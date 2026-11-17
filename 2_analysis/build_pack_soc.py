
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import build_queueing as queueing
import rebuild_inventory_chain as daily_inventory
from build_inventory import hp_factors, operations_record
from src import provenance
from src.loaders import DATA, PARAMS_FILE, RESULTS, params, ship_parameters, write_dir

TOL = 1e-7
RUN = "pack_replay"


@dataclass(frozen=True)
class VoyagePlan:
    ships: pd.DataFrame
    outfitting: pd.DataFrame
    swaps: pd.DataFrame


@dataclass(frozen=True)
class FleetCase:
    ports: list[str]
    days: pd.DatetimeIndex
    plans: dict[str, VoyagePlan]
    all_diesel_load_kwh: float
    retained_diesel_load_kwh: float
    fleet_km: float
    ships_total: int
    legacy_audit: dict


def full_pack_exchange(count_before, count_after, used_kwh, usable_kwh):
    """Return full issues and deficits of all packs unloaded at a call."""
    if not 0 <= used_kwh <= count_before * usable_kwh + TOL:
        raise ValueError("Voyage exceeds the available energy of its packs.")
    used_kwh = min(used_kwh, count_before * usable_kwh)
    used_packs = max(0, int(np.ceil((used_kwh - TOL) / usable_kwh)))
    deficits = np.clip(used_kwh - np.arange(used_packs) * usable_kwh, 0, usable_kwh)
    full_remaining = count_before - used_packs
    retained = min(full_remaining, count_after)
    returned = np.r_[deficits, np.zeros(full_remaining - retained)]
    issued = count_after - retained
    assert count_before + issued - len(returned) == count_after
    residual = count_before * usable_kwh - used_kwh
    assert np.isclose(
        residual + issued * usable_kwh - (len(returned) * usable_kwh - returned.sum()),
        count_after * usable_kwh, rtol=0, atol=TOL,
    )
    return issued, returned


def select_pack_counts(blocks, propulsion, auxiliary, displacement, rated_count,
                       pack_mass, usable_kwh, differential):
    candidates = np.arange(1, rated_count + 1)
    delta = displacement + (candidates - rated_count) * pack_mass
    if np.any(delta <= 0):
        raise ValueError("Reference displacement cannot support the pack-mass sensitivity.")
    factors = (delta / displacement) ** (2.0 / 3.0)
    counts, scales, fixed_mass_counts = [], [], []
    for start, end in blocks:
        prop = propulsion[start:end].sum()
        aux = auxiliary[start:end].sum()
        feasible = np.flatnonzero(prop * factors + aux <= candidates * usable_kwh + TOL)
        if not len(feasible) or feasible[-1] != rated_count - 1:
            raise ValueError("The original fixed pack allocation is infeasible.")
        choice = int(feasible[0]) if differential else rated_count - 1
        counts.append(int(candidates[choice]))
        scales.append(float(factors[choice]))
        fixed_mass_counts.append(int(candidates[np.flatnonzero(
            prop + aux <= candidates * usable_kwh + TOL,
        )[0]]))
    return counts, scales, fixed_mass_counts


def plan_voyage(legs, events, vessel, pack_mass, start_time, differential):
    """Check all intervals including legs after the last scheduled swap."""
    rated_count = int(vessel["packs"])
    rated_kwh = float(vessel["pack_capacity_kwh"])
    derating = float(vessel["depth_of_discharge"] * vessel["utilisation_efficiency"])
    assert rated_count * rated_kwh == vessel["rated_capacity_kwh"]
    usable = rated_kwh * derating
    marks = np.flatnonzero(legs["optimised swapping"].to_numpy())
    assert legs.iloc[marks].end.tolist() == events.port.tolist()
    boundaries = np.unique(np.r_[0, marks + 1, len(legs)])
    blocks = list(zip(boundaries[:-1], boundaries[1:], strict=True))
    base_load = legs["energy_kwh"].to_numpy() * derating
    auxiliary = legs["sailing_h"].to_numpy() * float(vessel["auxiliary_power_kw"])
    propulsion = base_load - auxiliary
    assert (propulsion >= 0).all()

    counts, scales, fixed_mass_counts = select_pack_counts(
        blocks, propulsion, auxiliary, float(vessel.avg_water), rated_count,
        pack_mass, usable, differential,
    )
    swaps = []
    total_used, minimum_soc, final_depletion = 0.0, 1.0, 0.0
    for block, (start, end) in enumerate(blocks):
        count = counts[block]
        consumed = np.cumsum(propulsion[start:end] * scales[block] + auxiliary[start:end])
        remaining = count * usable - consumed
        if remaining.min() < -TOL:
            raise ValueError("Usable SOC fell below zero during a voyage interval.")
        minimum_soc = min(minimum_soc, max(0.0, float(remaining.min()) / (count * usable)))
        used = float(consumed[-1])
        total_used += used
        if block < len(events):
            next_count = counts[block + 1] if block + 1 < len(counts) else count
            issued, deficits = full_pack_exchange(count, next_count, used, usable)
            event = events.iloc[block]
            if not differential:
                assert np.isclose(deficits.sum(), event.restored_kwh * derating, rtol=0, atol=TOL)
            swaps.append(dict(
                port=event.port, arrive=event.arrive, issued=issued,
                deficits_kwh=deficits, onboard_change=next_count - count,
                own_call=int(event.cargo_port), window_h=float(event.window_h),
            ))
        else:
            final_depletion = used

    assert np.isclose(
        total_used, sum(r["deficits_kwh"].sum() for r in swaps) + final_depletion,
        rtol=0, atol=TOL,
    )
    return (
        dict(load_kwh=total_used, fixed_weight_load_kwh=float(base_load.sum()),
             initial_packs=counts[0], maximum_packs=max(counts), rated_packs=rated_count,
             fixed_weight_minimum_initial_packs=fixed_mass_counts[0],
             minimum_usable_soc=minimum_soc, terminal_depletion_kwh=final_depletion,
             legs=len(legs), final_tail=len(blocks) > len(events)),
        dict(port=legs.start.iloc[0], arrive=start_time, packs=counts[0]),
        swaps,
    )


def audit_legacy(ops, ships, spans):
    ships_r1 = ships.copy()
    coefficients = pd.read_csv(RESULTS / "coefficient_correction/admiralty_pership.csv").set_index("mmsi")
    ships_r1["admiralty_coefficient"] = coefficients.C_new_code.reindex(ships.index)
    served = ops.loc[ops["feasible"] == 1]
    plans, _, _ = daily_inventory.build_plans(served, ships_r1, spans, hp_factors()[1],
                                          reference_filters=False)
    negative_load, tail, no_station = 0, 0, 0
    for plan in plans.values():
        calls = plan[plan["optimised swapping"] == 1]
        no_station += int(calls.empty)
        tail += int(not calls.empty and calls.index[-1] < plan.index[-1])
        adjustment = calls["packs_removed"] * calls["sign"].map({0: 0, 1: -1, 2: 1})
        negative_load += int(((calls["packs_swapped_das"] + adjustment) < 0).sum())
    return dict(negative_load_events=negative_load, ships_with_post_swap_tail=tail,
                ships_without_station_call=no_station)


def build_case() -> FleetCase:
    p = params()
    ports = p["case"]["plan2_stations_8"]
    days = pd.date_range(*p["case"]["ais_window"])
    ops = operations_record(ports)
    ships = ship_parameters().set_index("mmsi")
    _, _, spans = daily_inventory.load_control()
    events = queueing.arrival_stream(ports)
    event_by_ship = dict(tuple(events.groupby("mmsi", sort=False)))
    assert len(events) == 4634 and ops.mmsi.nunique() == 172

    leg_load = ops["energy_kwh"] * ops.mmsi.map(ships["depth_of_discharge"] * ships["utilisation_efficiency"])
    modes = {}
    pack_mass = float(p["inventory"]["battery_unit_mass_t"])
    for mode in ("EASM", "DASM"):
        rows, outfits, swaps = [], [], []
        for mmsi, legs in ops.loc[ops["feasible"] == 1].groupby("mmsi", sort=True):
            calls = event_by_ship.get(mmsi, events.iloc[:0])
            row, outfit, exchanged = plan_voyage(
                legs, calls, ships.loc[mmsi], pack_mass, pd.Timestamp(spans.loc[mmsi, "t0"]),
                differential=mode == "DASM",
            )
            rows.append(row)
            outfits.append(outfit)
            swaps.extend(dict(call, vessel=mmsi) for call in exchanged)
        plan = VoyagePlan(pd.DataFrame(rows), pd.DataFrame(outfits), pd.DataFrame(swaps))
        assert len(plan.ships) == 156 and len(plan.swaps) == len(events)
        modes[mode] = plan
    diesel = float(leg_load.loc[ops["feasible"] == 0].sum())
    total = float(leg_load.sum())
    assert np.isclose(modes["EASM"].ships.load_kwh.sum() + diesel, total, rtol=0, atol=TOL)
    distance = float(ops.distance.sum() * p["cost"]["nautical_mile_to_km"])
    return FleetCase(ports, days, modes, total, diesel, distance, int(ops.mmsi.nunique()),
                     audit_legacy(ops, ships, spans))


def report(case):
    rows = []
    pack_mass = float(params()["inventory"]["battery_unit_mass_t"])
    for mode, plan in case.plans.items():
        ships = plan.ships
        actual = float(ships.load_kwh.sum())
        fixed = float(ships.fixed_weight_load_kwh.sum())
        rows.append(dict(
            mode=mode, vessels=len(ships), checked_legs=int(ships.legs.sum()),
            checked_swaps=len(plan.swaps), outfitting_packs=int(ships.initial_packs.sum()),
            offsite_outfitting_packs=int(plan.outfitting.loc[
                ~plan.outfitting.port.isin(case.ports), "packs",
            ].sum()),
            sum_vessel_peak_packs=int(ships.maximum_packs.sum()),
            minimum_usable_soc=float(ships.minimum_usable_soc.min()),
            minimum_outfitting_packs_at_fixed_weight=int(ships.fixed_weight_minimum_initial_packs.sum()),
            avg_outfitting_mass_reduction_t=float((ships.rated_packs-ships.initial_packs).mean()*pack_mass),
            load_kwh=actual, load_at_fixed_weight_kwh=fixed,
            load_reduction_from_pack_mass_pct=100 * (1 - actual/fixed),
            terminal_depletion_kwh=float(ships.terminal_depletion_kwh.sum()),
            **case.legacy_audit,
        ))
    return pd.DataFrame(rows)


def source_files():
    upstream_dir = RESULTS / "coefficient_correction"
    return [
        Path(__file__), Path(queueing.__file__), Path(daily_inventory.__file__),
        Path(operations_record.__code__.co_filename),
        Path(daily_inventory.__file__).with_name("resolve_siting.py"),
        PARAMS_FILE, DATA / "02_channel_topology/port_topology.csv",
        DATA / "03_ship_parameters/172_ship_parameters.xlsx",
        DATA / "04_paper_baseline/segments_legs_with_timestamps.xlsx",
        upstream_dir / "segment_energy.csv", upstream_dir / "admiralty_pership.csv",
        upstream_dir / "ais_underway.csv", upstream_dir / "ais_berthing.csv",
    ]


def save(case):
    out = write_dir(RUN) / "pack_soc_check.csv"
    table = report(case)
    table.to_csv(out, index=False)
    provenance.record(out, source_files(), script=__file__, params_used={
        "routes": "Wuhu eight stations, same legs/speed and 156 served ships",
        "allocation": "integer; keep usable full packs; sequential discharge, final tail checked",
        "mass": "marginal 25 t per pack relative to the fixed-pack reference displacement",
        "energy": "usable-energy band already includes the DoD and utilisation factor",
    })
    return out


if __name__ == "__main__":
    save(build_case())
