
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import build_queueing as queueing
import rebuild_inventory_chain
from src import provenance
from src.loaders import DATA, PARAMS_FILE, RESULTS, params, ship_parameters, write_dir

RUN = "finite_power_replay"
TOL = 1e-7


def charge_batches(released_h, metered_kwh, power_kw):
    """Give the start and completion of each FIFO batch in release order."""
    if power_kw <= 0 or not np.isfinite(power_kw):
        raise ValueError("Charging input must be finite and positive.")
    if np.any(metered_kwh <= 0) or np.any(np.diff(released_h) < 0):
        raise ValueError("Use positive energy batches in release order.")
    start = np.empty(len(released_h))
    end = np.empty(len(released_h))
    available_h = 0.0
    for index, (release, work) in enumerate(zip(released_h, metered_kwh, strict=True)):
        start[index] = max(release, available_h)
        end[index] = start[index] + work / power_kw
        available_h = end[index]
    return start, end


def opening_stock(issue_h, issue_packs, ready_h, ready_packs, reported_stock):
    """Scheduled issue deficit and minimal opening stock in pack equivalents."""
    if np.any(np.diff(issue_h) < 0) or np.any(np.diff(ready_h) < 0):
        raise ValueError("The stock ledger requires time-ordered transactions.")
    ready_count = np.searchsorted(ready_h, issue_h, side="right")
    charged = np.r_[0.0, np.cumsum(ready_packs)][ready_count]
    gap = np.cumsum(issue_packs) - charged
    needed = max(0.0, float(gap.max(initial=0.0)))
    cumulative_topup = np.maximum.accumulate(np.maximum(0.0, gap - reported_stock))
    extra = np.diff(cumulative_topup, prepend=0.0)
    return needed, extra


def station_metrics(events, outfits, window_start, window_days, berths,
                    reported_stock, power_kw, rated_pack_kwh):
    """Run the stock ledger at the fixed berth schedule for one power case."""
    arrival_h = (events.arrive - window_start).dt.total_seconds().to_numpy() / 3600.0
    swap_packs = events.packs.to_numpy()
    q = params()["queueing"]
    service_h = q["fixed_hours"] + q["per_pack_hours"] * swap_packs
    wait_h = queueing.simulate(arrival_h, service_h, berths)
    swap_start_h = arrival_h + wait_h
    release_h = swap_start_h + service_h
    outfit_h = (pd.to_datetime(outfits["start_time"]) - window_start).dt.total_seconds() / 3600.0
    outfit_packs = outfits["packs_on_board_at_origin"] + outfits["packs_removed_at_origin"]
    all_times = np.r_[swap_start_h, outfit_h.to_numpy()]
    all_draws = np.r_[swap_packs, outfit_packs.to_numpy()]
    issue_order = np.argsort(all_times, kind="stable")

    returns = np.flatnonzero(swap_packs > 0)
    order = returns[np.argsort(release_h[returns], kind="stable")]
    released = release_h[order]
    metered = events.metered_kwh.to_numpy()[order]
    starts, completed = charge_batches(released, metered, power_kw)
    required, topups = opening_stock(
        all_times[issue_order], all_draws[issue_order], completed, swap_packs[order],
        reported_stock,
    )
    after_start = topups > TOL
    first = np.flatnonzero(after_start)
    first_time = None
    if len(first):
        first_time = window_start + pd.to_timedelta(all_times[issue_order[first[0]]], unit="h")
    horizon_h = 24.0 * len(window_days)
    charged_kwh = float((np.clip(horizon_h - starts, 0, completed - starts) * power_kw).sum())
    due_kwh = float(metered.sum())
    assert charged_kwh <= min(due_kwh, power_kw * horizon_h) + TOL
    assert np.isclose(
        topups.sum(), max(0.0, required - reported_stock), rtol=0, atol=TOL,
    )
    daily = events.groupby(events.arrive.dt.normalize()).metered_kwh.sum().reindex(
        window_days, fill_value=0.0,
    )
    assert np.isclose(float(daily.sum()), due_kwh)
    return dict(
        berths=berths,
        scheduled_swaps=len(events),
        own_port_swaps=int(events.cargo_port.sum()),
        outfitting_pack_equiv=float(outfit_packs.sum()),
        grid_input_kw=power_kw,
        grid_kwh_per_day_limit=24.0 * power_kw,
        horizon_days=len(window_days),
        metered_demand_kwh=due_kwh,
        mean_grid_input_kw=due_kwh / horizon_h,
        calendar_day_p95_metered_kwh=float(daily.quantile(.95)),
        calendar_day_max_metered_kwh=float(daily.max()),
        calendar_day_max_date=daily.idxmax().date().isoformat(),
        metered_kwh_charged_by_window_end=charged_kwh,
        scheduled_metered_kwh_unfinished_at_window_end=max(0.0, due_kwh - charged_kwh),
        charging_turnaround_p95_h=float(np.quantile(completed - released, .95)),
        charging_turnaround_max_h=float((completed - released).max()),
        daily_opening_pack_equiv=reported_stock,
        required_opening_pack_equiv=required,
        additional_opening_pack_equiv=max(0.0, required - reported_stock),
        scheduled_requests_needing_topup=int(after_start.sum()),
        first_topup_at=first_time.isoformat(timespec="seconds") if first_time is not None else "",
        rated_pack_kwh=rated_pack_kwh,
    )


def build_report(only_ports=None):
    p = params()
    paper8 = p["case"]["plan2_stations_8"]
    events, outfits = queueing.swap_schedule(paper8)
    served_outfits = outfits.loc[outfits["feasible"] == 1]
    assert len(served_outfits) == 156
    inventory_file = RESULTS / "paper/s5_inventory/inventory_summary_eas.csv"
    berths_file = RESULTS / "paper/s7_robustness/station_sizing_advice.csv"
    inventory = pd.read_csv(inventory_file).set_index("port")
    berth_advice = pd.read_csv(berths_file).set_index("port")

    pack_kwh = float(p["inventory"]["battery_unit_capacity_kwh"])
    ships = ship_parameters().set_index("mmsi")
    assert (ships["pack_capacity_kwh"] == pack_kwh).all()
    assert np.allclose(events.packs * pack_kwh, events.restored_kwh, rtol=0, atol=TOL)
    ship_factor = (ships["depth_of_discharge"] * ships["utilisation_efficiency"]).loc[events.mmsi].to_numpy()
    eta_c = float(p["lcc"]["energy_transfer_efficiency"])
    events["metered_kwh"] = events.restored_kwh * ship_factor / eta_c
    window_days = pd.date_range(*p["case"]["ais_window"])
    start = window_days[0]
    horizon_end = window_days[-1] + pd.Timedelta(days=1)
    assert events.arrive.ge(start).all() and events.arrive.lt(horizon_end).all()
    assert pd.to_datetime(served_outfits["start_time"]).ge(start).all()
    assert pd.to_datetime(served_outfits["start_time"]).lt(horizon_end).all()
    offset_days = ((events.arrive - events.arrive.min()).dt.total_seconds() // 86400).astype(int)
    events["model_day"] = offset_days
    grid_kw = float(p["lcc"]["station_power_kw"])

    rows = []
    for port in (paper8 if only_ports is None else only_ports):
        ev = events.loc[events.port == port]
        at_port = served_outfits.loc[served_outfits["origin_port"] == port]
        assert len(ev) > 0
        outfit_sum = (at_port["packs_on_board_at_origin"] + at_port["packs_removed_at_origin"]).sum()
        assert np.isclose(outfit_sum, inventory.loc[port, "outfitting_packs"])
        model_daily = ev.groupby("model_day").packs.sum().reindex(
            range(ev.model_day.min(), ev.model_day.max() + 1), fill_value=0.0,
        ) * pack_kwh
        model_p95 = float(model_daily.quantile(.95))
        assert np.isclose(model_p95, berth_advice.loc[port, "peak_day_kwh"], rtol=0, atol=TOL)
        for multiplier in (1, 2, 3, 4):
            result = station_metrics(
                ev, at_port, start, window_days, int(berth_advice.loc[port, "berths_required"]),
                float(inventory.loc[port, "initial"]), grid_kw * multiplier, pack_kwh,
            )
            rows.append(dict(port=port, mode="EASM",
                             r1_offset_day_p95_rated_equiv_kwh=model_p95, **result))
    return pd.DataFrame(rows)


def main():
    report = build_report()
    out = write_dir(RUN) / "station_charging_replay.csv"
    report.to_csv(out, index=False)
    upstream_dir = RESULTS / queueing.UPSTREAM_DEFAULT
    paper_dir = RESULTS / "paper"
    provenance.record(
        out,
        inputs=[
            Path(__file__), Path(queueing.__file__), Path(rebuild_inventory_chain.__file__),
            PARAMS_FILE, DATA / "02_channel_topology/port_topology.csv",
            DATA / "03_ship_parameters/172_ship_parameters.xlsx",
            DATA / "04_paper_baseline/segments_legs_with_timestamps.xlsx",
            upstream_dir / "segment_energy.csv", upstream_dir / "admiralty_pership.csv",
            upstream_dir / "ais_underway.csv", upstream_dir / "ais_berthing.csv",
            paper_dir / "s5_inventory/inventory_summary_eas.csv",
            paper_dir / "s7_robustness/station_sizing_advice.csv",
        ],
        script=__file__,
        params_used={
            "charge_policy": "24h aggregate grid input, FIFO batch release at charge completion",
            "swap_timing": "stock issue at FCFS berth start; return at berth finish",
            "opening_outfitting": "charged pack draw with no returned battery",
            "grid_power_multipliers": [1, 2, 3, 4],
            "efficiency_convention": "metered energy = rated equivalent * DoD * eta / eta_c",
            "scope": "the eight stations of the paper, EASM, no inter-port transfers",
        },
    )
    cols = ["port", "grid_input_kw", "mean_grid_input_kw",
            "required_opening_pack_equiv", "additional_opening_pack_equiv"]


if __name__ == "__main__":
    main()
