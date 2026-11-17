
from __future__ import annotations

from dataclasses import dataclass
from heapq import heapify, heappop, heappush, heapreplace
from pathlib import Path

import numpy as np
import pandas as pd

import build_pack_soc as fleet_model
import build_rule_based_transfer as rules
from build_transfer_model import port_distances
from src import provenance
from src.loaders import RESULTS, params, write_dir

RUN = "finite_power_replay"
PACK_CONNECTION_LIMIT_KW = 400.0
CLEARING_DAYS = 1
RULE_FLEET = 120
OPENING_PCT_GRID = range(50, 151, 5)
TOL = 1e-7


@dataclass(frozen=True)
class PortCalls:
    issue_h: np.ndarray
    issues: np.ndarray
    return_h: np.ndarray
    deficits_kwh: np.ndarray
    wait_h: np.ndarray
    arrival_delay_h: np.ndarray
    onboard_change_h: np.ndarray
    onboard_change: np.ndarray


@dataclass(frozen=True)
class NetworkScenario:
    name: str
    mode: str
    stations: pd.DataFrame
    total_battery_assets: int
    peak_simultaneous_onboard: int
    end_onboard: int
    rule_fleet: int


def berth_schedule(swaps, start, berth_counts):
    """FCFS at each port with observed berth slack along each vessel's route."""
    q = params()["queueing"]
    ref_h = (swaps.arrive - start).dt.total_seconds().to_numpy() / 3600.0
    ports = swaps.port.to_numpy()
    returns = swaps.deficits_kwh.map(len).to_numpy(dtype=int)
    actions = np.maximum(swaps.issued.to_numpy(dtype=int), returns)
    assert (actions > 0).all()
    service = q["fixed_hours"] + q["per_pack_hours"] * actions
    slack = np.where(swaps.own_call.to_numpy() == 1, swaps.window_h.to_numpy(), 0.0)
    assert np.isfinite(slack).all() and (slack >= 0).all()

    next_call = np.full(len(swaps), -1, dtype=int)
    requests = []
    for ship in swaps.groupby("vessel", sort=False).indices.values():
        assert (np.diff(ref_h[ship]) >= 0).all()
        next_call[ship[:-1]] = ship[1:]
        requests.append((ref_h[ship[0]], int(ship[0])))
    heapify(requests)
    free = {port: [0.0] * count for port, count in berth_counts.items()}
    arrives, starts, finishes = [np.empty(len(swaps)) for _ in range(3)]
    while requests:
        arrival, i = heappop(requests)
        arrives[i] = arrival
        pool = free[ports[i]]
        starts[i] = max(arrival, pool[0])
        finishes[i] = starts[i] + service[i]
        heapreplace(pool, finishes[i])
        j = next_call[i]
        if j >= 0:
            gap = ref_h[j] - ref_h[i]
            travel_and_other_dwell = max(0.0, gap - slack[i])
            heappush(requests, (max(ref_h[j], finishes[i] + travel_and_other_dwell), int(j)))

    previous = next_call >= 0
    assert (arrives >= ref_h).all()
    assert (arrives[next_call[previous]] >= finishes[previous] - TOL).all()
    return starts, finishes, starts - arrives, arrives - ref_h


def station_calls(plan, port, start, timings):
    calls = plan.swaps.loc[plan.swaps.port == port]
    at_port = (plan.swaps.port == port).to_numpy()
    starts, finishes, waits, delays = (x[at_port] for x in timings)
    returns = calls.deficits_kwh.map(len).to_numpy(dtype=int)
    outfits = plan.outfitting.loc[plan.outfitting.port == port]
    outfit_h = (outfits.arrive - start).dt.total_seconds().to_numpy() / 3600.0
    return PortCalls(
        np.r_[starts, outfit_h],
        np.r_[calls.issued.to_numpy(dtype=int), outfits.packs.to_numpy(dtype=int)],
        np.repeat(finishes, returns),
        np.concatenate(calls.deficits_kwh.to_list()),
        waits, delays, finishes,
        calls.onboard_change.to_numpy(dtype=int),
    )


def common_calls(case, initial_berths):
    q = params()["queueing"]
    counts = initial_berths.copy()
    while True:
        all_calls = {}
        for mode, plan in case.plans.items():
            timings = berth_schedule(plan.swaps, case.days[0], counts)
            all_calls[mode] = {port: station_calls(plan, port, case.days[0], timings)
                               for port in case.ports}
        failing = {
            port for port in case.ports for calls in all_calls.values()
            if np.quantile(calls[port].wait_h, .95) > q["service_level_wait_h"]
        }
        if not failing:
            return counts, all_calls
        for port in failing:
            counts[port] += 1
            if counts[port] > max(q["berth_sweep"]):
                raise ValueError(f"No tested berth configuration meets the wait target at {port}.")


def pack_ready_times(return_h, deficits_kwh, power_kw, eta_c,
                     connection_limit_kw=PACK_CONNECTION_LIMIT_KW):
    """Meter the recharge deficit and assign one returned pack to each free slot."""
    if power_kw <= 0 or connection_limit_kw <= 0 or not 0 < eta_c <= 1:
        raise ValueError("Invalid station power or charging efficiency.")
    slots = int(np.ceil(power_kw / connection_limit_kw))
    slot_kw = power_kw / slots
    ready = np.array(return_h, dtype=float, copy=True)
    assert (deficits_kwh >= 0).all()
    available = [0.0] * slots
    order = np.argsort(return_h, kind="stable")
    for idx in order[deficits_kwh[order] > TOL]:
        ready[idx] = max(available[0], return_h[idx]) + deficits_kwh[idx] / (eta_c * slot_kw)
        heapreplace(available, ready[idx])
    assert (ready >= return_h).all()
    return ready, slots, slot_kw


def daily_stock_flows(issue_h, issues, ready_h, days):
    """Compute exact intra-day stock floors for transfers arriving at midnight."""
    if np.any(issue_h < 0) or np.any(ready_h < 0) or np.any(issue_h >= days * 24):
        raise ValueError("An issue falls outside the case-study service horizon.")
    active = ready_h[ready_h < days * 24]
    h = np.r_[issue_h, active]
    net = np.r_[issues, -np.ones(len(active), dtype=int)]
    order = np.lexsort((net > 0, h))
    hours, changes = h[order], net[order]
    cut = np.searchsorted(hours, np.arange(days + 1) * 24)
    demand = np.zeros(days, dtype=int)
    available = np.zeros_like(demand)
    floor = np.zeros_like(demand)
    for day, (lo, hi) in enumerate(zip(cut[:-1], cut[1:], strict=True)):
        change = changes[lo:hi]
        demand[day] = np.maximum(change, 0).sum()
        available[day] = -np.minimum(change, 0).sum()
        floor[day] = np.cumsum(change).max(initial=0)
    assert demand.sum() == issues.sum() and available.sum() == len(active)
    return demand, available, floor


def no_transfer_opening(load, ready, day_floor):
    net = load - ready
    required = np.cumsum(net, axis=1) - net + day_floor
    return required.max(axis=1, initial=0)


def charging_case(calls, power, eta_c, days):
    ready, slots, slot_kw = pack_ready_times(calls.return_h, calls.deficits_kwh, power, eta_c)
    load, released, floor = daily_stock_flows(calls.issue_h, calls.issues, ready, days)
    return dict(
        load=load, ready=released, floor=floor, ready_times=ready, slots=slots, slot_kw=slot_kw,
        charge_kwh=float(calls.deficits_kwh.sum() / eta_c),
        pending_packs=int((ready >= 24 * days).sum()),
        last_ready_h=float(ready.max()),
    )


def run_stock(case, mode, port_inputs, fleet_size):
    """Size initial stock at the fixed station power and evaluate every issue."""
    ports = case.ports
    loads = np.stack([port_inputs[port]["load"] for port in ports])
    ready = np.stack([port_inputs[port]["ready"] for port in ports])
    floors = np.stack([port_inputs[port]["floor"] for port in ports])
    independent = no_transfer_opening(loads, ready, floors)
    if not fleet_size:
        net = loads - ready
        stock_end = independent - net.sum(axis=1)
        stock_at_day_open = independent[:, None] - (np.cumsum(net, axis=1) - net)
        assert (stock_at_day_open >= floors).all()
        return independent, stock_end, np.zeros(len(ports)), np.zeros(len(ports)), 100

    dist = port_distances(ports)
    travel = np.ceil(dist / params()["inventory"]["transfer_km_per_day"]).astype(int)
    transit = travel + (travel > 0)
    fleet = rules.equal_allocation(fleet_size, len(ports))
    for share in OPENING_PCT_GRID:
        opening = (independent * share + 99) // 100
        r = rules.replay(loads, ready, dist, transit, opening, fleet, day_opening_need=floors)
        if r["emergency"].sum() <= TOL:
            assert r["in_transit_packs"] == r["in_transit_trucks"] == 0
            loaded_km = (r["loaded"].sum(axis=0) * dist).sum(axis=1)
            empty_km = (r["empty"].sum(axis=0) * dist).sum(axis=1)
            return opening, r["stock"][:, -1].astype(int), loaded_km, empty_km, share
    raise ValueError(f"No zero-shortage case for {mode} on the stated opening-stock grid.")


def onboard_counts(plan, calls, start):
    first_h = (plan.outfitting.arrive - start).dt.total_seconds().to_numpy() / 3600.0
    hours = np.concatenate([first_h] + [c.onboard_change_h for c in calls.values()])
    values = np.concatenate([plan.outfitting.packs.to_numpy()] +
                            [c.onboard_change for c in calls.values()])
    carried = pd.Series(values, index=hours).groupby(level=0).sum().sort_index().cumsum()
    assert (carried >= 0).all()
    return int(carried.max()), int(carried.iloc[-1])


def build_scenario(case, mode, name, berth_counts, all_calls, powers, inputs, fleet_size=0):
    plan = case.plans[mode]
    calls = all_calls[mode]
    opening, closing, loaded, empty, sizing_share = run_stock(case, mode, inputs, fleet_size)
    n_days = len(case.days)
    rows = []
    for i, port in enumerate(case.ports):
        c = inputs[port]
        rows.append(dict(
            scenario=name, mode=mode, port=port, berths=berth_counts[port], grid_kw=powers[port],
            connections=c["slots"], kw_per_connection=c["slot_kw"],
            swaps=len(calls[port].wait_h),
            berth_wait_p95_h=float(np.quantile(calls[port].wait_h, .95)),
            propagated_arrival_delay_p95_h=float(np.quantile(calls[port].arrival_delay_h, .95)),
            recurring_recharge_grid_kwh=c["charge_kwh"],
            mean_recharge_grid_kw=c["charge_kwh"] / (24 * n_days),
            last_charge_h_relative_to_window_end=c["last_ready_h"] - 24 * n_days,
            opening_charged_packs=int(opening[i]), closing_charged_packs=int(closing[i]),
            still_charging_or_waiting_packs=c["pending_packs"],
            loaded_truck_km_departures=loaded[i], empty_truck_km_departures=empty[i],
            modeled_extra_supply_packs=0, stock_design_pct_of_no_transfer=sizing_share,
        ))
    by_port = pd.DataFrame(rows)
    offsite = int(plan.outfitting.loc[~plan.outfitting.port.isin(case.ports), "packs"].sum())
    peak, end = onboard_counts(plan, calls, case.days[0])
    assets = int(opening.sum()) + offsite
    assert peak <= assets
    assert int(closing.sum() + by_port.still_charging_or_waiting_packs.sum() + end) == assets
    return NetworkScenario(name, mode, by_port, assets, peak, end, fleet_size)


def build_cases(case):
    p = params()
    eta_c = float(p["lcc"]["energy_transfer_efficiency"])
    ref_kw = float(p["lcc"]["station_power_kw"])
    berths_file = RESULTS / "paper/s7_robustness/station_sizing_advice.csv"
    advice = pd.read_csv(berths_file).set_index("port")
    initial_counts = {port: int(advice.loc[port, "berths_required"]) for port in case.ports}
    berth_counts, all_calls = common_calls(case, initial_counts)

    observed_days = len(case.days)
    days = observed_days + CLEARING_DAYS
    power_candidates = [ref_kw * k for k in (1, 2, 3, 4)]
    selected = {}
    initial = {mode: {} for mode in case.plans}
    station_inputs = {mode: {} for mode in case.plans}
    for port in case.ports:
        for power in power_candidates:
            evaluated = {mode: charging_case(calls[port], power, eta_c, days)
                         for mode, calls in all_calls.items()}
            if power == ref_kw:
                for mode, result in evaluated.items():
                    initial[mode][port] = result
            if all(r["charge_kwh"] <= power * 24 * observed_days
                   and r["pending_packs"] == 0 for r in evaluated.values()):
                selected[port] = power
                for mode, result in evaluated.items():
                    station_inputs[mode][port] = result
                break
        else:
            raise ValueError(f"No tested station power clears the returned packs at {port}.")

    scenarios = []
    for mode in case.plans:
        scenarios.append(build_scenario(
            case, mode, f"{mode} 3 MW diagnostic", berth_counts, all_calls,
            dict.fromkeys(case.ports, ref_kw), initial[mode],
        ))
        scenarios.append(build_scenario(
            case, mode, f"{mode} no transfer", berth_counts, all_calls,
            selected, station_inputs[mode],
        ))
    scenarios.append(build_scenario(
        case, "DASM", "DASM rule dispatch", berth_counts, all_calls,
        selected, station_inputs["DASM"], RULE_FLEET,
    ))
    return scenarios


def save(scenarios):
    out = write_dir(RUN) / "integer_power_and_stock.csv"
    table = pd.concat([s.stations for s in scenarios], ignore_index=True)
    table.to_csv(out, index=False)
    inputs = fleet_model.source_files() + [
        Path(__file__), Path(rules.__file__), Path(port_distances.__code__.co_filename),
        RESULTS / "paper/s7_robustness/station_sizing_advice.csv",
    ]
    provenance.record(out, inputs, script=__file__, params_used={
        "power_factors": [1, 2, 3, 4],
        "connection_limit_kw": PACK_CONNECTION_LIMIT_KW,
        "clearing_days_after_90_day_arrivals": CLEARING_DAYS,
        "common_berths": "increase the base counts where linked arrivals exceed the one-hour p95",
        "event_clock": "propagate service beyond the measured previous berth window to later "
                       "calls of the same vessel; use zero slack at inserted calls",
        "common_station_power": "smallest tested power covering mean work and one-day clearing in both modes",
        "transport": "end-of-day dispatch, full 1 or 2 travel days following decision day",
        "rule_fleet": RULE_FLEET,
        "rule_opening_grid_pct_of_site_no_transfer_stock": list(OPENING_PCT_GRID),
        "opening_stock_design": "retrospective sizing for fixed arrivals, every scheduled issue served",
    })
    return out


if __name__ == "__main__":
    save(build_cases(fleet_model.build_case()))
