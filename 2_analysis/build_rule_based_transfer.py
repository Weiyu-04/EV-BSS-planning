
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import build_transfer_model as transfer_lp
import rebuild_inventory_chain
from src import provenance
from src.loaders import DATA, PARAMS_FILE, RESULTS, params, write_dir

RUN = "rule_based_transfer"
LOOKBACK_DAYS = 7
OPENING_STEP_PER_PORT = 10
TOL = 1e-7


def equal_allocation(total, sites):
    """Split an integer stock or fleet evenly, assigning remainders by site order."""
    each, remainder = divmod(total, sites)
    allocation = np.full(sites, each, dtype=int)
    allocation[:remainder] += 1
    return allocation


def ready_returns(unload, ratio, delay):
    """Apply the same daily release convention as the daily inventory model and the LP."""
    if delay < 0 or not 0 <= ratio <= 1:
        raise ValueError("Invalid charging share or delay.")
    if delay == 0:
        return unload.copy()
    ready = (1 - ratio) * unload
    if delay < unload.shape[1]:
        ready[:, delay:] += ratio * unload[:, :-delay]
    return ready


def replay(load, ready, dist, transit, opening, fleet_start, lookback_days=LOOKBACK_DAYS,
           day_opening_need=None, cover_days=None):
    """Replay daily dispatch, optionally using event-derived opening stock floors."""
    sites, days = load.shape
    if ready.shape != load.shape or lookback_days < 1:
        raise ValueError("Invalid inventory flows or lookback.")
    if day_opening_need is not None:
        if (day_opening_need.shape != load.shape or (day_opening_need < 0).any()
                or (day_opening_need < load - ready - TOL).any()):
            raise ValueError("Opening stock floors must cover each day's net draw.")
    routes = sorted(
        (int(transit[i, j]), dist[i, j], i, j)
        for i in range(sites) for j in range(sites) if i != j
    )
    if any(lag < 1 for lag, _, _, _ in routes):
        raise ValueError("Inter-port shipments require a positive travel time.")
    max_lag = max((r[0] for r in routes), default=0)
    cover_days = 1 + max_lag if cover_days is None else cover_days
    pack_arrivals = np.zeros((days + max_lag, sites), dtype=int)
    truck_arrivals = np.zeros_like(pack_arrivals)
    loaded = np.zeros((days, sites, sites), dtype=int)
    empty = np.zeros_like(loaded)
    emergency = np.zeros_like(load, dtype=float)
    stock_log = np.zeros_like(emergency)
    stock = np.asarray(opening, dtype=float).copy()
    fleet = np.asarray(fleet_start, dtype=int).copy()
    fleet_count = int(fleet.sum())
    stock_balance = float(stock.sum())
    net_draw = load - ready

    for day in range(days):
        fleet += truck_arrivals[day]
        if day_opening_need is None:
            stock += ready[:, day] + pack_arrivals[day]
            emergency[:, day] = np.maximum(0.0, load[:, day] - stock)
            stock += emergency[:, day] - load[:, day]
            day_routes = routes
        else:
            stock += pack_arrivals[day]
            emergency[:, day] = np.maximum(0.0, day_opening_need[:, day] - stock)
            stock += ready[:, day] + emergency[:, day] - load[:, day]
            day_routes = [route for route in routes if day + route[0] < days]
        stock_balance += float((emergency[:, day] - net_draw[:, day]).sum())

        start = max(0, day + 1 - lookback_days)
        recent_draw = np.maximum(net_draw[:, start:day + 1], 0.0)
        buffer = cover_days * recent_draw.mean(axis=1)
        inbound = pack_arrivals[day + 1:].sum(axis=0)
        surplus = np.floor(np.maximum(stock - buffer, 0.0)).astype(int)
        need = np.ceil(np.maximum(buffer - stock - inbound - TOL, 0.0)).astype(int)

        for lag, _, origin, dest in day_routes:
            count = min(fleet[origin], surplus[origin], need[dest])
            if count:
                fleet[origin] -= count
                stock[origin] -= count
                surplus[origin] -= count
                need[dest] -= count
                loaded[day, origin, dest] = count
                pack_arrivals[day + lag, dest] += count
                truck_arrivals[day + lag, dest] += count

        inbound_trucks = truck_arrivals[day + 1:].sum(axis=0)
        unmet_buffer = int(need.sum())
        donor_need = np.maximum(0, np.minimum(surplus, unmet_buffer) - fleet - inbound_trucks)
        for lag, _, origin, dest in day_routes:
            if surplus[origin] > 0:
                continue
            count = min(fleet[origin], donor_need[dest], unmet_buffer)
            if count:
                fleet[origin] -= count
                donor_need[dest] -= count
                unmet_buffer -= count
                empty[day, origin, dest] = count
                truck_arrivals[day + lag, dest] += count

        stock_log[:, day] = stock
        assert (stock >= -TOL).all()
        assert int(fleet.sum() + truck_arrivals[day + 1:].sum()) == fleet_count
        assert np.isclose(
            stock.sum() + pack_arrivals[day + 1:].sum(), stock_balance, rtol=0, atol=TOL,
        )
    return dict(
        total_i0=float(np.sum(opening)),
        emergency=emergency,
        stock=stock_log,
        loaded=loaded,
        empty=empty,
        moved=int(loaded.sum()),
        pack_km=float((loaded * dist).sum()),
        empty_km=float((empty * dist).sum()),
        in_transit_packs=int(pack_arrivals[days:].sum()),
        in_transit_trucks=int(truck_arrivals[days:].sum()),
    )


def summarize(case, policy, opening_allocation, result):
    emergency = result.get("emergency")
    topup = emergency[emergency > TOL] if emergency is not None else np.array([])
    return dict(
        comparison=case,
        policy=policy,
        opening_allocation=opening_allocation,
        total_initial_pack_equiv=result["total_i0"],
        topup_pack_equiv=float(topup.sum()),
        shortage_port_days=len(topup),
        loaded_pack_equiv=result["moved"],
        loaded_pack_km=result["pack_km"],
        empty_truck_km=result["empty_km"],
        ending_packs_in_transit=result.get("in_transit_packs", 0),
        ending_trucks_in_transit=result.get("in_transit_trucks", 0),
    )


def build_report(trucks=120):
    ports, daily, _, dist, calendar = transfer_lp.prepare_case()
    load, unload = transfer_lp.aligned_flows(
        ports, daily, calendar, "packs_landed_das", "packs_taken_das",
    )
    p = params()
    inv = p["inventory"]
    assert calendar.equals(pd.date_range(*p["case"]["ais_window"]))
    transit = np.ceil(dist / float(inv["transfer_km_per_day"])).astype(int)
    ratio, delay = float(inv["charging_ratio"]), int(inv["charging_days"])
    ready = ready_returns(unload, ratio, delay)
    no_transfer_i0 = np.array([
        rebuild_inventory_chain.recur(draw, ret, ratio, delay)["initial"]
        for draw, ret in zip(load, unload, strict=True)
    ])
    lp = transfer_lp.solve(load, unload, dist, 0.0, ratio, delay, transit, trucks=trucks)
    if not lp["ok"]:
        raise RuntimeError(lp["message"])
    reference = pd.read_csv(RESULTS / "coefficient_correction/transfer_frontier.csv")
    published = reference.loc[
        (reference["mode"] == "DAS") & ~reference.terminal &
        (reference.lam == 0) & (reference.trucks_per_day == trucks)
    ].iloc[0]
    assert np.isclose(no_transfer_i0.sum(), published.no_transfer, rtol=0, atol=TOL)
    assert np.isclose(lp["total_i0"], published.total_i0, rtol=0, atol=TOL)

    sites = len(ports)
    empty_fleet = np.zeros(sites, dtype=int)
    fleet = equal_allocation(trucks, sites)
    no_transfer = replay(load, ready, dist, transit, no_transfer_i0, empty_fleet)
    assert float(no_transfer["emergency"].sum()) <= TOL

    fixed_fleet = np.floor(lp["base"]).astype(int)
    remainder = trucks - fixed_fleet.sum()
    fractional = lp["base"] - fixed_fleet
    fixed_fleet[np.argsort(-fractional, kind="stable")[:remainder]] += 1
    shared_lp = transfer_lp.solve(
        load, unload, dist, 0.0, ratio, delay, transit,
        trucks=trucks, initial_trucks=fixed_fleet,
    )
    if not shared_lp["ok"]:
        raise RuntimeError(shared_lp["message"])
    assert np.allclose(shared_lp["base"], fixed_fleet, rtol=0, atol=TOL)
    assert shared_lp["total_i0"] >= lp["total_i0"] - TOL
    rule_shared = replay(load, ready, dist, transit, shared_lp["i0"], fixed_fleet)
    none_shared = replay(load, ready, dist, transit, shared_lp["i0"], empty_fleet)

    base_budget = int(np.ceil(lp["total_i0"]))
    start = equal_allocation(base_budget, sites)
    rule_low = replay(load, ready, dist, transit, start, fleet)
    none_low = replay(load, ready, dist, transit, start, empty_fleet)

    lower = int(np.ceil(lp["total_i0"] / sites / OPENING_STEP_PER_PORT))
    upper = int(np.ceil(no_transfer_i0.max() / OPENING_STEP_PER_PORT))
    for per_port in range(lower * OPENING_STEP_PER_PORT,
                          (upper + 1) * OPENING_STEP_PER_PORT,
                          OPENING_STEP_PER_PORT):
        tested = np.full(sites, per_port, dtype=int)
        rule_tested = replay(load, ready, dist, transit, tested, fleet)
        if float(rule_tested["emergency"].sum()) <= TOL:
            break
    else:
        raise RuntimeError("No zero-top-up rule case within the stated opening-stock sweep.")
    none_tested = replay(load, ready, dist, transit, tested, empty_fleet)

    rows = [
        summarize("zero shortage", "No transfer", "daily per-site minima", no_transfer),
        summarize("zero shortage", "Full-foresight LP", "optimized allocation", lp),
        summarize("Shared site stocks and truck locations", "Full-foresight LP",
                  "Optimized stocks with fixed initial truck locations", shared_lp),
        summarize("Shared site stocks and truck locations", "No transfer",
                  "Same site stocks as fixed-start LP", none_shared),
        summarize("Shared site stocks and truck locations", "Rule-based",
                  "Same site stocks and truck locations as fixed-start LP", rule_shared),
        summarize("Rounded LP inventory budget", "No transfer", "Equal by site", none_low),
        summarize("Rounded LP inventory budget", "Rule-based", "Equal by site", rule_low),
        summarize("First zero-top-up budget on the grid", "No transfer", "Equal by site", none_tested),
        summarize("First zero-top-up budget on the grid", "Rule-based", "Equal by site", rule_tested),
    ]
    return pd.DataFrame(rows)


def main():
    trucks = 120
    report = build_report(trucks)
    out = write_dir(RUN) / "transfer_policy_comparison.csv"
    report.to_csv(out, index=False)
    upstream_dir = RESULTS / "coefficient_correction"
    provenance.record(
        out,
        inputs=[
            Path(__file__), Path(transfer_lp.__file__), Path(rebuild_inventory_chain.__file__),
            PARAMS_FILE, DATA / "02_channel_topology/port_topology.csv",
            DATA / "03_ship_parameters/172_ship_parameters.xlsx",
            DATA / "04_paper_baseline/segments_legs_with_timestamps.xlsx",
            upstream_dir / "segment_energy.csv", upstream_dir / "admiralty_pership.csv",
            upstream_dir / "transfer_frontier.csv",
        ],
        script=__file__,
        params_used={
            "trucks": trucks,
            "lookback_days": LOOKBACK_DAYS,
            "opening_stock_grid_packs_per_site": OPENING_STEP_PER_PORT,
            "buffers": "1 + maximum truck transit days times mean positive net draw over trailing days",
            "decision_time": "end of day, after its demand and stock releases",
            "shared_start": "LP site stocks with integer truck locations fixed by largest remainder; "
                            "LP dispatch is relaxed, rule dispatch is integer",
            "equal_start_sensitivity": "stocks and trucks split evenly, tested grid only",
            "dispatch": "shortest transit/distance loaded trips then empty repositioning",
            "scope": "DASM daily release and transport assumptions, including first outfitting",
            "terminal": "no target; truck and stock in transit retained in balances",
        },
    )


if __name__ == "__main__":
    main()
