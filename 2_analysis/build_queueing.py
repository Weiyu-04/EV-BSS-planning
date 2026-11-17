
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401

from build_inventory import hp_factors, operations_record  # noqa: E402
from rebuild_inventory_chain import build_plans, load_control  # noqa: E402
from src import provenance  # noqa: E402
from src.loaders import (  # noqa: E402
    RESULTS, output_dir, params, segments_with_timestamps, ship_parameters,
)


UPSTREAM_DEFAULT = "coefficient_correction"
_UPSTREAM = UPSTREAM_DEFAULT


def leg_clock() -> pd.DataFrame:
    """Per AIS leg: its real end timestamp and its measured underway hours."""
    t = segments_with_timestamps().copy()
    t["start_time"] = pd.to_datetime(t["start_time"])
    t["end_time"] = pd.to_datetime(t["end_time"])
    t["leg"] = t.groupby("mmsi").cumcount()

    u = pd.read_csv(RESULTS / _UPSTREAM / "ais_underway.csv")
    if len(u) == 0:
        raise SystemExit("run build_ais_dwell.py first")
    k = u.set_index(["mmsi", "leg"])["underway_h"]
    idx = pd.MultiIndex.from_arrays([t.mmsi, t.leg])
    t["sail_h"] = k.reindex(idx).values
    miss = t.sail_h.isna() | (t.sail_h <= 0)
    t.loc[miss, "sail_h"] = t.loc[miss, "sailing_h"]
    t.attrs["fallback_legs"] = int(miss.sum())
    return t[["mmsi", "leg", "start", "end", "start_time", "end_time",
              "sail_h", "distance"]]


def berth_windows() -> pd.DataFrame:
    """Measured continuous berthings, indexed for lookup by ship and port."""
    b = pd.read_csv(RESULTS / _UPSTREAM / "ais_berthing.csv",
                    parse_dates=["start", "end"])
    return b[b.port.notna()].sort_values(["mmsi", "port", "start"])


def replenishment_energy(ops: pd.DataFrame) -> np.ndarray:
    """Energy each stop puts back, in the pack, aligned to `ops`."""
    out = np.zeros(len(ops))
    energy = ops.energy.values
    swapped = ops.swapped.values
    feasible = ops.feasible.values
    mmsi = ops.mmsi.values
    pending, prev = 0.0, None
    for i in range(len(ops)):
        if mmsi[i] != prev:
            pending, prev = 0.0, mmsi[i]
        if feasible[i] == 1:
            pending += energy[i]
            if swapped[i] == 1:
                out[i] = pending
                pending = 0.0
        else:
            pending = 0.0
    return out


def arrival_stream(ports: list[str], flexible: bool = True,
                   growth_pct: float = 0.0) -> pd.DataFrame:
    """The recurrent swaps used by the queueing and time analyses."""
    events, _ = swap_schedule(ports, flexible, growth_pct)
    return events


def swap_schedule(ports: list[str], flexible: bool = True,
                  growth_pct: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    import networkx as nx

    from resolve_siting import PORT_BIT, Router, TOPOLOGY, replay  # noqa: E402

    ships = ship_parameters().set_index("mmsi")
    per_ship = pd.read_csv(RESULTS / _UPSTREAM / "admiralty_pership.csv"
                           ).set_index("mmsi")
    ships_table = ships.copy()
    ships_table["admiralty_coefficient"] = per_ship["C_new_code"].reindex(ships_table.index)
    _, _, spans = load_control()
    _, hp_correct = hp_factors()

    topo = pd.read_csv(TOPOLOGY)
    g = nx.Graph()
    for _, r in topo.iterrows():
        g.add_edge(r.start, r.end, distance=float(r.distance))
    router = Router(g)
    seg = pd.read_csv(RESULTS / _UPSTREAM / "segment_energy.csv")
    seg["leg"] = seg.groupby("mmsi").cumcount()
    mask = sum(PORT_BIT[p] for p in ports)
    rows = []
    for mmsi, grp in seg.groupby("mmsi", sort=True):
        legs = grp[["start", "end", "distance", "energy_new"]].to_dict("records")
        battery = float(ships.loc[mmsi, "rated_capacity_kwh"]) * (1.0 + growth_pct / 100.0)
        rows += replay(legs, "energy_new", battery,
                       mask, router, mmsi, flexible=flexible)
    ops_full = pd.DataFrame(rows)
    ops_full["restored_kwh"] = replenishment_energy(ops_full)

    k = seg.set_index(["mmsi", "leg"])
    key = pd.MultiIndex.from_arrays([ops_full.mmsi, ops_full.leg])
    ops = ops_full[["mmsi", "start", "end", "distance"]].copy()
    ops["real_speed"] = k["v_water_new"].reindex(key).values
    ops["avg_wspeed"] = k["w_new"].reindex(key).values
    ops["energy_kwh"] = ops_full.energy.values
    ops["sailing_h"] = ops.distance / (ops.real_speed + ops.avg_wspeed)
    ops["optimised swapping"] = ops_full.swapped.values
    ops["feasible"] = ops_full.feasible.values
    plans, outfits, _ = build_plans(ops, ships_table, spans, hp_correct,
                              reference_filters=False)

    clock_df = leg_clock()
    clock = clock_df.set_index(["mmsi", "leg"])

    ops_full["cum_d"] = ops_full.groupby(["mmsi", "leg"]).distance.cumsum()
    ops_full["leg_d"] = ops_full.groupby(["mmsi", "leg"]).distance.transform("sum")
    arrive_end = clock["end_time"].reindex(key).values
    sail = clock["sail_h"].reindex(key).values
    share = np.where(ops_full.leg_d.values > 0,
                     ops_full.cum_d.values / ops_full.leg_d.values, 1.0)
    ops_full["arrive"] = arrive_end - pd.to_timedelta((1 - share) * sail, unit="h")

    b = berth_windows()
    win = np.zeros(len(ops_full))
    bykey = {k: g for k, g in b.groupby(["mmsi", "port"], sort=False)}
    arr = ops_full.arrive.values
    for i, (m, p, cargo) in enumerate(zip(ops_full.mmsi.values, ops_full.end.values,
                                          ops_full.cargo_port.values)):
        if cargo != 1:
            continue
        g = bykey.get((m, p))
        if g is None:
            continue
        j = int(np.argmin(np.abs(g.start.values - arr[i])))
        win[i] = g.hours.values[j]
    ops_full["window_h"] = win

    ev = ops_full[(ops_full.swapped == 1) & (ops_full.feasible == 1)].copy()
    ev = ev[ev.end.isin(ports)].copy()
    ev["order"] = ev.groupby("mmsi").cumcount()

    packs = []
    for mmsi, p in plans.items():
        s = p[p["optimised swapping"] == 1]
        packs.append(pd.DataFrame(dict(mmsi=mmsi, order=range(len(s)),
                                       packs=s["packs_swapped_eas"].values)))
    packs = pd.concat(packs, ignore_index=True)

    n_before = len(ev)
    ev = ev.merge(packs, on=["mmsi", "order"], how="inner")
    assert len(ev) == n_before, f"swap events lost in the pack join: {n_before} -> {len(ev)}"
    assert ev.arrive.notna().all(), "swap event without a clock time"
    ev = ev[["mmsi", "end", "arrive", "packs", "window_h", "cargo_port",
             "restored_kwh"]].rename(columns={"end": "port"})
    ev.attrs["feasible"] = ops_full.groupby("mmsi").feasible.min().to_dict()
    return ev, outfits


def simulate(arrivals: np.ndarray, service: np.ndarray, c: int) -> np.ndarray:
    """FCFS wait times at a station with `c` identical berths."""
    order = np.argsort(arrivals, kind="stable")
    free = np.zeros(c)
    wait = np.empty(len(arrivals))
    for i in order:
        b = int(np.argmin(free))
        start = max(arrivals[i], free[b])
        wait[i] = start - arrivals[i]
        free[b] = start + service[i]
    return wait


def main(run: str = UPSTREAM_DEFAULT,
         upstream: str = UPSTREAM_DEFAULT) -> None:
    global _UPSTREAM
    _UPSTREAM = upstream
    out = output_dir(run)
    q = params()["queueing"]
    ports = params()["case"]["plan2_stations_8"]

    ev = arrival_stream(ports)
    ev["service_h"] = q["fixed_hours"] + q["per_pack_hours"] * ev.packs

    med_full = q["fixed_hours"] + q["per_pack_hours"] * q["median_packs_full_swap"]
    assert abs(med_full - q["paper_swap_hours"]) < 1e-3, med_full
    assert int(ev.mmsi.nunique()) == 154, f"{ev.mmsi.nunique()} ships swapping"
    assert set(ev.port) <= set(ports), f"swap at a non-station: {set(ev.port) - set(ports)}"
    pct = float((ev.service_h < q["paper_swap_hours"]).mean() * 100)

    t0 = ev.arrive.min()
    ev["t_h"] = (ev.arrive - t0).dt.total_seconds() / 3600.0

    rows = []
    for port, g in ev.groupby("port"):
        a = g.t_h.values
        s = g.service_h.values
        for c in q["berth_sweep"]:
            w = simulate(a, s, c)
            rows.append(dict(port=port, berths=c, n=len(g),
                             wait_mean=w.mean(), wait_p50=np.percentile(w, 50),
                             wait_p95=np.percentile(w, 95), wait_max=w.max(),
                             util=s.sum() / (c * (a.max() - a.min())),
                             absorbed=float(np.mean(w + s <= g.window_h.values))))
    d = pd.DataFrame(rows)
    d.to_csv(out / "queueing.csv", index=False)

    need = {}
    for port, g in d.groupby("port"):
        g = g.sort_values("berths")
        ok = g[g.wait_p95 <= q["service_level_wait_h"]]
        c_need = int(ok.berths.iloc[0]) if len(ok) else int(g.berths.max())
        need[port] = c_need
        u = float(g[g.berths == c_need].util.iloc[0])
        p = {int(r.berths): r.wait_p95 for _, r in g.iterrows()}

    sized = {}
    for c_label, c in [("berths per station as tabulated above", None), ("per station 1  berths", 1)]:
        n_abs, n_tot, extra, per_ship = 0, 0, [], []
        for port, g in ev.groupby("port"):
            cc = need[port] if c is None else c
            w = simulate(g.t_h.values, g.service_h.values, cc)
            fits = (w + g.service_h.values) <= g.window_h.values
            n_abs += int(fits.sum()); n_tot += len(g)
            e = np.where(fits, 0.0, w + g.service_h.values - g.window_h.values)
            extra.append(e)
            per_ship.append(pd.DataFrame(dict(mmsi=g.mmsi.values, extra=e,
                                              service=g.service_h.values)))
        e = np.concatenate(extra)
        share = n_abs / n_tot
        if c is None:
            sized = dict(share=share, extra_total=float(e.sum()),
                         per_ship=pd.concat(per_ship, ignore_index=True))

    cargo = ev.cargo_port == 1

    sail = segments_with_timestamps().groupby("mmsi")["sailing_h"].sum()
    ps = sized["per_ship"].groupby("mmsi").agg(extra=("extra", "sum"),
                                               service=("service", "sum"))
    n_ev = ev.groupby("mmsi").size()
    base = sail.reindex(ps.index)
    flat = n_ev.reindex(ps.index) * q["paper_swap_hours"]

    svc = ev.groupby("mmsi").service_h.sum().reindex(ps.index)
    steps = [("1 paper convention: constant at 2 h, no queueing, berthing not credited", flat),
             ("2 service time only, scaled by pack count", svc),
             ("3 +queueing+credited against the measured berthing window", ps.extra)]
    prev = None
    for label, series in steps:
        pen = series / base * 100
        prev = pen
    p1 = (flat / base * 100).quantile(.75)
    p2 = (svc / base * 100).quantile(.75)
    p3 = (ps.extra / base * 100).quantile(.75)

    pd.DataFrame([dict(sail_total_h=float(sail.sum()),
                       paper_flat_h=float(flat.sum()),
                       full_chain_h=sized["extra_total"],
                       absorbed_share=sized["share"],
                       pen_flat_p75=float(p1), pen_service_only_p75=float(p2),
                       pen_full_p75=float(p3),
                       berths_total=sum(need.values()))]).to_csv(
        out / "queueing_summary.csv", index=False)
    provenance.record(out / "queueing_summary.csv", script=__file__,
                      inputs=[RESULTS / _UPSTREAM / "segment_energy.csv",
                              RESULTS / _UPSTREAM / "ais_berthing.csv"],
                      params_used=params()["queueing"])


if __name__ == "__main__":
    main()
