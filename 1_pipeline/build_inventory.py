
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401

from rebuild_inventory_chain import (  # noqa: E402
    build_plans, daily_plans, load_control, published, recur, table3,
)
from src.loaders import RESULTS, output_dir, params, ship_parameters  # noqa: E402
from src import provenance  # noqa: E402

COEF_RUN = RESULTS / "coefficient_correction"

STATION_SET = ["Nanjing Longtan", "Taicang", "Changshu Xinghua", "Zhangjiagang", "Yangzhou",
          "Jiangyin", "Yangshan", "Maanshan"]


def hp_factors() -> tuple[float, float]:
    inv = params()["inventory"]
    return inv["das_search_hp_factor_published"], inv["das_search_hp_factor_correct"]


def operations_record(stations: list[str],
                  source: Path | None = None,
                  flexible: bool = True,
                  growth_pct: float = 0.0) -> pd.DataFrame:
    """The operating record for one station set, in the chain's layout."""
    import networkx as nx
    from resolve_siting import PORT_BIT, Router, TOPOLOGY, replay

    topo = pd.read_csv(TOPOLOGY)
    graph = nx.Graph()
    for _, r in topo.iterrows():
        graph.add_edge(r.start, r.end, distance=float(r.distance))
    router = Router(graph)

    src = COEF_RUN if source is None else Path(source)
    seg = pd.read_csv(src / "segment_energy.csv")
    seg["leg"] = seg.groupby("mmsi").cumcount()
    ships = ship_parameters().set_index("mmsi")
    mask = sum(PORT_BIT[p] for p in stations)

    rows = []
    for mmsi, grp in seg.groupby("mmsi", sort=True):
        legs = grp[["start", "end", "distance", "energy_new"]].to_dict("records")
        battery = float(ships.loc[mmsi, "rated_capacity_kwh"]) * (1.0 + growth_pct / 100.0)
        rows += replay(legs, "energy_new", battery,
                       mask, router, mmsi, flexible=flexible)
    ops = pd.DataFrame(rows)

    k = seg.set_index(["mmsi", "leg"])
    idx = pd.MultiIndex.from_arrays([ops.mmsi, ops.leg])
    out = ops[["mmsi", "start", "end", "distance"]].copy()
    out["real_speed"] = k["v_water_new"].reindex(idx).values
    out["avg_wspeed"] = k["w_new"].reindex(idx).values
    out["energy_kwh"] = ops.energy.values
    out["sailing_h"] = out.distance / (out.real_speed + out.avg_wspeed)
    out["optimised swapping"] = ops.swapped.values
    out["feasible"] = ops.feasible.values
    out["cargo_port"] = ops.cargo_port.values
    return out


daily_by_run: dict = {}
trajectory_by_run: list = []


def run(label: str, ops: pd.DataFrame, ships: pd.DataFrame, spans: pd.DataFrame,
        ports: list[str], hp_factor: float, reference_filters: bool = True
        ) -> tuple[pd.DataFrame, dict]:
    plans, outfits, cohort = build_plans(ops, ships, spans, hp_factor,
                                         reference_filters)
    daily = daily_plans(plans, outfits, ports)
    daily_by_run[label] = daily
    traj: list = []
    t3 = table3(daily, traj)
    if traj:
        d = pd.concat(traj, ignore_index=True)
        d.insert(0, "run", label)
        trajectory_by_run.append(d)
    t3.insert(0, "run", label)
    cohort["served"] = int((outfits["feasible"] == 1).sum()) if len(outfits) else 0
    cohort.update(das_stability(plans, ships, daily))
    return t3, cohort


def das_stability(plans: dict, ships: pd.DataFrame, daily: dict) -> dict:
    """How hard the DAS rule is working, and whether it stays balanced."""
    swaps, amplitude, sheds, adds = [], [], 0, 0
    for mmsi, p in plans.items():
        swaps.append(int((p["optimised swapping"] == 1).sum()))
        held = int(float(ships.loc[mmsi, "rated_capacity_kwh"]) / float(ships.loc[mmsi, "pack_capacity_kwh"]))
        traj = []
        for k, kind in zip(p["packs_removed"], p["sign"]):
            held += -k if kind == 1 else (k if kind == 2 else 0)
            traj.append(held)
        if traj:
            amplitude.append(max(traj) - min(traj))
        sheds += int(((p["sign"] == 1) & (p["packs_removed"] > 0)).sum())
        adds += int((p["sign"] == 2).sum())

    das_l = sum(d["packs_landed_das"].sum() for d in daily.values() if len(d))
    das_u = sum(d["packs_taken_das"].sum() for d in daily.values() if len(d))
    worst = max(((abs(d["packs_landed_das"].sum() - d["packs_taken_das"].sum())
                  / max(d["packs_taken_das"].sum(), 1.0)), p)
                for p, d in daily.items() if len(d))
    return dict(swaps_median=float(np.median(swaps)) if swaps else 0.0,
                holding_amplitude=float(np.median(amplitude)) if amplitude else 0.0,
                add_share=adds / max(1, adds + sheds),
                das_imbalance=abs(das_l - das_u) / max(das_u, 1.0),
                worst_port_imbalance=worst[0], worst_port=worst[1])


def main() -> None:
    out = output_dir("coefficient_correction")
    paper_set = params()["case"]["plan2_stations_8"]
    ships = ship_parameters().set_index("mmsi")
    ops0, _, spans = load_control()

    per_ship = pd.read_csv(COEF_RUN / "admiralty_pership.csv").set_index("mmsi")
    ships_table = ships.copy()
    ships_table["admiralty_coefficient"] = per_ship["C_new_code"].reindex(ships_table.index)
    if ships_table["admiralty_coefficient"].isna().any():
        raise SystemExit("per-ship coefficients missing; run calibrate_admiralty_pership.py")

    hp_published, hp_correct = hp_factors()
    results, cohorts = [], {}
    a, cohorts["A control"] = run("A control", ops0, ships, spans, paper_set,
                                  hp_published)
    b, cohorts["B +hp"] = run("B +hp", ops0, ships, spans, paper_set, hp_correct)
    results += [a, b]

    ops3 = operations_record(paper_set)
    c, cohorts["C coefficient"] = run("C coefficient", ops3, ships_table, spans, paper_set, hp_correct)
    d, cohorts["D +cohort"] = run("D +cohort", ops3, ships_table, spans, paper_set,
                                  hp_correct, reference_filters=False)
    results += [c, d]

    t = pd.concat(results, ignore_index=True)
    t.to_csv(out / "inventory.csv", index=False)
    if trajectory_by_run:
        pd.concat(trajectory_by_run, ignore_index=True).to_csv(
            out / "inventory_daily.csv", index=False)

    for k, c in cohorts.items():
        pass

    for k, c in cohorts.items():
        worst = f"{c['worst_port']} {c['worst_port_imbalance']:.0%}"

    piv = t.pivot_table(index=["port", "mode"], columns="run", values="initial")
    order = [c for c in ["A control", "B +hp", "C coefficient", "D +cohort"]
             if c in piv.columns]
    piv = piv[order]
    pub = published()
    piv.insert(0, "published", [pub[p][m][0] if p in pub else np.nan
                                for p, m in piv.index])

    for metric, idx in [("average", 1), ("maximum", 2)]:
        p2 = t.pivot_table(index=["port", "mode"], columns="run", values=metric)[order]
        p2.insert(0, "published", [pub[p][m][idx] if p in pub else np.nan
                                   for p, m in p2.index])

    for metric in ("initial", "average", "maximum"):
        s = t.groupby(["run", "mode"])[metric].sum().unstack(0)[order]
        for mode in ("EAS", "DAS"):
            row = s.loc[mode]
            base = row["A control"]
            txt = "  ".join(f"{c}: {row[c]:>8.0f} ({row[c] / base - 1:+6.1%})"
                            for c in order)

    hor = []
    for name, dailies in daily_by_run.items():
        for mode, lc, uc in [("EAS", "packs_landed_eas", "packs_taken_eas"),
                             ("DAS", "packs_landed_das", "packs_taken_das")]:
            vals = []
            for nd in (30, 45, 60, 90):
                vals.append(sum(recur(d[lc].values[:nd], d[uc].values[:nd])["initial"]
                                for d in dailies.values() if len(d)))
            hor.append(dict(run=name, mode=mode, d30=vals[0], d45=vals[1],
                            d60=vals[2], d90=vals[3], ratio=vals[3] / max(vals[0], 1)))
    pd.DataFrame(hor).to_csv(out / "inventory_horizon.csv", index=False)
    provenance.record(out / "inventory_horizon.csv", script=__file__,
                      inputs=[COEF_RUN / "segment_energy.csv"],
                      params_used={"stations": paper_set})


if __name__ == "__main__":
    main()
