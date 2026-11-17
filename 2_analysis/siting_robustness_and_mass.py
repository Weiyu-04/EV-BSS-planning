
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401,E402

from resolve_siting import (  # noqa: E402
    CANDIDATE_PORTS, PORT_BIT, FlatFleet, Router, TOPOLOGY, plan2_feasible, relevant_mask, sweep,
)
from src.loaders import RESULTS, params, ship_parameters, write_dir  # noqa: E402

RUN = "siting_robustness_and_mass"
MAANSHAN_8 = ["Nanjing Longtan", "Taicang", "Changshu Xinghua", "Zhangjiagang", "Yangzhou", "Jiangyin", "Yangshan", "Maanshan"]


def load_legs():
    d = pd.read_csv(RESULTS / "coefficient_correction/segment_energy.csv")
    ships = ship_parameters().set_index("mmsi")
    legs, battery, mmsis = [], [], []
    for mmsi, g in d.groupby("mmsi", sort=True):
        v = ships.loc[mmsi]
        derate = float(v["depth_of_discharge"] * v["utilisation_efficiency"])
        hours = g.distance / (g.v_water_new + g.w_new)
        aux_rated = hours * float(v["auxiliary_power_kw"]) / derate
        rec = g[["start", "end", "distance", "energy_new"]].copy()
        rec["aux_rated"] = aux_rated.values
        legs.append(rec.to_dict("records"))
        battery.append(float(v["rated_capacity_kwh"]))
        mmsis.append(mmsi)
    return legs, np.asarray(battery), mmsis, ships


def layout_cover(legs, battery, mask, router, key="energy_new"):
    return np.array([plan2_feasible(l, key, b, mask, router) for l, b in zip(legs, battery)])


def main():
    p = params()
    topo = pd.read_csv(TOPOLOGY)
    g = nx.Graph()
    for _, r in topo.iterrows():
        g.add_edge(r.start, r.end, distance=float(r.distance))
    router = Router(g)
    legs, battery, mmsis, ships = load_legs()
    wuhu = sum(PORT_BIT[x] for x in p["case"]["plan2_stations_8"])
    maan = sum(PORT_BIT[x] for x in MAANSHAN_8)
    out = write_dir(RUN)

    rows = []
    for gr in p["case"]["growth_sweep"]:
        b = battery * (1 + gr / 100)
        rows.append(dict(growth_pct=gr,
                         wuhu_layout=int(layout_cover(legs, b, wuhu, router).sum()),
                         maanshan_layout=int(layout_cover(legs, b, maan, router).sum())))
    rob = pd.DataFrame(rows)
    rob.to_csv(out / "layout_coverage_by_density.csv", index=False)

    masks = [sum(1 << i for i in c) for c in itertools.combinations(range(len(CANDIDATE_PORTS)), 8)]
    cov = np.zeros(len(masks), int)
    for s, l in enumerate(legs):
        rel = relevant_mask(l, router)
        cache = {}
        for j, m in enumerate(masks):
            k = m & rel
            if k not in cache:
                cache[k] = plan2_feasible(l, "energy_new", battery[s], k, router)
            cov[j] += cache[k]
    ties = pd.Series(cov).value_counts().sort_index(ascending=False).head(4)
    ties.rename_axis("covered").rename("layouts").to_csv(out / "eight_station_coverage_distribution.csv")

    flat = FlatFleet(legs, "energy_new")
    ext = []
    for n in range(4, 13):
        ext += [r for r in sweep(legs, "energy_new", battery, router, flat, n, "N sweep")]
    pd.DataFrame(ext).to_csv(out / "coverage_4_to_12.csv", index=False)

    mass = float(p["inventory"]["battery_unit_mass_t"])
    heavy = []
    for l, mmsi in zip(legs, mmsis):
        v = ships.loc[mmsi]
        disp = float(v.avg_water)
        f = ((disp + int(v["packs"]) * mass) / disp) ** (2 / 3)
        heavy.append([dict(x, energy_heavy=x["aux_rated"] + (x["energy_new"] - x["aux_rated"]) * f)
                      for x in l])
    base_ok = layout_cover(legs, battery, wuhu, router)
    heavy_ok = layout_cover(heavy, battery, wuhu, router, "energy_heavy")
    e0 = sum(x["energy_new"] for l in legs for x in l)
    e1 = sum(x["energy_heavy"] for l in heavy for x in l)
    rel_mass = pd.Series({m: int(ships.loc[m, "packs"]) * mass / float(ships.loc[m, "avg_water"])
                          for m in mmsis})
    res = pd.Series(dict(
        covered_reference=int(base_ok.sum()), covered_with_battery_mass=int(heavy_ok.sum()),
        lost_ships=int((base_ok & ~heavy_ok).sum()),
        fleet_energy_increase_pct=100 * (e1 / e0 - 1),
        battery_mass_share_of_displacement_median_pct=100 * rel_mass.median(),
        battery_mass_share_max_pct=100 * rel_mass.max(),
        any_aux_negative_propulsion=int(sum(x["energy_new"] < x["aux_rated"] for l in legs for x in l)),
    ))
    res.to_csv(out / "battery_mass_upper_bound.csv", header=["value"])


if __name__ == "__main__":
    main()
