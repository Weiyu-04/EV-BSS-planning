
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401

from resolve_siting import (  # noqa: E402
    CANDIDATE_PORTS, PORT_BIT, FlatFleet, Router, TOPOLOGY, replay, sweep,
)
from src.loaders import RESULTS, output_dir, params, ship_parameters  # noqa: E402

COEF_RUN = RESULTS / "coefficient_correction"


def main(run: str = "coefficient_correction", energy_from: str | None = None) -> None:
    out = output_dir(run)
    src = output_dir(energy_from) if energy_from else out
    counts = ([int(a) for a in sys.argv[1:]] if len(sys.argv) > 1
              else params()["case"]["station_counts"])
    growths = params()["case"]["growth_sweep"]

    topo = pd.read_csv(TOPOLOGY)
    graph = nx.Graph()
    for _, r in topo.iterrows():
        graph.add_edge(r.start, r.end, distance=float(r.distance))
    router = Router(graph)

    d = pd.read_csv(src / "segment_energy.csv")
    ships = ship_parameters().set_index("mmsi")

    ship_legs, battery = [], []
    for mmsi, grp in d.groupby("mmsi", sort=True):
        ship_legs.append(grp[["start", "end", "distance",
                              "energy_field", "energy_new"]].to_dict("records"))
        battery.append(float(ships.loc[mmsi, "rated_capacity_kwh"]))
    battery = np.asarray(battery, float)

    rows = []
    for growth in growths:
        bat = battery * (1 + growth / 100.0)
        flat = FlatFleet(ship_legs, "energy_new")
        for n in counts:
            rows += [dict(r, growth=growth) for r in
                     sweep(ship_legs, "energy_new", bat, router, flat, n,
                           f"growth{growth}")]

    scen = pd.DataFrame(rows)
    scen.to_csv(out / "coverage_scenarios.csv", index=False)

    piv = scen.pivot_table(index=["n_stations", "plan"], columns="growth",
                         values="covered")
    piv.columns = [f"growth {c}%" for c in piv.columns]

    for growth in growths:
        r = scen[(scen.growth == growth) & (scen.n_stations == 8)
                 & (scen.plan == "plan2")].iloc[0]

    r0 = scen[(scen.growth == 0) & (scen.n_stations == 8) & (scen.plan == "plan2")].iloc[0]
    mask = sum(PORT_BIT[p] for p in r0.stations.split("|"))
    mmsis = sorted(d.mmsi.unique())
    ops = pd.DataFrame([row for s, legs in enumerate(ship_legs)
                        for row in replay(legs, "energy_new", battery[s], mask,
                                          router, mmsis[s])])
    ops.to_csv(out / "operations_8stn.csv", index=False)


if __name__ == "__main__":
    main()
