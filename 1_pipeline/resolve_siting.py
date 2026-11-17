
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401

from src.loaders import DATA, output_dir, params, ship_parameters  # noqa: E402

TOPOLOGY = (Path(__file__).resolve().parents[2] / "03_data" / "02_channel_topology"
            / "port_topology.csv")
SEGMENT_FILE = DATA / "04_paper_baseline" / "segments_legs_energy_and_time.xlsx"

CANDIDATE_PORTS = [
    "Taicang", "Nanjing Longtan", "Yangshan", "Zhangjiagang", "Yangzhou", "Changshu Xinghua",
    "Shanghai Waigaoqiao", "Luojing", "Wuhu", "Nantong", "Taizhou", "Jiangyin",
    "Jingjiang", "Jiangdu", "Maanshan", "Anqing", "Tongling", "Chizhou", "Beisha",
]
PORT_BIT = {p: 1 << i for i, p in enumerate(CANDIDATE_PORTS)}


class FlatFleet:
    """All ships' legs concatenated, for the vectorised Plan 1 test."""

    def __init__(self, ship_legs: list[list[dict]], energy_key: str):
        energy, port, ship = [], [], []
        for s, legs in enumerate(ship_legs):
            for leg in legs:
                energy.append(leg[energy_key])
                port.append(PORT_BIT.get(leg["end"], 0))
                ship.append(s)
        self.energy = np.asarray(energy, float)
        self.bit = np.asarray(port, np.int64)
        self.ship = np.asarray(ship, int)
        self.cum = np.cumsum(self.energy)
        self.starts = np.flatnonzero(np.diff(self.ship, prepend=-1) != 0)
        self.is_first = np.zeros(len(self.energy), bool)
        self.is_first[self.starts] = True
        self._first_base = self.cum[self.is_first] - self.energy[self.is_first]

    def feasible(self, mask: int, battery: np.ndarray) -> np.ndarray:
        reset = (self.bit & mask) != 0
        marker = np.where(reset, self.cum, -np.inf)
        shifted = np.empty_like(marker)
        shifted[0] = -np.inf
        shifted[1:] = marker[:-1]
        shifted[self.is_first] = self._first_base
        base = np.maximum.accumulate(shifted)
        worst = np.maximum.reduceat(self.cum - base, self.starts)
        return worst <= battery


class Router:
    """Shortest paths and distances on the port graph, precomputed."""

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.dist = dict(nx.all_pairs_dijkstra_path_length(graph, weight="distance"))
        self.path = dict(nx.all_pairs_dijkstra_path(graph, weight="distance"))

    def split(self, start: str, end: str, mask: int) -> list[tuple[str, float]]:
        on_path = [p for p in self.path[start][end]
                   if p not in (start, end) and (PORT_BIT.get(p, 0) & mask)]
        if not on_path:
            return [(end, self.dist[start][end])]
        out, cur = [], start
        while on_path:
            nearest = min(on_path, key=lambda p: self.dist[cur][p])
            out.append((nearest, self.dist[cur][nearest]))
            cur = nearest
            on_path.remove(nearest)
        out.append((end, self.dist[cur][end]))
        return out


def plan2_feasible(legs: list[dict], energy_key: str, battery: float,
                   mask: int, router: Router) -> bool:
    bottleneck, remaining = set(), battery
    for i, leg in enumerate(legs):
        e = leg[energy_key]
        if e > battery or e > remaining:
            bottleneck.add(i)
        remaining = battery - e if e > remaining else remaining - e
        if PORT_BIT.get(leg["end"], 0) & mask:
            remaining = battery

    remaining = battery
    for i, leg in enumerate(legs):
        if i in bottleneck:
            pieces = router.split(leg["start"], leg["end"], mask)
            span = leg["distance"]
            for end_port, d in pieces:
                e = leg[energy_key] * (d / span) if span > 0 else leg[energy_key]
                if e > battery or e > remaining:
                    return False
                remaining -= e
                if PORT_BIT.get(end_port, 0) & mask:
                    remaining = battery
        else:
            e = leg[energy_key]
            if e > battery or e > remaining:
                return False
            remaining -= e
            if PORT_BIT.get(leg["end"], 0) & mask:
                remaining = battery
    return True


def replay(legs: list[dict], energy_key: str, battery: float, mask: int,
           router: Router, mmsi, flexible: bool = True) -> list[dict]:
    bottleneck, remaining = set(), battery
    for i, leg in enumerate(legs):
        e = leg[energy_key]
        if e > battery or e > remaining:
            bottleneck.add(i)
        remaining = battery - e if e > remaining else remaining - e
        if PORT_BIT.get(leg["end"], 0) & mask:
            remaining = battery
    if not flexible:
        bottleneck = set()

    rows, remaining, cum, feasible = [], battery, 0.0, True
    for i, leg in enumerate(legs):
        span = leg["distance"]
        pieces = (router.split(leg["start"], leg["end"], mask)
                  if i in bottleneck else [(leg["end"], span)])
        for end_port, d in pieces:
            e = (leg[energy_key] * (d / span)
                 if (i in bottleneck and span > 0) else leg[energy_key])
            if e > battery or e > remaining:
                feasible = False
            else:
                remaining -= e
            cum += e
            swapped = bool(PORT_BIT.get(end_port, 0) & mask)
            if swapped:
                remaining = battery
            rows.append(dict(mmsi=mmsi, leg=i, start=leg["start"], end=end_port,
                             distance=d, energy=e, cum_energy=cum,
                             swapped=int(swapped),
                             cargo_port=int(end_port == leg["end"]),
                             split=int(i in bottleneck)))
    for r in rows:
        r["feasible"] = int(feasible)
    return rows


def relevant_mask(legs: list[dict], router: Router) -> int:
    m = 0
    for leg in legs:
        for p in router.path[leg["start"]][leg["end"]]:
            m |= PORT_BIT.get(p, 0)
    return m


def sweep(ship_legs, energy_key, battery, router, flat, n_stations, label):
    """Exhaustive fixed-N search over all combinations."""
    masks = np.array([sum(1 << i for i in c)
                      for c in itertools.combinations(range(len(CANDIDATE_PORTS)),
                                                      n_stations)], np.int64)

    cov1 = np.empty(len(masks), int)
    for j, m in enumerate(masks):
        cov1[j] = int(flat.feasible(int(m), battery).sum())

    ok2 = np.zeros((len(ship_legs), len(masks)), bool)
    sims = 0
    for s, legs in enumerate(ship_legs):
        rel = relevant_mask(legs, router)
        keys = masks & rel
        uniq, inverse = np.unique(keys, return_inverse=True)
        verdict = np.empty(len(uniq), bool)
        for u, key in enumerate(uniq):
            verdict[u] = plan2_feasible(legs, energy_key, battery[s], int(key), router)
        sims += len(uniq)
        ok2[s] = verdict[inverse]
    cov2 = ok2.sum(axis=0)

    b1, b2 = int(cov1.argmax()), int(cov2.argmax())

    def ports_of(mask_val: int) -> str:
        return "|".join(sorted(p for p, bit in PORT_BIT.items() if bit & int(mask_val)))

    return [dict(source=label, n_stations=n_stations, plan="plan1",
                 covered=int(cov1[b1]), fleet=len(battery),
                 coverage=cov1[b1] / len(battery), stations=ports_of(masks[b1])),
            dict(source=label, n_stations=n_stations, plan="plan2",
                 covered=int(cov2[b2]), fleet=len(battery),
                 coverage=cov2[b2] / len(battery), stations=ports_of(masks[b2]))]


def build_ship_legs(seg: pd.DataFrame) -> tuple[list[list[dict]], np.ndarray]:
    ships = ship_parameters().set_index("mmsi")
    ship_legs, battery = [], []
    for mmsi, grp in seg.groupby("mmsi", sort=True):
        ship_legs.append(grp[["start", "end", "distance", "energy_kwh", "energy_new"]]
                         .to_dict("records"))
        battery.append(float(ships.loc[mmsi, "rated_capacity_kwh"]))
    return ship_legs, np.asarray(battery, float)


def main() -> None:
    out = output_dir("flowfield_rebuild")
    counts = ([int(a) for a in sys.argv[1:]] if len(sys.argv) > 1
              else params()["case"]["station_counts"])

    topo = pd.read_csv(TOPOLOGY)
    graph = nx.Graph()
    for _, r in topo.iterrows():
        graph.add_edge(r.start, r.end, distance=float(r.distance))
    router = Router(graph)

    seg = pd.read_excel(SEGMENT_FILE)
    reb = pd.read_csv(out / "segment_energy_rebuilt.csv")
    seg = seg.assign(energy_new=reb.energy_new.values)
    ship_legs, battery = build_ship_legs(seg)

    rows = []
    for key, label in [("energy_kwh", "control (published energy)"),
                       ("energy_new", "rebuilt energy")]:
        flat = FlatFleet(ship_legs, key)
        for n in counts:
            rows += sweep(ship_legs, key, battery, router, flat, n, label)

    both = pd.DataFrame(rows)
    published = {(4, "plan1"): 107, (4, "plan2"): 142, (5, "plan1"): 117,
                 (5, "plan2"): 151, (6, "plan1"): 126, (6, "plan2"): 158,
                 (7, "plan1"): 132, (7, "plan2"): 163, (8, "plan1"): 136,
                 (8, "plan2"): 165, (9, "plan1"): 138, (9, "plan2"): 168}
    ctrl = both[both.source.str.startswith("control")].copy()
    ctrl["published"] = [published.get((r.n_stations, r.plan)) for _, r in ctrl.iterrows()]
    ctrl["matches"] = ctrl.covered == ctrl.published


    piv = both.pivot_table(index=["n_stations", "plan"], columns="source", values="covered")
    if piv.shape[1] == 2:
        piv["delta"] = piv["rebuilt energy"] - piv["control (published energy)"]

    for src in both.source.unique():
        sel = both[(both.source == src) & (both.n_stations == 8) & (both.plan == "plan2")]
        if len(sel):
            r = sel.iloc[0]

    both.to_csv(out / "coverage_resolved.csv", index=False)

    mmsis = sorted(seg.mmsi.unique())
    for key, label, tag in [("energy_kwh", "control (published energy)", "published"),
                            ("energy_new", "rebuilt energy", "rebuilt")]:
        sel = both[(both.source == label) & (both.n_stations == 8)
                   & (both.plan == "plan2")]
        if not len(sel):
            continue
        mask = sum(PORT_BIT[p] for p in sel.iloc[0].stations.split("|"))
        rows = []
        for s, legs in enumerate(ship_legs):
            rows += replay(legs, key, battery[s], mask, router, mmsis[s])
        ops = pd.DataFrame(rows)
        ops.to_csv(out / f"operations_8stn_{tag}.csv", index=False)
        covered = ops.groupby("mmsi").feasible.max().sum()


if __name__ == "__main__":
    main()
