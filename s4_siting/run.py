
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401,E402

from src import provenance  # noqa: E402
from src.loaders import DATA, ROOT, output_dir, params  # noqa: E402

RUN = "paper/s4_siting"
UPSTREAM = "paper/s3_energy"
CODE = Path(__file__).resolve().parents[1]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, CODE / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_reserve_sweep(out: Path) -> pd.DataFrame:
    det = _load("2_analysis/build_soc.py", "build_soc")
    rs = _load("1_pipeline/resolve_siting.py", "resolve_siting")
    import networkx as nx
    from src.loaders import ship_parameters
    graph = nx.Graph()
    for _, r in pd.read_csv(rs.TOPOLOGY).iterrows():
        graph.add_edge(r.start, r.end, distance=float(r.distance))
    router = rs.Router(graph)
    p = params()
    mask = 0
    for s_ in p["case"]["plan2_stations_8"]:
        mask |= rs.PORT_BIT.get(s_, 0)

    seg = pd.read_csv(output_dir(UPSTREAM) / "segment_energy.csv")
    s = ship_parameters().set_index("mmsi")
    zeta0 = float(s["depth_of_discharge"].iloc[0])
    eta = float(s["utilisation_efficiency"].iloc[0])

    rows = []
    served: dict[float, set] = {}
    for reserve in (0.20, 0.15, 0.10, 0.05, 0.00):
        zeta = 1.0 - reserve
        ok = set()
        for m, gg in seg.groupby("mmsi"):
            legs = [dict(e=r.energy_new, distance=r.distance,
                         start=r.start, end=r.end) for r in gg.itertuples()]
            bat = float(s.loc[m, "rated_capacity_kwh"]) * zeta / zeta0
            if det.feasible(legs, bat, mask, router):
                ok.add(m)
        served[zeta] = ok
        n = len(ok)
        rows.append(dict(reserve_held=reserve, depth_of_discharge=zeta,
                         usable_fraction=zeta * eta, feasible_ships=n,
                         fleet=172, coverage=n / 172,
                         is_manuscript_setting=abs(reserve - (1 - zeta0)) < 1e-9))
    t = pd.DataFrame(rows)
    t.to_csv(out / "soc_reserve.csv", index=False)

    levels = sorted(served)
    manuscript_zeta = round(zeta0, 10)
    flips = []
    for lo, hi in zip(levels, levels[1:]):
        lost = sorted(served[lo] - served[hi])
        flips.append(dict(comparison="step", zeta_from=lo, zeta_to=hi,
                          n_from=len(served[lo]), n_to=len(served[hi]),
                          n_newly_infeasible=len(lost),
                          mmsi="|".join(str(m) for m in lost)))
    top = max(levels)
    endpoint = sorted(served[manuscript_zeta] - served[top])
    flips.append(dict(comparison="manuscript_vs_zero",
                      zeta_from=manuscript_zeta, zeta_to=top,
                      n_from=len(served[manuscript_zeta]), n_to=len(served[top]),
                      n_newly_infeasible=len(endpoint),
                      mmsi="|".join(str(m) for m in endpoint)))
    pd.DataFrame(flips).to_csv(out / "reserve_flips_paper8.csv", index=False)
    return t


def build_tier_failure(out: Path) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Per-rating failure rates, and the nineteen-station ceiling."""
    import networkx as nx
    import resolve_siting as rs
    from src.loaders import ship_parameters

    det = _load("2_analysis/build_soc.py", "build_soc_for_tier")
    topo = pd.read_csv(rs.TOPOLOGY)
    graph = nx.Graph()
    for _, r in topo.iterrows():
        graph.add_edge(r.start, r.end, distance=float(r.distance))
    router = rs.Router(graph)

    seg = pd.read_csv(output_dir(UPSTREAM) / "segment_energy.csv")
    s = ship_parameters().set_index("mmsi")
    zeta0 = float(s["depth_of_discharge"].iloc[0])
    eta = float(s["utilisation_efficiency"].iloc[0])

    def mask_of(ports) -> int:
        m = 0
        for name in ports:
            m |= rs.PORT_BIT.get(name, 0)
        return m

    def feasible_set(mask: int, zeta: float) -> set:
        ok = set()
        for m, gg in seg.groupby("mmsi"):
            legs = [dict(e=r.energy_new, distance=r.distance,
                         start=r.start, end=r.end) for r in gg.itertuples()]
            bat = float(s.loc[m, "rated_capacity_kwh"]) * zeta / zeta0
            if det.feasible(legs, bat, mask, router):
                ok.add(m)
        return ok

    paper8 = params()["case"]["plan2_stations_8"]
    served = feasible_set(mask_of(paper8), zeta0)

    rows = []
    for rated, gg in s.groupby("rated_capacity_kwh"):
        ships = set(gg.index) & set(seg.mmsi.unique())
        if not ships:
            continue
        infeasible = ships - served
        max_leg = seg[seg.mmsi.isin(ships)].groupby("mmsi").distance.max()
        rows.append(dict(rated_kwh=float(rated), ships=len(ships),
                         infeasible=len(infeasible),
                         failure_rate=len(infeasible) / len(ships),
                         max_leg_nm_min=float(max_leg.min()),
                         max_leg_nm_p50=float(max_leg.median()),
                         max_leg_nm_max=float(max_leg.max())))
    tier = pd.DataFrame(rows).sort_values("rated_kwh").reset_index(drop=True)
    tier.to_csv(out / "tier_failure_paper8.csv", index=False)

    all19 = sorted(set(topo.start) | set(topo.end))
    cov_rows = []
    for label, ports in (("paper8", paper8), ("all19", all19)):
        mask = mask_of(ports)
        for reserve in (0.20, 0.15, 0.10, 0.05, 0.00):
            zeta = 1.0 - reserve
            n = len(feasible_set(mask, zeta))
            cov_rows.append(dict(station_set=label, n_stations=len(ports),
                                 reserve_held=reserve, depth_of_discharge=zeta,
                                 usable_fraction=zeta * eta,
                                 feasible_ships=n, fleet=172, coverage=n / 172,
                                 is_manuscript_setting=(label == "paper8"
                                                        and abs(reserve - (1 - zeta0)) < 1e-9)))
    cov19 = pd.DataFrame(cov_rows)
    cov19.to_csv(out / "coverage_all19.csv", index=False)

    fleet_p50 = float(seg.groupby("mmsi").distance.max().median())
    return tier, cov19, fleet_p50


def build_port_chainage(out: Path) -> pd.DataFrame:
    import networkx as nx

    topo = pd.read_csv(DATA / "02_channel_topology" / "port_topology.csv")
    g = nx.Graph()
    for _, r in topo.iterrows():
        g.add_edge(r.start, r.end, distance=float(r.distance))
    km = pd.Series({q: nx.shortest_path_length(g, "Anqing", q, weight="distance")
                    for q in g.nodes}).sort_values() * 1.852
    corridor = float(params()["case"]["corridor_km"])
    scale = corridor / float(km.max())
    t = pd.DataFrame(dict(port=km.index, km_topology=km.values,
                          km_from_anqing=(km * scale).values))
    t["share_of_corridor"] = t.km_from_anqing / corridor
    t.to_csv(out / "port_chainage.csv", index=False)
    return t


def build_siting_by_growth(out: Path, chain: pd.DataFrame) -> pd.DataFrame:
    """Where the stations move as batteries get denser."""
    cov = pd.read_csv(out / "coverage.csv")
    km = chain.set_index("port").km_from_anqing
    corridor = float(params()["case"]["corridor_km"])
    rows = []
    for _, r in cov.iterrows():
        ports = str(r.stations).split("|")
        d = km.reindex(ports)
        rows.append(dict(
            growth=int(r.growth), n_stations=int(r.n_stations), plan=r.plan,
            n_feasible=int(r.covered), stations="|".join(ports),
            mean_km=float(d.mean()), median_km=float(d.median()),
            min_km=float(d.min()), max_km=float(d.max()),
            spread_km=float(d.max() - d.min()),
            share_upstream_half=float((d < corridor / 2).mean())))
    t = pd.DataFrame(rows).sort_values(["plan", "n_stations", "growth"])
    t.to_csv(out / "siting_by_growth.csv", index=False)
    return t


def build_speed_sensitivity(out: Path) -> pd.DataFrame:
    import networkx as nx
    from src.loaders import ship_parameters
    rs = _load("1_pipeline/resolve_siting.py", "resolve_siting")
    det = _load("2_analysis/build_soc.py", "build_soc")
    graph = nx.Graph()
    for _, r in pd.read_csv(rs.TOPOLOGY).iterrows():
        graph.add_edge(r.start, r.end, distance=float(r.distance))
    router = rs.Router(graph)
    p = params()
    mask = 0
    for s_ in p["case"]["plan2_stations_8"]:
        mask |= rs.PORT_BIT.get(s_, 0)

    seg = pd.read_csv(output_dir(UPSTREAM) / "segment_energy.csv")
    s = ship_parameters().set_index("mmsi")
    cal = pd.read_csv(output_dir(UPSTREAM) / "admiralty_calibration.csv").set_index("mmsi")
    c = s["energy_curve"].str.strip("[] ").str.split(expand=True).astype(float)
    c.columns = list("abcd")
    scale = cal.C_new_kw / (s["admiralty_coefficient"].astype(float) * 1.36054)
    aux, ze = s["auxiliary_power_kw"], s["depth_of_discharge"] * s["utilisation_efficiency"]

    rows = []
    for k in (1.00, 0.95, 0.90, 0.85, 0.80):
        sog = (seg.real_speed + seg.avg_wspeed).values * k
        vw = sog - seg.w_new.values
        cf = c.reindex(seg.mmsi.values)
        pe = ((cf.a.values * vw ** 3 + cf.b.values * vw ** 2 + cf.c.values * vw
               + cf.d.values) / scale.reindex(seg.mmsi.values).values)
        e = ((pe + aux.reindex(seg.mmsi.values).values)
             / ze.reindex(seg.mmsi.values).values * (seg.distance.values / sog))
        d2 = seg.copy(); d2["e2"] = e
        n = 0
        for m, gg in d2.groupby("mmsi"):
            legs = [dict(e=r.e2, distance=r.distance, start=r.start, end=r.end)
                    for r in gg.itertuples()]
            if det.feasible(legs, float(s.loc[m, "rated_capacity_kwh"]), mask, router):
                n += 1
        ze_scalar = float((s["depth_of_discharge"] * s["utilisation_efficiency"]).iloc[0])
        rows.append(dict(speed_factor=k,
                         required_capacity_gwh=e.sum() / 1e6,
                         energy_drawn_gwh=e.sum() * ze_scalar / 1e6,
                         feasible_ships=n, fleet=172, coverage=n / 172))
    t = pd.DataFrame(rows)
    t.to_csv(out / "speed_scenarios.csv", index=False)
    return t


def main() -> None:
    out = output_dir(RUN)

    _load("1_pipeline/resolve_siting_scenarios.py", "resolve_siting_scenarios").main(
        RUN, UPSTREAM, "flowfield_rebuild")

    cov = pd.read_csv(out / "coverage_scenarios.csv")
    cov.to_csv(out / "coverage.csv", index=False)
    g0 = cov[(cov.growth == 0) & (cov.plan == "plan2")].sort_values("n_stations")
    for _, r in g0.iterrows():
        pass

    chain = build_port_chainage(out)
    sg = build_siting_by_growth(out, chain)
    for _, r in sg[(sg.plan == "plan2") & (sg.n_stations == 8)].iterrows():
        pass

    t = build_reserve_sweep(out)
    for _, r in t.iterrows():
        tag = "   <- paper as submitted (zeta=0.8)" if r.is_manuscript_setting else ""
    base = t[t.is_manuscript_setting].iloc[0]; top = t.iloc[-1]
    tier, cov19, fleet_p50 = build_tier_failure(out)
    lo = tier.iloc[0]
    worst = tier.loc[tier.infeasible.idxmax()]
    highest = tier[tier.infeasible > 0].iloc[-1]
    a19 = cov19[(cov19.station_set == "all19") & (cov19.reserve_held == 0.20)].iloc[0]
    flip = pd.read_csv(out / "reserve_flips_paper8.csv")
    step = flip[flip.comparison == "step"]
    distinct = {m for row in step.mmsi.fillna("") for m in row.split("|") if m}
    end = flip[flip.comparison == "manuscript_vs_zero"].iloc[0]

    sp = build_speed_sensitivity(out)
    for _, r in sp.iterrows():
        tag = "   <- this paper, observed speed" if r.speed_factor == 1.0 else ""
    slow = sp.loc[sp.speed_factor.idxmin()]

    inputs = [output_dir(UPSTREAM) / "segment_energy.csv",
              output_dir(UPSTREAM) / "admiralty_calibration.csv",
              DATA / "02_channel_topology" / "port_topology.csv"]
    inputs = [p for p in inputs if p.exists()]
    stamped = 0
    for prod in sorted(out.glob("*")):
        if prod.is_file() and not prod.name.endswith(provenance.SUFFIX):
            provenance.record(prod, script=__file__, inputs=inputs,
                              params_used={"case": params()["case"]})
            stamped += 1


if __name__ == "__main__":
    main()
