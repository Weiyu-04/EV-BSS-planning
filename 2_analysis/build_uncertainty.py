
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401

from resolve_siting import (  # noqa: E402
    PORT_BIT, FlatFleet, Router, TOPOLOGY, plan2_feasible,
)
from src.loaders import RESULTS, output_dir, params, ship_parameters, write_dir  # noqa: E402
from src import provenance  # noqa: E402

_R = params()["robustness"]
SEED = int(_R["seed"])
N_DRAWS = int(_R["n_draws"])

SIGMA_CURRENT_FLEET = float(_R["sigma_current_fleet"])
SIGMA_CURRENT_LEG = float(_R["sigma_current_leg"])
SIGMA_SPEED_LEG = float(_R["sigma_speed_leg"])
SIGMA_AUX_FLEET = float(_R["sigma_aux_fleet"])


UPSTREAM_DEFAULT = "coefficient_correction"


def load_chain(upstream: str = UPSTREAM_DEFAULT
               ) -> tuple[pd.DataFrame, pd.DataFrame, Router]:
    seg = pd.read_csv(RESULTS / upstream / "segment_energy.csv")
    ships = ship_parameters().set_index("mmsi")
    per_ship = pd.read_csv(RESULTS / upstream / "admiralty_pership.csv"
                           ).set_index("mmsi")

    coef = ships["energy_curve"].str.strip("[] ").str.split(expand=True).astype(float)
    coef.columns = ["a", "b", "c", "d"]
    k = params()["admiralty"]["code_convention_factor"]
    scale = (per_ship["C_new_kw"] / (ships["admiralty_coefficient"] * k)).reindex(ships.index)

    m = seg.mmsi.values
    seg["sog"] = seg.real_speed.values + seg.avg_wspeed.values
    seg["w"] = seg.w_new.values
    for col in ("a", "b", "c", "d"):
        seg[col] = coef[col].reindex(m).values
    seg["scale"] = scale.reindex(m).values
    seg["aux"] = ships["auxiliary_power_kw"].reindex(m).values
    seg["zeta"] = ships["depth_of_discharge"].reindex(m).values
    seg["eta"] = ships["utilisation_efficiency"].reindex(m).values

    topo = pd.read_csv(TOPOLOGY)
    g = nx.Graph()
    for _, r in topo.iterrows():
        g.add_edge(r.start, r.end, distance=float(r.distance))
    return seg, ships, Router(g)


def leg_energy(seg: pd.DataFrame, f_current: np.ndarray, f_speed: np.ndarray,
               f_aux: np.ndarray) -> np.ndarray:
    """Re-derive every leg's energy from the observables under one perturbation."""
    sog = seg.sog.values * f_speed
    v = sog - seg.w.values * f_current
    pe = (seg.a.values * v ** 3 + seg.b.values * v ** 2
          + seg.c.values * v + seg.d.values) / seg.scale.values
    hours = seg.distance.values / sog
    return (pe + seg.aux.values * f_aux) / (seg.zeta.values * seg.eta.values) * hours


def coverage(seg: pd.DataFrame, energy: np.ndarray, battery: np.ndarray,
             mask: int, router: Router, mmsis: np.ndarray) -> tuple[int, int]:
    """Ships completing their voyages, non-flexible (plan 1) and flexible (plan 2)."""
    d = seg[["mmsi", "start", "end", "distance"]].copy()
    d["e"] = energy
    ship_legs = [g[["start", "end", "distance", "e"]].to_dict("records")
                 for _, g in d.groupby("mmsi", sort=True)]
    flat = FlatFleet(ship_legs, "e")
    p1 = int(flat.feasible(mask, battery).sum())
    p2 = sum(plan2_feasible(legs, "e", battery[i], mask, router)
             for i, legs in enumerate(ship_legs))
    return p1, p2


def main(run: str = UPSTREAM_DEFAULT, upstream: str = UPSTREAM_DEFAULT) -> None:
    out = write_dir(run)
    ports = params()["case"]["plan2_stations_8"]
    mask = sum(PORT_BIT[p] for p in ports)
    seg, ships, router = load_chain(upstream)
    mmsis = np.sort(seg.mmsi.unique())
    battery = ships["rated_capacity_kwh"].reindex(mmsis).values.astype(float)
    one = np.ones(len(seg))

    e0 = leg_energy(seg, one, one, one)
    err = np.abs(e0 - seg.energy_new.values) / seg.energy_new.values
    assert err.max() < 1e-9, "zero-perturbation energy does not match the leg energy table"
    p1_0, p2_0 = coverage(seg, e0, battery, mask, router, mmsis)
    assert p2_0 == 156, f"control plan2 coverage {p2_0} != 156"
    assert p1_0 == 76, f"control plan1 coverage {p1_0} != 76"

    rng = np.random.default_rng(SEED)
    n = len(seg)
    rows = []
    for i in range(N_DRAWS):
        fc = (1 + rng.normal(0, SIGMA_CURRENT_FLEET)) * \
            (1 + rng.normal(0, SIGMA_CURRENT_LEG, n))
        fs = 1 + rng.normal(0, SIGMA_SPEED_LEG, n)
        fa = 1 + rng.normal(0, SIGMA_AUX_FLEET)
        e = leg_energy(seg, fc, fs, np.full(n, fa))
        p1, p2 = coverage(seg, e, battery, mask, router, mmsis)
        rows.append(dict(draw=i, plan1=p1, plan2=p2, energy=e.sum()))
    mc = pd.DataFrame(rows)
    mc.to_csv(out / "montecarlo.csv", index=False)
    provenance.record(out / "montecarlo.csv", script=__file__,
                      inputs=[RESULTS / upstream / "segment_energy.csv"],
                      params_used=params()["uncertainty"]
                      if "uncertainty" in params() else None)

    for col, base in [("plan2", p2_0), ("plan1", p1_0)]:
        s = mc[col]
    ze = params()["energy"]["depth_of_discharge"] * \
        params()["energy"]["utilisation_efficiency"]
    bias = p2_0 - mc.plan2.median()

    sens = []

    base_p1, base_p2 = p1_0, p2_0
    for soh in (1.0, 0.9, 0.8, 0.7):
        p1, p2 = coverage(seg, e0, battery * soh, mask, router, mmsis)
        sens.append(dict(factor="SOH", value=soh, plan1=p1, plan2=p2))
    s = pd.DataFrame(sens)
    a1, a2 = s[s.value == 1.0].plan1.iloc[0], s[s.value == 1.0].plan2.iloc[0]
    b1, b2 = s[s.value == 0.8].plan1.iloc[0], s[s.value == 0.8].plan2.iloc[0]
    d1, d2 = (a1 - b1) / 172 * 100, (a2 - b2) / 172 * 100
    r1, r2 = (1 - b1 / a1) * 100, (1 - b2 / a2) * 100

    for growth in params()["case"]["growth_rates"]:
        p1, p2 = coverage(seg, e0, battery * (1 + growth / 100), mask, router, mmsis)
        sens.append(dict(factor="growth", value=growth, plan1=p1, plan2=p2))

    now = ships["admiralty_coefficient"] * params()["admiralty"]["code_convention_factor"]
    for c_kw in params()["admiralty"]["sensitivity_c_kw"]:
        sc = (c_kw / now).reindex(seg.mmsi.values).values
        sg = seg.copy(); sg["scale"] = sc
        e = leg_energy(sg, one, one, one)
        p1, p2 = coverage(sg, e, battery, mask, router, mmsis)
        sens.append(dict(factor="C_kW", value=c_kw, plan1=p1, plan2=p2))

    c = pd.DataFrame([r for r in sens if r["factor"] == "C_kW"]).sort_values("value")
    mono1 = c.plan1.is_monotonic_increasing
    mono2 = c.plan2.is_monotonic_increasing
    if not mono2:
        pass

    pd.DataFrame(sens).to_csv(out / "sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
