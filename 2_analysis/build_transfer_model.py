
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401

from build_inventory import hp_factors, operations_record  # noqa: E402
from rebuild_inventory_chain import (  # noqa: E402
    build_plans, daily_plans, load_control, recur,
)
from resolve_siting import TOPOLOGY  # noqa: E402
from src.loaders import RESULTS, output_dir, params, ship_parameters, write_dir  # noqa: E402

TIE_BREAK = 1e-6
NM_TO_KM = 1.852         # port_topology.csv is in nautical miles
LAMBDAS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def port_distances(ports: list[str]) -> np.ndarray:
    topo = pd.read_csv(TOPOLOGY)
    g = nx.Graph()
    for _, r in topo.iterrows():
        g.add_edge(r.start, r.end, distance=float(r.distance))
    d = np.zeros((len(ports), len(ports)))
    for i, p in enumerate(ports):
        for j, q in enumerate(ports):
            if i != j:
                d[i, j] = nx.shortest_path_length(g, p, q, weight="distance") * NM_TO_KM
    return d


def solve(load: np.ndarray, unload: np.ndarray, dist: np.ndarray,
          lam: float, ratio: float, latency: int, transit: np.ndarray,
          terminal: np.ndarray | None = None,
          allow_transfer: bool = True, trucks: int = 0,
          initial_trucks: np.ndarray | None = None) -> dict:
    """Minimum opening stock, optionally allowing packs to be trucked."""
    P, T = load.shape
    n_i0, n_inv = P, P * T
    n_x = P * P * T
    n_y = P * P * T if trucks else 0
    n_tr = P * T if trucks else 0
    n_s = P if trucks else 0
    n = n_i0 + n_inv + n_x + n_y + n_tr + n_s

    def iv(p, t):
        return n_i0 + p * T + t

    def xv(p, q, t):
        return n_i0 + n_inv + (p * P + q) * T + t

    def yv(p, q, t):
        return n_i0 + n_inv + n_x + (p * P + q) * T + t

    def tv(p, t):
        return n_i0 + n_inv + n_x + n_y + p * T + t

    def sv(p):
        return n_i0 + n_inv + n_x + n_y + n_tr + p

    rows, cols, vals, rhs = [], [], [], []
    eq = 0
    for p in range(P):
        for t in range(T):
            rows.append(eq); cols.append(iv(p, t)); vals.append(1.0)
            if t == 0:
                rows.append(eq); cols.append(p); vals.append(-1.0)
            else:
                rows.append(eq); cols.append(iv(p, t - 1)); vals.append(-1.0)
            for q in range(P):
                if q == p:
                    continue
                rows.append(eq); cols.append(xv(p, q, t)); vals.append(1.0)
                lag = int(transit[q, p])
                if t - lag >= 0:
                    rows.append(eq); cols.append(xv(q, p, t - lag))
                    vals.append(-1.0)
            r = (1 - ratio) * unload[p, t] - load[p, t]
            if t >= latency:
                r += ratio * unload[p, t - latency]
            rhs.append(r)
            eq += 1

    if trucks:
        for p in range(P):
            for t_ in range(T):
                rows.append(eq); cols.append(tv(p, t_)); vals.append(1.0)
                if t_ > 0:
                    rows.append(eq); cols.append(tv(p, t_ - 1)); vals.append(-1.0)
                else:
                    rows.append(eq); cols.append(sv(p)); vals.append(-1.0)
                for q_ in range(P):
                    if q_ == p:
                        continue
                    for var in (xv, yv):
                        rows.append(eq); cols.append(var(p, q_, t_))
                        vals.append(1.0)
                        lag = int(transit[q_, p])
                        if t_ - lag >= 0:
                            rows.append(eq); cols.append(var(q_, p, t_ - lag))
                            vals.append(-1.0)
                rhs.append(0.0)
                eq += 1
        for p in range(P):
            rows.append(eq); cols.append(sv(p)); vals.append(1.0)
        rhs.append(float(trucks))
        eq += 1
    a_eq = coo_matrix((vals, (rows, cols)), shape=(eq, n)).tocsr()
    cap_rows, cap_cols, cap_vals, cap_rhs = [], [], [], []

    a_ub, b_ub = None, None
    if terminal is not None:
        rows, cols, vals = [], [], []
        for p in range(P):
            rows.append(p); cols.append(p); vals.append(1.0)
            rows.append(p); cols.append(iv(p, T - 1)); vals.append(-1.0)
        a_ub = coo_matrix((vals, (rows, cols)), shape=(P, n)).tocsr()
        b_ub = np.asarray(terminal, float)
    if cap_rhs:
        cap = coo_matrix((cap_vals, (cap_rows, cap_cols)),
                         shape=(len(cap_rhs), n)).tocsr()
        if a_ub is None:
            a_ub, b_ub = cap, np.asarray(cap_rhs, float)
        else:
            from scipy.sparse import vstack
            a_ub = vstack([a_ub, cap]).tocsr()
            b_ub = np.concatenate([b_ub, np.asarray(cap_rhs, float)])

    c = np.zeros(n)
    c[:n_i0] = 1.0
    for p in range(P):
        for q in range(P):
            if p != q:
                c[xv(p, q, 0):xv(p, q, T - 1) + 1] = lam * dist[p, q] + TIE_BREAK
                if trucks:
                    c[yv(p, q, 0):yv(p, q, T - 1) + 1] = (lam * dist[p, q]
                                                          + TIE_BREAK)

    bounds = [(0, None)] * n
    if initial_trucks is not None:
        fleet = np.asarray(initial_trucks, dtype=float)
        if (not trucks or fleet.shape != (P,) or not np.isfinite(fleet).all()
                or (fleet < 0).any() or not np.isclose(fleet.sum(), trucks)):
            raise ValueError("Fixed initial truck locations must sum to the fleet size.")
        for p, count in enumerate(fleet):
            bounds[sv(p)] = (float(count), float(count))
    if not allow_transfer:
        for p in range(P):
            for q in range(P):
                if p != q:
                    for t in range(T):
                        bounds[xv(p, q, t)] = (0, 0)

    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=np.asarray(rhs),
                  bounds=bounds, method="highs")
    if not res.success:
        return dict(ok=False, message=res.message)

    x = res.x[n_i0 + n_inv:n_i0 + n_inv + n_x].reshape(P, P, T)
    inv = res.x[n_i0:n_i0 + n_inv].reshape(P, T)
    empty_km = 0.0
    base = None
    if trucks:
        y = res.x[n_i0 + n_inv + n_x:
                  n_i0 + n_inv + n_x + n_y].reshape(P, P, T)
        empty_km = float(sum(y[p, q].sum() * dist[p, q]
                             for p in range(P) for q in range(P) if p != q))
        base = res.x[n_i0 + n_inv + n_x + n_y + n_tr:].copy()
    moved = x.sum()
    packkm = float(sum(x[p, q].sum() * dist[p, q]
                       for p in range(P) for q in range(P) if p != q))
    return dict(ok=True, i0=res.x[:n_i0].copy(), total_i0=float(res.x[:n_i0].sum()),
                moved=float(moved), pack_km=packkm, empty_km=empty_km,
                base=base, avg=inv.mean(axis=1), maximum=inv.max(axis=1))


def prepare_case():
    """The paper's common daily inventory and inter-port transport inputs."""
    ports = params()["case"]["plan2_stations_8"]
    _, hp_correct = hp_factors()

    ships = ship_parameters().set_index("mmsi")
    _, _, spans = load_control()
    per_ship = pd.read_csv(RESULTS / "coefficient_correction" / "admiralty_pership.csv"
                           ).set_index("mmsi")
    ships_table = ships.copy()
    ships_table["admiralty_coefficient"] = per_ship["C_new_code"].reindex(ships_table.index)

    plans, outfits, _ = build_plans(operations_record(ports), ships_table, spans,
                                    hp_correct, reference_filters=False)
    daily = daily_plans(plans, outfits, ports)
    dist = port_distances(ports)

    served = outfits[outfits["feasible"] == 1]
    outfit = {"EAS": np.zeros(len(ports)), "DAS": np.zeros(len(ports))}
    for i, p in enumerate(ports):
        s = served[served["origin_port"] == p]
        outfit["DAS"][i] = s["packs_on_board_at_origin"].sum()
        outfit["EAS"][i] = (s["packs_on_board_at_origin"] + s["packs_removed_at_origin"]).sum()

    calendar = pd.date_range(min(d["date"].min() for d in daily.values() if len(d)),
                             max(d["date"].max() for d in daily.values() if len(d)),
                             freq="D")
    return ports, daily, outfit, dist, calendar


def aligned_flows(ports, daily, calendar, load_column, unload_column):
    """Align all sites on dates before assembling their daily stock flows."""
    aligned = [daily[p].set_index("date").reindex(calendar, fill_value=0.0) for p in ports]
    return (np.stack([d[load_column].to_numpy() for d in aligned]),
            np.stack([d[unload_column].to_numpy() for d in aligned]))


def main() -> None:
    out = write_dir("coefficient_correction")
    inv_p = params()["inventory"]
    ratio, latency = inv_p["charging_ratio"], inv_p["charging_days"]
    km_day = float(inv_p["transfer_km_per_day"])
    trucks_sweep = [int(v) for v in inv_p["transfer_trucks_per_day"]]
    ports, daily, outfit, dist, calendar = prepare_case()

    slack = {}
    modes = {"EAS": ("packs_landed_eas", "packs_taken_eas"),
             "DAS": ("packs_landed_das", "packs_taken_das")}
    days = len(calendar)
    rows = []
    for mode, (lc, uc) in modes.items():
        load, unload = aligned_flows(ports, daily, calendar, lc, uc)

        base = sum(recur(load[i], unload[i])["initial"] for i in range(len(ports)))
        transit = np.ceil(dist / km_day).astype(int)
        np.fill_diagonal(transit, 0)
        zero = solve(load, unload, dist, 0.0, ratio, latency, transit,
                     allow_transfer=False)
        assert zero["ok"], zero.get("message")
        ok = abs(zero["total_i0"] - base) < 1.0
        assert ok, (f"[{mode}] transfer LP does not reduce to equation (27) when "
                    f"transfers are forbidden: LP {zero['total_i0']:.2f} vs "
                    f"Eq (27) {base:.2f}")

        allowance = outfit[mode] + ratio * unload[:, -1]
        net = load.sum(axis=1) - unload.sum(axis=1) + ratio * unload[:, -1]
        slack[mode] = float((allowance - net).sum())

        for lam in LAMBDAS:
            for term in (None, allowance):
                for trucks in sorted(set(trucks_sweep)):
                    r = solve(load, unload, dist, lam, ratio, latency,
                              transit, term, trucks=trucks)
                    if not r["ok"]:
                        continue
                    rows.append(dict(
                        mode=mode, lam=lam, terminal=term is not None,
                        trucks_per_day=trucks,
                        total_i0=r["total_i0"], moved=r["moved"],
                        pack_km=r["pack_km"], empty_km=r["empty_km"],
                        base_ports="|".join(
                            f"{ports[i]}:{v:.0f}"
                            for i, v in enumerate(r["base"]) if v > 0.5)
                        if r["base"] is not None else "",
                        i0_by_port="|".join(f"{ports[i]}:{v:.1f}"
                                            for i, v in enumerate(r["i0"])),
                        no_transfer=base))

    res = pd.DataFrame(rows)
    res["saving_pct"] = (1 - res.total_i0 / res.no_transfer) * 100
    res.to_csv(out / "transfer_frontier.csv", index=False)

    for mode in modes:
        m = res[(res["mode"] == mode) & (~res.terminal)
                & (res.trucks_per_day == 0)]
        for _, r in m.iterrows():
            pass
        t = res[(res["mode"] == mode) & (res.terminal)]
        if len(t):
            free = t[t.lam == 0.0]
            sl = slack[mode]


if __name__ == "__main__":
    main()
