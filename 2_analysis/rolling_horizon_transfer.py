
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401,E402

import build_rule_based_transfer as rules  # noqa: E402
import build_transfer_model as transfer_lp  # noqa: E402
import rebuild_inventory_chain  # noqa: E402
from src.loaders import params, write_dir  # noqa: E402

RUN = "rolling_horizon_transfer"
KM_WEIGHT = 1e-4
TRUCKS = 120
LOOKBACK = 7


def shared_case():
    ports, daily, _, dist, calendar = transfer_lp.prepare_case()
    load, unload = transfer_lp.aligned_flows(ports, daily, calendar, "packs_landed_das", "packs_taken_das")
    inv = params()["inventory"]
    transit = np.ceil(dist / float(inv["transfer_km_per_day"])).astype(int)
    ratio, delay = float(inv["charging_ratio"]), int(inv["charging_days"])
    ready = rules.ready_returns(unload, ratio, delay)
    lp = transfer_lp.solve(load, unload, dist, 0.0, ratio, delay, transit, trucks=TRUCKS)
    fleet = np.floor(lp["base"]).astype(int)
    frac = lp["base"] - fleet
    fleet[np.argsort(-frac, kind="stable")[:TRUCKS - fleet.sum()]] += 1
    shared = transfer_lp.solve(load, unload, dist, 0.0, ratio, delay, transit,
                            trucks=TRUCKS, initial_trucks=fleet)
    no_transfer = np.array([rebuild_inventory_chain.recur(d, r, ratio, delay)["initial"]
                            for d, r in zip(load, unload, strict=True)])
    return ports, load, ready, dist, transit, shared["i0"], fleet, no_transfer


def plan(stock, trucks, pack_in, truck_in, f_load, f_ready, dist, transit):
    """LP over H future days."""
    P, H = f_load.shape
    pairs = [(p, q) for p in range(P) for q in range(P) if p != q]
    D = H
    nx_ = len(pairs) * D
    ix = lambda k, i: k * len(pairs) + i  # noqa: E731
    iy = lambda k, i: nx_ + k * len(pairs) + i  # noqa: E731
    iI = lambda k, p: 2 * nx_ + k * P + p  # noqa: E731  (k = 0..H, 0 = after today's dispatch)
    iu = lambda k, p: 2 * nx_ + (H + 1) * P + (k - 1) * P + p  # noqa: E731 (k=1..H)
    iT = lambda k, p: 2 * nx_ + (H + 1) * P + H * P + k * P + p  # noqa: E731 trucks idle after dispatch day k
    n = 2 * nx_ + (H + 1) * P + H * P + D * P
    rows, cols, vals, rhs = [], [], [], []
    eq = 0

    def add(entries, b):
        nonlocal eq
        for c, v in entries:
            rows.append(eq); cols.append(c); vals.append(v)
        rhs.append(b); eq += 1

    for k in range(H + 1):
        for p in range(P):
            e = [(iI(k, p), 1.0)]
            b = 0.0
            if k == 0:
                b = stock[p]
            else:
                e.append((iI(k - 1, p), -1.0))
                e.append((iu(k, p), -1.0))
                b = f_ready[p, k - 1] - f_load[p, k - 1] + pack_in[k, p]
            if k < D:
                for i, (a, q) in enumerate(pairs):
                    if a == p:
                        e.append((ix(k, i), 1.0))
            for i, (a, q) in enumerate(pairs):
                lag = transit[a, q]
                if q == p and 1 <= k and 0 <= k - lag < D and k - lag < k:
                    e.append((ix(k - lag, i), -1.0))
            add(e, b)
    for k in range(D):
        for p in range(P):
            e = [(iT(k, p), 1.0)]
            b = trucks[p] if k == 0 else truck_in[k, p]
            if k > 0:
                e.append((iT(k - 1, p), -1.0))
            for i, (a, q) in enumerate(pairs):
                if a == p:
                    e += [(ix(k, i), 1.0), (iy(k, i), 1.0)]
                lag = transit[a, q]
                if q == p and k > 0 and 0 <= k - lag < k:
                    e += [(ix(k - lag, i), -1.0), (iy(k - lag, i), -1.0)]
            add(e, b)
    A = coo_matrix((vals, (rows, cols)), shape=(eq, n)).tocsr()
    c = np.zeros(n)
    for k in range(D):
        for i, (a, q) in enumerate(pairs):
            c[ix(k, i)] = c[iy(k, i)] = KM_WEIGHT * dist[a, q]
    for k in range(1, H + 1):
        for p in range(P):
            c[iu(k, p)] = 1.0
    res = linprog(c, A_eq=A, b_eq=np.asarray(rhs), bounds=[(0, None)] * n, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    x0 = np.zeros((P, P)); y0 = np.zeros((P, P))
    for i, (a, q) in enumerate(pairs):
        x0[a, q] = res.x[ix(0, i)]; y0[a, q] = res.x[iy(0, i)]
    return x0, y0


def simulate(load, ready, dist, transit, opening, fleet, horizon, mode, floor=None):
    P, T = load.shape
    lag_max = int(transit.max())
    pack_arr = np.zeros((T + lag_max + 2, P)); truck_arr = np.zeros_like(pack_arr)
    stock = opening.astype(float).copy(); trucks = fleet.astype(float).copy()
    topup = np.zeros((P, T)); loaded_km = empty_km = moved = 0.0
    for t in range(T):
        trucks += truck_arr[t]
        if floor is None:
            stock += ready[:, t] + pack_arr[t]
            topup[:, t] = np.maximum(0.0, load[:, t] - stock)
            stock += topup[:, t] - load[:, t]
        else:
            stock += pack_arr[t]
            topup[:, t] = np.maximum(0.0, floor[:, t] - stock)
            stock += ready[:, t] + topup[:, t] - load[:, t]
        H = min(horizon, T - 1 - t)
        if H <= 0:
            continue
        if mode == "lookahead":
            f_load, f_ready = load[:, t + 1:t + 1 + H], ready[:, t + 1:t + 1 + H]
        else:
            lo = max(0, t + 1 - LOOKBACK)
            f_load = np.repeat(load[:, lo:t + 1].mean(axis=1, keepdims=True), H, axis=1)
            f_ready = np.repeat(ready[:, lo:t + 1].mean(axis=1, keepdims=True), H, axis=1)
        pin = np.zeros((H + 1, P)); tin = np.zeros((H + 1, P))
        for k in range(1, H + 1):
            pin[k] = pack_arr[t + k]; tin[k] = truck_arr[t + k]
        x, y = plan(stock, trucks, pin, tin, f_load, f_ready, dist, transit)
        x = np.floor(x + 1e-6); y = np.floor(y + 1e-6)
        for p in range(P):
            out = x[p].sum()
            if out > min(stock[p], trucks[p]):
                x[p] *= np.floor(min(stock[p], trucks[p])) / out if out else 0
                x[p] = np.floor(x[p])
            room = trucks[p] - x[p].sum()
            if y[p].sum() > room:
                y[p] = np.floor(y[p] * max(room, 0) / y[p].sum())
        for p in range(P):
            for q in range(P):
                if p == q:
                    continue
                lag = transit[p, q]
                if x[p, q]:
                    stock[p] -= x[p, q]; trucks[p] -= x[p, q]
                    pack_arr[t + lag, q] += x[p, q]; truck_arr[t + lag, q] += x[p, q]
                    loaded_km += x[p, q] * dist[p, q]; moved += x[p, q]
                if y[p, q]:
                    trucks[p] -= y[p, q]; truck_arr[t + lag, q] += y[p, q]
                    empty_km += y[p, q] * dist[p, q]
        assert (stock >= -1e-6).all() and (trucks >= -1e-6).all()
    top = topup[topup > 1e-7]
    return dict(topup_pack_equiv=float(top.sum()), shortage_port_days=len(top),
                moved_packs=moved, loaded_pack_km=loaded_km, empty_truck_km=empty_km)


def main():
    ports, load, ready, dist, transit, opening, fleet, no_transfer = shared_case()
    rows = []
    zero = np.zeros_like(fleet)
    base = rules.replay(load, ready, dist, transit, opening, zero)
    rule = rules.replay(load, ready, dist, transit, opening, fleet)
    for name, r in (("No transfer", base), ("Rule (3-day buffer, 7-day look-back)", rule)):
        e = r["emergency"]
        rows.append(dict(policy=name, information="observed only",
                         topup_pack_equiv=float(e.sum()), shortage_port_days=int((e > 1e-7).sum()),
                         moved_packs=r["moved"], loaded_pack_km=r["pack_km"], empty_truck_km=r["empty_km"]))
    for mode, label in (("forecast", "7-day mean forecast"), ("lookahead", "exact lookahead")):
        for H in (1, 3, 7, 14):
            r = simulate(load, ready, dist, transit, opening, fleet, H, mode)
            rows.append(dict(policy=f"Rolling horizon H={H}", information=label, **r))
    rows.append(dict(policy="Full-foresight LP (90 days)", information="all 90 days",
                     topup_pack_equiv=0.0, shortage_port_days=0))
    table = pd.DataFrame(rows)
    table.insert(2, "opening_pack_equiv", float(opening.sum()))
    out = write_dir(RUN) / "rolling_horizon_comparison.csv"
    table.to_csv(out, index=False)

    issues = float(load.sum())
    curve = []
    zero = np.zeros_like(fleet)
    for k in (1.0, 1.5, 2.0, 3.0):
        start = opening * k
        cases = {"No transfer": rules.replay(load, ready, dist, transit, start, zero)["emergency"].sum(),
                 "Rule (3-day buffer, 7-day look-back)": rules.replay(load, ready, dist, transit, start, fleet)["emergency"].sum()}
        for h, mode, lab in ((7, "forecast", "RH H=7, 7-day mean forecast"),
                             (14, "forecast", "RH H=14, 7-day mean forecast"),
                             (14, "lookahead", "RH H=14, exact lookahead")):
            cases[lab] = simulate(load, ready, dist, transit, start, fleet, h, mode)["topup_pack_equiv"]
        for name, top in cases.items():
            curve.append(dict(scale=k, opening_pack_equiv=float(start.sum()), policy=name,
                              topup_pack_equiv=float(top), topup_share_of_issues_pct=100 * float(top) / issues))
    pd.DataFrame(curve).to_csv(write_dir(RUN) / "shortage_vs_opening_stock.csv", index=False)


if __name__ == "__main__":
    main()
