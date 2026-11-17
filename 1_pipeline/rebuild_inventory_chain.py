
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401

from src.loaders import (  # noqa: E402
    DATA, inventory_summary, output_dir, params, ship_parameters,
    segments_with_timestamps, write_dir,
)

PACK_MASS_T = float(params()["inventory"]["battery_unit_mass_t"])

PORT_EN = {"Nanjing Longtan": "Nanjing", "Changshu Xinghua": "Changshu", "Jiangyin": "Jiangyin",
           "Wuhu": "Wuhu", "Yangzhou": "Yangzhou", "Taicang": "Taicang",
           "Yangshan": "Yangshan", "Zhangjiagang": "Zhangjiagang", "Maanshan": "Maanshan"}

def published() -> dict:
    a = inventory_summary().copy()
    a["port"] = a["Port"].str.replace("_Port", "", regex=False)
    a["mode"] = a["Mode"].map({"Equal": "EAS", "Differential": "DAS"})
    out: dict = {}
    for _, r in a.iterrows():
        out.setdefault(r["port"], {})[r["mode"]] = (
            float(r["Initial Inventory"]), float(r["Average Inventory"]),
            float(r["Max Inventory"]))
    return out


def power_curve(vmin: float, vmax: float, volume: float, c_code: float) -> np.ndarray:
    v = np.linspace(vmin, vmax, 100)
    pe = volume ** (2 / 3) * v ** 3 / c_code
    popt, _ = curve_fit(lambda x, a, b, c, d: a * x ** 3 + b * x ** 2 + c * x + d,
                        v, pe)
    return popt


def leg_energy_terms(legs: pd.DataFrame, popt: np.ndarray, aux: float,
                     zeta: float, eta: float, hp_factor: float = 1.0
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Split each leg's energy into the displacement-dependent part and the rest."""
    a, b, c, d = popt
    v = legs.real_speed.values
    pe = (a * v ** 3 + b * v ** 2 + c * v + d) * hp_factor
    hours = legs.distance.values / (v + legs.avg_wspeed.values)
    return pe * hours / (zeta * eta), aux * hours / (zeta * eta)


def best_adjustment(a_terms: np.ndarray, b_terms: np.ndarray, base: float,
                    volume: float, capacity: float, max_reduce: int,
                    max_increase: int, unit: float) -> tuple:
    """Largest feasible pack shedding, else smallest feasible addition."""
    cum_a, cum_b = np.cumsum(a_terms), np.cumsum(b_terms)

    def check(new_volume: float, new_capacity: float):
        cum = cum_a * (new_volume / base) ** (2 / 3) + cum_b
        return (cum.max() <= new_capacity) if len(cum) else True, (
            cum[-1] if len(cum) else 0.0)

    for k in range(max_reduce, -1, -1):
        nv, nc = volume - k * PACK_MASS_T, capacity - k * unit
        ok, total = check(nv, nc)
        if ok:
            return k, 1, total, nc, nv
    for k in range(1, max_increase + 1):
        nv, nc = volume + k * PACK_MASS_T, capacity + k * unit
        ok, total = check(nv, nc)
        if ok:
            return k, 2, total, nc, nv
    return 0, 0, 0.0, capacity, volume


def ship_plan(legs: pd.DataFrame, info: pd.Series, span: tuple,
              hp_factor: float = 1.0, reference_filters: bool = True
              ) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    legs = legs.reset_index(drop=True).copy()
    unit = float(info["pack_capacity_kwh"])
    capacity = float(info["rated_capacity_kwh"])
    total_packs = int(capacity / unit)
    base = float(info["avg_water"])

    t0, t1 = span
    hours = (t1 - t0).total_seconds() / 3600.0
    share = legs["sailing_h"] / legs["sailing_h"].sum()
    ends = t0 + pd.to_timedelta((share * hours).cumsum(), unit="h")
    legs["end_time"] = ends
    legs["start_time"] = [t0] + list(ends[:-1])

    legs["packs_swapped_eas"] = 0.0
    run = 0.0
    for i, r in legs.iterrows():
        run += r["energy_kwh"]
        if r["optimised swapping"] == 1 and run > 0:
            legs.at[i, "packs_swapped_eas"] = run / unit
            run = 0.0

    legs["packs_removed"] = 0
    legs["sign"] = 0
    legs["packs_swapped_das"] = 0.0

    swaps = legs.index[legs["optimised swapping"] == 1].tolist()
    if not swaps and reference_filters:
        return None, None

    popt = power_curve(float(info["speed_min_kn"]), float(info["speed_max_kn"]),
                       base, float(info["admiralty_coefficient"]))
    a_all, b_all = leg_energy_terms(legs, popt, float(info["auxiliary_power_kw"]),
                                    float(info["depth_of_discharge"]),
                                    float(info["utilisation_efficiency"]), hp_factor)

    volume, cap = base, capacity
    max_reduce, max_increase = total_packs, 0

    if not swaps:
        k, kind, _, cap_new, _ = best_adjustment(
            a_all, b_all, base, volume, cap, max_reduce, max_increase, unit)
        held = int(cap_new / unit) if kind > 0 else total_packs
        return legs, dict(mmsi=int(info.name), origin_port=legs.iloc[0]["start"],
                          peak_packs=held,
                          packs_on_board_at_origin=held,
                          packs_removed_at_origin=k if kind == 1 else 0,
                          feasible=int(legs["feasible"].iloc[0]), start_time=t0)

    lo, hi = 0, swaps[0] + 1
    k, kind, total, cap_new, vol_new = best_adjustment(
        a_all[lo:hi], b_all[lo:hi], base, volume, cap, max_reduce, max_increase, unit)
    if kind > 0:
        legs.at[swaps[0], "packs_removed"] = k
        legs.at[swaps[0], "sign"] = kind
        legs.at[swaps[0], "packs_swapped_das"] = total / unit
        cap, volume = cap_new, vol_new
        held = int(cap / unit)
        max_reduce, max_increase = held, total_packs - held
    else:
        held = total_packs

    peak = held
    outfit = dict(mmsi=int(info.name), origin_port=legs.iloc[0]["start"],
                  packs_on_board_at_origin=held, packs_removed_at_origin=k if kind == 1 else 0,
                  feasible=int(legs["feasible"].iloc[0]), start_time=t0)

    for i in range(len(swaps) - 1):
        lo, hi = swaps[i] + 1, swaps[i + 1] + 1
        k, kind, total, cap_new, vol_new = best_adjustment(
            a_all[lo:hi], b_all[lo:hi], base, volume, cap,
            max_reduce, max_increase, unit)
        if kind > 0:
            legs.at[swaps[i], "packs_removed"] = k
            legs.at[swaps[i], "sign"] = kind
            legs.at[swaps[i + 1], "packs_swapped_das"] = total / unit
            cap, volume = cap_new, vol_new
            held = int(cap / unit)
            peak = max(peak, held)
            max_reduce, max_increase = held, total_packs - held

    outfit["peak_packs"] = peak
    return legs, outfit


def build_plans(ops: pd.DataFrame, ships: pd.DataFrame, spans: pd.DataFrame,
                hp_factor: float = 1.0, reference_filters: bool = True
                ) -> tuple[dict, pd.DataFrame, dict]:
    plans, outfits = {}, []
    skipped_short, skipped_noswap = [], []
    for mmsi, legs in ops.groupby("mmsi", sort=True):
        if mmsi not in spans.index:
            skipped_short.append(mmsi)
            continue
        s = spans.loc[mmsi]
        if s.legs < 3 and reference_filters:
            skipped_short.append(mmsi)
            continue
        plan, outfit = ship_plan(legs, ships.loc[mmsi], (s.t0, s.t1), hp_factor,
                                 reference_filters)
        if plan is None:
            skipped_noswap.append(mmsi)
            continue
        plans[mmsi] = plan
        outfits.append(outfit)
    cohort = dict(fleet=int(ops.mmsi.nunique()), short=skipped_short,
                  noswap=skipped_noswap, plans=sorted(plans))
    return plans, pd.DataFrame(outfits), cohort


def daily_plans(plans: dict, outfits: pd.DataFrame,
                ports: list[str]) -> dict[str, pd.DataFrame]:
    rows = {p: [] for p in ports}

    served = outfits[outfits["feasible"] == 1].copy()
    served["date"] = pd.to_datetime(served["start_time"]).dt.normalize()
    served["packs_landed_eas"] = served["packs_on_board_at_origin"] + served["packs_removed_at_origin"]
    first = served.groupby(["origin_port", "date"]).agg(
        packs_landed_das=("packs_on_board_at_origin", "sum"),
        packs_landed_eas=("packs_landed_eas", "sum")).reset_index()
    for _, r in first.iterrows():
        if r["origin_port"] in rows:
            rows[r["origin_port"]].append(dict(
                date=r["date"], packs_landed_das=r["packs_landed_das"], packs_taken_das=0.0,
                packs_landed_eas=r["packs_landed_eas"], packs_taken_eas=0.0))

    for plan in plans.values():
        p = plan[plan["feasible"] == 1]
        if p.empty:
            continue
        for _, r in p.iterrows():
            if r["end"] not in rows:
                continue
            shed = r["packs_removed"]
            load = (r["packs_swapped_das"] - shed if r["sign"] == 1 else
                    r["packs_swapped_das"] + shed if r["sign"] == 2 else
                    r["packs_swapped_das"])
            rows[r["end"]].append(dict(
                date=pd.Timestamp(r["end_time"]).normalize(),
                packs_landed_das=load, packs_taken_das=r["packs_swapped_das"],
                packs_landed_eas=r["packs_swapped_eas"], packs_taken_eas=r["packs_swapped_eas"]))

    out = {}
    cols = ["packs_landed_das", "packs_taken_das", "packs_landed_eas", "packs_taken_eas"]
    for p in ports:
        if not rows[p]:
            out[p] = pd.DataFrame(columns=["date"] + cols)
            continue
        d = pd.DataFrame(rows[p]).groupby("date", as_index=False).sum()
        span = pd.date_range(d["date"].min(), d["date"].max(), freq="D")
        out[p] = (pd.DataFrame({"date": span}).merge(d, on="date", how="left")
                  .fillna(0.0))
    return out


def recur(load: np.ndarray, unload: np.ndarray,
          ratio: float | None = None, delay: int | None = None) -> dict:
    p = params()["inventory"]
    a_ratio = float(p["charging_ratio"]) if ratio is None else float(ratio)
    tau = int(p["charging_days"]) if delay is None else int(delay)
    immediate = 1.0 - a_ratio

    L = np.asarray(load, float)
    U = np.asarray(unload, float)
    n = len(L)
    if n == 0:
        return dict(initial=0.0, average=0.0, maximum=0.0,
                    series=np.asarray([], float))

    cum_u, cum_l = np.cumsum(U), np.cumsum(L)
    lagged = np.array([cum_u[t - tau] if t - tau >= 0 else 0.0 for t in range(n)])
    a = immediate * cum_u + a_ratio * lagged - cum_l
    i0 = max(0.0, float(-a.min()))

    stock, series = i0, []
    for t in range(n):
        released = U[t - tau] if t - tau >= 0 else 0.0
        stock = stock + immediate * U[t] + a_ratio * released - L[t]
        series.append(stock)
    s = np.asarray(series, float)

    scale = max(1.0, float(np.abs(s).max()))
    assert np.allclose(s, i0 + a, rtol=0, atol=1e-9 * scale), (
        "the forward simulation and the closed form of equation (27) disagree; "
        "one of the two loops has been edited without the other")
    return dict(initial=i0, average=float(s.mean()), maximum=float(s.max()),
                series=s)


def table3(daily: dict[str, pd.DataFrame],
           trajectory: list | None = None) -> pd.DataFrame:
    """The three summary statistics per port and mode."""
    rows = []
    for cn, d in daily.items():
        if d.empty:
            continue
        for mode, lc, uc in [("EAS", "packs_landed_eas", "packs_taken_eas"),
                             ("DAS", "packs_landed_das", "packs_taken_das")]:
            r = recur(d[lc].values.astype(float), d[uc].values.astype(float))
            s = r.pop("series")
            if trajectory is not None:
                trajectory.append(pd.DataFrame(dict(
                    port=PORT_EN.get(cn, cn), port_cn=cn, mode=mode,
                    day=np.arange(1, len(s) + 1), stock=s,
                    load=d[lc].values.astype(float),
                    unload=d[uc].values.astype(float))))
            r.update(port=PORT_EN.get(cn, cn), port_cn=cn, mode=mode, days=len(d))
            rows.append(r)
    return pd.DataFrame(rows)


def load_control() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ops = pd.read_excel(DATA / "04_paper_baseline" / "siting_result" / "n_stations_8"
                        / "growth 0%-Plan 2-8-optimised swapping.xlsx")
    timed = segments_with_timestamps()
    spans = timed.groupby("mmsi").agg(t0=("start_time", "first"),
                                      t1=("end_time", "last"), legs=("mmsi", "size"))
    return ops, ship_parameters().set_index("mmsi"), spans


def main() -> None:
    out = write_dir("coefficient_correction")
    ports = params()["case"]["plan2_stations_8"]
    ops, ships, spans = load_control()

    plans, outfits, cohort = build_plans(ops, ships, spans)
    daily = daily_plans(plans, outfits, ports)
    traj: list = []
    t3 = table3(daily, traj)
    pd.concat(traj, ignore_index=True).to_csv(out / "inventory_daily_baseline.csv",
                                              index=False)

    served = set(outfits[outfits["feasible"] == 1].mmsi)
    feasible = set(ops.groupby("mmsi")["feasible"].max().pipe(lambda s: s[s == 1].index))


    pub = published()
    for key, idx in [("initial", 0), ("average", 1), ("maximum", 2)]:
        t3[f"pub_{key}"] = [pub[p][m][idx] for p, m in zip(t3.port, t3["mode"])]
        t3[f"d_{key}"] = t3[key] - t3[f"pub_{key}"]

    for mode in ("EAS", "DAS"):
        m = t3[t3["mode"] == mode].sort_values("port")
    ok = int((t3[["d_initial", "d_average", "d_maximum"]].abs() < 0.05).sum().sum())
    worst = float(t3[["d_initial", "d_average", "d_maximum"]].abs().max().max())

    t3.to_csv(out / "inventory_chain_control.csv", index=False)
    pd.DataFrame(dict(
        role=(["short"] * len(cohort["short"]) + ["noswap"] * len(cohort["noswap"])
              + ["missing_from_table3"] * len(feasible - served)),
        mmsi=(cohort["short"] + cohort["noswap"] + sorted(feasible - served)),
    )).to_csv(out / "inventory_cohort.csv", index=False)


if __name__ == "__main__":
    main()
