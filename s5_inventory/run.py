
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401,E402

from src import provenance  # noqa: E402
from src.loaders import DATA, ROOT, output_dir, params, ship_parameters  # noqa: E402

RUN = "paper/s5_inventory"
UPSTREAM_ENERGY = "paper/s3_energy"
CODE = Path(__file__).resolve().parents[1]

MODES = [("EAS", "packs_landed_eas", "packs_taken_eas"),
         ("DAS", "packs_landed_das", "packs_taken_das")]
FIGURE_PORTS = ["Changshu Xinghua", "Taicang", "Yangshan"]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, CODE / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ADMIRALTY_COLUMN = "C_new_code"
USE_ARCHIVED_FILTERS = False


def build_daily(inv, biv, growth_pct: float = 0.0) -> tuple[dict, pd.DataFrame, dict]:
    ports = params()["case"]["plan2_stations_8"]
    src = output_dir(UPSTREAM_ENERGY)
    ops = biv.operations_record(ports, source=src, growth_pct=growth_pct)

    _, ships, spans = inv.load_control()
    per_ship = pd.read_csv(src / "admiralty_pership.csv").set_index("mmsi")
    ships_table = ships.copy()
    ships_table["admiralty_coefficient"] = per_ship[ADMIRALTY_COLUMN].reindex(ships_table.index)
    if growth_pct:
        f = 1.0 + growth_pct / 100.0
        for col in ("rated_capacity_kwh", "pack_capacity_kwh"):
            ships_table[col] = ships_table[col] * f
    hp_published, hp_used = biv.hp_factors()

    plans, outfits, cohort = inv.build_plans(ops, ships_table, spans,
                                             hp_used,
                                             reference_filters=USE_ARCHIVED_FILTERS)
    replayed = sorted(set(ops.loc[ops["optimised swapping"] == 1, "end"].dropna().unique()))
    caliber = {
        "ops_stations": list(ports),
        "ops_stations_replayed": replayed,
        "ops_source": src,
        "admiralty_file": src / "admiralty_pership.csv",
        "admiralty_column": ADMIRALTY_COLUMN,
        "admiralty_coefficients": ships_table["admiralty_coefficient"],
        "hp_factor": hp_used,
        "hp_factor_published": hp_published,
        "reference_filters": USE_ARCHIVED_FILTERS,
        "cohort": cohort,
    }
    return inv.daily_plans(plans, outfits, ports), outfits, caliber


def build_ship_side(inv, outfits: pd.DataFrame, ports: list) -> pd.DataFrame:
    sv = outfits[outfits["feasible"] == 1].copy()
    das = float(sv["packs_on_board_at_origin"].sum())
    eas = float((sv["packs_on_board_at_origin"] + sv["packs_removed_at_origin"]).sum())
    das_peak = float(sv["peak_packs"].sum())
    rated = float(ship_parameters().set_index("mmsi")["packs"]
                  .reindex(sv["mmsi"]).sum())
    n = len(sv)
    reloading = int((sv["peak_packs"] > sv["packs_on_board_at_origin"]).sum())
    pack_t = float(params()["inventory"]["battery_unit_mass_t"])

    def row(q, packs):
        return dict(quantity=q, packs=packs, share_of_rated=packs / rated,
                    per_ship=packs / n)

    rows = [row("outfitting packs, EASM", eas),
            row("outfitting packs, DASM", das),
            row("packs not carried at outfitting under DASM", eas - das),
            row("peak packs carried, DASM", das_peak),
            row("packs not carried at peak under DASM", eas - das_peak)]
    t = pd.DataFrame(rows)
    t["tonnes"] = t.packs * pack_t
    t["ships"] = n
    t["ships_reloading_later"] = reloading
    t["pack_weight_t"] = pack_t

    by_ship = sv.set_index("mmsi")
    rated_each = ship_parameters().set_index("mmsi")["packs"].reindex(by_ship.index)
    for col, what in (("packs_on_board_at_origin", "DASM outfitting"), ("peak_packs", "peak held")):
        over = by_ship.index[by_ship[col] > rated_each + 1e-9]
    off = sv[~sv["origin_port"].isin(ports)]
    t["ships_starting_off_station"] = len(off)
    t["eas_packs_outfitted_off_station"] = float(
        (off["packs_on_board_at_origin"] + off["packs_removed_at_origin"]).sum())
    t["das_packs_outfitted_off_station"] = float(off["packs_on_board_at_origin"].sum())
    return t


def reconcile_ship_and_station(ship: pd.DataFrame, outfits: pd.DataFrame,
                               summary: pd.DataFrame, ports: list) -> None:
    """The ship-side table and the station-side summary must agree on outfitting."""
    sv = outfits[outfits["feasible"] == 1]
    on = sv[sv["origin_port"].isin(ports)]
    ship_side = {"EAS": float((on["packs_on_board_at_origin"] + on["packs_removed_at_origin"]).sum()),
                 "DAS": float(on["packs_on_board_at_origin"].sum())}
    for mode, station_side in summary.groupby("mode")["outfitting_packs"]:
        got, want = ship_side[mode], float(station_side.sum())
    off_eas = float(ship.iloc[0].eas_packs_outfitted_off_station)
    total_eas = float(ship.iloc[0].packs)


def build_switch_deltas(inv, biv, summary: pd.DataFrame) -> pd.DataFrame:
    ports = params()["case"]["plan2_stations_8"]
    src = output_dir(UPSTREAM_ENERGY)
    ops_replayed = biv.operations_record(ports, source=src)
    ops_published, ships, spans = inv.load_control()
    per_ship = pd.read_csv(src / "admiralty_pership.csv").set_index("mmsi")
    hp_published, hp_correct = biv.hp_factors()

    def measure(replay: bool, s3_column: bool, hp_ok: bool, filters_off: bool) -> dict:
        sh = ships.copy()
        if s3_column:
            sh["admiralty_coefficient"] = per_ship[ADMIRALTY_COLUMN].reindex(sh.index)
        plans, outfits, cohort = inv.build_plans(
            ops_replayed if replay else ops_published, sh, spans,
            hp_correct if hp_ok else hp_published,
            reference_filters=not filters_off)
        d = inv.daily_plans(plans, outfits, ports)
        row = {"ships_planned": len(cohort["plans"])}
        for mode, load, unload in MODES:
            for win in (30, 90):
                row[f"{mode}_d{win}"] = sum(
                    inv.recur(d[p][load].to_numpy()[:win],
                              d[p][unload].to_numpy()[:win])["initial"]
                    for p in ports)
        return row

    spec = [
        ("published_baseline", (False, False, False, False), None,
         "baseline settings"),
        ("ops_source", (True, False, False, False), "published_baseline",
         "operating record from the published xlsx changed to the leg energy replay"),
        ("admiralty_column", (False, True, False, False), "published_baseline",
         "per ship C taken from the reference column instead of the calibrated coefficient; touches only DAS"),
        ("hp_factor", (False, False, True, False), "published_baseline",
         "DAS inside the unloading search hp_factor from 1.0 changed to 0.735; touches only DAS"),
        ("reference_filters_off", (False, False, False, True), "published_baseline",
         "reference early exits disabled; 151->172  ships"),
        ("all_four", (True, True, True, True), "published_baseline",
         "final_convention; should equal the formal output of this segment"),
        ("ops_source_given_filters_off", (True, False, False, True),
         "reference_filters_off",
         "same as ops_source, but in (3) measured with the restriction lifted--DeltaEAS changes sign here"),
    ]
    got = {name: measure(*flags) for name, flags, _, _ in spec}
    metrics = [f"{m}_d{w}" for m, _, _ in MODES for w in (30, 90)]

    rows = []
    for name, _, ref, note in spec:
        r = dict(switch=name, delta_vs=ref or "", note=note, **got[name])
        for m in metrics:
            r[f"d_{m}"] = (got[name][m] - got[ref][m]) if ref else 0.0
        rows.append(r)
    table = pd.DataFrame(rows)

    singles = table[table.switch.isin(
        ["ops_source", "admiralty_column", "hp_factor", "reference_filters_off"])]
    dead = singles[singles[[f"d_{m}" for m in metrics]].abs().max(axis=1) == 0.0]

    all_four = table[table.switch == "all_four"].iloc[0]
    for mode, _, _ in MODES:
        produced = float(summary[summary["mode"] == mode].initial.sum())
    return table


def build_trajectories(inv, daily) -> pd.DataFrame:
    """The daily stock series, with the opening balance as day 0."""
    rows = []
    for port, d in daily.items():
        if d.empty:
            continue
        for mode, lc, uc in MODES:
            L = d[lc].values.astype(float)
            U = d[uc].values.astype(float)
            r = inv.recur(L, U)
            rows.append(pd.DataFrame(dict(
                port=port, mode=mode, day=np.arange(0, len(L) + 1),
                stock=np.r_[r["initial"], r["series"]],
                load=np.r_[0.0, L], unload=np.r_[0.0, U])))
    return pd.concat(rows, ignore_index=True)


def build_summary(inv, daily, outfits) -> pd.DataFrame:
    served = outfits[outfits["feasible"] == 1].copy()
    served["packs_landed_eas"] = (served["packs_on_board_at_origin"]
                              + served["packs_removed_at_origin"])
    outfit = {
        "EAS": served.groupby("origin_port")["packs_landed_eas"].sum(),
        "DAS": served.groupby("origin_port")["packs_on_board_at_origin"].sum(),
    }
    rows = []
    for port, d in daily.items():
        if d.empty:
            continue
        for mode, lc, uc in MODES:
            L = d[lc].values.astype(float)
            U = d[uc].values.astype(float)
            r = inv.recur(L, U)
            s = r["series"]
            of = float(outfit[mode].get(port, 0.0))
            rows.append(dict(
                port=port, mode=mode, days=len(L),
                initial=r["initial"], average=r["average"], maximum=r["maximum"],
                minimum=float(s.min()), terminal=float(s[-1]),
                band=float(s.max() - s.min()),
                drift=float(s[-1] - r["initial"]),
                outfitting_packs=of,
                net_drain=float(L.sum() - U.sum()),
                drift_excl_outfitting=float(s[-1] - r["initial"] + of)))
    return pd.DataFrame(rows)


def build_growth_inventory(inv, biv, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inventory across the energy-density scenarios."""
    daily_rows, summary_rows = [], []
    for growth in params()["case"]["growth_sweep"]:
        daily, outfits, _cal = build_daily(inv, biv, growth_pct=growth)
        traj = build_trajectories(inv, daily)
        traj = traj.assign(growth=growth)
        daily_rows.append(traj)
        summ = build_summary(inv, daily, outfits).assign(growth=growth)
        summary_rows.append(summ)
        for (port, mode), g in traj.groupby(["port", "mode"]):
            g = g.sort_values("day")
            tail = g[g.day > g.day.max() - 45]
            summary_rows[-1].loc[
                (summary_rows[-1].port == port)
                & (summary_rows[-1]["mode"] == mode), "net_last45"] = (
                    float(tail.stock.iloc[-1] - tail.stock.iloc[0]))
    d = pd.concat(daily_rows, ignore_index=True)
    t = pd.concat(summary_rows, ignore_index=True)
    d.to_csv(out / "inventory_daily_by_growth.csv", index=False)
    t.to_csv(out / "inventory_summary_by_growth.csv", index=False)

    base = pd.read_csv(out / "inventory_daily_paper8.csv")
    got = (d[d.growth == 0].sort_values(["port", "mode", "day"])
           .stock.values)
    want = base.sort_values(["port", "mode", "day"]).stock.values
    return d, t


def build_window_sensitivity(inv, daily) -> pd.DataFrame:
    rows = []
    for w in (30, 45, 60, 90):
        for mode, lc, uc in MODES:
            tot = 0.0
            for port, d in daily.items():
                if d.empty:
                    continue
                L = d[lc].values.astype(float)[:w]
                U = d[uc].values.astype(float)[:w]
                if len(L) == 0:
                    continue
                i0 = inv.recur(L, U)["initial"]
                tot += i0
                rows.append(dict(window_days=w, mode=mode, port=port,
                                 initial=i0, days_available=min(w, len(d))))
            rows.append(dict(window_days=w, mode=mode, port="__fleet__",
                             initial=tot, days_available=w))
    return pd.DataFrame(rows)


def build_self_balance(summary: pd.DataFrame) -> pd.DataFrame:
    p = params()["inventory"]
    beta = float(p["self_balance_band_over_i0_max"])
    gamma = float(p["self_balance_drift_over_i0_max"])
    d = summary.copy()
    d["band_over_i0"] = d.band / d.initial
    d["drift_excl_over_i0"] = d.drift_excl_outfitting.abs() / d.initial
    d["band_ok"] = d.band_over_i0 <= beta
    d["drift_ok"] = d.drift_excl_over_i0 <= gamma
    d["self_balancing"] = d.band_ok & d.drift_ok
    d["band_threshold"] = beta
    d["drift_threshold"] = gamma
    for b in p["self_balance_sweep"]:
        d[f"self_balancing_at_band_{b}"] = (d.band_over_i0 <= b) & d.drift_ok
    return d


def build_balance_scales(traj: pd.DataFrame) -> pd.DataFrame:
    p = params()["inventory"]
    beta = float(p["self_balance_band_over_i0_max"])
    scales = [int(k) for k in p["balance_judging_scales_days"]]
    rows = []
    for (port, mode), g in traj.groupby(["port", "mode"], sort=False):
        s = g.sort_values("day").stock.values
        i0 = s[0]
        d1 = np.diff(s)
        var1 = float(np.var(d1, ddof=1))
        for k in scales:
            if len(s) <= k:
                continue
            dk = s[k:] - s[:-k]
            vr = float(np.var(dk, ddof=1) / (k * var1)) if var1 > 0 else np.nan
            rows.append(dict(
                port=port, mode=mode, scale_days=k,
                abs_change_median=float(np.median(np.abs(dk))),
                abs_change_p95=float(np.percentile(np.abs(dk), 95)),
                abs_change_max=float(np.abs(dk).max()),
                p95_over_i0=float(np.percentile(np.abs(dk), 95) / i0) if i0 else np.nan,
                variance_ratio=vr,
                mean_reverting=bool(vr < 1.0) if var1 > 0 else False,
                would_signal_transfer=float(np.mean(np.abs(dk) > beta * i0))
                if i0 else np.nan))
    return pd.DataFrame(rows)


def build_terminal_balance(inv, daily, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    a_ratio = float(params()["inventory"]["charging_ratio"])
    for _, r in summary.iterrows():
        d = daily[r.port]
        lc, uc = next((lc, uc) for m, lc, uc in MODES if m == r["mode"])
        U = d[uc].values.astype(float)
        gap = r.terminal - r.initial
        rows.append(dict(
            port=r.port, mode=r["mode"], initial=r.initial, terminal=r.terminal,
            terminal_minus_initial=gap,
            predicted_gap=-(r.net_drain) - a_ratio * float(U[-1]),
            net_drain=r.net_drain, outfitting_packs=r.outfitting_packs,
            still_charging_last_day=a_ratio * float(U[-1]),
            gap_excluding_outfitting=r.drift_excl_outfitting,
            satisfiable_by_holding_more_stock=False,
            terminal_balance_holds=bool(gap >= -1e-9)))
    t = pd.DataFrame(rows)
    worst = float((t.terminal_minus_initial - t.predicted_gap).abs().max())
    assert worst < 1e-6, (
        f"the closing balance does not match its closed form (worst {worst:.2e}); "
        f"the trajectory and equation (27) have diverged")
    return t


def draw_figure(traj: pd.DataFrame, scales: pd.DataFrame, out: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return None

    figs = out / "figs" / "internal"
    figs.mkdir(parents=True, exist_ok=True)
    en = _load("1_pipeline/rebuild_inventory_chain.py",
               "rebuild_inventory_chain").PORT_EN
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.4), sharex="row")
    for j, port in enumerate(FIGURE_PORTS):
        g = traj[(traj.port == port) & (traj["mode"] == "EAS")].sort_values("day")
        ax = axes[0][j]
        ax.plot(g.day, g.stock, lw=1.2, color="#1f4e79")
        ax.axhline(g.stock.iloc[0], ls="--", lw=0.9, color="#c00000")
        ax.set_title(f"{en.get(port, port)}  (EAS)", fontsize=10)
        ax.set_xlabel("day")
        if j == 0:
            ax.set_ylabel("packs in stock")

        s = scales[(scales.port == port) & (scales["mode"] == "EAS")]
        ax2 = axes[1][j]
        colours = ["#bfbfbf" if int(k) == 1 else "#4472c4" for k in s.scale_days]
        ax2.bar([str(int(k)) for k in s.scale_days], s.variance_ratio,
                color=colours, width=0.55)
        ax2.axhline(1.0, ls="--", lw=0.9, color="#808080")
        ax2.set_ylim(0, 1.2)
        ax2.set_xlabel("judging scale (days)")
        if j == 0:
            ax2.set_ylabel("variance ratio")
            ax2.annotate("k = 1 is 1.0 by definition", xy=(0, 1.0),
                         xytext=(0.05, 1.1), fontsize=7.5, color="#595959")
    fig.suptitle("Daily stock and how far a swing reverses itself "
                 "(variance ratio: 1 = random walk, below 1 = self-correcting)",
                 fontsize=11)
    fig.tight_layout()
    path = figs / "inventory_balance_scales_WORKING.png"
    fig.savefig(path, dpi=200, metadata={"Software": None})
    plt.close(fig)
    return path


def main() -> None:
    out = output_dir(RUN)
    inv = _load("1_pipeline/rebuild_inventory_chain.py", "rebuild_inventory_chain")
    biv = _load("1_pipeline/build_inventory.py", "build_inventory")

    daily, outfits, _caliber = build_daily(inv, biv)

    traj = build_trajectories(inv, daily)
    traj.to_csv(out / "inventory_daily_paper8.csv", index=False)

    summary = build_summary(inv, daily, outfits)
    for mode in ("EAS", "DAS"):
        m = summary[summary["mode"] == mode]
        m.to_csv(out / f"inventory_summary_{mode.lower()}.csv", index=False)

    win = build_window_sensitivity(inv, daily)
    win.to_csv(out / "window_sensitivity.csv", index=False)
    fleet = win[win.port == "__fleet__"].pivot(index="mode", columns="window_days",
                                               values="initial")
    for mode in ("EAS", "DAS"):
        r = fleet.loc[mode]

    bal = build_self_balance(summary)
    bal.to_csv(out / "port_self_balance.csv", index=False)
    p = params()["inventory"]
    for mode in ("EAS", "DAS"):
        m = bal[bal["mode"] == mode]
        ok = m[m.self_balancing]

    tb = build_terminal_balance(inv, daily, summary)
    tb.to_csv(out / "terminal_balance.csv", index=False)
    for mode in ("EAS", "DAS"):
        m = tb[tb["mode"] == mode]

    scales = build_balance_scales(traj)
    scales.to_csv(out / "balance_scales_1_3_5d.csv", index=False)
    judged = scales[scales.scale_days > 1]
    mr = int(judged.mean_reverting.sum())
    vr5 = scales[scales.scale_days == 5].variance_ratio
    fig = draw_figure(traj, scales, out)
    if fig:
        pass

    ports = params()["case"]["plan2_stations_8"]
    ship = build_ship_side(inv, outfits, ports)
    reconcile_ship_and_station(ship, outfits, summary, ports)
    ship.to_csv(out / "ship_side_packs.csv", index=False)
    e, d, sv, pk, pks = (ship.iloc[i] for i in range(5))

    gd, gt = build_growth_inventory(inv, biv, out)
    gs = sorted(gt.growth.unique())
    for mode in ("EAS", "DAS"):
        cells = "  ".join(
            f"+{int(g)}%={gt[(gt.growth == g) & (gt['mode'] == mode)].initial.sum():.0f}"
            for g in gs)
    flips = []
    for (port, mode), g in gt[gt["mode"] == "DAS"].groupby(["port", "mode"]):
        g = g.sort_values("growth")
        v = g.net_last45.values
        for k in range(1, len(v)):
            if v[k - 1] * v[k] < 0:
                flips.append(f"{port} in +{int(g.growth.values[k-1])}%~"
                             f"+{int(g.growth.values[k])}% between")
                break
    for f in flips:
        pass

    deltas = build_switch_deltas(inv, biv, summary)
    deltas.to_csv(out / "inventory_delta_by_switch.csv", index=False)
    base = deltas[deltas.switch == "published_baseline"].iloc[0]
    for _, r in deltas[deltas.delta_vs != ""].iterrows():
        pass
    flip = deltas[deltas.switch == "ops_source_given_filters_off"].iloc[0]
    single = deltas[deltas.switch == "ops_source"].iloc[0]

    inputs = [output_dir(UPSTREAM_ENERGY) / "segment_energy.csv",
              DATA / "02_channel_topology" / "port_topology.csv",
              DATA / "03_ship_parameters" / "172_ship_parameters.xlsx"]
    inputs = [q for q in inputs if q.exists()]
    n = 0
    for prod in sorted(out.rglob("*")):
        if prod.is_file() and not prod.name.endswith(provenance.SUFFIX):
            provenance.record(prod, script=__file__, inputs=inputs,
                              params_used={"inventory": params()["inventory"]})
            n += 1


if __name__ == "__main__":
    main()
