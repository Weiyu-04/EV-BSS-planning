
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401,E402

from src import provenance  # noqa: E402
from src.loaders import DATA, ROOT, RESULTS, output_dir, params  # noqa: E402

RUN = "paper/s7_robustness"
CODE = Path(__file__).resolve().parents[1]

UPSTREAM = "coefficient_correction"
UPSTREAM_ENERGY = "paper/s3_energy"
DELIVERED_INVENTORY = "paper/s5_inventory"


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, CODE / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def quartiles(x, prefix: str = "") -> dict:
    a = np.asarray(x, float)
    return {f"{prefix}p25": float(np.percentile(a, 25)),
            f"{prefix}p50": float(np.percentile(a, 50)),
            f"{prefix}p75": float(np.percentile(a, 75)),
            f"{prefix}p05": float(np.percentile(a, 5)),
            f"{prefix}p95": float(np.percentile(a, 95)),
            f"{prefix}mean": float(a.mean())}


def _delivered_calibre(inv, biv):
    ports = params()["case"]["plan2_stations_8"]
    src = output_dir(UPSTREAM_ENERGY)
    ops = biv.operations_record(ports, source=src)
    _, ships, spans = inv.load_control()
    ships = ships.copy()
    ships["admiralty_coefficient"] = pd.read_csv(
        src / "admiralty_pership.csv").set_index("mmsi")[
            "C_new_code"].reindex(ships.index)
    _, hp = biv.hp_factors()
    return ops, ships, spans, hp


def _inventory_i0(inv, ops, ships, spans, hp_factor: float,
                  ship_ratio: pd.Series | None = None,
                  mode: str = "EAS") -> float:
    """EAS opening stock for the eight stations under one draw."""
    load, unload = (("packs_landed_eas", "packs_taken_eas") if mode == "EAS"
                    else ("packs_landed_das", "packs_taken_das"))
    scaled = ops.copy()
    if ship_ratio is not None:
        scaled["energy_kwh"] = scaled["energy_kwh"].values * ship_ratio.reindex(scaled.mmsi).values
    plans, outfits, _ = inv.build_plans(scaled, ships, spans, hp_factor,
                                        reference_filters=False)
    daily = inv.daily_plans(plans, outfits, params()["case"]["plan2_stations_8"])
    return float(sum(
        inv.recur(d[load].values.astype(float),
                  d[unload].values.astype(float))["initial"]
        for d in daily.values() if not d.empty))


def build_montecarlo(unc, inv, biv, out: Path) -> pd.DataFrame:
    P = params()["robustness"]
    from resolve_siting import PORT_BIT           # noqa: E402
    mask = sum(PORT_BIT[p] for p in params()["case"]["plan2_stations_8"])

    seg, ships, router = unc.load_chain(UPSTREAM)
    mmsis = np.sort(seg.mmsi.unique())
    battery = ships["rated_capacity_kwh"].reindex(mmsis).values.astype(float)
    n = len(seg)
    base_ship = seg.groupby("mmsi").energy_new.sum()

    ops_c, ships_c, spans, hp = _delivered_calibre(inv, biv)
    n_inv, n_draws = int(P["n_draws_inventory"]), int(P["n_draws"])

    for mode in ("EAS", "DAS"):
        delivered = float(pd.read_csv(
            output_dir(DELIVERED_INVENTORY)
            / f"inventory_summary_{mode.lower()}.csv").initial.sum())
        unperturbed = _inventory_i0(inv, ops_c, ships_c, spans, hp, mode=mode)

    rng = np.random.default_rng(int(P["seed"]))
    rows = []
    for i in range(n_draws):
        fc = ((1 + rng.normal(0, P["sigma_current_fleet"]))
              * (1 + rng.normal(0, P["sigma_current_leg"], n)))
        fs = 1 + rng.normal(0, P["sigma_speed_leg"], n)
        fa = 1 + rng.normal(0, P["sigma_aux_fleet"])
        e = unc.leg_energy(seg, fc, fs, np.full(n, fa))
        p1, p2 = unc.coverage(seg, e, battery, mask, router, mmsis)
        row = dict(draw=i, plan1=p1, plan2=p2, energy=float(e.sum()))
        if i < n_inv:
            ratio = (pd.Series(e, index=seg.mmsi.values).groupby(level=0).sum()
                     / base_ship)
            row["inventory_eas"] = _inventory_i0(inv, ops_c, ships_c, spans,
                                                 hp, ratio)
        rows.append(row)

    mc = pd.DataFrame(rows)
    mc.to_csv(out / "montecarlo.csv", index=False)
    return mc


def build_sensitivity_matrix(unc, out: Path) -> pd.DataFrame:
    from resolve_siting import PORT_BIT           # noqa: E402
    mask = sum(PORT_BIT[p] for p in params()["case"]["plan2_stations_8"])
    seg, ships, router = unc.load_chain(UPSTREAM)
    mmsis = np.sort(seg.mmsi.unique())
    battery = ships["rated_capacity_kwh"].reindex(mmsis).values.astype(float)
    one = np.ones(len(seg))
    e0 = unc.leg_energy(seg, one, one, one)

    rows = []

    def add(axis, value, e, batt, note=""):
        p1, p2 = unc.coverage(seg, e, batt, mask, router, mmsis)
        rows.append(dict(axis=axis, value=value, plan1=p1, plan2=p2,
                         plan1_pct=p1 / 172 * 100, plan2_pct=p2 / 172 * 100,
                         note=note))

    for soh in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
        add("SOH", soh, e0, battery * soh)
    for g in params()["case"]["growth_rates"]:
        add("energy_density_growth_pct", float(g), e0, battery * (1 + g / 100))

    band = yaml.safe_load(
        (output_dir("paper/s3_energy") / "energy_error_band.yaml").read_text("utf-8"))
    for tag, err in (("lower", 0.0), ("headline", float(band["error_min"])),
                     ("upper", float(band["error_max"]))):
        add("energy_model_accuracy", err, e0 * (1 + err), battery,
            f"energy_error_band.yaml: {tag} ({band['reported_as']} headline)")

    d = pd.DataFrame(rows)
    d.to_csv(out / "sensitivity_matrix.csv", index=False)
    return d


def build_soh_by_calibre(unc, out: Path) -> pd.DataFrame:
    from resolve_siting import PORT_BIT           # noqa: E402
    mask = sum(PORT_BIT[p] for p in params()["case"]["plan2_stations_8"])
    seg, ships, router = unc.load_chain(UPSTREAM)
    mmsis = np.sort(seg.mmsi.unique())
    battery = ships["rated_capacity_kwh"].reindex(mmsis).values.astype(float)

    rows = []
    for calibre, col in (("published record", "energy_kwh"),
                         ("rebuilt current", "energy_field"),
                         ("corrected coefficient", "energy_new")):
        e = seg[col].values.astype(float)
        for soh in (1.0, 0.9, 0.8, 0.7):
            p1, p2 = unc.coverage(seg, e, battery * soh, mask, router, mmsis)
            rows.append(dict(calibre=calibre, soh=soh, plan1=p1, plan2=p2,
                             plan1_pct=p1 / 172 * 100, plan2_pct=p2 / 172 * 100))
    d = pd.DataFrame(rows)

    out_rows = []
    for calibre, g in d.groupby("calibre", sort=False):
        g = g.set_index("soh")
        for plan in ("plan1", "plan2"):
            base, low = g.loc[1.0, plan], g.loc[0.8, plan]
            out_rows.append(dict(
                calibre=calibre, plan=plan,
                start_pct=base / 172 * 100,
                headroom_pp=(172 - base) / 172 * 100,
                drop_pp_1p0_to_0p8=(base - low) / 172 * 100,
                relative_loss_pct=(1 - low / base) * 100 if base else np.nan))
    summary = pd.DataFrame(out_rows)
    d.to_csv(out / "soh_by_calibre.csv", index=False)
    summary.to_csv(out / "soh_reversal_summary.csv", index=False)
    return summary


def build_charge_window_fit(q_mod, out: Path) -> pd.DataFrame:
    ch = params()["charging"]
    eta_c = float(params()["lcc"]["energy_transfer_efficiency"])
    zeta_eta = float(params()["energy"]["depth_of_discharge"]
                     * params()["energy"]["utilisation_efficiency"])
    ports = params()["case"]["plan2_stations_8"]
    ev = q_mod.arrival_stream(list(ports))
    ev = ev.copy()
    ev["kwh_delivered"] = ev.restored_kwh * zeta_eta

    rows = []
    for kw in ch["power_sweep_kw"]:
        hours = ev.kwh_delivered / (float(kw) * eta_c)
        fits = hours <= ev.window_h
        for port, g in ev.groupby("port"):
            m = g.index
            own = g.cargo_port == 1
            rows.append(dict(
                port=port, power_kw=int(kw),
                events=int(len(g)), events_at_own_call=int(own.sum()),
                events_midway=int((~own).sum()),
                share_fits_of_own=float(fits[m][own.values].mean())
                if own.any() else float("nan"),
                share_fits_of_all=float(fits[m].mean()),
                median_kwh=float(g.kwh_delivered.median()),
                median_window_h=float(g.loc[own[own].index, "window_h"].median())
                if own.any() else float("nan")))
        rows.append(dict(
            port="__network__", power_kw=int(kw), events=int(len(ev)),
            events_at_own_call=int((ev.cargo_port == 1).sum()),
            events_midway=int((ev.cargo_port == 0).sum()),
            share_fits_of_own=float(fits[ev.cargo_port == 1].mean()),
            share_fits_of_all=float(fits.mean()),
            median_kwh=float(ev.kwh_delivered.median()),
            median_window_h=float(ev.loc[ev.cargo_port == 1, "window_h"].median())))
    t = pd.DataFrame(rows)
    t.to_csv(out / "charge_window_fit.csv", index=False)

    q = params()["queueing"]
    sail_h = float(q_mod.segments_with_timestamps()["sailing_h"].sum())
    drawn_kwh = float(ev.kwh_delivered.sum())
    own = ev[ev.cargo_port == 1]
    swap_h = q["fixed_hours"] + q["per_pack_hours"] * own.packs
    swap_kw = float((own.kwh_delivered / swap_h).median())
    kw_ship = float(ch["ship_charge_power_kw"])
    power = pd.DataFrame([
        dict(quantity="drawn while under way", kw=drawn_kwh / sail_h,
             basis="energy delivered over the fleet's sailing hours"),
        dict(quantity="delivered by charging", kw=kw_ship * eta_c,
             basis=f"{kw_ship:.0f} kW at eta_c {eta_c:.2f}"),
        dict(quantity="delivered by swapping", kw=swap_kw,
             basis=(f"median call, {float(own.kwh_delivered.median()):,.0f} kWh "
                    f"over {float(swap_h.median()):.2f} h")),
    ])
    power.to_csv(out / "replenishment_power.csv", index=False)

    mid = ev[ev.cargo_port == 0]
    for port, g in t[t.port != "__network__"].groupby("port"):
        v = g.sort_values("power_kw").share_fits_of_own.values
    return t


def build_voyage_time(q_mod, out: Path) -> pd.DataFrame:
    """Voyage time under all three plans, 4 to 9 stations."""
    q = params()["queueing"]
    ch = params()["charging"]
    eta_c = float(params()["lcc"]["energy_transfer_efficiency"])
    kw = float(ch["ship_charge_power_kw"])
    absorb = bool(ch["absorb_into_berthing"])
    zeta_eta = float(params()["energy"]["depth_of_discharge"]
                     * params()["energy"]["utilisation_efficiency"])

    cov = pd.read_csv(output_dir("paper/s4_siting") / "coverage.csv")
    g0 = cov[cov.growth == 0]
    legs = q_mod.segments_with_timestamps()
    sail = legs.groupby("mmsi")["sailing_h"].sum()
    leg_nm = legs.groupby("mmsi").distance.median()
    lo, hi = params()["case"]["distance_bands_nm"]
    band = pd.Series(np.where(leg_nm < lo, "short",
                              np.where(leg_nm < hi, "medium", "long")),
                     index=leg_nm.index)
    paper8 = params()["case"]["plan2_stations_8"]

    def stream(ports, flexible, growth=0.0):
        ev = q_mod.arrival_stream(list(ports), flexible=flexible,
                                  growth_pct=growth)
        feasible = ev.attrs["feasible"]
        ev = ev.copy()
        ev["swap_h"] = q["fixed_hours"] + q["per_pack_hours"] * ev.packs
        ev["charge_h"] = ev.restored_kwh * zeta_eta / (kw * eta_c)
        return ev, feasible

    def tabulate(ev, feasible, plan, station_set, n, growth=0.0):
        own = ev.cargo_port.values == 1
        if plan == "plan3":
            svc = np.where(own, ev.swap_h.values, ev.charge_h.values)
            flat = np.where(own, q["paper_swap_hours"], ev.charge_h.values)
        else:
            svc = ev.swap_h.values
            flat = np.full(len(ev), float(q["paper_swap_hours"]))
        win = ev.window_h.values if absorb else np.zeros(len(ev))
        d = pd.DataFrame(dict(
            mmsi=ev.mmsi.values, own=own.astype(int),
            restored_kwh=ev.restored_kwh.values,
            add_flat_h=flat, add_bypack_h=svc,
            add_dwell_h=np.maximum(0.0, svc - win)))
        g = d.groupby("mmsi").agg(
            n_events=("mmsi", "size"), n_own_stops=("own", "sum"),
            restored_kwh=("restored_kwh", "sum"),
            add_flat_h=("add_flat_h", "sum"),
            add_bypack_h=("add_bypack_h", "sum"),
            add_dwell_h=("add_dwell_h", "sum"))
        g = g.reindex(sorted(feasible)).fillna(0.0)
        g["n_midway_stops"] = g.n_events - g.n_own_stops
        g["feasible"] = [int(feasible[m] == 1) for m in g.index]
        g["sail_h"] = sail.reindex(g.index).values
        g["leg_nm_median"] = leg_nm.reindex(g.index).values
        g["distance_band"] = band.reindex(g.index).values
        for c in ("flat", "bypack", "dwell"):
            g["pct_" + c] = g["add_" + c + "_h"] / g.sail_h * 100
        g = g.reset_index()
        g.insert(0, "n_stations", n)
        g.insert(0, "growth", growth)
        g.insert(0, "plan", plan)
        g.insert(0, "station_set", station_set)
        return g

    jobs = []
    for n in sorted(g0.n_stations.unique()):
        p1 = g0[(g0.n_stations == n) & (g0.plan == "plan1")].stations.iloc[0]
        p2 = g0[(g0.n_stations == n) & (g0.plan == "plan2")].stations.iloc[0]
        jobs.append(("optimal", int(n), p1.split("|"), p2.split("|")))
    jobs.append(("paper8", 8, paper8, paper8))

    frames = []
    for station_set, n, ports1, ports2 in jobs:
        ev1, f1 = stream(ports1, flexible=False)
        ev2, f2 = stream(ports2, flexible=True)
        frames.append(tabulate(ev1, f1, "plan1", station_set, n))
        frames.append(tabulate(ev2, f2, "plan2", station_set, n))
        frames.append(tabulate(ev2, f2, "plan3", station_set, n))

    for growth in params()["case"]["growth_sweep"]:
        if float(growth) == 0.0:
            continue
        e1, q1 = stream(paper8, flexible=False, growth=float(growth))
        e2, q2 = stream(paper8, flexible=True, growth=float(growth))
        frames.append(tabulate(e1, q1, "plan1", "paper8", 8, float(growth)))
        frames.append(tabulate(e2, q2, "plan2", "paper8", 8, float(growth)))
        frames.append(tabulate(e2, q2, "plan3", "paper8", 8, float(growth)))
    d = pd.concat(frames, ignore_index=True)
    d.to_csv(out / "voyage_time_by_plan.csv", index=False)

    for station_set, n, _p1, _p2 in jobs:
        if station_set != "optimal":
            continue
        for plan in ("plan1", "plan2"):
            got = int(d[(d.growth == 0) & (d.station_set == "optimal")
                        & (d.n_stations == n)
                        & (d.plan == plan)].feasible.sum())
            want = int(g0[(g0.n_stations == n)
                          & (g0.plan == plan)].covered.iloc[0])
    for plan in ("plan1", "plan2"):
        base = d[(d.growth == 0) & (d.station_set == "paper8")
                 & (d.plan == plan)].set_index("mmsi").feasible
        cov, flips = {}, {}
        for growth in sorted(d.growth.unique()):
            now = d[(d.growth == growth) & (d.station_set == "paper8")
                    & (d.plan == plan)].set_index("mmsi").feasible
            cov[growth] = int(now.sum())
            flips[growth] = sum(1 for m in base.index
                                if base[m] == 1 and now.get(m, 0) == 0)
        levels = sorted(cov)
        dips = [(a, b) for a, b in zip(levels, levels[1:]) if cov[b] < cov[a]]
        if dips:
            pass
        worst = max(flips.values())
        if worst:
            pass

    for plan in ("plan1", "plan2", "plan3"):
        r = d[(d.growth == 0) & (d.station_set == "paper8")
              & (d.plan == plan) & (d.feasible == 1)]
    return d


def build_queueing_products(q_mod, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = params()["queueing"]
    ports = params()["case"]["plan2_stations_8"]
    ev = q_mod.arrival_stream(ports)
    ev["service_h"] = q["fixed_hours"] + q["per_pack_hours"] * ev.packs
    t0 = ev.arrive.min()
    ev["t_h"] = (ev.arrive - t0).dt.total_seconds() / 3600.0

    unit_kwh = float(params()["inventory"]["battery_unit_capacity_kwh"])
    span_days = (ev.t_h.max() - ev.t_h.min()) / 24.0

    wait_rows, power_rows, advice = [], [], []
    for port, g in ev.groupby("port"):
        a, s = g.t_h.values, g.service_h.values
        for c in q["berth_sweep"]:
            w = q_mod.simulate(a, s, c)
            wait_rows.append(dict(port=port, berths=c, n_events=len(g),
                                  util=s.sum() / (c * (a.max() - a.min())),
                                  **quartiles(w, "wait_"),
                                  wait_max=float(w.max())))
        day = (g.t_h // 24).astype(int)
        per_day = g.groupby(day).packs.sum() * unit_kwh
        full = per_day.reindex(range(int(day.min()), int(day.max()) + 1),
                               fill_value=0.0)
        power_rows.append(dict(port=port, n_events=len(g),
                               packs_total=float(g.packs.sum()),
                               kwh_total=float(g.packs.sum() * unit_kwh),
                               **quartiles(full.values, "kwh_per_day_"),
                               kw_if_spread_over_24h=float(full.mean() / 24.0)))

    w = pd.DataFrame(wait_rows)
    p = pd.DataFrame(power_rows)

    for port, g in w.groupby("port"):
        g = g.sort_values("berths")
        ok = g[g.wait_p95 <= q["service_level_wait_h"]]
        c_need = int(ok.berths.iloc[0]) if len(ok) else int(g.berths.max())
        row = g[g.berths == c_need].iloc[0]
        pr = p[p.port == port].iloc[0]
        advice.append(dict(
            port=port, berths_required=c_need,
            criterion=f"p95 wait <= {q['service_level_wait_h']:.0f} h",
            wait_p95_at_that_c=float(row.wait_p95),
            utilisation_at_that_c=float(row.util),
            peak_day_kwh=float(pr.kwh_per_day_p95),
            connection_kw_if_peak_day_over_24h=float(pr.kwh_per_day_p95 / 24.0),
            note=("berths are set by clustering, not by average load -- "
                  f"utilisation at the required c is only {row.util:.1%}")))
    a = pd.DataFrame(advice)

    need = dict(zip(a.port, a.berths_required))
    rows = []
    for port, g in ev.groupby("port"):
        wq = q_mod.simulate(g.t_h.values, g.service_h.values, int(need[port]))
        fits = (wq + g.service_h.values) <= g.window_h.values
        rows.append(pd.DataFrame(dict(
            mmsi=g.mmsi.values,
            extra=np.where(fits, 0.0,
                           wq + g.service_h.values - g.window_h.values),
            service=g.service_h.values)))
    ps = (pd.concat(rows, ignore_index=True).groupby("mmsi")
          .agg(with_overlap=("extra", "sum"), by_pack=("service", "sum")))
    sail = q_mod.segments_with_timestamps().groupby("mmsi")["sailing_h"].sum()
    n_ev = ev.groupby("mmsi").size()
    tp = ps.join(sail.rename("sail_h"), how="left")
    tp["flat_2h"] = n_ev.reindex(tp.index).fillna(0) * q["paper_swap_hours"]
    for col in ("flat_2h", "by_pack", "with_overlap"):
        tp["pct_" + col] = tp[col] / tp.sail_h * 100
    tp = tp.reset_index()
    tp.to_csv(out / "time_penalty_per_ship.csv", index=False)

    w.to_csv(out / "queueing_by_station.csv", index=False)
    p.to_csv(out / "power_demand_by_station.csv", index=False)
    a.to_csv(out / "station_sizing_advice.csv", index=False)
    return w, a


def build_working_figure(mc: pd.DataFrame, w: pd.DataFrame, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].boxplot([mc.plan2, mc.plan1], tick_labels=["plan2", "plan1"], whis=(5, 95))
    ax[0].axhline(156, ls="--", lw=1, color="tab:red")
    ax[0].axhline(76, ls="--", lw=1, color="tab:orange")
    ax[0].set_title("coverage (dashed = deterministic)")
    ax[0].set_ylabel("ships / 172")

    ax[1].boxplot([mc.inventory_eas.dropna()], tick_labels=["I0 EAS"], whis=(5, 95))
    ax[1].set_title(f"opening stock, first {int(mc.inventory_eas.notna().sum())} draws")
    ax[1].set_ylabel("packs")

    piv = w.pivot(index="berths", columns="port", values="wait_p95")
    for port in piv.columns:
        ax[2].plot(piv.index, piv[port], marker="o", ms=3, label=port)
    ax[2].axhline(float(params()["queueing"]["service_level_wait_h"]),
                  ls="--", lw=1, color="k")
    ax[2].set_yscale("symlog", linthresh=0.1)
    ax[2].set_title("p95 wait vs berths")
    ax[2].set_xlabel("berths")
    ax[2].legend(fontsize=6, ncol=2)

    fig.tight_layout()
    d = out / "figs" / "internal"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "robustness_distributions_WORKING.png"
    matplotlib.rcParams["svg.hashsalt"] = "etran"
    fig.savefig(path, dpi=110, metadata={"Software": None})
    plt.close(fig)
    return path


def main() -> None:
    out = output_dir(RUN)
    unc = _load("2_analysis/build_uncertainty.py", "s7_unc_run")
    inv = _load("1_pipeline/rebuild_inventory_chain.py", "s7_inv_run")
    biv = _load("1_pipeline/build_inventory.py", "s7_biv_run")
    q_mod = _load("2_analysis/build_queueing.py", "s7_q_run")
    P = params()["robustness"]

    mc = build_montecarlo(unc, inv, biv, out)
    n_inv = int(mc.inventory_eas.notna().sum())
    for col in ("plan2", "plan1"):
        s = quartiles(mc[col])
    inv_q = quartiles(mc.inventory_eas.dropna())

    sm = build_sensitivity_matrix(unc, out)
    for axis, g in sm.groupby("axis", sort=False):
        pass

    soh = build_soh_by_calibre(unc, out)
    for _, r in soh[soh.plan == "plan2"].iterrows():
        pass

    w, advice = build_queueing_products(q_mod, out)
    for _, r in advice.iterrows():
        pass

    vt = build_voyage_time(q_mod, out)
    p8 = vt[(vt.station_set == "paper8") & (vt.feasible == 1)]
    for b in ("short", "medium", "long"):
        cells = []
        for plan in ("plan1", "plan2", "plan3"):
            r = p8[(p8.plan == plan) & (p8.distance_band == b)]
            cells.append(f"{plan} {r.pct_dwell.quantile(.75):5.1f}%")

    cwf = build_charge_window_fit(q_mod, out)
    net = cwf[cwf.port == "__network__"].set_index("power_kw")
    for kw in sorted(net.index):
        pass
    kw0 = int(params()["charging"]["ship_charge_power_kw"])
    per = (cwf[(cwf.power_kw == kw0) & (cwf.port != "__network__")]
           .sort_values("share_fits_of_own"))
    lo, hi = per.iloc[0], per.iloc[-1]
    pw = pd.read_csv(out / "replenishment_power.csv")
    for _, r in pw.iterrows():
        pass

    fp = build_working_figure(mc, w, out)

    inputs = [RESULTS / UPSTREAM / "segment_energy.csv",
              RESULTS / UPSTREAM / "admiralty_pership.csv",
              RESULTS / UPSTREAM / "ais_underway.csv",
              RESULTS / UPSTREAM / "ais_berthing.csv",
              output_dir("paper/s3_energy") / "energy_error_band.yaml",
              DATA / "02_channel_topology" / "port_topology.csv",
              DATA / "03_ship_parameters" / "172_ship_parameters.xlsx"]
    inputs = [q for q in inputs if q.exists()]
    n = 0
    for prod in sorted(out.rglob("*")):
        if prod.is_file() and not prod.name.endswith(provenance.SUFFIX):
            provenance.record(prod, script=__file__, inputs=inputs,
                              params_used={"robustness": params()["robustness"],
                                           "queueing": params()["queueing"]})
            n += 1


if __name__ == "__main__":
    main()
