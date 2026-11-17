
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401,E402

from src import provenance  # noqa: E402
from src.loaders import DATA, RESULTS, ROOT, output_dir, params  # noqa: E402

RUN = "paper/s6_cost"
UPSTREAM_ENERGY = "paper/s3_energy"
CODE = Path(__file__).resolve().parents[1]

_READS: set[Path] = set()


def _install_read_tracer() -> None:
    def _hook(event, args):
        if event != "open":
            return
        path, mode = args[0], (args[1] if len(args) > 1 else "r")
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        if isinstance(mode, str) and set(mode) & set("wax"):
            return
        try:
            q = Path(os.path.abspath(os.fspath(path)))
        except (TypeError, ValueError):
            return
        if q.name.endswith(provenance.SUFFIX) or q.suffix in (".py", ".pyc"):
            return
        if str(q).startswith((str(RESULTS), str(DATA))):
            _READS.add(q)
    sys.addaudithook(_hook)


def _sha256(q: Path) -> str:
    return hashlib.sha256(q.read_bytes()).hexdigest()


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, CODE / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_cost_per_km(out: Path) -> pd.DataFrame:
    """One table, every per-km figure, calibre columns mandatory."""
    conv = pd.read_csv(out / "cost_convention.csv")
    p3 = pd.read_csv(out / "plan3_cost.csv")

    rows = []
    for _, r in conv.iterrows():
        rows.append(dict(plan="diesel", run=r.run, n_feasible=int(r.n_feasible),
                         cny_per_km=r.diesel,
                         convention="fleet blend; whole fleet on diesel"))
        rows.append(dict(plan="plan2_swap", run=r.run,
                         n_feasible=int(r.n_feasible), cny_per_km=r.plan2,
                         convention=("fleet blend; served ships at swap "
                                     "price, unserved stay diesel")))
    for _, r in p3.iterrows():
        rows.append(dict(plan="plan3_mixed", run=r.run,
                         n_feasible=(156 if str(r.run).startswith(("C", "att"))
                                     else 165),
                         cny_per_km=r.plan3,
                         convention="mixed supply; per build_plan3 calibre"))
    d = pd.DataFrame(rows)
    d.to_csv(out / "cost_per_km_by_plan.csv", index=False)
    return d


def build_item_coverage(out: Path, cycles_per_yr: float,
                        calendar_cycles: float) -> pd.DataFrame:
    p = params()["lcc"]
    q = params()["queueing"]
    rows = [
        dict(item="station investment", status="modelled",
             parameter="lcc.station_cny_per_kw x lcc.station_power_kw + swap equipment",
             value=f"{p['station_cny_per_kw']} CNY/kW x {p['station_power_kw']} kW",
             grade="A (Dongjiadu tender; low/high from 2026 MSA awards)",
             note="annualised over station_life_yr at discount_rate; both swept",
             note_en="Annualized over the station life at the discount rate",
             treatment_en="Priced"),
        dict(item="spare batteries", status="modelled",
             parameter="lcc.battery_cny_per_kwh x packs x 1600 kWh",
             value=f"{p['battery_cny_per_kwh']} CNY/kWh",
             grade="A/B (Jiangsu Port Group; low 950 A, high 2000 B)",
             note="the dominant term; pack count from the inventory chain, checked against the energy side",
             note_en="The dominant term, over 80% of the annualized cost",
             treatment_en="Priced"),
        dict(item="charging equipment", status="inside-capex",
             parameter="lcc.station_cny_per_kw",
             value="not separable",
             grade="A (tender scope)",
             note="chargers, switchgear and grid connection are priced together "
                  "inside the whole-station investment intensity, same source "
                  "and same reason as grid reinforcement. Swap handling gear "
                  f"({p['swap_equipment_cny']:,} CNY lightweight robot / "
                  f"{p['swap_equipment_cny_crane']:,} 60 t crane, grade B) is "
                  "priced separately as swapping equipment",
             note_en="Chargers and switchgear are priced with the station",
             treatment_en="Within station investment"),
        dict(item="grid reinforcement", status="inside-capex",
             parameter="lcc.station_cny_per_kw",
             value="not separable",
             grade="A (tender scope)",
             note="the grade-A whole-station intensities price switchgear, grid "
                  "connection, monitoring and civil works together; claiming a "
                  "dedicated grid model would overstate. Per-station power "
                  "demand is quantified separately by the queueing chain",
             note_en="Connection and civil works are priced with the station",
             treatment_en="Within station investment"),
        dict(item="maintenance", status="modelled",
             parameter="lcc.om_rate_per_yr x station capex",
             value=f"{p['om_rate_per_yr']:.0%}/yr", grade="C (trade journal)",
             note="rate assumed and declared; swept implicitly via capex sweep",
             note_en="A stated rate on the station capital, swept with it",
             treatment_en="Priced"),
        dict(item="battery degradation", status="modelled",
             parameter="lcc.battery_calendar_life_yr (annuity horizon); "
                       "SOH sweep in build_uncertainty",
             value=f"{p['battery_calendar_life_yr']} yr; measured "
                   f"{cycles_per_yr:.1f} cycles/pack/yr -> {calendar_cycles:.0f} "
                   f"cycles over that horizon, vs a "
                   f"{p['battery_cycle_life_low']:,}-{p['battery_cycle_life']:,} "
                   f"cycle rating",
             grade="B (EVE LF280K, named vessels)",
             note="calendar life sets the annuity horizon; cycle life does not "
                  "bind at this fleet's swap frequency (~48 cycles/pack/yr, so "
                  "~477 over ten years against a 6,000+ rating) and is therefore "
                  "not priced. Coverage under SOH 1.0->0.7 is swept in sensitivity.csv; "
                  "the flexible plan loses relatively less, not absolutely less",
             note_en="Calendar life sets the horizon, not cycle life",
             treatment_en="Priced"),
        dict(item="inventory holding", status="modelled",
             parameter="battery_annual (capital carrying cost of station stock)",
             value="annuity(discount_rate, battery life) x battery capex",
             grade="derived",
             note="the held stock is the spare-battery fleet, so its carrying "
                  "cost is the annualised battery capital; no separate "
                  "warehousing term is claimed",
             note_en="The held inventory is the spare-battery fleet",
             treatment_en="Priced"),
        dict(item="rebalancing (transfer)", status="modelled",
             parameter="lcc.transfer_cny_per_pack_km x frozen annual pack-km",
             value=f"{p['transfer_cny_per_pack_km']} CNY/(pack.km), "
                   f"sweep {p['transfer_cny_per_pack_km_sweep']}",
             grade="assumed (stated): full-truckload road freight rate, 25 t/pack",
             note="annual pack-km from the transfer program, priced at the stated haulage rate",
             note_en="A stated road freight rate",
             treatment_en="Priced"),
        dict(item="delay costs", status="quantified-not-priced",
             parameter="queueing chain: waiting-time distribution per station",
             value=f"berth sweep {q['berth_sweep']}, service level "
                   f"p95 wait <= {q['service_level_wait_h']} h",
             grade="n/a",
             note="waiting is reported in hours with p25/p50/p75; "
                  "no defensible CNY/h exists for these fleets",
             note_en="No defensible value of time for these fleets",
             treatment_en="Reported in hours"),
    ]
    d = pd.DataFrame(rows)
    too_long = d[d.note_en.str.len() > 90]
    bad = {r.item: r.note_en for _, r in d.iterrows()
           if any(ord(ch) > 127 for ch in r.note_en)}

    TREATMENT = {"modelled": "Priced",
                 "inside-capex": "Within station investment",
                 "quantified-not-priced": "Reported in hours"}
    for _i, r in d.iterrows():
        pass
    printed = " ".join(list(d.treatment_en) + list(d.note_en)).lower()
    for word in ("capex", "modelled", "not-priced", "lcc.", "params", "grade"):
        pass


    unpriced = set(d[d.status != "modelled"].item)


    lcc_keys = set(params()["lcc"])
    lcc_src = (CODE / "2_analysis" / "build_lcc.py").read_text(encoding="utf-8")
    import re as _re
    for _, r in d[d.status == "modelled"].iterrows():
        words = set(_re.findall(r"[a-z_][a-z0-9_]*", r.parameter))
        named = {w for w in words if w in lcc_keys}
        named |= set(_re.findall(r"lcc\.([a-z_]+)", r.parameter))
        for k in named:
            pass

    d.to_csv(out / "cost_item_coverage.csv", index=False)
    return d


COST_CLASSES = ("station investment", "spare batteries", "charging equipment",
                 "grid reinforcement", "maintenance", "battery degradation",
                 "inventory holding", "rebalancing (transfer)", "delay costs")

UNPRICED_CLASSES = {"charging equipment", "grid reinforcement", "delay costs"}


def annual_cycles_per_pack(out: Path) -> tuple[float, float]:
    """How often a station pack actually cycles, and therefore which life binds."""
    L = params()["lcc"]
    daily = pd.read_csv(output_dir("paper/s5_inventory") / "inventory_daily_paper8.csv")
    eas = daily[daily["mode"] == "EAS"]
    window_days = int(eas.day.max())
    issued = float(eas["load"].sum())
    stock = float(pd.read_csv(
        output_dir("paper/s5_inventory") / "window_sensitivity.csv"
    ).query("port == '__fleet__' and window_days == 90 and mode == 'EAS'").initial.iloc[0])

    per_year = issued * 365.0 / window_days
    cycles = per_year / stock
    calendar_cycles = cycles * float(L["battery_calendar_life_yr"])
    binding = min(float(L["battery_cycle_life"]), float(L["battery_cycle_life_low"]))
    return cycles, calendar_cycles


def build_step_attribution(out: Path) -> pd.DataFrame:
    lcc = _load("2_analysis/build_lcc.py", "build_lcc_attr")
    inv = _load("1_pipeline/build_inventory.py", "build_inventory_attr")
    from src import metrics
    from src.loaders import RESULTS
    import numpy as np

    P = params()
    c, e, L = P["cost"], P["energy"], P["lcc"]
    ze = float(e["depth_of_discharge"]) * float(e["utilisation_efficiency"])
    eta_c = float(L["energy_transfer_efficiency"])

    def per_ship(ops):
        return ops.groupby("mmsi").agg(energy=("energy", "sum"),
                                       dist=("distance", "sum"),
                                       feasible=("feasible", "max"))

    def columns(ps):
        metered = ps.energy * ze / eta_c
        electric = metered * c["swapping_price"]
        shaft = ps.energy * ze / c["diesel_combustion_efficiency"]
        diesel = shaft / c["diesel_lhv_kwh_per_litre"] * c["diesel_price_per_litre"]
        blended = pd.Series(np.where(ps.feasible == 1, electric, diesel),
                            index=ps.index)
        return (metrics._per_km(diesel, ps.dist),
                metrics._per_km(blended, ps.dist))

    ops0 = pd.read_csv(RESULTS / "flowfield_rebuild" / "operations_8stn_baseline.csv")
    rec3 = inv.operations_record(P["case"]["plan2_stations_8"])
    ops3 = rec3.rename(columns={"energy_kwh": "energy", "feasible": "feasible"})[
        ["mmsi", "start", "end", "distance", "energy", "feasible"]]
    ps0, ps3 = per_ship(ops0), per_ship(ops3)

    mid = ps0.copy()
    mid["feasible"] = ps3.feasible.reindex(ps0.index).fillna(0)

    d_D, p_D = columns(ps0)
    d_M, p_M = columns(mid)
    d_E, p_E = columns(ps3)

    rows = []
    for col, (D, M, E) in (("diesel", (d_D, d_M, d_E)),
                           ("plan2", (p_D, p_M, p_E))):
        total = E - D
        rows.append(dict(column=col, step="1 cohort only (165->156 ships)",
                         cny_per_km_from=D, cny_per_km_to=M, delta=M - D,
                         share_of_total=(M - D) / total))
        rows.append(dict(column=col, step="2 energy field",
                         cny_per_km_from=M, cny_per_km_to=E, delta=E - M,
                         share_of_total=(E - M) / total))
        rows.append(dict(column=col, step="D->E total",
                         cny_per_km_from=D, cny_per_km_to=E, delta=total,
                         share_of_total=1.0))

    d = pd.DataFrame(rows)


    conv = pd.read_csv(out / "cost_convention.csv")
    for tag, ps_cols in (("D ", (d_D, p_D)), ("E ", (d_E, p_E))):
        row = conv[conv.run.str.startswith(tag)].iloc[0]

    d.to_csv(out / "convention_step_attribution.csv", index=False)
    return d


def build_cost_by_stations(out: Path) -> pd.DataFrame:
    lcc = _load("2_analysis/build_lcc.py", "build_lcc_by_n")
    biv = _load("1_pipeline/build_inventory.py", "biv_by_n")
    c, p_ = params()["cost"], params()["lcc"]
    zeta_eta = (params()["energy"]["depth_of_discharge"]
                * params()["energy"]["utilisation_efficiency"])
    eta_c = p_["energy_transfer_efficiency"]

    cov = pd.read_csv(output_dir("paper/s4_siting") / "coverage.csv")
    all0 = cov[cov.growth == 0]
    g0 = all0[all0.plan == "plan2"]

    def priced_over(ports, flexible):
        rec3 = biv.operations_record(list(ports), source=output_dir(UPSTREAM_ENERGY),
                               flexible=flexible)
        ops = rec3.rename(columns={"energy_kwh": "energy", "feasible": "feasible"})[
            ["mmsi", "start", "end", "distance", "energy", "feasible"]]
        return lcc.price(ops, c, zeta_eta, zeta_eta, eta_c)

    rows = []
    paper8 = params()["case"]["plan2_stations_8"]
    for _, r in g0.sort_values("n_stations").iterrows():
        n = int(r.n_stations)
        ports = list(paper8) if n == 8 else str(r.stations).split("|")
        priced = priced_over(ports, True)
        ports1 = str(all0[(all0.n_stations == n)
                          & (all0.plan == "plan1")].stations.iloc[0]).split("|")
        p1 = priced_over(ports1, False)
        rows.append(dict(n_stations=n,
                         stations="|".join(ports),
                         station_set=("manuscript (D1)" if n == 8
                                      else "re-optimised"),
                         n_feasible=int(priced["n_feasible"]),
                         diesel=priced["diesel"],
                         plan1=p1["plan2"],
                         plan2=priced["plan2"],
                         plan1_stations="|".join(ports1),
                         plan1_n_feasible=int(p1["n_feasible"]),
                         saving_pct=(priced["diesel"] - priced["plan2"])
                         / priced["diesel"] * 100,
                         plan1_saving_pct=(priced["diesel"] - p1["plan2"])
                         / priced["diesel"] * 100))
    t = pd.DataFrame(rows)


    conv = pd.read_csv(out / "cost_convention.csv")
    e = conv[conv.run.str.startswith("E ")].iloc[0]
    eight = t[t.n_stations == 8].iloc[0]
    for col in ("diesel", "plan2"):
        pass
    t.to_csv(out / "cost_by_stations.csv", index=False)
    return t


def build_price_grid(out: Path) -> pd.DataFrame:
    """Cost surface over (electricity price x station count)."""
    lcc = _load("2_analysis/build_lcc.py", "build_lcc_grid")
    p3 = _load("2_analysis/build_plan3.py", "build_plan3_grid")
    biv = _load("1_pipeline/build_inventory.py", "biv_grid")
    c0, p_ = params()["cost"], params()["lcc"]
    zeta_eta = (params()["energy"]["depth_of_discharge"]
                * params()["energy"]["utilisation_efficiency"])
    eta_c = p_["energy_transfer_efficiency"]
    ratio = c0["charging_price"] / c0["swapping_price"]

    cov = pd.read_csv(output_dir("paper/s4_siting") / "coverage.csv")
    g0 = cov[(cov.growth == 0) & (cov.plan == "plan2")]
    paper8 = params()["case"]["plan2_stations_8"]
    own = p3.own_ports(pd.read_csv(output_dir(UPSTREAM_ENERGY)
                                   / "segment_energy.csv"))

    sweep = params()["cost"]["swapping_price_sweep"]
    prices = [round(x, 2) for x in np.arange(sweep["low"], sweep["high"] + 1e-9,
                                              sweep["step"])]
    rows = []
    for _, r in g0.sort_values("n_stations").iterrows():
        n = int(r.n_stations)
        ports = list(paper8) if n == 8 else str(r.stations).split("|")
        rec3 = biv.operations_record(ports, source=output_dir(UPSTREAM_ENERGY))
        ops = rec3.rename(columns={"energy_kwh": "energy", "feasible": "feasible"})[
            ["mmsi", "start", "end", "distance", "energy", "feasible"]]
        rec = rec3.rename(columns={"energy_kwh": "energy", "feasible": "feasible",
                                 "optimised swapping": "swapped"})[
            ["mmsi", "start", "end", "distance", "energy", "feasible",
             "swapped", "cargo_port"]].copy()
        rec["cargo_port"] = p3.paper_cargo_flag(rec.mmsi, rec.end, own)
        for tar in prices:
            c = dict(c0, swapping_price=tar, charging_price=round(tar * ratio, 6))
            swap = lcc.price(ops, c, zeta_eta, zeta_eta, eta_c)["plan2"]
            mixed = p3.price_mixed(rec, c, zeta_eta, zeta_eta, eta_c)
            rows.append(dict(n_stations=n, swapping_price=tar,
                             charging_price=round(tar * ratio, 6),
                             station_set="manuscript (D1)" if n == 8
                                         else "re-optimised",
                             plan2=swap, plan3=mixed, difference=swap - mixed))
    t = pd.DataFrame(rows)
    cell = t[(t.n_stations == 8) & (np.isclose(t.swapping_price,
                                               c0["swapping_price"]))]
    conv = pd.read_csv(out / "cost_convention.csv")
    e = conv[conv.run.str.startswith("E ")].iloc[0]
    p3c = pd.read_csv(out / "plan3_cost.csv")
    c_run = p3c[p3c.run.str.startswith("C ")].iloc[0]
    t.to_csv(out / "cost_price_grid.csv", index=False)
    return t


def build_density_grid(out: Path) -> pd.DataFrame:
    """Cost over (electricity price x energy density x station count)."""
    lcc = _load("2_analysis/build_lcc.py", "build_lcc_dens")
    p3 = _load("2_analysis/build_plan3.py", "build_plan3_dens")
    biv = _load("1_pipeline/build_inventory.py", "biv_dens")
    c0, p_ = params()["cost"], params()["lcc"]
    zeta_eta = (params()["energy"]["depth_of_discharge"]
                * params()["energy"]["utilisation_efficiency"])
    eta_c = p_["energy_transfer_efficiency"]
    ratio = c0["charging_price"] / c0["swapping_price"]

    cov = pd.read_csv(output_dir("paper/s4_siting") / "coverage.csv")
    paper8 = params()["case"]["plan2_stations_8"]
    own = p3.own_ports(pd.read_csv(output_dir(UPSTREAM_ENERGY)
                                   / "segment_energy.csv"))
    sweep = params()["cost"]["swapping_price_sweep"]
    prices = [round(x, 2) for x in np.arange(sweep["low"], sweep["high"] + 1e-9,
                                             sweep["step"])]

    rows = []
    for growth in params()["case"]["growth_sweep"]:
        for n in params()["case"]["station_counts"]:
            sel = cov[(cov.growth == growth) & (cov.n_stations == n)]
            ports2 = (list(paper8) if (growth == 0 and n == 8)
                      else str(sel[sel.plan == "plan2"].stations.iloc[0]).split("|"))
            ports1 = str(sel[sel.plan == "plan1"].stations.iloc[0]).split("|")

            def record(ports, flexible):
                return biv.operations_record(list(ports),
                                         source=output_dir(UPSTREAM_ENERGY),
                                         flexible=flexible, growth_pct=growth)

            v2, v1 = record(ports2, True), record(ports1, False)
            v1s = record(ports2, False)
            cols = ["mmsi", "start", "end", "distance", "energy", "feasible"]
            ops2 = v2.rename(columns={"energy_kwh": "energy",
                                      "feasible": "feasible"})[cols]
            ops1 = v1.rename(columns={"energy_kwh": "energy",
                                      "feasible": "feasible"})[cols]
            ops1s = v1s.rename(columns={"energy_kwh": "energy",
                                        "feasible": "feasible"})[cols]
            rec = v2.rename(columns={"energy_kwh": "energy", "feasible": "feasible",
                                     "optimised swapping": "swapped"})[
                cols + ["swapped", "cargo_port"]].copy()
            rec["cargo_port"] = p3.paper_cargo_flag(rec.mmsi, rec.end, own)
            for tar in prices:
                c = dict(c0, swapping_price=tar,
                         charging_price=round(tar * ratio, 6))
                sw = lcc.price(ops2, c, zeta_eta, zeta_eta, eta_c)
                nf = lcc.price(ops1, c, zeta_eta, zeta_eta, eta_c)
                nfs = lcc.price(ops1s, c, zeta_eta, zeta_eta, eta_c)
                mixed = p3.price_mixed(rec, c, zeta_eta, zeta_eta, eta_c)
                rows.append(dict(
                    growth=growth, n_stations=n, swapping_price=tar,
                    charging_price=round(tar * ratio, 6),
                    station_set=("manuscript (D1)" if (growth == 0 and n == 8)
                                 else "re-optimised at this density"),
                    stations="|".join(ports2),
                    n_feasible=int(sw["n_feasible"]),
                    diesel=sw["diesel"], plan1=nf["plan2"], plan2=sw["plan2"],
                    plan3=mixed, difference=sw["plan2"] - mixed,
                    plan1_same_set=nfs["plan2"],
                    plan1_same_set_n_feasible=int(nfs["n_feasible"]),
                    plan1_own_n_feasible=int(nf["n_feasible"]),
                    plan1_stations="|".join(ports1)))
    t = pd.DataFrame(rows)

    conv = pd.read_csv(out / "cost_convention.csv")
    e = conv[conv.run.str.startswith("E ")].iloc[0]
    cell = t[(t.growth == 0) & (t.n_stations == 8)
             & (np.isclose(t.swapping_price, c0["swapping_price"]))]
    p3c = pd.read_csv(out / "plan3_cost.csv")
    c_run = p3c[p3c.run.str.startswith("C ")].iloc[0]
    t.to_csv(out / "cost_density_grid.csv", index=False)
    return t


def build_total_cost(out: Path) -> pd.DataFrame:
    """Operating cost plus the stations it takes, per km."""
    lcc = _load("2_analysis/build_lcc.py", "build_lcc_total")
    p_, c0 = params()["lcc"], params()["cost"]
    grid = pd.read_csv(out / "cost_density_grid.csv")
    base = grid[np.isclose(grid.swapping_price, c0["swapping_price"])]

    seg = pd.read_csv(output_dir(UPSTREAM_ENERGY) / "segment_energy.csv")
    window = pd.to_datetime(params()["case"]["ais_window"])
    span_days = (window[1] - window[0]).days
    fleet_km_year = (seg.distance.sum() * c0["nautical_mile_to_km"]
                     * 365.0 / span_days)

    crf = lcc.annuity(p_["discount_rate"], p_["station_life_yr"])
    prices = [("low", p_["station_cny_per_kw_low"]),
              ("central", p_["station_cny_per_kw"]),
              ("high", p_["station_cny_per_kw_high"])]

    rows = []
    for _, r in base.iterrows():
        for label, kw_price in prices:
            capex = kw_price * p_["station_power_kw"] + p_["swap_equipment_cny"]
            annual = capex * crf + capex * p_["om_rate_per_yr"]
            infra = int(r.n_stations) * annual / fleet_km_year
            for plan in ("plan1", "plan2", "plan3"):
                rows.append(dict(
                    growth=int(r.growth), n_stations=int(r.n_stations),
                    plan=plan, station_price=label,
                    station_cny_per_kw=kw_price,
                    station_annual_cny=annual,
                    operating_cny_per_km=float(r[plan]),
                    infrastructure_cny_per_km=infra,
                    total_cny_per_km=float(r[plan]) + infra,
                    n_feasible=int(r.n_feasible)))
    t = pd.DataFrame(rows)

    for label, _kw in prices:
        z = t[(t.growth == 100) & (t.plan == "plan2")
              & (t.station_price == label)].set_index("n_stations")
    t.to_csv(out / "cost_total_by_stations.csv", index=False)
    return t


def main() -> None:
    _install_read_tracer()
    out = output_dir(RUN)

    _load("2_analysis/build_lcc.py", "build_lcc_main").main(run=RUN)

    _load("2_analysis/build_plan3.py", "build_plan3_main").main(run=RUN)

    d = build_cost_per_km(out)

    cbs = build_cost_by_stations(out)
    for _, r in cbs.iterrows():
        pass

    grid = build_price_grid(out)

    dens = build_density_grid(out)
    c0 = params()["cost"]
    base = dens[np.isclose(dens.swapping_price, c0["swapping_price"])]
    for g in sorted(dens.growth.unique()):
        r = base[(base.growth == g) & (base.n_stations == 8)].iloc[0]
    nine = base[(base.growth == 100) & (base.n_stations == 9)].iloc[0]
    eight = base[(base.growth == 100) & (base.n_stations == 8)].iloc[0]

    tot = build_total_cost(out)
    z = tot[(tot.growth == 100) & (tot.plan == "plan2")
            & (tot.station_price == "central")].set_index("n_stations")
    d89 = z.loc[9, "total_cny_per_km"] - z.loc[8, "total_cny_per_km"]
    for lab in ("low", "central", "high"):
        y = tot[(tot.growth == 100) & (tot.plan == "plan2")
                & (tot.station_price == lab)].set_index("n_stations")

    att = build_step_attribution(out)
    for col in ("diesel", "plan2"):
        g = att[att.column == col]
        tot = g[g.step == "D->E total"].iloc[0]
        for _, r in g[g.step.str.startswith(("1", "2"))].iterrows():
            pass

    cycles_per_yr, calendar_cycles = annual_cycles_per_pack(out)
    cov = build_item_coverage(out, cycles_per_yr, calendar_cycles)
    for _, r in cov.iterrows():
        pass

    paper_energy = output_dir("paper/s3_energy") / "segment_energy.csv"
    energy_path = RESULTS / "coefficient_correction" / "segment_energy.csv"
    inputs = [
        paper_energy,
        output_dir("paper/s4_siting") / "coverage.csv",
        output_dir("paper/s5_inventory") / "inventory_daily_paper8.csv",
        output_dir("paper/s5_inventory") / "window_sensitivity.csv",
        RESULTS / "flowfield_rebuild" / "operations_8stn_baseline.csv",
        energy_path,
        RESULTS / "coefficient_correction" / "inventory.csv",
        RESULTS / "coefficient_correction" / "transfer_frontier.csv",
        DATA / "02_channel_topology" / "port_topology.csv",
        DATA / "03_ship_parameters" / "172_ship_parameters.xlsx",
        DATA / "04_paper_baseline" / "segments_legs_energy_and_time.xlsx",
    ] + [DATA / "04_paper_baseline" / "siting_result" / f"n_stations{k}"
         / f"growth 0%-mixed supply-{k}-optimised swapping.xlsx" for k in (4, 8, 9)]
    missing = [str(q.relative_to(ROOT)) for q in inputs if not q.exists()]


    declared = {Path(os.path.abspath(q)) for q in inputs}
    traced = {q for q in _READS if not str(q).startswith(str(out))}
    undeclared = sorted(str(q.relative_to(ROOT)) for q in traced - declared)
    untraced = sorted(str(q.relative_to(ROOT)) for q in declared - traced)

    upstream = [q for q in inputs
                if str(q).startswith(str(RESULTS))
                and not str(q).startswith(str(RESULTS / "paper"))]
    n = 0
    for prod in sorted(out.rglob("*")):
        if prod.is_file() and not prod.name.endswith(provenance.SUFFIX):
            provenance.record(prod, script=__file__, inputs=inputs,
                              params_used={"lcc": params()["lcc"],
                                           "cost": params()["cost"]})
            n += 1
    for q in upstream:
        pass
    if untraced:
        pass


if __name__ == "__main__":
    main()
