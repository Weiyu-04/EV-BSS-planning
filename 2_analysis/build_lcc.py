
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401

from build_inventory import operations_record  # noqa: E402
from src import metrics  # noqa: E402
from src.loaders import RESULTS, output_dir, params, ship_parameters, write_dir  # noqa: E402
from src import provenance  # noqa: E402

FIELD_RUN = RESULTS / "flowfield_rebuild"
COEF_RUN = RESULTS / "coefficient_correction"
WINDOW_DAYS = 90
YEAR = 365.0 / WINDOW_DAYS


def price(ops: pd.DataFrame, c: dict, elec_factor: float, diesel_factor: float,
          eta_c: float = 1.0) -> dict:
    """Per-km cost under one convention."""
    per_ship = ops.groupby("mmsi").agg(energy=("energy", "sum"),
                                       dist=("distance", "sum"),
                                       feasible=("feasible", "max"))
    metered = per_ship.energy * elec_factor / eta_c
    electric = metered * c["swapping_price"]

    shaft = per_ship.energy * diesel_factor / c["diesel_combustion_efficiency"]
    diesel = shaft / c["diesel_lhv_kwh_per_litre"] * c["diesel_price_per_litre"]

    blended = pd.Series(np.where(per_ship.feasible == 1, electric, diesel),
                        index=per_ship.index)
    return dict(diesel=metrics._per_km(diesel, per_ship.dist),
                plan2=metrics._per_km(blended, per_ship.dist),
                metered_kwh=float(metered.sum()),
                metered_kwh_served=float(metered[per_ship.feasible == 1].sum()),
                n_feasible=int((per_ship.feasible == 1).sum()))


def part1(c: dict, p: dict, zeta_eta: float) -> pd.DataFrame:
    ops0 = pd.read_csv(FIELD_RUN / "operations_8stn_baseline.csv")
    rec3 = operations_record(params()["case"]["plan2_stations_8"])
    ops3 = rec3.rename(columns={"energy_kwh": "energy", "feasible": "feasible"})[
        ["mmsi", "start", "end", "distance", "energy", "feasible"]]
    reference_diesel = c["diesel_electric_to_mechanical_efficiency"] * c["diesel_transmission_efficiency"]
    eta_c = p["energy_transfer_efficiency"]

    rows = []
    for label, ops, ef, df_, ce in [
            ("A control, paper convention", ops0, 1.0, reference_diesel, 1.0),
            ("B electric side corrected", ops0, zeta_eta, reference_diesel, 1.0),
            ("C +diesel side corrected", ops0, zeta_eta, zeta_eta, 1.0),
            ("D +energy transfer efficiency", ops0, zeta_eta, zeta_eta, eta_c),
            ("E corrected energy, main result", ops3, zeta_eta, zeta_eta, eta_c)]:
        r = price(ops, c, ef, df_, ce)
        r.update(run=label, eta_c=ce)
        rows.append(r)

    d = pd.DataFrame(rows)
    d["saving"] = d.diesel - d.plan2
    d["saving_pct"] = d.saving / d.diesel * 100

    for _, r in d.iterrows():
        pass

    a = d.iloc[0]
    assert abs(a.diesel - 54.20) < 0.15, f"control diesel {a.diesel} != 54.20"
    assert abs(a.saving_pct - 33.0) < 0.3, f"control saving {a.saving_pct} != 33.0"

    e = d.iloc[4]
    assert int(e.n_feasible) == 156, (
        f"run E has {int(e.n_feasible)} feasible ships; the paper's eight "
        f"stations give 156")
    fleet_energy_total = float(pd.read_csv(COEF_RUN / "segment_energy.csv").energy_new.sum())
    assert abs(e.metered_kwh - fleet_energy_total * zeta_eta / eta_c) / e.metered_kwh < 1e-9, (
        "run E's metered energy does not match the leg energy table")
    assert abs(eta_c - 0.90) < 1e-12, (
        f"energy transfer efficiency is {eta_c}, not the declared assumption 0.90; "
        f"changing it moves the headline saving and the sensitivity table")
    served_gwh = float(e.metered_kwh_served) * YEAR / 1e6
    assert abs(served_gwh - 150.3) < 0.5, (
        f"served annual energy {served_gwh:.1f} GWh != the reported 150.3 GWh; "
        f"either the cohort or the energy transfer efficiency has moved")
    assert 53.0 < a.diesel < 55.5, (
        f"control diesel {a.diesel:.2f} CNY/km is outside the plausible band; "
        f"a nm/km slip would read as {a.diesel * 1.852:.1f} or {a.diesel / 1.852:.1f}")


    for ce in p["energy_transfer_efficiency_sweep"]:
        r = price(ops3, c, zeta_eta, zeta_eta, ce)
        s = (r["diesel"] - r["plan2"]) / r["diesel"] * 100
    return d


def annuity(rate: float, years: float) -> float:
    """Capital recovery factor; falls back to straight-line at zero rate."""
    if rate <= 0:
        return 1.0 / years
    return rate / (1 - (1 + rate) ** -years)


def lcc(packs: float, metered_kwh_year: float, p: dict, c: dict,
        battery_price: float, station_kw_price: float, swap_equip: float,
        rate: float, n_stations: int = 8) -> dict:
    """Annualised life-cycle cost of the eight-station system."""
    station_capex = n_stations * (station_kw_price * p["station_power_kw"]
                                  + swap_equip)
    battery_capex = packs * p_pack_kwh() * battery_price

    crf_station = annuity(rate, p["station_life_yr"])
    crf_battery = annuity(rate, p["battery_calendar_life_yr"])

    return dict(
        station_annual=station_capex * crf_station,
        battery_annual=battery_capex * crf_battery,
        om_annual=station_capex * p["om_rate_per_yr"],
        energy_station=metered_kwh_year * p["electricity_cny_per_kwh"],
        energy_ship=metered_kwh_year * c["swapping_price"],
        station_capex=station_capex, battery_capex=battery_capex, packs=packs)


def p_pack_kwh() -> float:
    """Pack capacity in kWh, from params.yaml."""
    return float(params()["inventory"]["battery_unit_capacity_kwh"])


def _served_first_load(stations: list[str]) -> pd.DataFrame:
    """Per-ship first-day pack load for the served ships, with its start port."""
    ops = operations_record(stations)
    ships = ship_parameters().set_index("mmsi")
    served = ops.groupby("mmsi")["feasible"].max()
    served = served[served == 1].index
    first = ops.groupby("mmsi", sort=False).first()["start"]
    packs = ships["rated_capacity_kwh"] / p_pack_kwh()
    df = pd.DataFrame({"start": first, "packs": packs}).loc[list(served)]
    assert df.packs.notna().all(), "ship/leg tables failed to align on mmsi"
    return df


def _onboard_packs(stations: list[str]) -> float:
    """Packs carried by the served ships themselves, the asset base's floor."""
    return float(_served_first_load(stations).packs.sum())


def _offsite_first_load(stations: list[str]) -> float:
    """Packs loaded on day one at ports outside the eight stations."""
    df = _served_first_load(stations)
    off = ~df.start.isin(stations)
    return float(df.loc[off, "packs"].sum()), int((~off).sum()), int(off.sum())


def main(run: str = "coefficient_correction",
         upstream: str = "coefficient_correction") -> None:
    out = write_dir(run)
    src = output_dir(upstream)
    c, p = params()["cost"], params()["lcc"]
    inv = params()["inventory"]
    zeta_eta = params()["energy"]["depth_of_discharge"] * \
        params()["energy"]["utilisation_efficiency"]

    d1 = part1(c, p, zeta_eta)
    d1.to_csv(out / "cost_convention.csv", index=False)
    provenance.record(out / "cost_convention.csv", script=__file__,
                      inputs=[FIELD_RUN / "operations_8stn_baseline.csv",
                              COEF_RUN / "segment_energy.csv"],
                      params_used={"cost": c, "zeta_eta": zeta_eta})

    metered_year = float(d1.iloc[4].metered_kwh_served) * YEAR
    served_share = d1.iloc[4].metered_kwh_served / d1.iloc[4].metered_kwh

    t3 = pd.read_csv(src / "inventory.csv")
    eas = t3[(t3.run == "D +cohort") & (t3["mode"] == "EAS")].initial.sum()

    offsite, n_inside, n_outside = _offsite_first_load(
        params()["case"]["plan2_stations_8"])
    n_feasible = int(d1.iloc[4].n_feasible)
    assert n_inside + n_outside == n_feasible, (
        f"battery cohort {n_inside}+{n_outside} != electricity cohort "
        f"{n_feasible}; the two sides of the LCC are counting different ships")

    fr = pd.read_csv(src / "transfer_frontier.csv")
    fr = fr[fr.trucks_per_day == 0] if "trucks_per_day" in fr.columns else fr
    front = fr[(fr["mode"] == "EAS") & (~fr.terminal) & (fr.moved > 0)].sort_values("lam")
    op = front[front.lam == 0.001].iloc[0]
    packs_no_tr, packs_tr = float(eas) + offsite, float(op.total_i0) + offsite
    pack_km = float(op.pack_km) * YEAR
    mean_haul_km = float(op.pack_km) / float(op.moved)
    assert 100.0 < mean_haul_km < 400.0, (
        f"mean transfer haul {mean_haul_km:.1f} km is implausible for the lower "
        f"Yangtze; a dropped nm->km conversion would read as "
        f"{mean_haul_km / 1.852:.1f} km")


    rows = []
    base_eta = p["energy_transfer_efficiency"]
    for rate in p["discount_rate_sweep"]:
      for life in p["station_life_sweep"]:
        for eta_c in p["energy_transfer_efficiency_sweep"]:
          for bp, bl in [(p["battery_cny_per_kwh_low"], "battery 950"),
                         (p["battery_cny_per_kwh"], "battery 1400"),
                         (p["battery_cny_per_kwh_high"], "battery 2000")]:
            for sp, sl in [(p["station_cny_per_kw_low"], "station 2000/kW"),
                           (p["station_cny_per_kw"], "station 5312/kW"),
                           (p["station_cny_per_kw_high"], "station 6050/kW")]:
                for eq, eql in [(p["swap_equipment_cny"], "lightweight swap robot"),
                                (p["swap_equipment_cny_crane"], "60t harbour crane")]:
                    pp = dict(p, station_life_yr=life)
                    r = lcc(packs_no_tr, metered_year * base_eta / eta_c,
                            pp, c, bp, sp, eq, rate)
                    total = (r["station_annual"] + r["battery_annual"]
                             + r["om_annual"] + r["energy_station"])
                    rows.append(dict(rate=rate, life=life, eta=eta_c, battery=bl,
                                     station=sl, equip=eql,
                                     is_baseline=(rate == p["discount_rate"]
                                                  and life == p["station_life_yr"]
                                                  and eta_c == base_eta
                                                  and bp == p["battery_cny_per_kwh"]
                                                  and sp == p["station_cny_per_kw"]
                                                  and eq == p["swap_equipment_cny"]),
                                     **{k: v for k, v in r.items()}, total=total))
    d2 = pd.DataFrame(rows)
    d2.to_csv(out / "lcc_sensitivity.csv", index=False)

    base = d2[(d2.rate == 0.05) & (d2.life == p["station_life_yr"])
              & (d2.eta == base_eta) & (d2.battery == "battery 1400")
              & (d2.station == "station 5312/kW") & (d2.equip == "lightweight swap robot")].iloc[0]
    for k, lab in [("station_annual", "station investment annualised"), ("battery_annual", "spare batteries annualised"),
                   ("om_annual", "maintenance"), ("energy_station", "electricity purchased, station side, 0.8 CNY/kWh)")]:
        pass
    ship_total = base.total - base.energy_station + base.energy_ship

    lo, hi = d2.total.min(), d2.total.max()
    sh = d2.energy_station / d2.total * 100
    bt = d2.battery_annual / d2.total * 100

    onboard = float(_onboard_packs(params()["case"]["plan2_stations_8"]))
    for lab, i0 in [("no transfer, paper model", float(eas))] + [
            (f"transfer allowed lambda={lm:g}", float(front[front.lam == lm].total_i0.iloc[0]))
            for lm in (0.0, 0.001) if (front.lam == lm).any()]:
        tot = i0 + offsite

    saved_packs = packs_no_tr - packs_tr
    crf = annuity(p["discount_rate"], p["battery_calendar_life_yr"])
    freight = p["transfer_cny_per_pack_km"]
    rows_t = []
    for bp in (p["battery_cny_per_kwh_low"], p["battery_cny_per_kwh"],
               p["battery_cny_per_kwh_high"]):
        saved = saved_packs * p_pack_kwh() * bp * crf
        cost = pack_km * freight
        breakeven = saved / pack_km if pack_km > 0 else float("nan")
        rows_t.append(dict(battery_cny_per_kwh=bp, saved_cny=saved,
                           transfer_cny=cost, net_cny=saved - cost,
                           breakeven_cny_per_pack_km=breakeven))
    pd.DataFrame(rows_t).to_csv(out / "transfer_economics.csv", index=False)
    for _, r in front.iterrows():
        saved_r = packs_no_tr - (r.total_i0 + offsite)
        sv = saved_r * p_pack_kwh() * p["battery_cny_per_kwh"] * crf
        pk = r.pack_km * YEAR
        be = sv / pk if pk > 0 else float("nan")
        flag = "  <- same order as the assumed freight rate" if be < freight * 3 else ""
    lo_be = min(sv / (r.pack_km * YEAR) for _, r in front.iterrows()
                for sv in [(packs_no_tr - (r.total_i0 + offsite)) * p_pack_kwh()
                           * p["battery_cny_per_kwh"] * crf] if r.pack_km > 0)


if __name__ == "__main__":
    main()
