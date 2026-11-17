import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401,E402

import build_coupled_inventory as ci  # noqa: E402
import build_pack_soc as fleet_model  # noqa: E402
import build_transfer_model as transfer_lp  # noqa: E402
import rule_transfer_allocation as rule_alloc  # noqa: E402
from build_lcc import annuity  # noqa: E402
from build_transfer_model import port_distances  # noqa: E402
from src.loaders import RESULTS, params, write_dir  # noqa: E402

PREMIUM, MAINT, LIFE = 1_000_000.0, 200_000.0, 10


def main():
    p = params(); lcc = p["lcc"]
    case = fleet_model.build_case()
    inputs = rule_alloc.station_inputs(case)
    load, ready, floor = (x.astype(float) for x in rule_alloc.stack(case, inputs["DASM"]))
    dist = port_distances(case.ports)
    travel = np.ceil(dist / p["inventory"]["transfer_km_per_day"]).astype(int)
    transit = travel + (travel > 0)
    lp = transfer_lp.solve(load, ready, dist, 0.0, 0.0, 0, transit, trucks=ci.RULE_FLEET)
    bound = int(np.ceil(lp["total_i0"]))
    factor = 365.0 / len(case.days)
    cost = pd.read_csv(RESULTS / "service_cost/annualized_cost_same_service.csv")
    crf_b = annuity(lcc["discount_rate"], lcc["battery_calendar_life_yr"])
    crf_p = annuity(lcc["discount_rate"], LIFE)
    rows = []
    for prices, g in cost.groupby("prices", sort=False):
        d = g[g.scenario == "Diesel fleet"].iloc[0]
        e = g[g.scenario == "EASM no transfer"].iloc[0]
        b = g[g.scenario == "DASM no transfer"].iloc[0]
        pk = 1600 * b.battery_cny_per_kwh * crf_b
        n_el = int(e.ships_electric)
        machinery = n_el * PREMIUM * crf_p
        maint = -n_el * MAINT
        dasm = dict(
            diesel=b.annual_diesel_cny, grid=b.annual_grid_cny,
            onboard=b.annual_onboard_battery_cny,
            other=(bound + b.initial_offsite_packs - b.peak_simultaneous_onboard_packs) * pk,
            facility=b.annual_station_facility_cny, equipment=b.annual_swap_equipment_cny,
            om=b.annual_station_om_cny,
            loaded=lp["pack_km"] * factor * lcc["transfer_cny_per_pack_km"],
            empty=lp["empty_km"] * factor * lcc["transfer_cny_per_pack_km"],
            premium=machinery, maint=maint, packs=bound + int(b.initial_offsite_packs),
            station_packs=bound, grid_kwh=b.annual_grid_kwh, ship=b.annual_ship_operating_cny)
        easm = dict(
            diesel=e.annual_diesel_cny, grid=e.annual_grid_cny,
            onboard=e.annual_onboard_battery_cny, other=e.annual_other_battery_cny,
            facility=e.annual_station_facility_cny, equipment=e.annual_swap_equipment_cny,
            om=e.annual_station_om_cny, loaded=0.0, empty=0.0, premium=machinery, maint=maint,
            packs=int(e.battery_assets_packs), station_packs=int(e.initial_station_packs),
            grid_kwh=e.annual_grid_kwh, ship=e.annual_ship_operating_cny)
        dsl = dict(diesel=d.annual_diesel_cny, grid=0, onboard=0, other=0, facility=0, equipment=0,
                   om=0, loaded=0, empty=0, premium=0, maint=0, packs=0, station_packs=0,
                   grid_kwh=0, ship=d.annual_ship_operating_cny)
        km = d.annual_fleet_km
        keys = ["diesel", "grid", "onboard", "other", "facility", "equipment", "om",
                "loaded", "empty", "premium", "maint"]
        for name, c in (("Diesel", dsl), ("EASM", easm), ("DASM with transfers (bound)", dasm)):
            total = sum(c[k] for k in keys)
            batt = c["onboard"] + c["other"]
            row = dict(prices=prices, case=name, battery_cny_per_kwh=b.battery_cny_per_kwh,
                       **{f"{k}_mcny": c[k] / 1e6 for k in keys},
                       total_mcny=total / 1e6, system_cny_per_km=total / km,
                       shipowner_cny_per_km=c["ship"] / km, battery_packs=c["packs"],
                       station_opening_packs=c["station_packs"],
                       battery_share=batt / total if total else 0.0)
            if batt:
                row["break_even_battery_cny_per_kwh"] = (d.annual_diesel_cny - (total - batt)) / (
                    batt / b.battery_cny_per_kwh)
                row["operator_break_even_fee_cny_per_kwh"] = (
                    total - c["diesel"] - c["premium"] - c["maint"]) / c["grid_kwh"]
                row["diesel_parity_fee_cny_per_kwh"] = (d.annual_diesel_cny - c["diesel"]) / c["grid_kwh"]
            rows.append(row)
    t = pd.DataFrame(rows)
    out = write_dir("system_cost_accounts")
    t.to_csv(out / "01_G3G4.csv", index=False)
    with pd.option_context("display.width", 250):
        pass


if __name__ == "__main__":
    main()
