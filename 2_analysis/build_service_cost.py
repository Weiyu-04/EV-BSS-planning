
from __future__ import annotations

from pathlib import Path

import pandas as pd

import build_coupled_inventory as stock_model
import build_pack_soc as fleet_model
from build_lcc import annuity
from src import provenance
from src.loaders import PARAMS_FILE, params, write_dir

RUN = "service_cost"


def fuel_litres(load_kwh, diesel):
    """The cost code uses useful work over diesel efficiency and LHV."""
    return load_kwh / (diesel["diesel_combustion_efficiency"] * diesel["diesel_lhv_kwh_per_litre"])


def operating_costs(fleet, scenario, assumptions):
    lcc, cost = assumptions["lcc"], assumptions["cost"]
    factor = 365.0 / len(fleet.days)
    diesel_case = scenario is None
    if diesel_case:
        n_electric, metered_kwh = 0, 0.0
        litres = fuel_litres(fleet.all_diesel_load_kwh * factor, cost)
    else:
        plan = fleet.plans[scenario.mode]
        n_electric = len(plan.ships)
        metered_kwh = plan.ships.load_kwh.sum() * factor / lcc["energy_transfer_efficiency"]
        litres = fuel_litres(fleet.retained_diesel_load_kwh * factor, cost)
    diesel_cost = litres * cost["diesel_price_per_litre"]
    return dict(
        ships_electric=n_electric, ships_diesel=fleet.ships_total - n_electric,
        sample_days=len(fleet.days), annual_fleet_km=fleet.fleet_km * factor,
        annual_diesel_litres=litres, annual_grid_kwh=metered_kwh,
        annual_diesel_cny=diesel_cost,
        annual_grid_cny=metered_kwh * lcc["electricity_cny_per_kwh"],
        annual_ship_operating_cny=diesel_cost + metered_kwh * cost["swapping_price"],
    )


def asset_costs(scenario, assumptions, battery_price, station_price, annual_factor):
    p = assumptions["lcc"]
    if scenario is None:
        power_kw, berths, opening, onboard, assets = 0, 0, 0, 0, 0
        loaded_km, empty_km = 0.0, 0.0
    else:
        stations = scenario.stations
        assert (stations.modeled_extra_supply_packs == 0).all()
        assert (stations.still_charging_or_waiting_packs == 0).all()
        assert (stations.berth_wait_p95_h <= assumptions["queueing"]["service_level_wait_h"]).all()
        power_kw = float(stations.grid_kw.sum())
        berths = int(stations.berths.sum())
        opening = int(stations.opening_charged_packs.sum())
        onboard = scenario.peak_simultaneous_onboard
        assets = scenario.total_battery_assets
        loaded_km = float(stations.loaded_truck_km_departures.sum()) * annual_factor
        empty_km = float(stations.empty_truck_km_departures.sum()) * annual_factor
    spare = assets - onboard
    battery_recovery = annuity(p["discount_rate"], p["battery_calendar_life_yr"])
    station_recovery = annuity(p["discount_rate"], p["station_life_yr"])
    pack_annual = assumptions["inventory"]["battery_unit_capacity_kwh"] * battery_price * battery_recovery
    facility_capex = power_kw * station_price
    equipment_capex = berths * p["swap_equipment_cny"]
    return dict(
        grid_nameplate_kw=power_kw, swap_berths=berths,
        initial_station_packs=opening, initial_offsite_packs=assets-opening,
        peak_simultaneous_onboard_packs=onboard, other_packs_at_onboard_peak=spare,
        battery_assets_packs=assets, modeled_haulage_fleet=0 if scenario is None else scenario.rule_fleet,
        annual_onboard_battery_cny=onboard * pack_annual,
        annual_other_battery_cny=spare * pack_annual,
        annual_station_facility_cny=facility_capex * station_recovery,
        annual_swap_equipment_cny=equipment_capex * station_recovery,
        annual_station_om_cny=(facility_capex + equipment_capex) * p["om_rate_per_yr"],
        annual_loaded_truck_km=loaded_km, annual_empty_truck_km=empty_km,
        annual_loaded_transport_cny=loaded_km * p["transfer_cny_per_pack_km"],
        annual_empty_transport_cny=empty_km * p["transfer_cny_per_pack_km"],
    )


def cost_table(fleet, scenarios):
    assumptions = params()
    p = assumptions["lcc"]
    priced = [s for s in scenarios if not s.name.endswith("diagnostic")]
    diesel = operating_costs(fleet, None, assumptions)
    diesel_annual = diesel["annual_diesel_cny"]
    annual_factor = 365.0 / len(fleet.days)
    rows = []
    for case, battery_key, station_key in (
        ("central", "battery_cny_per_kwh", "station_cny_per_kw"),
        ("low", "battery_cny_per_kwh_low", "station_cny_per_kw_low"),
        ("high", "battery_cny_per_kwh_high", "station_cny_per_kw_high"),
    ):
        battery_price, station_price = p[battery_key], p[station_key]
        for scenario in [None, *priced]:
            opex = operating_costs(fleet, scenario, assumptions)
            assets = asset_costs(scenario, assumptions, battery_price, station_price, annual_factor)
            annual = opex["annual_diesel_cny"] + opex["annual_grid_cny"] + sum(
                assets[key] for key in (
                    "annual_onboard_battery_cny", "annual_other_battery_cny",
                    "annual_station_facility_cny", "annual_swap_equipment_cny",
                    "annual_station_om_cny", "annual_loaded_transport_cny",
                    "annual_empty_transport_cny",
                )
            )
            km = opex["annual_fleet_km"]
            annual_battery = assets["annual_onboard_battery_cny"] + assets["annual_other_battery_cny"]
            break_even_battery = None
            if annual_battery:
                other_costs = annual - annual_battery
                break_even_battery = (diesel_annual - other_costs) / (annual_battery / battery_price)
            rows.append(dict(
                prices=case, scenario="Diesel fleet" if scenario is None else scenario.name,
                battery_cny_per_kwh=battery_price, facility_cny_per_kw=station_price,
                **opex, **assets,
                annual_system_resource_cny=annual, system_cny_per_km=annual/km,
                ship_operating_cny_per_km=opex["annual_ship_operating_cny"]/km,
                diesel_minus_system_cny_per_km=(diesel_annual-annual)/km,
                break_even_battery_cny_per_kwh=break_even_battery,
            ))
    return pd.DataFrame(rows)


def main():
    fleet = fleet_model.build_case()
    scenarios = stock_model.build_cases(fleet)
    soc = fleet_model.save(fleet)
    stock = stock_model.save(scenarios)
    report = cost_table(fleet, scenarios)
    out = write_dir(RUN) / "annualized_cost_same_service.csv"
    report.to_csv(out, index=False)
    provenance.record(out, [Path(__file__), Path(annuity.__code__.co_filename), PARAMS_FILE, soc, stock],
                      script=__file__, params_used={
        "comparison": "172-ship AIS distance, 156 electric and 16 diesel in swapping cases",
        "annualization": "CRF for assets; 365/90 for observed service, energy and truck distance",
        "battery_stock": "initial station assets plus first outfit at other ports; split at peak onboard",
        "station_capital": "facility per kW and one swap device per modeled berth",
        "replenishment_energy": "include full voyage energy including last unreturned depletion",
        "fee_treatment": "ship expenditure reported separately; internal swapping payment cancels",
        "transport": "same distance rate for loaded and empty outsourced truck movements",
        "scope": "propulsion energy and modeled battery/station/transfer costs; no vessel retrofit or value of time",
    })
    cols = ["prices", "scenario", "battery_assets_packs", "annual_system_resource_cny",
            "system_cny_per_km", "ship_operating_cny_per_km", "diesel_minus_system_cny_per_km"]


if __name__ == "__main__":
    main()
