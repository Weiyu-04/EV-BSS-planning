
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401
from build_inventory import operations_record
from build_lcc import annuity
from build_plan3 import price_mixed
from src import provenance
from src.loaders import DATA, PARAMS_FILE, RESULTS, params, ship_parameters, write_dir

RUN = "service_cost"
ENERGY = RESULTS / "paper/s3_energy"
COVERAGE = RESULTS / "paper/s4_siting/coverage.csv"


def energy_cost(ops, derating, cost, eta_c):
    useful = ops["energy_kwh"] * derating
    km = float(ops.distance.sum()) * cost["nautical_mile_to_km"]
    served = ops["feasible"].eq(1)
    fuel_rate = (cost["diesel_price_per_litre"] /
                 (cost["diesel_combustion_efficiency"] * cost["diesel_lhv_kwh_per_litre"]))
    return (float(useful[served].sum()) / eta_c / km,
            float(useful[~served].sum()) * fuel_rate / km,
            float(useful.sum()) * fuel_rate / km, km)


def mixed_energy_per_km(ops, derating, cost, eta_c, diesel_cny_per_km):
    """Separate swap and charge purchases using the replay's stop-level flag."""
    record = ops.rename(columns={"energy_kwh": "energy", "feasible": "feasible", "optimised swapping": "swapped"})
    swap_only = dict(cost, swapping_price=1.0, charging_price=0.0)
    charge_only = dict(cost, swapping_price=0.0, charging_price=1.0)
    swap = price_mixed(record, swap_only, derating, derating, eta_c, weighted=True)
    charge = price_mixed(record, charge_only, derating, derating, eta_c, weighted=True)
    return swap - diesel_cny_per_km, charge - diesel_cny_per_km


def build_table(cost_reference: Path):
    provenance.require(cost_reference)
    central = pd.read_csv(cost_reference)
    p = params()
    cost, lcc = p["cost"], p["lcc"]
    eta_c = lcc["energy_transfer_efficiency"]
    coverage = pd.read_csv(COVERAGE)
    ship = ship_parameters()
    factors = ship["depth_of_discharge"] * ship["utilisation_efficiency"]
    assert factors.nunique() == 1
    derating = float(factors.iloc[0])
    days = len(pd.date_range(*p["case"]["ais_window"]))
    capex = lcc["station_power_kw"] * lcc["station_cny_per_kw"] + lcc["swap_equipment_cny"]
    annual_station = capex * (annuity(lcc["discount_rate"], lcc["station_life_yr"])
                              + lcc["om_rate_per_yr"])
    rows = []
    for growth in p["case"]["growth_sweep"]:
        for n in p["case"]["station_counts"]:
            if n != 8 and growth not in p["case"]["growth_rates"]:
                continue
            selected = coverage[(coverage.growth == growth) & (coverage.n_stations == n)].set_index("plan")
            fixed_case = growth == 0 and n == 8
            ports = (p["case"]["plan2_stations_8"] if fixed_case
                     else selected.loc["plan2", "stations"].split("|"))
            ops = operations_record(ports, source=ENERGY, growth_pct=growth)
            power, fuel, diesel, km = energy_cost(ops, derating, cost, eta_c)
            n2 = ops.loc[ops["feasible"].eq(1), "mmsi"].nunique()
            station = n * annual_station / (km * 365 / days)
            grid_fuel = fuel + power * lcc["electricity_cny_per_kwh"]
            swap, charge = mixed_energy_per_km(ops, derating, cost, eta_c, fuel)
            assert np.isclose(swap + charge, power)
            row = dict(
                growth=growth, stations=n, covered=n2,
                maximum_coverage_ships=int(selected.loc["plan2", "covered"]),
                plan2_ports="|".join(ports), annual_fleet_km=km * 365 / days, diesel=diesel,
                diesel_component=fuel, flexible_grid_kwh_per_km=power,
                mixed_swap_kwh_per_km=swap, mixed_direct_kwh_per_km=charge,
                plan2=fuel + power * cost["swapping_price"],
                plan3=fuel + swap * cost["swapping_price"] + charge * cost["charging_price"],
                plan2_grid_and_diesel=grid_fuel,
                reference_station=station, energy_and_station=grid_fuel + station,
            )
            if growth == 0:
                ports1 = selected.loc["plan1", "stations"].split("|")
                ops1 = operations_record(ports1, source=ENERGY, flexible=False)
                power1, fuel1, diesel1, km1 = energy_cost(ops1, derating, cost, eta_c)
                assert np.isclose(km, km1) and np.isclose(diesel, diesel1)
                row.update(plan1=fuel1 + power1 * cost["swapping_price"],
                           plan1_energy_and_station=fuel1 + power1 * lcc["electricity_cny_per_kwh"] + station)
            rows.append(row)
    report = pd.DataFrame(rows)
    anchor = central[(central.prices == "central") & (central.scenario == "EASM no transfer")].iloc[0]
    fixed = report[(report.growth == 0) & (report.stations == 8)].iloc[0]
    assert fixed.covered == 156 and fixed.maximum_coverage_ships == 157
    assert np.isclose(fixed.plan2, anchor.ship_operating_cny_per_km)
    assert np.isclose(fixed.annual_fleet_km, anchor.annual_fleet_km)
    return report


def main():
    out = write_dir(RUN) / "station_count_cost_sensitivity.csv"
    cost_reference = out.with_name("annualized_cost_same_service.csv")
    table = build_table(cost_reference)
    table.to_csv(out, index=False)
    provenance.record(out, [
        Path(__file__), Path(operations_record.__code__.co_filename),
        Path(operations_record.__code__.co_filename).with_name("resolve_siting.py"),
        Path(price_mixed.__code__.co_filename), Path(annuity.__code__.co_filename),
        COVERAGE, ENERGY / "segment_energy.csv", PARAMS_FILE,
        DATA / "03_ship_parameters/172_ship_parameters.xlsx", DATA / "02_channel_topology/port_topology.csv",
        cost_reference,
    ], script=__file__, params_used={
        "annualization": "365/90, using total modeled fleet km",
        "station_reference": "3 MW and one swap device per site",
        "resource_scope": "fuel, grid electricity, reference facilities and station O&M",
        "shipowner_scope": "fuel and metered-equivalent purchases at swap or direct charge prices",
        "mixed_pricing": "original calls swap; added part-way calls and terminal residual charge",
    })
    sweep = table[table.growth.isin(params()["case"]["growth_rates"])]


if __name__ == "__main__":
    main()
