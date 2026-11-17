
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401,E402

from build_inventory import operations_record  # noqa: E402
from build_lcc import annuity  # noqa: E402
from build_station_cost_sensitivity import ENERGY, energy_cost  # noqa: E402
from resolve_siting import FlatFleet, Router, TOPOLOGY, sweep  # noqa: E402
from src.loaders import params, ship_parameters, write_dir  # noqa: E402

RUN = "station_count_extended"
COUNTS = range(4, 13)


def main():
    p = params()
    cost, lcc = p["cost"], p["lcc"]
    eta_c = lcc["energy_transfer_efficiency"]
    topo = pd.read_csv(TOPOLOGY)
    g = nx.Graph()
    for _, r in topo.iterrows():
        g.add_edge(r.start, r.end, distance=float(r.distance))
    router = Router(g)
    d = pd.read_csv(ENERGY / "segment_energy.csv")
    ships = ship_parameters().set_index("mmsi")
    legs, battery = [], []
    for mmsi, grp in d.groupby("mmsi", sort=True):
        legs.append(grp[["start", "end", "distance", "energy_new"]].to_dict("records"))
        battery.append(float(ships.loc[mmsi, "rated_capacity_kwh"]))
    battery = np.asarray(battery)
    derating = float((ships["depth_of_discharge"] * ships["utilisation_efficiency"]).iloc[0])
    days = len(pd.date_range(*p["case"]["ais_window"]))
    capex = lcc["station_power_kw"] * lcc["station_cny_per_kw"] + lcc["swap_equipment_cny"]
    annual_station = capex * (annuity(lcc["discount_rate"], lcc["station_life_yr"]) + lcc["om_rate_per_yr"])

    rows = []
    for growth in p["case"]["growth_rates"]:
        bat = battery * (1 + growth / 100)
        flat = FlatFleet(legs, "energy_new")
        for n in COUNTS:
            best = [r for r in sweep(legs, "energy_new", bat, router, flat, n, f"g{growth}")
                    if r["plan"] == "plan2"][0]
            ports = (p["case"]["plan2_stations_8"] if (growth == 0 and n == 8)
                     else best["stations"].split("|"))
            ops = operations_record(ports, source=ENERGY, growth_pct=growth)
            power, fuel, diesel, km = energy_cost(ops, derating, cost, eta_c)
            station = n * annual_station / (km * 365 / days)
            grid_fuel = fuel + power * lcc["electricity_cny_per_kwh"]
            rows.append(dict(growth=growth, stations=n,
                             covered=int(ops.loc[ops["feasible"].eq(1), "mmsi"].nunique()),
                             maximum_coverage_ships=best["covered"], plan2_ports="|".join(ports),
                             diesel=diesel, plan2_grid_and_diesel=grid_fuel,
                             reference_station=station, energy_and_station=grid_fuel + station,
                             plan2_shipowner=fuel + power * cost["swapping_price"]))
    t = pd.DataFrame(rows)
    out = write_dir(RUN) / "energy_and_station_cost_4_to_12.csv"
    t.to_csv(out, index=False)


if __name__ == "__main__":
    main()
