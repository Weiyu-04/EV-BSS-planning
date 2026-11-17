
from __future__ import annotations

import numpy as np
import pandas as pd

from . import loaders


def _diesel_cost(energy_kwh: pd.Series) -> pd.Series:
    """Fuel cost for legs a ship cannot electrify."""
    c = loaders.params()["cost"]
    shaft = energy_kwh * c["diesel_electric_to_mechanical_efficiency"] * c["diesel_transmission_efficiency"] / c["diesel_combustion_efficiency"]
    return shaft / c["diesel_lhv_kwh_per_litre"] * c["diesel_price_per_litre"]


def _per_km(cost_per_ship: pd.Series, distance_nm: pd.Series) -> float:
    """Unweighted mean of per-ship unit cost, converted from CNY/nm to CNY/km."""
    nm_to_km = loaders.params()["cost"]["nautical_mile_to_km"]
    return float((cost_per_ship / distance_nm).mean() / nm_to_km)


def coverage(n_stations: int, plan: str, growth: int = 0) -> tuple[int, int, float]:
    """Ships able to complete their observed voyages, out of the fleet."""
    df = loaders.siting_result(n_stations, plan, growth)
    feasible = df.groupby("mmsi")["feasible"].max()
    n_ok = int((feasible == 1).sum())
    return n_ok, len(feasible), n_ok / len(feasible)


def diesel_baseline_cost(n_stations: int = 4, growth: int = 0) -> float:
    """Cost if the whole fleet stays on diesel."""
    df = loaders.siting_result(n_stations, "plan1_non_flexible", growth)
    per_ship = df.groupby("mmsi").agg(dist=("distance", "sum"), energy=("cumulative_energy_kwh", "max"))
    return _per_km(_diesel_cost(per_ship["energy"]), per_ship["dist"])


def swapping_cost(n_stations: int, plan: str, growth: int = 0) -> float:
    """Cost for a swapping plan: electricity for feasible ships, diesel otherwise."""
    price = loaders.params()["cost"]["swapping_price"]
    df = loaders.siting_result(n_stations, plan, growth)
    per_ship = df.groupby("mmsi").agg(
        dist=("distance", "sum"), energy=("cumulative_energy_kwh", "max"), feasible=("feasible", "max")
    )
    cost = np.where(
        per_ship["feasible"] == 1,
        per_ship["energy"] * price,
        _diesel_cost(per_ship["energy"]),
    )
    return _per_km(pd.Series(cost, index=per_ship.index), per_ship["dist"])


def time_penalty_p75(n_stations: int, plan: str, growth: int = 0) -> float:
    """Relative increase in the 75th percentile of cumulative sailing time."""
    base = loaders.segments_with_energy().groupby("mmsi")["cumulative_sailing_h"].max()
    df = loaders.siting_result(n_stations, plan, growth)
    now = df.groupby("mmsi")["cumulative_sailing_h"].max()
    common = now.index.intersection(base.index)
    p75_now = np.percentile(now[common], 75)
    p75_base = np.percentile(base[common], 75)
    return float((p75_now - p75_base) / p75_base)


def inventory_by_port() -> pd.DataFrame:
    """Initial / average / maximum inventory per port, EAS versus DAS."""
    df = loaders.inventory_summary().copy()
    df["port"] = df["Port"].str.replace("_Port", "", regex=False)
    df["mode"] = df["Mode"].map({"Equal": "EAS", "Differential": "DAS"})
    return df.pivot(index="port", columns="mode",
                    values=["Initial Inventory", "Average Inventory",
                            "Max Inventory", "Std Deviation"])
