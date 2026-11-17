
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401

from build_inventory import COEF_RUN, operations_record  # noqa: E402
from src import loaders, metrics  # noqa: E402
from src.loaders import output_dir, params  # noqa: E402
from src import provenance  # noqa: E402

def price_mixed(rec: pd.DataFrame, c: dict, elec_factor: float,
                diesel_factor: float, eta_c: float = 1.0,
                weighted: bool = False, lhv: float | None = None) -> float:
    """CNY/km under the charging plan's per-leg price rule."""
    swap, charge = c["swapping_price"], c["charging_price"]
    nm_to_km = c["nautical_mile_to_km"]
    lhv = c["diesel_lhv_kwh_per_litre"] if lhv is None else lhv
    unit, dist = [], []
    for _, g in rec.groupby("mmsi"):
        total, pending = 0.0, 0.0
        for _, r in g.iterrows():
            if r.feasible == 1:
                pending += r.energy
                if r.swapped == 1:
                    metered = pending * elec_factor / eta_c
                    total += metered * (swap if r.cargo_port == 1 else charge)
                    pending = 0.0
            else:
                if pending > 0:
                    total += pending * elec_factor / eta_c * charge
                    pending = 0.0
                shaft = r.energy * diesel_factor / c["diesel_combustion_efficiency"]
                total += shaft / lhv * c["diesel_price_per_litre"]
        if pending > 0:
            total += pending * elec_factor / eta_c * charge
        unit.append(total)
        dist.append(g.distance.sum())
    u, d = pd.Series(unit), pd.Series(dist)
    return float((u.sum() / d.sum() if weighted else (u / d).mean()) / nm_to_km)
