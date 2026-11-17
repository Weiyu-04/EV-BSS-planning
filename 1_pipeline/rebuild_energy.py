
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401

from src.loaders import DATA, output_dir, ship_parameters  # noqa: E402

SEGMENT_FILE = DATA / "04_paper_baseline" / "segments_legs_energy_and_time.xlsx"


def energy_curve_coefficients(ships: pd.DataFrame) -> pd.DataFrame:
    """Parse the per-ship cubic power curve stored as a bracketed string."""
    coef = ships["energy_curve"].str.strip("[] ").str.split(expand=True).astype(float)
    coef.columns = ["a", "b", "c", "d"]
    return coef.set_index(ships["mmsi"].values)


def propulsion_power(v_knots: np.ndarray, coef: pd.DataFrame) -> np.ndarray:
    v = np.asarray(v_knots, float)
    return (coef.a.values * v ** 3 + coef.b.values * v ** 2
            + coef.c.values * v + coef.d.values)


def energy(v_water: np.ndarray, hours: np.ndarray, aux_kw: np.ndarray,
           zeta: np.ndarray, eta: np.ndarray, coef: pd.DataFrame) -> np.ndarray:
    return (propulsion_power(v_water, coef) + aux_kw) / (zeta * eta) * hours


def speed_through_water(sog: np.ndarray, v_w: np.ndarray) -> np.ndarray:
    """Sign of v_w in Constraint (15): positive downstream."""
    return np.asarray(sog, float) - np.asarray(v_w, float)


def main(run: str = "flowfield_rebuild",
         current_from: str | None = None) -> None:
    """`run` is the stage written to; `current_from` is the stage whose current field is read."""
    out = output_dir(run)
    src = output_dir(current_from) if current_from else out
    seg = pd.read_excel(SEGMENT_FILE)
    ships = ship_parameters()

    per_ship = ships.set_index("mmsi")
    coef = energy_curve_coefficients(ships).loc[seg.mmsi]
    aux = per_ship["auxiliary_power_kw"].loc[seg.mmsi].values
    zeta = per_ship["depth_of_discharge"].loc[seg.mmsi].values
    eta = per_ship["utilisation_efficiency"].loc[seg.mmsi].values

    sog = seg.real_speed.values + seg.avg_wspeed.values
    hours = seg.distance.values / sog
    e_ctrl = energy(seg.real_speed.values, hours, aux, zeta, eta, coef)
    rel = np.abs(e_ctrl - seg["energy_kwh"].values) / np.maximum(seg["energy_kwh"].values, 1e-9)
    if np.median(rel) > 5e-3:
        raise RuntimeError("cannot reproduce the published energy chain")

    reb = pd.read_csv(src / "port_pair_currents.csv")
    reb.columns = ["start", "end"] + list(reb.columns[2:])
    lookup = reb.set_index(["start", "end"]).rebuilt
    w_new = pd.MultiIndex.from_arrays([seg.start, seg.end]).map(lookup)
    if pd.isna(w_new).any():
        raise RuntimeError(f"{int(pd.isna(w_new).sum())} segments have no rebuilt current")

    _check = speed_through_water(sog, seg.avg_wspeed.values)
    assert np.allclose(_check, seg.real_speed.values, rtol=0, atol=1e-9), (
        "the sign of v_w in Constraint (15) is inverted: applying it to the "
        "published current does not return the published speed through water")
    v_water_new = speed_through_water(sog, np.asarray(w_new, float))

    lo = per_ship["speed_min_kn"].loc[seg.mmsi].values
    hi = per_ship["speed_max_kn"].loc[seg.mmsi].values
    below, above = (v_water_new < lo).sum(), (v_water_new > hi).sum()

    e_new = energy(v_water_new, hours, aux, zeta, eta, coef)

    d = seg[["mmsi", "start", "end", "distance", "real_speed", "avg_wspeed",
             "sailing_h", "energy_kwh", "rated_capacity_kwh"]].copy()
    d["w_new"] = np.asarray(w_new, float)
    d["v_water_new"] = v_water_new
    d["energy_new"] = e_new
    d["ratio"] = e_new / seg["energy_kwh"].values

    up, down = seg.avg_wspeed < 0, seg.avg_wspeed > 0

    o, n = seg["energy_kwh"].sum(), e_new.sum()

    per_ship_e = d.groupby("mmsi").agg(old=("energy_kwh", "sum"), new=("energy_new", "sum"))
    per_ship_e["pct"] = (per_ship_e.new / per_ship_e.old - 1) * 100
    for thr in (1, 5, 10):
        pass

    d.to_csv(out / "segment_energy_rebuilt.csv", index_label="seg_id")
    per_ship_e.to_csv(out / "ship_energy_delta.csv")


if __name__ == "__main__":
    main()
