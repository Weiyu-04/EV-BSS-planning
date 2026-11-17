
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401

from src.loaders import DATA, RESULTS, output_dir, params, ship_parameters  # noqa: E402
from src import provenance  # noqa: E402

FIELD_RUN = RESULTS / "flowfield_rebuild"


def energy_curve(ships: pd.DataFrame) -> pd.DataFrame:
    c = ships["energy_curve"].str.strip("[] ").str.split(expand=True).astype(float)
    c.columns = ["a", "b", "c", "d"]
    return c.set_index(ships["mmsi"].values)


def main(run: str = "coefficient_correction",
         energy_from: str = "flowfield_rebuild",
         calib_from: str | None = None) -> None:
    out = output_dir(run)
    src = output_dir(energy_from)
    calib = output_dir(calib_from) if calib_from else out
    p = params()
    adm = p["admiralty"]
    if not adm["apply_correction"]:
        raise SystemExit("admiralty.apply_correction is false; the correction would be a no-op")

    ships = ship_parameters()
    per_ship = ships.set_index("mmsi")
    coef = energy_curve(ships)

    c_code = ships.set_index("mmsi")["admiralty_coefficient"].astype(float)
    c_kw_now = c_code * adm["code_convention_factor"]

    ps = pd.read_csv(calib / "admiralty_pership.csv")
    c_kw_new = ps.set_index("mmsi")["C_new_kw"].reindex(c_kw_now.index)
    if c_kw_new.isna().any():
        raise RuntimeError("per-ship coefficients missing; run calibrate_admiralty_pership.py")

    target = float(adm["calibrated_c_kw"])
    fleet_scale = target / float(c_kw_now.median())

    scale_series = c_kw_new / c_kw_now
    scale = scale_series

    seg = pd.read_excel(DATA / "04_paper_baseline" / "segments_legs_energy_and_time.xlsx")
    field = pd.read_csv(src / "segment_energy_rebuilt.csv")

    aux = per_ship["auxiliary_power_kw"].loc[seg.mmsi].values
    zeta = per_ship["depth_of_discharge"].loc[seg.mmsi].values
    eta = per_ship["utilisation_efficiency"].loc[seg.mmsi].values
    cf = coef.loc[seg.mmsi]

    v_water = field.v_water_new.values
    hours = seg.distance.values / (seg.real_speed.values + seg.avg_wspeed.values)
    pe_base = (cf.a.values * v_water ** 3 + cf.b.values * v_water ** 2
             + cf.c.values * v_water + cf.d.values)
    pe_new = pe_base / scale.reindex(seg.mmsi).values

    e_leg = (pe_new + aux) / (zeta * eta) * hours

    d = seg[["mmsi", "start", "end", "distance", "real_speed", "avg_wspeed",
             "energy_kwh", "rated_capacity_kwh"]].copy()
    d["w_new"] = field.w_new.values
    d["v_water_new"] = v_water
    d["energy_field"] = field.energy_new.values
    d["energy_new"] = e_leg
    d.to_csv(out / "segment_energy.csv", index_label="seg_id")

    ze = float(zeta[0] * eta[0])
    for label, col in [("published record", seg["energy_kwh"].values),
                       ("current field only", field.energy_new.values),
                       ("current field and coefficient", e_leg)]:
        pass

    ps = d.groupby("mmsi").agg(field=("energy_field", "sum"), corrected=("energy_new", "sum"))
    ps["pct"] = (ps.corrected / ps.field - 1) * 100

    pd.Series({"scale_median": float(scale.median()), "c_kw_target": target,
               "c_kw_before": float(c_kw_now.median()),
               "c_kw_after": float(c_kw_new.median()),
               "fleet_energy_published": float(seg["energy_kwh"].sum()),
               "fleet_energy_field": float(field.energy_new.sum()),
               "fleet_energy": float(np.sum(e_leg)),
               "fleet_E_act_published": float(seg["energy_kwh"].sum() * zeta[0] * eta[0]),
               "fleet_E_act_field": float(field.energy_new.sum() * zeta[0] * eta[0]),
               "fleet_E_act": float(np.sum(e_leg) * zeta[0] * eta[0])}).rename_axis(
        "quantity").rename("value").to_csv(
        out / "correction_summary.csv")
    provenance.record(out / "correction_summary.csv", script=__file__,
                      inputs=[calib / "admiralty_pership.csv"],
                      params_used=params()["admiralty"])


if __name__ == "__main__":
    main()
