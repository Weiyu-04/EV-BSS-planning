
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401,E402

from src import provenance  # noqa: E402
from src.loaders import DATA, ROOT, output_dir, params, ship_parameters  # noqa: E402

RUN = "paper/s3_energy"
UPSTREAM = "paper/s2_flowfield"
CODE = Path(__file__).resolve().parents[1]
KN_TO_KMH = 1.852

REPORT = dict(disp_t=12_600.0, v_kn=8.0, battery_kwh=57_600.0,
              endurance_h=44.0, range_km=652.0)

CANDIDATES = [
    (356.2, "700TEU report, back-solved from its own 389 kW x 2", True),
    (366.0, "Zhou et al. (2019) Yangtze in-service regression", False),
    (387.0, "20-point calibration at 75% MCR, report excluded", False),
]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, CODE / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def endurance_km(c_kw: float, zeta: float = 0.8, eta: float = 0.9,
                 aux_kw: float = 150.0) -> float:
    pe = REPORT["disp_t"] ** (2 / 3) * REPORT["v_kn"] ** 3 / c_kw
    hours = REPORT["battery_kwh"] * zeta * eta / (pe + aux_kw)
    return hours * REPORT["v_kn"] * KN_TO_KMH


def build_leg_energy(out: Path) -> pd.DataFrame:
    seg = pd.read_csv(out / "segment_energy.csv")
    ships = ship_parameters().set_index("mmsi")
    zeta = ships["depth_of_discharge"].loc[seg.mmsi].values
    eta = ships["utilisation_efficiency"].loc[seg.mmsi].values
    rated = ships["rated_capacity_kwh"].loc[seg.mmsi].values

    d = seg[["mmsi", "start", "end", "distance"]].copy()
    d["zeta"] = zeta
    d["eta_utilisation"] = eta
    d["energy_actual_kwh"] = seg.energy_new.values * zeta * eta
    d["capacity_usable_kwh"] = rated * zeta * eta
    d["capacity_rated_kwh"] = rated
    d["feasible"] = d.energy_actual_kwh <= d.capacity_usable_kwh
    d["published_e_kwh"] = seg.energy_new.values
    assert bool((d.feasible == (d.published_e_kwh <= d.capacity_rated_kwh)).all()), \
        "Constraints (14)-(15) and the rated-capacity form disagree on feasibility"
    d.to_csv(out / "leg_energy.csv", index=False)
    return d


def build_validation(out: Path) -> tuple[pd.DataFrame, dict]:
    rows = []
    for c, source, circular in CANDIDATES:
        km = endurance_km(c)
        rows.append(dict(coefficient_c_kw=c, source=source,
                         predicted_range_km=round(km, 1),
                         reported_range_km=REPORT["range_km"],
                         relative_error=round(km / REPORT["range_km"] - 1, 6),
                         circular=circular,
                         counts_towards_band=not circular))
    t = pd.DataFrame(rows)
    t.to_csv(out / "validation_vessels.csv", index=False)

    indep = t[~t.circular]
    band = dict(
        source="s3_energy/run.py",
        basis=("End-to-end closure of the 700TEU design study's stated range "
               "(~652 km) using coefficients derived independently of it."),
        n_independent_points=int(len(indep)),
        error_min=float(indep.relative_error.min()),
        error_max=float(indep.relative_error.max()),
        headline=float(indep.relative_error.abs().min()),
        reported_as="+3.8%",
        reported_decision="Computed 3.8491%, reported as +3.8% (one-decimal rounding).",
        excluded_circular={"coefficient_c_kw": 356.2,
                           "why": ("back-solved from the same report's propulsion "
                                   "figure; closing that report with it is circular")},
        excluded_range=("+1.5% to +9%, whose lower bound is the excluded circular row"),
        consumed_by="s7_robustness",
    )
    (out / "energy_error_band.yaml").write_text(
        yaml.safe_dump(band, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return t, band


def main() -> None:
    out = output_dir(RUN)

    _load("1_pipeline/calibrate_admiralty_pership.py",
          "calibrate_admiralty_pership").main(RUN)
    cal = pd.read_csv(out / "admiralty_pership.csv")
    cal.to_csv(out / "admiralty_calibration.csv", index=False)

    _load("1_pipeline/rebuild_energy.py", "rebuild_energy").main(RUN, UPSTREAM)

    _load("1_pipeline/build_operating_record.py", "build_operating_record").main(RUN, RUN, RUN)

    d = build_leg_energy(out)

    t, band = build_validation(out)
    for _, r in t.iterrows():
        mark = "  cycle, not counted" if r.circular else ""

    inputs = [output_dir(UPSTREAM) / "port_pair_currents.csv",
              DATA / "03_ship_parameters" / "172_ship_parameters.xlsx",
              DATA / "04_paper_baseline" / "segments_legs_energy_and_time.xlsx"]
    inputs = [p for p in inputs if p.exists()]
    stamped = 0
    for prod in sorted(out.glob("*")):
        if prod.is_file() and not prod.name.endswith(provenance.SUFFIX):
            provenance.record(prod, script=__file__, inputs=inputs,
                              params_used={"energy": params()["energy"]})
            stamped += 1


if __name__ == "__main__":
    main()
