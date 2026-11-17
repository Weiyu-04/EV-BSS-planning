
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401,E402

from src import provenance  # noqa: E402
from src.geometry import ChannelNetwork  # noqa: E402
from src.loaders import DATA, ROOT, output_dir, params, waypoints  # noqa: E402
from src.profile import ALPHA_UNIFORM_SECTION_MEAN  # noqa: E402

RUN = "paper/s2_flowfield"
PIPE = Path(__file__).resolve().parents[1] / "1_pipeline"


def _load(name: str):
    """`1_pipeline` starts with a digit, so it cannot be imported by name."""
    spec = importlib.util.spec_from_file_location(name, PIPE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_channel_geometry(out: Path) -> pd.DataFrame:
    """Per-waypoint geometry: where each point sits, and how the axis bends."""
    wp = pd.read_csv(out / "waypoint_velocities.csv")
    net = ChannelNetwork(waypoints())
    label = {i: (lb if isinstance(lb, str) and lb.strip() else "")
             for i, lb in net.label.items()}
    g = wp[["index", "section", "local_axis", "chainage_km", "lateral_m", "r",
            "curvature", "width_m"]].copy()
    g.insert(1, "label", g["index"].map(label).fillna(""))
    g = g.rename(columns={"curvature": "curvature_1_per_m"})
    g["radius_km"] = np.where(g.curvature_1_per_m.abs() > 1e-9,
                              1.0 / g.curvature_1_per_m.abs() / 1000.0, np.nan)
    g["bank_side"] = np.select([g.r > 0, g.r < 0], ["left", "right"], "axis")
    g.to_csv(out / "channel_geometry.csv", index=False)
    return g


def build_profile_params(out: Path, geom: pd.DataFrame) -> pd.DataFrame:
    p = params()["flow_field"]
    cur = pd.read_csv(out / "current_by_reach.csv")
    k = geom.curvature_1_per_m.abs()
    wide = geom.width_m

    rows = [
        dict(quantity="section-average current",
             symbol="v_avg", value=f"{p['v_avg_anchor_knots']:.3f}",
             unit="knots (at the anchor reach)",
             provenance="verified (external gauge)",
             source=("Datong gauge, Zhu et al. (2021): flood 1.24 m/s, dry 0.73 m/s, "
                     f"weighted {p['ais_window_flood_days']}/{p['ais_window_dry_days']} days "
                     "of the observation window"),
             paper_location="Eq (1), section 3.2.2",
             caveat=("Magnitude from one gauge; the along-corridor shape comes from the AIS "
                     "up/down SOG half-difference. Zero over 41 km of the tidal reach "
                     "(600-640 km), where no correction is applied.")),
        dict(quantity="profile normalisation",
             symbol="alpha", value=f"{ALPHA_UNIFORM_SECTION_MEAN}",
             unit="dimensionless",
             provenance="verified (analytic)",
             source=("Not fitted. Requiring the lateral mean of the profile to equal v_avg "
                     "fixes alpha at 3/2 exactly, for any bend term"),
             paper_location="Eq (1), section 3.2.2",
             caveat=("Analytic identity, not evidence about the river. The parabolic shape "
                     "is what the AIS lateral bands test, and they support it only weakly "
                     "(see profile_fit.csv).")),
        dict(quantity="bend strength",
             symbol="b", value=f"{p['bend_factor_b']}",
             unit="dimensionless",
             provenance="assumed(stated)",
             source=f"Not calibrated. Swept over {p['bend_factor_sweep']}",
             paper_location="Eq (2), section 3.2.2",
             caveat=("Not calibrated. Reported because the sweep shows the result barely "
                     "moves: max |delta| per port pair 0.069 kn at b = 0.3 versus b = 0. "
                     "No further sensitivity study is added.")),
        dict(quantity="signed curvature",
             symbol="k", value=f"median {k.median():.3e}, p90 {k.quantile(.9):.3e}",
             unit="1/m",
             provenance="verified (computed)",
             source=("Finite differences on the resampled, smoothed centreline "
                     f"(resample {p['centreline_resample_m']} m, "
                     f"smoothing window {p['centreline_smooth_window']})"),
             paper_location="Eq (2)-(4), section 3.2.3",
             caveat=("Signed, not absolute: the bend term needs to know which bank is the "
                     "outer one. Waypoints were hand-placed, so curvature is "
                     "taken after smoothing and the window is reported.")),
        dict(quantity="channel width",
             symbol="w", value=f"median {wide.median():.0f}",
             unit="m",
             provenance="verified (from AIS)",
             source=f"{p['width_source']}: p1-p99 lateral spread of AIS positions per reach",
             paper_location="Eq (3), section 3.2.3",
             caveat=("This is a navigable-band width, not a hydraulic channel width. "
                     "It bounds where ships actually go, which is what eq (3) needs, "
                     "but it must not be quoted as river width.")),
    ]
    t = pd.DataFrame(rows)
    t.to_csv(out / "velocity_profile_params.csv", index=False)
    return t


def build_hydrology_cross_check(out: Path) -> pd.DataFrame:
    from src.loaders import DATA
    HYD = DATA / "05_hydrology"
    if not (HYD / "daily_discharge_5_gauges_2024.xlsx").exists():
        return pd.DataFrame()

    fp = params()["flow_field"]
    win = ("2024-09-01", "2024-11-29")

    q = pd.read_excel(HYD / "daily_discharge_5_gauges_2024.xlsx")
    q.columns = ["station", "date", "time", "value"]
    q["kind"] = "discharge_m3s"
    lv = pd.read_excel(HYD / "daily_stage_6_gauges_2024.xlsx")
    lv.columns = ["station", "date", "time", "value"]
    lv["kind"] = "stage_m"
    g = pd.concat([q, lv], ignore_index=True)
    g["date"] = pd.to_datetime(g["date"], format="mixed")
    g["value"] = pd.to_numeric(g["value"], errors="coerce")
    g = g[(g.date >= win[0]) & (g.date <= win[1])].dropna(subset=["value"])

    rows = []
    for (kind, st), d in g.groupby(["kind", "station"]):
        m = {int(mm): float(dd.value.mean()) for mm, dd in d.groupby(d.date.dt.month)}
        if not {9, 10, 11} <= set(m):
            continue
        rows.append(dict(kind=kind, station=st, days=len(d),
                         sep=m[9], oct=m[10], nov=m[11],
                         sep_over_oct=m[9] / m[10], nov_over_oct=m[11] / m[10]))
    d = pd.DataFrame(rows).sort_values(["kind", "station"], kind="stable")

    EXP = 0.45
    dis = d[d.kind == "discharge_m3s"]
    sep_r = float(dis.sep_over_oct.median()) ** EXP
    nov_r = float(dis.nov_over_oct.median()) ** EXP

    flood = set(fp["flood_months"])
    clim = {m: (1.0 if m in flood else float(fp["datong_dry_season_mps"])
                / float(fp["datong_flood_season_mps"])) for m in (9, 10, 11)}

    days = {9: 30, 10: 31, 11: 29}
    obs = {9: sep_r, 10: 1.0, 11: nov_r}
    obs_mean = sum(days[m] * obs[m] for m in days) / sum(days.values())
    clim_mean = sum(days[m] * clim[m] for m in days) / sum(days.values())

    shape = pd.DataFrame([
        dict(month=m,
             climatology_relative=clim[m] / clim_mean,
             observed_relative=obs[m] / obs_mean,
             ratio=(obs[m] / obs_mean) / (clim[m] / clim_mean))
        for m in (9, 10, 11)])
    worst = float((shape.observed_relative - 1).abs().max())
    band = float(params()["robustness"]["sigma_current_fleet"])

    shape["max_abs_deviation_from_window_mean"] = worst
    shape["reported_systematic_band"] = band
    shape["inside_reported_band"] = worst < band

    ORDER = ["Hankou", "Jiujiang", "Hukou", "Zhenjiang 2", "Yanglin", "Wusongkou"]
    IN_REACH = {"Zhenjiang 2", "Yanglin", "Wusongkou"}
    st = d[d.kind == "stage_m"].set_index("station")
    tt = []
    for name in ORDER:
        if name not in st.index:
            continue
        r = st.loc[name]
        tt.append(dict(station=name, in_modelled_reach=name in IN_REACH,
                       sep_mean_m=r.sep, oct_mean_m=r["oct"], nov_mean_m=r.nov,
                       seasonal_range_m=r.sep - r.nov))
    trans = pd.DataFrame(tt)
    if len(trans):
        base = float(trans[trans.station == "Zhenjiang 2"].seasonal_range_m.iloc[0])
        trans["decay_vs_zhenjiang"] = 1 - trans.seasonal_range_m / base
        trans.to_csv(out / "hydrology_tidal_transition.csv", index=False)

    d.to_csv(out / "hydrology_gauges_monthly.csv", index=False)
    shape.to_csv(out / "hydrology_seasonal_shape.csv", index=False)

    return shape


def main() -> None:
    out = output_dir(RUN)

    _load("calibrate_flow_field").main(RUN)

    _load("rebuild_flow_velocities").main(RUN)

    geom = build_channel_geometry(out)

    t = build_profile_params(out, geom)
    for _, r in t.iterrows():
        pass
    weak = t[t.provenance.str.startswith("assumed")]

    pp = pd.read_csv(out / "port_pair_currents.csv")
    pp.rename(columns={pp.columns[0]: "start", pp.columns[1]: "end"}
              ).to_csv(out / "leg_currents.csv", index=False)

    shape = build_hydrology_cross_check(out)
    if len(shape):
        for _, r in shape.iterrows():
            pass
        w = float(shape.max_abs_deviation_from_window_mean.iloc[0])
        b = float(shape.reported_systematic_band.iloc[0])
        tt = out / "hydrology_tidal_transition.csv"
        if tt.exists():
            t = pd.read_csv(tt)
            for _, r in t.iterrows():
                tag = " <-inside the study reach" if r.in_modelled_reach else ""
    else:
        pass

    inputs = [DATA / "02_channel_topology" / "nodes_and_waypoints.xlsx",
              DATA / "02_channel_topology" / "nodes_and_waypoints_current.xlsx",
              DATA / "02_channel_topology" / "port_topology.csv",
              DATA / "04_paper_baseline" / "segments_legs_energy_and_time.xlsx"]
    inputs = [p for p in inputs if p.exists()]
    stamped = 0
    for prod in sorted(out.glob("*")):
        if prod.is_file() and not prod.name.endswith(provenance.SUFFIX):
            provenance.record(prod, script=__file__, inputs=inputs,
                              params_used={"flow_field": params()["flow_field"]})
            stamped += 1


if __name__ == "__main__":
    main()
