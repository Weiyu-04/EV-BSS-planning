
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401

from src.geometry import ChannelNetwork, to_local_xy  # noqa: E402
from src.loaders import RESULTS, DATA, output_dir, waypoints  # noqa: E402
from src.profile import ALPHA_UNIFORM_SECTION_MEAN  # noqa: E402

AIS_DIR = DATA / "01_raw_ais" / "yangtze_ais_labelled"

MIN_SOG_KNOTS = 3.0
MAX_DIST_FROM_CENTRELINE_M = 5_000.0
MIN_ALONG_ALIGNMENT = 0.7

REACH_BIN_KM = 10.0
MIN_POINTS_PER_DIRECTION = 300
LATERAL_BANDS = 7


def load_centreline():
    net = ChannelNetwork(waypoints())
    path = net.path(1, 93)
    return net, path, net.centreline(path, resample_m=500.0, smooth_window=9)


def project_all_ais(cl) -> pd.DataFrame:
    """Project every AIS fix onto the centreline."""
    tree = cKDTree(cl.xy)
    files = sorted(p for p in AIS_DIR.iterdir() if p.suffix in (".csv", ".xlsx"))
    if not files:
        raise FileNotFoundError(f"no AIS files under {AIS_DIR}")

    frames = []
    for i, f in enumerate(files, 1):
        d = pd.read_csv(f) if f.suffix == ".csv" else pd.read_excel(f)
        d = d.dropna(subset=["lon", "lat", "speed", "course"])
        if d.empty:
            continue
        x, y = to_local_xy(d["lon"].values, d["lat"].values, cl.lon0, cl.lat0)
        p = np.column_stack([x, y])
        dist, idx = tree.query(p)
        idx = np.clip(idx, 0, len(cl.tangents) - 1)

        t, n = cl.tangents[idx], cl.normals[idx]
        rel = p - cl.xy[idx]
        ang = np.radians(d["course"].values)
        heading = np.column_stack([np.sin(ang), np.cos(ang)])

        frames.append(pd.DataFrame(dict(
            mmsi=d["mmsi"].values,
            s_km=cl.arclength[idx] / 1000.0,
            lateral_m=np.einsum("ij,ij->i", rel, n),
            curvature=cl.curvature[idx],
            dist_m=dist,
            sog=d["speed"].values,
            along=np.einsum("ij,ij->i", heading, t),
        )))
        if i % 40 == 0:
            pass
    return pd.concat(frames, ignore_index=True)


def filter_underway(a: pd.DataFrame) -> pd.DataFrame:
    f = a[(a.sog >= MIN_SOG_KNOTS)
          & (a.dist_m < MAX_DIST_FROM_CENTRELINE_M)
          & (a.along.abs() > MIN_ALONG_ALIGNMENT)].copy()
    f["downstream"] = f.along > 0
    f["reach_km"] = (f.s_km // REACH_BIN_KM * REACH_BIN_KM).astype(int)
    return f


def longitudinal_current(f: pd.DataFrame) -> pd.DataFrame:
    """v_avg along the river: half-difference of the two directions per reach."""
    g = (f.groupby(["reach_km", "downstream"])
           .agg(sog_med=("sog", "median"), sog_p25=("sog", lambda s: s.quantile(.25)),
                sog_p75=("sog", lambda s: s.quantile(.75)), n=("sog", "size"))
           .unstack("downstream"))
    g.columns = [f"{a}_{'down' if b else 'up'}" for a, b in g.columns]
    g = g.dropna()
    g = g[(g.n_down >= MIN_POINTS_PER_DIRECTION) & (g.n_up >= MIN_POINTS_PER_DIRECTION)]
    g["v_current_kn"] = (g.sog_med_down - g.sog_med_up) / 2.0
    g["v_still_kn"] = (g.sog_med_down + g.sog_med_up) / 2.0
    return g.reset_index()


def lateral_profile(f: pd.DataFrame, width_by_reach: pd.Series) -> pd.DataFrame:
    """v(r) within each reach, plus the same-sign control that identifies it."""
    f = f.copy()
    w = f.reach_km.map(width_by_reach)
    f = f[w.notna()]
    f["r"] = np.clip(2.0 * f.lateral_m / w[w.notna()], -1.0, 1.0)
    edges = np.linspace(-1, 1, LATERAL_BANDS + 1)
    f["band"] = pd.cut(f.r, edges, labels=(edges[:-1] + edges[1:]) / 2)

    g = (f.groupby(["band", "downstream"], observed=True)
           .agg(sog_med=("sog", "median"), n=("sog", "size"))
           .unstack("downstream"))
    g.columns = [f"{a}_{'down' if b else 'up'}" for a, b in g.columns]
    g = g.dropna()
    g = g[(g.n_down >= MIN_POINTS_PER_DIRECTION) & (g.n_up >= MIN_POINTS_PER_DIRECTION)]
    g["current_kn"] = (g.sog_med_down - g.sog_med_up) / 2.0
    g["common_kn"] = (g.sog_med_down + g.sog_med_up) / 2.0
    g["common_kn"] -= g["common_kn"].mean()
    return g.reset_index().rename(columns={"band": "r_mid"})


def fit_alpha(prof: pd.DataFrame) -> dict:
    """Test whether the lateral current follows the parabola of equation (1)."""
    r = prof.r_mid.astype(float).values
    v = prof.current_kn.values
    shape = 1.0 - r ** 2

    X = np.column_stack([np.ones_like(shape), shape])
    coef, *_ = np.linalg.lstsq(X, v, rcond=None)
    C, A = float(coef[0]), float(coef[1])
    ss_res = float(np.sum((v - X @ coef) ** 2))
    ss_const = float(np.sum((v - v.mean()) ** 2))
    return dict(intercept_kn=C, parabola_amplitude_kn=A,
                r2_vs_constant=1 - ss_res / ss_const if ss_const > 0 else np.nan,
                shape_supported=bool(A > 0 and ss_res < ss_const),
                note="alpha is analytic (3/2); this tests shape only")


def keep_side_convention(f: pd.DataFrame) -> pd.DataFrame:
    """Evidence that the two directions do not share lateral positions."""
    side = np.sign(f.lateral_m)
    return (pd.crosstab(f.downstream, side, normalize="index")
              .rename(columns={-1.0: "right_of_axis", 1.0: "left_of_axis"},
                      index={True: "downstream", False: "upstream"}))


def channel_width(f: pd.DataFrame) -> pd.DataFrame:
    """Navigable width per reach, from the p1-p99 lateral spread of traffic."""
    g = f.groupby("reach_km").lateral_m.agg(
        p1=lambda s: s.quantile(.01), p99=lambda s: s.quantile(.99), n="size")
    g["width_m"] = g.p99 - g.p1
    return g[g.n >= 500].reset_index()


def compare_with_published(long: pd.DataFrame, cl, net, path) -> pd.DataFrame:
    seg = pd.read_excel(DATA / "04_paper_baseline" / "segments_legs_energy_and_time.xlsx")
    published = seg.groupby(["start", "end"]).avg_wspeed.first().abs()
    per_segment = published.groupby(level=[0, 1]).first().reset_index()
    return pd.DataFrame(dict(
        published_abs_kn=[per_segment.avg_wspeed.median(),
                          per_segment.avg_wspeed.min(),
                          per_segment.avg_wspeed.max()],
        ais_derived_kn=[long.v_current_kn.median(),
                        long.v_current_kn.min(),
                        long.v_current_kn.max()],
    ), index=["median", "min", "max"])


def main(run: str = "flowfield_rebuild") -> None:
    out = output_dir(run)
    net, path, cl = load_centreline()

    a = project_all_ais(cl)

    f = filter_underway(a)

    width = channel_width(f)
    long = longitudinal_current(f)
    prof = lateral_profile(f, width.set_index("reach_km").width_m)
    fit = fit_alpha(prof)
    sides = keep_side_convention(f)
    cmp = compare_with_published(long, cl, net, path)


    span_cur = prof.current_kn.max() - prof.current_kn.min()
    span_com = prof.common_kn.max() - prof.common_kn.min()


    long.to_csv(out / "current_by_reach.csv", index=False)
    width.to_csv(out / "channel_width_by_reach.csv", index=False)
    prof.to_csv(out / "lateral_profile.csv", index=False)
    cmp.to_csv(out / "published_vs_ais.csv")
    sides.to_csv(out / "keep_to_starboard.csv")
    pd.Series(fit).to_csv(out / "profile_fit.csv", header=False)


if __name__ == "__main__":
    main()
