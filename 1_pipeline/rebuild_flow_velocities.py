
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401

from src.geometry import Centreline, ChannelNetwork, normalised_lateral  # noqa: E402
from src.loaders import DATA, output_dir, params, waypoints  # noqa: E402
from src.profile import (  # noqa: E402
    ProfileParams, bend_factor, segment_mean_velocity, velocity_at,
)

KNOTS_PER_MPS = 1.0 / 0.514444
WAYPOINT_FILE = DATA / "02_channel_topology" / "nodes_and_waypoints.xlsx"
FROZEN_VELOCITY_FILE = DATA / "02_channel_topology" / "nodes_and_waypoints_current.xlsx"
SEGMENT_FILE = DATA / "04_paper_baseline" / "segments_legs_energy_and_time.xlsx"


def load_sections() -> pd.DataFrame:
    """The 20 sections, each spanning a contiguous run of waypoint indices."""
    w = pd.read_excel(WAYPOINT_FILE)
    s = w.dropna(subset=["leg_id"])[
        ["leg_id", "origin_id", "destination_id", "leg_origin", "leg_destination"]].copy()
    s.columns = ["section", "first_wp", "last_wp", "from_port", "to_port"]
    return s.astype({"section": int, "first_wp": int, "last_wp": int})


def port_graph(sections: pd.DataFrame, current: pd.Series,
               known_ports: set) -> nx.Graph:
    """Ports as nodes, sections as edges carrying a directed current."""
    def canonical(name: str) -> str:
        return name if name in known_ports else (
            name + "port" if name + "port" in known_ports else name)

    g = nx.Graph()
    for _, r in sections.iterrows():
        a, b = canonical(r.from_port), canonical(r.to_port)
        if g.has_edge(a, b):
            continue
        g.add_edge(a, b, current=float(current[r.section]), downstream=(a, b))
    return g


def directed_current(g: nx.Graph, a: str, b: str) -> float | None:
    """Mean section current along the shortest path, signed by travel direction."""
    if a not in g or b not in g:
        return None
    try:
        path = nx.shortest_path(g, a, b)
    except nx.NetworkXNoPath:
        return None
    vals = []
    for u, v in zip(path[:-1], path[1:]):
        e = g[u][v]
        vals.append(e["current"] if e["downstream"] == (u, v) else -e["current"])
    return float(np.mean(vals))


def all_directed_currents(g: nx.Graph, pairs) -> pd.Series:
    return pd.Series({(a, b): directed_current(g, a, b) for a, b in pairs})


def reproduce_published(sections: pd.DataFrame, published_od: pd.Series,
                        known_ports: set) -> None:
    frozen = pd.read_excel(FROZEN_VELOCITY_FILE)
    col = next(c for c in frozen.columns if "mean_current_kn" in c)
    per_section = (frozen.dropna(subset=["leg_id"])
                   .set_index(frozen.dropna(subset=["leg_id"])["leg_id"].astype(int))[col])

    g = port_graph(sections, per_section, known_ports)
    calc = all_directed_currents(g, published_od.index)
    err = (calc - published_od).abs()
    exact = int((err < 1e-6).sum())

    off = err[err >= 1e-6]
    for (a, b), e in off.items():
        pass
    if exact < len(err) - 2:
        raise RuntimeError("aggregation rule does not reproduce the published field")


def section_average_speed(out_dir: Path, chainage_km: np.ndarray,
                          p: dict) -> tuple[np.ndarray, dict]:
    """v_avg along the corridor: gauge magnitude x AIS longitudinal shape."""
    cur = pd.read_csv(out_dir / "current_by_reach.csv").sort_values("reach_km")
    shape = cur.set_index("reach_km").v_current_kn.rolling(3, center=True,
                                                           min_periods=1).median()

    anchor_km = p["v_avg_anchor_chainage_km"]
    anchor_reach = int(anchor_km // 10 * 10)
    at_anchor = float(shape.reindex([anchor_reach]).iloc[0])
    if not np.isfinite(at_anchor) or at_anchor <= 0:
        raise RuntimeError(f"no usable AIS shape at the anchor reach {anchor_reach}")

    scaled = shape / at_anchor * p["v_avg_anchor_knots"]
    v = np.interp(chainage_km, scaled.index.values, scaled.values,
                  left=scaled.iloc[0], right=scaled.iloc[-1])

    n_neg = int((v < 0).sum())
    v = np.maximum(v, 0.0)

    zero = v <= 0.0
    if zero.any():
        idx = np.flatnonzero(zero)
        zero_from, zero_to = float(chainage_km[idx[0]]), float(chainage_km[idx[-1]])
    else:
        zero_from = zero_to = float("nan")
    info = dict(anchor_reach=anchor_reach, ais_at_anchor_kn=at_anchor,
                scale=p["v_avg_anchor_knots"] / at_anchor, floored=n_neg,
                zero_km=int(zero.sum()), zero_from_km=zero_from,
                zero_to_km=zero_to)
    return v, info


def build(bend_b: float, wp: pd.DataFrame, global_cl, width_by_km, v_avg_by_km,
          sections: pd.DataFrame, smooth_window: int, samples_per_segment: int = 1
          ) -> tuple[pd.Series, pd.DataFrame]:
    """Waypoint velocities, equations (1)-(4), then section means (Section 3.2.4)."""
    rows = []
    for _, s in sections.iterrows():
        sub = wp[(wp["index"] >= s.first_wp) & (wp["index"] <= s.last_wp)]
        if len(sub) < 3:
            local_cl = global_cl
            local = False
        else:
            sub = sub.sort_values("index")
            local_cl = Centreline(sub["lon"].values, sub["lat"].values,
                                  resample_m=200.0,
                                  smooth_window=min(smooth_window,
                                                    max(3, len(sub) // 2 * 2 + 1)))
            local = True

        pts = []
        rec = sub[["index", "lon", "lat"]].to_records(index=False)
        for i, (idx, lon, lat) in enumerate(rec):
            pts.append((int(idx), float(lon), float(lat), True))
            if samples_per_segment > 1 and i + 1 < len(rec):
                _, lon1, lat1 = rec[i + 1]
                for f in np.linspace(0.0, 1.0, samples_per_segment + 1)[1:-1]:
                    pts.append((int(idx), float(lon) + f * (float(lon1) - float(lon)),
                                float(lat) + f * (float(lat1) - float(lat)), False))

        for idx, lon, lat, is_wp in pts:
            r = {"index": idx}
            pr = local_cl.project(lon, lat)                # equations (3)-(4)
            km = global_cl.project(lon, lat).arclength_m / 1000.0
            w = float(np.interp(km, width_by_km[0], width_by_km[1]))
            r_norm = normalised_lateral(pr.lateral_m, w)
            v_avg = float(np.interp(km, np.arange(len(v_avg_by_km)) * 1.0, v_avg_by_km))
            pp = ProfileParams(v_avg_knots=v_avg, channel_width_m=w,
                               bend_strength=bend_b)
            beta = float(bend_factor(r_norm, pr.curvature_1_per_m, pp))   # equation (2)
            rows.append(dict(index=int(r["index"]), is_waypoint=is_wp,
                             section=int(s.section),
                             local_axis=local, chainage_km=km,
                             lateral_m=pr.lateral_m, r=r_norm,
                             curvature=pr.curvature_1_per_m, width_m=w,
                             v_avg_kn=v_avg, beta=beta,
                             v_kn=float(velocity_at(r_norm, pr.curvature_1_per_m,
                                                    pp))))

    per_wp = pd.DataFrame(rows)
    per_section = (per_wp.groupby("section").v_kn.mean())     # Section 3.2.4
    _shared = per_wp.groupby("section").v_kn.apply(
        lambda v: segment_mean_velocity(v.values))
    assert np.allclose(per_section.values, _shared.reindex(per_section.index).values,
                       rtol=0, atol=1e-12), (
        "the section mean no longer agrees with src.profile.segment_mean_velocity; "
        "the pipeline's section mean and the tested one have diverged")
    per_wp = (per_wp[per_wp.is_waypoint].drop_duplicates("index")
              .set_index("index").sort_index())
    return per_section, per_wp


def main(run: str = "flowfield_rebuild") -> None:
    out = output_dir(run)
    p = params()["flow_field"]
    wp = waypoints()
    sections = load_sections()

    seg = pd.read_excel(SEGMENT_FILE)
    published_od = seg.groupby(["start", "end"]).avg_wspeed.first()
    known_ports = set(seg.start) | set(seg.end)

    reproduce_published(sections, published_od, known_ports)

    net = ChannelNetwork(wp)
    cl = net.centreline(net.path(1, 93),
                        resample_m=p["centreline_resample_m"],
                        smooth_window=p["centreline_smooth_window"])
    total_km = cl.arclength[-1] / 1000.0
    grid = np.arange(0.0, total_km + 1.0, 1.0)

    v_avg_grid, anchor_info = section_average_speed(out, grid, p)
    if anchor_info["zero_km"]:
        pass

    width = pd.read_csv(out / "channel_width_by_reach.csv").sort_values("reach_km")
    width_by_km = (width.reach_km.values.astype(float), width.width_m.values)

    results = {}
    for b in p["bend_factor_sweep"]:
        per_section, per_wp = build(b, wp, cl, width_by_km, v_avg_grid, sections,
                                    p["centreline_smooth_window"])
        g = port_graph(sections, per_section, known_ports)
        results[b] = all_directed_currents(g, published_od.index)
        if b == p["bend_factor_b"]:
            per_wp.to_csv(out / "waypoint_velocities.csv")
            per_section.to_csv(out / "section_currents.csv", header=["current_kn"])
            main_wp = per_wp

    base_od = None
    for n_s in (1, 2, 5, 10, 20):
        ps, _ = build(p["bend_factor_b"], wp, cl, width_by_km, v_avg_grid,
                      sections, p["centreline_smooth_window"], samples_per_segment=n_s)
        od = all_directed_currents(port_graph(sections, ps, known_ports),
                                   published_od.index)
        if base_od is None:
            base_od = od

    new_od = results[p["bend_factor_b"]]
    cmp = pd.DataFrame({"published": published_od, "rebuilt": new_od})
    for b in p["bend_factor_sweep"]:
        cmp[f"rebuilt_b{b}"] = results[b]
    cmp["delta"] = cmp.rebuilt - cmp.published
    cmp["sign_agrees"] = np.sign(cmp.rebuilt) == np.sign(cmp.published)


    for b in p["bend_factor_sweep"]:
        d = (cmp[f"rebuilt_b{b}"] - cmp["rebuilt_b0.0"]).abs()

    wp_km = main_wp.chainage_km
    for lo, hi in [(0, 200), (200, 400), (400, 600), (600, 760)]:
        sel = main_wp[(wp_km >= lo) & (wp_km < hi)]
        if len(sel):
            pass

    cmp.to_csv(out / "port_pair_currents.csv")

    anchor_info = dict(anchor_info)
    anchor_info["zero_waypoints"] = int((main_wp.v_avg_kn == 0).sum())
    anchor_info["n_waypoints"] = int(len(main_wp))
    _zero_secs = set(main_wp.loc[main_wp.v_avg_kn == 0, "section"].unique())
    anchor_info["sections_with_a_zero"] = int(len(_zero_secs))
    anchor_info["sections_all_zero"] = int((per_section.abs() < 1e-12).sum())
    anchor_info["n_sections"] = int(len(per_section))
    anchor_info["zero_port_pairs"] = int((cmp["rebuilt"].abs() < 1e-12).sum())
    anchor_info["n_port_pairs"] = int(len(cmp))
    pd.Series(anchor_info).to_csv(out / "v_avg_anchor.csv", header=False)


if __name__ == "__main__":
    main()
