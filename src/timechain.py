
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LABEL = ROOT / "03_data" / "01_raw_ais" / "yangtze_ais_labelled"
RAW_JSON = ROOT / "03_data" / "00_raw_json"
STOPS = ROOT / "03_data" / "06_screening_list" / "all_vessel_routes.xlsx"

BERTH, ANCHOR, OTHER = "berth", "anchor", "other"


def port_centroids() -> dict[str, tuple[float, float]]:
    from scipy.spatial import ConvexHull
    st = pd.read_excel(STOPS)
    out = {}
    for p, g in st.groupby("port"):
        pts = g[["lon", "lat"]].to_numpy(float)
        if len(pts) >= 3:
            try:
                pts = pts[ConvexHull(pts).vertices]
            except Exception:
                pass
        out[p] = tuple(pts.mean(axis=0))
    return out


def status_by_time(mmsi: int) -> dict[pd.Timestamp, int]:
    f = RAW_JSON / f"{mmsi}.json"
    if not f.exists():
        return {}
    rows = json.load(f.open(encoding="utf-8")).get("data") or []
    out = {}
    for r in rows:
        t = r.get("postime")
        s = r.get("status")
        if t is not None and s is not None:
            out[pd.Timestamp(t)] = int(s)
    return out


def stop_runs(d: pd.DataFrame, moving_kn: float, min_points: int,
              status_share: float) -> pd.DataFrame:
    """Maximal runs of stopped points, typed by navigational status."""
    d = d.sort_values("updatetime").reset_index(drop=True)
    stopped = pd.to_numeric(d["speed"], errors="coerce") <= moving_kn
    grp = (stopped != stopped.shift()).cumsum()
    rows = []
    for _, g in d[stopped].groupby(grp[stopped]):
        if len(g) < min_points:
            continue
        if "status" in g.columns:
            st = pd.to_numeric(g["status"], errors="coerce")
            f5 = float((st == 5).mean()) if st.notna().any() else 0.0
            f1 = float((st == 1).mean()) if st.notna().any() else 0.0
        else:
            f5 = f1 = 0.0
        port = g["port"].mode()
        rows.append(dict(
            t0=g["updatetime"].iloc[0], t1=g["updatetime"].iloc[-1],
            hours=(g["updatetime"].iloc[-1] - g["updatetime"].iloc[0]).total_seconds() / 3600,
            lon=float(g["lon"].median()), lat=float(g["lat"].median()),
            port=(port.iloc[0] if len(port) else None),
            cls=(BERTH if f5 >= status_share else
                 ANCHOR if f1 >= status_share else OTHER),
            n=len(g)))
    return pd.DataFrame(rows)


def _zone(lon: float, lat: float, cent: dict, tol: float) -> str | None:
    best, bd = None, np.inf
    for p, (px, py) in cent.items():
        dd = ((lon - px) ** 2 + (lat - py) ** 2) ** 0.5
        if dd < bd:
            best, bd = p, dd
    return best if bd <= tol else None


def port_calls(runs: pd.DataFrame, cent: dict, cfg: dict) -> pd.DataFrame:
    """Assemble runs into port calls, then split each call into components."""
    if runs.empty:
        return pd.DataFrame()
    runs = runs.sort_values("t0").reset_index(drop=True)
    runs["zone"] = [_zone(r.lon, r.lat, cent, cfg["zone_deg"])
                    for r in runs.itertuples()]

    call_id, ids = 0, []
    prev = None
    for r in runs.itertuples():
        if prev is not None:
            gap = (r.t0 - prev.t1).total_seconds() / 3600
            if r.zone != prev.zone or gap > cfg["gap_max_h"]:
                call_id += 1
        ids.append(call_id)
        prev = r
    runs["call"] = ids

    out = []
    for cid, g in runs.groupby("call"):
        z = g.zone.iloc[0]
        if z is None or not (g.port == z).any():
            continue
        berth = g[g.cls == BERTH]
        if len(berth):
            anchor_at, mode = berth.index[0], "status"
        else:
            cand = g[(g.port == z) & (g.hours >= cfg["min_berth_h"])]
            if not len(cand):
                anchor_at, mode = None, "no_time"
            else:
                anchor_at, mode = cand.hours.idxmax(), "dwell"
        timed = anchor_at is not None
        before = g.loc[:anchor_at].iloc[:-1] if timed else g.iloc[:0]
        out.append(dict(
            call=cid, port=z, mode=mode,
            t_first=g.t0.iloc[0], t_last=g.t1.iloc[-1],
            i_first=int(g.index[0]), i_last=int(g.index[-1]),
            wait_h=(float(before.loc[before.cls == ANCHOR, "hours"].sum())
                    if timed else np.nan),
            wait_all_h=float(before.hours.sum()) if timed else np.nan,
            berth_h=(float(berth.hours.max()) if mode == "status"
                     else float(g.loc[anchor_at, "hours"]) if timed else np.nan),
            stay_h=float(g.hours.sum()), n_runs=len(g)))
    calls = pd.DataFrame(out)
    if calls.empty:
        return calls

    calls = calls.sort_values("t_first").reset_index(drop=True)
    prev_end = calls.t_last.shift()
    next_start = calls.t_first.shift(-1)
    free_before = (calls.t_first - prev_end).dt.total_seconds() / 3600
    free_after = (next_start - calls.t_last).dt.total_seconds() / 3600

    outside = runs[runs.zone.isna()]
    appr, dep = [], []
    for r in calls.itertuples():
        pre = outside[outside.t1 <= r.t_first]
        post = outside[outside.t0 >= r.t_last]
        appr.append((r.t_first - pre.t1.iloc[-1]).total_seconds() / 3600
                    if len(pre) else np.nan)
        dep.append((post.t0.iloc[0] - r.t_last).total_seconds() / 3600
                   if len(post) else np.nan)
    calls["appr_h"] = np.minimum(appr, free_before)
    calls["dep_h"] = np.minimum(dep, free_after)
    return calls
