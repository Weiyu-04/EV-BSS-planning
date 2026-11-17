
from __future__ import annotations

import numpy as np

UNMARKED = 0
NOISE = -1
EARTH_R_M = 6371009.0


def _great_circle_m(lon0: float, lat0: float,
                    lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Great-circle distance in metres from one point to many."""
    p0, p = np.radians(lat0), np.radians(lat)
    dl = np.radians(lon - lon0)
    d = np.arccos(np.clip(np.sin(p0) * np.sin(p) +
                          np.cos(p0) * np.cos(p) * np.cos(dl), -1.0, 1.0))
    return EARTH_R_M * d


def st_dbscan(lon: np.ndarray, lat: np.ndarray, t_s: np.ndarray,
              eps_spatial_m: float, eps_temporal_min: float,
              min_pts: int) -> np.ndarray:
    """Labels in {NOISE} u {1..K}, one per input point, in input order."""
    n = len(lon)
    labels = np.full(n, UNMARKED, dtype=np.int64)
    eps_t = eps_temporal_min * 60.0
    cluster = 0

    def neighbours(i: int) -> np.ndarray:
        lo = int(np.searchsorted(t_s, t_s[i] - eps_t, side="left"))
        hi = int(np.searchsorted(t_s, t_s[i] + eps_t, side="right"))
        idx = np.arange(lo, hi)
        if len(idx) == 0:
            return idx
        d = _great_circle_m(lon[i], lat[i], lon[idx], lat[idx])
        return idx[d <= eps_spatial_m]

    for i in range(n):
        if labels[i] != UNMARKED:
            continue
        seeds = neighbours(i)
        if len(seeds) < min_pts:
            labels[i] = NOISE
            continue
        cluster += 1
        labels[seeds] = cluster
        stack = [j for j in seeds.tolist() if j != i]
        while stack:
            j = stack.pop()
            more = neighbours(j)
            if len(more) < min_pts:
                continue
            for k in more.tolist():
                if labels[k] in (UNMARKED, NOISE):
                    if labels[k] == UNMARKED:
                        stack.append(k)
                    labels[k] = cluster
    return labels
