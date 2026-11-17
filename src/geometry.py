
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def to_local_xy(lon: np.ndarray, lat: np.ndarray,
                lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    """Equirectangular projection to metres about (lon0, lat0)."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    x = np.radians(lon - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    return x, y


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance in metres."""
    lon1, lat1, lon2, lat2 = map(lambda a: np.radians(np.asarray(a, float)),
                                 (lon1, lat1, lon2, lat2))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


@dataclass
class Projection:
    """Where a query point falls relative to the centreline."""
    segment_index: int
    arclength_m: float
    lateral_m: float
    curvature_1_per_m: float
    tangent: np.ndarray
    normal: np.ndarray


class Centreline:
    """A polyline through the channel, with tangents, normals and curvature."""

    def __init__(self, lon: np.ndarray, lat: np.ndarray,
                 resample_m: float = 500.0, smooth_window: int = 9):
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        self.lon0, self.lat0 = float(np.mean(lon)), float(np.mean(lat))
        x, y = to_local_xy(lon, lat, self.lon0, self.lat0)
        self.raw_xy = np.column_stack([x, y])

        self.xy = self._resample(self.raw_xy, resample_m)
        if smooth_window > 1:
            self.xy = self._smooth(self.xy, smooth_window)

        seg = np.diff(self.xy, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.tangents = seg / self.seg_len[:, None]
        self.normals = np.column_stack([-self.tangents[:, 1], self.tangents[:, 0]])
        self.arclength = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.curvature = self._curvature(self.xy)

    @staticmethod
    def _resample(xy: np.ndarray, step_m: float) -> np.ndarray:
        d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        n = max(int(d[-1] // step_m) + 1, 2)
        target = np.linspace(0, d[-1], n)
        return np.column_stack([np.interp(target, d, xy[:, 0]),
                                np.interp(target, d, xy[:, 1])])

    @staticmethod
    def _smooth(xy: np.ndarray, window: int) -> np.ndarray:
        if window % 2 == 0:
            window += 1
        pad = window // 2
        kernel = np.ones(window) / window
        out = np.empty_like(xy)
        for i in range(2):
            padded = np.pad(xy[:, i], pad, mode="edge")
            out[:, i] = np.convolve(padded, kernel, mode="valid")
        return out

    @staticmethod
    def _curvature(xy: np.ndarray) -> np.ndarray:
        """Signed curvature k = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)."""
        dx, dy = np.gradient(xy[:, 0]), np.gradient(xy[:, 1])
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        denom = np.power(dx ** 2 + dy ** 2, 1.5)
        k = np.divide(dx * ddy - dy * ddx, denom,
                      out=np.zeros_like(denom), where=denom > 0)
        k[~np.isfinite(k)] = 0.0
        return k

    def project(self, lon: float, lat: float) -> Projection:
        """Project one point onto the centreline."""
        px, py = to_local_xy(np.array([lon]), np.array([lat]), self.lon0, self.lat0)
        p = np.array([px[0], py[0]])

        a = self.xy[:-1]
        d = p - a                                        # equation (3)
        along = np.einsum("ij,ij->i", d, self.tangents)
        along = np.clip(along, 0.0, self.seg_len)
        foot = a + along[:, None] * self.tangents
        dist = np.linalg.norm(p - foot, axis=1)
        i = int(np.argmin(dist))

        lateral = float(np.dot(p - a[i], self.normals[i]))  # equation (3), signed as in (4)
        return Projection(
            segment_index=i,
            arclength_m=float(self.arclength[i] + along[i]),
            lateral_m=lateral,
            curvature_1_per_m=float(self.curvature[i]),
            tangent=self.tangents[i],
            normal=self.normals[i],
        )


class ChannelNetwork:
    """The waypoint graph, and centrelines cut from it."""

    def __init__(self, waypoints):
        import networkx as nx

        self.pos = {int(r["index"]): (float(r["lon"]), float(r["lat"]))
                    for _, r in waypoints.iterrows()}
        self.label = {int(r["index"]): r["lable"] for _, r in waypoints.iterrows()}
        self.graph = nx.Graph()
        for _, row in waypoints.iterrows():
            i = int(row["index"])
            for token in str(row["connect_index"]).split(","):
                token = token.strip()
                if not token.isdigit():
                    continue
                j = int(token)
                lon_i, lat_i = self.pos[i]
                lon_j, lat_j = self.pos[j]
                self.graph.add_edge(i, j, weight=float(haversine_m(lon_i, lat_i, lon_j, lat_j)))

    def path(self, start: int, end: int) -> list[int]:
        """Shortest waypoint path between two waypoint indices."""
        import networkx as nx
        return nx.shortest_path(self.graph, start, end, weight="weight")

    def path_length_m(self, start: int, end: int) -> float:
        import networkx as nx
        return float(nx.shortest_path_length(self.graph, start, end, weight="weight"))

    def centreline(self, node_path: list[int], **kwargs) -> "Centreline":
        """Build a centreline along an explicit waypoint path."""
        lon = np.array([self.pos[i][0] for i in node_path])
        lat = np.array([self.pos[i][1] for i in node_path])
        return Centreline(lon, lat, **kwargs)


def normalised_lateral(lateral_m: float, channel_width_m: float) -> float:
    """Equation (3): lateral offset over half the channel width, clipped to [-1, 1]."""
    r = 2.0 * lateral_m / channel_width_m
    return float(np.clip(r, -1.0, 1.0))
