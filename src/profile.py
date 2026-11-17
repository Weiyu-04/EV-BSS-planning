
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ALPHA_UNIFORM_SECTION_MEAN = 1.5

KNOTS_PER_MPS = 1.0 / 0.514444


@dataclass(frozen=True)
class ProfileParams:
    """Parameters of manuscript equations (1)-(2)."""

    v_avg_knots: float
    channel_width_m: float
    alpha: float = ALPHA_UNIFORM_SECTION_MEAN
    bend_strength: float = 0.0

    def __post_init__(self) -> None:
        if self.channel_width_m <= 0:
            raise ValueError("channel width must be positive")
        if self.bend_strength < 0:
            raise ValueError(
                "bend_strength is a magnitude; the outer-bank direction is "
                "handled by the sign convention, not by a negative b"
            )


def bend_factor(r: np.ndarray | float,
                curvature_1_per_m: np.ndarray | float,
                params: ProfileParams) -> np.ndarray:
    """Manuscript equation (2), with the half-width scaling and the bank sign."""
    r = np.asarray(r, float)
    k = np.asarray(curvature_1_per_m, float)
    half_width = params.channel_width_m / 2.0
    beta = 1.0 - params.bend_strength * half_width * k * r
    return np.maximum(beta, 0.0)


def velocity_at(r: np.ndarray | float,
                curvature_1_per_m: np.ndarray | float,
                params: ProfileParams) -> np.ndarray:
    """Manuscript equation (1): v_base(r) = v_avg * alpha * (1 - r^2) * beta."""
    r = np.clip(np.asarray(r, float), -1.0, 1.0)
    beta = bend_factor(r, curvature_1_per_m, params)
    return params.v_avg_knots * params.alpha * (1.0 - r ** 2) * beta


def section_mean(curvature_1_per_m: float, params: ProfileParams,
                 n: int = 2001) -> float:
    """Lateral average of the profile across the full section."""
    r = np.linspace(-1.0, 1.0, n)
    return float(np.trapezoid(velocity_at(r, curvature_1_per_m, params), r) / 2.0)


def segment_mean_velocity(velocities: np.ndarray) -> float:
    """Section 3.2.4: V_sec = (1/n) * sum_j v_seg,j."""
    v = np.asarray(velocities, float)
    if v.size == 0:
        raise ValueError("no velocities to average")
    return float(np.mean(v))


def signed_current_along_track(velocity_knots: np.ndarray | float,
                               heading_is_downstream: np.ndarray | bool) -> np.ndarray:
    """Current projected onto the direction of travel, as v_w of Constraint (15)."""
    v = np.asarray(velocity_knots, float)
    down = np.asarray(heading_is_downstream, bool)
    return np.where(down, v, -v)
