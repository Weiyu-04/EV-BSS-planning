
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "03_data"
RESULTS = ROOT / "05_result"
PARAMS_FILE = ROOT / "params.yaml"

PLAN_FILE_KEY = {
    "plan1_non_flexible": "Plan 1",
    "plan2_flexible_swap": "Plan 2",
    "plan3_flexible_charge": "mixed supply",
}


@lru_cache(maxsize=1)
def params() -> dict:
    """Load params.yaml once."""
    with PARAMS_FILE.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def ship_parameters() -> pd.DataFrame:
    """172 ships x 25 columns: dimensions, battery configuration, speed limits."""
    return pd.read_excel(DATA / "03_ship_parameters" / "172_ship_parameters.xlsx")


@lru_cache(maxsize=1)
def segments_baseline() -> pd.DataFrame:
    """4360 port-to-port legs with the frozen `avg_wspeed` current speeds."""
    return pd.read_excel(DATA / "04_paper_baseline" / "segments_leg.xlsx")


@lru_cache(maxsize=1)
def segments_with_energy() -> pd.DataFrame:
    """Baseline legs plus per-leg energy, sailing time and cumulative totals."""
    return pd.read_excel(DATA / "04_paper_baseline" / "segments_legs_energy_and_time.xlsx")


@lru_cache(maxsize=1)
def segments_with_timestamps() -> pd.DataFrame:
    """Baseline legs plus absolute AIS start/end timestamps (90-day window)."""
    return pd.read_excel(DATA / "04_paper_baseline" / "segments_legs_with_timestamps.xlsx")


@lru_cache(maxsize=1)
def waypoints() -> pd.DataFrame:
    """93 channel waypoints: coordinates, connectivity, section membership."""
    return pd.read_excel(DATA / "02_channel_topology" / "nodes_and_waypoints.xlsx")


@lru_cache(maxsize=1)
def frozen_flow_velocities() -> pd.DataFrame:
    return pd.read_excel(DATA / "02_channel_topology" / "nodes_and_waypoints_current.xlsx")


def siting_result(n_stations: int, plan: str, growth: int = 0) -> pd.DataFrame:
    key = PLAN_FILE_KEY[plan]
    path = (DATA / "04_paper_baseline" / "siting_result" / f"n_stations{n_stations}"
            / f"growth{growth}%-{key}-{n_stations}-optimised swapping.xlsx")
    return pd.read_excel(path)


@lru_cache(maxsize=1)
def inventory_summary() -> pd.DataFrame:
    """Per-port initial/average/maximum inventory under EAS and DAS."""
    return pd.read_excel(DATA / "04_paper_baseline" / "inventory_growth_0pct" / "summary_all_ports.xlsx")


def output_dir(run: str) -> Path:
    path = RESULTS / run
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_dir(run: str) -> Path:
    root = os.environ.get("EVBSS_WRITE_ROOT")
    path = (Path(root) if root else RESULTS) / run
    path.mkdir(parents=True, exist_ok=True)
    return path
