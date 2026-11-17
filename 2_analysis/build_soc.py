
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401

from resolve_siting import (  # noqa: E402
    CANDIDATE_PORTS, PORT_BIT, Router, TOPOLOGY, plan2_feasible,
)
from src.loaders import RESULTS, output_dir, params, ship_parameters, write_dir  # noqa: E402

NM_TO_KM = 1.852
D_MAX_SWEEP_KM = [0, 5, 10, 20, 30, 50]
SOC_FLOORS = [0.0, 0.05, 0.10, 0.20]

RESCUE_CALLS = 0
RESCUE_HITS = 0
RESCUE_SITES: list[tuple[str, str]] = []


def feasible(legs, battery, mask, router, all_split=False) -> bool:
    bottleneck, remaining = set(), battery
    if not all_split:
        for i, leg in enumerate(legs):
            e = leg["e"]
            if e > battery or e > remaining:
                bottleneck.add(i)
            remaining = battery - e if e > remaining else remaining - e
            if PORT_BIT.get(leg["end"], 0) & mask:
                remaining = battery

    remaining = battery
    for i, leg in enumerate(legs):
        span = leg["distance"]
        per_km = leg["e"] / span if span > 0 else 0.0
        pieces = (router.split(leg["start"], leg["end"], mask)
                  if (all_split or i in bottleneck) else [(leg["end"], span)])
        cur = leg["start"]
        for end_port, d in pieces:
            e = leg["e"] * (d / span) if span > 0 else leg["e"]
            if e > battery or e > remaining:
                return False
            else:
                remaining -= e
            if PORT_BIT.get(end_port, 0) & mask:
                remaining = battery
            cur = end_port
    return True


def floor_explicit(legs, battery, mask, router, f: float,
                   reset_to_full: bool) -> bool:
    """Safety floor written out, so the reset semantics can be varied."""
    cap = battery if reset_to_full else battery * (1 - f)
    floor = f * battery
    bottleneck, remaining = set(), cap
    for i, leg in enumerate(legs):
        e = leg["e"]
        if e > cap - floor or e > remaining - floor:
            bottleneck.add(i)
        remaining = cap - e if e > remaining - floor else remaining - e
        if PORT_BIT.get(leg["end"], 0) & mask:
            remaining = cap

    remaining = cap
    for i, leg in enumerate(legs):
        span = leg["distance"]
        pieces = (router.split(leg["start"], leg["end"], mask)
                  if i in bottleneck else [(leg["end"], span)])
        for end_port, d in pieces:
            e = leg["e"] * (d / span) if span > 0 else leg["e"]
            if e > cap - floor or e > remaining - floor:
                return False
            remaining -= e
            if PORT_BIT.get(end_port, 0) & mask:
                remaining = cap
    return True
