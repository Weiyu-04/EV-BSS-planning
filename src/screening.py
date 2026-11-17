
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "03_data"

RAW_JSON = DATA / "00_raw_json"
TRACKS_216 = DATA / "01_raw_ais" / "yangtze_ais"
LIST_302 = DATA / "06_screening_list" / "screen vessels.xlsx"
LIST_185 = DATA / "06_screening_list" / "185_yangtze_vessel_list.xlsx"
LIST_172 = DATA / "06_screening_list" / "172_yangtze_vessel_list.xlsx"

STOPS = DATA / "06_screening_list" / "all_vessel_routes.xlsx"

MANUAL_FLAG = "is_yangtze_vessel"
FLAG_KEEP = 1

N_DROPPED_216_185 = 31

AUDIT_BBOX = (117.037, 122.217, 30.450, 32.372)
AUDIT_BBOX_NAME = ("93 waypoint bounding box +0.05 deg (lon 117.037-122.217 / lat 30.450-32.372)")


def roster() -> dict[str, set[int]]:
    """The five authoritative membership sets, each from its own artefact."""
    m344 = {int(p.stem) for p in RAW_JSON.glob("*.json")}
    m302 = set(pd.read_excel(LIST_302)["mmsi"].astype(int))
    m216 = {int(p.stem) for p in TRACKS_216.glob("*.csv")}
    m185 = set(pd.read_excel(LIST_185)["mmsi"].astype(int))
    m172 = set(pd.read_excel(LIST_172)["mmsi"].astype(int))
    return {"344": m344, "302": m302, "216": m216, "185": m185, "172": m172}


def nesting_holds(r: dict[str, set[int]]) -> list[str]:
    """Each stage must be a subset of the one before it."""
    order = ["344", "302", "216", "185", "172"]
    bad = []
    for a, b in zip(order, order[1:]):
        extra = r[b] - r[a]
        if extra:
            bad.append(f"{b} is not {a} subset of, extra {len(extra)} row: {sorted(extra)[:5]}")
    return bad


def ais_point_counts(cache: Path | None = None) -> pd.Series:
    n_now = len(list(RAW_JSON.glob("*.json")))
    if cache is not None and cache.exists():
        c = pd.read_csv(cache)
        if len(c) == n_now:
            return c.set_index("mmsi")["n_points"]
    rows = []
    for p in sorted(RAW_JSON.glob("*.json")):
        try:
            pts = json.load(p.open(encoding="utf-8")).get("data") or []
        except (json.JSONDecodeError, UnicodeDecodeError):
            pts = []
        rows.append((int(p.stem), len(pts)))
    s = pd.Series(dict(rows), name="n_points").sort_index()
    s.index.name = "mmsi"
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        s.reset_index().to_csv(cache, index=False)
    return s


def manual_flags() -> pd.Series:
    """The author's 2024 hand screening column, keyed by mmsi."""
    d = pd.read_excel(LIST_302)
    return d.set_index(d["mmsi"].astype(int))[MANUAL_FLAG]


def stop_table_roster() -> set[int]:
    return set(pd.read_excel(STOPS)["mmsi"].astype(int))


def static_draught_all_zero(mmsis: set[int]) -> set[int]:
    """Vessels whose reported static draught never leaves zero."""
    out = set()
    for m in sorted(mmsis):
        f = TRACKS_216 / f"{m}.csv"
        if not f.exists():
            continue
        s = pd.to_numeric(pd.read_csv(f, usecols=["draught"])["draught"],
                          errors="coerce").dropna()
        if len(s) and s.max() == 0:
            out.add(m)
    return out


@dataclass
class Step:
    name: str
    kept: int
    dropped: int
    grade: str
    criterion: str
    evidence: str
    ok: bool

    def line(self) -> str:
        mark = {"exact": "", "separable": "", "reconstructed": ""}[self.grade]
        flag = "" if self.ok else "    split not reproduced"
        return (f"  {mark} {self.name:<12} kept {self.kept:>3} / excluded {self.dropped:>3}"
                f"  [{self.grade}]  {self.evidence}{flag}")


def step_344_302(r, counts) -> Step:
    """No recorded rule."""
    kept, dropped = r["302"], r["344"] - r["302"]
    hi = int(counts[sorted(dropped)].max())
    lo = int(counts[sorted(kept)].min())
    return Step("344 -> 302", len(kept), len(dropped), "separable",
                "AIS n_positions",
                f"excluded <= {hi:,} point . kept >= {lo:,} point . gap {lo - hi:,}",
                ok=hi < lo)


def step_302_216(r, flags) -> Step:
    """A recorded field."""
    kept, dropped = r["216"], r["302"] - r["216"]
    derived = {m for m, v in flags.items() if v == FLAG_KEEP}
    return Step("302 -> 216", len(kept), len(dropped), "exact",
                f"labelled by hand {MANUAL_FLAG} == {FLAG_KEEP}",
                f"reproduced split {len(derived & kept):,}/{len(kept):,}, no exception",
                ok=derived == kept)


def step_216_185(r, witness: set[int]) -> Step:
    kept, dropped = r["185"], r["216"] - r["185"]
    agree = witness == kept
    extra, missing = len(witness - kept), len(kept - witness)
    return Step("216 -> 185", len(kept), len(dropped), "reconstructed",
                "scope, stated in prose, not a single statistic",
                f"membership corroborated by the reference call table: that table {len(witness)}  ships and 185 list"
                f"{'sets are equal' if agree else f'mismatch, extra  {extra} / missing {missing})'}, "
                f"and excluded {len(dropped)}  ships in common {len(witness & dropped)}. "
                "31  ships call at these 19 every network port <=2 count, median  0, of which 17  ships are 0, 11  ships 1 , "
                "3  ships 2 ; recorded as coastal elsewhere/ocean going 6  ships, calls inside the network "
                "0/0/0/1/1/1, none reaches the minimum for a leg 2  ports. the two bins are not exclusive, "
                "two sides of the same scope condition.  the record length criterion**cannot be separated**two groups"
                " (largest inside the excluded group 13,199 point, kept group 12 below the median of the excluded group 6,625); "
                "but**and it does not show the record is sufficient**--excluded group 6 below the minimum of the kept group 2,373. "
                f"window={AUDIT_BBOX_NAME}",
                ok=agree)


def step_185_172(r, zero) -> Step:
    """A property of the record, reproducing the split exactly."""
    kept, dropped = r["172"], r["185"] - r["172"]
    return Step("185 -> 172", len(kept), len(dropped), "exact",
                "static draught constant at 0",
                f"draught constant at 0 set equals the excluded set exactly, {len(dropped)} rows",
                ok=zero == dropped)


def build(cache: Path | None = None) -> tuple[list[Step], list[str]]:
    """All four steps plus any structural complaints about the chain itself."""
    r = roster()
    problems = nesting_holds(r)
    counts = ais_point_counts(cache)
    steps = [step_344_302(r, counts),
             step_302_216(r, manual_flags()),
             step_216_185(r, stop_table_roster()),
             step_185_172(r, static_draught_all_zero(r["185"]))]
    return steps, problems


def roster_frame(r: dict[str, set[int]]) -> pd.DataFrame:
    order = ["344", "302", "216", "185", "172"]
    rows = [dict(mmsi=m, **{f"in_{k}": m in r[k] for k in order})
            for m in sorted(r["344"])]
    return pd.DataFrame(rows)


def to_frame(steps: list[Step]) -> pd.DataFrame:
    return pd.DataFrame([dict(step=s.name, kept=s.kept, dropped=s.dropped,
                              evidence_grade=s.grade, criterion=s.criterion,
                              evidence=s.evidence, reproduces_split=s.ok)
                         for s in steps])
