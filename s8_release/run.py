
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401,E402

from src import provenance  # noqa: E402
from src.loaders import DATA, ROOT, output_dir, params  # noqa: E402

RUN = "paper/s8_release"
HERE = Path(__file__).resolve().parent


def pseudonyms(mmsis) -> dict:
    """Stable vessel codes, deterministic and order independent."""
    salt = "ETRAN-D-26-00296"
    ranked = sorted(mmsis, key=lambda m: hashlib.sha256(
        f"{salt}:{m}".encode()).hexdigest())
    return {m: f"V{i + 1:03d}" for i, m in enumerate(ranked)}


def build_voyages(out: Path) -> pd.DataFrame:
    legs = pd.read_excel(DATA / "04_paper_baseline" / "segments_legs_with_timestamps.xlsx")
    names = pd.read_csv(HERE / "port_names_en.csv")
    en = dict(zip(names.port_zh, names.port_en))

    missing = (set(legs["start"]) | set(legs["end"])) - set(en)

    code = pseudonyms(sorted(legs.mmsi.unique()))
    d = pd.DataFrame({
        "vessel_id": legs.mmsi.map(code),
        "leg_seq": legs.groupby("mmsi").cumcount() + 1,
        "origin_port": legs["start"].map(en),
        "destination_port": legs["end"].map(en),
        "distance_nm": legs["distance"].astype(float),
        "departure_utc8": pd.to_datetime(legs["start_time"]),
        "arrival_utc8": pd.to_datetime(legs["end_time"]),
        "sailing_hours": legs["sailing_h"].astype(float),
        "design_speed_kn": legs["75%_speed"].astype(float),
        "current_along_track_kn": legs["avg_wspeed"].astype(float),
        "teu_capacity": legs["teu"].astype(int),
        "battery_rated_kwh": legs["rated_capacity_kwh"].astype(int),
        "leg_energy_kwh": legs["energy_kwh"].astype(float),
    }).sort_values(["vessel_id", "leg_seq"], kind="stable").reset_index(drop=True)

    assert "mmsi" not in d.columns
    assert d.vessel_id.nunique() == legs.mmsi.nunique() == 172
    assert len(d) == len(legs) == 4360

    d.to_csv(out / "anonymised_voyages.csv", index=False)
    names.to_csv(out / "port_names.csv", index=False)
    return d


def build_readme_table(d: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Column dictionary."""
    units = {
        "vessel_id": ("stable pseudonym, salted hash of the vessel identifier", "-"),
        "leg_seq": ("leg order within the vessel's 90 day record", "-"),
        "origin_port": ("port of departure", "-"),
        "destination_port": ("port of arrival", "-"),
        "distance_nm": ("along channel distance on the shortest path", "nautical mile"),
        "departure_utc8": ("departure timestamp, China Standard Time", "-"),
        "arrival_utc8": ("arrival timestamp, China Standard Time", "-"),
        "sailing_hours": ("time under way as recorded", "hour"),
        "design_speed_kn": ("75th percentile of observed speed over ground", "knot"),
        "current_along_track_kn": ("current projected on the track, positive downstream",
                                   "knot"),
        "teu_capacity": ("nominal container capacity", "TEU"),
        "battery_rated_kwh": ("rated battery energy, multiples of 1600 kWh", "kWh"),
        "leg_energy_kwh": ("leg energy over depth of discharge and utilization factor, compared with rated capacity (Section 3.4.2)", "kWh"),
    }
    rows = [dict(column=c, description=units[c][0], unit=units[c][1],
                 dtype=str(d[c].dtype)) for c in d.columns]
    t = pd.DataFrame(rows)
    t.to_csv(out / "anonymised_voyages_columns.csv", index=False)
    return t


def main() -> None:
    out = output_dir(RUN)

    d = build_voyages(out)

    t = build_readme_table(d, out)

    inputs = [DATA / "04_paper_baseline" / "segments_legs_with_timestamps.xlsx",
              HERE / "port_names_en.csv"]
    n = 0
    for prod in sorted(out.rglob("*")):
        if prod.is_file() and not prod.name.endswith(provenance.SUFFIX):
            provenance.record(prod, script=__file__, inputs=inputs, params_used=None)
            n += 1


if __name__ == "__main__":
    main()
