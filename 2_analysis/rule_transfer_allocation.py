
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401,E402

import build_coupled_inventory as ci  # noqa: E402
import build_pack_soc as fleet_model  # noqa: E402
import build_rule_based_transfer as rules  # noqa: E402
from build_transfer_model import port_distances  # noqa: E402
from src.loaders import RESULTS, params, write_dir  # noqa: E402

RUN = "rule_transfer_allocation"
WINDOWS = (7, 14, 30, 45, 60, 91)
SHARES = range(50, 301, 5)
COVERS = (None, 7, 14, 21, 28, 35, 45)


def station_inputs(case):
    """Reproduce the selected powers and per-port charging inputs."""
    p = params()
    eta_c = float(p["lcc"]["energy_transfer_efficiency"])
    ref_kw = float(p["lcc"]["station_power_kw"])
    advice = pd.read_csv(RESULTS / "paper/s7_robustness/station_sizing_advice.csv").set_index("port")
    initial = {port: int(advice.loc[port, "berths_required"]) for port in case.ports}
    _, all_calls = ci.common_calls(case, initial)
    days = len(case.days) + ci.CLEARING_DAYS
    out = {mode: {} for mode in case.plans}
    for port in case.ports:
        for power in [ref_kw * k for k in (1, 2, 3, 4)]:
            ev = {m: ci.charging_case(c[port], power, eta_c, days) for m, c in all_calls.items()}
            if all(r["charge_kwh"] <= power * 24 * len(case.days) and r["pending_packs"] == 0
                   for r in ev.values()):
                for m, r in ev.items():
                    out[m][port] = r
                break
    return out


def stack(case, inputs):
    get = lambda key: np.stack([inputs[p][key] for p in case.ports])  # noqa: E731
    return get("load"), get("ready"), get("floor")


def opening_requirement(load, ready, floor, days):
    return ci.no_transfer_opening(load[:, :days], ready[:, :days], floor[:, :days])


def pooled_bound(load, ready, floor):
    """Single pooled stock: a lower bound for any transfer policy at these flows."""
    net = (load - ready).sum(axis=0)
    need = np.cumsum(net) - net + floor.sum(axis=0)
    return int(max(need.max(), 0))


def main():
    case = fleet_model.build_case()
    inputs = station_inputs(case)
    ports = case.ports
    dist = port_distances(ports)
    travel = np.ceil(dist / params()["inventory"]["transfer_km_per_day"]).astype(int)
    transit = travel + (travel > 0)
    fleet = rules.equal_allocation(ci.RULE_FLEET, len(ports))

    load, ready, floor = stack(case, inputs["DASM"])
    full = ready.shape[1]
    no_transfer = opening_requirement(load, ready, floor, full)
    rows = []
    for cover in COVERS:
      for w in WINDOWS:
        base = opening_requirement(load, ready, floor, min(w, full))
        for share in SHARES:
            opening = (base * share + 99) // 100
            r = rules.replay(load, ready, dist, transit, opening, fleet,
                             day_opening_need=floor, cover_days=cover)
            top = r["emergency"]
            if top.sum() <= ci.TOL:
                rows.append(dict(
                    cover_days="default" if cover is None else cover,
                    window_days=w, share_pct=share, opening_packs=int(opening.sum()),
                    vs_no_transfer_pct=100 * opening.sum() / no_transfer.sum(),
                    loaded_pack_km=r["pack_km"], empty_truck_km=r["empty_km"],
                    moved_packs=r["moved"], closing_packs=int(r["stock"][:, -1].sum()),
                    **{f"open_{p}": int(v) for p, v in zip(ports, opening)},
                ))
                break
        else:
            rows.append(dict(cover_days="default" if cover is None else cover,
                             window_days=w, share_pct=None, opening_packs=None))
    table = pd.DataFrame(rows)
    bounds = dict(pooled_lower_bound=pooled_bound(load, ready, floor),
                  no_transfer=int(no_transfer.sum()))

    diag = []
    for mode in case.plans:
        l, rd, f = stack(case, inputs[mode])
        for d in (30, 45, 60, full):
            diag.append(dict(mode=mode, horizon_days=d,
                             no_transfer_opening=int(opening_requirement(l, rd, f, d).sum()),
                             pooled_bound=pooled_bound(l[:, :d], rd[:, :d], f[:, :d])))
    diag = pd.DataFrame(diag)

    out = write_dir(RUN)
    table.to_csv(out / "rule_allocation_by_cover_window.csv", index=False)
    diag.to_csv(out / "no_transfer_stock_by_window.csv", index=False)
    pd.Series(bounds).to_csv(out / "lower_bounds.csv", header=["packs"])


if __name__ == "__main__":
    main()
