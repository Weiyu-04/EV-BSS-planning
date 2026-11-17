
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _path  # noqa: F401

from src.loaders import output_dir, params, ship_parameters  # noqa: E402

KN_PER_KMH = 1 / 1.852
DESIGN_SPEED_KMH = 20.0
SERVICE_FRACTION = 0.75

ZHOU_T3 = [(135, 22.6, 2648), (88, 14.8, 880), (70.5, 12.6, 516),
           (86.8, 16.2, 882), (36.4, 7.1, 128.7), (52.8, 10, 272),
           (38.1, 7.05, 136), (94.2, 15.2, 882), (59.9, 11.2, 368),
           (100, 14.8, 1104)]


def main(run: str = "coefficient_correction") -> None:
    out = output_dir(run)
    ships = ship_parameters()
    adm = params()["admiralty"]
    k = adm["code_convention_factor"]

    z = pd.DataFrame(ZHOU_T3, columns=["L", "B", "P"])
    z["LB"] = z.L * z.B
    co = np.polyfit(z.LB, z.P, 2)
    pred = np.polyval(co, z.LB)
    r2 = 1 - ((z.P - pred) ** 2).sum() / ((z.P - z.P.mean()) ** 2).sum()

    LB = ships.length * ships.width
    inside = ((LB >= z.LB.min()) & (LB <= z.LB.max())).mean() * 100
    ships["P_mcr"] = np.polyval(co, LB)

    v = DESIGN_SPEED_KMH * KN_PER_KMH
    d23 = ships.avg_water ** (2 / 3)
    table = {}
    for frac in [1.00, 0.85, 0.75, 0.65]:
        c_kw = d23 * v ** 3 / (ships.P_mcr * frac)
        table[frac] = c_kw
        inside_range = ((c_kw >= 300) & (c_kw <= 400)).mean() * 100

    c_new = table[SERVICE_FRACTION]
    c_old_kw = ships["admiralty_coefficient"] * k


    res = ships[["mmsi", "teu", "length", "width", "avg_draught", "avg_water"]].copy()
    res["C_paper_code"] = ships["admiralty_coefficient"]
    res["C_paper_kw"] = c_old_kw
    res["P_mcr_est"] = ships.P_mcr
    res["C_new_kw"] = c_new
    res["C_new_code"] = c_new / k
    res["ratio"] = c_new / c_old_kw
    res.to_csv(out / "admiralty_pership.csv", index=False)


if __name__ == "__main__":
    main()
