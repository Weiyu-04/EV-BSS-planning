
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SUPERSEDED = "9_superseded"

_stages = sorted(d for d in ROOT.iterdir()
                 if d.is_dir() and d.name[:1].isdigit() and d.name != SUPERSEDED)
for _d in reversed([ROOT] + _stages):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

_retired = ROOT / SUPERSEDED
if _retired.is_dir() and str(_retired) not in sys.path:
    sys.path.append(str(_retired))
