
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

SUFFIX = ".prov.json"
_CACHE: dict[tuple[str, int], str] = {}


def _sha(path: Path) -> str:
    path = Path(path)
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    hit = _CACHE.get(key)
    if hit:
        return hit
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    _CACHE[key] = h.hexdigest()
    return _CACHE[key]


def _norm(path) -> Path:
    return Path(os.path.abspath(path))


def _root() -> Path:
    return _norm(Path(__file__).parents[2])


def _resolve(rel: str, product: Path) -> Path | None:
    """Read explicit product-relative paths and older project-relative entries."""
    beside_product = _norm(product).parent / rel
    if rel.startswith("./"):
        return beside_product if beside_product.exists() else None
    for candidate in (_root() / rel, beside_product):
        if candidate.exists():
            return candidate
    return None


def _rel(path: Path) -> str:
    """Project-relative POSIX path, so sidecars are machine-independent."""
    root = _root()
    p = _norm(path)
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.name


def _input_key(path: Path, product: Path) -> str:
    p = _norm(path)
    root = _root()
    if p.is_relative_to(root):
        return p.relative_to(root).as_posix()
    return "./" + Path(os.path.relpath(p, _norm(product).parent)).as_posix()


def sidecar(product: Path) -> Path:
    return Path(str(product) + SUFFIX)


def record(product: Path, inputs: list, script: str | Path,
           params_used: dict | None = None) -> Path:
    """Stamp a freshly written product with the bytes it was built from."""
    product = Path(product)
    if not product.exists():
        raise FileNotFoundError(f"nothing to stamp at {product}")
    missing = [str(p) for p in inputs if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"declared input(s) absent: {missing}")

    doc = {
        "product": _rel(product),
        "product_sha256": _sha(product),
        "script": _rel(Path(script)),
        "inputs": {_input_key(Path(p), product): _sha(Path(p)) for p in sorted(map(str, inputs))},
    }
    if params_used is not None:
        doc["params"] = params_used
    out = sidecar(product)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2,
                              sort_keys=True) + "\n", encoding="utf-8")
    return out


def check(product: Path) -> list[str]:
    """Complaints about a product's freshness; empty means it is current."""
    product = Path(product)
    side = sidecar(product)
    if not side.exists():
        return []
    doc = json.loads(side.read_text(encoding="utf-8"))
    bad = []
    if product.exists() and _sha(product) != doc.get("product_sha256"):
        bad.append(f"{product.name} has been edited by hand")
    for rel, sha in (doc.get("inputs") or {}).items():
        p = _resolve(rel, product)
        if p is None:
            bad.append(f"upstream {rel} no longer present")
        elif _sha(p) != sha:
            bad.append(f"upstream {rel} updated, and {product.name} still the old one")
    return bad


def require(*products: Path) -> None:
    """Stop the run if any product was built from bytes that have since moved."""
    bad = [c for p in products for c in check(p)]
    if bad:
        pass
