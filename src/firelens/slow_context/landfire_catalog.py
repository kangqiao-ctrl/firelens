# src/firelens/slow_context/landfire_catalog.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_LANDFIRE_ROOT = Path("/home/groups/data/landfire")

DEFAULT_VINTAGES = ["LF2016", "LF2020", "LF2022", "LF2023", "LF2024"]

DEFAULT_PRODUCTS = ["cc", "ch", "fbfm40", "fdist"]

PRODUCT_FILENAME_TOKENS = {
    "cc": "CC",
    "ch": "CH",
    "fbfm40": "FBFM40",
    "fdist": "FDist",
}

PRODUCT_TIF_DIR = "Tif"
PRODUCT_CSV_DIR = "CSV_Data"


def _normalize_contains(
    contains: str | Iterable[str] | None,
) -> list[str]:
    if contains is None:
        return []

    if isinstance(contains, str):
        return [contains.lower()]

    return [str(c).lower() for c in contains]


def _has_dir_component(path: Path, dirname: str) -> bool:
    dirname_lower = dirname.lower()
    return any(part.lower() == dirname_lower for part in path.parts)


def _candidate_rank(
    path: Path,
    *,
    root: Path,
    contains_norm: list[str],
    required_dir: str | None,
) -> tuple:
    """
    Deterministic ranking for matched files.

    Preference:
    - file lives under required_dir
    - product token appears in file name, not just parent path
    - shallower path
    - shorter path
    - alphabetical
    """
    rel = path.relative_to(root)
    rel_str = rel.as_posix().lower()
    name_lower = path.name.lower()

    dir_rank = 0
    if required_dir is not None:
        dir_rank = 0 if _has_dir_component(rel, required_dir) else 1

    token_in_name_rank = 0
    if contains_norm:
        token_in_name_rank = 0 if all(c in name_lower for c in contains_norm) else 1

    return (
        dir_rank,
        token_in_name_rank,
        rel_str.count("/"),
        len(rel_str),
        rel_str,
    )


def find_unzipped_file(
    search_root: str | Path,
    *,
    exts: Iterable[str],
    contains: str | Iterable[str] | None = None,
    required_dir: str | None = None,
) -> Path:
    """
    Find one unzipped LANDFIRE file under search_root.

    Parameters
    ----------
    search_root
        Usually root / vintage, for example:
        /scratch/users/qiaok/data/landfire/LF2024
    exts
        File extensions to match, e.g. [".tif", ".tiff"] or [".csv"].
    contains
        Product token(s) that should appear in the relative path.
    required_dir
        Require the file to live somewhere under a directory named this.
        For LANDFIRE products, usually "Tif" or "CSV_Data".
    """
    search_root = Path(search_root)

    if not search_root.exists():
        raise FileNotFoundError(f"Search root does not exist: {search_root}")

    exts_norm = tuple(e.lower() for e in exts)
    contains_norm = _normalize_contains(contains)

    candidates: list[Path] = []

    for path in search_root.rglob("*"):
        if not path.is_file():
            continue

        if not path.name.lower().endswith(exts_norm):
            continue

        rel = path.relative_to(search_root)
        rel_str = rel.as_posix().lower()

        if required_dir is not None and not _has_dir_component(rel, required_dir):
            continue

        if contains_norm and not all(c in rel_str for c in contains_norm):
            continue

        candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No matching file found under {search_root}. "
            f"exts={exts_norm}, contains={contains_norm}, required_dir={required_dir!r}"
        )

    candidates = sorted(
        candidates,
        key=lambda p: _candidate_rank(
            p,
            root=search_root,
            contains_norm=contains_norm,
            required_dir=required_dir,
        ),
    )

    return candidates[0]


def default_product_root_map(
    *,
    root: str | Path = DEFAULT_LANDFIRE_ROOT,
    vintages: Iterable[str] = DEFAULT_VINTAGES,
    products: Iterable[str] = DEFAULT_PRODUCTS,
    check_exists: bool = True,
) -> dict[str, dict[str, str]]:
    """
    Build vintage -> product -> search-root mapping for unzipped LANDFIRE data.

    In the unzipped workflow, each product search root is the vintage directory.
    resolve_product_files() then finds the product-specific TIF and CSV inside
    Tif/ and CSV_Data/ subdirectories.
    """
    root = Path(root)

    product_roots: dict[str, dict[str, str]] = {}

    for vintage in vintages:
        vintage_root = root / vintage

        if check_exists and not vintage_root.exists():
            raise FileNotFoundError(f"Missing LANDFIRE vintage directory: {vintage_root}")

        product_roots[vintage] = {}

        for product in products:
            if product not in PRODUCT_FILENAME_TOKENS:
                raise KeyError(f"Unknown LANDFIRE product: {product}")

            product_roots[vintage][product] = str(vintage_root)

    return product_roots


def resolve_product_files(
    product_key: str,
    search_root: str | Path,
) -> dict[str, str]:
    """
    Resolve product TIFF and CSV paths from an unzipped LANDFIRE vintage directory.

    Expected layout somewhere under search_root:

      Tif/*.tif
      CSV_Data/*.csv
    """
    search_root = Path(search_root)

    token = PRODUCT_FILENAME_TOKENS.get(product_key, product_key)

    tif_path = find_unzipped_file(
        search_root,
        exts=[".tif", ".tiff"],
        contains=token,
        required_dir=PRODUCT_TIF_DIR,
    )

    csv_path = find_unzipped_file(
        search_root,
        exts=[".csv"],
        contains=token,
        required_dir=PRODUCT_CSV_DIR,
    )

    return {
        "root": str(search_root),
        "tif_path": str(tif_path),
        "tif_uri": str(tif_path),   # rasterio can open local path directly
        "csv_path": str(csv_path),
        "csv_uri": str(csv_path),
    }


def read_product_csv(
    product_key: str,
    search_root: str | Path,
) -> pd.DataFrame:
    files = resolve_product_files(product_key, search_root)
    return pd.read_csv(files["csv_path"])