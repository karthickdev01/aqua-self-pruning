"""Small helpers for turning experiment results into committed artefacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

__all__ = ["toJsonable", "writeJson", "writeCsv", "readCsv", "ensureDirectory"]


def toJsonable(value: Any) -> Any:
    """Recursively convert NumPy scalars/arrays into plain Python types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): toJsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [toJsonable(item) for item in value]
    return value


def ensureDirectory(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def writeJson(path: Path, payload: Any) -> Path:
    """Write ``payload`` as pretty-printed JSON."""
    ensureDirectory(Path(path).parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(toJsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return Path(path)


def writeCsv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldNames: Optional[Sequence[str]] = None,
) -> Path:
    """Write ``rows`` as CSV with a stable column order."""
    ensureDirectory(Path(path).parent)
    if not rows:
        raise ValueError("writeCsv() requires at least one row.")
    columns = list(fieldNames) if fieldNames is not None else list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: toJsonable(row.get(column)) for column in columns})
    return Path(path)


def readCsv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of dictionaries."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
