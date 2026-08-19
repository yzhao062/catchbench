"""Shared PRE corpus conversion helpers for harvest tools."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (ROOT / "data").resolve()
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from catchbench.pre_static_scanner import derive_spec_features  # noqa: E402


def deidentify_row(row: dict[str, Any]) -> dict[str, Any]:
    """Replace one raw task specification with scanner sufficient statistics."""
    if "task_or_role_spec" not in row:
        if "spec_tokens" in row:
            return dict(row)
        raise KeyError("row has neither task_or_role_spec nor spec_tokens")
    spec = row["task_or_role_spec"]
    if not isinstance(spec, str):
        raise TypeError("task_or_role_spec must be a string")
    capabilities = row.get("declared_capabilities")
    if not isinstance(capabilities, list):
        raise TypeError("declared_capabilities must be a list")
    converted = dict(row)
    del converted["task_or_role_spec"]
    converted.update(derive_spec_features(spec, capabilities))
    return converted


def deidentify_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [deidentify_row(row) for row in rows]


def write_json_rows(rows: list[dict[str, Any]], path: Path, *, sort_keys: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=True, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )


def write_local_prose(rows: list[dict[str, Any]], path: Path | None) -> None:
    """Write an explicit local raw copy, rejecting every path below the repository data directory."""
    if path is None:
        return
    resolved = path.resolve()
    if resolved == DATA_DIR or DATA_DIR in resolved.parents:
        raise ValueError("--retain-prose must point outside the repository data directory")
    write_json_rows(rows, resolved)


def add_retain_prose_argument(parser) -> None:
    parser.add_argument(
        "--retain-prose",
        type=Path,
        metavar="LOCAL_JSON",
        help="Opt in to a raw local copy for judge regeneration. Paths below data/ are rejected.",
    )
