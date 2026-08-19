"""Offline checks for package imports and the committed PRE board."""
from __future__ import annotations

import importlib
import pkgutil
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SKIPPED = {
    "catchbench._reuse": "requires local GRADE and auditable checkout bridges",
    "catchbench.detection": "requires the GRADE checkout bridge",
    "catchbench.gold": "requires the GRADE and auditable checkout bridges",
    "catchbench.live": "requires the GRADE checkout bridge",
    "catchbench.llm_judge": "requires the GRADE checkout bridge",
    "catchbench.namedvalue": "requires the GRADE checkout bridge",
    "catchbench.post": "requires the GRADE and auditable checkout bridges",
}

EXPECTED_COUNTS = {
    "crewai": 298,
    "injecagent": 340,
    "mcp": 144,
    "n8n": 219,
    "sweagent": 130,
    "synthetic": 56,
}


def check_imports() -> None:
    package = importlib.import_module("catchbench")
    names = sorted(m.name for m in pkgutil.iter_modules(package.__path__, package.__name__ + "."))
    unknown_skips = set(SKIPPED) - set(names)
    assert not unknown_skips, f"skip list names missing from package: {sorted(unknown_skips)}"

    imported = []
    for name in names:
        if name not in SKIPPED:
            importlib.import_module(name)
            imported.append(name)

    print("imported: " + ", ".join([package.__name__, *imported]))
    for name in sorted(SKIPPED):
        print(f"skipped: {name} ({SKIPPED[name]})")


def check_pre_board() -> None:
    from catchbench.pre import FlagAllMethod, FlagNoneMethod, PreOverPrivilege

    task = PreOverPrivilege()
    task.setup()
    counts = Counter("synthetic" if row.source.startswith("synth") else row.source
                     for row in task.instances)
    assert len(task.instances) == 1187, f"PRE config count: expected 1187, found {len(task.instances)}"
    assert dict(counts) == EXPECTED_COUNTS, (
        f"PRE corpus split: expected {EXPECTED_COUNTS}, found {dict(counts)}"
    )

    flag_all = dict(FlagAllMethod().evaluate(task))
    flag_none = dict(FlagNoneMethod().evaluate(task))
    assert flag_all["recall"] == 1.0, f"flag_all recall changed: {flag_all}"
    assert flag_all["f1"] == 0.601, f"flag_all F1 changed: {flag_all}"
    assert flag_none == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "coverage": 1.0}, (
        f"flag_none floor changed: {flag_none}"
    )
    # Every method that answers the whole corpus must say so. A silently missing or partial
    # coverage value is how an abstaining method gets compared against a complete one.
    assert flag_all["coverage"] == 1.0, f"flag_all coverage changed: {flag_all}"
    print(f"PRE configs: {len(task.instances)} {dict(counts)}")
    print(f"PRE floors: flag_all={flag_all}, flag_none={flag_none}")


def main() -> None:
    try:
        check_imports()
        check_pre_board()
    except Exception as exc:
        raise SystemExit(f"CI smoke failed: {exc}") from exc
    print("CI smoke passed")


if __name__ == "__main__":
    main()
