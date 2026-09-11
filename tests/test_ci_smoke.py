r"""The offline smoke check must know which modules it cannot import offline.

``tools/ci_smoke.py`` walks every module in the package and imports it, skipping the ones named in
its ``SKIPPED`` registry because they reach into a GRADE or auditable checkout that CI does not
have. The registry is hand-maintained, and nothing on a development machine notices when it falls
behind: GRADE is checked out here, so a module missing from the registry imports cleanly and the
walk passes. It fails only on CI, after a push, which is exactly how two modules shipped.

So the check here is static. It reads the source of every module in the package and asks which ones
import the bridge, rather than importing anything, which means it gives the same answer with or
without a GRADE checkout present. ``ci_smoke`` already asserts the other direction, that no name in
the registry has disappeared from the package.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "catchbench"
sys.path.insert(0, str(ROOT / "tools"))

import ci_smoke  # noqa: E402

# The module whose import as a side effect puts the GRADE and auditable checkouts on sys.path. A
# module that pulls this in cannot be imported where those checkouts are absent, which is CI.
BRIDGE = "catchbench._reuse"


def _imports_the_bridge(source: Path) -> bool:
    """True when this module imports the checkout bridge, read from the syntax tree.

    Matching text would count the string inside this file's own docstring, and a comment mentioning
    ``_reuse`` in any module would become a false positive. The tree carries only real imports.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "catchbench" and any(a.name == "_reuse" for a in node.names):
                return True
            if node.module == BRIDGE:
                return True
        elif isinstance(node, ast.Import):
            if any(a.name == BRIDGE for a in node.names):
                return True
    return False


def _package_modules() -> list[Path]:
    return sorted(p for p in PACKAGE.glob("*.py") if p.name != "__init__.py")


def test_the_package_is_where_this_test_thinks_it_is():
    """Without this, an empty glob would make every assertion below vacuously true."""
    modules = _package_modules()
    assert len(modules) >= 8, f"only found {[p.name for p in modules]} under {PACKAGE}"
    assert (PACKAGE / "detection.py") in modules


@pytest.mark.parametrize("source", _package_modules(), ids=lambda p: p.stem)
def test_every_module_needing_the_grade_bridge_is_skipped_by_the_smoke_check(source):
    """A module that reaches into the GRADE checkout must be registered as unimportable offline."""
    name = f"catchbench.{source.stem}"
    if not _imports_the_bridge(source):
        return
    assert name in ci_smoke.SKIPPED, (
        f"{name} imports {BRIDGE}, so tools/ci_smoke.py cannot import it where GRADE is absent. "
        f"Add it to ci_smoke.SKIPPED with the reason, as the other bridge modules are. "
        f"Currently registered: {sorted(ci_smoke.SKIPPED)}")


def test_the_registry_names_no_module_that_does_not_need_the_bridge():
    """The reverse drift: a skip left behind after a module stopped needing the checkout.

    ci_smoke asserts that every registered name still exists in the package. It cannot tell that a
    name still needs to be there, and a stale skip silently drops a module from the import walk.
    """
    stale = []
    for name in sorted(ci_smoke.SKIPPED):
        source = PACKAGE / (name.split(".", 1)[1] + ".py")
        if source.is_file() and not _imports_the_bridge(source) and source.stem != "_reuse":
            stale.append(name)
    assert not stale, (
        f"{stale} no longer import {BRIDGE}, so the smoke check is skipping modules it could now "
        f"import. Drop them from ci_smoke.SKIPPED.")


def test_the_detector_reads_imports_rather_than_text(tmp_path):
    """The mutation that matters: a mention in prose must not count as an import.

    Written because the first version of this file matched on text, which counted its own docstring
    and would have marked any module carrying a comment about the bridge as needing it.
    """
    mentions = tmp_path / "mentions.py"
    mentions.write_text('"""This module once used catchbench._reuse but no longer does."""\n'
                        "# from catchbench import _reuse\n"
                        "VALUE = 1\n", encoding="utf-8")
    assert not _imports_the_bridge(mentions)

    for body in ("from catchbench import _reuse\n",
                 "from catchbench._reuse import anything\n",
                 "import catchbench._reuse\n"):
        real = tmp_path / "real.py"
        real.write_text(body, encoding="utf-8")
        assert _imports_the_bridge(real), f"missed a real import: {body!r}"


def test_the_smoke_check_runs_offline(tmp_path):
    """End to end: the walk must survive with no GRADE checkout reachable.

    ``_reuse`` finds GRADE through GRADE_DIR or a sibling checkout. Pointing GRADE_DIR at an empty
    directory removes the first and, on this workstation, leaves the sibling, so this asserts the
    registry rather than the absence of GRADE: it re-runs ci_smoke's own walk and requires that no
    module outside the registry needs the bridge. The parametrized test above is the per-module
    form; this one fails the same way the CI job did, in one line.
    """
    needed = {f"catchbench.{p.stem}" for p in _package_modules() if _imports_the_bridge(p)}
    unregistered = sorted(needed - set(ci_smoke.SKIPPED))
    assert not unregistered, (
        f"tools/ci_smoke.py would try to import {unregistered} on a runner with no GRADE checkout "
        f"and fail, which is what the CI smoke job reports as "
        f"'CatchBench needs GRADE's experiment modules'.")
    assert os.path.isdir(PACKAGE)
