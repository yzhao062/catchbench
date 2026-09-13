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
import subprocess
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
    for node in _import_time_nodes(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "catchbench" and any(a.name == "_reuse" for a in node.names):
                return True
            if node.module == BRIDGE:
                return True
        elif isinstance(node, ast.Import):
            if any(a.name == BRIDGE for a in node.names):
                return True
    return False


def _import_time_nodes(tree: ast.AST):
    """Yield the nodes that execute when the module is imported.

    ``ast.walk`` descends into function bodies, so it cannot tell an import that runs on import from
    one that runs when somebody calls a function. ``cli`` defers its bridge import into ``main``
    inside a try that reports the missing checkout and exits, so importing it offline succeeds, and
    a whole-tree walk reads that as a module-level dependency and is wrong.

    Class bodies execute at import time and are therefore still followed. Function and lambda bodies
    are not.

    The contract is deliberately narrow and this rule has two known gaps, so it is a screen rather
    than the evidence. It misses an import that a module-level call reaches::

        def load():
            import catchbench._reuse
        load()

    and it over-reports imports under ``if TYPE_CHECKING`` or ``if __name__ == "__main__"``, which
    do not run on an ordinary import. The first gap has no syntactic fix: deciding whether a call
    reaches a bridge import is the general problem. ``test_every_registered_module_really_imports_
    offline`` is the authoritative check, because it imports the modules instead of reading them.
    """
    stack = [tree]
    while stack:
        node = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            yield child
            stack.append(child)


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


@pytest.mark.parametrize("body,expected", [
    ("import catchbench._reuse", True),
    ("class C:\n    import catchbench._reuse", True),
    ("def f():\n    import catchbench._reuse", False),
    ("if TYPE_CHECKING:\n    import catchbench._reuse", True),
], ids=["module-level", "class-body", "function-body", "type-checking-block"])
def test_the_import_time_rule_is_the_documented_one(body, expected, tmp_path):
    """Pin the screen's rule, including the case it knowingly over-reports.

    The type-checking case is asserted as True to record present behaviour rather than to endorse
    it: that import does not run, and the docstring says so. Writing it down means a later change
    to it is a decision rather than an accident.
    """
    source = tmp_path / "probe.py"
    source.write_text(body + "\n", encoding="utf-8")
    assert _imports_the_bridge(source) is expected


_BLOCKED_BRIDGE_CHILD = """
import importlib, os, pathlib, sys

SRC = %(src)r
GRADE_TOP = %(grade)r

# Bind the source under review. Without this the child can satisfy "import catchbench" from an
# installed wheel and report that a modified working tree is fine.
sys.path.insert(0, SRC)
os.environ.pop("PYTHONPATH", None)


# Refuse the GRADE top-level names however they would otherwise be found. GRADE_DIR cannot do this
# job: _resolve_grade checks the installed module locations first and returns before reading it, so
# an empty GRADE_DIR leaves an installed GRADE fully reachable.
class _RefuseGrade:

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in GRADE_TOP:
            raise ImportError("blocked for this test: " + name)
        return None


sys.meta_path.insert(0, _RefuseGrade())

# Prove the block before trusting it. If the bridge still loads, every assertion after this would
# pass for the wrong reason.
try:
    importlib.import_module("catchbench._reuse")
except Exception as exc:
    if type(exc).__name__ != "MissingGradeBridge":
        raise SystemExit("bridge failed for an unexpected reason: %%r" %% (exc,))
else:
    raise SystemExit("the GRADE bridge loaded despite the block; this test proves nothing")

for name in %(offline)r:
    module = importlib.import_module(name)
    where = pathlib.Path(module.__file__).resolve()
    if SRC not in [str(p) for p in where.parents]:
        raise SystemExit("%%s resolved to %%s, outside the source under review" %% (name, where))
"""


def test_every_registered_module_really_imports_offline(tmp_path):
    """Import each module the registry calls offline-safe, with GRADE blocked at the finder.

    The static screen cannot see an import reached through a module-level call, so this imports the
    modules rather than reading them. Two things it must get right, both of which it got wrong
    first. An empty ``GRADE_DIR`` blocks nothing, because ``_resolve_grade`` returns on the installed
    module locations before reading it, so the block is a meta-path finder and the child proves the
    block works before testing anything with it. And the child must import the source under review
    rather than whichever CatchBench happens to be installed, so it binds ``src`` and checks where
    each module resolved.
    """
    offline = [f"catchbench.{p.stem}" for p in _package_modules()
               if f"catchbench.{p.stem}" not in ci_smoke.SKIPPED]
    assert "catchbench.cli" in offline, "the CLI must stay in the offline import walk"

    program = _BLOCKED_BRIDGE_CHILD % {
        "src": str((ROOT / "src").resolve()),
        "grade": tuple(_reuse_grade_names()),
        "offline": tuple(offline),
    }
    done = subprocess.run([sys.executable, "-I", "-c", program], capture_output=True, text=True,
                          cwd=str(tmp_path), timeout=300)
    assert done.returncode == 0, (
        "the offline import walk failed:\n" + (done.stderr or done.stdout)[-2000:])


def _reuse_grade_names() -> tuple[str, ...]:
    """The GRADE top-level names, read from the resolver rather than restated here."""
    source = (ROOT / "src" / "catchbench" / "_reuse.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_GRADE_MODULES" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("_GRADE_MODULES not found in _reuse.py")
