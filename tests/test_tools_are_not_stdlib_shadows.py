"""No file in tools/ may take a standard-library module's name.

Running `python tools/<script>.py` puts `tools/` at `sys.path[0]`, ahead of the standard library.
A file named `tools/ast.py` therefore becomes the `ast` that every tools script imports, including
`tools/check_test_report.py`, which is the one script whose job is to say whether the suite really
ran. The collision needs no bad intent: `types.py`, `random.py`, `statistics.py`, and `json.py` are
all names a tools directory reaches for on its own.

The obvious fix, `-P` or `PYTHONSAFEPATH=1`, needs Python 3.11, and the CI matrix still carries
3.10. So the hazard is checked here instead, where it holds on every version the project supports.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _tools_modules() -> list[str]:
    return sorted(
        p.stem for p in TOOLS.glob("*.py") if not p.name.startswith("_")
    )


def test_tools_dir_exists_and_is_populated():
    assert TOOLS.is_dir(), TOOLS
    assert _tools_modules(), "no tools scripts found; the shadowing check would pass vacuously"


@pytest.mark.parametrize("name", _tools_modules())
def test_tool_name_does_not_shadow_a_stdlib_module(name):
    assert name not in sys.stdlib_module_names, (
        "tools/%s.py shadows the standard-library module %r for every script run as "
        "`python tools/<script>.py`, because that puts tools/ at sys.path[0]. Rename it."
        % (name, name)
    )


def test_the_check_test_report_imports_are_covered():
    """The names the report checker imports are the ones a shadow would hurt most."""
    source = (TOOLS / "check_test_report.py").read_text(encoding="utf-8")
    for imported in ("argparse", "ast", "re", "xml"):
        assert imported in source, imported
        assert imported in sys.stdlib_module_names, imported
        assert not (TOOLS / (imported + ".py")).exists(), imported
