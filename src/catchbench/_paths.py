"""Resolve the committed data directories in a source checkout and in an installed wheel.

A checkout keeps the records at ``<repo>/data``, where every tool, test, and CI job reads them. The
wheel carries a copy at ``catchbench/data`` through the force-include in ``pyproject.toml``, so an
installed package finds them beside its own modules rather than at the environment root. Both
layouts are checked here, in that order, so a checkout is never served stale data from a wheel that
happens to be installed in the same environment.

The lookup returns a path rather than raising, because the callers glob it and an empty directory is
their own concern. What must not happen is silently reading the wrong one.
"""
from __future__ import annotations

from pathlib import Path


_PACKAGE_DIR = Path(__file__).resolve().parent
_CHECKOUT_ROOT = _PACKAGE_DIR.parents[1]


def _in_checkout() -> bool:
    """True when this module is the one under <repo>/src/catchbench rather than an installed copy."""
    return (_CHECKOUT_ROOT / "src" / "catchbench").resolve() == _PACKAGE_DIR


def data_root() -> Path:
    """The directory holding the committed data directories, checkout first."""
    if _in_checkout():
        checkout = _CHECKOUT_ROOT / "data"
        if checkout.is_dir():
            return checkout
    return _PACKAGE_DIR / "data"


def data_dir(name: str) -> Path:
    """A committed data directory by name, for example ``pre`` or ``llm_judge``."""
    return data_root() / name


def golden_board() -> Path:
    """The board the comparison targets. Its checkout home is tests/, which no wheel carries."""
    if _in_checkout():
        checkout = _CHECKOUT_ROOT / "tests" / "golden" / "board.txt"
        if checkout.is_file():
            return checkout
    return _PACKAGE_DIR / "data" / "golden" / "board.txt"
