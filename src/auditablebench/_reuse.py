"""Resolve GRADE's experiment modules and the ``auditable`` dependency.

Installed modules take precedence. GRADE's experiment modules are not included in its wheel, so
an editable sibling checkout or ``GRADE_DIR`` is used as a fallback. ``AUDITABLE_DIR`` remains a
development fallback for ``auditable`` when the declared package dependency is unavailable.
Importing this module may add fallback checkout directories to ``sys.path``.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path


_LOGGER = logging.getLogger(__name__)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_GRADE_MODULES = (
    "grade",
    "agent_failure_detection",
    "agent_failure_localization",
    "agent_graph_characterization",
    "agent_graph_swegym",
    "agent_graph_tau_bench",
)


def _module_location(name: str) -> str | None:
    try:
        spec = importlib.util.find_spec(name)
    except (AttributeError, ImportError, ValueError):
        return None
    if spec is None:
        return None
    return spec.origin or next(iter(spec.submodule_search_locations or ()), None)


def _prepend(paths: tuple[Path, ...]) -> None:
    for path in reversed(paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _resolve_grade() -> str:
    locations = [_module_location(name) for name in _GRADE_MODULES]
    if all(locations):
        source = f"installed modules ({locations[0]})"
        _LOGGER.debug("Using GRADE from %s", source)
        return source

    configured = os.environ.get("GRADE_DIR")
    checkout = Path(configured).expanduser() if configured else _REPOSITORY_ROOT.parent / "grade"
    paths = (checkout / "experiment", checkout / "src")
    if all(path.is_dir() for path in paths):
        _prepend(paths)
        if all(_module_location(name) for name in _GRADE_MODULES):
            source = f"checkout bridge ({checkout})"
            _LOGGER.debug("Using GRADE from %s", source)
            return source

    raise ImportError(
        "AuditableBench needs GRADE's experiment modules, which are not distributed in GRADE's "
        "wheel. Clone GRADE next to this repository and install its experiment dependencies:\n"
        "  git clone https://github.com/yzhao062/grade.git ../grade\n"
        "  python -m pip install -e \"../grade[experiments]\"\n"
        "Alternatively, set GRADE_DIR to the GRADE checkout. The checkout must contain "
        "experiment/ and src/."
    )


def _resolve_auditable() -> str:
    location = _module_location("auditable")
    if location:
        source = f"installed package ({location})"
        _LOGGER.debug("Using auditable from %s", source)
        return source

    configured = os.environ.get("AUDITABLE_DIR")
    if configured:
        path = Path(configured).expanduser() / "src"
        if path.is_dir():
            _prepend((path,))
            location = _module_location("auditable")
            if location:
                source = f"checkout bridge ({path.parent})"
                _LOGGER.debug("Using auditable from %s", source)
                return source

    raise ImportError(
        "AuditableBench requires auditable>=0.2.0. Install it with "
        "`python -m pip install \"auditable>=0.2.0\"`, or set AUDITABLE_DIR to a source "
        "checkout containing src/auditable."
    )


SOURCES = {
    "auditable": _resolve_auditable(),
    "grade": _resolve_grade(),
}

# GRADE's localization eval uses a conda BLAS stack that wants this set (matches the seed).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
