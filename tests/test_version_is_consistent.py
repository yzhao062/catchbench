"""The version is written in two places, so hold them equal.

`pyproject.toml` carries the version the wheel is published under, and `catchbench.__version__`
carries the one a user reads at runtime. Nothing else compares them. On 2026-08-22 they had drifted:
the package said `0.0.1` while the distribution on PyPI said `0.1.0`, and neither a test nor a
workflow noticed, because no test referenced `__version__` at all.

A mismatch is quiet in the worst way. Someone reporting a bug quotes the runtime value, and the
maintainer looks at a different release.

Both checks read files in the repository. Comparing against whatever `importlib.metadata` reports
was tried and removed: an editable install records its version once and does not rewrite it on a
bump, so the answer depended on what happened to be installed and on test order rather than on the
repository. A flaky check in a suite whose point is checked numbers is worse than no check.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import catchbench


ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"

_VERSION = re.compile(r'^version\s*=\s*"([^"]+)"', re.M)
# PEP 440 in the shapes this project uses: release, and the pre/post/dev suffixes.
_PEP440 = re.compile(r"^\d+\.\d+(\.\d+)?((a|b|rc)\d+|\.post\d+|\.dev\d+)?$")


@pytest.fixture(scope="module")
def declared() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    # The [project] table is first, so the first match is the distribution version rather than a
    # version pin further down.
    match = _VERSION.search(text)
    assert match, "pyproject.toml declares no version"
    return match.group(1)


def test_the_package_and_the_distribution_agree(declared):
    assert catchbench.__version__ == declared, (
        f"catchbench.__version__ is {catchbench.__version__!r} and pyproject.toml declares "
        f"{declared!r}; a user reporting a bug would quote a release that was never published")


def test_the_version_is_a_release_identifier(declared):
    assert _PEP440.match(declared), f"{declared!r} is not a version pip will sort correctly"
