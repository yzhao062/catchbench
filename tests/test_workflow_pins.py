"""The two workflows pin the same third-party versions, so hold them equal.

verify.yml and board.yml each install torch and each clone GRADE at a commit, and both pins are
written out in full in both files. Nothing else compares them. A board scored against one torch
build while the graph contract tests run against another is a difference that produces no error and
that nobody would think to look for, and a GRADE bridge at two different commits means the corpus
loaders differ between the job that checks the board and the job that checks the tests.

The pins are duplicated rather than centralized because a GitHub Actions workflow cannot import a
value from another workflow. A repository variable could hold them, but it moves the pin out of the
diff a reviewer reads and into a settings page nobody opens. Keeping the duplication visible and
asserting the equality here is the trade this file makes.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"
VERIFY = WORKFLOWS / "verify.yml"
BOARD = WORKFLOWS / "board.yml"

_GRADE_CLONE = re.compile(r"grade\.git\s")
_GRADE_PIN = re.compile(r"git -C \S+ checkout --quiet ([0-9a-f]{40})")
_TORCH_PIN = re.compile(r'TORCH_VERSION:\s*"([^"]+)"')
_TORCH_INSTALL = re.compile(r'torch==\$\{TORCH_VERSION\}')
_PYG_INDEX = re.compile(r'data\.pyg\.org/whl/torch-\$\{TORCH_VERSION\}\+cpu\.html')


@pytest.fixture(scope="module")
def texts() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in (VERIFY, BOARD)}


def test_both_workflows_exist(texts):
    assert set(texts) == {"verify.yml", "board.yml"}


def test_the_grade_commit_pin_is_the_same_in_both(texts):
    pins = {name: set(_GRADE_PIN.findall(text)) for name, text in texts.items()}
    for name, found in pins.items():
        assert len(found) == 1, f"{name} pins GRADE at {sorted(found) or 'nothing'}"
    assert pins["verify.yml"] == pins["board.yml"], (
        f"verify.yml clones GRADE at {pins['verify.yml']} and board.yml at {pins['board.yml']}; "
        "the two jobs would load the corpora through different code")


def test_the_torch_version_pin_is_the_same_in_both(texts):
    pins = {name: set(_TORCH_PIN.findall(text)) for name, text in texts.items()}
    for name, found in pins.items():
        assert len(found) == 1, f"{name} declares TORCH_VERSION as {sorted(found) or 'nothing'}"
    assert pins["verify.yml"] == pins["board.yml"], (
        f"verify.yml pins torch {pins['verify.yml']} and board.yml pins {pins['board.yml']}; "
        "the board would be scored against a different build than the contract tests use")


@pytest.mark.parametrize("name", ["verify.yml", "board.yml"])
def test_the_pyg_index_is_derived_from_the_pin_rather_than_written_again(texts, name):
    """A literal version in the index URL is how the two halves drift apart within one file."""
    text = texts[name]
    assert _TORCH_INSTALL.search(text), f"{name} should install torch=={{TORCH_VERSION}}"
    assert _PYG_INDEX.search(text), (
        f"{name} should point pyg_lib at the index for ${{TORCH_VERSION}}+cpu rather than at a "
        "version written out a second time")


def test_board_does_not_ask_for_a_runner_that_has_to_be_registered(texts):
    """The reason this file exists: board.yml queued forever against a runner nobody had.

    A self-hosted label is a legitimate choice, but it is one that fails silently. The job neither
    passes nor fails; it waits, and the workflow's status never reaches the commit. If this pin is
    ever moved back to self-hosted, that has to be a decision someone makes with a runner already
    registered, rather than a line that looks like every other runs-on.
    """
    assert "self-hosted" not in texts["board.yml"] or "runs-on: ubuntu" in texts["board.yml"], (
        "board.yml asks for a self-hosted runner; confirm one is registered for this repository "
        "before merging, because an unregistered label makes the job queue silently forever")
