"""The golden comparison tolerates a torch row moving, and nothing else.

The board is compared against a committed golden byte for byte. The first CI run of that comparison
found the golden, generated on Windows, disagreeing with ubuntu-latest on exactly two lines:

    g-safeguard (sup GNN)   0.828 -> 0.829
    pygod-anomalydae        0.490 -> 0.487

A second, independent CI run reproduced the same two values, so the difference is stable between
platforms rather than random between runs, and it comes from the float kernels underneath torch. The
paper already reports 0.824 +/- 0.007 for the larger of the two over five seeds, so a movement of
0.001 is inside the uncertainty it publishes.

A tolerance is a hole in a check, so this file bounds it: the exact values CI produced pass, a torch
row that moves further than the tolerance fails, and a non-torch row that moves by one digit in the
last place fails. Without the third of those the tolerance could quietly widen to the whole board,
which is most of the check's value, because the rule methods are exact by construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "golden" / "board.txt"
sys.path.insert(0, str(ROOT / "tools"))

import check_board  # noqa: E402


# The two lines CI moved, verbatim from tests/golden/board.txt and from the CI log.
_CI_MOVED = (
    ("g-safeguard (sup GNN)     0.828", "g-safeguard (sup GNN)     0.829"),
    ("pygod-anomalydae          0.490", "pygod-anomalydae          0.487"),
)


@pytest.fixture(scope="module")
def golden_lines() -> list[str]:
    return check_board.normalize(GOLDEN.read_text(encoding="utf-8"))


def _replace(lines: list[str], old: str, new: str) -> list[str]:
    hits = [i for i, line in enumerate(lines) if line.strip() == old.strip()]
    assert len(hits) == 1, f"expected one {old!r} row in the golden, found {len(hits)}"
    out = list(lines)
    out[hits[0]] = out[hits[0]].replace(old.strip(), new.strip())
    return out


def test_the_two_rows_ci_moved_are_still_in_the_golden(golden_lines):
    """If a row is renamed or dropped, this file's premises need rereading before its assertions."""
    for old, _ in _CI_MOVED:
        assert any(line.strip() == old.strip() for line in golden_lines), old


@pytest.mark.parametrize("old,new", _CI_MOVED)
def test_the_exact_values_ci_produced_reconcile(golden_lines, old, new):
    produced = _replace(golden_lines, old, new)
    merged, got, folded = check_board.reconcile_neural_rows(golden_lines, produced)
    assert merged == got, "the value CI produces should not be reported as drift"
    assert folded == 1


def test_a_torch_row_beyond_the_tolerance_still_fails(golden_lines):
    old = "g-safeguard (sup GNN)     0.828"
    produced = _replace(golden_lines, old, "g-safeguard (sup GNN)     0.798")
    merged, got, folded = check_board.reconcile_neural_rows(golden_lines, produced)
    assert merged != got and folded == 0


def test_a_non_torch_row_fails_on_the_last_digit(golden_lines):
    """The tolerance must not leak onto the rule and size rows, which are exact by construction."""
    old = "size (flat)               0.663"
    produced = _replace(golden_lines, old, "size (flat)               0.664")
    merged, got, folded = check_board.reconcile_neural_rows(golden_lines, produced)
    assert merged != got and folded == 0


def test_a_torch_row_whose_label_changed_is_not_reconciled():
    """Only the numbers may move. A renamed method is a different method."""
    assert not check_board.within_neural_tolerance(
        "  pygod-anomalydae          0.490", "  pygod-dominant            0.491")


def test_a_board_that_gained_a_line_is_never_reconciled(golden_lines):
    """A structural change is not a float-kernel difference, whatever the numbers do."""
    produced = list(golden_lines) + ["  pygod-extra               0.500"]
    merged, got, folded = check_board.reconcile_neural_rows(golden_lines, produced)
    assert folded == 0 and merged is golden_lines or merged == golden_lines


def test_the_tolerance_is_smaller_than_the_published_seed_variance():
    """0.005 is a claim about the paper, so it fails here if someone widens it past that claim."""
    assert check_board.NEURAL_TOLERANCE < 0.007
