"""The golden comparison tolerates a torch row moving, and nothing else.

The board is compared against a committed golden byte for byte. The first CI run of that comparison
found the golden, generated on Windows, disagreeing with ubuntu-latest on exactly two lines:

    g-safeguard (sup GNN)   0.828 -> 0.829
    pygod-anomalydae        0.490 -> 0.487

A second, independent CI run reproduced the same two values, which is why the difference was first
recorded as stable between platforms rather than random between runs. A scheduled run on 2026-09-08
withdrew that reading by producing a third value, pygod-anomalydae 0.485, for the same row on the
same commit. It also showed how narrow the movement is: of 274 board lines it changed exactly one.
The cause is the float kernels underneath torch. The paper already reports 0.824 +/- 0.007 for the
larger of the two over five seeds, so a movement of 0.001 is inside the uncertainty it publishes.

A tolerance is a hole in a check, so this file bounds it four ways: the exact values CI produced
pass, a difference sitting exactly on the documented bound passes, a torch row that moves further
than the tolerance fails, and rows that no produced board has ever shown moving are compared
exactly, as is any non-torch row that moves by one digit in the last place. Without the last two the
tolerance would quietly widen to the whole board, which is most of the check's value, because the
rule methods are exact by construction.
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
    """Only the numbers may move. A renamed method is a different method.

    Both lines name a tolerated row, so this reaches the label comparison rather than stopping at
    the is-this-row-covered guard, which is what the test is for.
    """
    assert not check_board.within_neural_tolerance(
        "  pygod-anomalydae          0.490", "  g-safeguard (sup GNN)     0.491")


def test_a_board_that_gained_a_line_is_never_reconciled(golden_lines):
    """A structural change is not a float-kernel difference, whatever the numbers do."""
    produced = list(golden_lines) + ["  pygod-extra               0.500"]
    merged, got, folded = check_board.reconcile_neural_rows(golden_lines, produced)
    assert folded == 0 and merged is golden_lines or merged == golden_lines


def test_the_tolerance_is_smaller_than_the_published_seed_variance():
    """0.005 is a claim about the paper, so it fails here if someone widens it past that claim."""
    assert check_board.NEURAL_TOLERANCE < 0.007


def test_a_difference_sitting_exactly_on_the_bound_is_tolerated():
    """The regression this covers rejected a run for being 4.3e-18 over its own documented bound.

    A scheduled CI job produced pygod-anomalydae 0.485 against a golden 0.490. That difference is
    exactly NEURAL_TOLERANCE, so it has to pass, but in binary floating point the subtraction is
    0.0050000000000000044 and the comparison failed. The check now subtracts decimals, which is what
    the board prints.
    """
    assert check_board.within_neural_tolerance(
        "  pygod-anomalydae          0.490", "  pygod-anomalydae          0.485")


def test_the_bound_is_exact_at_several_magnitudes():
    """One passing pair could be luck in binary. These differ by exactly the bound as well."""
    for want, got in (("0.500", "0.495"), ("0.300", "0.295"), ("0.828", "0.823")):
        assert check_board.within_neural_tolerance(
            "  pygod-anomalydae          %s" % want,
            "  pygod-anomalydae          %s" % got), (want, got)


def test_one_printed_step_past_the_bound_still_fails():
    """The control in the red direction: fixing the boundary must not widen the tolerance."""
    assert not check_board.within_neural_tolerance(
        "  pygod-anomalydae          0.490", "  pygod-anomalydae          0.484")


def test_a_torch_row_never_observed_to_move_is_compared_exactly():
    """The narrowing itself, in the red direction.

    pygod-gaan, pygod-conad and the guardian and pygod family rows are scored by torch too, but no
    produced board has ever shown them move. They were tolerated by family prefix, which meant a real
    regression on any of them would have been folded into agreement. They are exact again.
    """
    for label in ("pygod-gaan               ", "pygod-conad              ",
                  "guardian (recon-AE)      ", "pygod (graph AD)         "):
        assert not check_board.within_neural_tolerance(
            "  %s 0.490" % label, "  %s 0.489" % label), label


def test_the_tolerated_rows_are_the_ones_the_paper_names():
    """The paper states the exception as two torch-backed cells. This is that sentence, mechanically.

    It fails if someone re-widens the list to families without also changing 08_statements.tex.
    """
    assert len(check_board.TORCH_ROW_PREFIXES) == 2
    assert set(check_board.TORCH_ROW_PREFIXES) == {"g-safeguard (sup GNN)", "pygod-anomalydae"}
