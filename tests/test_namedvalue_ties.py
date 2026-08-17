"""The Gold v2 evaluator must score ties in expectation, like the Gold v1 evaluator.

The two substrates keep separate implementations, because v1 reports a Top-1/Top-3/MRR triple while
v2 reports Top-1 alone. That makes drift possible, so each side carries a test pinning the same
property: a control that assigns one constant score to every candidate lands on the analytic random
floor, rather than taking the first pool slot and reading as a leak. `tests/test_gold_ties.py` is
the v1 half of this pair.
"""
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from tools import namedvalue_admissibility as adm
except ImportError as exc:  # pragma: no cover - exercised only without a GRADE checkout
    pytest.skip(f"needs a GRADE checkout: {exc}", allow_module_level=True)


@pytest.mark.parametrize("pool_size,injected_at", [(1, 0), (2, 0), (2, 1), (5, 3), (9, 8)])
def test_constant_scores_land_on_the_analytic_floor(pool_size, injected_at):
    """A k-way tie containing the target is worth 1/k, wherever the target sits in the pool."""
    assert adm._tie_aware_top1([0.0] * pool_size, injected_at) == pytest.approx(1.0 / pool_size)


def test_position_in_the_pool_does_not_change_the_score():
    """Order sensitivity was the defect. The same tie must score the same from any slot."""
    scores = [adm._tie_aware_top1([7.5] * 4, i) for i in range(4)]
    assert scores == [pytest.approx(0.25)] * 4


def test_a_strict_winner_still_scores_one_and_a_strict_loser_zero():
    """Expectation scoring must not blunt a real ranking."""
    assert adm._tie_aware_top1([9.0, 1.0, 0.5], 0) == pytest.approx(1.0)
    assert adm._tie_aware_top1([9.0, 1.0, 0.5], 2) == pytest.approx(0.0)


def test_a_tie_for_the_lead_that_excludes_the_target_scores_zero():
    assert adm._tie_aware_top1([4.0, 4.0, 1.0], 2) == pytest.approx(0.0)


def test_a_tie_below_the_leader_scores_zero_not_a_share():
    """Only a tie that reaches rank 1 can earn Top-1 credit."""
    assert adm._tie_aware_top1([9.0, 4.0, 4.0], 1) == pytest.approx(0.0)
