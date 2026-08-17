"""Tie-handling regression tests for the Gold localization boards."""
import re

import numpy as np
import pytest

try:
    from auditablebench import gold
except ImportError as exc:  # pragma: no cover - exercised only without a GRADE checkout
    pytest.skip(f"needs a GRADE checkout: {exc}", allow_module_level=True)


class _SyntheticGoldTask:
    groups = np.array([0, 0, 0, 0, 0, 1, 1])
    injected_row = {0: 0, 1: 5}
    targets = [0, 0]
    kinds = ["stale", "dropped"]

    def setup(self):
        pass


class _ConstantMethod:
    method_id = "constant"

    def scores(self, task):
        return [np.zeros(5), np.zeros(2)]


def _harmonic_floor(size):
    return (
        1.0 / size,
        min(3, size) / size,
        sum(1.0 / rank for rank in range(1, size + 1)) / size,
    )


def _reported_metrics(report, label="constant"):
    line = next(line for line in report.splitlines() if line.strip().startswith(label))
    return [tuple(float(value) for value in triple.split("/"))
            for triple in re.findall(r"\d+\.\d+/\d+\.\d+/\d+\.\d+", line)]


def test_tie_straddling_rank_three_gets_fractional_credit():
    """A tie that spans the Top-3 boundary must earn partial Top-3 and MRR credit.

    The constant-score cases below never place an item strictly above the tie, so they would not
    catch logic that ignores the count of strictly-higher scores. Here one item is strictly higher
    and a four-way tie occupies ranks 2 through 5: Top-1 is 0, Top-3 credit is the 2 of 4 tied ranks
    that land at 3 or better, and MRR is the mean of 1/2, 1/3, 1/4, 1/5.
    """
    scores = np.array([9.0, 7.0, 7.0, 7.0, 7.0, 0.0])
    expected = (0.0, 0.5, float(np.mean([1 / 2, 1 / 3, 1 / 4, 1 / 5])))

    np.testing.assert_allclose(gold._tie_aware(scores, 3), expected)


def test_constant_scores_hit_analytic_random_floor_in_full_and_matched_pools(monkeypatch):
    task = _SyntheticGoldTask()
    method = _ConstantMethod()

    full = _reported_metrics(gold.gold_breakdown(task, [method]))
    full_expected = [
        tuple(np.mean(values) for values in zip(_harmonic_floor(5), _harmonic_floor(2))),
        _harmonic_floor(5),
        _harmonic_floor(2),
    ]

    pools = {0: [0, 1], 1: [0]}
    monkeypatch.setattr(gold, "_matched_pools", lambda unused: pools)
    matched = _reported_metrics(gold.gold_matched_breakdown(task, [method]))
    matched_expected = [
        tuple(np.mean(values) for values in zip(_harmonic_floor(2), _harmonic_floor(1))),
        _harmonic_floor(2),
        _harmonic_floor(1),
    ]

    np.testing.assert_allclose(full, full_expected, atol=5e-4)
    np.testing.assert_allclose(matched, matched_expected, atol=5e-4)
