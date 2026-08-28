from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from statistical_tests import (  # noqa: E402
    exact_mcnemar,
    holm_adjust,
    paired_delong,
    poisson_binomial_tail,
    single_delong,
)


def test_holm_enforces_monotonicity() -> None:
    raw = [0.8607443, 0.87567154, 0.96961387]
    adjusted = holm_adjust(raw)
    assert adjusted == [1.0, 1.0, 1.0]


def test_exact_mcnemar_uses_discordant_pairs() -> None:
    result = exact_mcnemar([True, True, True, False], [False, False, True, True])
    assert result["b10"] == 2
    assert result["b01"] == 1
    assert result["discordant"] == 3
    assert result["p"] == 1.0


def test_paired_delong_identical_scores() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.4, 0.3, 0.7, 0.8])
    result = paired_delong(labels, scores, scores)
    assert result["difference"] == 0.0
    assert result["p"] == 1.0
    single = single_delong(labels, scores, 0.5)
    assert np.isclose(single["p_less"] + single["p_greater"], 1.0)
    # The sum alone also passes with the two tails swapped, which is the bug this guards.
    assert single["auc"] > 0.5
    assert single["p_greater"] < 0.5 < single["p_less"]


def test_poisson_binomial_tails() -> None:
    probabilities = [0.5, 0.5]
    assert poisson_binomial_tail(probabilities, 2, "greater") == 0.25
    assert poisson_binomial_tail(probabilities, 0, "less") == 0.25
