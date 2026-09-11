r"""The LIVE prefix size control: what the effect is, and what the branch is allowed to read.

Part A of the declaration asks one question at each prefix: how much of the dependency increment
survives once the size reference is allowed to bend. Two things about that question are easy to lose
in a regenerated record, and both are what these tests hold.

The first is which contrast the number is. Four arms are scored at every cell and three contrasts can
be taken between them, so a row that carried the wrong pair, or the right pair on the wrong estimand,
would still look like an effect. The first group below recomputes every shipped point from the
out-of-fold vectors stored beside it in the same record and requires exact equality, and it pins the
declared estimand against the two neighbours it is most often confused with: the mean of the 25 fold
AUCs, which is what the board prints, and the mean of the five per-seed pooled AUCs.

The second is what the branch is entitled to see. A5 nominated ``D(swegym, 0.25)`` before any of the
eight existed and A8 forbids promoting a better result at 50 or 75 percent, so the second group pins
that ``classify_outcome`` moves with the primary cell alone and does not move when another cell is
replaced by a large one. The third group binds the rest of the record to its sources: the frozen RNG
labels as literal text, the interval method each corpus is entitled to, A5's refusal to give the
reproduction quantity an interval, and the A10.4 gate's exemption to the single field the
declaration's dated entry names.

No corpus, GRADE checkout, or network is needed. The record carries its own labels and out-of-fold
probabilities, so every effect here is recomputed from the file rather than refitted.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from catchbench import live_size_audit as audit  # noqa: E402

RECORD = ROOT / "tools" / "live_size_control_audit_results.json"

# A2 names this number in the declaration: the matched arm against the flexible size reference on
# seed-averaged out-of-fold predictions, at the endpoint, on SWE-Gym. It is the one value the batch
# shares with the frozen text, so it is the one binding that catches a route that drifted wholesale.
DECLARED_ENDPOINT_EFFECT = -0.005404
DECLARED_CELLS = tuple((corpus, percent) for corpus in ("swegym", "tau")
                       for percent in (25, 50, 75, 100))


@pytest.fixture(scope="module")
def record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _cell(record, row):
    return record["corpora"][row["corpus"]]["by_prefix"][str(row["prefix_percent"])]


def _oof(cell, arm):
    return np.asarray(cell["arms"][arm]["oof_proba"], dtype=float)


# --- which contrast the number is -----------------------------------------------------------------

def test_every_effect_recomputes_from_the_vectors_stored_beside_it(record):
    high, low = audit.MATCHED_CONTRAST
    for row in record["effects"]:
        cell = _cell(record, row)
        labels = np.asarray(record["corpora"][row["corpus"]]["labels"], dtype=int)
        point = (roc_auc_score(labels, _oof(cell, high).mean(axis=0))
                 - roc_auc_score(labels, _oof(cell, low).mean(axis=0)))
        assert row["point"] == point, row["cell"]
        assert row["high"] == high and row["low"] == low


def test_the_estimand_is_not_the_board_number_and_not_the_mean_of_per_seed_aucs(record):
    """A5 keeps three quantities apart that the same arrays all support. This is that separation.

    On SWE-Gym at 25 percent the published contrast is +0.101 on seed-averaged out-of-fold scores and
    about 0.112 on the displayed board numbers, and taking one for the other is the specific mistake
    A5 was written to prevent. The mean of the five per-seed pooled AUCs is the third neighbour: it
    averages five AUCs instead of scoring one averaged vector, and it is not the estimand either.
    """
    high, low = audit.MATCHED_CONTRAST
    for row in record["effects"]:
        cell = _cell(record, row)
        labels = np.asarray(record["corpora"][row["corpus"]]["labels"], dtype=int)
        board = cell["arms"][high]["mean_fold_roc_auc"] - cell["arms"][low]["mean_fold_roc_auc"]
        assert row["reproduction_quantity"]["mean_fold_difference"] == board
        per_seed = float(np.mean([roc_auc_score(labels, a) - roc_auc_score(labels, b)
                                  for a, b in zip(_oof(cell, high), _oof(cell, low))]))
        assert row["point"] != per_seed


def test_only_the_size_columns_are_splined_in_the_matched_arm(record):
    """A2's boundary, read off the shipped estimator rather than off a claim that it was respected.

    Splining all eight ``flatdep`` columns is a different control and a different number: on the
    committed endpoint records it gives -0.016325 where the matched arm gives -0.005404. The matched
    arm therefore has to be the one that passes a block through, and the flexible size reference has
    to be the one with no passthrough at all.
    """
    high, low = audit.MATCHED_CONTRAST
    for row in record["effects"]:
        cell = _cell(record, row)
        assert cell["arms"][high]["layer"] == "flatdep"
        assert cell["arms"][low]["layer"] == "flat"
        assert "passthrough" in cell["arms"][high]["estimator"]
        assert "passthrough" not in cell["arms"][low]["estimator"]
        assert cell["arms"][high]["n_columns"] == 2 * cell["arms"][low]["n_columns"]


def test_the_endpoint_reproduces_the_number_the_declaration_names(record):
    endpoint = next(row for row in record["effects"] if row["cell"] == "swegym.100")
    assert round(endpoint["point"], 6) == DECLARED_ENDPOINT_EFFECT
    assert endpoint["role"] == "reproduction control"
    assert endpoint["independent_new_evidence"] is False


def test_all_eight_declared_cells_are_present_and_labelled(record):
    rows = {(row["corpus"], row["prefix_percent"]): row for row in record["effects"]}
    assert sorted(rows) == sorted(DECLARED_CELLS)
    for (corpus, percent), row in rows.items():
        assert row["role"] == ("primary" if (corpus, percent) == ("swegym", 25)
                               else "reproduction control" if percent == 100 else "declared cell")
        assert row["independent_new_evidence"] == (percent != 100)


# --- what the branch is allowed to read -----------------------------------------------------------

def test_the_branch_is_the_declared_rule_applied_to_the_primary_cell(record):
    primary = record["primary"]
    assert primary["cell"] == "swegym.25"
    assert (primary["corpus"], primary["prefix"]) == audit.PRIMARY_CELL
    expected = audit.classify_outcome(primary["point"], *primary["interval_95"])
    assert record["outcome"]["branch"] == expected["branch"]
    assert record["outcome"]["primary_cell"] == primary["cell"]
    assert record["outcome"]["substantial_threshold"] == audit.SUBSTANTIAL_EFFECT == 0.03


def test_the_four_branch_names_are_the_declared_ones_and_the_rule_is_support_then_size():
    assert audit.classify_outcome(0.05, 0.01, 0.09)["branch"] == "substantial-positive"
    assert audit.classify_outcome(0.030, 1e-9, 0.06)["branch"] == "substantial-positive"
    assert audit.classify_outcome(0.029, 1e-9, 0.06)["branch"] == "small-positive"
    assert audit.classify_outcome(-0.02, -0.05, -1e-9)["branch"] == "erased-or-reversed"
    # support first: a point over the threshold whose interval contains zero is unresolved, not
    # substantial, however far over the threshold the point sits.
    assert audit.classify_outcome(0.40, -1e-9, 0.90)["branch"] == "unresolved"
    assert audit.classify_outcome(0.0, 0.0, 0.09)["branch"] == "unresolved"


def test_the_branch_reads_the_primary_cell_and_no_other(record):
    """A8's last bullet: if 25 percent disappoints, 50 or 75 percent is not promoted to primary.

    Pinned structurally rather than by hunting for an absent line of code. The shipped outcome
    carries the three numbers the rule was applied to, and they have to be the primary cell's own,
    to the bit. That is what a promoted cell would break: the strongest cell in this record has its
    own point and its own interval, and neither appears in the outcome unless it is the primary.
    """
    primary = record["primary"]
    assert record["outcome"]["point"] == primary["point"]
    assert record["outcome"]["interval_95"] == primary["interval_95"]
    assert record["outcome"]["branch"] == audit.classify_outcome(
        primary["point"], *primary["interval_95"])["branch"]
    strongest = max((row for row in record["effects"] if row["cell"] != primary["cell"]),
                    key=lambda row: row["point"])
    assert strongest["point"] > primary["point"]  # the temptation A8 names actually exists here
    assert record["outcome"]["point"] != strongest["point"]
    assert record["outcome"]["interval_95"] != strongest["interval_95"]


def test_the_temporal_contrast_is_the_difference_of_two_shipped_effects(record):
    temporal = record["temporal_contrast"]
    early = next(row for row in record["effects"] if row["cell"] == "swegym.25")
    late = next(row for row in record["effects"] if row["cell"] == "swegym.100")
    assert temporal["point"] == early["point"] - late["point"]
    assert temporal["components"] == {"D(swegym, 0.25)": early["point"],
                                      "D(swegym, 1.00)": late["point"]}
    assert temporal["rng_label"] == "live_size_audit.swegym.25_minus_100"
    assert temporal["rng_base_seed"] == 20260907
    assert temporal["replicates"] == 10_000
    assert temporal["discarded_single_class_draws"] == 0  # class-stratified: no draw can be one class
    assert temporal["additional_fits"] == 0               # A7: no additional classifier fit
    assert temporal["interval_95"][0] <= temporal["point"] <= temporal["interval_95"][1]


# --- the record bound to its sources ---------------------------------------------------------------

def test_every_rng_label_is_the_frozen_literal(record):
    """A6 and A7 froze the labels as text. Pinned as text, not as ``RNG_LABELS[...]``.

    Comparing the record against the dictionary that produced it would pass on any label the
    dictionary later built, which is the one failure this check exists to catch.
    """
    for row in record["effects"]:
        assert row["rng_label"] == (f"live_size_audit.{row['corpus']}."
                                    f"{row['prefix_percent']}.mixed_minus_spline")
        assert row["rng_base_seed"] == 20260907


def test_each_corpus_carries_the_interval_a6_gives_it(record):
    for row in record["effects"]:
        if row["corpus"] == "swegym":
            assert row["interval_95"] == row["delong"]["interval_95"]
            assert "clustered_interval" not in row
            assert row["crosscheck_stratified_bootstrap"]["role"].startswith("crosscheck only")
        else:
            assert row["interval_95"] == row["clustered_interval"]["interval_95"]
            assert row["interval_95"] != row["delong"]["interval_95"]
            assert row["clustered_interval"]["replicates"] == 10_000
            assert row["clustered_interval"]["n_clusters"] == 165
            assert row["clustered_interval"]["clusters_per_stratum"] == {"airline": 50,
                                                                        "retail": 115}
            assert row["reproduction_diagnostic"]["role"].startswith("reproduction diagnostic")
        assert row["delong_point_agrees_with_the_estimand"]["within_tolerance"]


def test_the_reproduction_quantity_never_carries_an_interval(record):
    # A5: the mean of fold AUCs never receives the interval computed for the estimand. The way to
    # obey that in an artifact is to store it where an interval visibly is not.
    for row in record["effects"]:
        assert row["reproduction_quantity"]["interval"] is None


def test_every_arm_at_every_cell_is_declared_and_none_is_a_board_entrant(record):
    for corpus, corpus_record in record["corpora"].items():
        assert sorted(corpus_record["by_prefix"]) == ["100", "25", "50", "75"]
        for cell in corpus_record["by_prefix"].values():
            assert sorted(cell["arms"]) == sorted(audit.ARM_ROLES)
            for arm_id, arm in cell["arms"].items():
                assert arm["board_entrant"] is False
                assert arm["role"] == audit.ARM_ROLES[arm_id]
                assert arm["nonfinite_fold_scores"] == 0
                assert arm["nonfinite_oof_values"] == 0


def test_the_folds_are_five_seeds_of_five_and_every_row_is_held_out_once_per_seed(record):
    for corpus_record in record["corpora"].values():
        n_rows = corpus_record["n_runs"]
        membership = np.asarray(corpus_record["fold_membership"], dtype=int)
        assert membership.shape == (5, n_rows)
        for row in membership:
            counts = np.bincount(row, minlength=5)
            assert len(counts) == 5 and counts.sum() == n_rows
        for cell in corpus_record["by_prefix"].values():
            for arm in cell["arms"].values():
                assert np.asarray(arm["oof_proba"]).shape == (5, n_rows)
                assert np.asarray(arm["fold_roc_auc"]).shape == (5, 5)


def test_the_population_is_the_one_a3_declared(record):
    for corpus, corpus_record in record["corpora"].items():
        assert corpus_record["n_runs"] == audit.DECLARED_POPULATION[corpus]
        assert 0 < corpus_record["n_failed"] < corpus_record["n_runs"]


def test_every_spline_bearing_fit_was_inspected_inside_its_own_fold(record):
    spline_arms = {"size (spline)", "size-spline + linear-deps"}
    for corpus_record in record["verification"]["per_corpus"].values():
        locality = corpus_record["training_only_transforms"]
        assert sorted(locality) == ["100", "25", "50", "75"]
        for prefix_record in locality.values():
            assert set(prefix_record["arms_inspected"]) == spline_arms
            assert prefix_record["fits_inspected"] == 50
            for arm in prefix_record["per_arm"].values():
                assert arm["fits"] == 25
                assert arm["fits_whose_knots_differ_from_the_whole_matrix"] > 0


def test_the_a10_4_gate_exempts_only_the_field_the_dated_entry_names(record):
    gate = record["preflight"]["gate"]
    assert gate["blocking_disagreements"] == []
    assert gate["admits_the_batch"] is True
    exempt = gate["exempted_field"]
    assert exempt["source"] == "statistical_tests_results.json"
    assert exempt["quantity"] == "mean_fold_roc_auc"
    assert exempt["field"] == "variance_axes.board_point_estimate.value"
    for row in gate["record_defects_carried_forward"]:
        assert row["source"] == exempt["source"] and row["quantity"] == exempt["quantity"]
        assert row["corpus"] == "swegym" and row["prefix"] == "25"
    # the exemption is a narrower reading, not a widened tolerance
    assert record["declaration"]["frozen_before_the_run"]["parity_tolerance"] == 1e-12
    assert audit.PARITY_ATOL == 1e-12


def test_the_record_declares_the_quantities_this_module_still_carries(record):
    frozen = record["declaration"]["frozen_before_the_run"]
    assert tuple(frozen["prefixes"]) == tuple(audit.AUDIT_PREFIXES) == (0.25, 0.5, 0.75, 1.0)
    assert frozen["declared_population"] == audit.DECLARED_POPULATION
    assert frozen["arms"] == audit.ARM_ROLES
    assert frozen["matched_contrast"] == list(audit.MATCHED_CONTRAST)
    assert frozen["primary_cell"] == {"corpus": "swegym", "prefix": 0.25}
    assert frozen["temporal_rng_label"] == audit.TEMPORAL_RNG_LABEL
    assert frozen["rng_base_seed"] == audit.BOOTSTRAP_BASE_SEED == 20260907
    assert frozen["bootstrap_draws"] == audit.BOOTSTRAP_DRAWS == 10_000
    assert frozen["substantial_threshold"] == audit.SUBSTANTIAL_EFFECT
    assert record["settings"]["declared_fits"] == 800  # 4 arms x 2 corpora x 4 prefixes x 5 x 5
