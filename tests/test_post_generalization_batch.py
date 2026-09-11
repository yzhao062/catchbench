r"""The POST generalization batch: what it fitted, what it measured, and how it read the result.

Section J gives this analysis one batch of 200 fits and one reporting pass, so the record it wrote
is the only copy of six quantities that will not be recomputed. These tests hold that record to the
four sections that define it, and they hold it by recomputation rather than by restatement wherever
recomputation is possible.

The first group is section E. Each of the six points is recomputed from the out-of-fold vectors the
record stores beside it, so an artifact whose printed effect drifted from its own arms fails here
rather than being read as a result.

The second group is section F. The two ``S`` intervals are regenerated from the stored vectors and
the frozen RNG labels and must come back bit for bit, which is the one property no field of the
record can assert about itself. The resampling unit, the draw count and the discarded-draw count
are held beside them.

The third group is section G. The branch is re-derived by handing the frozen classifier the
record's own six numbers, so the printed branch has to follow from them and not from anything a
reader might have preferred. The traps section G names are held on synthetic readings, where a case
can be constructed rather than waited for.

The fourth group is the gate and the bindings that make a cleared preflight mean anything: the
batch must have fitted the rows, the columns and the 25 partitions the preflight checked, and a
disagreement in any of them must be a blocker rather than a note.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import emit_post_generalization_audit as driver  # noqa: E402
import statistical_tests as st  # noqa: E402

from catchbench import post_generalization_audit as audit  # noqa: E402

RECORD = ROOT / "tools" / "post_generalization_audit_results.json"

CORPORA = ("openhands", "scienceworld")


@pytest.fixture(scope="module")
def record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _rows(record) -> dict:
    """The six effect rows, keyed by ``corpus.estimand``, which is how the record names a cell."""
    return {row["cell"]: row for row in record["effects"]}


# --- section E: the six estimands, recomputed from the vectors they were built from ---------------

def test_every_estimand_recomputes_from_the_out_of_fold_vectors_stored_beside_it(record):
    """The point and the interval have to be reading the same two or four vectors.

    Two arms recorded separately can each be right while the difference is taken between the wrong
    pair or read on the wrong scale. Recomputing the difference from the record's own ``oof_proba``
    arrays is what rules that out, and it is exact: the estimand is a deterministic function of
    arrays the record carries, so an approximate match would mean something moved.
    """
    from sklearn.metrics import roc_auc_score

    for corpus in CORPORA:
        corpus_record = record["corpora"][corpus]
        labels = np.asarray(corpus_record["labels"], dtype=int)
        scored = {name: float(roc_auc_score(labels, driver._seed_averaged(corpus_record, name)))
                  for name in corpus_record["arms"]}
        linear = scored[audit.LINEAR_CONTRAST[0]] - scored[audit.LINEAR_CONTRAST[1]]
        matched = scored[audit.MATCHED_CONTRAST[0]] - scored[audit.MATCHED_CONTRAST[1]]
        expected = {"L": linear, "D": matched, "S": linear - matched}
        for estimand, point in expected.items():
            row = _rows(record)[f"{corpus}.{estimand}"]
            assert row["point"] == point, row["cell"]
            for arm, values in row["arm_scores"].items():
                assert values["seed_averaged_oof_roc_auc"] == scored[arm], (row["cell"], arm)


def test_s_is_l_minus_d_and_not_a_third_number(record):
    """``S = L - D`` is its definition, so the record stores the identity rather than a recompute."""
    for corpus in CORPORA:
        rows = _rows(record)
        assert rows[f"{corpus}.S"]["definition"] == "L - D"
        assert (rows[f"{corpus}.S"]["point"]
                == rows[f"{corpus}.L"]["point"] - rows[f"{corpus}.D"]["point"])
        effects = record["corpora"][corpus]["effects"]
        for scale in ("seed_averaged_oof_difference", "mean_fold_difference"):
            assert effects["S"][scale] == effects["L"][scale] - effects["D"][scale]


def test_the_declared_estimand_is_not_the_board_shaped_number_beside_it(record):
    """Section E puts the estimands on seed-averaged out-of-fold probabilities. The mean of the 25
    fold AUCs is what a board prints, and the two differ; the record keeps the second somewhere an
    interval visibly is not, so a reader cannot take one for the other."""
    for row in record["effects"]:
        reproduction = row["reproduction_quantity"]
        assert reproduction["interval"] is None
        assert reproduction["mean_fold_difference"] != row["point"], row["cell"]
        assert "never receives the interval" in reproduction["why_no_interval"]


def test_every_arm_is_a_declared_arm_in_its_declared_role_and_none_is_a_board_entrant(record):
    for corpus in CORPORA:
        arms = record["corpora"][corpus]["arms"]
        assert sorted(arms) == sorted(audit.ARM_ROLES)
        for arm_id, arm in arms.items():
            assert arm["role"] == audit.ARM_ROLES[arm_id]
            assert arm["board_entrant"] is False
            assert arm["nonfinite_fold_scores"] == 0 and arm["nonfinite_oof_values"] == 0
    assert record["board_entrants_added"] == []


def test_the_record_preserves_everything_section_e_lists(record):
    """Five out-of-fold vectors, labels, ordered row identities, group keys, exact fold membership,
    fold scores, warnings, and source and feature hashes."""
    for corpus in CORPORA:
        corpus_record = record["corpora"][corpus]
        for field in ("labels", "seeds", "fold_membership", "run_identities", "fold_protocol",
                      "resampling_unit", "rng_labels", "feature_hashes", "label_hash", "arms",
                      "effects"):
            assert field in corpus_record, (corpus, field)
        rows = len(corpus_record["labels"])
        assert np.asarray(corpus_record["fold_membership"]).shape == (5, rows)
        for arm in corpus_record["arms"].values():
            assert np.asarray(arm["oof_proba"]).shape == (5, rows)
            assert np.asarray(arm["fold_roc_auc"]).shape == (5, 5)
        # Every layer GRADE builds is hashed, and the two the arms read have to be among them and
        # have to differ: ``flat`` is the leading block of ``flatdep``, so equal hashes would mean
        # the dependency columns are absent and the two contrasts are one contrast.
        hashes = corpus_record["feature_hashes"]
        assert {"flat", "flatdep"} <= set(hashes)
        assert hashes["flat"] != hashes["flatdep"]
        assert len(set(hashes.values())) == len(hashes)
    assert record["corpora"]["openhands"]["run_identities"]["group_key_field"] == "instance_id"


# --- section F: the intervals, regenerated -------------------------------------------------------

def test_both_s_intervals_regenerate_bit_for_bit_from_the_stored_vectors_and_frozen_labels(record):
    """The one property no field of the record can assert about itself.

    ``S`` is the difference of differences, so its draw carries all four score vectors, and a
    regeneration that came back different would mean either the vectors are not the vectors the
    interval was computed from or the frozen label is not the label that seeded it. Both are exactly
    the failures section F's frozen labels exist to make visible. The primary quantity is one of the
    two regenerated here.
    """
    for corpus in CORPORA:
        corpus_record = record["corpora"][corpus]
        labels = np.asarray(corpus_record["labels"], dtype=int)
        units, _ = driver._resampling_units(corpus, corpus_record)
        vectors = [driver._seed_averaged(corpus_record, name)
                   for name in (*audit.LINEAR_CONTRAST, *audit.MATCHED_CONTRAST)]
        row = _rows(record)[f"{corpus}.S"]
        regenerated = driver._unit_bootstrap(
            labels, units, vectors, lambda a: (a[0] - a[1]) - (a[2] - a[3]),
            audit.BOOTSTRAP_DRAWS, st._rng_for(row["rng_label"], audit.BOOTSTRAP_BASE_SEED))
        assert regenerated["interval_95"] == row["interval_95"], corpus
        assert regenerated["usable_replicates"] == row["usable_replicates"]
        assert regenerated["discarded_single_class_draws"] == row["discarded_single_class_draws"]


def test_every_interval_carries_the_frozen_label_the_declared_draws_and_a_discard_count(record):
    """Section F freezes the labels as text, so they are held as text: comparing the record against
    the dictionary that produced them would pass on whatever that dictionary later built."""
    for corpus in CORPORA:
        for estimand in ("L", "D", "S"):
            row = _rows(record)[f"{corpus}.{estimand}"]
            assert row["rng_label"] == f"post_generalization.{corpus}.{estimand}"
            assert row["rng_base_seed"] == audit.BOOTSTRAP_BASE_SEED == 20260907
            assert row["replicates"] == audit.BOOTSTRAP_DRAWS == 10_000
            assert row["discarded_single_class_draws"] >= 0
            assert (row["usable_replicates"] + row["discarded_single_class_draws"]
                    == audit.BOOTSTRAP_DRAWS)
            low, high = row["interval_95"]
            assert low <= row["point"] <= high, row["cell"]
    assert record["settings"]["bootstrap"]["percentiles"] == [0.025, 0.975]


def test_s_carries_all_four_score_vectors_in_every_draw_and_l_and_d_carry_two(record):
    """Section F: ``S`` is a difference of differences, so a draw that carried only two vectors
    would be measuring two contrasts on unrelated resamples and would not describe ``S``."""
    for corpus in CORPORA:
        rows = _rows(record)
        assert rows[f"{corpus}.L"]["arms_carried_in_every_draw"] == list(audit.LINEAR_CONTRAST)
        assert rows[f"{corpus}.D"]["arms_carried_in_every_draw"] == list(audit.MATCHED_CONTRAST)
        assert (rows[f"{corpus}.S"]["arms_carried_in_every_draw"]
                == list(audit.LINEAR_CONTRAST) + list(audit.MATCHED_CONTRAST))


def test_the_resampling_unit_is_the_issue_group_for_openhands_and_the_row_for_scienceworld(record):
    for corpus in CORPORA:
        for estimand in ("L", "D", "S"):
            row = _rows(record)[f"{corpus}.{estimand}"]
            assert row["resampling_unit"] == audit.RESAMPLING_UNIT[corpus]
    identities = record["corpora"]["openhands"]["run_identities"]
    assert _rows(record)["openhands.S"]["n_units"] == identities["n_distinct_group"]
    assert (_rows(record)["scienceworld.S"]["n_units"]
            == len(record["corpora"]["scienceworld"]["labels"]))


def test_a_unit_covers_every_row_exactly_once_and_never_splits_an_issue(record):
    """The grouped unit is what carries every attempt at an issue into a draw together. Read off the
    record's own group keys rather than off the count the record reports."""
    for corpus in CORPORA:
        corpus_record = record["corpora"][corpus]
        units, name = driver._resampling_units(corpus, corpus_record)
        covered = np.concatenate(units)
        assert sorted(covered.tolist()) == list(range(len(corpus_record["labels"])))
        assert name == audit.RESAMPLING_UNIT[corpus]
    keys = np.asarray(record["corpora"]["openhands"]["run_identities"]["group_key"])
    units, _ = driver._resampling_units("openhands", record["corpora"]["openhands"])
    for unit in units:
        assert len(set(keys[unit].tolist())) == 1
    assert sum(len(unit) for unit in units) == len(keys)


def test_a_single_class_draw_is_discarded_and_counted_rather_than_scored():
    """Section F says single-class draws are discarded AND counted, which is a specification of an
    unstratified draw: a class-stratified one cannot produce a single-class resample at all.

    Two units, one per class, so a draw picking the same unit twice is single-class and has no
    ROC-AUC. With four rows the discard rate is high enough that the counter cannot stay at zero.
    """
    labels = np.array([0, 0, 1, 1])
    units = [np.array([0, 1]), np.array([2, 3])]
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    result = driver._unit_bootstrap(labels, units, [scores], lambda a: a[0], 200,
                                    np.random.default_rng(0))
    assert result["discarded_single_class_draws"] > 0
    assert result["usable_replicates"] + result["discarded_single_class_draws"] == 200
    assert result["interval_95"] == [1.0, 1.0]


# --- section G: the branch, from the record's own numbers ----------------------------------------

def test_the_branch_is_the_frozen_rule_applied_to_both_corpora_readings(record):
    """Re-derived by handing ``classify_branch`` the record's own six numbers, so the printed branch
    has to follow from them. The classifier was written before any of the six existed."""
    readings = {}
    for corpus in CORPORA:
        rows = {estimand: _rows(record)[f"{corpus}.{estimand}"] for estimand in ("L", "D", "S")}
        readings[corpus] = audit.corpus_reading(
            {name: row["point"] for name, row in rows.items()},
            {name: row["interval_95"] for name, row in rows.items()},
        )
        assert readings[corpus] == record["branch"]["per_corpus"][corpus]
    assert audit.classify_branch(readings)["branch"] == record["branch"]["branch"]
    assert record["branch"]["branch"] in audit.BRANCHES
    assert list(audit.BRANCHES) == ["substantial-attenuation", "mixed-informative",
                                    "unresolved-or-modest", "stable-positive-both"]


def test_no_result_at_one_corpus_is_promoted_to_stand_for_both(record):
    """Section G's own sentence, written as a refusal rather than as a note."""
    one = {"openhands": record["branch"]["per_corpus"]["openhands"]}
    with pytest.raises(AssertionError, match="both corpora"):
        audit.classify_branch(one)


def test_each_named_condition_needs_every_clause_the_declaration_gives_it():
    """Constructed readings, because the two conditions are what decide the branch and a corpus
    that happened not to meet one would leave the other clause untested.

    Substantial attenuation needs the size, the support and a positive linear increment.
    The stable-positive boundary needs a supported ``D`` at the declared size AND an ``S`` interval
    inside the declared band; a supported ``D`` alone is not a boundary.
    """
    def reading(**cells):
        return audit.corpus_reading({name: value[0] for name, value in cells.items()},
                                    {name: value[1:] for name, value in cells.items()})

    attenuating = reading(L=(0.14, 0.09, 0.19), D=(0.01, -0.02, 0.04), S=(0.13, 0.08, 0.18))
    assert attenuating["substantial_attenuation"] is True
    assert reading(L=(0.14, 0.09, 0.19), D=(0.01, -0.02, 0.04),
                   S=(0.13, -0.01, 0.28))["substantial_attenuation"] is False
    assert reading(L=(0.14, -0.01, 0.29), D=(0.01, -0.02, 0.04),
                   S=(0.13, 0.08, 0.18))["substantial_attenuation"] is False

    bounding = reading(L=(0.09, 0.05, 0.13), D=(0.08, 0.04, 0.12), S=(0.01, -0.02, 0.025))
    assert bounding["informative_stable_positive_boundary"] is True
    wide = reading(L=(0.09, 0.05, 0.13), D=(0.08, 0.04, 0.12), S=(0.01, -0.05, 0.08))
    assert wide["informative_stable_positive_boundary"] is False
    assert wide["unresolved"] is True
    small = reading(L=(0.09, 0.05, 0.13), D=(0.01, 0.005, 0.02), S=(0.01, -0.02, 0.025))
    assert small["informative_stable_positive_boundary"] is False


def test_failing_to_reject_s_equals_zero_is_reported_as_unresolved(record):
    """Section G's third bullet, and the trap it names. An ``S`` interval that contains both zero
    and the declared magnitude does not establish stability, and the record has to say unresolved
    rather than treating the containment as a result."""
    for corpus in CORPORA:
        reading = record["branch"]["per_corpus"][corpus]
        if reading["s_interval_contains_zero_and_the_threshold"]:
            assert reading["unresolved"] is True
            assert reading["informative_stable_positive_boundary"] is False
    assert audit.SUBSTANTIAL_EFFECT == 0.03
    for reading in record["branch"]["per_corpus"].values():
        assert reading["substantial_threshold"] == 0.03
        assert "not a deployment threshold" in reading["threshold_is"]


# --- the gate, and what makes a cleared preflight mean anything ----------------------------------

def test_the_batch_ran_behind_a_preflight_that_cleared_and_spent_no_batch_fit(record):
    gate = record["preflight"]["gate"]
    assert gate["admits_the_batch"] is True
    assert gate["cleared"] is True
    assert gate["findings"] == [] == record["preflight"]["findings"]
    assert gate["fits_of_the_declared_batch"] == 0
    assert sorted(gate["corpora_covered"]) == sorted(CORPORA)
    assert gate["corpora_missing"] == []


def test_the_gate_refuses_a_preflight_that_did_not_clear_or_that_spent_a_batch_fit():
    """Each of the three conditions on its own, because a gate that passed on two of three would
    admit a batch over rows or folds nothing checked."""
    good = {"cleared": True, "findings": [], "fits_of_the_declared_batch": 0,
            "corpora": {"openhands": {}, "scienceworld": {}}}
    assert driver.preflight_gate(good, list(CORPORA))["admits_the_batch"] is True
    assert driver.preflight_gate({**good, "cleared": False},
                                 list(CORPORA))["admits_the_batch"] is False
    assert driver.preflight_gate({**good, "findings": ["a disagreement"]},
                                 list(CORPORA))["admits_the_batch"] is False
    assert driver.preflight_gate({**good, "fits_of_the_declared_batch": 25},
                                 list(CORPORA))["admits_the_batch"] is False
    partial = driver.preflight_gate({**good, "corpora": {"openhands": {}}}, list(CORPORA))
    assert partial["admits_the_batch"] is False
    assert partial["corpora_missing"] == ["scienceworld"]


def test_the_batch_fitted_the_rows_the_columns_and_the_partitions_the_preflight_cleared(record):
    for corpus in CORPORA:
        bound = record["verification"]["per_corpus"][corpus]["matches_the_cleared_preflight"]
        rows = len(record["corpora"][corpus]["labels"])
        assert bound["fold_membership_cells_compared"] == 5 * rows
        assert bound["column_boundary_fields_compared"] == 4
        assert bound["population_observed"] == record["populations"][corpus]["observed"]
    binding = record["verification"]["openhands_ids_match_the_measured_overlap"]
    assert binding["ids_compared"] == len(record["corpora"]["openhands"]["labels"])


def test_a_partition_that_moved_between_the_preflight_and_the_batch_is_a_blocker():
    """A cleared preflight over a partition the batch did not fit on establishes nothing, so the
    comparison raises rather than recording a note beside the scores."""
    population = {"observed": {"n_runs": 4}}
    boundaries = {"splined_columns": [0, 1], "passthrough_columns": [2, 3],
                  "flat_shape": [4, 2], "flatdep_shape": [4, 4]}
    membership = [[0, 1, 0, 1]]
    stored = {"corpora": {"scienceworld": {"population": population,
                                           "column_boundaries": boundaries,
                                           "fold_membership": membership}}}
    assert driver._check_matches_preflight("scienceworld", stored, population, boundaries,
                                           membership)["column_boundary_fields_compared"] == 4
    with pytest.raises(AssertionError, match="fold membership"):
        driver._check_matches_preflight("scienceworld", stored, population, boundaries,
                                        [[1, 0, 0, 1]])
    with pytest.raises(AssertionError, match="the corpus moved"):
        driver._check_matches_preflight("scienceworld", stored, {"observed": {"n_runs": 5}},
                                        boundaries, membership)
    with pytest.raises(AssertionError, match="splined_columns"):
        driver._check_matches_preflight("scienceworld", stored, population,
                                        {**boundaries, "splined_columns": [0, 1, 2]}, membership)


def test_the_batch_performed_the_declared_number_of_fits_and_no_more(record):
    """Section J: 2 corpora x 4 arms x 5 seeds x 5 folds. A record that fitted more than it declared
    would have spent something the declaration did not budget."""
    assert record["settings"]["declared_fits"] == 200
    assert record["settings"]["fits_performed"] == 200
    assert record["declaration"]["frozen_before_the_run"]["declared_fits"] == 200
    assert record["declaration"]["frozen_before_the_run"]["seeds"] == [0, 1, 2, 3, 4]
    assert record["declaration"]["frozen_before_the_run"]["folds"] == audit.N_SPLITS == 5


def test_every_spline_bearing_fit_was_inspected_inside_the_fold_that_produced_it(record):
    """Section D fits knots, scaling and the classifier on training rows only. The comparison is per
    fit and by exact array equality, and an arm whose knots never differ from the whole-matrix knots
    is rejected, because there the comparison would have no negative control."""
    for corpus in CORPORA:
        locality = record["verification"]["per_corpus"][corpus]["training_only_transforms"]
        assert locality["arms_inspected"] == ["size (spline)", "size-spline + linear-deps"]
        assert locality["fits_inspected"] == 50
        for arm, row in locality["per_arm"].items():
            assert row["fits"] == 25, arm
            assert row["fits_whose_knots_differ_from_the_whole_matrix"] > 0, arm


# --- sections H and I: what the record reports and what it bars ----------------------------------

def test_the_record_reports_the_populations_the_fold_protocol_and_the_measured_overlap(record):
    """Section H names these three in every branch."""
    for corpus in CORPORA:
        population = record["populations"][corpus]
        assert population["declared"] == audit.DECLARED_POPULATION[corpus]
        for key, value in population["declared"].items():
            assert population["observed"][key] == value, (corpus, key)
        folds = record["fold_protocol"][corpus]
        assert folds["protocol"] == audit.FOLD_PROTOCOL[corpus]
        assert folds["n_seeds"] == 5 and folds["n_splits"] == 5
        assert folds["minimum_class_count_in_any_fold"] > 0
        assert folds["splits_compared"] == 50
    assert record["fold_protocol"]["openhands"]["grouped_by"] == "instance_id"
    assert record["fold_protocol"]["scienceworld"]["grouped_by"] is None


def test_the_measured_overlap_is_nonzero_so_the_disjoint_issue_set_claim_is_barred(record):
    """Section I permits a disjoint-issue-set claim only if the overlap measures zero. It did not,
    and the shared issue is disclosed rather than dropped."""
    overlap = record["swegym_overlap"]
    assert overlap["n_shared_issues"] == len(overlap["shared_issues"]) >= 1
    assert (overlap["n_distinct_openhands_issues"]
            == record["corpora"]["openhands"]["run_identities"]["n_distinct_group"])
    assert "no row is dropped" in overlap["no_row_is_dropped"].lower()
    assert "disjoint" in record["interpretation"]["no_disjoint_issue_set"]
    assert len(record["corpora"]["openhands"]["labels"]) == audit.DECLARED_POPULATION[
        "openhands"]["n_runs"]


def test_neither_corpus_was_added_to_a_loader_or_a_board_roster(record):
    from catchbench import detection

    assert sorted(detection._LOADERS) == ["swegym", "tau"]
    assert not set(audit.DECLARED_POPULATION) & set(detection._LOADERS)
    board = {method.method_id for method in detection.post_detection_methods()}
    assert "size (spline)" not in board and "size-spline + linear-deps" not in board
    assert record["board_entrants_added"] == []
    assert "untouched" in record["rosters_unchanged"]
    assert "live_streaming_methods()" in record["rosters_unchanged"]


def test_the_declaration_hash_is_recorded_at_both_moments_rather_than_repaired(record):
    """The declaration is append-only once fitting begins, so a dated deviation entry added between
    the preflight and the batch moves the file hash. A record that carried one hash would be
    claiming an immutability the document does not have; it carries the pair and says which is
    which, and nothing rewrites either."""
    declaration = record["inputs"]["declaration"]
    assert declaration["sha256"] and declaration["sha256_at_preflight"]
    assert declaration["changed_since_preflight"] == (
        declaration["sha256"] != declaration["sha256_at_preflight"])
    assert "append-only" in declaration["why_it_can_change"]
    assert record["declaration"]["file"] == audit.DECLARATION


def test_no_failure_or_warning_was_absorbed(record):
    """Section H reports every failure. An empty list has to mean nothing happened, so the fields a
    failure would have landed in are checked to be empty rather than absent."""
    assert record["failures"] == []
    assert record["warnings"] == []
    assert record["primary"] is not None
    assert record["primary"]["cell"] == "openhands.S"
    assert audit.PRIMARY_ESTIMAND == ("openhands", "S")
    assert record["primary"]["role"] == "primary"
    for row in record["effects"]:
        assert row["point_recomputed_by_the_bootstrap_scorer"]["within_tolerance"], row["cell"]
    assert record["inputs"]["repositories"]["paper"]["written_to"] is False
