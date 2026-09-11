r"""The POST generalization preflight: which population it measured, and which folds it built.

Sections C and D of the declaration are the two places where this audit can be wrong in a way no
later number reveals. A moved population still produces four arm scores. A fold that lost a class,
or a grouped split that let one attempt at an issue train against the other, still produces four arm
scores, and better ones. So the checks below hold the two things the scores cannot show.

The first group binds the shipped record to section C's counts, and pins them twice: against the
module's ``DECLARED_POPULATION`` and against the literals the declaration froze, because comparing
the record only against the dictionary that produced it would pass on whatever that dictionary later
said. It also holds the two measurements section C ordered because the loaders drop them, the
ScienceWorld task identities and the SWE-Gym issue overlap, down to the route each was measured
through.

The second group holds section D's four assertions about the partition, read off the record's own
fold membership and group keys rather than off the flag that says a check ran.

The third group is the one that does not depend on the record at all. Section D requires the
saved-index fold path to reproduce ``detection._cv_estimator_detail`` exactly, and a regenerated
record could report that agreement without it holding. So the two paths are run against each other
here on synthetic rows, bit for bit, with no corpus, GRADE checkout or network involved.

The fourth group binds the frozen quantities and section G's branch rule, which was written before
any of the six quantities existed and is therefore a rule rather than a reading.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from catchbench import post_generalization_audit as audit  # noqa: E402

RECORD = ROOT / "tools" / "post_generalization_preflight.json"

# Section C's counts, as literals. Pinned as text beside the module constant, so a change to the
# module cannot carry the record and this file along with it.
DECLARED = {
    "openhands": {"n_runs": 600, "n_solved": 288, "n_failed": 312, "n_distinct_group": 574,
                  "group_multiplicity": {"1": 548, "2": 26}, "n_missing_group": 0},
    "scienceworld": {"n_runs": 128, "n_solved": 64, "n_failed": 64},
}


@pytest.fixture(scope="module")
def record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _synthetic(rows: int = 60, seed: int = 12345):
    """A ``flat`` / ``flatdep`` pair and balanced labels, in the shape the arms read.

    Eight columns whose leading four are the flat block, which is the invariant
    ``_MixedSplineAUC.columns`` refuses to proceed without, and enough distinct values per column
    that the quantile knots are not degenerate.
    """
    rng = np.random.default_rng(seed)
    flat = rng.normal(size=(rows, 4)) + np.arange(rows).reshape(-1, 1) / rows
    deps = rng.normal(size=(rows, 4))
    labels = np.array([0, 1] * (rows // 2))
    return {"flat": flat, "flatdep": np.hstack([flat, deps])}, labels


class _StubTask:
    """The surface ``detection``'s arms and this module's checks read, and nothing else."""

    task_id = "post_detection"

    def __init__(self, layers, labels, groups=None, corpus="scienceworld"):
        self.corpus = corpus
        self.dataset = corpus
        self.layers = layers
        self.y = labels
        self.groups = groups
        self.identities = {"available": False, "reason": "synthetic rows, no corpus behind them"}

    def setup(self):
        return None


# --- section C: the populations, and the two things the loaders drop -----------------------------

def test_every_declared_population_number_reproduces_exactly_in_the_record(record):
    for corpus, declared in DECLARED.items():
        population = record["corpora"][corpus]["population"]
        assert population["declared"] == declared == audit.DECLARED_POPULATION[corpus]
        for key, value in declared.items():
            assert population["observed"][key] == value, (corpus, key)


def test_the_eligibility_rule_is_posts_and_not_lives(record):
    """Section C: at least two parsed steps, and the LIVE four-step filter is not applied."""
    assert audit.MIN_STEPS == 2
    for corpus in DECLARED:
        population = record["corpora"][corpus]["population"]
        assert "2 parsed steps" in population["eligibility"]
        assert "LIVE four-step filter is not applied" in population["eligibility"]
        assert population["label_1_means"] == "failure"


def test_the_openhands_grouping_key_is_the_issue_id_and_every_row_carries_one(record):
    identities = record["corpora"]["openhands"]["population"]["identities"]
    assert identities["group_key_field"] == "instance_id"
    assert identities["n_missing_group"] == 0
    keys = identities["group_key"]
    assert len(keys) == DECLARED["openhands"]["n_runs"]
    assert len(set(keys)) == DECLARED["openhands"]["n_distinct_group"]
    counts = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    histogram = {}
    for count in counts.values():
        histogram[str(count)] = histogram.get(str(count), 0) + 1
    assert histogram == DECLARED["openhands"]["group_multiplicity"]


def test_the_scienceworld_identities_were_replayed_from_the_raw_json(record):
    """Section C ordered this measurement because the loader drops all three of these fields.

    The mixture is what section I's limitation is stated against, so the counts are held rather than
    the fact that something was recorded, and the source file is held as a single name: the loader
    balances within one source precisely so the features cannot separate source instead of outcome,
    and a second file among the selected rows would break that.
    """
    identities = record["corpora"]["scienceworld"]["population"]["identities"]
    assert "raw JSON" in identities["source"] or "raw" in identities["source"]
    assert identities["n_distinct_task_var_pairs"] == 128
    assert identities["n_distinct_task_names"] == 18
    assert identities["source_file"] == (
        "test trajs/EEF/NAT3I99p-ep6-6fm-ep6_sciworld_0_-200.json")
    assert sum(identities["runs_per_task_name"].values()) == 128
    assert len(identities["runs_per_task_name"]) == identities["n_distinct_task_names"]
    assert len(identities["task_name"]) == len(identities["var_num"]) == 128
    assert len(set(zip(identities["task_name"], identities["var_num"]))) == 128
    # one model, one scaffold: the confound control the loader's single-source rule exists for
    assert identities["llm_name"] == ["NAT3I99p-ep6-6fm-ep6"]
    assert identities["agent_arch"] == ["think_act"]
    assert identities["repository"] == "lclan/webshop_expert_trajectories"
    assert "webshop_expert_trajectories" in identities["repository_name_is_wrong"]


def test_the_swegym_overlap_was_measured_through_the_raw_shard_and_costs_no_row(record):
    """Section C: the adapter drops ``instance_id``, so the overlap is measured past it, not through
    it, and a shared issue is disclosed rather than dropped."""
    overlap = record["swegym_overlap"]
    assert "instance_id" in overlap["why_not_the_loader"]
    assert "parquet" in overlap["measured_through"]
    assert overlap["n_distinct_openhands_issues"] == DECLARED["openhands"]["n_distinct_group"]
    assert overlap["n_shared_issues"] == len(overlap["shared_issues"]) == 1
    assert overlap["shared_issues"] == ["iterative__dvc-1651"]
    assert "no row is dropped" in overlap["no_row_is_dropped"].lower()
    assert len(overlap["alignment_verified"]) == 2


def test_an_alignment_that_does_not_hold_is_recorded_as_verified_nowhere(record):
    """Every replayed identity says which two comparisons established it, because an unverified
    identifier is worse than none: the grouped folds are built from one of them."""
    for corpus in DECLARED:
        identities = record["corpora"][corpus]["population"]["identities"]
        assert identities["available"] is True
        assert identities["alignment_verified"] == [
            "labels equal the scored labels row for row",
            "replayed step counts equal the n_steps column row for row",
        ]


# --- section D: the partition -------------------------------------------------------------------

def test_every_fold_at_every_seed_carries_both_classes(record):
    """Section D asserts this and blocks if it does not hold. A single-class fold has no ROC-AUC."""
    for corpus in DECLARED:
        folds = record["corpora"][corpus]["fold_protocol"]
        assert folds["n_seeds"] == 5 and folds["n_splits"] == audit.N_SPLITS == 5
        seen = 0
        for seed in folds["per_seed"]:
            assert len(seed["folds"]) == 5
            for fold in seed["folds"]:
                assert fold["n_failed"] > 0 and fold["n_solved"] > 0, (corpus, seed["seed"])
                seen += 1
        assert seen == 25
        assert folds["minimum_class_count_in_any_fold"] > 0


def test_every_row_is_held_out_exactly_once_per_seed(record):
    for corpus, declared in DECLARED.items():
        membership = np.asarray(record["corpora"][corpus]["fold_membership"], dtype=int)
        assert membership.shape == (5, declared["n_runs"])
        for row in membership:
            counts = np.bincount(row, minlength=5)
            assert len(counts) == 5
            assert counts.sum() == declared["n_runs"]
            assert counts.min() > 0


def test_no_issue_group_is_split_across_a_fold_boundary(record):
    """Read off the record's own fold membership and group keys, not off the flag that a check ran.

    This is the property the grouping exists for: 26 OpenHands issues are attempted twice, and a row
    split would score one attempt with a model trained on the other.
    """
    groups = np.asarray(record["corpora"]["openhands"]["population"]["identities"]["group_key"])
    membership = np.asarray(record["corpora"]["openhands"]["fold_membership"], dtype=int)
    for seed, row in enumerate(membership):
        for group in set(groups.tolist()):
            folds = set(row[groups == group].tolist())
            assert len(folds) == 1, (seed, group, sorted(folds))


def test_the_grouped_protocol_reaches_the_corpus_that_repeats_a_unit_and_no_other(record):
    assert audit.GROUPED_CORPORA == ("openhands",)
    openhands = record["corpora"]["openhands"]["fold_protocol"]
    scienceworld = record["corpora"]["scienceworld"]["fold_protocol"]
    assert openhands["grouped_by"] == "instance_id"
    assert openhands["protocol"] == audit.FOLD_PROTOCOL["openhands"]
    assert openhands["protocol"].startswith("StratifiedGroupKFold(n_splits=5, shuffle=True, "
                                            "random_state=seed)")
    assert scienceworld["grouped_by"] is None
    assert scienceworld["protocol"] == audit.FOLD_PROTOCOL["scienceworld"]
    assert scienceworld["protocol"] == ("StratifiedKFold(n_splits=5, shuffle=True, "
                                        "random_state=seed)")


def test_all_four_arms_share_one_partition_on_every_feature_layer(record):
    """Section D: all four arms share the same partitions. Verified on the real matrices, because
    a splitter reading only ``len(X)`` is scikit-learn's behaviour rather than this code's."""
    layers = sorted({arm.layer for arm in audit.audit_arms()})
    for corpus in DECLARED:
        folds = record["corpora"][corpus]["fold_protocol"]
        assert folds["layers_compared"] == layers
        assert folds["splits_compared"] == len(layers) * 25


# --- section D: the new fold path did not change the fitting --------------------------------------

def test_the_saved_index_loop_reproduces_the_shipped_fold_loop_bit_for_bit():
    """The check that does not depend on the record, run on synthetic rows with no corpus involved.

    ``_cv_estimator_on_splits`` exists because ``_cv_estimator_detail`` builds its partition inside
    its own loop and cannot be handed a grouped one. Everything else was kept identical, and this is
    what turns that from a claim into a measurement: handed the partition ``_cv_estimator_detail``
    would have built, the two must agree exactly, on every arm and on both outputs.
    """
    from sklearn.model_selection import StratifiedKFold

    from catchbench.detection import _cv_estimator_detail

    layers, labels = _synthetic()
    task = _StubTask(layers, labels)
    splits = {int(seed): [(train.copy(), test.copy()) for train, test in
                          StratifiedKFold(n_splits=5, shuffle=True,
                                          random_state=seed).split(layers["flat"], labels)]
              for seed in audit.SEEDS}
    for arm in audit.audit_arms():
        matrix = layers[arm.layer]
        factory = arm.estimator_factory(task)
        ours_fold, ours_oof, ours_chosen = audit._cv_estimator_on_splits(
            factory, matrix, labels, splits)
        theirs_fold, theirs_oof, theirs_chosen = _cv_estimator_detail(factory, matrix, labels)
        assert np.array_equal(ours_fold, theirs_fold), arm.method_id
        assert np.array_equal(ours_oof, theirs_oof), arm.method_id
        assert ours_chosen == theirs_chosen == []


def test_the_saved_index_loop_refuses_a_partition_that_is_not_the_declared_shape():
    """A partition with the wrong seeds or the wrong fold count would silently misalign the rows of
    the fold-score array against the seeds they came from."""
    from sklearn.model_selection import StratifiedKFold

    layers, labels = _synthetic()
    task = _StubTask(layers, labels)
    factory = audit.audit_arms()[0].estimator_factory(task)
    full = {int(seed): [(train.copy(), test.copy()) for train, test in
                        StratifiedKFold(n_splits=5, shuffle=True,
                                        random_state=seed).split(layers["flat"], labels)]
            for seed in audit.SEEDS}
    with pytest.raises(AssertionError, match="seeds"):
        audit._cv_estimator_on_splits(factory, layers["flat"], labels,
                                      {seed: full[seed] for seed in (0, 1, 2)})
    with pytest.raises(AssertionError, match="folds"):
        audit._cv_estimator_on_splits(factory, layers["flat"], labels,
                                      {seed: folds[:4] for seed, folds in full.items()})


def test_the_batch_scorer_runs_and_preserves_everything_section_e_lists():
    """Section J allows the batch one pass, so the pass has to work the first time.

    Run here on synthetic rows rather than on either corpus, which exercises the whole batch path,
    ``score_arm`` through ``_effects`` through the hashes, without producing any of section E's six
    quantities. ``S`` is checked as an identity against ``L`` and ``D`` rather than recomputed a
    second way, because ``S = L - D`` is its definition and an artifact that stored a third number
    there would be storing something else.
    """
    layers, labels = _synthetic()
    task = _StubTask(layers, labels)
    result = audit.post_generalization_audit(task)

    for field in ("labels", "seeds", "fold_membership", "run_identities", "fold_protocol",
                  "resampling_unit", "rng_labels", "feature_hashes", "label_hash", "arms",
                  "effects"):
        assert field in result, field
    assert sorted(result["arms"]) == sorted(audit.ARM_ROLES)
    for arm_id, arm in result["arms"].items():
        assert arm["role"] == audit.ARM_ROLES[arm_id]
        assert arm["board_entrant"] is False
        assert arm["nonfinite_fold_scores"] == 0 and arm["nonfinite_oof_values"] == 0
        assert np.asarray(arm["oof_proba"]).shape == (5, len(labels))
        assert np.asarray(arm["fold_roc_auc"]).shape == (5, 5)

    effects = result["effects"]
    assert sorted(effects) == ["D", "L", "S"]
    assert (effects["L"]["high"], effects["L"]["low"]) == audit.LINEAR_CONTRAST
    assert (effects["D"]["high"], effects["D"]["low"]) == audit.MATCHED_CONTRAST
    for scale in ("seed_averaged_oof_difference", "mean_fold_difference"):
        assert effects["S"][scale] == effects["L"][scale] - effects["D"][scale]
    assert effects["S"]["definition"] == "L - D"
    assert sorted(result["feature_hashes"]) == ["flat", "flatdep"]
    assert len(set(result["feature_hashes"].values())) == 2


def test_the_parity_check_refuses_the_grouped_corpus():
    """Section D allows the comparison only where the two partitions agree by construction. On a
    grouped corpus the two paths would be compared on different splits and would measure nothing."""
    layers, labels = _synthetic()
    grouped = _StubTask(layers, labels, groups=np.arange(len(labels)) // 2, corpus="openhands")
    with pytest.raises(AssertionError, match="grouped"):
        audit.check_saved_split_parity(grouped, {})


def test_the_record_shows_the_two_paths_agreed_on_every_arm(record):
    parity = record["corpora"]["scienceworld"]["saved_split_parity"]
    assert parity["compared_against"] == "catchbench.detection._cv_estimator_detail"
    assert parity["tolerance"] == audit.PARITY_ATOL
    assert parity["arms_disagreeing"] == 0
    assert sorted(parity["arms_compared"]) == sorted(audit.ARM_ROLES)
    for arm, row in parity["per_arm"].items():
        assert row["within_tolerance"], arm
        assert row["bit_identical_fold_auc"] and row["bit_identical_oof"], arm
        assert row["fits_each_path"] == 25
    assert record["corpora"]["openhands"]["saved_split_parity"] is None


# --- the record bound to its declaration ----------------------------------------------------------

def test_the_tolerance_was_declared_before_the_first_fit_and_was_not_widened(record):
    assert audit.PARITY_ATOL == 1e-12
    assert record["tolerance"]["value"] == audit.PARITY_ATOL
    assert "before the first fit" in record["tolerance"]["declared"]
    assert record["frozen_before_the_run"]["parity_tolerance"] == 1e-12


def test_every_rng_label_is_the_frozen_literal(record):
    """Section F froze the labels as text, so they are pinned as text rather than as ``RNG_LABELS``:
    comparing the record against the dictionary that produced it would pass on any label the
    dictionary later built."""
    labels = record["frozen_before_the_run"]["rng_labels"]
    assert labels == {f"{corpus}.{estimand}": f"post_generalization.{corpus}.{estimand}"
                      for corpus in ("openhands", "scienceworld")
                      for estimand in ("D", "L", "S")}
    assert record["frozen_before_the_run"]["rng_base_seed"] == audit.BOOTSTRAP_BASE_SEED == 20260907
    assert record["frozen_before_the_run"]["bootstrap_draws"] == audit.BOOTSTRAP_DRAWS == 10_000


def test_only_the_size_columns_are_splined_in_the_matched_arm(record):
    """Section B's boundary. Splining all eight ``flatdep`` columns is a different control and a
    different number, ``-0.016325`` against ``-0.005404`` on the committed endpoint records."""
    for corpus in DECLARED:
        boundary = record["corpora"][corpus]["column_boundaries"]
        assert boundary["checked_by"] == "catchbench.detection._MixedSplineAUC.columns"
        assert boundary["splined_columns"] == [0, 1, 2, 3]
        assert boundary["passthrough_columns"] == [4, 5, 6, 7]
        assert boundary["flat_shape"][1] == 4 and boundary["flatdep_shape"][1] == 8


def test_the_four_arms_are_detections_own_objects_in_the_declared_roles():
    from catchbench.detection import _EstimatorAUC, _MixedSplineAUC

    arms = audit.audit_arms()
    assert [arm.method_id for arm in arms] == list(audit.ARM_ROLES)
    assert [type(arm) for arm in arms] == [_EstimatorAUC] * 3 + [_MixedSplineAUC]
    assert [arm.layer for arm in arms] == ["flat", "flatdep", "flat", "flatdep"]
    assert audit.LINEAR_CONTRAST == ("auditable (size+deps)", "size (flat)")
    assert audit.MATCHED_CONTRAST == ("size-spline + linear-deps", "size (spline)")


def test_neither_corpus_was_added_to_a_loader_or_a_board_roster(record):
    """Section C: neither corpus is added to ``_LOADERS``, to any board roster, or to
    ``live_streaming_methods()``."""
    from catchbench import detection

    assert sorted(detection._LOADERS) == ["swegym", "tau"]
    assert not set(audit.DECLARED_POPULATION) & set(detection._LOADERS)
    board = {method.method_id for method in detection.post_detection_methods()}
    assert "size (spline)" not in board and "size-spline + linear-deps" not in board
    assert record["board_entrants_added"] == []


def test_the_preflight_fitted_no_cell_of_the_declared_batch(record):
    """Section J makes the batch one pass of 200 fits, so a preflight that scored a cell of it would
    have spent it. The parity comparison fits estimators and produces no declared quantity."""
    assert record["fits_of_the_declared_batch"] == 0
    assert record["frozen_before_the_run"]["declared_fits"] == 200
    parity = record["corpora"]["scienceworld"]["saved_split_parity"]
    assert "scores" in parity["scores_withheld"]
    for row in parity["per_arm"].values():
        assert "seed_averaged_oof_roc_auc" not in row and "fold_roc_auc" not in row


def test_the_primary_estimand_is_the_one_section_e_nominated(record):
    assert audit.PRIMARY_ESTIMAND == ("openhands", "S")
    assert record["frozen_before_the_run"]["primary_estimand"] == {"corpus": "openhands",
                                                                   "estimand": "S"}
    assert audit.ESTIMANDS == ("L", "D", "S")
    assert audit.RESAMPLING_UNIT == {"openhands": "issue group", "scienceworld": "row"}


# --- section G's branch, fixed before any of the six quantities existed ---------------------------

def _reading(**estimates):
    """One corpus's reading from three points and three intervals, given as (point, low, high)."""
    return audit.corpus_reading({name: value[0] for name, value in estimates.items()},
                                {name: value[1:] for name, value in estimates.items()})


def test_substantial_attenuation_needs_the_size_the_support_and_a_positive_linear_increment():
    supported = _reading(L=(0.14, 0.09, 0.19), D=(0.01, -0.02, 0.04), S=(0.13, 0.08, 0.18))
    assert supported["substantial_attenuation"] is True
    # the point clears +0.03 but the interval does not exclude zero: unresolved, not substantial
    unsupported = _reading(L=(0.14, 0.09, 0.19), D=(0.01, -0.02, 0.04), S=(0.13, -0.01, 0.28))
    assert unsupported["substantial_attenuation"] is False
    assert unsupported["unresolved"] is True
    # section G asks for evidence of a positive L alongside; without it the attenuation is of
    # nothing, and this implementation reads "evidence" as an interval excluding zero
    without_l = _reading(L=(0.14, -0.01, 0.29), D=(0.01, -0.02, 0.04), S=(0.13, 0.08, 0.18))
    assert without_l["evidence_of_a_positive_linear_increment"] is False
    assert without_l["substantial_attenuation"] is False


def test_the_stable_positive_boundary_needs_a_supported_d_and_an_s_inside_the_declared_band():
    inside = _reading(L=(0.09, 0.05, 0.13), D=(0.08, 0.04, 0.12), S=(0.01, -0.02, 0.025))
    assert inside["informative_stable_positive_boundary"] is True
    # the same supported D with an S interval reaching past the declared band is not a boundary
    outside = _reading(L=(0.09, 0.05, 0.13), D=(0.08, 0.04, 0.12), S=(0.01, -0.02, 0.05))
    assert outside["informative_stable_positive_boundary"] is False
    assert outside["unresolved"] is True


def test_failing_to_reject_s_equals_zero_is_reported_as_unresolved():
    """Section G's third bullet: an interval that includes zero while also including 0.03 does not
    establish stability, and is reported as unresolved."""
    reading = _reading(L=(0.05, 0.01, 0.09), D=(0.04, 0.00, 0.08), S=(0.01, -0.02, 0.05))
    assert reading["s_interval_contains_zero_and_the_threshold"] is True
    assert reading["unresolved"] is True
    assert reading["substantial_attenuation"] is False
    assert reading["informative_stable_positive_boundary"] is False


def test_the_four_branches_are_the_declared_ones_and_the_overlapping_rows_are_disjoint_here():
    """Section G's first two rows overlap as written, so the reading is fixed in code and held here.

    Row 1's "an informative replication or boundary" contains row 2's "informative stable-positive
    boundary", which under a literal first-match reading of the table would make row 2 unreachable.
    The boundary case therefore goes to the row that names it and the replication case stays with
    row 1, which is the only reading under which all four branches can occur.
    """
    assert audit.BRANCHES == ("substantial-attenuation", "mixed-informative",
                              "unresolved-or-modest", "stable-positive-both")
    attenuating = _reading(L=(0.14, 0.09, 0.19), D=(0.01, -0.02, 0.04), S=(0.13, 0.08, 0.18))
    bounding = _reading(L=(0.09, 0.05, 0.13), D=(0.08, 0.04, 0.12), S=(0.01, -0.02, 0.025))
    nothing = _reading(L=(0.01, -0.03, 0.05), D=(0.01, -0.03, 0.05), S=(0.00, -0.04, 0.04))
    both_attenuate = audit.classify_branch({"openhands": attenuating,
                                            "scienceworld": attenuating})
    assert both_attenuate["branch"] == "substantial-attenuation"
    mixed = audit.classify_branch({"openhands": attenuating, "scienceworld": bounding})
    assert mixed["branch"] == "mixed-informative"
    assert audit.classify_branch({"openhands": bounding,
                                  "scienceworld": bounding})["branch"] == "stable-positive-both"
    assert audit.classify_branch({"openhands": attenuating,
                                  "scienceworld": nothing})["branch"] == "unresolved-or-modest"
    assert audit.classify_branch({"openhands": nothing,
                                  "scienceworld": nothing})["branch"] == "unresolved-or-modest"
    assert {both_attenuate["branch"], mixed["branch"]} <= set(audit.BRANCHES)


def test_no_result_at_one_corpus_is_promoted_to_stand_for_both():
    """Section G: the branch classification requires both corpora."""
    attenuating = _reading(L=(0.14, 0.09, 0.19), D=(0.01, -0.02, 0.04), S=(0.13, 0.08, 0.18))
    with pytest.raises(AssertionError, match="both corpora"):
        audit.classify_branch({"openhands": attenuating})


def test_the_declared_threshold_travels_with_the_words_section_g_gives_it():
    reading = _reading(L=(0.14, 0.09, 0.19), D=(0.01, -0.02, 0.04), S=(0.13, 0.08, 0.18))
    assert reading["substantial_threshold"] == audit.SUBSTANTIAL_EFFECT == 0.03
    assert "not a deployment threshold" in reading["threshold_is"]
    assert "not a validated minimum useful effect" in reading["threshold_is"]


def test_the_preflight_cleared_and_recorded_the_reading_it_had_to_fix(record):
    assert record["cleared"] is True and record["findings"] == []
    fixed = record["fixed_by_this_implementation"]
    assert "mixed-informative" in fixed["section_g_row_overlap"]
    assert "evidence of a positive L" in fixed["evidence_of_a_positive_L"]
    assert record["declaration"]["file"] == audit.DECLARATION
    assert record["declaration"]["sections"] == ["C", "D"]
