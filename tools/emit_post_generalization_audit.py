r"""The declared POST generalization batch: 200 fits, six estimands, six intervals, one branch.

The analysis is ``research/catchbench-post-generalization-declaration-2026-09-11.md``, frozen at
2026-09-11T01:05:49Z before any arm was fitted on either new corpus.
``catchbench.post_generalization_audit`` holds the arms, the saved-split fold path, the checks and
section G's branch rule; ``tools/post_generalization_preflight.py`` is the gate that has to clear
first; this script is the driver that spends the batch. The split is the one
``tools/emit_live_size_audit.py`` keeps from ``catchbench.live_size_audit``.

    GRADE_DIR=/path/to/grade python tools/emit_post_generalization_audit.py --run \
        --declaration /path/to/catchbench-post-generalization-declaration-2026-09-11.md

**Section J makes this one pass.** Four arms on two corpora at five seeds and five folds is 200
classifier fits, and a disappointing effect is not permission to run them again. So the gate is read
off the committed preflight record rather than recomputed: the preflight already fitted the
ScienceWorld parity comparison, and running it a second time here would spend those fits to learn
what the record already says. What this script does recompute is everything that ties the batch to
that record. Section C's counts are re-measured on today's corpora and re-blocked, the matched arm's
column boundary is re-read through ``_MixedSplineAUC.columns``, section D's four fold assertions are
re-run, and the resulting partition is compared index for index against the ``fold_membership`` the
preflight recorded. A cleared preflight over a partition the batch did not fit on would establish
nothing.

Section F's intervals are written here rather than reached for in ``tools/statistical_tests.py``,
and the two shipped bootstraps are the reason. ``_task_clustered_auc_bootstrap`` resamples through
``_cluster_blocks``, which requires equal-sized clusters and refuses a ragged grid on purpose:
tau-bench's grid is exactly four attempts per task, and a ragged one there would mean the replay had
drifted. OpenHands is ragged by construction, 548 issues attempted once and 26 attempted twice, so
that function cannot express its unit. ``_stratified_auc_bootstrap`` draws positives and negatives
separately, which makes a single-class draw impossible, and section F asks for single-class draws to
be discarded AND COUNTED, which is a specification of an unstratified draw. What is reused is
everything that does not turn on the unit: ``st._rng_for`` for the frozen labels, ``st._auc_vectorized``
for the inner AUC, and the 2.5 / 97.5 percentile rule.

``_auc_vectorized`` rather than ``_auc_fast`` for the same reason the clustered bootstrap uses it: a
unit resample duplicates whole rows, so exact ties are guaranteed on every draw and the tie handling
is the hot path rather than an edge case. The two return the same value and
``tests/test_tau_cluster_bootstrap.py`` pins that equality on tie-heavy input.

``S`` is a difference of differences and its interval carries all four score vectors in every draw,
which section F requires and which matters here: ``L`` and ``D`` are the same rows scored on the
same 25 splits, so they are strongly correlated and an interval built from two unrelated resamples
would be wider than the quantity deserves.

Fold locality is checked through ``emit_detection_audit._FoldLocality``, one instance per corpus,
driven from the runner's ``inspect`` hook so all 100 fits of a corpus are inspected inside the fold
loop that produced them and none is refitted. Its ``record()`` is deliberately NOT called. That
method ends by fitting 50 further estimators to score a leaked basis, which produces no declared
quantity, and its prose states a measured fact about SWE-Gym and tau-bench ("in opposite directions
on the two corpora") that is not a fact about these two. The per-fit array comparisons are the
load-bearing half and they are what runs here; the summary is assembled below from the counters that
half fills in.

Nothing here is a board entrant. ``detection._LOADERS`` is untouched, no roster gains a row, and the
nine boards, the 72 entrants and the 138-record registry are what they were. The two new Hugging
Face repositories stay out of ``catchbench.corpora.CORPUS_REVISIONS``, because that tuple feeds
``revision_header`` onto the board; their heads are recorded instead.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import statistical_tests as st  # noqa: E402

# The preflight is this declaration's own driver, not another declaration's, so its provenance
# helpers and its Hub retry wrapper are imported rather than copied. ``emit_live_size_audit.py``
# copies the same four helpers from a DIFFERENT declaration's driver on purpose; the reason given
# there is that coupling two independent documents' drivers buys nothing, and it does not apply
# between the two halves of one document.
from post_generalization_preflight import (  # noqa: E402
    REPOSITORIES, _git_commit, _git_worktree, _hub_revisions, _sha256, _wrap_hub_listing,
)

CORPORA = ("openhands", "scienceworld")

DEFAULT_PREFLIGHT = ROOT / "tools" / "post_generalization_preflight.json"
DEFAULT_OUTPUT = ROOT / "tools" / "post_generalization_audit_results.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", required=True,
                        help="spend section J's one batch of 200 fits; required, so the batch is "
                             "never a side effect of asking this script anything else")
    parser.add_argument("--corpora", nargs="+", choices=CORPORA, default=list(CORPORA))
    parser.add_argument("--preflight-record", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--declaration", type=Path, default=None,
                        help="the frozen declaration, recorded by path and sha256")
    parser.add_argument("--no-hub-revisions", action="store_true",
                        help="skip resolving the dataset heads; for an offline rerun over a cache "
                             "a previous verified run already populated")
    return parser.parse_args()


# --- the gate: what the committed preflight has to say before the batch is spent ------------------


def preflight_gate(record: dict, corpora: list[str]) -> dict:
    """Whether the committed preflight admits the batch, read off its fields rather than its prose.

    Three conditions, and each is a different way the batch could be spent on the wrong thing. The
    preflight must have cleared with no findings, which is section C's and section D's verdict. It
    must have fitted none of the declared batch, because a preflight that had scored a cell would
    have made that cell exist before section E nominated its primary. And it must cover every
    corpus this run scores, since a corpus absent from the preflight has had neither its population
    nor its folds checked.

    Returns the reading rather than raising, so a refusal travels into the record with the fields it
    was made from. Nothing here relaxes anything: the preflight's tolerance was fixed before its
    first fit and this reads its result, not its inputs.
    """
    missing = sorted(set(corpora) - set(record.get("corpora", {})))
    fits = record.get("fits_of_the_declared_batch")
    admits = bool(record.get("cleared") and not record.get("findings") and fits == 0 and not missing)
    return {
        "admits_the_batch": admits,
        "cleared": record.get("cleared"),
        "findings": list(record.get("findings") or []),
        "fits_of_the_declared_batch": fits,
        "corpora_covered": sorted(record.get("corpora", {})),
        "corpora_missing": missing,
        "reading": "sections C and D cleared with no findings, the preflight spent none of section "
                   "J's 200 fits, and every corpus this batch scores was checked by it",
    }


def _check_matches_preflight(corpus: str, record: dict, population: dict, boundaries: dict,
                             membership: list) -> dict:
    """Bind today's batch to the cleared preflight: same rows, same columns, same 25 partitions.

    The preflight's verdict is about the corpora and the folds it saw. Carrying that verdict into a
    batch that fitted something else is the failure this rules out, and it is not hypothetical: both
    corpora are read live from the Hugging Face Hub and neither is pinned in ``CORPUS_REVISIONS``,
    so a moved head between the two runs is exactly the thing section C's counts exist to catch.

    Exact equality throughout, and no tolerance anywhere. Fold membership is compared index for
    index because an approximate match would mean these are not the same partitions; the observed
    population and the column boundary are compared field for field for the same reason. A
    disagreement raises: a batch fitted on rows or folds the preflight did not clear is a different
    analysis, and section J does not admit repairing one into the other.
    """
    stored = record["corpora"][corpus]
    if population["observed"] != stored["population"]["observed"]:
        raise AssertionError(
            f"{corpus}: the population this batch scored is {population['observed']} and the "
            f"preflight cleared {stored['population']['observed']}; the corpus moved between the "
            "two runs and the preflight's verdict does not cover these rows"
        )
    for field in ("splined_columns", "passthrough_columns", "flat_shape", "flatdep_shape"):
        if boundaries[field] != stored["column_boundaries"][field]:
            raise AssertionError(
                f"{corpus}/{field}: {boundaries[field]} against the preflight's "
                f"{stored['column_boundaries'][field]}; the matched arm is not splining the block "
                "the preflight checked"
            )
    if membership != stored["fold_membership"]:
        differing = sum(1 for seed_now, seed_then in zip(membership, stored["fold_membership"])
                        for now, then in zip(seed_now, seed_then) if now != then)
        raise AssertionError(
            f"{corpus}: the batch's fold membership differs from the preflight's in {differing} "
            "row-seed cells, so the four arms were not fitted on the partition section D's "
            "assertions were checked against"
        )
    return {
        "establishes": "the rows, the column boundary and all 25 partitions this batch fitted on "
                       "are the ones the preflight cleared, compared exactly and with no tolerance",
        "population_observed": dict(population["observed"]),
        "fold_membership_cells_compared": sum(len(seed) for seed in membership),
        "column_boundary_fields_compared": 4,
    }


def _openhands_groups_match_preflight(record: dict, task) -> dict:
    """The OpenHands side of the carried-forward overlap, re-derived and compared row for row.

    Section H reports the measured SWE-Gym overlap and the preflight measured it, through a raw
    parquet replay the adapter cannot reach. Re-reading three parquet shards here to recompute a
    number a cleared record already carries would add Hub traffic and no evidence, so the overlap is
    carried forward. What makes carrying it sound is this: the OpenHands half of that intersection
    is recomputed inside ``task.setup()`` every run, and it is compared here against the ids the
    preflight intersected, in order and exactly. The SWE-Gym half is pinned in
    ``CORPUS_REVISIONS`` and its fetches are verified there.
    """
    stored = record["corpora"]["openhands"]["population"]["identities"]["group_key"]
    today = [str(value) for value in task.identities["group_key"]]
    if today != stored:
        differing = sum(1 for now, then in zip(today, stored) if now != then)
        raise AssertionError(
            f"the OpenHands issue ids differ from the preflight's in {differing} of "
            f"{len(stored)} rows, so the recorded overlap was measured against other rows"
        )
    return {
        "establishes": "the OpenHands issue ids this batch grouped by are the ids the preflight "
                       "intersected with the scored SWE-Gym population, compared in order",
        "ids_compared": len(today),
    }


# --- section F's intervals -------------------------------------------------------------------------


def _resampling_units(corpus: str, record: dict) -> tuple[list[np.ndarray], str]:
    """Section F's resampling unit for one corpus, as the row indices each unit carries.

    The issue group for OpenHands, so every attempt at an issue enters a draw together or not at
    all; the row for ScienceWorld, which has no repeated unit to carry. The unit list is in
    first-appearance order over the scored rows rather than sorted, so a reader replaying it from
    the ``group_key`` array this record already stores reconstructs the same list without depending
    on a sort order.
    """
    from catchbench.post_generalization_audit import RESAMPLING_UNIT

    unit = RESAMPLING_UNIT[corpus]
    n_rows = len(record["labels"])
    if unit == "row":
        return [np.array([row]) for row in range(n_rows)], unit
    keys = [str(value) for value in record["run_identities"]["group_key"]]
    if len(keys) != n_rows:
        raise AssertionError(f"{corpus}: {len(keys)} group keys for {n_rows} scored rows")
    order: list[str] = []
    rows: dict[str, list[int]] = {}
    for row, key in enumerate(keys):
        if key not in rows:
            rows[key] = []
            order.append(key)
        rows[key].append(row)
    return [np.array(rows[key]) for key in order], unit


def _unit_bootstrap(labels: np.ndarray, units: list[np.ndarray], columns: list[np.ndarray],
                    combine, draws: int, rng: np.random.Generator) -> dict:
    """Section F's paired percentile bootstrap over whole resampling units.

    Every score vector is indexed by the SAME draw, which is what makes this an interval about a
    contrast rather than about two quantities measured on unrelated resamples. A unit that enters a
    draw enters it in every arm, so the correlation between the arms is carried; for ``S`` that is
    the whole point, since ``L`` and ``D`` are the same rows scored on the same 25 splits.

    A draw whose resampled labels are one class has no ROC-AUC. Section F says such draws are
    discarded and counted, so they are, and the count travels in the result rather than being
    silently absorbed into a smaller denominator.
    """
    values = np.empty(draws, dtype=float)
    usable = 0
    discarded = 0
    for _ in range(draws):
        chosen = rng.integers(0, len(units), size=len(units))
        indices = np.concatenate([units[index] for index in chosen])
        boot_y = labels[indices]
        if boot_y.min() == boot_y.max():
            discarded += 1
            continue
        values[usable] = combine([st._auc_vectorized(boot_y, column[indices])
                                  for column in columns])
        usable += 1
    if usable < 2:
        raise RuntimeError(f"the bootstrap kept {usable} usable draws of {draws}")
    drawn = values[:usable]
    low, high = np.quantile(drawn, (0.025, 0.975))
    lower_tail = (int(np.sum(drawn <= 0)) + 1) / (usable + 1)
    upper_tail = (int(np.sum(drawn >= 0)) + 1) / (usable + 1)
    return {
        "interval_95": [float(low), float(high)],
        "replicates": draws,
        "usable_replicates": usable,
        "discarded_single_class_draws": discarded,
        "bootstrap_mean": float(drawn.mean()),
        "bootstrap_sd": float(drawn.std(ddof=1)),
        "two_sided_tail_p": min(1.0, 2 * min(lower_tail, upper_tail)),
    }


def _seed_averaged(record: dict, arm: str) -> np.ndarray:
    """One arm's five-seed-averaged out-of-fold failure probabilities: section E's input.

    Averaged over the five split seeds and then scored once, which is the declared estimand and NOT
    the mean of the five per-seed pooled AUCs the same array also supports. The record carries both
    under names that say which is which; this is the one the estimands and the intervals are on.
    """
    return np.asarray(record["arms"][arm]["oof_proba"], dtype=float).mean(axis=0)


def _estimand_rows(corpus: str, record: dict, tolerance: float) -> list[dict]:
    """Section E's three estimands on one corpus, each with section F's interval beside it.

    The point is recomputed here from the saved out-of-fold vectors and then required to equal the
    one the runner's ``_effects`` stored, exactly and with no tolerance. Two arms recorded separately
    can each be right while the difference is taken between the wrong pair or read on the wrong
    scale, and this is the comparison that rules that out: the interval and the scorer have to be
    reading the same vectors for a row to be emitted at all.

    The mean of fold AUCs rides along under ``reproduction_quantity`` with no interval attached.
    Section E defines the estimands on the seed-averaged out-of-fold quantity, and the way to keep a
    reader from taking the board-shaped number for the declared one is to store it somewhere an
    interval visibly is not.
    """
    from sklearn.metrics import roc_auc_score

    from catchbench.post_generalization_audit import (
        BOOTSTRAP_BASE_SEED, BOOTSTRAP_DRAWS, LINEAR_CONTRAST, MATCHED_CONTRAST, PRIMARY_ESTIMAND,
        RESAMPLING_UNIT, RNG_LABELS,
    )

    labels = np.asarray(record["labels"], dtype=int)
    units, unit_name = _resampling_units(corpus, record)
    vectors = {name: _seed_averaged(record, name)
               for name in (*LINEAR_CONTRAST, *MATCHED_CONTRAST)}
    scored = {name: float(roc_auc_score(labels, vector)) for name, vector in vectors.items()}

    linear = scored[LINEAR_CONTRAST[0]] - scored[LINEAR_CONTRAST[1]]
    matched = scored[MATCHED_CONTRAST[0]] - scored[MATCHED_CONTRAST[1]]
    points = {"L": linear, "D": matched, "S": linear - matched}
    stored = record["effects"]
    for name, point in points.items():
        if point != stored[name]["seed_averaged_oof_difference"]:
            raise AssertionError(
                f"{corpus} {name}: the estimand recomputed from the saved out-of-fold vectors is "
                f"{point!r} and the runner stored "
                f"{stored[name]['seed_averaged_oof_difference']!r}; the interval below would not "
                "be the interval of the point above it"
            )

    # Which score vectors each draw carries. L and D each need their own pair; S is a difference of
    # differences, so section F makes it carry all four together in every draw.
    carried = {
        "L": ([LINEAR_CONTRAST[0], LINEAR_CONTRAST[1]], lambda a: a[0] - a[1]),
        "D": ([MATCHED_CONTRAST[0], MATCHED_CONTRAST[1]], lambda a: a[0] - a[1]),
        "S": ([LINEAR_CONTRAST[0], LINEAR_CONTRAST[1],
               MATCHED_CONTRAST[0], MATCHED_CONTRAST[1]],
              lambda a: (a[0] - a[1]) - (a[2] - a[3])),
    }

    rows = []
    for name in ("L", "D", "S"):
        arm_names, combine = carried[name]
        label = RNG_LABELS[(corpus, name)]
        interval = _unit_bootstrap(labels, units, [vectors[arm] for arm in arm_names], combine,
                                   BOOTSTRAP_DRAWS, st._rng_for(label, BOOTSTRAP_BASE_SEED))
        resampled_point = combine([st._auc_vectorized(labels, vectors[arm]) for arm in arm_names])
        rows.append({
            "corpus": corpus,
            "estimand": name,
            "cell": f"{corpus}.{name}",
            "definition": {"L": f"AUC({LINEAR_CONTRAST[0]}) - AUC({LINEAR_CONTRAST[1]})",
                           "D": f"AUC({MATCHED_CONTRAST[0]}) - AUC({MATCHED_CONTRAST[1]})",
                           "S": "L - D"}[name],
            "point": points[name],
            "interval_95": interval["interval_95"],
            "interval_method": "paired percentile bootstrap over resampled "
                               f"{unit_name}s, carrying {len(arm_names)} score vector(s) in every "
                               "draw",
            "interval_axis": f"{unit_name} resampling",
            "resampling_unit": RESAMPLING_UNIT[corpus],
            "n_units": len(units),
            "arms_carried_in_every_draw": list(arm_names),
            "rng_label": label,
            "rng_base_seed": BOOTSTRAP_BASE_SEED,
            "role": ("primary" if (corpus, name) == PRIMARY_ESTIMAND else "co-reported"),
            "labelled_as": ("the quantity section E nominated before any of the six existed"
                            if (corpus, name) == PRIMARY_ESTIMAND
                            else "co-reported; section G's branch requires both corpora"),
            "arm_scores": {arm: {"seed_averaged_oof_roc_auc": scored[arm],
                                 "mean_fold_roc_auc": record["arms"][arm]["mean_fold_roc_auc"]}
                           for arm in sorted(vectors)},
            "reproduction_quantity": {
                "mean_fold_difference": stored[name]["mean_fold_difference"],
                "interval": None,
                "why_no_interval": "section E defines the estimands on the seed-averaged "
                                   "out-of-fold quantity. The mean of the 25 fold AUCs is what a "
                                   "board prints and never receives the interval computed for the "
                                   "quantity above it.",
            },
            "point_recomputed_by_the_bootstrap_scorer": {
                "value": float(resampled_point),
                "abs_difference": abs(float(resampled_point) - points[name]),
                "tolerance": tolerance,
                "within_tolerance": bool(abs(float(resampled_point) - points[name]) <= tolerance),
                "why_they_can_differ_at_all": "the bootstrap's AUC is the tie-corrected placement "
                                              "mean and the estimand is scored by sklearn's "
                                              "trapezoid rule; the same quantity computed two ways, "
                                              "agreeing to float noise",
                "tolerance_scope_note": "the declaration scopes PARITY_ATOL to the comparison "
                                        "between the two fold paths. It is applied here to a "
                                        "comparison of the same kind, two computations of one "
                                        "ROC-AUC in one environment, and it is not widened for it.",
            },
            **{key: interval[key] for key in ("replicates", "usable_replicates",
                                              "discarded_single_class_draws", "bootstrap_mean",
                                              "bootstrap_sd", "two_sided_tail_p")},
            "conditional_on_the_saved_fits": (
                "section F: these intervals are conditional on the saved fits. They do not include "
                "resample-and-refit training uncertainty, and folds and seeds are not treated as "
                "independent runs."
            ),
        })
    return rows


# --- the batch --------------------------------------------------------------------------------------


def _fold_locality(task, arms_expected: int):
    """``emit_detection_audit._FoldLocality`` and the summary this audit takes from it.

    Returns the inspector and a closure that assembles the summary. ``record()`` is not used, for
    the two reasons the module docstring gives: it fits 50 further estimators to score a leaked
    basis, which produces no declared quantity, and its prose asserts a measured fact about SWE-Gym
    and tau-bench that is not a fact about these corpora. The per-fit comparisons are the part that
    establishes anything, and they run unchanged inside the batch's own fold loop.
    """
    from catchbench import detection

    from emit_detection_audit import _FoldLocality

    locality = _FoldLocality(task, detection)

    def summary() -> dict:
        if not locality.per_arm:
            raise AssertionError(f"{task.dataset}: no spline-bearing arm was inspected")
        for name, row in sorted(locality.per_arm.items()):
            if row["fits"] != 25:
                raise AssertionError(
                    f"{task.dataset}/{name}: {row['fits']} fits were checked, not the 25 the arm "
                    "is scored on, so the check no longer covers the arm"
                )
            if row["fits_whose_knots_differ_from_the_whole_matrix"] == 0:
                raise AssertionError(
                    f"{task.dataset}/{name}: the fitted knots equal the whole-matrix knots on "
                    "every one of the 25 fits, so this arm has no negative control and the "
                    "training-fold comparison cannot tell a leak from a clean fit"
                )
        return {
            "checked_by": "tools/emit_detection_audit.py::_FoldLocality, driven from the runner's "
                          "inspect hook so every fit is inspected without being refitted",
            "establishes": "for every fit of every spline-bearing arm, the fitted knot vectors are "
                           "exactly those of a transformer fitted on that fold's training rows and "
                           "the fitted scaler's moments are exactly those of the transformed "
                           "training rows; and on at least one fit per arm those knots differ from "
                           "the whole-matrix knots, so the comparison discriminates rather than "
                           "passing vacuously",
            "record_not_called": "_FoldLocality.record() ends by fitting 50 further estimators to "
                                 "score a leaked basis, which produces no declared quantity, and "
                                 "its prose states a measured fact about SWE-Gym and tau-bench "
                                 "that is not a fact about these two corpora",
            "arms_inspected": sorted(locality.per_arm),
            "fits_inspected": sum(row["fits"] for row in locality.per_arm.values()),
            "arms_expected": arms_expected,
            "per_arm": locality.per_arm,
        }

    return locality, summary


def _score_corpus(corpus: str, preflight_record: dict) -> dict:
    """Fit the four declared arms on one corpus, with every check the batch owes bound to the fits.

    The checks run before the arms and after them for different reasons. Section C's population and
    section B's column boundary have to hold before a fit is worth doing. Section D's fold
    assertions run on the partition the arms will be handed, and the comparison against the
    preflight's recorded membership runs on that same object, so the three cannot disagree about
    which partition the batch used.
    """
    from catchbench import post_generalization_audit as audit

    task = audit.PostGeneralizationTask(corpus)
    task.setup()

    population = audit.check_population(task)
    boundaries = audit.check_column_boundaries(task)
    splits = audit.fold_splits(task)
    folds = audit.check_fold_protocol(task, splits)
    membership = audit.fold_membership(splits, len(task.y))
    bound = _check_matches_preflight(corpus, preflight_record, population, boundaries, membership)

    locality, summary = _fold_locality(task, arms_expected=len(audit.audit_arms()))
    audited = audit.post_generalization_audit(task, splits=splits, inspect=locality)
    audited["corpus_line"] = task.corpus_line()

    return {
        "task": task,
        "audited": audited,
        "checks": {
            "population": population,
            "column_boundaries": boundaries,
            "fold_protocol": folds,
            "matches_the_cleared_preflight": bound,
            "training_only_transforms": summary(),
        },
    }


def run_batch(corpora: list[str], preflight_record: dict, tolerance: float) -> dict:
    """Section E's six estimands, section F's six intervals and section G's branch, in one pass.

    A corpus whose fits raise is caught here rather than allowed to end the run. Section J keeps one
    batch and one reporting pass, the two corpora share no fitted object, and a partial result plus
    a recorded failure is worth more than nothing. The traceback is kept, because a recorded failure
    that does not say where it happened cannot be told from a disappointing effect, and those two
    are what section J most needs kept apart.

    Section G's branch needs both corpora and ``classify_branch`` refuses to guess from one, which
    is section G's own "no result at one corpus is promoted to stand for both" written as code. A
    corpus that failed therefore leaves the branch unrecorded rather than leaving it to the survivor.
    """
    from catchbench import post_generalization_audit as audit

    records: dict[str, dict] = {}
    checks: dict[str, dict] = {}
    effects: list[dict] = []
    readings: dict[str, dict] = {}
    failures: list[dict] = []
    warned: list[dict] = []
    overlap_binding = None

    for corpus in corpora:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                scored = _score_corpus(corpus, preflight_record)
                if corpus == "openhands":
                    overlap_binding = _openhands_groups_match_preflight(preflight_record,
                                                                        scored["task"])
                rows = _estimand_rows(corpus, scored["audited"], tolerance)
            for warning in caught:
                warned.append({"corpus": corpus, "category": warning.category.__name__,
                               "message": str(warning.message),
                               "where": f"{Path(warning.filename).name}:{warning.lineno}"})
            records[corpus] = scored["audited"]
            checks[corpus] = scored["checks"]
            effects += rows
            readings[corpus] = audit.corpus_reading(
                {row["estimand"]: row["point"] for row in rows},
                {row["estimand"]: row["interval_95"] for row in rows},
            )
            for arm_id, arm in sorted(scored["audited"]["arms"].items()):
                if arm["nonfinite_fold_scores"] or arm["nonfinite_oof_values"]:
                    failures.append({
                        "kind": "nonfinite_output", "corpus": corpus, "arm": arm_id,
                        "fold_scores": arm["nonfinite_fold_scores"],
                        "oof_values": arm["nonfinite_oof_values"],
                    })
        except Exception as error:  # noqa: BLE001 - section J wants this recorded, not raised
            failures.append({
                "kind": "corpus_batch_raised", "corpus": corpus,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "consequence": f"the three {corpus} estimands are absent from this record and "
                               "section G's branch cannot be classified, because it requires both "
                               "corpora",
            })

    branch = None
    try:
        branch = audit.classify_branch(readings)
    except Exception as error:  # noqa: BLE001 - same reason as the corpus loop above
        failures.append({
            "kind": "branch_not_classified",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        })

    primary_corpus, primary_estimand = audit.PRIMARY_ESTIMAND
    primary = next((row for row in effects if row["corpus"] == primary_corpus
                    and row["estimand"] == primary_estimand), None)
    if primary is None:
        failures.append({
            "kind": "primary_estimand_missing", "corpus": primary_corpus,
            "estimand": primary_estimand,
            "error": "the primary quantity was not computed, so the batch has no primary result",
        })

    return {
        "corpora": records,
        "verification": {"per_corpus": checks,
                         "openhands_ids_match_the_measured_overlap": overlap_binding},
        "effects": effects,
        "primary": primary,
        "branch": branch,
        "failures": failures,
        "warnings": warned,
    }


# --- provenance and the record ----------------------------------------------------------------------


def _inputs(declaration: Path | None, preflight_path: Path, preflight_record: dict,
            corpus_lines: dict) -> dict:
    """Every file this batch read, by path and sha256, and the repositories it read them at.

    The declaration is hashed as it stands now AND compared against the hash the preflight recorded.
    The two differ whenever a dated deviation entry was appended between the two runs, which the
    declaration's own append-only rule provides for, and a reader needs to see the pair rather than
    a single hash that silently belongs to one moment. A difference is reported, never repaired.
    """
    import importlib.metadata
    import os
    import platform

    import agent_failure_detection

    grade = Path(agent_failure_detection.__file__).resolve()
    paper_dir = os.environ.get("CATCHBENCH_PAPER_DIR")
    repositories = {
        "catchbench": {"path": str(ROOT), "commit": _git_commit(ROOT), **_git_worktree(ROOT)},
        "grade": {"path": str(grade.parents[1]), "commit": _git_commit(grade.parents[1]),
                  **_git_worktree(grade.parents[1])},
    }
    if paper_dir:
        repositories["paper"] = {"path": paper_dir, "commit": _git_commit(Path(paper_dir)),
                                 **_git_worktree(Path(paper_dir)), "written_to": False}
    packages = {}
    for package in ("numpy", "scipy", "scikit-learn"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover - environment
            packages[package] = "not installed"
    inputs = {
        "code": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in (
                ("src/catchbench/detection.py", ROOT / "src" / "catchbench" / "detection.py"),
                ("src/catchbench/post_generalization_audit.py",
                 ROOT / "src" / "catchbench" / "post_generalization_audit.py"),
                ("tools/statistical_tests.py", ROOT / "tools" / "statistical_tests.py"),
                ("tools/emit_detection_audit.py", ROOT / "tools" / "emit_detection_audit.py"),
                ("tools/post_generalization_preflight.py",
                 ROOT / "tools" / "post_generalization_preflight.py"),
                ("tools/emit_post_generalization_audit.py", Path(__file__).resolve()),
                ("grade/experiment/agent_failure_detection.py", grade),
                ("grade/experiment/agent_graph_openhands.py",
                 grade.parent / "agent_graph_openhands.py"),
                ("grade/experiment/agent_graph_scienceworld.py",
                 grade.parent / "agent_graph_scienceworld.py"),
                ("grade/experiment/agent_graph_swegym.py", grade.parent / "agent_graph_swegym.py"),
            )
        },
        "preflight_record": {"path": str(preflight_path), "sha256": _sha256(preflight_path)},
        "repositories": repositories,
        "package_versions": packages,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "corpora": corpus_lines,
        "records_consumed_by_the_batch": [preflight_path.name],
        "note": "the batch reads feature matrices and fits arms. The only record it consumes is "
                "the preflight's, and it consumes it as a gate rather than as a source of numbers.",
    }
    if declaration is not None:
        at_preflight = (preflight_record.get("inputs", {}).get("declaration", {}) or {}).get(
            "sha256")
        now = _sha256(declaration)
        inputs["declaration"] = {
            "path": str(declaration),
            "sha256": now,
            "sha256_at_preflight": at_preflight,
            "changed_since_preflight": bool(at_preflight is not None and at_preflight != now),
            "why_it_can_change": "the declaration is append-only once fitting begins and a "
                                 "deviation gets a dated entry at the bottom. Entries appended "
                                 "between the preflight and this batch move the file hash. Both "
                                 "hashes are recorded; neither is repaired.",
        }
    return inputs


def build_result(batch: dict, gate: dict, preflight_path: Path, preflight_record: dict,
                 declaration: Path | None, revisions: dict, started: float) -> dict:
    """The section H record: both corpora, all four arm scores, L, D and S with intervals."""
    from catchbench import detection
    from catchbench import post_generalization_audit as audit

    corpus_lines = {name: record.get("corpus_line")
                    for name, record in sorted(batch["corpora"].items())}
    return {
        "schema_version": "1.0.0",
        "generated_by": "tools/emit_post_generalization_audit.py --run",
        "question": "Contribution (2) rests on two corpora: SWE-Gym's structural increment changes "
                    "sign under a flexible size control and tau-bench's does not. On two further "
                    "corpora, how much of the linear dependency increment survives the same change "
                    "of size control.",
        "declaration": {
            "file": audit.DECLARATION,
            "frozen_utc": "2026-09-11T01:05:49Z",
            "chronology": "proposed after Part A's result and after Round 31 of the review. A "
                          "prospectively specified extension of an exploratory analysis whose "
                          "neighbouring results are known, not an independent confirmatory "
                          "replication, and labelled exploratory whichever way it comes out.",
            "frozen_before_the_run": {
                "declared_population": dict(audit.DECLARED_POPULATION),
                "fold_protocol": dict(audit.FOLD_PROTOCOL),
                "arms": dict(audit.ARM_ROLES),
                "linear_contrast": list(audit.LINEAR_CONTRAST),
                "matched_contrast": list(audit.MATCHED_CONTRAST),
                "estimands": list(audit.ESTIMANDS),
                "primary_estimand": {"corpus": audit.PRIMARY_ESTIMAND[0],
                                     "estimand": audit.PRIMARY_ESTIMAND[1]},
                "rng_labels": {f"{corpus}.{estimand}": label
                               for (corpus, estimand), label in sorted(audit.RNG_LABELS.items())},
                "rng_base_seed": audit.BOOTSTRAP_BASE_SEED,
                "bootstrap_draws": audit.BOOTSTRAP_DRAWS,
                "resampling_unit": dict(audit.RESAMPLING_UNIT),
                "substantial_threshold": audit.SUBSTANTIAL_EFFECT,
                "branches": list(audit.BRANCHES),
                "parity_tolerance": audit.PARITY_ATOL,
                "seeds": [int(seed) for seed in audit.SEEDS],
                "folds": audit.N_SPLITS,
                "spline": dict(detection._SPLINE),
                "declared_fits": 200,
            },
            "fixed_by_this_implementation": {
                "section_g_row_overlap": "section G's first two rows overlap as written, so a "
                                         "first-match reading leaves mixed-informative "
                                         "unreachable. classify_branch sends the boundary case to "
                                         "the row that names it. Written into the module before "
                                         "the batch and before any of the six quantities existed, "
                                         "and confirmed by the declaration's dated entry of "
                                         "2026-09-11.",
                "evidence_of_a_positive_L": "section G's substantial-attenuation condition asks "
                                            "for 'evidence of a positive L' without giving it a "
                                            "test. corpus_reading applies the support-then-size "
                                            "order the rest of section G uses: a positive point "
                                            "whose section F interval excludes zero. Confirmed by "
                                            "the same dated entry.",
                "bootstrap_is_written_in_the_driver": "neither shipped bootstrap can express "
                                                      "section F's unit. _task_clustered_auc_"
                                                      "bootstrap requires equal-sized clusters and "
                                                      "OpenHands's issue groups are ragged; "
                                                      "_stratified_auc_bootstrap cannot produce a "
                                                      "single-class draw and section F requires "
                                                      "such draws to be discarded and counted. "
                                                      "_rng_for, _auc_vectorized and the "
                                                      "percentile rule are reused.",
                "overlap_carried_forward": "section H reports the measured SWE-Gym overlap and the "
                                           "preflight measured it through the raw parquet. It is "
                                           "carried rather than re-measured, and the OpenHands "
                                           "half of the intersection is re-derived here and "
                                           "compared row for row.",
                "fold_locality_summary": "_FoldLocality's per-fit comparisons run inside the "
                                         "batch's fold loop; its record() is not called, because "
                                         "it fits 50 further estimators for a descriptive score "
                                         "and states a measured fact about the two ORIGINAL "
                                         "corpora.",
                "partial_results": "a corpus whose fits raise is recorded with its traceback; "
                                   "section G's branch then stays unrecorded, because it requires "
                                   "both corpora",
            },
        },
        "settings": {
            "confidence_level": 0.95,
            "declared_fits": 200,
            "fits_performed": sum(len(record["arms"]) * len(record["seeds"]) * audit.N_SPLITS
                                  for record in batch["corpora"].values()),
            "corpora_scored": sorted(batch["corpora"]),
            "bootstrap": {
                "method": "paired percentile bootstrap over whole resampling units, single-class "
                          "draws discarded and counted",
                "draws": audit.BOOTSTRAP_DRAWS,
                "percentiles": [0.025, 0.975],
                "interpolation": "linear (numpy.quantile default)",
                "unit": dict(audit.RESAMPLING_UNIT),
                "inner_auc": "statistical_tests._auc_vectorized, the tie-corrected placement mean",
                "s_carries": "all four score vectors in every draw, because S is a difference of "
                             "differences",
            },
        },
        "inputs": _inputs(declaration, preflight_path, preflight_record, corpus_lines),
        "hub_revisions": revisions,
        "hub_revisions_are": "recorded, not enforced, for the two corpora absent from "
                             "catchbench.corpora.CORPUS_REVISIONS; section C's counts are what "
                             "catch a population that moved",
        "preflight": {
            "record": str(preflight_path),
            "cleared": preflight_record.get("cleared"),
            "findings": list(preflight_record.get("findings") or []),
            "fits_of_the_declared_batch": preflight_record.get("fits_of_the_declared_batch"),
            "saved_split_parity": {
                corpus: record.get("saved_split_parity")
                for corpus, record in preflight_record.get("corpora", {}).items()
                if record.get("saved_split_parity") is not None
            },
            "gate": gate,
        },
        # Section H names the measured populations, the fold protocol and the measured overlap as
        # things the record reports in every branch. They are lifted to the top level here rather
        # than left inside the per-corpus verification block, so a reader looking for what section H
        # asks for finds it without knowing how this driver files its checks.
        "populations": {corpus: checks["population"]
                        for corpus, checks in batch["verification"]["per_corpus"].items()},
        "fold_protocol": {corpus: checks["fold_protocol"]
                          for corpus, checks in batch["verification"]["per_corpus"].items()},
        "swegym_overlap": preflight_record.get("swegym_overlap"),
        "corpora": batch["corpora"],
        "verification": batch["verification"],
        "effects": batch["effects"],
        "primary": batch["primary"],
        "branch": batch["branch"],
        "failures": batch["failures"],
        "warnings": batch["warnings"],
        "board_entrants_added": [],
        "rosters_unchanged": "detection._LOADERS, every board roster and live_streaming_methods() "
                             "are untouched; the nine boards, the 72 entrants and the 138-record "
                             "registry are what they were",
        "interpretation": {
            "exploratory": "the declaration's chronology section: this audit was proposed after "
                           "Part A's result and after Round 31, on data whose neighbouring results "
                           "are known. It stays labelled exploratory whichever way it comes out.",
            "grouped_protocol_differs": "section D: the grouped OpenHands protocol differs from the "
                                        "row-split protocol the original POST boards use. The "
                                        "manuscript states that difference rather than presenting "
                                        "the numbers as like for like.",
            "not_a_board": "section I: neither corpus is a board evaluation and CatchBench's POST "
                           "board does not cover four corpora.",
            "no_disjoint_issue_set": "section I as its dated entry of 2026-09-11 amends it: the "
                                     "measured overlap is nonzero, so the issue collections may "
                                     "not be described as disjoint and the shared issue is "
                                     "disclosed in the appendix.",
            "no_independent_scaffold": "section I: SWE-Gym's scored runs already use the OpenHands "
                                       "scaffold and GRADE's SWE-Gym adapter imports "
                                       "agent_graph_openhands.to_steps directly. This is a change "
                                       "of producing model and issue collection.",
            "no_unseen_task_family": "section I: inference on ScienceWorld is conditional on the "
                                     "selected source file and its measured task-family mixture, "
                                     "which is uneven.",
            "failing_to_reject_is_not_stability": "section G: an interval for S that includes zero "
                                                  "while also including the declared magnitude is "
                                                  "unresolved and is reported as unresolved.",
            "one_batch": "section J: one declared fit batch and one reporting pass. A disappointing "
                         "effect is not an implementation defect and is not grounds for a rerun.",
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
    }


def _report(result: dict) -> None:
    print(f"\nPOST generalization audit :: declared batch ({result['declaration']['file']})")
    frozen = result["declaration"]["frozen_before_the_run"]
    print(f"  L = AUC({frozen['linear_contrast'][0]}) - AUC({frozen['linear_contrast'][1]})")
    print(f"  D = AUC({frozen['matched_contrast'][0]}) - AUC({frozen['matched_contrast'][1]})")
    print("  S = L - D, all on five-seed-averaged out-of-fold failure probabilities")
    for corpus, record in sorted(result["corpora"].items()):
        print(f"\n[{corpus}] {record['corpus_line']}")
        for arm_id, arm in sorted(record["arms"].items()):
            print(f"    {arm_id:<28} oof {arm['seed_averaged_oof_roc_auc']:.6f}   "
                  f"fold-mean {arm['mean_fold_roc_auc']:.6f}   ({arm['role']})")
    header = f"\n  {'cell':<24}{'point':>11}{'low':>11}{'high':>11}   role"
    print(header)
    print("  " + "-" * (len(header) - 4))
    for row in result["effects"]:
        low, high = row["interval_95"]
        print(f"  {row['cell']:<24}{row['point']:>+11.6f}{low:>+11.6f}{high:>+11.6f}   "
              f"{row['role']}")
    branch = result["branch"]
    if branch is not None:
        print(f"\n  section G branch: {branch['branch'].upper()}")
        for corpus, reading in sorted(branch["per_corpus"].items()):
            print(f"    {corpus:<14} substantial attenuation {reading['substantial_attenuation']}, "
                  f"stable-positive boundary "
                  f"{reading['informative_stable_positive_boundary']}, "
                  f"positive L {reading['evidence_of_a_positive_linear_increment']}")
    if result["failures"]:
        print(f"\n  {len(result['failures'])} recorded failure(s):")
        for failure in result["failures"]:
            print(f"    [{failure['kind']}] {failure.get('corpus', '-')}: "
                  f"{failure.get('error', '')}")
    else:
        print("\n  no recorded failures")
    if result["warnings"]:
        print(f"  {len(result['warnings'])} warning(s) captured:")
        for entry in result["warnings"]:
            print(f"    [{entry['corpus']}] {entry['category']} at {entry['where']}: "
                  f"{entry['message']}")
    else:
        print("  no warnings captured")


def _write(path: Path, payload: dict) -> None:
    """Write JSON through a temporary sibling, so a run that dies mid-write leaves the old record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(st._jsonable(payload), indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = _parse_args()
    started = time.time()
    from catchbench.post_generalization_audit import PARITY_ATOL

    preflight_record = json.loads(args.preflight_record.read_text(encoding="utf-8"))
    gate = preflight_gate(preflight_record, list(args.corpora))
    if not gate["admits_the_batch"]:
        print(f"\nthe preflight does not admit the batch: cleared={gate['cleared']}, "
              f"{len(gate['findings'])} finding(s), "
              f"fits_of_the_declared_batch={gate['fits_of_the_declared_batch']}, "
              f"missing corpora {gate['corpora_missing']}")
        for finding in gate["findings"]:
            print(f"  {finding}")
        return 1

    _wrap_hub_listing()
    revisions = {} if args.no_hub_revisions else _hub_revisions()
    if revisions:
        print("\nhub heads resolved:")
        for name, entry in sorted(revisions.items()):
            print(f"  {name:<26} {entry['repo_id']}  {entry['head']}")
        for name in REPOSITORIES:
            if name not in revisions:  # pragma: no cover - _hub_revisions covers every key
                print(f"  {name:<26} unresolved")

    batch = run_batch(list(args.corpora), preflight_record, PARITY_ATOL)
    payload = build_result(batch, gate, args.preflight_record, preflight_record, args.declaration,
                           revisions, started)
    _write(args.output, payload)
    _report(payload)
    print(f"\nwrote {args.output}")
    return 1 if payload["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
