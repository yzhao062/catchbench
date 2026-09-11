r"""Run the declared LIVE prefix size control, and the preflight that has to clear before it does.

The analysis is Part A of ``research/catchbench-live-size-control-declaration-2026-09-10.md``,
frozen before any prefix spline was fitted. ``catchbench.live_size_audit`` holds the arms and the
scorer; this script is the driver, and the split is the same one ``emit_detection_audit.py`` keeps
from ``catchbench.detection``.

Two modes, and the default is the cautious one::

    GRADE_DIR=/path/to/grade python tools/emit_live_size_audit.py --preflight-only
    GRADE_DIR=/path/to/grade python tools/emit_live_size_audit.py --run

``--preflight-only`` runs A10 and writes its record. It fits the two LINEAR arms at all four
prefixes, because reproducing the published prefix numbers is what admits the route, and it fits the
two spline-bearing arms at the 100 percent prefix ONLY, because that cell is a reproduction control
whose value is already committed and because reading a prefix the declaration nominates as primary
before the batch would defeat the point of nominating it. The spline arms at 25, 50 and 75 percent
are the batch's to fit, and the preflight refuses to fit them.

A10's five checks, and where each is answered:

  1. the mixed arm's column boundary at every prefix    ``live_size_audit.check_column_boundaries``
  2. exact row and fold alignment across arms/prefixes  ``check_row_and_fold_alignment``
  3. training-only transforms                           ``emit_detection_audit._FoldLocality``
  4. linear-path parity with the published numbers      ``_check_published_prefix_numbers``
  5. the 100 percent cells reproduce the POST controls  ``_check_hundred_percent_matches_post``

``--run`` clears that gate and then fits the declared batch: four arms at four prefixes on two
corpora, five split seeds and five folds each, 800 fits. It computes ``D(c, p)`` as A5 defines it,
attaches A6's interval to each of the eight, computes A7's temporal contrast ``T``, applies A8's
branch to the primary cell, and writes all of it to ``tools/live_size_control_audit_results.json``.
A11 makes that one batch and one reporting pass, so nothing here retries a cell, and a corpus whose
fits raise is caught, recorded as a failure with its traceback, and the other corpus still finishes.

The gate is A10.4 as the declaration's dated entry of 2026-09-10 narrows it. The preflight's own
``cleared`` flag is left exactly as computed and is not what admits the batch: two diagnostic fields
inside ``statistical_tests_results.json`` do not reproduce, and that entry records them as findings
about that artifact rather than as a failure of this route. ``_parity_gate`` is that reading written
as code, read off the preflight's structured comparisons rather than off its message strings, and it
still stops the batch on any disagreement in a published number. ``PARITY_ATOL`` is untouched.

Nothing here is a board entrant and nothing here changes the LIVE board. ``live_streaming_methods``
is untouched, so the printed board, its entrant inventory and ``tests/golden/board.txt`` are what
they were.

Two reuse decisions are worth stating, because both are load-bearing rather than tidy. The fold
locality check is ``emit_detection_audit._FoldLocality`` itself, driven through a prefix view: it
reads ``task.layers`` and ``arm.layer`` and nothing else, so it verifies the prefix fits by the same
array comparisons it verifies the endpoint fits by, and a weakening of that check cannot reach one
audit without reaching the other. The tau-bench run keys come from
``statistical_tests._tau_live_clustering``, which already replays the corpus under LIVE's own
``>= 4`` filter and verifies its alignment against LIVE's labels and 100 percent step counts; a
second replay here would be a second thing to keep true.

Comparisons are on unrounded values against ``live_size_audit.PARITY_ATOL``, which was written into
the module before any prefix was fitted. Warnings are captured and non-finite outputs are counted,
so a run that produced a score by way of a numerical complaint says so rather than printing the
score alone.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

# The shipped DeLong, the two bootstraps, the RNG derivation and the LIVE task clustering. A6 names
# this module's helpers by file and line, so the batch reaches them rather than restating them; a
# second copy of the tau-bench clustering or of the percentile rule would be a second specification
# to keep in step with the one the committed intervals came from.
import statistical_tests as st  # noqa: E402

CORPORA = ("swegym", "tau")
CORPUS_NAMES = {"swegym": "SWE-Gym", "tau": "tau-bench"}

# The corpus token the statistical record's LIVE claim ids are written with. It is not the loader's
# corpus name, and reading one for the other would compare SWE-Gym numbers against tau-bench cells.
CLAIM_CORPUS = {"swegym": "swe", "tau": "tau"}

# The prefix at which the LIVE cells and the POST audit answer the same question. Every other prefix
# has no POST counterpart, which is the whole reason this audit exists.
ENDPOINT = 1.00

STATISTICAL_RECORD = ROOT / "tools" / "statistical_tests_results.json"
POST_AUDIT_RECORD = ROOT / "tools" / "detection_audit_results.json"
DEFAULT_OUTPUT = ROOT / "tools" / "live_size_audit_preflight.json"
DEFAULT_RESULT_OUTPUT = ROOT / "tools" / "live_size_control_audit_results.json"

# The two arms whose published prefix numbers admit the route, and the two that are the control.
LINEAR_ARMS = ("size (flat)", "auditable (size+deps)")
SPLINE_ARMS = ("size (spline)", "size-spline + linear-deps")

# The record field the A10.4 deviation entry of 2026-09-10 names. ``_check_published_prefix_numbers``
# reads ``variance_axes.board_point_estimate.value`` into ``mean_fold_roc_auc``, and it is that field,
# in that record, that the entry treats as a finding about the artifact rather than as a published
# number the route has to reproduce. Nothing else is exempt: the same quantity compared against the
# POST audit still stops the batch, because the entry establishes that the endpoint reproduces there
# at 0.000e+00 on both estimands.
RECORD_DIAGNOSTIC = {
    "source": STATISTICAL_RECORD.name,
    "quantity": "mean_fold_roc_auc",
    "field": "variance_axes.board_point_estimate.value",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true",
                      help="run A10 and stop; fits no spline arm away from the endpoint")
    mode.add_argument("--run", action="store_true",
                      help="run the preflight and then the declared fit batch")
    parser.add_argument("--corpora", nargs="+", choices=CORPORA, default=list(CORPORA))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--result-output", type=Path, default=DEFAULT_RESULT_OUTPUT,
                        help="where --run writes the declared batch; ignored by --preflight-only")
    parser.add_argument("--declaration", type=Path, default=None,
                        help="the frozen Part A declaration, recorded by path and sha256")
    parser.add_argument("--no-corpus-preflight", action="store_true",
                        help="skip the Hub revision preflight; for an offline rerun over a cache "
                             "a previous verified run already populated")
    return parser.parse_args()


def _close(observed: float, recorded: float, tolerance: float) -> bool:
    """Unrounded absolute comparison. No relative term: these are ROC-AUCs on one bounded scale."""
    return bool(np.isfinite(observed) and np.isfinite(recorded)
                and abs(float(observed) - float(recorded)) <= tolerance)


# --- A10.4: the linear arms reproduce the published prefix numbers -------------------------------


def _published_prefix_cells(record: dict, corpus: str) -> dict:
    """The committed LIVE prefix cells for one corpus, keyed by (arm, prefix).

    Each cell carries two different numbers and the audit is checked against both. ``estimate.a`` is
    the ROC-AUC of the seed-averaged out-of-fold probabilities, which is the estimand and the scale
    the published ``+0.101`` contrast is on. ``variance_axes.board_point_estimate.value`` is the
    mean of the 25 fold AUCs, which is what the board prints. Checking only one of them would leave
    the other free to move.
    """
    token = CLAIM_CORPUS[corpus]
    cells = {}
    for claim in record["claims"]:
        claim_id = claim["id"]
        if not claim_id.startswith(f"live.{token}.bar."):
            continue
        _, _, _, prefix, method = claim_id.split(".", 4)
        cells[(method, prefix)] = {
            "seed_averaged_oof_roc_auc": float(claim["estimate"]["a"]),
            "mean_fold_roc_auc": float(claim["variance_axes"]["board_point_estimate"]["value"]),
            "n_runs": int(claim["variance_axes"]["run_sampling"]["n"]),
            "claim_id": claim_id,
        }
    return cells


def _check_route_is_the_board(view, detection) -> dict:
    """The swapped-estimator loop reproduces GRADE's ``_cv`` on the prefix matrices, cell for cell.

    ``emit_detection_audit._check_folds_are_the_boards`` establishes this at the endpoint and the
    same argument holds at a prefix: ``_cv_estimator`` differs from ``_cv`` in the estimator it is
    handed, so handing it the estimator ``_cv`` builds must return ``_cv``'s array exactly. This is
    the strongest available form of A10's linear-path parity, because ``_cv`` is what
    ``_PrefixLayerAUC`` calls and therefore what the LIVE board itself prints.

    Exact equality, no tolerance: an approximate match would mean the folds, the rows or the model
    moved, and every contrast measured through this route would carry that move.
    """
    checked = {}
    for layer in ("flat", "flatdep"):
        matrix = view.layers[layer]
        theirs = detection._cv(matrix, view.y)
        ours = detection._cv_estimator(detection._linear_estimator, matrix, view.y)
        if not np.array_equal(theirs, ours):
            raise AssertionError(
                f"{view.dataset}/{layer}: the swapped-estimator loop does not reproduce GRADE's "
                f"_cv; max |difference| {np.abs(theirs - ours).max():.3e}"
            )
        checked[layer] = float(theirs.mean())
    return checked


def _check_published_prefix_numbers(corpus: str, scored: dict, record: dict,
                                    tolerance: float) -> tuple[dict, list[str]]:
    """Both linear arms, both estimands, all four prefixes, against the committed record.

    Returns its findings rather than raising on the first one. A partial comparison cannot say
    whether a disagreement is one cell or the whole route, and that difference decides what the
    finding means, so every cell is compared and the record carries all of them. The caller refuses
    to run the batch when the list is nonempty; nothing here relaxes ``tolerance`` to empty it.

    Each cell carries two numbers and both are checked. ``estimate.a`` is the declared estimand.
    ``variance_axes.board_point_estimate.value`` is the mean of the 25 fold AUCs, which the
    statistical record computes through its own ``supervised_oof`` rather than through GRADE's
    ``_cv``; the two paths score the same fitted models by ``predict_proba`` and by
    ``decision_function`` respectively, so they can differ where the sigmoid ties two margins that
    were distinct. ``_check_route_is_the_board`` is the comparison against the board's own path.
    """
    published = _published_prefix_cells(record, corpus)
    if not published:
        raise AssertionError(f"{corpus}: the statistical record carries no LIVE prefix cells")
    checked = {}
    failures = []
    for prefix_key, cell in sorted(scored.items(), key=lambda item: int(item[0])):
        for arm in LINEAR_ARMS:
            want = published.get((arm, prefix_key))
            if want is None:
                raise AssertionError(f"{corpus}: no published cell for {arm} at {prefix_key}%")
            got = cell["arms"][arm]
            row = {"claim_id": want["claim_id"]}
            for quantity in ("seed_averaged_oof_roc_auc", "mean_fold_roc_auc"):
                difference = float(got[quantity]) - want[quantity]
                agrees = _close(got[quantity], want[quantity], tolerance)
                row[quantity] = {"observed": float(got[quantity]),
                                 "published": want[quantity],
                                 "difference": difference,
                                 "within_tolerance": agrees}
                if not agrees:
                    failures.append(f"{corpus} {prefix_key}% {arm} {quantity}: "
                                    f"{got[quantity]!r} against published {want[quantity]!r} "
                                    f"(difference {difference:+.3e})")
            checked[f"{prefix_key}/{arm}"] = row
    return {
        "source": STATISTICAL_RECORD.name,
        "tolerance": tolerance,
        "cells_compared": len(checked),
        "quantities_compared": 2 * len(checked),
        "quantities_disagreeing": len(failures),
        "max_absolute_difference": max(
            abs(row[quantity]["difference"])
            for row in checked.values()
            for quantity in ("seed_averaged_oof_roc_auc", "mean_fold_roc_auc")
        ),
        "per_cell": checked,
    }, failures


def _check_published_linear_contrast(corpus: str, scored: dict, record: dict,
                                     tolerance: float) -> tuple[dict, list[str]]:
    """The published contrast itself, not only the two arms it is built from.

    Two arms can each be right while their difference is read on the wrong scale, which is the
    mistake A5 names: the SWE-Gym 25 percent contrast is ``+0.101`` on seed-averaged out-of-fold
    scores and about ``0.112`` on the displayed board numbers. Both are recomputed and both are
    checked against the arms they came from, so the artifact carries the pair rather than one number
    a later reader could take for the other.
    """
    published = _published_prefix_cells(record, corpus)
    high, low = "auditable (size+deps)", "size (flat)"
    rows = {}
    failures = []
    for prefix_key, cell in sorted(scored.items(), key=lambda item: int(item[0])):
        want = {quantity: published[(high, prefix_key)][quantity]
                - published[(low, prefix_key)][quantity]
                for quantity in ("seed_averaged_oof_roc_auc", "mean_fold_roc_auc")}
        got = cell["effects"][f"{high} - {low}"]
        pairs = (("seed_averaged_oof_roc_auc", got["seed_averaged_oof_difference"]),
                 ("mean_fold_roc_auc", got["mean_fold_difference"]))
        row = {}
        for quantity, observed in pairs:
            agrees = _close(observed, want[quantity], tolerance)
            row[quantity] = {"observed": float(observed), "published": want[quantity],
                             "difference": float(observed) - want[quantity],
                             "within_tolerance": agrees}
            if not agrees:
                failures.append(
                    f"{corpus} {prefix_key}% {high} - {low} on {quantity}: {observed!r} against "
                    f"the published {want[quantity]!r}"
                )
        rows[prefix_key] = row
    return {"contrast": f"{high} - {low}", "tolerance": tolerance, "per_prefix": rows}, failures


# --- A10.5: the 100 percent cells reproduce the POST controls -------------------------------------


def _check_hundred_percent_matches_post(corpus: str, live_task, endpoint: dict,
                                        post_record: dict,
                                        tolerance: float) -> tuple[dict, list[str]]:
    """The endpoint LIVE cell and the POST audit are the same cell, checked on inputs and on scores.

    LIVE filters at ``len(steps) >= 4`` and POST at ``>= 2``, so the two populations coinciding is a
    fact about today's corpora rather than a guarantee. The scores can only be compared once the
    inputs are shown to be identical, and identical here means exactly identical: the labels, the
    ``flat`` matrix and the ``flatdep`` matrix compared with ``array_equal`` and no tolerance, since
    an approximate match would mean these are not the same rows.

    Only then are the four arms' scores compared against the committed POST audit, on both
    estimands. The POST record stores per-seed out-of-fold probabilities rather than their pooled
    AUC, so that quantity is recomputed from the stored arrays instead of read.
    """
    from sklearn.metrics import roc_auc_score

    from catchbench.detection import PostDetection

    post = PostDetection(corpus)
    post.setup()
    live_labels = np.asarray(live_task.y)
    post_labels = np.asarray(post.y)
    if not np.array_equal(live_labels, post_labels):
        raise AssertionError(
            f"{corpus}: the LIVE population is {len(live_labels)} runs and the POST population is "
            f"{len(post_labels)}, or their labels differ; the endpoint cells are not the same cell"
        )
    live_layers = live_task.layers_at[ENDPOINT]
    shared = sorted(set(live_layers) & set(post.layers))
    for layer in ("flat", "flatdep"):
        if layer not in shared:
            raise AssertionError(f"{corpus}: the {layer!r} layer is missing from one side")
    for layer in shared:
        if not np.array_equal(live_layers[layer], post.layers[layer]):
            moved = int(np.count_nonzero(live_layers[layer] != post.layers[layer]))
            raise AssertionError(
                f"{corpus}/{layer}: the 100% LIVE matrix differs from the POST matrix in {moved} "
                "cells, so the endpoint arms are not reading the POST control's inputs"
            )

    committed = post_record["corpora"][post.dataset]
    recorded_labels = np.asarray(committed["labels"])
    if not np.array_equal(recorded_labels, post_labels):
        raise AssertionError(f"{corpus}: the committed POST audit labels are not today's labels")

    scores = {}
    failures = []
    for arm_id, arm in endpoint["arms"].items():
        stored = committed["arms"].get(arm_id)
        if stored is None:
            raise AssertionError(f"{corpus}: the committed POST audit has no arm {arm_id!r}")
        if stored["layer"] != arm["layer"]:
            raise AssertionError(f"{corpus} {arm_id}: layer {arm['layer']!r} against the POST "
                                 f"audit's {stored['layer']!r}")
        stored_oof = np.asarray(stored["oof_proba"], dtype=float)
        want = {
            "mean_fold_roc_auc": float(stored["mean_roc_auc"]),
            "seed_averaged_oof_roc_auc": float(roc_auc_score(post_labels,
                                                             stored_oof.mean(axis=0))),
        }
        row = {}
        for quantity, recorded in want.items():
            observed = float(arm[quantity])
            agrees = _close(observed, recorded, tolerance)
            row[quantity] = {"observed": observed, "post_audit": recorded,
                             "difference": observed - recorded, "within_tolerance": agrees}
            if not agrees:
                failures.append(
                    f"{corpus} 100% {arm_id} {quantity}: {observed!r} against the POST audit's "
                    f"{recorded!r}"
                )
        scores[arm_id] = row
    return {
        "source": POST_AUDIT_RECORD.name,
        "tolerance": tolerance,
        "n_runs": int(len(post_labels)),
        "n_failed": int(post_labels.sum()),
        "layers_compared_exactly": shared,
        "establishes": "the 100% LIVE feature matrices and labels are the POST detection inputs "
                       "cell for cell, and all four arms reproduce the committed POST audit scores "
                       "on both estimands",
        "per_arm": scores,
        "quantities_disagreeing": len(failures),
        "max_absolute_difference": max(abs(row[quantity]["difference"])
                                       for row in scores.values() for quantity in row),
    }, failures


# --- the preflight ---------------------------------------------------------------------------


def _score_cell(arms, view, locality):
    """Score the given arms at one prefix, with each fit handed to the locality check."""
    from catchbench.live_size_audit import score_arm

    scored = {}
    for arm in arms:
        watcher = None
        if locality is not None:
            def watcher(seed, fold, train, fitted, arm=arm):
                locality(arm, seed, fold, train, fitted)
        scored[arm.method_id] = score_arm(arm, view, inspect=watcher)
    return scored


def preflight(corpora: list[str], corpus_preflight: bool) -> dict:
    """Every A10 check, on every requested corpus, before any prefix score is interpreted.

    Two kinds of failure and they are handled differently on purpose.

    A structural check raises. A moved population, a moved column boundary, a split that differs
    between arms, a fitted transformer that saw its own test fold, or a 100 percent matrix that is
    not the POST matrix all mean the cells are not the cells the declaration describes, and no
    record of such a run is worth writing.

    A parity comparison against a number a previous run recorded is collected instead. Whether a
    disagreement is one cell or the whole route is what decides what it means, and stopping at the
    first one cannot tell those apart. Every comparison therefore runs, all of them are recorded,
    and the caller refuses the batch when any failed. Nothing here relaxes the tolerance to empty
    that list; A10 fixed it before the first fit for exactly this situation.
    """
    from catchbench import detection
    from catchbench import live_size_audit as audit
    from catchbench.corpora import verify_corpus_heads, verify_pinned_fetches
    from catchbench.live import LiveStreaming

    from emit_detection_audit import _FoldLocality

    names = {CORPUS_NAMES[corpus] for corpus in corpora}
    revisions = verify_corpus_heads(names=names) if corpus_preflight else {}

    statistical = json.loads(STATISTICAL_RECORD.read_text(encoding="utf-8"))
    post_record = json.loads(POST_AUDIT_RECORD.read_text(encoding="utf-8"))

    per_corpus: dict[str, dict] = {}
    captured: list[dict] = []
    findings: list[str] = []
    for corpus in corpora:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            task = LiveStreaming(corpus)
            task.setup()
            if corpus_preflight:
                verify_pinned_fetches(names={CORPUS_NAMES[corpus]})

            population = audit.check_population(task)
            prefixes = audit.check_prefix_set(task)
            boundaries = audit.check_column_boundaries(task)
            alignment = audit.check_row_and_fold_alignment(task)

            arms = {arm.method_id: arm for arm in audit.live_size_audit_arms()}
            linear = [arms[name] for name in LINEAR_ARMS]
            spline = [arms[name] for name in SPLINE_ARMS]

            scored: dict[str, dict] = {}
            locality_record = None
            board_route: dict[str, dict] = {}
            for prefix in task.prefixes:
                view = audit.prefix_view(task, prefix)
                board_route[audit._prefix_key(prefix)] = _check_route_is_the_board(view, detection)
                cell = _score_cell(linear, view, None)
                if prefix == ENDPOINT:
                    # The only prefix whose spline arms the preflight is allowed to fit: it is a
                    # declared reproduction control and its value is already committed, so nothing
                    # the declaration nominates as primary is read here.
                    locality = _FoldLocality(view, detection)
                    cell.update(_score_cell(spline, view, locality))
                    locality_record = locality.record()
                scored[audit._prefix_key(prefix)] = {
                    "prefix": float(prefix),
                    "arms": cell,
                    "effects": audit._effects(cell),
                }
            fitted_away_from_endpoint = {
                key: sorted(set(cell["arms"]) & set(SPLINE_ARMS))
                for key, cell in scored.items() if key != audit._prefix_key(ENDPOINT)
            }
            if any(fitted_away_from_endpoint.values()):
                raise AssertionError(
                    "the preflight fitted a spline arm away from the endpoint: "
                    f"{fitted_away_from_endpoint}"
                )

            published, published_findings = _check_published_prefix_numbers(
                corpus, scored, statistical, audit.PARITY_ATOL)
            contrast, contrast_findings = _check_published_linear_contrast(
                corpus, scored, statistical, audit.PARITY_ATOL)
            endpoint_match, endpoint_findings = _check_hundred_percent_matches_post(
                corpus, task, scored[audit._prefix_key(ENDPOINT)], post_record, audit.PARITY_ATOL)
            findings += published_findings + contrast_findings + endpoint_findings

            nonfinite = {
                f"{key}/{arm_id}": {"fold_scores": arm["nonfinite_fold_scores"],
                                    "oof_values": arm["nonfinite_oof_values"]}
                for key, cell in scored.items() for arm_id, arm in cell["arms"].items()
                if arm["nonfinite_fold_scores"] or arm["nonfinite_oof_values"]
            }
            if nonfinite:
                raise AssertionError(f"{corpus}: non-finite output in {sorted(nonfinite)}")

        for warning in caught:
            captured.append({"corpus": corpus, "category": warning.category.__name__,
                             "message": str(warning.message),
                             "where": f"{Path(warning.filename).name}:{warning.lineno}"})

        per_corpus[corpus] = {
            "population": population,
            "prefix_set": prefixes,
            "column_boundaries": boundaries,
            "row_and_fold_alignment": alignment,
            "training_only_transforms": locality_record,
            "swapped_estimator_reproduces_grade_cv": board_route,
            "published_prefix_numbers": published,
            "published_linear_contrast": contrast,
            "hundred_percent_matches_post": endpoint_match,
            "nonfinite_output": nonfinite,
            "linear_arm_scores": {
                key: {arm_id: {"mean_fold_roc_auc": arm["mean_fold_roc_auc"],
                               "seed_averaged_oof_roc_auc": arm["seed_averaged_oof_roc_auc"]}
                      for arm_id, arm in cell["arms"].items()}
                for key, cell in scored.items()
            },
        }

    return {
        "metadata": _metadata(revisions, corpora),
        "cleared": not findings,
        "findings": findings,
        "checks": {
            "a10_1_column_boundary": "catchbench.live_size_audit.check_column_boundaries, through "
                                     "detection._MixedSplineAUC.columns at every prefix",
            "a10_2_row_and_fold_alignment":
                "catchbench.live_size_audit.check_row_and_fold_alignment",
            "a10_3_training_only_transforms":
                "emit_detection_audit._FoldLocality, at the 100% prefix on each spline arm",
            "a10_4_linear_path_parity": "_check_route_is_the_board against GRADE's _cv (exact), "
                                        "then _check_published_prefix_numbers and "
                                        "_check_published_linear_contrast against the committed "
                                        "statistical record",
            "a10_5_endpoint_matches_post": "_check_hundred_percent_matches_post",
        },
        "tolerance": {
            "value": _tolerance_value(),
            "scope": "every ROC-AUC compared against a number a previous run recorded",
            "exact_comparisons": ["fold index arrays", "the mixed arm's column boundary",
                                  "the 100% LIVE feature matrices against the POST matrices",
                                  "fitted knot vectors and scaler moments"],
            "declared": "in catchbench.live_size_audit.PARITY_ATOL, before any prefix was fitted",
        },
        "warnings": captured,
        "corpora": per_corpus,
    }


def _tolerance_value() -> float:
    from catchbench.live_size_audit import PARITY_ATOL

    return PARITY_ATOL


def _metadata(revisions: dict, corpora: list[str]) -> dict:
    from catchbench import live_size_audit as audit
    from catchbench import detection

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "corpus_revisions": revisions,
        "declaration": audit.DECLARATION,
        "corpora": list(corpora),
        "prefixes": [float(prefix) for prefix in audit.AUDIT_PREFIXES],
        "declared_population": dict(audit.DECLARED_POPULATION),
        "arm_roles": dict(audit.ARM_ROLES),
        "spline": dict(detection._SPLINE),
        "primary_cell": {"corpus": audit.PRIMARY_CELL[0], "prefix": audit.PRIMARY_CELL[1]},
        "rng_labels": {f"{corpus}.{int(round(prefix * 100))}": label
                       for (corpus, prefix), label in sorted(audit.RNG_LABELS.items())},
        "temporal_rng_label": audit.TEMPORAL_RNG_LABEL,
        "board_entrants_added": [],
        "note": "no arm here is a board entrant; live_streaming_methods() and the CLI roster are "
                "unchanged and the LIVE board reports what it reported",
    }


def _report(result: dict) -> None:
    print(f"\nLIVE prefix size control :: preflight ({result['metadata']['declaration']})")
    print(f"  tolerance {result['tolerance']['value']:.1e}, declared "
          f"{result['tolerance']['declared']}")
    for corpus, record in result["corpora"].items():
        population = record["population"]
        print(f"\n[{corpus}] {population['n_runs']} runs "
              f"({population['n_failed']} failed), declared {population['declared']}")
        boundary = next(iter(record["column_boundaries"]["per_prefix"].values()))
        print(f"  column boundary  : splined {len(boundary['splined_columns'])} of "
              f"{boundary['flatdep_shape'][1]} at all "
              f"{len(record['column_boundaries']['per_prefix'])} prefixes")
        alignment = record["row_and_fold_alignment"]
        print(f"  row/fold alignment: {alignment['splits_compared']} splits compared, all equal")
        locality = record["training_only_transforms"]
        print(f"  training-only    : {locality['fits_inspected']} spline-bearing fits inspected "
              f"({', '.join(locality['arms_inspected'])})")
        route = record["swapped_estimator_reproduces_grade_cv"]
        print(f"  board route      : GRADE _cv reproduced cell for cell at all {len(route)} "
              f"prefixes on flat and flatdep")
        published = record["published_prefix_numbers"]
        agreeing = published["quantities_compared"] - published["quantities_disagreeing"]
        print(f"  linear parity    : {agreeing}/{published['quantities_compared']} recorded "
              f"quantities within {published['tolerance']:.0e}, max |difference| "
              f"{published['max_absolute_difference']:.3e}")
        endpoint = record["hundred_percent_matches_post"]
        print(f"  100% vs POST     : layers {', '.join(endpoint['layers_compared_exactly'])} equal "
              f"cell for cell, {endpoint['quantities_disagreeing']} score(s) outside tolerance, "
              f"max |difference| {endpoint['max_absolute_difference']:.3e}")
        for key, cell in sorted(record["linear_arm_scores"].items(), key=lambda item: int(item[0])):
            row = "  ".join(f"{name}: fold {arm['mean_fold_roc_auc']:.6f} "
                            f"oof {arm['seed_averaged_oof_roc_auc']:.6f}"
                            for name, arm in sorted(cell.items()))
            print(f"    {key + '%':>5s}  {row}")
    if result["warnings"]:
        print(f"\n  {len(result['warnings'])} warning(s) captured:")
        for entry in result["warnings"]:
            print(f"    [{entry['corpus']}] {entry['category']} at {entry['where']}: "
                  f"{entry['message']}")
    else:
        print("\n  no warnings captured")
    if result["cleared"]:
        print("\npreflight cleared: every recorded quantity reproduces within the declared "
              "tolerance")
        return
    print(f"\npreflight DID NOT clear: {len(result['findings'])} recorded quantity/quantities "
          "outside the declared tolerance")
    for finding in result["findings"]:
        print(f"  {finding}")
    print("\nThe declaration fixed the tolerance before the first fit. Widening it here is not a "
          "resolution;\nthe disagreement is the finding.")


# --- the A10.4 gate, as the 2026-09-10 deviation entry narrows it --------------------------------


def _disagreement(values: dict, reference_key: str) -> dict:
    """One failed parity comparison, with the two record shapes normalized onto one set of keys.

    The two checks name the number they compare against differently, ``published`` against the
    statistical record and ``post_audit`` against the POST audit, because each says which artifact
    it read. The gate keys on the artifact rather than on the key name, so the difference is
    flattened here instead of being carried into the classification.
    """
    return {"observed": values["observed"], "reference": values[reference_key],
            "difference": values["difference"]}


def _parity_disagreements(record: dict) -> list[dict]:
    """Every parity comparison in one corpus's preflight record that fell outside the tolerance.

    Read off the structured comparisons rather than off ``preflight``'s ``findings`` strings. The
    gate has to know which record and which quantity a disagreement sits in, and a sentence cannot be
    keyed on without parsing it back into the fields it was assembled from. The two lists agree by
    construction: every comparison the sentences were built from is walked here.
    """
    found: list[dict] = []
    published = record["published_prefix_numbers"]
    for cell, row in sorted(published["per_cell"].items()):
        prefix_key, arm = cell.split("/", 1)
        for quantity in ("seed_averaged_oof_roc_auc", "mean_fold_roc_auc"):
            if not row[quantity]["within_tolerance"]:
                found.append({"check": "published_prefix_numbers", "source": published["source"],
                              "prefix": prefix_key, "arm": arm, "quantity": quantity,
                              "claim_id": row["claim_id"],
                              **_disagreement(row[quantity], "published")})
    contrast = record["published_linear_contrast"]
    for prefix_key, row in sorted(contrast["per_prefix"].items()):
        for quantity, values in sorted(row.items()):
            if not values["within_tolerance"]:
                found.append({"check": "published_linear_contrast", "source": published["source"],
                              "prefix": prefix_key, "arm": contrast["contrast"],
                              "quantity": quantity, "claim_id": None,
                              **_disagreement(values, "published")})
    endpoint = record["hundred_percent_matches_post"]
    for arm, row in sorted(endpoint["per_arm"].items()):
        for quantity, values in sorted(row.items()):
            if not values["within_tolerance"]:
                found.append({"check": "hundred_percent_matches_post", "source": endpoint["source"],
                              "prefix": "100", "arm": arm, "quantity": quantity, "claim_id": None,
                              **_disagreement(values, "post_audit")})
    return found


def _parity_gate(result: dict) -> dict:
    """What A10.4 requires of this route, and whether today's preflight met it.

    The declaration's dated entry of 2026-09-10 narrows A10.4 to the PUBLISHED prefix numbers, and
    it names exactly what does not reproduce: ``variance_axes.board_point_estimate.value`` inside
    ``statistical_tests_results.json``, at the SWE-Gym 25 percent prefix, on two cells, both by
    2.845e-05. That field is a diagnostic inside a record and is not computed the way the board
    computes it; the board's own path is checked separately by ``_check_route_is_the_board``, with
    exact equality and no tolerance, and it reproduces cell for cell.

    So the split below is by artifact and quantity, not by size. A disagreement in that one field of
    that one record is recorded as a finding about the artifact and does not stop the batch. Every
    other disagreement does stop it, including the same quantity read against the POST audit, where
    the entry establishes that the endpoint reproduces at 0.000e+00 on both estimands. Nothing here
    widens ``PARITY_ATOL``, and the preflight's own ``cleared`` flag is carried through untouched:
    this is a second, narrower reading placed beside it, not a replacement for it.
    """
    blocking: list[dict] = []
    defects: list[dict] = []
    for corpus, record in sorted(result["corpora"].items()):
        for row in _parity_disagreements(record):
            row = {"corpus": corpus, **row}
            covered = (row["source"] == RECORD_DIAGNOSTIC["source"]
                       and row["quantity"] == RECORD_DIAGNOSTIC["quantity"])
            (defects if covered else blocking).append(row)
    return {
        "admits_the_batch": not blocking,
        "reading": "A10.4 requires the published prefix numbers to reproduce: the board cells, the "
                   "declared A5 estimand at every supervised LIVE cell, the manuscript's +0.101 "
                   "contrast, and the endpoint against the POST audit. Recorded by the "
                   "declaration's dated entry of 2026-09-10, before the batch ran.",
        "established_by": {
            "board_route_exactly": "_check_route_is_the_board reproduces GRADE's _cv cell for cell "
                                   "at every prefix on flat and flatdep, by exact array equality",
            "declared_estimand": "_check_published_prefix_numbers on seed_averaged_oof_roc_auc, "
                                 "which is the record's estimate.a",
            "published_contrast": "_check_published_linear_contrast on seed_averaged_oof_roc_auc",
            "endpoint": "_check_hundred_percent_matches_post on both estimands",
        },
        "exempted_field": dict(RECORD_DIAGNOSTIC),
        "exemption_is_not": "a widened tolerance. PARITY_ATOL is 1e-12 and was written into "
                            "catchbench.live_size_audit before any fit; it is unchanged.",
        "preflight_cleared_on_its_own_terms": result["cleared"],
        "blocking_disagreements": blocking,
        "record_defects_carried_forward": defects,
    }


# --- A5's eight effects, A6's intervals, A7's temporal contrast -----------------------------------


def _matched_vectors(cell: dict) -> tuple[np.ndarray, np.ndarray]:
    """The two seed-averaged out-of-fold vectors A5's contrast is between, at one prefix.

    Averaged over the five split seeds and then scored once, which is the declared estimand and NOT
    the mean of the five per-seed pooled AUCs the same arrays also support. The two differ, and the
    record carries both under names that say which is which.
    """
    from catchbench.live_size_audit import MATCHED_CONTRAST

    high, low = MATCHED_CONTRAST
    return (np.asarray(cell["arms"][high]["oof_proba"], dtype=float).mean(axis=0),
            np.asarray(cell["arms"][low]["oof_proba"], dtype=float).mean(axis=0))


def _effect(corpus: str, prefix_key: str, cell: dict, labels: np.ndarray,
            clustering: dict | None, tolerance: float) -> dict:
    """One of the eight ``D(c, p)``: the declared point, A6's interval, and what binds them together.

    The point is recomputed here from the saved vectors and then required to equal the one
    ``live_size_audit._effects`` stored, exactly. Two arms recorded separately can each be right
    while the difference is taken between the wrong pair or on the wrong scale, and this is the
    comparison that rules that out: the interval and the scorer have to be reading the same two
    vectors for a row to be emitted at all.

    A6 splits by corpus and the split is not cosmetic. SWE-Gym's 376 rows carry 376 distinct tasks,
    so paired DeLong conditional on the saved fits is the interval its core contrasts already use.
    tau-bench's 660 rows are 165 task instances each attempted by four agent models, so a run-level
    interval would count four attempts at one task as four observations; the printed interval there
    is the shipped domain-stratified task-cluster bootstrap, and DeLong travels beside it as a
    reproduction diagnostic under the independence assumption, which is how the committed LIVE rows
    already carry the pair.

    The mean of fold AUCs rides along under ``reproduction_quantity`` with no interval attached. A5
    forbids giving it the one computed for the averaged-probability quantity, and the way to obey
    that in an artifact is to store it somewhere an interval visibly is not.
    """
    from catchbench.live_size_audit import (
        BOOTSTRAP_BASE_SEED, BOOTSTRAP_DRAWS, MATCHED_CONTRAST, PRIMARY_CELL, RNG_LABELS,
    )
    from sklearn.metrics import roc_auc_score

    high_name, low_name = MATCHED_CONTRAST
    high, low = _matched_vectors(cell)
    prefix = float(cell["prefix"])
    point = float(roc_auc_score(labels, high) - roc_auc_score(labels, low))
    stored = cell["effects"][f"{high_name} - {low_name}"]
    if point != stored["seed_averaged_oof_difference"]:
        raise AssertionError(
            f"{corpus} {prefix_key}%: the effect recomputed from the saved out-of-fold vectors is "
            f"{point!r} and the scorer stored {stored['seed_averaged_oof_difference']!r}; the "
            "interval below would not be the interval of the point above it"
        )

    label = RNG_LABELS[(corpus, prefix)]
    delong = st.paired_delong(labels, high, low)
    row: dict[str, Any] = {
        "corpus": corpus,
        "prefix": prefix,
        "prefix_percent": int(prefix_key),
        "cell": f"{corpus}.{prefix_key}",
        "high": high_name,
        "low": low_name,
        "point": point,
        "role": ("primary" if (corpus, prefix) == PRIMARY_CELL
                 else "reproduction control" if prefix == ENDPOINT else "declared cell"),
        "independent_new_evidence": prefix != ENDPOINT,
        "labelled_as": ("the cell A5 nominated before any of the eight was computed"
                        if (corpus, prefix) == PRIMARY_CELL
                        else "a reproduction control, not independent new evidence (A9)"
                        if prefix == ENDPOINT else "one of the eight declared cells"),
        "rng_label": label,
        "rng_base_seed": BOOTSTRAP_BASE_SEED,
        "estimand": "ROC-AUC of the matched structural arm's five-seed-averaged out-of-fold failure "
                    "probabilities minus that of the fixed flexible size arm (A5)",
        "arm_scores": {name: {"seed_averaged_oof_roc_auc": cell["arms"][name][
                                  "seed_averaged_oof_roc_auc"],
                              "mean_fold_roc_auc": cell["arms"][name]["mean_fold_roc_auc"]}
                       for name in sorted(cell["arms"])},
        "reproduction_quantity": {
            "mean_fold_difference": stored["mean_fold_difference"],
            "interval": None,
            "why_no_interval": "A5: the mean of fold AUCs is a separate reproduction quantity and "
                               "never receives the interval computed for the averaged-probability "
                               "quantity",
        },
        "delong": delong,
    }
    if corpus == "tau":
        if clustering is None:
            raise AssertionError("the tau-bench interval needs the LIVE task clustering")
        clustered = st._task_clustered_auc_bootstrap(
            labels, clustering, high, low, None, BOOTSTRAP_DRAWS,
            st._rng_for(label, BOOTSTRAP_BASE_SEED),
        )
        row["interval_95"] = list(clustered["interval_95"])
        row["interval_method"] = st.TAU_CLUSTER_METHOD
        row["interval_axis"] = st.TAU_CLUSTER_AXIS
        row["clustered_interval"] = clustered
        row["reproduction_diagnostic"] = st._clustered_reproduction_diagnostic(
            delong["interval_95"], "paired DeLong",
            {"z": delong["z"], "se": delong["se"], "n_positive": delong["n_positive"],
             "n_negative": delong["n_negative"]}, delong["p"],
        )
    else:
        row["interval_95"] = list(delong["interval_95"])
        row["interval_method"] = "paired DeLong, conditional on the saved fits"
        row["interval_axis"] = "run-level sampling"
        row["crosscheck_stratified_bootstrap"] = {
            **st._stratified_auc_bootstrap(labels, high, low, BOOTSTRAP_DRAWS,
                                           st._rng_for(label, BOOTSTRAP_BASE_SEED)),
            "role": "crosscheck only; A6 makes paired DeLong the reported SWE-Gym interval, and "
                    "this is the stratified paired bootstrap the core contrasts already carry "
                    "beside DeLong. It is what consumes the frozen swegym RNG label.",
        }
    delta = abs(delong["difference"] - point)
    row["delong_point_agrees_with_the_estimand"] = {
        "delong_difference": delong["difference"],
        "estimand": point,
        "abs_difference": delta,
        "tolerance": tolerance,
        "within_tolerance": bool(delta <= tolerance),
        "why_they_can_differ_at_all": "DeLong's AUC is the tie-corrected placement mean and the "
                                      "estimand is scored by sklearn's trapezoid rule; the two are "
                                      "the same quantity computed two ways and agree to float noise",
    }
    row["conditional_on_the_saved_fits"] = (
        "A6: folds and seeds are not independent new runs. This interval is conditional on the "
        "saved fits and does not include resample-and-refit training uncertainty."
    )
    return row


def _temporal_contrast(labels: np.ndarray, early: tuple[np.ndarray, np.ndarray],
                       late: tuple[np.ndarray, np.ndarray], early_point: float,
                       late_point: float, tolerance: float) -> dict:
    """A7's ``T = D(swegym, 0.25) - D(swegym, 1.00)``, from one class-stratified run bootstrap.

    All four score vectors are indexed by the SAME draw, which is what makes this an interval about
    the difference of two contrasts rather than about two contrasts measured on unrelated resamples.
    A run that enters a draw enters it at both prefixes and in both arms, so the correlation between
    the early and endpoint contrasts is carried. That correlation is large, because the two
    contrasts are the same 376 runs scored on the same 25 splits, and ignoring it would inflate the
    interval by roughly the amount the pairing removes.

    Class-stratified: the positive and negative counts are the corpus's own in every draw, so no
    draw can be single-class and none is discarded. The draw itself is the one
    ``_stratified_auc_bootstrap`` makes, positives then negatives off the same generator, so the two
    consume their labels the same way. Nothing is refitted; A7 says so, and the four vectors are the
    ones the batch already saved.
    """
    from catchbench.live_size_audit import (
        BOOTSTRAP_BASE_SEED, BOOTSTRAP_DRAWS, TEMPORAL_RNG_LABEL,
    )

    y = np.asarray(labels, dtype=int)
    columns = np.column_stack([early[0], early[1], late[0], late[1]]).astype(float)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    rng = st._rng_for(TEMPORAL_RNG_LABEL, BOOTSTRAP_BASE_SEED)
    values = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        indices = np.concatenate([
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ])
        boot_y = y[indices]
        block = columns[indices]
        values[draw] = ((st._auc_fast(boot_y, block[:, 0]) - st._auc_fast(boot_y, block[:, 1]))
                        - (st._auc_fast(boot_y, block[:, 2]) - st._auc_fast(boot_y, block[:, 3])))
    interval = np.quantile(values, (0.025, 0.975))
    point = float(early_point - late_point)
    resampled_point = float(
        (st._auc_fast(y, columns[:, 0]) - st._auc_fast(y, columns[:, 1]))
        - (st._auc_fast(y, columns[:, 2]) - st._auc_fast(y, columns[:, 3]))
    )
    lower = (int(np.sum(values <= 0)) + 1) / (BOOTSTRAP_DRAWS + 1)
    upper = (int(np.sum(values >= 0)) + 1) / (BOOTSTRAP_DRAWS + 1)
    return {
        "quantity": "T = D(swegym, 0.25) - D(swegym, 1.00)",
        "point": point,
        "components": {"D(swegym, 0.25)": float(early_point), "D(swegym, 1.00)": float(late_point)},
        "interval_95": [float(interval[0]), float(interval[1])],
        "interval_method": "class-stratified paired run percentile bootstrap carrying all four "
                           "score vectors in every draw",
        "interval_axis": "run-level sampling, stratified by class",
        "replicates": BOOTSTRAP_DRAWS,
        "discarded_single_class_draws": 0,
        "rng_label": TEMPORAL_RNG_LABEL,
        "rng_base_seed": BOOTSTRAP_BASE_SEED,
        "two_sided_tail_p": min(1.0, 2 * min(lower, upper)),
        "bootstrap_mean": float(values.mean()),
        "bootstrap_sd": float(values.std(ddof=1)),
        "point_recomputed_by_the_bootstrap_scorer": {
            "value": resampled_point,
            "abs_difference": abs(resampled_point - point),
            "tolerance": tolerance,
            "within_tolerance": bool(abs(resampled_point - point) <= tolerance),
        },
        "additional_fits": 0,
        "role": "secondary. A7: it never replaces a failed primary result, and if D(swegym, 0.25) "
                "is unresolved a positive T does not rescue an early-signal claim.",
    }


# --- the declared batch ---------------------------------------------------------------------------


def _score_corpus(corpus: str, corpus_preflight: bool) -> dict:
    """Fit the four declared arms at all four prefixes on one corpus, checking every spline fit.

    ``_FoldLocality`` is the POST audit's own check, one instance per prefix view, so all 200
    spline-bearing fits of a corpus are inspected inside the fold loop that produced them and by the
    same array comparisons the endpoint audit uses. Its ``record()`` refuses a prefix where every
    fit's knots equal the whole-matrix knots, because such a prefix has no negative control and the
    training-fold comparison there cannot tell a leak from a clean fit.
    """
    from catchbench import detection
    from catchbench import live_size_audit as audit
    from catchbench.corpora import verify_pinned_fetches
    from catchbench.live import LiveStreaming

    from emit_detection_audit import _FoldLocality

    task = LiveStreaming(corpus)
    task.setup()
    if corpus_preflight:
        verify_pinned_fetches(names={CORPUS_NAMES[corpus]})

    localities: dict[str, Any] = {}

    def inspect(arm, view, seed, fold, train, fitted):
        key = audit._prefix_key(view.prefix)
        locality = localities.get(key)
        if locality is None:
            locality = localities[key] = _FoldLocality(view, detection)
        locality(arm, seed, fold, train, fitted)

    clustering = st._tau_live_clustering(task) if corpus == "tau" else None
    audited = audit.live_size_audit(task, inspect=inspect)
    audited["corpus_line"] = task.corpus_line()
    audited["run_identities"] = _run_identities(task, detection, clustering)
    locality_record = {key: localities[key].record() for key in sorted(localities, key=int)}
    return {"task": task, "audited": audited, "clustering": clustering,
            "training_only_transforms": locality_record}


def _run_identities(task, detection, clustering: dict | None) -> dict:
    """Per-row identifiers for one LIVE corpus, where the loaders leave any behind.

    tau-bench's come from ``statistical_tests._tau_live_clustering``, which replays the corpus under
    LIVE's own ``>= 4``-step filter and verifies the replay against LIVE's labels and 100 percent
    step counts. What it returns is a clustering rather than a list of keys, so the per-row identity
    recorded here is the cluster index and the domain, which is what an interval or a re-analysis
    needs; recovering the task-id strings would mean a second replay, and the module this batch is
    driven from exists partly so there is only one.

    SWE-Gym's branch is ``detection._swegym_run_identities`` unchanged. It reads the loader source
    and nothing off the task, so it is as true of the LIVE corpus as of the POST one, and it hashes
    that source so the record cannot go stale silently if the loader later starts selecting an
    identifier column.
    """
    if clustering is None:
        return detection._swegym_run_identities(task)
    blocks = clustering["blocks"]
    cluster_of_row = np.empty(blocks.size, dtype=int)
    cluster_of_row[blocks.reshape(-1)] = np.repeat(np.arange(blocks.shape[0]), blocks.shape[1])
    stratum_of_cluster = np.empty(blocks.shape[0], dtype=object)
    for name, rows in clustering["strata"].items():
        stratum_of_cluster[rows] = name
    return {
        "available": True,
        "source": "tools/statistical_tests.py::_tau_live_clustering",
        "task_cluster": [int(value) for value in cluster_of_row],
        "domain": [str(stratum_of_cluster[value]) for value in cluster_of_row],
        "n_distinct_task_clusters": int(blocks.shape[0]),
        "runs_per_task_cluster": int(blocks.shape[1]),
        "clusters_per_stratum": dict(clustering["record"]["clusters_per_stratum"]),
        "identifier_is": "the task cluster index and its domain, not the tau-bench task-id string. "
                         "_tau_live_clustering returns a clustering rather than the keys it checked "
                         "for uniqueness, and a second replay to recover the strings would be a "
                         "second thing to keep aligned with the prefix feature matrices.",
    }


def run_batch(corpora: list[str], corpus_preflight: bool, tolerance: float) -> dict:
    """A5's eight effects, A6's intervals, A7's contrast and A8's branch, in one pass.

    A corpus whose fits raise is caught here rather than allowed to end the run. A11 preserves valid
    partial results and their failure record, and the two corpora share no fitted object, so a
    SWE-Gym failure has no bearing on the tau-bench cells and there is no reason for it to discard
    them. The traceback is kept, because "a recorded failure" that does not say where it happened
    cannot be told from a disappointing effect, and the two are what A11 most needs kept apart.
    """
    from catchbench import live_size_audit as audit

    corpora_records: dict[str, dict] = {}
    verification: dict[str, dict] = {}
    effects: list[dict] = []
    failures: list[dict] = []
    warned: list[dict] = []
    temporal: dict | None = None
    outcome: dict | None = None

    for corpus in corpora:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                scored = _score_corpus(corpus, corpus_preflight)
                task, audited = scored["task"], scored["audited"]
                labels = np.asarray(task.y)
                for key in sorted(audited["by_prefix"], key=int):
                    effects.append(_effect(corpus, key, audited["by_prefix"][key], labels,
                                           scored["clustering"], tolerance))
            for warning in caught:
                warned.append({"corpus": corpus, "category": warning.category.__name__,
                               "message": str(warning.message),
                               "where": f"{Path(warning.filename).name}:{warning.lineno}"})
            corpora_records[corpus] = audited
            verification[corpus] = {"training_only_transforms": scored["training_only_transforms"]}
            for key, cell in sorted(audited["by_prefix"].items(), key=lambda item: int(item[0])):
                for arm_id, arm in sorted(cell["arms"].items()):
                    if arm["nonfinite_fold_scores"] or arm["nonfinite_oof_values"]:
                        failures.append({
                            "kind": "nonfinite_output", "corpus": corpus, "prefix": key,
                            "arm": arm_id, "fold_scores": arm["nonfinite_fold_scores"],
                            "oof_values": arm["nonfinite_oof_values"],
                        })
        except Exception as error:  # noqa: BLE001 - A11 wants this recorded, not raised
            failures.append({
                "kind": "corpus_batch_raised", "corpus": corpus,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "consequence": f"the four {corpus} cells are absent from this record; A11 preserves "
                               "the partial results and this failure rather than rerunning",
            })

    primary_corpus, primary_prefix = audit.PRIMARY_CELL
    primary = next((row for row in effects
                    if row["corpus"] == primary_corpus and row["prefix"] == primary_prefix), None)
    if primary is not None:
        outcome = audit.classify_outcome(primary["point"], *primary["interval_95"])
        outcome["primary_cell"] = primary["cell"]
        outcome["interval_method"] = primary["interval_method"]
    else:
        failures.append({
            "kind": "primary_cell_missing", "corpus": primary_corpus,
            "prefix": audit._prefix_key(primary_prefix),
            "error": "the primary cell was not scored, so A8 has no branch to apply",
        })

    (early_corpus, early_prefix), (late_corpus, late_prefix) = audit.TEMPORAL_CONTRAST
    try:
        record = corpora_records[early_corpus]
        early_key, late_key = audit._prefix_key(early_prefix), audit._prefix_key(late_prefix)
        temporal = _temporal_contrast(
            np.asarray(record["labels"]),
            _matched_vectors(record["by_prefix"][early_key]),
            _matched_vectors(record["by_prefix"][late_key]),
            next(row["point"] for row in effects
                 if row["corpus"] == early_corpus and row["prefix"] == early_prefix),
            next(row["point"] for row in effects
                 if row["corpus"] == late_corpus and row["prefix"] == late_prefix),
            tolerance,
        )
    except Exception as error:  # noqa: BLE001 - same reason as the corpus loop above
        failures.append({
            "kind": "temporal_contrast_failed",
            "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
        })

    return {
        "corpora": corpora_records,
        "verification": {"per_corpus": verification},
        "effects": effects,
        "primary": primary,
        "outcome": outcome,
        "temporal_contrast": temporal,
        "failures": failures,
        "warnings": warned,
    }


# --- provenance ------------------------------------------------------------------------------------
# Defined here rather than shared with tools/live_tau_bar_design.py. The declaration opens by saying
# that Part A and Part B are independent and that a failure of one does not license a change to the
# other, and importing that tool's module to reach three five-line helpers would make one of them a
# dependency of the other for no gain.


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(path: Path) -> str:
    """The checked-out commit of a repository, or a reason string; never an exception."""
    try:
        completed = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                                   capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:  # pragma: no cover - environment
        return f"unavailable ({error})"
    return completed.stdout.strip()


def _git_worktree(path: Path) -> dict:
    """Whether the worktree is clean, and what is in it if not.

    The paths matter as much as the flag. This run writes its own record into the repository it
    reads, so ``clean`` is false by the time the record is emitted even when every input was read at
    the declared commit. Listing what is uncommitted lets a reader see that rather than take the
    flag at face value.
    """
    try:
        completed = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                                   capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:  # pragma: no cover - environment
        return {"clean": f"unavailable ({error})", "uncommitted": None}
    entries = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {"clean": not entries, "uncommitted": entries}


def _inputs(declaration: Path | None, preflight_output: Path, corpus_lines: dict) -> dict:
    """Every file this batch read, by path and sha256, and the three repositories it read them at.

    The declaration's frozen-inputs table names CODE, PAPER and GRADE by commit and three sources by
    sha256. All three commits are recorded, PAPER through ``CATCHBENCH_PAPER_DIR`` because this tool
    has no other way to find it and writes nothing to it either way.
    """
    import agent_failure_detection

    grade_root = Path(agent_failure_detection.__file__).resolve().parents[1]
    paper_dir = os.environ.get("CATCHBENCH_PAPER_DIR")
    repositories = {
        "catchbench": {"path": str(ROOT), "commit": _git_commit(ROOT), **_git_worktree(ROOT)},
        "grade": {"path": str(grade_root), "commit": _git_commit(grade_root),
                  **_git_worktree(grade_root)},
    }
    if paper_dir:
        repositories["paper"] = {"path": paper_dir, "commit": _git_commit(Path(paper_dir)),
                                 **_git_worktree(Path(paper_dir)),
                                 "written_to": False}
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
                ("src/catchbench/live.py", ROOT / "src" / "catchbench" / "live.py"),
                ("src/catchbench/detection.py", ROOT / "src" / "catchbench" / "detection.py"),
                ("src/catchbench/live_size_audit.py",
                 ROOT / "src" / "catchbench" / "live_size_audit.py"),
                ("tools/statistical_tests.py", ROOT / "tools" / "statistical_tests.py"),
                ("tools/emit_live_size_audit.py", Path(__file__).resolve()),
                ("grade/experiment/agent_failure_detection.py",
                 Path(agent_failure_detection.__file__).resolve()),
            )
        },
        "records_read_by_the_preflight": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in (("statistical_tests_results.json", STATISTICAL_RECORD),
                               ("detection_audit_results.json", POST_AUDIT_RECORD))
        },
        "preflight_record": {"path": str(preflight_output), "sha256": _sha256(preflight_output)},
        "repositories": repositories,
        "package_versions": packages,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "corpora": corpus_lines,
        "records_consumed_by_the_batch": [],
        "note": "the batch reads feature matrices and fits arms. It consumes no field of "
                "statistical_tests_results.json; that record is read by the preflight alone.",
    }
    if declaration is not None:
        inputs["declaration"] = {"path": str(declaration), "sha256": _sha256(declaration),
                                 "part": "Part A, the matched size control at LIVE prefixes"}
    return inputs


def build_result(batch: dict, gate: dict, preflight_result: dict, declaration: Path | None,
                 preflight_output: Path, started: float) -> dict:
    """The A9 record: both corpora, all four prefixes, all four arm scores, the eight effects, T."""
    from catchbench import detection
    from catchbench import live_size_audit as audit

    corpus_lines = {name: record.get("corpus_line")
                    for name, record in sorted(batch["corpora"].items())}
    return {
        "schema_version": "1.0.0",
        "generated_by": "tools/emit_live_size_audit.py --run",
        "question": "At each LIVE prefix, how much of the dependency increment survives once the "
                    "size reference is allowed to bend: the ROC-AUC of the matched structural arm "
                    "minus that of the fixed flexible size arm.",
        "declaration": {
            "file": "research/catchbench-live-size-control-declaration-2026-09-10.md",
            "part": "A",
            "frozen_before_the_run": {
                "prefixes": [float(value) for value in audit.AUDIT_PREFIXES],
                "declared_population": dict(audit.DECLARED_POPULATION),
                "arms": dict(audit.ARM_ROLES),
                "matched_contrast": list(audit.MATCHED_CONTRAST),
                "linear_reproduction_contrast": list(audit.LINEAR_CONTRAST),
                "primary_cell": {"corpus": audit.PRIMARY_CELL[0], "prefix": audit.PRIMARY_CELL[1]},
                "rng_labels": {f"{corpus}.{int(round(prefix * 100))}": label
                               for (corpus, prefix), label in sorted(audit.RNG_LABELS.items())},
                "temporal_rng_label": audit.TEMPORAL_RNG_LABEL,
                "rng_base_seed": audit.BOOTSTRAP_BASE_SEED,
                "bootstrap_draws": audit.BOOTSTRAP_DRAWS,
                "substantial_threshold": audit.SUBSTANTIAL_EFFECT,
                "parity_tolerance": audit.PARITY_ATOL,
                "seeds": [int(value) for value in audit.SEEDS],
                "folds": 5,
                "spline": dict(detection._SPLINE),
            },
            "fixed_by_this_implementation": {
                "a8_branch_order": "support first, size second. An interval containing zero is "
                                   "unresolved whatever the point is; only a supported sign is then "
                                   "split by the declared +0.03. Written into "
                                   "catchbench.live_size_audit.classify_outcome before the batch.",
                "a8_fourth_bullet": "an upper endpoint below +0.03 travels as a field rather than "
                                    "as a fifth branch, because A8 gives it no branch name",
                "swegym_rng_labels": "A6 makes paired DeLong the reported SWE-Gym interval and "
                                     "freezes eight labels, four of which no DeLong consumes. They "
                                     "seed the stratified paired bootstrap the core contrasts "
                                     "already carry beside DeLong, emitted as a crosscheck and "
                                     "never as the reported interval.",
                "delong_point_check": "every row compares DeLong's own AUC difference against the "
                                      "declared estimand at PARITY_ATOL, so a row cannot pair an "
                                      "interval with a point computed a different way",
                "fold_locality": "emit_detection_audit._FoldLocality, one instance per prefix view, "
                                 "so all 200 spline-bearing fits per corpus are inspected inside "
                                 "the fold loop that produced them",
                "partial_results": "a corpus whose fits raise is recorded with its traceback and "
                                   "the other corpus still finishes (A11)",
            },
        },
        "settings": {
            "confidence_level": 0.95,
            "z_975": st.Z975,
            "declared_fits": 4 * len(CORPORA) * len(audit.AUDIT_PREFIXES) * 5 * 5,
            "corpora_scored": sorted(batch["corpora"]),
            "bootstrap": {
                "tau_bench": {"method": st.TAU_CLUSTER_METHOD, "axis": st.TAU_CLUSTER_AXIS,
                              "unit": st.TAU_CLUSTER_UNIT, "stratum": st.TAU_CLUSTER_STRATUM,
                              "draws": audit.BOOTSTRAP_DRAWS,
                              "monte_carlo_noise_on_one_endpoint": st.TAU_CLUSTER_MC_NOISE},
                "swegym": {"method": "paired DeLong, conditional on the saved fits",
                           "crosscheck": "stratified paired percentile bootstrap"},
                "temporal": {"method": "class-stratified paired run percentile bootstrap carrying "
                                       "all four score vectors in every draw",
                             "draws": audit.BOOTSTRAP_DRAWS},
            },
        },
        "inputs": _inputs(declaration, preflight_output, corpus_lines),
        "preflight": {
            "record": str(preflight_output),
            "cleared_on_its_own_terms": preflight_result["cleared"],
            "findings": preflight_result["findings"],
            "gate": gate,
        },
        "corpora": batch["corpora"],
        "verification": batch["verification"],
        "effects": batch["effects"],
        "primary": batch["primary"],
        "outcome": batch["outcome"],
        "temporal_contrast": batch["temporal_contrast"],
        "failures": batch["failures"],
        "warnings": batch["warnings"],
        "interpretation": {
            "exploratory": "A2 and the chronology section: this extension was proposed after the "
                           "linear LIVE prefix scores and the POST specification-sensitivity "
                           "results were already in the manuscript. It stays labelled exploratory "
                           "whichever way it comes out.",
            "endpoint_cells_are_controls": "A9: the 100 percent effects reproduce a cell whose "
                                           "value is already committed. They are reproduction "
                                           "controls, not independent new evidence.",
            "no_deployable_alarm": "A3: the prefix fractions index a retrospective sweep over final "
                                   "trace length. A positive result establishes neither a "
                                   "deployable clock-time alarm nor knowledge of the eventual "
                                   "horizon during execution.",
            "tau_is_still_row_level_cv": "A4: row-level cross-validation on tau-bench is the "
                                         "existing evaluation convention. The clustered interval "
                                         "describes sampling uncertainty and does not convert the "
                                         "evaluation into an unseen-task evaluation.",
            "no_simultaneous_coverage": "A9: no new Holm family and no search over prefixes. These "
                                        "are marginal readings with one nominated primary quantity, "
                                        "and the 138-record registry is unchanged.",
            "one_batch": "A11: one declared fit batch and one reporting pass. A disappointing "
                         "effect is not an implementation defect and is not grounds for a rerun.",
        },
        "elapsed_seconds": time.time() - started,
    }


def _report_batch(result: dict) -> None:
    print(f"\nLIVE prefix size control :: declared batch ({result['declaration']['file']}, Part A)")
    high, low = result["declaration"]["frozen_before_the_run"]["matched_contrast"]
    print(f"  effect D(c, p) = {high}  minus  {low}, on seed-averaged out-of-fold probabilities")
    header = f"  {'cell':<12}{'D':>11}{'low':>11}{'high':>11}  {'interval':<38}role"
    print(f"\n{header}\n  {'-' * (len(header) - 2)}")
    for row in result["effects"]:
        lo, hi = row["interval_95"]
        print(f"  {row['cell']:<12}{row['point']:>+11.6f}{lo:>+11.6f}{hi:>+11.6f}  "
              f"{row['interval_method'][:36]:<38}{row['role']}")
    temporal = result["temporal_contrast"]
    if temporal is not None:
        lo, hi = temporal["interval_95"]
        print(f"\n  {temporal['quantity']} = {temporal['point']:+.6f} "
              f"[{lo:+.6f}, {hi:+.6f}]  (secondary)")
    outcome = result["outcome"]
    if outcome is not None:
        lo, hi = outcome["interval_95"]
        print(f"\n  A8 branch: {outcome['branch'].upper()}  "
              f"primary {outcome['primary_cell']} = {outcome['point']:+.6f} "
              f"[{lo:+.6f}, {hi:+.6f}], threshold {outcome['substantial_threshold']:+.2f}")
    if result["failures"]:
        print(f"\n  {len(result['failures'])} recorded failure(s):")
        for failure in result["failures"]:
            print(f"    [{failure['kind']}] {failure.get('corpus', '-')}: "
                  f"{failure.get('error', '')}")
    else:
        print("\n  no recorded failures")
    if result["warnings"]:
        print(f"  {len(result['warnings'])} warning(s) captured")


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
    result = preflight(list(args.corpora), corpus_preflight=not args.no_corpus_preflight)
    _write(args.output, result)
    _report(result)
    print(f"\nwrote {args.output}")

    gate = _parity_gate(result)
    if gate["record_defects_carried_forward"]:
        print(f"\n{len(gate['record_defects_carried_forward'])} disagreement(s) in "
              f"{RECORD_DIAGNOSTIC['source']}::{RECORD_DIAGNOSTIC['field']}, carried forward as "
              "findings about that\nrecord under the declaration's dated A10.4 entry rather than "
              "as a failure of this route:")
        for row in gate["record_defects_carried_forward"]:
            print(f"  {row['corpus']} {row['prefix']}% {row['arm']}: {row['observed']!r} against "
                  f"the recorded {row['reference']!r} (difference {row['difference']:+.3e})")
    if not gate["admits_the_batch"]:
        print(f"\nA10.4 is NOT satisfied: {len(gate['blocking_disagreements'])} published "
              "quantity/quantities outside the declared tolerance.")
        for row in gate["blocking_disagreements"]:
            print(f"  {row['corpus']} {row['prefix']}% {row['arm']} {row['quantity']}: "
                  f"{row['observed']!r} against {row['reference']!r}")
        return 1
    if not args.run:
        return 0

    batch = run_batch(list(args.corpora), corpus_preflight=not args.no_corpus_preflight,
                      tolerance=_tolerance_value())
    payload = build_result(batch, gate, result, args.declaration, args.output, started)
    _write(args.result_output, payload)
    _report_batch(payload)
    print(f"\nwrote {args.result_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
