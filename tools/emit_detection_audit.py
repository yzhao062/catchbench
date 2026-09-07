"""Score the POST detection controlled audit and write it as data.

The board answers one question per row. This audit answers the question the board rows raise
between them: how much of the dependency increment on ``post_detection`` is a property of the
dependency features, and how much is a property of the model that the size reference was given.
It scores the board's linear size references, a flexible ``size (spline)`` arm, a spline size model
with the dependency block entering linearly, a matched gradient-boosted-tree pair, and the two
nested-selection variants that estimate what a knot-selecting procedure would have scored.

**No arm here is a board entrant.** The three linear references are, and they are read out of the
registry rather than rebuilt. Everything else is an audit arm, written here rather than promoted,
so the board keeps one row per question and its entrant inventory does not move. Registering the
spline would add a row to both detection blocks and take the inventory from 72 to 73.

The output is JSON and not a table, because two different intervals can be computed from these
runs and the choice belongs to whoever is making the claim. Every arm therefore carries its
per-(seed, fold) ROC-AUC, from which the split-seed interval follows, and its per-seed out-of-fold
probabilities, from which the paper's paired run-level test follows. Neither is a substitute for
the other: a split-seed interval describes redrawing the folds on these runs, and says nothing
about other runs. Where the loader leaves a row identifier behind, the record carries that too, so
a third interval that clusters repeated attempts at one task is computable without refitting. That
holds on tau-bench and not on SWE-Gym, and ``run_identities`` in each corpus record says which case
applies and how it was established.

Six of the eight contrasts are matched pairs, differing in one respect. Two are not, and they are
labelled rather than dropped, because they are the pair the manuscript's own sentence is about.
The artifact carries ``matched`` and ``confound`` on every contrast so the label travels with the
number.

Run from the repository root with the benchmark Python environment::

    GRADE_DIR=/path/to/grade python tools/emit_detection_audit.py \
        --output tools/detection_audit_results.json

Every run verifies, before it writes anything, that the audit path is the board path: that the
swapped-estimator cross-validation reproduces GRADE's ``_cv`` exactly on the linear arms, that
``size (spline)`` scores identically through ``RunPipeline`` and through the audit path, that every one
of the 25 fits of every spline-bearing arm has knots and scaling belonging to its own training
fold, and that the gradient-boosted-tree configuration still matches GRADE's ``_m_gbt`` wherever
that script is present.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CORPORA = ("swegym", "tau")
CORPUS_NAMES = {"swegym": "SWE-Gym", "tau": "tau-bench"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpora", nargs="+", choices=CORPORA, default=list(CORPORA))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "tools" / "detection_audit_results.json")
    parser.add_argument("--no-preflight", action="store_true",
                        help="skip the Hub revision preflight; for an offline rerun over a cache "
                             "that a previous verified run already populated")
    return parser.parse_args()


# --- verification: the audit path is the board path -------------------------------------------


def _check_folds_are_the_boards(task, detection) -> dict:
    """The swapped-estimator loop must reproduce GRADE's ``_cv`` exactly on the linear arms.

    ``_cv_estimator`` differs from ``_cv`` in one respect, the estimator it is handed. Feeding it
    the estimator ``_cv`` builds must therefore return ``_cv``'s array, cell for cell. If it does
    not, the folds moved, and every contrast below would be comparing arms scored on different
    splits.
    """
    checked = {}
    for layer in ("flat", "flatdep", "full"):
        matrix = task.layers[layer]
        theirs = detection._cv(matrix, task.y)
        ours = detection._cv_estimator(detection._linear_estimator, matrix, task.y)
        if not np.array_equal(theirs, ours):
            raise AssertionError(
                f"{task.dataset}/{layer}: the swapped-estimator loop does not reproduce GRADE's "
                f"_cv; max |difference| {np.abs(theirs - ours).max():.3e}"
            )
        checked[layer] = float(theirs.mean())
    return checked


def _check_detail_matches_cross_val_score(task, detection, arms, audited) -> dict:
    """The recorded fold scores must be the fold scores ``cross_val_score`` returns.

    The audit keeps the fitted estimators so it can record out-of-fold probabilities, which
    ``cross_val_score`` throws away. That is the only reason its fold loop is written out, so the
    scores it produces have to be the same numbers. The comparison is against the array actually
    written to the artifact, so it checks what ships rather than a fresh recomputation of it.
    """
    checked = {}
    for arm in arms:
        recorded = np.array(audited["arms"][arm.method_id]["fold_roc_auc"])
        through_sklearn = detection._cv_estimator(
            arm.estimator_factory(task), task.layers[arm.layer], task.y)
        if not np.array_equal(through_sklearn, recorded):
            raise AssertionError(
                f"{task.dataset}/{arm.method_id}: the recorded fold scores and cross_val_score "
                f"disagree; max |difference| {np.abs(through_sklearn - recorded).max():.3e}"
            )
        checked[arm.method_id] = float(recorded.mean())
    return checked


def _spline_blocks(fitted):
    """Every fitted ``SplineTransformer`` inside one fitted arm, with the columns it was given.

    Three shapes occur among the arms: the transformer as the first step of a pipeline (columns
    ``None``, meaning all of them), the transformer inside a ``ColumnTransformer`` that splines the
    size columns and passes the dependency columns through, and either of those wrapped in a
    ``GridSearchCV`` whose ``best_estimator_`` is refitted on the whole training set it was handed.
    An arm with no spline in it yields nothing and is skipped.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import SplineTransformer

    pipeline = fitted.best_estimator_ if isinstance(fitted, GridSearchCV) else fitted
    steps = getattr(pipeline, "steps", None)
    if steps is None:
        return pipeline, []
    found = []
    for _, step in steps:
        if isinstance(step, SplineTransformer):
            found.append((step, None))
        elif isinstance(step, ColumnTransformer):
            for _, transformer, columns in step.transformers_:
                if isinstance(transformer, SplineTransformer):
                    found.append((transformer, list(columns)))
    return pipeline, found


class _FoldLocality:
    """Check every spline-bearing fit of every arm, inside the fold loop that produced it.

    Called by ``detection_audit`` after each of the 25 fits of each arm, so it reaches all 300
    fitted estimators of the twelve arms without refitting one of them. That matters for the four
    nested arms, where a refit costs as much again as scoring the arm did.

    Per fit it establishes two things, both by exact array comparison:

    - The knot vector inside the fitted transformer is the knot vector a fresh transformer with the
      same parameters produces from that fold's training rows alone, and the scaler's moments are
      the moments of the transformed training rows. If any test row had reached ``fit``, the
      quantile knots and the moments would both move.
    - Those knots differ from the knots the same transformer fitted on the complete matrix would
      have. Without this the comparison above could pass on an implementation that leaked, because
      a knot vector fitted on all rows is also "the knots of something".

    The second comparison can legitimately tie. Quantile knots at a small ``n_knots`` are order
    statistics of a discrete column, and an 80 percent subsample often has the same minimum,
    median and maximum as the whole matrix. The counts are therefore recorded per arm rather than
    required to be 25, and only an arm where the comparison never separates is an error, because
    for that arm the first comparison would have no negative control at all.
    """

    def __init__(self, task, detection) -> None:
        self.task = task
        self.detection = detection
        self.per_arm: dict[str, dict] = {}
        self._whole: dict = {}

    def _whole_matrix_reference(self, transformer, matrix, columns):
        from sklearn.base import clone

        key = (id(matrix), None if columns is None else tuple(columns),
               transformer.n_knots, transformer.degree, transformer.knots,
               transformer.extrapolation, transformer.include_bias)
        if key not in self._whole:
            rows = matrix if columns is None else matrix[:, columns]
            self._whole[key] = clone(transformer).fit(rows)
        return self._whole[key]

    def __call__(self, arm, seed, fold, train, fitted) -> None:
        from sklearn.base import clone
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        matrix = self.task.layers[arm.layer]
        pipeline, blocks = _spline_blocks(fitted)
        if not blocks:
            return
        record = self.per_arm.setdefault(arm.method_id, {
            "layer": arm.layer, "fits": 0, "spline_blocks": 0,
            "fits_whose_knots_differ_from_the_whole_matrix": 0,
        })
        differs = False
        for transformer, columns in blocks:
            rows = matrix[train] if columns is None else matrix[train][:, columns]
            training_only = clone(transformer).fit(rows)
            for got, want in zip(transformer.bsplines_, training_only.bsplines_):
                if not np.array_equal(got.t, want.t):
                    raise AssertionError(
                        f"{self.task.dataset}/{arm.method_id} seed {seed} fold {fold}: the fitted "
                        "knots are not the knots of this fold's training rows"
                    )
            whole = self._whole_matrix_reference(transformer, matrix, columns)
            if any(not np.array_equal(got.t, ref.t)
                   for got, ref in zip(transformer.bsplines_, whole.bsplines_)):
                differs = True
            record["spline_blocks"] += 1

        names = [name for name, _ in pipeline.steps]
        if "standardscaler" in names:
            index = names.index("standardscaler")
            scaler = pipeline.steps[index][1]
            before = Pipeline(pipeline.steps[:index]).transform(matrix[train])
            want = StandardScaler().fit(before)
            if not (np.array_equal(scaler.mean_, want.mean_)
                    and np.array_equal(scaler.scale_, want.scale_)):
                raise AssertionError(
                    f"{self.task.dataset}/{arm.method_id} seed {seed} fold {fold}: the fitted "
                    "scaling is not the scaling of this fold's training rows"
                )
        record["fits"] += 1
        record["fits_whose_knots_differ_from_the_whole_matrix"] += int(differs)

    def record(self) -> dict:
        """Finish the check and return what it establishes, with the score comparison beside it.

        The score comparison is descriptive and carries no assertion. Scoring the whole-matrix
        expansion measures how much this particular leak would have moved the mean fold AUC, and
        the answer on both corpora is under a thousandth, in opposite directions. A test that a
        leak moves a score by four ten-thousandths is not a test; the array comparisons above are
        what establishes fold locality, and this number is recorded so a reader can see how little
        the score alone would have revealed.
        """
        from sklearn.preprocessing import SplineTransformer, StandardScaler

        detection, task = self.detection, self.task
        if not self.per_arm:
            raise AssertionError(f"{task.dataset}: no spline-bearing arm was inspected")
        for name, record in sorted(self.per_arm.items()):
            if record["fits"] != 25:
                raise AssertionError(
                    f"{task.dataset}/{name}: {record['fits']} fits were checked, not the 25 the "
                    "arm is scored on, so the check no longer covers the arm"
                )
            if record["fits_whose_knots_differ_from_the_whole_matrix"] == 0:
                raise AssertionError(
                    f"{task.dataset}/{name}: the fitted knots equal the whole-matrix knots on "
                    "every one of the 25 fits, so this arm has no negative control and the "
                    "training-fold comparison cannot tell a leak from a clean fit"
                )
        if "size (spline)" not in self.per_arm:
            raise AssertionError(f"{task.dataset}: the board's spline entrant was not inspected")

        matrix, labels = task.layers["flat"], task.y
        whole = SplineTransformer(**detection._SPLINE).fit(matrix)
        honest = detection._cv_estimator(detection._spline_estimator, matrix, labels).mean()
        leaked = StandardScaler().fit_transform(whole.transform(matrix))
        leaky = detection._cv_estimator(lambda seed: detection._logistic(), leaked, labels).mean()
        return {
            "establishes": "for every one of the 25 fits of every spline-bearing arm, the fitted "
                           "knot vectors are exactly those of a transformer fitted on that fold's "
                           "training rows, and the fitted scaler's moments are exactly those of "
                           "the transformed training rows; and on at least one fit per arm those "
                           "knots differ from the whole-matrix knots, so the comparison "
                           "discriminates rather than passing vacuously",
            "does_not_establish": "that a leak would be visible in the score. Refitting the basis "
                                  "on the complete matrix moves the mean fold AUC by less than a "
                                  "thousandth, and in opposite directions on the two corpora, so "
                                  "the score comparison below is descriptive and has no power as "
                                  "a leak test",
            "arms_inspected": sorted(self.per_arm),
            "fits_inspected": sum(record["fits"] for record in self.per_arm.values()),
            "per_arm": self.per_arm,
            "declared_arm_mean_roc_auc": float(honest),
            "leaked_basis_mean_roc_auc": float(leaky),
            "difference": float(honest - leaky),
        }


def _check_gbt_matches_grade(detection) -> dict:
    """Compare the restated gradient-boosted-tree parameters against GRADE's ``_m_gbt``.

    ``size_baseline_nonlinear.py`` is not in GRADE's committed tree, so a clean checkout does not
    have it and this check reports that instead of failing. Where it is present, the parameters
    have to agree, which is what stops the restated copy from drifting.
    """
    try:
        from size_baseline_nonlinear import _m_gbt
    except Exception as exc:  # pragma: no cover - depends on the local GRADE checkout
        return {"compared": False, "reason": f"{type(exc).__name__}: {exc}"}
    for seed in (0, 3):
        theirs = _m_gbt(seed).get_params()
        ours = detection._gbt_estimator(seed).get_params()
        if theirs != ours:
            differing = {key: (theirs.get(key), ours.get(key))
                         for key in set(theirs) | set(ours) if theirs.get(key) != ours.get(key)}
            raise AssertionError(f"_gbt_estimator has drifted from GRADE's _m_gbt: {differing}")
    return {"compared": True, "source": "grade/experiment/size_baseline_nonlinear.py::_m_gbt"}


def _check_reference_paths(tasks, detection, core) -> dict:
    """Score the reference rows through ``RunPipeline`` and require the audit to agree exactly.

    The three linear references are selected out of ``post_detection_methods()`` rather than
    constructed here, so this fails if any is dropped from the registry, and they are scored by the
    same pipeline that produces the printed board. Only the feature-layer rows are run: the graph and
    LLM entrants answer other questions and would pull in a torch stack for no gain here.

    The spline arm is appended from the audit's own factory instead of being read out of the
    registry, because **it is deliberately not a board entrant**. Registering it would add a row to
    both detection blocks, move the inventory from 72 entrants to 73, and fail ``parse_board``
    against its pinned count. The audit needs the arm; the board does not have it. Scoring it through
    the same ``RunPipeline`` keeps the equality assertion below meaningful either way.
    """
    wanted = {"size (flat)", "auditable (size+deps)", "full"}
    methods = [method for method in detection.post_detection_methods()
               if method.method_id in wanted]
    missing = wanted - {method.method_id for method in methods}
    if missing:
        raise AssertionError(f"post_detection_methods() no longer registers {sorted(missing)}")
    methods.append(next(arm for arm in detection.post_detection_audit_arms()
                        if arm.method_id == "size (spline)"))
    rows = core.RunPipeline(tasks, methods).run()
    return {f"{row.dataset}/{row.method}": float(row.metrics["roc_auc"]) for row in rows}


# --- the feature-block disclosure ---------------------------------------------------------------


def _feature_identities(task, detection) -> dict:
    """The exact identities and the constant column inside the historical feature blocks.

    Recorded, and deliberately not repaired. Dropping the redundant column changes a regularized
    fit, because it changes the penalty geometry, so a deduplicated block is a different
    specification with its own values. The one below is a labelled diagnostic and is not an arm.
    """
    from grade import feature_vector

    names = {layer: feature_vector(task.graphs[0], layer=layer)[0]
             for layer in ("flat", "flatdep", "full")}
    flat, flatdep, full = (task.layers[layer] for layer in ("flat", "flatdep", "full"))
    keep = [index for index, name in enumerate(names["flat"]) if name != "n_decisions"]
    reduced = detection._cv_estimator(detection._linear_estimator, flat[:, keep], task.y)
    return {
        "columns": names,
        "centered_rank": {layer: detection._rank(task.layers[layer])
                          for layer in ("flat", "flatdep", "full")},
        "n_columns": {layer: int(task.layers[layer].shape[1])
                      for layer in ("flat", "flatdep", "full")},
        "n_decisions_equals_n_steps_minus_n_tool_calls":
            bool(np.array_equal(flat[:, 2], flat[:, 0] - flat[:, 1])),
        "n_agents_distinct_values": [float(value) for value in np.unique(flat[:, 3])],
        "hub_shares_equal_rows": {
            "flatdep": int(np.count_nonzero(flatdep[:, 6] == flatdep[:, 7])),
            "full": int(np.count_nonzero(full[:, 10] == full[:, 11])),
            "n_runs": int(len(task.y)),
        },
        "diagnostic_linear_flat_without_n_decisions": {
            "mean_roc_auc": float(reduced.mean()),
            "seed_mean_roc_auc": [float(value) for value in reduced.mean(axis=1)],
            "note": "a different specification, reported so the redundancy is visible; it is not "
                    "a board arm and does not replace the historical flat block",
        },
    }


# --- driver -------------------------------------------------------------------------------------


def build(corpora: list[str], preflight: bool) -> dict:
    from catchbench import core
    from catchbench.corpora import verify_corpus_heads, verify_pinned_fetches
    from catchbench.detection import PostDetection

    from catchbench import detection

    names = {CORPUS_NAMES[corpus] for corpus in corpora}
    revisions = verify_corpus_heads(names=names) if preflight else {}

    tasks = [PostDetection(corpus) for corpus in corpora]
    for task in tasks:
        task.setup()
    if preflight:
        verify_pinned_fetches(names=names)

    arms = detection.post_detection_audit_arms()
    verification = {
        "reference_rows_through_runpipeline": _check_reference_paths(
            tasks, detection, core),
        "gbt_matches_grade": _check_gbt_matches_grade(detection),
        "per_corpus": {},
    }
    result = {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "corpus_revisions": revisions,
            "seed_interval": detection._SEED_INTERVAL_NOTE,
            "spline": dict(detection._SPLINE),
            "knot_grid": {key: list(value) for key, value in detection._KNOT_GRID.items()},
            "knot_c_grid": {key: list(value) for key, value in detection._KNOT_C_GRID.items()},
            "board_entrants_added": [],
            "note": "the three linear references are board entrants; every other arm here, the spline arms included, is a controlled-audit arm and is on no board",
        },
        "verification": verification,
        "features": {},
        "corpora": {},
    }
    for task in tasks:
        locality = _FoldLocality(task, detection)
        audited = detection.detection_audit(task, arms, inspect=locality)
        result["corpora"][task.dataset] = audited
        result["features"][task.dataset] = _feature_identities(task, detection)
        verification["per_corpus"][task.dataset] = {
            "swapped_estimator_reproduces_grade_cv": _check_folds_are_the_boards(task, detection),
            "recorded_folds_match_cross_val_score": _check_detail_matches_cross_val_score(
                task, detection, arms, audited),
            "spline_is_fitted_inside_the_training_fold": locality.record(),
        }

    for task in tasks:
        piped = verification["reference_rows_through_runpipeline"][f"{task.dataset}/size (spline)"]
        audited = result["corpora"][task.dataset]["arms"]["size (spline)"]["mean_roc_auc"]
        if piped != audited:
            raise AssertionError(
                f"{task.dataset}: RunPipeline scores size (spline) at {piped!r} and the audit at "
                f"{audited!r}; one of the two paths has changed"
            )
    return result


def _report(result: dict) -> None:
    for corpus, record in result["corpora"].items():
        print(f"\n[POST] post_detection :: {corpus}  "
              f"({record['n_runs']} runs, {record['n_failed']} failed)")
        width = max(len(name) for name in record["arms"])
        for name, arm in record["arms"].items():
            mark = "board" if arm["board_entrant"] else "audit"
            print(f"  {name:{width}s}  {arm['mean_roc_auc']:.6f}  [{mark}] {arm['layer']}")
        print("  contrasts (split-seed t interval; NOT run-level inference)")
        for label, contrast in record["contrasts"].items():
            low, high = contrast["seed_interval_95"]
            mark = "matched   " if contrast["matched"] else "CONFOUNDED"
            print(f"    [{mark}] {label:{width * 2}s} {contrast['seed_mean_difference']:+.6f} "
                  f"[{low:+.6f}, {high:+.6f}]  {contrast['seeds_positive']}/"
                  f"{contrast['seeds']} seeds positive")
        for label, contrast in record["contrasts"].items():
            if contrast["confound"]:
                print(f"    CONFOUNDED {label}: {contrast['confound']}")
        blocks = result["features"][corpus]
        ranks = ", ".join(f"{layer} {blocks['centered_rank'][layer]}/{blocks['n_columns'][layer]}"
                          for layer in ("flat", "flatdep", "full"))
        print(f"  centered rank: {ranks}")
        identities = record["run_identities"]
        if identities["available"]:
            print(f"  run identities: {identities['n_distinct_task_clusters']} task clusters, "
                  f"runs per cluster {identities['runs_per_task_cluster']}")
        else:
            print(f"  run identities: unavailable ({identities['reason']})")


def main() -> int:
    args = _parse_args()
    result = build(list(args.corpora), preflight=not args.no_preflight)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    _report(result)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
