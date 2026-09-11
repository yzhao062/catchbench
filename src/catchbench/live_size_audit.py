r"""LIVE prefix size control: the endpoint audit's matched arms, read at every prefix.

The POST board reports the dependency increment against a LINEAR size reference. ``detection_audit``
exists because that reference is a modelling choice rather than a fact about the features, and on
SWE-Gym the choice decides the sign: the increment is ``+0.142`` against the linear size model and
``-0.010`` against a fixed cubic spline over the same four size columns. The LIVE board carries the
linear reading at four prefixes and carries no flexible size reference anywhere, so the same
question is open at the 25 percent prefix, which is the cell the early-warning claim rests on. This
module scores the matched control there, and at the other three prefixes beside it.

**Only the size columns receive the spline.** The arm the manuscript's endpoint number comes from is
``detection._MixedSplineAUC``: a ``ColumnTransformer`` that applies the declared ``_SPLINE`` to the
four flat size and count columns and passes the four normalized dependency columns through
unchanged. Splining all eight ``flatdep`` columns is a DIFFERENT control and is a different number.
On the committed endpoint records the two separate by more than a factor of three, ``-0.005404``
against ``-0.016325`` on seed-averaged out-of-fold predictions, so the distinction is not a detail
that rounds away.

Nothing here restates that arm. ``live_size_audit_arms`` returns ``detection``'s own arm objects,
and ``prefix_view`` hands each of them one prefix of a ``LiveStreaming`` task shaped like the
``PostDetection`` task they were written against. The column split, its check against that prefix's
own ``flat`` matrix, and the ``ColumnTransformer`` therefore all come from ``_MixedSplineAUC``
itself. A change to GRADE's feature order fails inside that class at every prefix instead of
quietly splining the wrong block here, and a prefix arm cannot drift away from the endpoint arm
because it is the endpoint arm.

**No arm here is a board entrant.** ``live.live_streaming_methods`` is unchanged, the CLI roster is
unchanged, and ``live_breakdown`` still prints what it printed. A control belongs beside the rows it
interprets, and registering one would move the entrant inventory and change what the LIVE board is.
This is the same separation ``detection_audit`` keeps from ``post_detection_methods``.

Fitting goes through ``detection._cv_estimator_detail`` rather than ``_cv_estimator``. The declared
estimand is the ROC-AUC of an arm's five-seed-averaged out-of-fold failure probabilities, and the
interval is paired on those vectors; a mean fold score discards both the vectors and the pairing.
The mean of fold AUCs is kept beside it as a separate reproduction quantity, because it is what the
board prints and what the published prefix cells are checked against. The two are not
interchangeable: on SWE-Gym at 25 percent the published out-of-fold contrast is ``+0.101`` while the
displayed ``0.742`` minus ``0.629`` is about ``0.112``.

Row identifiers are not recovered here. ``detection.run_identities`` is typed to ``PostDetection``
and replays under POST's ``len(steps) >= 2`` filter, and ``tools/statistical_tests.py`` already
carries ``_tau_live_clustering``, the replay under LIVE's own ``>= 4`` predicate that verifies its
alignment against LIVE's labels and 100 percent step counts. The driver reads the run keys from
there and embeds them, so this module holds one replay's worth of that logic rather than two.

The analysis this module serves is declared in
``research/catchbench-live-size-control-declaration-2026-09-10.md``, Part A, frozen before any
prefix spline was fitted. ``DECLARATION`` names it, and the constants below are its frozen
quantities written down in code so a later run cannot quietly choose a different one.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from catchbench import _reuse  # noqa: F401  side effect: sets sys.path for grade + auditable

import numpy as np  # noqa: E402

from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from agent_failure_detection import SEEDS  # noqa: E402  GRADE's split seeds, unchanged

from catchbench.detection import (  # noqa: E402
    _EstimatorAUC,
    _MixedSplineAUC,
    _cv_estimator_detail,
    _estimator_specification,
    _linear_estimator,
    _spline_estimator,
)
from catchbench.live import _PREFIXES, LiveStreaming  # noqa: E402


DECLARATION = "research/catchbench-live-size-control-declaration-2026-09-10.md, Part A"

# The four prefixes are frozen by A3 and are the board's own, imported rather than retyped so a
# change to the board's prefix set fails the identity check below instead of silently scoring a
# different sweep under the declared labels.
AUDIT_PREFIXES = _PREFIXES

# A3, confirmed by preflight and not accommodated: a different population is a blocker, because
# every published prefix number this audit is measured against was computed on these rows.
DECLARED_POPULATION = {"swegym": 376, "tau": 660}

# A10 fixes the numerical tolerance BEFORE any fit, so a comparison that fails cannot be rescued by
# widening it afterwards. Two levels, and the split is not a convenience:
#
#   PARITY_ATOL applies to a float score compared against a number a previous run recorded. The
#   expected difference is exactly zero, because the same code re-runs on the same rows in the same
#   environment; the allowance covers only the last bits of a float64 that has been through a JSON
#   round trip. A difference above it means the numerical path moved and is a finding, not a
#   tolerance to loosen.
#
#   Structure carries NO tolerance. Fold index arrays, column boundaries, feature matrices, knot
#   vectors, and scaler moments are compared with exact equality, because an approximate match there
#   would mean the rows or the splits are not the same rows or splits, and no epsilon makes that
#   acceptable.
PARITY_ATOL = 1e-12

# A5 nominates the primary cell before any of the eight is computed. It is recorded here so the
# nomination travels with the code that produces the number.
PRIMARY_CELL = ("swegym", 0.25)

# A6 and A7 freeze the RNG labels literally. Written down now, before any fit, because a label
# chosen after a bootstrap has been seen is not a frozen label.
RNG_LABELS = {
    (corpus, prefix): f"live_size_audit.{corpus}.{int(round(prefix * 100))}.mixed_minus_spline"
    for corpus in ("swegym", "tau")
    for prefix in AUDIT_PREFIXES
}
TEMPORAL_RNG_LABEL = "live_size_audit.swegym.25_minus_100"
TEMPORAL_CONTRAST = (("swegym", 0.25), ("swegym", 1.00))

# A6 fixes the tau-bench interval at 10,000 draws seeded through ``_rng_for(label, 20260907)``, and
# A7 fixes the temporal contrast at 10,000 draws off the same base seed. Both are written down here
# beside the labels they are paired with rather than at the driver, for the reason the labels are:
# a draw count or a base seed that lives where a rerun can pass a different one is not frozen.
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_BASE_SEED = 20260907

# A8's threshold, and A8's own words about what it is. It is the size at which a supported positive
# is called substantial for this paper; it is not a validated operational threshold and nothing here
# treats it as one.
SUBSTANTIAL_EFFECT = 0.03

# The declaration's four arms, keyed by the method id the committed records already use. The role
# names are A2's, kept beside the ids so the artifact says which arm is which control without a
# reader having to map the two vocabularies themselves.
ARM_ROLES = {
    "size (flat)": "linear size reference",
    "auditable (size+deps)": "linear structural reference",
    "size (spline)": "fixed flexible size reference",
    "size-spline + linear-deps": "matched structural control",
}

# The declared contrast, high arm minus low arm, at every prefix. One pair, named here rather than
# searched for: A5 makes the matched control against the flexible size reference the estimand, and
# the two linear arms are the reproduction route rather than a second contrast.
MATCHED_CONTRAST = ("size-spline + linear-deps", "size (spline)")

# The reproduction route: the pair whose difference the manuscript's published prefix numbers are.
LINEAR_CONTRAST = ("auditable (size+deps)", "size (flat)")


class _PrefixView:
    """One prefix of a ``LiveStreaming`` task, shaped like the ``PostDetection`` task it feeds.

    ``detection``'s arms read ``task.layers[layer]`` and call ``task.setup()``; a ``LiveStreaming``
    task keeps ``task.layers_at[prefix][layer]`` instead and holds no graphs. This view is the
    adapter between the two, and it exists so the arms can be REUSED rather than reimplemented.
    That matters most for ``_MixedSplineAUC``, whose ``columns`` reads the size width off
    ``layers["flat"]`` and refuses to proceed unless the flat block is still the leading columns of
    ``flatdep``: handed this view, that check runs against the prefix's own two matrices, at every
    prefix, in the class the endpoint number came from.

    ``dataset`` carries the prefix so an assertion raised inside a reused arm or a reused
    verification names the cell it failed on. ``setup`` is a no-op because the underlying task is
    already built; the view never loads anything and never mutates the task.
    """

    def __init__(self, task: LiveStreaming, prefix: float) -> None:
        task.setup()
        if prefix not in task.layers_at:
            raise KeyError(f"{task.dataset}: no feature matrices at prefix {prefix!r}")
        self.task = task
        self.prefix = float(prefix)
        self.corpus = task.corpus
        self.dataset = f"{task.dataset}@{int(round(self.prefix * 100))}%"
        self.layers = task.layers_at[prefix]
        self.y = task.y

    def setup(self) -> None:
        """Already built by the underlying task; present because the reused arms call it."""
        return None


def prefix_view(task: LiveStreaming, prefix: float) -> _PrefixView:
    """One prefix of ``task``, in the shape ``detection``'s arms and verifications expect."""
    return _PrefixView(task, prefix)


def live_size_audit_arms() -> list:
    """The four declared arms, in table order, as ``detection``'s own arm objects.

    They are not prefix-shaped copies. The endpoint number the manuscript cites is what these
    objects produce, and the whole point of the control is that a prefix cell is the same arm read
    at a different prefix. Constructing them here from ``detection``'s factories means the spline
    configuration, the column split, and the classifier reach every prefix from one place.

    The first two reproduce the published linear prefix numbers and are the route check. The last
    two are the new controlled pair. No other arm is fitted: boosted trees, all-column splines and
    nested knot selection are outside the declaration, and ``detection.post_detection_audit_arms``
    is where those live for the endpoint.
    """
    return [
        _EstimatorAUC("size (flat)", "flat", _linear_estimator),
        _EstimatorAUC("auditable (size+deps)", "flatdep", _linear_estimator),
        _EstimatorAUC("size (spline)", "flat", _spline_estimator),
        _MixedSplineAUC(),
    ]


def fold_splits(labels: Sequence[int]) -> dict[int, list[tuple[np.ndarray, np.ndarray]]]:
    """The (train, test) index arrays every arm at every prefix is scored on.

    ``StratifiedKFold(shuffle=True, random_state=seed).split(X, y)`` reads only ``len(X)`` and
    ``y``, so identical rows get identical assignments whatever matrix is passed. That is the
    property A4 requires and it is a property of scikit-learn's implementation rather than of this
    code, which is exactly why the preflight checks it on the real matrices instead of citing it.
    """
    labels = np.asarray(labels)
    splits = {}
    for seed in SEEDS:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        splits[int(seed)] = [(train.copy(), test.copy())
                             for train, test in cv.split(np.zeros(len(labels)), labels)]
    return splits


def fold_membership(labels: Sequence[int]) -> list[list[int]]:
    """Per seed, the fold each row was held out in: the pairing an interval needs, stored compactly.

    A list of five (train, test) index pairs per seed says the same thing in twenty-five arrays.
    One fold index per row per seed reconstructs those pairs exactly and survives a JSON round trip
    without a reader having to trust that the order was preserved.
    """
    labels = np.asarray(labels)
    membership = []
    for seed, folds in sorted(fold_splits(labels).items()):
        row = np.full(len(labels), -1, dtype=int)
        for index, (_, test) in enumerate(folds):
            row[test] = index
        if int(row.min()) < 0:
            raise AssertionError(f"seed {seed}: {int((row < 0).sum())} rows were never held out")
        membership.append([int(value) for value in row])
    return membership


def score_arm(arm, view: _PrefixView, inspect=None) -> dict:
    """Score one arm at one prefix, keeping everything a fold score alone throws away.

    ``inspect`` is handed straight to ``detection._cv_estimator_detail``, which calls it after each
    of the 25 fits and before the fit is scored. That is how a fold-locality verification reaches
    every fitted estimator without refitting one, and it is the only route by which anything outside
    this function sees a fitted object.
    """
    labels = np.asarray(view.y)
    matrix = view.layers[arm.layer]
    factory = arm.estimator_factory(view)
    fold_auc, oof, selected = _cv_estimator_detail(factory, matrix, labels, inspect=inspect)
    per_seed_pooled = [float(roc_auc_score(labels, row)) for row in oof]
    specification = _estimator_specification(factory)
    return {
        "role": ARM_ROLES.get(arm.method_id),
        "layer": arm.layer,
        "board_entrant": False,
        "estimator": specification["estimator"],
        "params": specification["params_at_seed_0"],
        "params_vary_with_seed": specification["varies_with_seed"],
        "n_columns": int(matrix.shape[1]),
        # The declared estimand: the ROC-AUC of the five-seed-averaged out-of-fold probabilities.
        "seed_averaged_oof_roc_auc": float(roc_auc_score(labels, oof.mean(axis=0))),
        "per_seed_oof_roc_auc": per_seed_pooled,
        # The separate reproduction quantity: the mean of the 25 fold AUCs, which is what the board
        # prints. It never receives an interval computed for the quantity above.
        "mean_fold_roc_auc": float(fold_auc.mean()),
        "seed_mean_fold_roc_auc": [float(value) for value in fold_auc.mean(axis=1)],
        "fold_roc_auc": [[float(value) for value in row] for row in fold_auc],
        "oof_proba": [[float(value) for value in row] for row in oof],
        "selected_hyperparameters": selected,
        "nonfinite_fold_scores": int(np.count_nonzero(~np.isfinite(fold_auc))),
        "nonfinite_oof_values": int(np.count_nonzero(~np.isfinite(oof))),
    }


def live_size_audit(task: LiveStreaming, arms: list | None = None,
                    prefixes: Sequence[float] | None = None, inspect=None) -> dict:
    """Score the declared arms at every declared prefix on one LIVE corpus.

    ``inspect``, when given, is called as ``inspect(arm, view, seed, fold, train, fitted)`` after
    every fit, so a verification reaches all 400 fitted estimators of the batch without refitting
    one. It receives the view as well as the arm because a finding has to name the prefix it was
    found at.

    Returns the arms, the matched effect at each prefix, and the labels and fold membership the
    intervals are computed from. It writes no file and prints nothing; A9's result file and its
    appendix table are the driver's business.
    """
    task.setup()
    labels = np.asarray(task.y)
    arms = list(live_size_audit_arms() if arms is None else arms)
    prefixes = tuple(task.prefixes if prefixes is None else prefixes)

    per_prefix: dict[str, dict] = {}
    for prefix in prefixes:
        view = prefix_view(task, prefix)
        scored: dict[str, dict] = {}
        for arm in arms:
            watcher = None
            if inspect is not None:
                def watcher(seed, fold, train, fitted, arm=arm, view=view):
                    inspect(arm, view, seed, fold, train, fitted)
            scored[arm.method_id] = score_arm(arm, view, inspect=watcher)
        per_prefix[_prefix_key(prefix)] = {
            "prefix": float(prefix),
            "arms": scored,
            "effects": _effects(scored),
        }

    return {
        "corpus": task.dataset,
        "task_id": task.task_id,
        "declaration": DECLARATION,
        "n_runs": int(len(labels)),
        "n_failed": int(labels.sum()),
        "labels": [int(value) for value in labels],
        "seeds": [int(seed) for seed in SEEDS],
        "fold_membership": fold_membership(labels),
        "prefixes": [float(prefix) for prefix in prefixes],
        "arm_roles": dict(ARM_ROLES),
        "rng_labels": {_prefix_key(prefix): RNG_LABELS[(task.corpus, prefix)]
                       for prefix in prefixes if (task.corpus, prefix) in RNG_LABELS},
        "by_prefix": per_prefix,
    }


def _prefix_key(prefix: float) -> str:
    """``0.25`` as ``'25'``: the form the declaration's RNG labels and claim ids are written in."""
    return str(int(round(float(prefix) * 100)))


def _effects(scored: Mapping[str, dict]) -> dict:
    """The declared contrast and the reproduction contrast at one prefix, on both estimands.

    Both are reported on the seed-averaged out-of-fold quantity, which is the estimand, and on the
    mean of fold AUCs, which is what the board prints. Reporting one without the other is what makes
    ``+0.101`` and ``0.112`` look like the same number.
    """
    effects = {}
    for label, (high, low) in (("matched", MATCHED_CONTRAST), ("linear", LINEAR_CONTRAST)):
        if high not in scored or low not in scored:
            continue
        effects[f"{high} - {low}"] = {
            "kind": label,
            "high": high,
            "low": low,
            "seed_averaged_oof_difference": (scored[high]["seed_averaged_oof_roc_auc"]
                                             - scored[low]["seed_averaged_oof_roc_auc"]),
            "mean_fold_difference": (scored[high]["mean_fold_roc_auc"]
                                     - scored[low]["mean_fold_roc_auc"]),
        }
    return effects


def classify_outcome(point: float, low: float, high: float) -> dict:
    """A8's branch for the primary effect and its marginal interval.

    Written before the batch was run, which is the only property that makes it a rule rather than a
    reading. A8's four bullets are applied in the order support-then-size, because that is the order
    they constrain: an interval containing zero supports no sign whatever the point estimate is, and
    only once a sign is supported does the declared ``+0.03`` decide which positive claim is
    licensed. A supported negative is the endpoint's own outcome read at a prefix, the increment
    erased or reversed once the size reference is allowed to bend, and it is the branch A1's
    ``-0.010`` would fall into.

    ``upper_endpoint_below_substantial`` carries A8's fourth bullet as a field rather than a fifth
    branch. That bullet says something the unresolved branch does not: an interval whose upper end
    sits below ``+0.03`` rules out that planning magnitude under this conditional analysis, while
    still establishing no positive effect and no zero. A8 gives it no name of its own, so it travels
    beside the branch instead of replacing one.

    Nothing here reads a prefix other than the one it is handed. A5 nominates the primary cell and
    A8 forbids promoting a better result at 50 or 75 percent, so the driver passes the primary cell
    and this function has no way to see the others.
    """
    excludes_zero = bool(low > 0.0 or high < 0.0)
    if not excludes_zero:
        branch = "unresolved"
    elif high < 0.0:
        branch = "erased-or-reversed"
    elif point >= SUBSTANTIAL_EFFECT:
        branch = "substantial-positive"
    else:
        branch = "small-positive"
    return {
        "branch": branch,
        "point": float(point),
        "interval_95": [float(low), float(high)],
        "substantial_threshold": SUBSTANTIAL_EFFECT,
        "interval_excludes_zero": excludes_zero,
        "point_at_least_substantial": bool(point >= SUBSTANTIAL_EFFECT),
        "upper_endpoint_below_substantial": bool(high < SUBSTANTIAL_EFFECT),
        "rule": "A8: substantial-positive when the point is at least +0.03 AND the marginal "
                "interval excludes zero; small-positive when the interval excludes zero on the "
                "positive side at a smaller point; erased-or-reversed when it excludes zero on the "
                "negative side; unresolved when it contains zero",
        "threshold_is": "a declared editorial judgment about what would be a worthwhile result for "
                        "this paper, stated as such in A8; not a validated operational threshold",
    }


# --- execution validity: the checks A10 requires before any score is read ------------------------
#
# These establish that the cells are what they are declared to be. They raise on failure rather than
# returning a verdict, because a cell whose rows or folds moved is not a weaker result, it is a
# different analysis, and the declaration's stopping rule does not admit repairing one into the
# other. Each returns what it established, so the run artifact carries the evidence and not only the
# fact that something passed.


def check_population(task: LiveStreaming) -> dict:
    """The corpus is the one the published prefix numbers were computed on.

    A3 fixes 376 SWE-Gym runs and 660 tau-bench runs. A different count is a blocker: every number
    this audit reproduces or is measured against came from those rows, and re-deriving the
    comparison on a moved population would compare two different things.
    """
    task.setup()
    labels = np.asarray(task.y)
    declared = DECLARED_POPULATION.get(task.corpus)
    if declared is None:
        raise AssertionError(f"no declared population for LIVE corpus {task.corpus!r}")
    if len(labels) != declared:
        raise AssertionError(
            f"{task.dataset}: {len(labels)} runs today against the declared {declared}. A3 makes "
            "this a blocker rather than something to accommodate."
        )
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise AssertionError(f"{task.dataset}: labels are not the two-class failure labels")
    return {
        "corpus": task.corpus,
        "n_runs": int(len(labels)),
        "n_failed": int(labels.sum()),
        "declared": int(declared),
    }


def check_column_boundaries(task: LiveStreaming,
                            prefixes: Sequence[float] | None = None) -> dict:
    """The mixed arm's size / dependency split, at every prefix, through the arm's own check.

    ``_MixedSplineAUC.columns`` reads the size width off that prefix's ``flat`` matrix and refuses
    to proceed unless the flat block is still the leading columns of ``flatdep``. Calling it here is
    the check: a changed feature order raises inside the class the endpoint number came from rather
    than producing a spline over the wrong block. The widths are returned so the artifact shows the
    boundary was 4 and 8 at every prefix rather than merely that nothing raised.
    """
    task.setup()
    prefixes = tuple(task.prefixes if prefixes is None else prefixes)
    mixed = _MixedSplineAUC()
    boundaries = {}
    for prefix in prefixes:
        view = prefix_view(task, prefix)
        size, deps = mixed.columns(view)
        flat = view.layers["flat"]
        flatdep = view.layers[mixed.layer]
        if size != list(range(flat.shape[1])) or deps != list(range(flat.shape[1],
                                                                   flatdep.shape[1])):
            raise AssertionError(f"{view.dataset}: the mixed arm's column split is not contiguous")
        if not np.array_equal(flatdep[:, size], flat):
            raise AssertionError(f"{view.dataset}: the splined block is not the flat matrix")
        boundaries[_prefix_key(prefix)] = {
            "splined_columns": size,
            "passthrough_columns": deps,
            "flat_shape": [int(value) for value in flat.shape],
            "flatdep_shape": [int(value) for value in flatdep.shape],
        }
    widths = {(entry["flat_shape"][1], entry["flatdep_shape"][1]) for entry in boundaries.values()}
    if len(widths) != 1:
        raise AssertionError(f"{task.dataset}: the layer widths move across prefixes: {widths}")
    return {
        "checked_by": "catchbench.detection._MixedSplineAUC.columns, per prefix",
        "establishes": "at every prefix the splined block is exactly that prefix's flat matrix and "
                       "the passthrough block is the remaining flatdep columns, so the dependency "
                       "columns enter linearly and only the size columns are splined",
        "per_prefix": boundaries,
    }


def check_row_and_fold_alignment(task: LiveStreaming,
                                 prefixes: Sequence[float] | None = None) -> dict:
    """Row ``i`` is the same run at every prefix, and every arm at every prefix shares the folds.

    Two separate facts, and neither follows from the other.

    Rows: each prefix's matrices are built from ``task.runs`` in one comprehension, so row order is
    shared by construction. The check binds that to the data instead of to the construction: the
    ``n_steps`` column at prefix ``p`` must equal ``len(_prefix_steps(steps, p))`` for the run in
    that row, which is ``max(2, ceil(p * len(steps)))`` capped at the run's length. A row that had
    drifted to another run would have to carry that run's truncated step count to pass.

    Folds: ``StratifiedKFold.split`` reads only ``len(X)`` and ``y``, so identical rows get
    identical assignments whatever matrix is passed. That is scikit-learn's behaviour rather than
    something this code arranges, so it is verified on the real matrices, arm by arm and prefix by
    prefix, by exact array comparison against one reference set of splits.
    """
    from catchbench.live import _prefix_steps

    task.setup()
    prefixes = tuple(task.prefixes if prefixes is None else prefixes)
    labels = np.asarray(task.y)
    arms = live_size_audit_arms()
    layers = sorted({arm.layer for arm in arms})

    from grade.features import feature_vector
    from grade import build_graph

    names = feature_vector(build_graph(_prefix_steps(task.runs[0], 1.0), dependency="explicit",
                                       shared_resource=False), layer="flat")[0]
    step_column = names.index("n_steps")

    rows = {}
    for prefix in prefixes:
        view = prefix_view(task, prefix)
        expected = np.array([float(len(_prefix_steps(run, prefix))) for run in task.runs])
        observed = view.layers["flat"][:, step_column]
        if not np.array_equal(expected, observed):
            disagreeing = int(np.count_nonzero(expected != observed))
            raise AssertionError(
                f"{view.dataset}: {disagreeing} rows carry an n_steps that is not the truncated "
                "length of the run in that row, so the prefix matrices are not row-aligned"
            )
        for layer in layers:
            if view.layers[layer].shape[0] != len(labels):
                raise AssertionError(f"{view.dataset}/{layer}: row count is not the corpus size")
        rows[_prefix_key(prefix)] = {
            "n_rows": int(len(labels)),
            "min_prefix_steps": int(expected.min()),
            "max_prefix_steps": int(expected.max()),
        }

    reference = fold_splits(labels)
    compared = 0
    for prefix in prefixes:
        view = prefix_view(task, prefix)
        for layer in layers:
            matrix = view.layers[layer]
            for seed in SEEDS:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                for index, (train, test) in enumerate(cv.split(matrix, labels)):
                    want_train, want_test = reference[int(seed)][index]
                    if not (np.array_equal(train, want_train) and np.array_equal(test, want_test)):
                        raise AssertionError(
                            f"{view.dataset}/{layer} seed {seed} fold {index}: the split differs "
                            "from the shared reference split, so the arms are not scored on the "
                            "same folds"
                        )
                    compared += 1
    return {
        "establishes": "row i is the same run at every prefix, and all four arms at all four "
                       "prefixes are scored on one identical set of 25 splits",
        "splits_compared": compared,
        "n_seeds": len(list(SEEDS)),
        "per_prefix": rows,
        "fold_sizes": [[int(len(test)) for _, test in reference[int(seed)]] for seed in SEEDS],
    }


def check_prefix_set(task: LiveStreaming) -> dict:
    """The task sweeps the declared prefixes, and no others.

    A3 freezes ``(0.25, 0.50, 0.75, 1.00)``. A task constructed with a different sweep would score
    cells the declaration does not name and would leave the named ones uncomputed, so the identity
    is checked rather than assumed from the default argument.
    """
    task.setup()
    if tuple(task.prefixes) != tuple(AUDIT_PREFIXES):
        raise AssertionError(
            f"{task.dataset}: prefixes {tuple(task.prefixes)!r} are not the declared "
            f"{tuple(AUDIT_PREFIXES)!r}"
        )
    return {"prefixes": [float(prefix) for prefix in AUDIT_PREFIXES],
            "rule": "catchbench.live._prefix_steps: steps[: max(2, ceil(p * len(steps)))]"}
