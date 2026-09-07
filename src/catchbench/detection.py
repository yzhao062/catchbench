"""POST pillar (forensics): run-level failure prediction (detection).

Predict whether a run failed, scored by ROC-AUC, on trusted outcome labels. The board carries one
size reference, ``size (flat)``, which reads the four recorded size and count columns under the
board's linear model. A dependency increment measured against a linear size model is a different
quantity from the increment measured against a flexible one, and on SWE-Gym the two disagree in
sign. That question is answered in ``detection_audit`` rather than on the board: it measures the
increment under matched learners and writes the arms and their intervals instead of a verdict.
The flexible arms stay off the board so its entrant inventory does not move.

The board answers the detection question across four method families. A random floor and the two
size references bound the trivial end. Unsupervised anomaly detectors read the run without labels:
PyOD on the flat features (shallow
tabular AD), PyGOD's DOMINANT over the typed graph, and GUARDIAN's reconstruction autoencoder (the
agent-specific graph sibling). The supervised feature layers reuse GRADE's verified eval (typed-graph
layers, seed-averaged 5-fold stratified CV, standardized logistic regression, each run one row), and
G-Safeguard adds a supervised graph-classification GNN. The wider PyOD / PyGOD detector arena plugs in
through ``pyod_extra`` and ``pygod_extra`` via ``run.py``.

  - random               : a floor (random scores; ROC-AUC ~ 0.5).
  - size (flat)          : run size and counts only, under the board's linear model. The flexible
                           spline counterpart is an audit arm, not a board row (see ``_SPLINE``).
  - pyod-flatten (ECOD)  : unsupervised PyOD on the flat per-run features (shallow tabular AD).
  - pygod (graph AD)     : unsupervised DOMINANT over the typed graph (generic graph AD).
  - guardian (recon-AE)  : GUARDIAN's unsupervised reconstruction autoencoder (agent-specific graph AD).
  - auditable (size+deps): size plus the size-normalized dependency block (the dependency signal
                           auditable surfaces), read against BOTH size rows.
  - full                 : flat plus execution plus raw dependency features (reference).
  - g-safeguard (sup GNN): G-Safeguard's supervised graph-classification GNN over the dependency graph.

The spline arms run under a fixed configuration, so no knot count was chosen on a score. What a
flexible size model changes for the dependency reading is an empirical question the board does not
settle on its own: ``detection_audit`` scores the matched controls, including a spline size model
with the dependency block entering linearly and a gradient-boosted-tree pair on the same two
feature sets. Every one of those arms is an analysis arm and none is a board entrant;
``tools/emit_detection_audit.py`` writes them, their per-seed arrays and their out-of-fold
probabilities to JSON.

`auditable` is one entry on the board, not the referee. A method runs across every detection
dataset (the same Task contract over a different corpus), which is the dataset-as-the-asset point.

Datasets:
  - swegym : SWE-Gym OpenHands-sampled trajectories (balanced resolved / unresolved); GRADE's
             cited detection corpus (ROC-AUC ~0.805 for the dependency signal).
  - tau    : tau-bench retail / airline trajectories (MIT), labelled by db_match outcome.
Both load from HuggingFace through the GRADE loaders on first use.
"""
from __future__ import annotations

from typing import Mapping

from catchbench import _reuse  # noqa: F401  side effect: sets sys.path for grade + auditable

import numpy as np  # noqa: E402

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import get_scorer  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import SplineTransformer, StandardScaler  # noqa: E402

from agent_failure_detection import (  # noqa: E402  GRADE's verified detection eval
    SEEDS,
    _cv,
    _gain,
    _layer_matrices,
    load_swegym,
    load_tau,
)

_LOADERS = {"swegym": load_swegym, "tau": load_tau}


class PostDetection:
    """POST / detection Task: run-level failure prediction over a corpus with outcome labels."""

    task_id = "post_detection"
    pillar = "POST"
    granularity = "run"

    def __init__(self, corpus: str = "swegym") -> None:
        if corpus not in _LOADERS:
            raise ValueError(f"unsupported detection corpus: {corpus!r}")
        self.corpus = corpus
        self.dataset = corpus
        self._loaded = False

    def setup(self) -> None:
        if self._loaded:
            return
        graphs, labels = _LOADERS[self.corpus]()  # downloads from HuggingFace on first call
        self.graphs = graphs  # typed networkx graphs, for the graph-AD baseline
        self.y = np.array(labels)
        self.layers = _layer_matrices(graphs)  # {layer_name: feature matrix}
        self._loaded = True

    def corpus_line(self) -> str:
        self.setup()
        n, n_fail = len(self.y), int(self.y.sum())
        return (f"{self.dataset}: {n} runs ({n_fail} failed, {n - n_fail} resolved), "
                f"run-level outcome labels.")


class _LayerAUC:
    """A detection method scored as the seed-averaged 5-fold stratified-CV ROC-AUC of one layer."""

    def __init__(self, method_id: str, layer: str) -> None:
        self.method_id = method_id
        self.layer = layer
        self.supports = {"post_detection"}

    def evaluate(self, task: PostDetection) -> Mapping[str, float]:
        task.setup()
        per_seed_fold = _cv(task.layers[self.layer], task.y)  # (n_seeds, n_folds) ROC-AUC
        return {"roc_auc": float(per_seed_fold.mean())}


# --- the declared flexible size reference, and the controlled-audit estimators ----------------
#
# ``_SPLINE`` is the configuration declared before any of these arms was fitted: cubic B-splines,
# five quantile knots per input column, constant extrapolation past the outer knots, and no bias
# column. It is a fixed estimator, so nothing here selects the knot count on a score. The nested
# arms below estimate what a knot-selecting PROCEDURE would score instead, and they are reported
# beside the fixed one rather than in place of it.
_SPLINE = {
    "degree": 3,
    "n_knots": 5,
    "knots": "quantile",
    "extrapolation": "constant",
    "include_bias": False,
}

# Knot grids for the two nested-selection arms, and the offsets their inner splitters carry.
#
# The offsets guarantee nothing about leakage, and it would be wrong to read them as the reason
# these arms are honest. ``GridSearchCV`` partitions only the rows it is handed, which are the
# outer training rows, so the outer test fold is absent from every inner split whatever random
# state the inner splitter carries. What keeps the outer test fold out is ``_cv_estimator_detail``
# handing ``fit`` the training rows alone, and ``_FoldLocality`` in
# ``tools/emit_detection_audit.py`` verifies that directly on every fitted object.
#
# The offsets are hygiene. They keep the inner and outer random states textually distinct in a
# parameter dump, so a reader tracing a selected knot count back to a seed cannot mistake one
# splitter for the other.
_KNOT_GRID = {"splinetransformer__n_knots": [3, 4, 5, 6, 8, 10, 15]}
_KNOT_C_GRID = {
    "splinetransformer__n_knots": [3, 5, 8],
    "logisticregression__C": [0.03, 0.1, 0.3, 1.0, 3.0],
}
_INNER_KNOT_OFFSET = 1000
_INNER_KNOT_C_OFFSET = 2000


def _logistic():
    """The board's own classifier, unchanged: GRADE's ``_cv`` builds exactly this one."""
    return LogisticRegression(solver="liblinear", class_weight="balanced")


def _linear_estimator(seed):
    """The board's linear feature-layer model, rebuilt here so the audit can rerun it."""
    return make_pipeline(StandardScaler(), _logistic())


def _spline_estimator(seed):
    """``size (spline)``: an additive cubic spline in the recorded columns, then the board's model.

    The spline sits in front of the scaler, so the logistic regression still sees standardized
    inputs, which is what ``_cv`` guarantees for the linear rows.
    """
    return make_pipeline(SplineTransformer(**_SPLINE), StandardScaler(), _logistic())


def _gbt_estimator(seed):
    """GRADE's own gradient-boosted-tree configuration (``_m_gbt`` in size_baseline_nonlinear).

    Restated here rather than imported, because that experiment script is not part of GRADE's
    committed tree and is absent from a clean checkout. ``tools/emit_detection_audit.py`` compares
    these parameters against ``_m_gbt`` whenever that module can be imported, so the copy cannot
    drift silently away from its source.
    """
    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        class_weight="balanced",
        random_state=seed,
    )


def _nested_knot_estimator(seed):
    """The size spline with its knot count selected inside the training folds, never on the board.

    ``GridSearchCV`` splits the training rows it is handed and never sees the outer test fold, so
    the selected knot count is a training-fold quantity like the knots themselves.
    """
    return GridSearchCV(
        make_pipeline(SplineTransformer(**_SPLINE), StandardScaler(), _logistic()),
        _KNOT_GRID,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=_INNER_KNOT_OFFSET + seed),
        n_jobs=1,
        error_score="raise",
    )


def _nested_knot_c_estimator(seed):
    """The same, selecting the knot count and the regularization strength together."""
    return GridSearchCV(
        make_pipeline(SplineTransformer(**_SPLINE), StandardScaler(), _logistic()),
        _KNOT_C_GRID,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=_INNER_KNOT_C_OFFSET + seed),
        n_jobs=1,
        error_score="raise",
    )


def _cv_estimator(factory, X, y):
    """GRADE's ``_cv`` fold construction with the estimator swapped, and nothing else swapped.

    Same ``SEEDS``, same ``StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)``, same
    ``roc_auc`` scoring, same ``cross_val_score`` call, so every arm here is scored on the folds
    the board's linear rows were scored on. Passing ``_linear_estimator`` reproduces ``_cv``.

    ``factory`` returns an UNFITTED estimator. ``cross_val_score`` clones it and calls
    ``fit(X[train], y[train])``, and a ``Pipeline`` fits each step on what the step before it
    produced FROM THE TRAINING ROWS. A spline's quantile knots and a scaler's moments are
    therefore training-fold quantities, and the held-out rows only ever reach ``transform``. No
    arm can be handed a transformer that has already seen the fold it is about to be scored on,
    because no arm is ever handed a fitted transformer at all.
    """
    rows = []
    for seed in SEEDS:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        rows.append(cross_val_score(factory(seed), X, y, cv=cv, scoring="roc_auc",
                                    error_score="raise"))
    return np.array(rows)  # shape (n_seeds, n_folds)


def _cv_estimator_detail(factory, X, y, inspect=None):
    """``_cv_estimator``'s folds, keeping what a fold score alone throws away.

    Returns the per-(seed, fold) ROC-AUC, the out-of-fold probability of failure per seed, and any
    hyperparameter a nested arm selected. The fold loop is written out here rather than delegated,
    because ``cross_val_score`` returns scores and discards the fitted estimators; the scorer is
    the same object ``scoring="roc_auc"`` resolves to, so the scores it produces are the same
    numbers. Out-of-fold probabilities are what the paper's paired run-level convention consumes,
    and split-seed intervals are recoverable from the fold scores, so a reader of the artifact can
    compute either without refitting.

    ``inspect``, when given, is called as ``inspect(seed, fold, train, fitted)`` after each fit and
    before it is scored. It is how a verification reaches all 25 fitted estimators without refitting
    any of them, which matters for the nested arms: refitting one of those to look inside it costs
    as much again as scoring it. An inspector reads the fitted object and returns nothing; it is
    handed no test rows, and nothing it does can reach ``fold_auc`` or ``oof``.
    """
    seeds = list(SEEDS)
    scorer = get_scorer("roc_auc")
    fold_auc = np.zeros((len(seeds), 5))
    oof = np.zeros((len(seeds), len(y)))
    selected = []
    for row, seed in enumerate(seeds):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(cv.split(X, y)):
            fitted = factory(seed).fit(X[train], y[train])
            if inspect is not None:
                inspect(int(seed), int(fold), train, fitted)
            fold_auc[row, fold] = scorer(fitted, X[test], y[test])
            oof[row, test] = fitted.predict_proba(X[test])[:, 1]
            chosen = getattr(fitted, "best_params_", None)
            if chosen is not None:
                selected.append({"seed": int(seed), "fold": int(fold),
                                 **{str(k): v for k, v in chosen.items()}})
    return fold_auc, oof, selected


class _EstimatorAUC:
    """``_LayerAUC`` with the estimator named as well as the layer.

    Same contract, same folds; the only difference is that the model is a declared factory rather
    than the board's built-in linear one. Used by every controlled-audit arm. No arm built this way
    is a board entrant.
    """

    def __init__(self, method_id: str, layer: str, factory) -> None:
        self.method_id = method_id
        self.layer = layer
        self.factory = factory
        self.supports = {"post_detection"}

    def estimator_factory(self, task: "PostDetection"):
        """The unfitted-estimator factory this arm scores. ``task`` is unused here; the mixed arm
        needs it, and the audit reads every arm through this one method."""
        return self.factory

    def evaluate(self, task: "PostDetection") -> Mapping[str, float]:
        task.setup()
        per_seed_fold = _cv_estimator(self.estimator_factory(task), task.layers[self.layer], task.y)
        return {"roc_auc": float(per_seed_fold.mean())}


class _MixedSplineAUC:
    """``size-spline + linear-deps``: the declared spline over the size columns, with the
    dependency block entering linearly exactly as it enters ``auditable (size+deps)``.

    This is the arm the "beyond run size" sentence actually implies once the size reference is
    allowed to bend: the dependency features are unchanged and enter the model the way they always
    did, and only the size term becomes flexible. A controlled-audit arm, not a board entrant.
    """

    def __init__(self, method_id: str = "size-spline + linear-deps") -> None:
        self.method_id = method_id
        self.layer = "flatdep"
        self.supports = {"post_detection"}

    def columns(self, task: "PostDetection") -> tuple[list[int], list[int]]:
        """The size columns and the dependency columns of the ``flatdep`` matrix.

        Read from the layer widths rather than written down, and checked against the ``flat``
        matrix, so a change to GRADE's feature order fails here instead of quietly splining the
        wrong block.
        """
        task.setup()
        size = task.layers["flat"].shape[1]
        matrix = task.layers[self.layer]
        if not np.array_equal(matrix[:, :size], task.layers["flat"]):
            raise ValueError(
                f"the flat block is no longer the leading {size} columns of {self.layer!r}; "
                "the mixed audit arm cannot tell the size columns from the dependency columns"
            )
        return list(range(size)), list(range(size, matrix.shape[1]))

    def estimator_factory(self, task: "PostDetection"):
        size, deps = self.columns(task)

        def factory(seed):
            return make_pipeline(
                ColumnTransformer([("size", SplineTransformer(**_SPLINE), size),
                                   ("deps", "passthrough", deps)]),
                StandardScaler(),
                _logistic(),
            )

        return factory

    def evaluate(self, task: "PostDetection") -> Mapping[str, float]:
        task.setup()
        per_seed_fold = _cv_estimator(self.estimator_factory(task), task.layers[self.layer], task.y)
        return {"roc_auc": float(per_seed_fold.mean())}


class PyODFlatten:
    """Shallow tabular baseline: unsupervised PyOD (ECOD) on the flat per-run features."""

    method_id = "pyod-flatten (ECOD)"
    supports = {"post_detection"}

    def evaluate(self, task: PostDetection) -> Mapping[str, float]:
        from pyod.models.ecod import ECOD
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        task.setup()
        features = StandardScaler().fit_transform(task.layers["flat"])
        detector = ECOD()
        detector.fit(features)
        # Unsupervised: higher decision score = more anomalous; score against failed = 1.
        return {"roc_auc": float(roc_auc_score(task.y, detector.decision_scores_))}


class PyGODDetection:
    """PyGOD graph-AD as a run-level detector: a DOMINANT autoencoder scores each node's anomaly
    over the run's typed graph, and the run's mean node anomaly is its failure score. Unsupervised
    and structure-aware, the graph sibling of the flat PyOD baseline. Deterministic given the seed."""

    method_id = "pygod (graph AD)"
    supports = {"post_detection"}

    def evaluate(self, task: PostDetection) -> Mapping[str, float]:
        from sklearn.metrics import roc_auc_score

        from catchbench.graph_ad import nx_to_graph, pygod_node_scores

        task.setup()
        graphs = [nx_to_graph(graph) for graph in task.graphs]
        per_run = pygod_node_scores(graphs)
        run_scores = np.array([scores.mean() for scores in per_run])
        return {"roc_auc": float(roc_auc_score(task.y, run_scores))}


class GuardianDetection:
    """GUARDIAN (Zhou et al., 2025): an UNSUPERVISED reconstruction autoencoder over the typed graph,
    the agent-specific sibling of PyGOD's generic DOMINANT. A run's failure score is the mean per-node
    reconstruction error. See ``agent_detectors`` for what is implemented versus simplified."""

    method_id = "guardian (recon-AE)"
    supports = {"post_detection"}

    def evaluate(self, task: PostDetection) -> Mapping[str, float]:
        from sklearn.metrics import roc_auc_score

        from catchbench.agent_detectors import guardian_run_scores
        from catchbench.graph_ad import nx_to_graph

        task.setup()
        scores = guardian_run_scores([nx_to_graph(graph) for graph in task.graphs])
        return {"roc_auc": float(roc_auc_score(task.y, scores))}


class GSafeguardDetection:
    """G-Safeguard (Wang et al., 2025): a SUPERVISED graph-classification GNN, trained with
    seed-averaged stratified cross-validation to predict run failure from the dependency graph. The
    benchmark's first supervised GNN, so it tests whether learning over the graph beats the
    feature-layer logistic models."""

    method_id = "g-safeguard (sup GNN)"
    supports = {"post_detection"}

    def evaluate(self, task: PostDetection) -> Mapping[str, float]:
        from catchbench.agent_detectors import gsafeguard_cv_auc
        from catchbench.graph_ad import nx_to_graph

        task.setup()
        return {"roc_auc": gsafeguard_cv_auc([nx_to_graph(graph) for graph in task.graphs], task.y)}


class RandomDetection:
    """The ROC-AUC floor: random run scores (averages to ~0.5)."""

    method_id = "random"
    supports = {"post_detection"}

    def evaluate(self, task: PostDetection) -> Mapping[str, float]:
        from sklearn.metrics import roc_auc_score

        task.setup()
        rng = np.random.RandomState(0)
        aucs = [roc_auc_score(task.y, rng.rand(len(task.y))) for _ in range(5)]
        return {"roc_auc": float(np.mean(aucs))}


def post_detection_methods() -> list:
    """The detection board's baseline set, in display order: a floor and the linear size reference;
    the unsupervised anomaly detectors (PyOD tabular, PyGOD graph, GUARDIAN recon-AE); the
    supervised feature layers and the G-Safeguard GNN.

    The flexible size model that interprets this block is deliberately **not** here. It is an audit
    arm in the controlled audit below, because registering it would add a row to both detection
    blocks and move the entrant inventory from 72 to 73, and a control belongs beside the rows it
    interprets rather than competing with them. ``tools/emit_detection_audit.py`` scores it.
    """
    return [
        RandomDetection(),
        _LayerAUC("size (flat)", "flat"),
        PyODFlatten(),
        PyGODDetection(),
        GuardianDetection(),
        _LayerAUC("auditable (size+deps)", "flatdep"),
        _LayerAUC("full", "full"),
        GSafeguardDetection(),
    ]


# --- the controlled audit: matched arms that interpret the board, and stay off it ---------------
#
# These read the two size references against each other under matched conditions. They are not
# board entrants: promoting them would spend board and inventory space on controls whose purpose
# is clearer in one table, and the point of a control is what it says about the rows already
# there. ``tools/emit_detection_audit.py`` scores them and writes the artifact.
#
# The three linear arms are the board's own rows, rebuilt through the swapped-estimator path. They
# earn their place twice: they are the references every contrast below is measured against, and
# reproducing the board's published values through this path is the check that the path did not
# change the folds.


def post_detection_audit_arms() -> list:
    """Every arm of the controlled audit, board rows included, in table order."""
    return [
        _EstimatorAUC("size (flat)", "flat", _linear_estimator),
        _EstimatorAUC("auditable (size+deps)", "flatdep", _linear_estimator),
        _EstimatorAUC("full", "full", _linear_estimator),
        _EstimatorAUC("size (spline)", "flat", _spline_estimator),
        _MixedSplineAUC(),
        _EstimatorAUC("size+deps (spline)", "flatdep", _spline_estimator),
        _EstimatorAUC("size (GBT)", "flat", _gbt_estimator),
        _EstimatorAUC("size+deps (GBT)", "flatdep", _gbt_estimator),
        _EstimatorAUC("size (spline, nested knots)", "flat", _nested_knot_estimator),
        _EstimatorAUC("size+deps (spline, nested knots)", "flatdep", _nested_knot_estimator),
        _EstimatorAUC("size (spline, nested knots+C)", "flat", _nested_knot_c_estimator),
        _EstimatorAUC("size+deps (spline, nested knots+C)", "flatdep", _nested_knot_c_estimator),
    ]


# (higher arm, lower arm), in two groups, because two of the eight are not matched pairs and the
# artifact has to say so rather than let a reader infer it from the name.
#
# Matched: the two arms differ in exactly one respect, named beside each pair. A difference between
# them is attributable to that one respect.
DETECTION_AUDIT_CONTRASTS_MATCHED = (
    ("size (spline)", "size (flat)"),                # the model over the same four columns
    ("size-spline + linear-deps", "size (spline)"),  # the dependency block, entering linearly
    ("size+deps (spline)", "size (spline)"),         # the dependency block, entering splined
    ("size+deps (GBT)", "size (GBT)"),               # the dependency block, under the trees
    ("size+deps (spline, nested knots)", "size (spline, nested knots)"),
    ("size+deps (spline, nested knots+C)", "size (spline, nested knots+C)"),
)

# Confounded: TWO things differ at once. In both pairs the high arm is the board's published linear
# dependency row and the low arm is a spline size model, so going from the high arm to the low one
# moves the feature layer from ``flatdep`` to ``flat`` AND the size model from linear to a spline.
# A difference between them cannot be attributed to either change alone.
#
# They stay here, labelled, because they are the comparison the manuscript's "beyond run size"
# sentence is actually about: the published dependency row read against a flexible size reference.
# Dropping them would remove the number the prose is making a claim about.
#
# The first has an exact matched reading among the arms, ``size-spline + linear-deps`` against
# ``size (spline)``, which holds the dependency block and its linear entry fixed and moves only the
# size model. The second does not. Its exact counterpart would be a knot-selecting size spline with
# the dependency block still entering linearly, and no arm is that; the nearest one holds the
# knot-selecting learner fixed on both sides but splines the dependency block as well.
DETECTION_AUDIT_CONTRASTS_CONFOUNDED = (
    ("auditable (size+deps)", "size (spline)"),
    ("auditable (size+deps)", "size (spline, nested knots)"),
)

DETECTION_AUDIT_CONTRASTS = (
    DETECTION_AUDIT_CONTRASTS_MATCHED + DETECTION_AUDIT_CONTRASTS_CONFOUNDED
)

# What each confounded pair confounds, recorded per contrast in the artifact so the label travels
# with the number instead of living only in this comment.
_CONFOUNDS = {
    ("auditable (size+deps)", "size (spline)"):
        "two changes at once: the feature layer moves from flatdep to flat and the size model "
        "moves from linear to the declared fixed spline. The matched reading of the same question "
        "is 'size-spline + linear-deps - size (spline)'.",
    ("auditable (size+deps)", "size (spline, nested knots)"):
        "two changes at once: the feature layer moves from flatdep to flat and the size model "
        "moves from linear to a knot-selecting spline. No arm is the exact matched reading, which "
        "would be a knot-selecting size spline with the dependency block still entering linearly. "
        "The nearest is 'size+deps (spline, nested knots) - size (spline, nested knots)', which "
        "holds the knot-selecting learner fixed on both sides but splines the dependency block as "
        "well instead of passing it through.",
}

# What the numbers in the artifact are, said once, in the artifact, so a later reader cannot
# mistake one for the other. GRADE's ``_gain`` interval is a t interval over the five
# cross-validation SPLIT SEEDS. It describes how the estimate moves when the folds are redrawn on
# the same runs. It is not an interval over runs, and it does not answer whether a difference
# would survive on other runs. The paper's paired run-level test consumes ``oof_proba`` instead,
# which is why every arm carries it.
_SEED_INTERVAL_NOTE = (
    "t interval over the 5 cross-validation split seeds (GRADE _gain, half-width "
    "2.776 * sd(ddof=1) / sqrt(5)); describes fold-redraw variability on these runs only, and is "
    "not run-level inference. Recompute run-level inference from oof_proba and labels."
)


def _rank(matrix: np.ndarray) -> int:
    """Rank of the column-centered matrix: the directions a model with an intercept can use."""
    return int(np.linalg.matrix_rank(matrix - matrix.mean(axis=0)))


def _json_params(estimator) -> dict:
    """``get_params(deep=True)`` with anything unserializable rendered as its repr."""
    params = {}
    for name, value in sorted(estimator.get_params(deep=True).items()):
        if isinstance(value, (bool, int, float, str, type(None))):
            params[name] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(item, (bool, int, float, str, type(None))) for item in value
        ):
            params[name] = list(value)
        else:
            params[name] = repr(value)
    return params


def _estimator_specification(factory) -> dict:
    """Enough of an arm's estimator to rebuild EVERY seed's fit from the artifact alone.

    An arm is scored five times, once per split seed, and ``factory(seed)`` is called afresh each
    time. Recording ``factory(0)`` alone therefore under-specifies any arm whose estimator depends
    on the seed, and six of the twelve arms do:

      - ``size (GBT)`` and ``size+deps (GBT)`` move ``random_state``, which sets the tree learner's
        own randomness;
      - the four nested-selection arms move ``cv``, whose ``StratifiedKFold`` carries
        ``random_state = _INNER_KNOT_OFFSET + seed`` or ``_INNER_KNOT_C_OFFSET + seed``, which sets
        the inner split the knot count is selected on.

    The other six are fixed: ``factory`` ignores its argument and all five seeds build an identical
    estimator, so the only thing the seed changes for them is the outer fold split.

    Returned: ``repr`` and the full deep parameters at seed 0, plus ``varies_with_seed``, which
    holds every parameter whose value is not the same for all five seeds, with that parameter's
    value under each seed. ``varies_with_seed`` is empty for a fixed arm, and that emptiness is the
    positive statement that the arm does not move with the seed.
    """
    seeds = [int(seed) for seed in SEEDS]
    by_seed = {seed: _json_params(factory(seed)) for seed in seeds}
    base = by_seed[seeds[0]]
    varies: dict[str, dict] = {}
    for name in sorted({key for params in by_seed.values() for key in params}):
        values = {str(seed): by_seed[seed].get(name) for seed in seeds}
        if len({repr(value) for value in values.values()}) > 1:
            varies[name] = values
    return {
        "estimator": repr(factory(seeds[0])),
        "seeds": seeds,
        "params_at_seed_0": base,
        "varies_with_seed": varies,
    }


def _tau_run_identities(task: PostDetection) -> dict:
    """Row identifiers for the tau-bench corpus, replayed from the loader and checked to align.

    ``load_tau`` reads records, keeps the ones passing three filters, and returns only graphs and
    labels, so the identifiers are dropped rather than absent. Replaying the same iteration under
    the same three filters recovers them in the same order.

    The alignment is verified, not assumed: the replayed labels must equal ``task.y`` and the
    replayed step counts must equal the ``n_steps`` column of the flat matrix, both exactly and
    row for row. A mismatch raises, because a silently misaligned identifier is worse than none.
    """
    import agent_graph_tau_bench as tau
    from grade import feature_vector

    records = []
    for record in tau.load_runs(tau._ensure_files()):
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            continue
        outcome = record.get("eval_result") or {}
        if "db_match" not in outcome:
            continue
        steps = tau.to_steps(messages, record.get("model_path", "model"))
        if len(steps) < 2:
            continue
        records.append((record, len(steps)))

    labels = np.array([0 if bool((record.get("eval_result") or {})["db_match"]) else 1
                       for record, _ in records])
    if not np.array_equal(labels, np.asarray(task.y)):
        raise AssertionError(
            f"{task.dataset}: the replayed tau-bench labels no longer match the scored labels "
            f"({len(labels)} replayed against {len(task.y)} scored); the identifiers cannot be "
            "trusted to line up with the feature matrices"
        )
    names = feature_vector(task.graphs[0], layer="flat")[0]
    steps_column = task.layers["flat"][:, names.index("n_steps")]
    replayed_steps = np.array([float(count) for _, count in records])
    if not np.array_equal(replayed_steps, steps_column):
        raise AssertionError(
            f"{task.dataset}: the replayed tau-bench step counts do not match the n_steps column, "
            "so the replay is not in the order the feature matrices are in"
        )

    models = [record["model_path"] for record, _ in records]
    clusters = [(record.get("meta") or {})["id"] for record, _ in records]
    domains = [record["task_name"] for record, _ in records]
    keys = [f"{model}|{cluster}" for model, cluster in zip(models, clusters)]
    if len(set(keys)) != len(keys):
        raise AssertionError(f"{task.dataset}: the run key is not unique across the scored rows")
    per_cluster: dict[str, int] = {}
    for cluster in clusters:
        per_cluster[cluster] = per_cluster.get(cluster, 0) + 1
    histogram: dict[str, int] = {}
    for count in per_cluster.values():
        histogram[str(count)] = histogram.get(str(count), 0) + 1
    return {
        "available": True,
        "source": "agent_graph_tau_bench.load_runs(_ensure_files()), replayed under the three "
                  "filters agent_failure_detection.load_tau applies: len(messages) >= 3, "
                  "'db_match' in eval_result, len(steps) >= 2",
        "alignment_verified": ["labels equal task.y row for row",
                               "replayed step counts equal the n_steps column row for row",
                               "the run key is unique across rows"],
        "run_key": keys,
        "run_key_fields": ["model_path", "meta.id"],
        "model": models,
        "task_cluster": clusters,
        "domain": domains,
        "n_distinct_task_clusters": len(per_cluster),
        "runs_per_task_cluster": histogram,
        "note": "the rows are not independent draws. Each tau-bench task instance is run by every "
                "agent model, so the corpus is a task-by-model grid and a run-level interval "
                "treats repeated attempts at one task as separate observations. A task-clustered "
                "interval answers a different question and needs task_cluster to compute.",
    }


def _swegym_run_identities(task: PostDetection) -> dict:
    """Why SWE-Gym rows carry no identifier, checked against the loader source rather than asserted.

    ``agent_graph_swegym.load_runs`` reads three columns out of the parquet, ``messages``,
    ``resolved`` and ``run_id``, and the SWE-bench ``instance_id`` column is not among them, so no
    instance identifier is ever read. ``run_id`` is read but is used only to pick and balance one
    scaffold; the records the loader returns hold ``resolved`` and ``steps`` and nothing else, so it
    stops there too. Nothing identifying reaches ``load_swegym``, the graphs, or the feature
    matrices.

    That is a statement about this boundary, not about the corpus. ``instance_id`` is in the parquet
    and two routes reach it. One is a GRADE-side change that carries it through ``load_runs`` into
    the record, after which this function would record it the way it records the tau-bench
    identifiers. The other is an out-of-tree replay that adds ``instance_id`` to the requested
    columns and otherwise reproduces the loader's balancing and single-scaffold selection exactly,
    which recovers the identifiers without touching GRADE but re-implements the selection rather
    than reading it, so its alignment has to be re-established against the shipped matrices every
    time the loader moves. Neither route is taken here, and the tau-bench branch is the only one
    whose identifiers this artifact carries.

    The source is hashed and scanned so this record cannot go stale silently: if the loader later
    starts selecting an identifier column, ``selects_an_instance_identifier`` turns true and the
    hash moves, which is the signal to revisit this and record real identifiers.
    """
    from hashlib import sha256
    from inspect import getsource

    import agent_graph_swegym as swegym

    source = getsource(swegym.load_runs)
    return {
        "available": False,
        "reason": "agent_graph_swegym.load_runs selects only [messages, resolved, run_id] from the "
                  "parquet, so the SWE-bench instance_id column is never read, and the records it "
                  "returns carry only 'resolved' and 'steps', so run_id stops there as well. No "
                  "run or instance identifier reaches the feature matrices.",
        "loader": "agent_graph_swegym.load_runs",
        "loader_source_sha256": sha256(source.encode("utf-8")).hexdigest(),
        "selects_an_instance_identifier": "instance_id" in source,
        "what_would_change_this": [
            "a GRADE-side change that carries an identifier through load_runs into the record, "
            "after which this audit records it the way it records the tau-bench identifiers",
            "an out-of-tree replay that adds instance_id to the requested parquet columns and "
            "reproduces the balancing and single-scaffold selection exactly; that recovers the "
            "identifiers without touching GRADE, but it re-implements the selection rather than "
            "reading it, so the row alignment needs re-establishing whenever the loader moves",
        ],
    }


def run_identities(task: PostDetection) -> dict:
    """Per-row identifiers for one detection corpus, where the loader leaves any behind.

    Recoverable on tau-bench and not on SWE-Gym, and the difference is a property of the two GRADE
    loaders rather than of this boundary in general. Both branches return a record saying which
    case applies and how it was established.
    """
    task.setup()
    if task.corpus == "tau":
        return _tau_run_identities(task)
    if task.corpus == "swegym":
        return _swegym_run_identities(task)
    return {"available": False,
            "reason": f"no identifier replay is implemented for corpus {task.corpus!r}"}


def detection_audit(task: PostDetection, arms: list | None = None, inspect=None) -> dict:
    """Score every controlled-audit arm on one detection corpus and return the whole record.

    Per arm: the estimator specification for every seed, the per-(seed, fold) ROC-AUC, the per-seed
    means, the out-of-fold probability of failure per seed, and whatever a nested arm selected. Per
    contrast: the split-seed mean difference with GRADE's own interval, the per-seed differences,
    and whether the pair is matched or confounded.

    ``inspect``, when given, is called as ``inspect(arm, seed, fold, train, fitted)`` after every
    fit, so a verification can read all 25 fitted estimators of every arm without refitting one.
    """
    task.setup()
    arms = list(post_detection_audit_arms() if arms is None else arms)
    labels = np.asarray(task.y)

    blocks = {}
    for layer, matrix in sorted(task.layers.items()):
        blocks[layer] = {
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "centered_rank": _rank(matrix),
            "constant_columns": [int(i) for i in range(matrix.shape[1])
                                 if float(np.ptp(matrix[:, i])) == 0.0],
        }

    board_names = {method.method_id for method in post_detection_methods()}
    scored: dict[str, dict] = {}
    for arm in arms:
        factory = arm.estimator_factory(task)
        watcher = None
        if inspect is not None:
            def watcher(seed, fold, train, fitted, arm=arm):
                inspect(arm, seed, fold, train, fitted)
        fold_auc, oof, selected = _cv_estimator_detail(
            factory, task.layers[arm.layer], labels, inspect=watcher)
        specification = _estimator_specification(factory)
        scored[arm.method_id] = {
            "board_entrant": arm.method_id in board_names,
            "layer": arm.layer,
            "estimator": specification["estimator"],
            "params": specification["params_at_seed_0"],
            "params_vary_with_seed": specification["varies_with_seed"],
            "mean_roc_auc": float(fold_auc.mean()),
            "seed_mean_roc_auc": [float(value) for value in fold_auc.mean(axis=1)],
            "fold_roc_auc": [[float(value) for value in row] for row in fold_auc],
            "selected_hyperparameters": selected,
            "oof_proba": [[float(value) for value in row] for row in oof],
        }

    contrasts = {}
    for high, low in DETECTION_AUDIT_CONTRASTS:
        if high not in scored or low not in scored:
            continue
        hi = np.array(scored[high]["fold_roc_auc"])
        lo = np.array(scored[low]["fold_roc_auc"])
        mean, half, positive, count = _gain(hi, lo)
        per_seed = hi.mean(axis=1) - lo.mean(axis=1)
        contrasts[f"{high} - {low}"] = {
            "high": high,
            "low": low,
            "matched": (high, low) in DETECTION_AUDIT_CONTRASTS_MATCHED,
            "confound": _CONFOUNDS.get((high, low)),
            "seed_mean_difference": float(mean),
            "seed_interval_95": [float(mean - half), float(mean + half)],
            "seeds_positive": int(positive),
            "seeds": int(count),
            "per_seed_difference": [float(value) for value in per_seed],
            "inference": _SEED_INTERVAL_NOTE,
        }

    return {
        "corpus": task.dataset,
        "task_id": task.task_id,
        "n_runs": int(len(labels)),
        "n_failed": int(labels.sum()),
        "labels": [int(value) for value in labels],
        "run_identities": run_identities(task),
        "seeds": [int(seed) for seed in SEEDS],
        "feature_blocks": blocks,
        "arms": scored,
        "contrasts": contrasts,
    }
