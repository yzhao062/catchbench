"""POST pillar (forensics): run-level failure prediction (detection).

Predict whether a run failed, scored by ROC-AUC, on trusted outcome labels. Reuses GRADE's
verified detection eval: typed-graph feature layers plus seed-averaged 5-fold stratified
cross-validation over runs (standardized logistic regression; each run is one row). The supervised "methods" on this board are feature layers,
which is GRADE's keystone question posed as a leaderboard: does dependency structure predict
failure beyond run size? PyOD (the shallow tabular path) and a random floor round out the board.

  - random              : a floor (random scores; ROC-AUC ~ 0.5).
  - size (flat)          : run size and counts only, the honest trivial baseline.
  - pyod-flatten (ECOD)  : unsupervised PyOD on the flat per-run features (shallow tabular AD).
  - auditable (structure): size plus the size-normalized dependency block (the structural signal
                          auditable surfaces). Beating "size (flat)" is the benchmark's point.
  - full                 : flat plus execution plus raw dependency features (reference).

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

from auditablebench import _reuse  # noqa: F401  side effect: sets sys.path for grade + auditable

import numpy as np  # noqa: E402

from agent_failure_detection import (  # noqa: E402  GRADE's verified detection eval
    _cv,
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

        from auditablebench.graph_ad import nx_to_graph, pygod_node_scores

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

        from auditablebench.agent_detectors import guardian_run_scores
        from auditablebench.graph_ad import nx_to_graph

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
        from auditablebench.agent_detectors import gsafeguard_cv_auc
        from auditablebench.graph_ad import nx_to_graph

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
    """The detection board's baseline set, in display order: a floor and a size baseline; the
    unsupervised anomaly detectors (PyOD tabular, PyGOD graph, GUARDIAN recon-AE); the supervised
    feature layers and the G-Safeguard GNN."""
    return [
        RandomDetection(),
        _LayerAUC("size (flat)", "flat"),
        PyODFlatten(),
        PyGODDetection(),
        GuardianDetection(),
        _LayerAUC("auditable (structure)", "flatdep"),
        _LayerAUC("full", "full"),
        GSafeguardDetection(),
    ]
