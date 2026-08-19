"""Extra unsupervised PyOD tabular baselines for the POST detection board.

Five additional shallow tabular anomaly detectors (IForest, KNN, LOF, COPOD, HBOS) following
the same contract as ``PyODFlatten`` in ``detection.py``: fit on the flat per-run feature matrix
(StandardScaler-normalized), score each run, and report ROC-AUC against the outcome label
(failure = 1, higher decision score = more anomalous).

Use ``pyod_extra_methods()`` to get the full list of instances in display order.
"""
from __future__ import annotations

from typing import Mapping


class IForestDetection:
    """Isolation Forest (deterministic, random_state=0) on the flat per-run features."""

    method_id = "pyod-iforest"
    supports = {"post_detection"}

    def evaluate(self, task) -> Mapping[str, float]:
        from pyod.models.iforest import IForest
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        task.setup()
        features = StandardScaler().fit_transform(task.layers["flat"])
        detector = IForest(random_state=0)
        detector.fit(features)
        return {"roc_auc": float(roc_auc_score(task.y, detector.decision_scores_))}


class KNNDetection:
    """k-Nearest Neighbours (PyOD) on the flat per-run features."""

    method_id = "pyod-knn"
    supports = {"post_detection"}

    def evaluate(self, task) -> Mapping[str, float]:
        from pyod.models.knn import KNN
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        task.setup()
        features = StandardScaler().fit_transform(task.layers["flat"])
        detector = KNN()
        detector.fit(features)
        return {"roc_auc": float(roc_auc_score(task.y, detector.decision_scores_))}


class LOFDetection:
    """Local Outlier Factor (PyOD) on the flat per-run features."""

    method_id = "pyod-lof"
    supports = {"post_detection"}

    def evaluate(self, task) -> Mapping[str, float]:
        from pyod.models.lof import LOF
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        task.setup()
        features = StandardScaler().fit_transform(task.layers["flat"])
        detector = LOF()
        detector.fit(features)
        return {"roc_auc": float(roc_auc_score(task.y, detector.decision_scores_))}


class COPODDetection:
    """COPOD (Copula-Based Outlier Detection, PyOD) on the flat per-run features."""

    method_id = "pyod-copod"
    supports = {"post_detection"}

    def evaluate(self, task) -> Mapping[str, float]:
        from pyod.models.copod import COPOD
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        task.setup()
        features = StandardScaler().fit_transform(task.layers["flat"])
        detector = COPOD()
        detector.fit(features)
        return {"roc_auc": float(roc_auc_score(task.y, detector.decision_scores_))}


class HBOSDetection:
    """HBOS (Histogram-Based Outlier Score, PyOD) on the flat per-run features."""

    method_id = "pyod-hbos"
    supports = {"post_detection"}

    def evaluate(self, task) -> Mapping[str, float]:
        from pyod.models.hbos import HBOS
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        task.setup()
        features = StandardScaler().fit_transform(task.layers["flat"])
        detector = HBOS()
        detector.fit(features)
        return {"roc_auc": float(roc_auc_score(task.y, detector.decision_scores_))}


def pyod_extra_methods() -> list:
    """All five extra PyOD tabular baselines, in display order."""
    return [
        IForestDetection(),
        KNNDetection(),
        LOFDetection(),
        COPODDetection(),
        HBOSDetection(),
    ]
