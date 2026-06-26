"""Extra unsupervised PyGOD graph anomaly detectors as POST detection baselines.

Adds CONAD, AnomalyDAE, and GAAN alongside the DOMINANT baseline already in graph_ad.py and
detection.py. Each detector follows the same full-batch + self-loop-isolated-nodes pattern
required by the sparse GRADE dependency graphs: chunked batches (at most max_chunk_nodes nodes
per chunk), a single full-batch fit per chunk (batch_size=0), and self-loops added for isolated
nodes only. Run-level failure score is each run's mean node anomaly score; ROC-AUC vs task.y is
the reported metric.

Heavy imports (torch, torch_geometric, pygod) are deferred inside functions, matching graph_ad.py.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from typing import List, Sequence, Tuple, Type

import numpy as np

# Mirrors the Graph type from graph_ad.py: (node_features [n, d], edge_index [2, m]).
Graph = Tuple[np.ndarray, np.ndarray]


def _pygod_extra_node_scores(
    detector_cls: Type,
    graphs: Sequence[Graph],
    *,
    hid_dim: int = 32,
    num_layers: int = 2,
    epoch: int = 40,
    max_chunk_nodes: int = 3000,
    seed: int = 0,
    **detector_kwargs,
) -> List[np.ndarray]:
    """Fit ``detector_cls`` over the per-run graphs and return per-node anomaly scores split back
    per run, each array in the graph's original node order.

    Follows the same chunked full-batch + self-loop-isolated-nodes pattern as
    ``graph_ad.pygod_node_scores``. Node features are standardized across the whole set before
    training. Runs are grouped into chunks capped at ``max_chunk_nodes`` total nodes; each chunk
    is fit full-batch (one disconnected PyG Batch, no neighbor sampling). Self-loops are added
    only for truly isolated nodes to avoid PyG's NeighborLoader index crash on sparse graphs.
    """
    import torch
    from torch_geometric.data import Batch, Data
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(seed)
    np.random.seed(seed)

    sizes = [len(np.asarray(x)) for x, _ in graphs]
    stacked = StandardScaler().fit_transform(
        np.vstack([np.asarray(x, dtype=float) for x, _ in graphs])
    )

    datas, offset = [], 0
    for (x, edges), n in zip(graphs, sizes):
        node_x = torch.tensor(stacked[offset:offset + n], dtype=torch.float)
        offset += n
        e = np.asarray(edges, dtype=np.int64).reshape(2, -1)
        touched = set(e.flatten().tolist()) if e.shape[1] else set()
        isolated = [v for v in range(n) if v not in touched]
        if isolated:
            loops = np.array(isolated, dtype=np.int64)
            e = np.concatenate([e, np.stack([loops, loops])], axis=1)
        datas.append(Data(x=node_x, edge_index=torch.tensor(e, dtype=torch.long)))

    results: List[np.ndarray] = [np.zeros(0)] * len(datas)

    def _fit_chunk(members: list) -> None:
        if not members:
            return
        batch = Batch.from_data_list([datas[i] for i in members])
        detector = detector_cls(
            hid_dim=hid_dim,
            num_layers=num_layers,
            epoch=epoch,
            batch_size=0,
            gpu=-1,
            verbose=0,
            **detector_kwargs,
        )
        detector.fit(batch)
        scores = np.asarray(detector.decision_score_, dtype=float)
        membership = batch.batch.numpy()
        for local_k, gi in enumerate(members):
            results[gi] = scores[membership == local_k]

    members, chunk_nodes = [], 0
    for gi, n in enumerate(sizes):
        if members and chunk_nodes + n > max_chunk_nodes:
            _fit_chunk(members)
            members, chunk_nodes = [], 0
        members.append(gi)
        chunk_nodes += n
    _fit_chunk(members)
    return results


class _PyGODExtraDetection:
    """Shared base for extra PyGOD run-level detectors.

    Subclasses set ``method_id`` and ``_detector_cls``; extra constructor kwargs to the PyGOD
    detector (beyond hid_dim / num_layers / epoch / batch_size / gpu / verbose) go in
    ``_detector_kwargs``.
    """

    method_id: str
    supports = {"post_detection"}
    _detector_cls: Type
    _detector_kwargs: dict = {}

    def evaluate(self, task) -> dict:
        from sklearn.metrics import roc_auc_score
        from auditablebench.graph_ad import nx_to_graph

        task.setup()
        graphs = [nx_to_graph(g) for g in task.graphs]
        per_run = _pygod_extra_node_scores(
            self._detector_cls,
            graphs,
            hid_dim=32,
            num_layers=2,
            epoch=40,
            **self._detector_kwargs,
        )
        run_scores = np.array([scores.mean() for scores in per_run])
        return {"roc_auc": float(roc_auc_score(task.y, run_scores))}


class CONADDetection(_PyGODExtraDetection):
    """CONAD (Contrastive Attributed Network Anomaly Detection) as a run-level failure detector.

    Ding et al., 2022. Trains a GCN autoencoder with contrastive augmentation; mean node anomaly
    score per run is the failure score. Unsupervised and structure-aware.
    """

    method_id = "pygod-conad"

    @property
    def _detector_cls(self):
        from pygod.detector import CONAD
        return CONAD


class AnomalyDAEDetection(_PyGODExtraDetection):
    """AnomalyDAE (Deep Autoencoder for Network Anomaly Detection) as a run-level failure detector.

    Fan et al., 2020. Dual autoencoder over node attributes and graph structure; mean node
    anomaly score per run is the failure score. Unsupervised and structure-aware.
    """

    method_id = "pygod-anomalydae"

    @property
    def _detector_cls(self):
        from pygod.detector import AnomalyDAE
        return AnomalyDAE


class GAANDetection(_PyGODExtraDetection):
    """GAAN (Generative Adversarial Attribution Networks) as a run-level failure detector.

    Chen et al., 2020. GAN-based graph anomaly detector; mean node anomaly score per run is the
    failure score. Unsupervised and structure-aware.
    """

    method_id = "pygod-gaan"

    @property
    def _detector_cls(self):
        from pygod.detector import GAAN
        return GAAN


def pygod_extra_methods() -> list:
    """Return the three extra PyGOD graph-AD baseline instances."""
    return [CONADDetection(), AnomalyDAEDetection(), GAANDetection()]
