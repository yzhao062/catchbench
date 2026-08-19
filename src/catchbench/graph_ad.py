"""Graph anomaly detection via PyGOD, the graph-AD counterpart to PyOD on this benchmark.

One DOMINANT graph autoencoder (Ding et al., 2019) is fit over a batch of per-run graphs: it
reconstructs node attributes and graph structure, and a node the model reconstructs poorly scores
as anomalous. The detector is unsupervised (it never sees the failure / fault labels), so it is the
structure-aware sibling of the unsupervised PyOD baseline that reads flat per-run features only. The
question the pairing poses on the leaderboard: does graph structure help an off-the-shelf anomaly
detector find failures and faults beyond what a flat detector sees?

PyGOD trains through PyG's ``NeighborLoader``, which needs the ``pyg-lib`` / ``torch-sparse``
backend; both are installed for this torch build. Training runs on CPU for reproducibility.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from typing import List, Sequence, Tuple

import numpy as np

# A per-run graph: (node features [n, d], edge_index [2, m]).
Graph = Tuple[np.ndarray, np.ndarray]


def flat_disconnected(datas: Sequence, members: Sequence[int]):
    """Join several per-run graphs into one disconnected graph, without PyG's ``Batch`` wrapper.

    ``Batch`` exposes a ``batch_size`` attribute equal to its graph count. PyGOD's ``DeepDetector``
    reads that attribute off the object the loader yields and writes scores for only that many
    nodes (``pygod/detector/base.py``: ``batch_size = sampled_data.batch_size`` then
    ``decision_score_[node_idx[:batch_size]] = score``). Feeding it a ``Batch`` of g graphs
    therefore scores the first g nodes of the chunk and silently leaves every other node at its
    initial zero, which reads downstream as a detector that ranked nothing.

    A plain ``Data`` carrying the same disconnected graph has no ``batch_size`` attribute, so the
    loader supplies the true seed count and every node is scored. Returns the joined ``Data`` and an
    integer membership vector giving each node's index into ``members``.
    """
    import torch
    from torch_geometric.data import Data

    xs, edges, membership, offset = [], [], [], 0
    for local_k, gi in enumerate(members):
        d = datas[gi]
        n = int(d.x.shape[0])
        xs.append(d.x)
        edges.append(d.edge_index + offset)
        membership.append(np.full(n, local_k, dtype=np.int64))
        offset += n
    joined = Data(x=torch.cat(xs, dim=0), edge_index=torch.cat(edges, dim=1))
    return joined, np.concatenate(membership)


def pygod_node_scores(
    graphs: Sequence[Graph], *, hid_dim: int = 32, epoch: int = 40, max_chunk_nodes: int = 3000,
    seed: int = 0,
) -> List[np.ndarray]:
    """Fit DOMINANT autoencoders over the per-run graphs and return per-node anomaly scores split
    back per run, each array in the graph's original node order.

    Node features are standardized across the whole set before training (degrees and shares live on
    different scales). Runs are grouped into chunks of at most ``max_chunk_nodes`` nodes, and each
    chunk is fit FULL-BATCH as one disconnected ``Data`` built by :func:`flat_disconnected`, whose
    docstring explains why a PyG ``Batch`` here scores only the chunk's first few nodes. Full-batch
    keeps
    DOMINANT's dense O(n^2) adjacency reconstruction bounded by the chunk cap, and sidesteps a PyG
    minibatch-sampling index bug on graphs with many isolated nodes (the sparse dependency-only
    graphs the Gold board builds). A run's nodes always stay in one chunk, so its within-run ranking
    comes from a single model.
    """
    import torch
    from torch_geometric.data import Data
    from pygod.detector import DOMINANT
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(seed)
    np.random.seed(seed)

    sizes = [len(np.asarray(x)) for x, _ in graphs]
    stacked = StandardScaler().fit_transform(
        np.vstack([np.asarray(x, dtype=float) for x, _ in graphs]))

    datas, offset = [], 0
    for (x, edges), n in zip(graphs, sizes):
        node_x = torch.tensor(stacked[offset:offset + n], dtype=torch.float)
        offset += n
        e = np.asarray(edges, dtype=np.int64).reshape(2, -1)
        touched = set(e.flatten().tolist()) if e.shape[1] else set()
        isolated = [v for v in range(n) if v not in touched]  # only these trip PyG's sampler
        if isolated:  # self-loop ONLY isolated nodes; keep real adjacency for the rest
            loops = np.array(isolated, dtype=np.int64)
            e = np.concatenate([e, np.stack([loops, loops])], axis=1)
        datas.append(Data(x=node_x, edge_index=torch.tensor(e, dtype=torch.long)))

    results: List[np.ndarray] = [np.zeros(0)] * len(datas)

    def _fit_chunk(members: list) -> None:
        if not members:
            return
        joined, membership = flat_disconnected(datas, members)
        detector = DOMINANT(hid_dim=hid_dim, num_layers=2, epoch=epoch, batch_size=0,
                            gpu=-1, verbose=0)  # batch_size=0 => one full batch, no neighbor sampling
        detector.fit(joined)
        scores = np.asarray(detector.decision_score_, dtype=float)
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


def full_context_edges(n: int) -> np.ndarray:
    """The full-context DAG on ``n`` ordered steps: every step depends on all prior steps, the
    dependency assumption the Who&When debate corpus uses. Edges point current -> prior (step i ->
    each prior j), matching GRADE's ``depends_on`` direction and the Gold step graphs."""
    src, dst = [], []
    for i in range(n):
        for j in range(i):
            src.append(i)  # step i depends on prior step j: edge i -> j (current -> prior)
            dst.append(j)
    return np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)


def nx_to_graph(graph) -> Graph:
    """Convert a GRADE typed networkx graph to (node features, edge_index) for PyGOD. Node features
    are structural and type-based: in / out degree (total degree if undirected) and a one-hot over
    decision / tool-call / other node types, so the detector reads the typed graph, not run size."""
    nodes = list(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    directed = graph.is_directed()
    feats = []
    for node in nodes:
        ntype = graph.nodes[node].get("ntype", "")
        in_deg = float(graph.in_degree(node)) if directed else float(graph.degree(node))
        out_deg = float(graph.out_degree(node)) if directed else 0.0
        feats.append([
            in_deg,
            out_deg,
            1.0 if ntype == "decision" else 0.0,
            1.0 if ntype == "tool_call" else 0.0,
            1.0 if ntype not in ("decision", "tool_call") else 0.0,
        ])
    src, dst = [], []
    for u, v in graph.edges():
        src.append(index[u])
        dst.append(index[v])
    edges = np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)
    return np.array(feats, dtype=float), edges
