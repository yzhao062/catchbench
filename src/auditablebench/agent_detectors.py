"""Agent-specific published detectors as benchmark baselines: GUARDIAN and G-Safeguard.

Both are run on the dependency-graph substrate so they compete on the same data as every other method,
and both are implemented as their core mechanism with the simplifications noted, in the ADBench / BOND
tradition of porting a published method onto the benchmark's representation rather than gesturing at it.

  - GUARDIAN (Zhou et al., 2025, arXiv:2505.19234) safeguards multi-agent collaboration with an
    UNSUPERVISED reconstruction autoencoder over a temporal attributed graph: it reconstructs node
    attributes and structure and scores a node by reconstruction error. Here it is a directed-GCN
    attribute-reconstruction autoencoder over the per-run dependency graph (the temporal direction is
    the step -> dependency edge); a node it reconstructs poorly is anomalous, and a run's failure score
    is its mean node error. The explicit adjacency-reconstruction term and the Information-Bottleneck
    compression are simplified to an attribute-reconstruction objective that stays structure-aware
    through message passing. Unsupervised: it never sees the failure label.

  - G-Safeguard (Wang et al., 2025, arXiv:2502.11127) detects injected / anomalous agents with a
    SUPERVISED GNN over the multi-agent utterance graph and then remediates topologically. Here it is a
    supervised graph-classification GNN (GCN layers, mean pooling, a linear head) trained with
    seed-averaged stratified cross-validation to predict run failure from the dependency graph. The
    topological remediation step is out of scope because the board scores detection, not intervention.

Both train on CPU with fixed seeds for reproducibility, the same posture as the PyGOD baseline.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from typing import List, Sequence, Tuple

import numpy as np

Graph = Tuple[np.ndarray, np.ndarray]  # (node features [n, d], edge_index [2, m])


def _batched(graphs: Sequence[Graph]):
    """Standardize node features across all graphs and pack them into one disconnected PyG batch,
    self-looping isolated nodes (sparse dependency graphs have many) so message passing is defined."""
    import torch
    from torch_geometric.data import Batch, Data
    from sklearn.preprocessing import StandardScaler

    sizes = [len(np.asarray(x)) for x, _ in graphs]
    stacked = StandardScaler().fit_transform(
        np.vstack([np.asarray(x, dtype=float) for x, _ in graphs]))
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
    return Batch.from_data_list(datas), len(datas)


def guardian_run_scores(graphs: Sequence[Graph], *, seed: int = 0, hid: int = 16,
                        epochs: int = 50) -> np.ndarray:
    """GUARDIAN's unsupervised reconstruction-AE core: fit a directed-GCN attribute autoencoder over
    the run graphs and return each run's mean per-node reconstruction error (higher = more anomalous).
    Seed-averaged over a few inits, because a single GNN init is noisy at this graph scale."""
    import torch
    from torch_geometric.nn import GCNConv

    batch, n_runs = _batched(graphs)
    in_dim = batch.x.shape[1]

    class _AE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc1 = GCNConv(in_dim, hid * 2)
            self.enc2 = GCNConv(hid * 2, hid)          # latent bottleneck (the IB nod)
            self.dec1 = GCNConv(hid, hid * 2)
            self.dec2 = GCNConv(hid * 2, in_dim)       # reconstruct attributes

        def forward(self, x, edge_index):
            z = self.enc2(torch.relu(self.enc1(x, edge_index)), edge_index)
            x_hat = self.dec2(torch.relu(self.dec1(z, edge_index)), edge_index)
            return x_hat

    per_seed = []
    for s in range(3):
        torch.manual_seed(seed + s)
        model = _AE()
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            x_hat = model(batch.x, batch.edge_index)
            loss = ((batch.x - x_hat) ** 2).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            x_hat = model(batch.x, batch.edge_index)
            node_err = ((batch.x - x_hat) ** 2).mean(dim=1).numpy()
        per_seed.append(np.array([node_err[batch.batch.numpy() == k].mean() for k in range(n_runs)]))
    return np.mean(per_seed, axis=0)


def gsafeguard_cv_auc(graphs: Sequence[Graph], y: np.ndarray, *, seed: int = 0, hid: int = 32,
                      epochs: int = 60, n_splits: int = 5) -> float:
    """G-Safeguard's supervised-GNN detector: a graph-classification GCN trained with seed-averaged
    stratified K-fold cross-validation to predict run failure, returning the mean held-out ROC-AUC.
    Grouped at the run, since each run is one graph and one label."""
    import torch
    from torch_geometric.data import Batch
    from torch_geometric.nn import GCNConv, global_mean_pool
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    batch, n_runs = _batched(graphs)
    datas = batch.to_data_list()
    in_dim = batch.x.shape[1]
    y = np.asarray(y)

    class _GNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c1 = GCNConv(in_dim, hid)
            self.c2 = GCNConv(hid, hid)
            self.lin = torch.nn.Linear(hid, 1)

        def forward(self, b):
            h = torch.relu(self.c1(b.x, b.edge_index))
            h = torch.relu(self.c2(h, b.edge_index))
            return self.lin(global_mean_pool(h, b.batch)).squeeze(-1)

    aucs = []
    for s in range(3):  # seed-average for a stable held-out estimate
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + s)
        for tr, te in skf.split(np.zeros(n_runs), y):
            torch.manual_seed(seed + s)
            model = _GNN()
            opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            loss_fn = torch.nn.BCEWithLogitsLoss()
            train_batch = Batch.from_data_list([datas[i] for i in tr])
            train_y = torch.tensor(y[tr], dtype=torch.float)
            model.train()
            for _ in range(epochs):
                opt.zero_grad()
                loss = loss_fn(model(train_batch), train_y)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                test_batch = Batch.from_data_list([datas[i] for i in te])
                prob = torch.sigmoid(model(test_batch)).numpy()
            if len(set(y[te].tolist())) > 1:
                aucs.append(roc_auc_score(y[te], prob))
    return float(np.mean(aucs)) if aucs else 0.5
