"""A graph-AD detector must score every node, not just the head of its chunk.

PyGOD's ``DeepDetector`` decides how many nodes to write scores for by reading ``batch_size`` off
the object its loader yields. A PyG ``Batch`` carries that attribute set to its graph count, so a
chunk of g graphs used to come back with only its first g nodes scored and every other node left at
zero. The board read that as a detector that ranked nothing. These tests lock the two halves of the
fix: the joiner must not reintroduce the attribute, and the detector must return a score that
varies within a run.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("pygod")

from torch_geometric.data import Data  # noqa: E402

from auditablebench.graph_ad import flat_disconnected, pygod_node_scores  # noqa: E402


def _toy_graphs(n_graphs=6, n_nodes=12, n_feat=3, seed=0):
    rng = np.random.RandomState(seed)
    return [(rng.randn(n_nodes, n_feat),
             np.stack([rng.randint(0, n_nodes, 20), rng.randint(0, n_nodes, 20)]))
            for _ in range(n_graphs)]


def test_joined_graph_carries_no_batch_size_attribute():
    """The attribute is the whole bug. If it comes back, the scores silently truncate again."""
    rng = np.random.RandomState(0)
    datas = [Data(x=torch.tensor(rng.randn(5, 3), dtype=torch.float),
                  edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long))
             for _ in range(4)]

    joined, membership = flat_disconnected(datas, list(range(4)))

    assert getattr(joined, "batch_size", None) is None
    assert joined.x.shape[0] == 20
    assert membership.shape == (20,)
    assert [int(v) for v in np.unique(membership)] == [0, 1, 2, 3]
    # Edges must be re-offset into the joined node numbering, never left pointing at graph 0.
    assert int(joined.edge_index.max()) == 17


def test_offsets_keep_each_graph_edges_inside_its_own_block():
    rng = np.random.RandomState(1)
    sizes = [3, 5, 2]
    datas = [Data(x=torch.tensor(rng.randn(n, 2), dtype=torch.float),
                  edge_index=torch.tensor([[0, n - 1], [n - 1, 0]], dtype=torch.long))
             for n in sizes]

    joined, membership = flat_disconnected(datas, [0, 1, 2])

    src, dst = joined.edge_index.numpy()
    assert (membership[src] == membership[dst]).all()


def test_pygod_scores_vary_within_a_run():
    """The regression itself: a constant score per run is what the truncation produced.

    ``max_chunk_nodes`` is set low enough to force several chunks. With one chunk the member index
    and the graph index coincide, so a future bug that indexed membership globally instead of within
    the chunk would still pass.
    """
    scores = pygod_node_scores(_toy_graphs(), hid_dim=8, epoch=5, max_chunk_nodes=24)

    assert len(scores) == 6
    assert all(len(s) == 12 for s in scores)
    varying = sum(float(np.ptp(s)) > 0 for s in scores)
    assert varying == 6, f"only {varying}/6 runs received a non-constant score"


def test_scores_come_back_in_original_per_run_node_order(monkeypatch):
    """Shape and variation are not enough: the split must preserve each run's node order.

    A stub detector returns the standardized first feature as its score, so the expected output is
    known exactly. Graph sizes are uneven and the chunk cap forces a split, which is where a
    membership or offset error would show up as a permutation rather than as a crash.
    """
    import pygod.detector

    class _EchoFirstFeature:
        def __init__(self, **kwargs):
            pass

        def fit(self, data):
            self.decision_score_ = data.x[:, 0].numpy()

    monkeypatch.setattr(pygod.detector, "DOMINANT", _EchoFirstFeature)

    from sklearn.preprocessing import StandardScaler

    empty = np.empty((2, 0), dtype=np.int64)
    features = [np.array(v, dtype=float)[:, None]
                for v in ([10, 11, 12], [20], [30, 31, 32, 33], [40, 41])]
    got = pygod_node_scores([(x, empty) for x in features], epoch=1, max_chunk_nodes=4)

    assert [len(x) for x in got] == [3, 1, 4, 2]
    want = StandardScaler().fit_transform(np.vstack(features))[:, 0]
    np.testing.assert_allclose(np.concatenate(got), want, rtol=1e-6, atol=1e-7)
