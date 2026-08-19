"""Report how much of the PyGOD board is decided by a constant score.

A detector that returns one value for every node in a run cannot rank anything inside that run.
Under tie-aware scoring it lands on the analytic random floor, which reads on the board as a
detector that was tried and failed. The two readings differ: a detector that ranks and ranks badly
is evidence about graph anomaly detection, while a detector that emits a flat vector is evidence
about the port. This script measures which one the Gold board is showing.

Run it with the repository's usual environment (``GRADE_DIR`` set or a ``grade`` sibling present)::

    python tools/pygod_degeneracy_diagnostic.py                 # the shipped scorer
    python tools/pygod_degeneracy_diagnostic.py --legacy-batch  # the pre-repair scorer

It prints, over the Gold localization graphs, the fraction of runs whose PyGOD node scores are
constant, the distribution of distinct score values per run, and the within-run spread.

``--legacy-batch`` restores the joiner the repair replaced, so the defect this tool exists to
measure can be reproduced rather than merely asserted. It rebuilds each chunk with PyG's
``Batch.from_data_list``, whose ``batch_size`` attribute equals its graph count; PyGOD's detector
reads that attribute to decide how many nodes to write scores for, so all but the first few nodes
of a chunk keep their initial value. The swap is a module-global rebind and ``pygod_extra`` imports
the name at call time, so it is restored in a ``finally`` block: a caller that imports ``main`` or
runs it twice gets the shipped joiner back either way.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))

from catchbench import graph_ad  # noqa: E402
from catchbench.gold import GoldLocalization  # noqa: E402
from catchbench.graph_ad import pygod_node_scores  # noqa: E402


def _legacy_batch_joiner(datas, members):
    """The pre-repair joiner, kept only so its defect stays reproducible.

    Returns the same pair as ``graph_ad.flat_disconnected`` but builds a PyG ``Batch``, which is
    exactly what triggered the truncation.
    """
    from torch_geometric.data import Batch

    batch = Batch.from_data_list([datas[i] for i in members])
    return batch, batch.batch.numpy()

# Below this spread a run's ranking is float noise rather than a decision the detector made.
FLAT_TOL = 1e-6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--legacy-batch", action="store_true",
                    help="score with the pre-repair PyG Batch joiner, to reproduce the defect")
    args = ap.parse_args()

    # The rebind is module-global and pygod_extra imports the name at call time, so an unrestored
    # swap would leave the pre-repair joiner active for every later caller in the same process.
    original_joiner = graph_ad.flat_disconnected
    try:
        if args.legacy_batch:
            graph_ad.flat_disconnected = _legacy_batch_joiner
            print("scoring with the PRE-REPAIR joiner (Batch.from_data_list)\n")

        task = GoldLocalization()
        task.setup()
        scores = pygod_node_scores(task.graphs)
    finally:
        graph_ad.flat_disconnected = original_joiner

    exact_flat, tol_flat, distinct, spreads = 0, 0, [], []
    for per_node in scores:
        arr = np.asarray(per_node, dtype=float)
        if arr.size == 0:
            continue
        spread = float(arr.max() - arr.min())
        spreads.append(spread)
        distinct.append(int(np.unique(arr).size))
        exact_flat += int(np.unique(arr).size == 1)
        tol_flat += int(spread <= FLAT_TOL)

    n = len(spreads)
    if not n:
        print("no graphs scored")
        return 1

    distinct = np.asarray(distinct)
    spreads = np.asarray(spreads)
    print(f"runs scored: {n}")
    print(f"exactly constant:            {exact_flat:4d}  ({100.0 * exact_flat / n:.1f}%)")
    print(f"constant within {FLAT_TOL:g}:     {tol_flat:4d}  ({100.0 * tol_flat / n:.1f}%)")
    print(f"distinct values per run:     min {distinct.min()}, median "
          f"{int(np.median(distinct))}, max {distinct.max()}")
    print(f"within-run score spread:     min {spreads.min():.3e}, median "
          f"{np.median(spreads):.3e}, max {spreads.max():.3e}")
    node_counts = np.asarray([len(np.asarray(x)) for x, _ in task.graphs])
    print(f"nodes per run:               min {node_counts.min()}, median "
          f"{int(np.median(node_counts))}, max {node_counts.max()}")
    print()
    if tol_flat / n > 0.5:
        print("VERDICT: the PyGOD rows are a port artifact, not a measurement of graph anomaly "
              "detection. Report the degeneracy beside the rows and make no family-level claim.")
        return 0
    print("VERDICT: PyGOD ranks within most runs, so its board position is a real measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
