"""Measure seed stability of the repaired PyGOD graph-AD board rows.

The script runs five detector-initialization seeds by default. It covers all four PyGOD
post-detection entrants on SWE-Gym and tau-bench, plus the DOMINANT row that is actually
registered on the Who&When and Gold localization boards. For Gold, the canonical injection
seed remains fixed at zero so detector noise is not mixed with injection-sampling noise.

Detection diagnostics include run-score correlation with node count, node-count-only ROC-AUC,
and an exact-size-matched ROC-AUC. The latter compares failed and resolved runs only when their
integer node counts are identical, leaving no residual size variation within a comparison.

Run from the repository root with the benchmark Python environment::

    GRADE_DIR=/path/to/grade python tools/pygod_seed_stability.py \
        --output tools/pygod_seed_stability_results.json

The output is checkpointed after every expensive seed fit and can be resumed by rerunning the
same command. Use ``--no-resume`` to replace an existing output.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

METRICS = ("top1", "top3", "mrr")
DETECTORS = ("pygod (graph AD)", "pygod-conad", "pygod-anomalydae", "pygod-gaan")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=("detection", "whoandwhen", "gold", "leaders"),
        default=["detection", "whoandwhen", "gold", "leaders"],
    )
    parser.add_argument("--corpora", nargs="+", choices=("swegym", "tau"),
                        default=["swegym", "tau"])
    parser.add_argument("--output", type=Path,
                        default=ROOT / "tools" / "pygod_seed_stability_results.json")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _new_results(seeds: Sequence[int]) -> dict:
    return {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "seeds": list(seeds),
            "gold_injection_seed": 0,
            "epoch": 40,
            "max_chunk_nodes": 3000,
            "size_control": "exact integer node-count matched, pair-weighted ROC-AUC",
        },
        "detection": {},
        "localization": {},
        "leaders": {},
        "timing": {"invocations": []},
    }


def _load_results(path: Path, seeds: Sequence[int], resume: bool) -> dict:
    if resume and path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        old_seeds = result.get("metadata", {}).get("seeds")
        if old_seeds != list(seeds):
            raise ValueError(f"checkpoint seeds {old_seeds} do not match requested seeds {list(seeds)}")
        return result
    return _new_results(seeds)


def _checkpoint(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25)


def _summary(values: Sequence[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "values": [float(value) for value in arr],
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)


def _pygod_scores(detector: str, graphs: Sequence, seed: int):
    _seed_everything(seed)
    if detector == "pygod (graph AD)":
        from auditablebench.graph_ad import pygod_node_scores

        return pygod_node_scores(graphs, seed=seed)

    from auditablebench.pygod_extra import _pygod_extra_node_scores
    if detector == "pygod-conad":
        from pygod.detector import CONAD

        cls = CONAD
    elif detector == "pygod-anomalydae":
        from pygod.detector import AnomalyDAE

        cls = AnomalyDAE
    elif detector == "pygod-gaan":
        from pygod.detector import GAAN

        cls = GAAN
    else:
        raise ValueError(f"unknown detector: {detector}")
    return _pygod_extra_node_scores(
        cls, graphs, hid_dim=32, num_layers=2, epoch=40, max_chunk_nodes=3000, seed=seed
    )


def _exact_size_matched_auc(labels: np.ndarray, scores: np.ndarray,
                            sizes: np.ndarray) -> tuple[float, dict]:
    """Pair-weighted concordance over positive-negative pairs with exactly equal node counts."""
    from sklearn.metrics import roc_auc_score

    weighted_auc = 0.0
    total_pairs = 0
    matchable_runs: set[int] = set()
    matchable_sizes = []
    for size in np.unique(sizes):
        idx = np.flatnonzero(sizes == size)
        y = labels[idx]
        positives = int(y.sum())
        negatives = int(len(y) - positives)
        pairs = positives * negatives
        if not pairs:
            continue
        auc = float(roc_auc_score(y, scores[idx]))
        weighted_auc += pairs * auc
        total_pairs += pairs
        matchable_runs.update(int(i) for i in idx)
        matchable_sizes.append(int(size))
    if total_pairs == 0:
        return float("nan"), {
            "pair_count": 0, "run_count": 0, "stratum_count": 0, "sizes": []
        }
    return weighted_auc / total_pairs, {
        "pair_count": total_pairs,
        "run_count": len(matchable_runs),
        "stratum_count": len(matchable_sizes),
        "sizes": matchable_sizes,
    }


def _run_detection(result: dict, path: Path, seeds: Sequence[int], corpora: Sequence[str]) -> None:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    from auditablebench.detection import PostDetection
    from auditablebench.graph_ad import nx_to_graph

    for corpus in corpora:
        print(f"loading detection corpus {corpus}", flush=True)
        setup_start = time.perf_counter()
        task = PostDetection(corpus)
        task.setup()
        graphs = [nx_to_graph(graph) for graph in task.graphs]
        sizes = np.asarray([len(x) for x, _ in graphs], dtype=int)
        labels = np.asarray(task.y, dtype=int)
        corpus_result = result["detection"].setdefault(corpus, {})
        corpus_result.update({
            "runs": int(len(labels)),
            "failed": int(labels.sum()),
            "nodes": int(sizes.sum()),
            "node_count_auc": float(roc_auc_score(labels, sizes)),
            "setup_seconds": float(time.perf_counter() - setup_start),
        })
        detector_results = corpus_result.setdefault("detectors", {})
        for detector in DETECTORS:
            detector_result = detector_results.setdefault(detector, {"seeds": {}})
            for seed in seeds:
                key = str(seed)
                if key in detector_result["seeds"]:
                    print(f"resume {corpus} {detector} seed {seed}", flush=True)
                    continue
                print(f"fit {corpus} {detector} seed {seed}", flush=True)
                started = time.perf_counter()
                per_run = _pygod_scores(detector, graphs, seed)
                run_scores = np.asarray([np.asarray(scores, dtype=float).mean()
                                         for scores in per_run])
                matched_auc, match_info = _exact_size_matched_auc(labels, run_scores, sizes)
                rho = float(spearmanr(run_scores, sizes).statistic)
                detector_result["seeds"][key] = {
                    "roc_auc": float(roc_auc_score(labels, run_scores)),
                    "score_size_spearman": rho,
                    "exact_size_matched_auc": float(matched_auc),
                    "exact_size_match": match_info,
                    "seconds": float(time.perf_counter() - started),
                }
                _checkpoint(path, result)
            ordered = [detector_result["seeds"][str(seed)] for seed in seeds]
            for field in ("roc_auc", "score_size_spearman", "exact_size_matched_auc", "seconds"):
                detector_result[field] = _summary([row[field] for row in ordered])
            detector_result["exact_size_match"] = ordered[0]["exact_size_match"]
            _checkpoint(path, result)


def _run_whoandwhen(result: dict, path: Path, seeds: Sequence[int]) -> None:
    from auditablebench.graph_ad import full_context_edges
    from auditablebench.post import PostLocalization
    from agent_failure_localization import _rank_metrics, _seed_metrics, _step_matrix

    print("loading Who&When localization", flush=True)
    setup_start = time.perf_counter()
    task = PostLocalization()
    task.setup()
    graphs = [(_step_matrix(run), full_context_edges(len(run["steps"]))) for run in task.runs]
    section = result["localization"].setdefault("whoandwhen", {
        "runs": int(len(task.runs)), "nodes": int(sum(len(x) for x, _ in graphs)),
        "setup_seconds": float(time.perf_counter() - setup_start),
        "pygod (graph AD)": {"seeds": {}},
    })
    detector_result = section["pygod (graph AD)"]
    for seed in seeds:
        key = str(seed)
        if key in detector_result["seeds"]:
            print(f"resume Who&When DOMINANT seed {seed}", flush=True)
            continue
        print(f"fit Who&When DOMINANT seed {seed}", flush=True)
        started = time.perf_counter()
        per_run = _pygod_scores("pygod (graph AD)", graphs, seed)
        metrics = _rank_metrics(np.concatenate(per_run), task.groups, task.mistake_row)
        detector_result["seeds"][key] = {
            **{name: float(value) for name, value in zip(METRICS, metrics)},
            "seconds": float(time.perf_counter() - started),
        }
        _checkpoint(path, result)
    ordered = [detector_result["seeds"][str(seed)] for seed in seeds]
    for field in (*METRICS, "seconds"):
        detector_result[field] = _summary([row[field] for row in ordered])

    # GRADE returns one metric triple per split seed for the registered supervised reference.
    structure = _seed_metrics(task.X, task.y, task.groups, task.mistake_row, "structure")
    structure_seeds = range(len(structure))  # fixed by GRADE, independent of the PyGOD seed sweep
    leader = section.setdefault("exec-rank (sup.)", {})
    leader["seeds"] = {
        str(seed): {name: float(value) for name, value in zip(METRICS, row)}
        for seed, row in zip(structure_seeds, structure)
    }
    for column, field in enumerate(METRICS):
        leader[field] = _summary(structure[:, column])
    _checkpoint(path, result)


def _run_gold(result: dict, path: Path, seeds: Sequence[int]) -> None:
    from auditablebench.gold import (
        GoldLocalization,
        _kind_summary,
        _matched_pools,
        _ranks,
    )

    print("loading Gold localization at canonical injection seed 0", flush=True)
    setup_start = time.perf_counter()
    task = GoldLocalization(seed=0)
    task.setup()
    all_groups = sorted(set(task.groups.tolist()))
    subsets = {
        "overall": all_groups,
        "stale_state": [group for group in all_groups if task.kinds[group] == "stale"],
        "dropped_grounding": [group for group in all_groups if task.kinds[group] == "dropped"],
    }
    pools = _matched_pools(task)
    section = result["localization"].setdefault("gold", {
        "runs": int(len(task.graphs)),
        "nodes": int(sum(len(x) for x, _ in task.graphs)),
        "injection_seed": 0,
        "setup_seconds": float(time.perf_counter() - setup_start),
        "pygod (graph AD)": {"seeds": {}},
    })
    detector_result = section["pygod (graph AD)"]
    for seed in seeds:
        key = str(seed)
        if key in detector_result["seeds"] and all(
            field in detector_result["seeds"][key]
            for field in ("full_pool_by_kind", "eligibility_matched_by_kind")
        ):
            print(f"resume Gold DOMINANT seed {seed}", flush=True)
            continue
        print(f"fit Gold DOMINANT seed {seed}", flush=True)
        started = time.perf_counter()
        per_run = _pygod_scores("pygod (graph AD)", task.graphs, seed)
        full_ranks = _ranks(task, per_run)
        matched_ranks = _ranks(task, per_run, pools)
        full_by_kind = {
            name: _kind_summary(full_ranks, groups) for name, groups in subsets.items()
        }
        matched_by_kind = {
            name: _kind_summary(matched_ranks, groups) for name, groups in subsets.items()
        }
        full = full_by_kind["overall"]
        matched = matched_by_kind["overall"]
        detector_result["seeds"][key] = {
            "full_pool": {name: float(value) for name, value in zip(METRICS, full)},
            "eligibility_matched": {
                name: float(value) for name, value in zip(METRICS, matched)
            },
            "full_pool_by_kind": {
                kind: {name: float(value) for name, value in zip(METRICS, values)}
                for kind, values in full_by_kind.items()
            },
            "eligibility_matched_by_kind": {
                kind: {name: float(value) for name, value in zip(METRICS, values)}
                for kind, values in matched_by_kind.items()
            },
            "seconds": float(time.perf_counter() - started),
        }
        _checkpoint(path, result)
    ordered = [detector_result["seeds"][str(seed)] for seed in seeds]
    for pool in ("full_pool", "eligibility_matched"):
        detector_result[pool] = {
            field: _summary([row[pool][field] for row in ordered]) for field in METRICS
        }
    for pool in ("full_pool_by_kind", "eligibility_matched_by_kind"):
        detector_result[pool] = {
            kind: {
                field: _summary([row[pool][kind][field] for row in ordered])
                for field in METRICS
            }
            for kind in subsets
        }
    detector_result["seconds"] = _summary([row["seconds"] for row in ordered])
    _checkpoint(path, result)


def _gsafeguard_seed_auc(graphs: Sequence, labels: np.ndarray, seed: int,
                         hid: int = 32, epochs: int = 60, n_splits: int = 5) -> float:
    """One initialization/split seed of the registered G-Safeguard implementation."""
    import torch
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from torch_geometric.data import Batch
    from torch_geometric.nn import GCNConv, global_mean_pool

    from auditablebench.agent_detectors import _to_data

    _seed_everything(seed)
    labels = np.asarray(labels)
    input_dim = np.asarray(graphs[0][0]).shape[1]

    class GNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = GCNConv(input_dim, hid)
            self.conv2 = GCNConv(hid, hid)
            self.linear = torch.nn.Linear(hid, 1)

        def forward(self, batch):
            hidden = torch.relu(self.conv1(batch.x, batch.edge_index))
            hidden = torch.relu(self.conv2(hidden, batch.edge_index))
            return self.linear(global_mean_pool(hidden, batch.batch)).squeeze(-1)

    aucs = []
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, test in folds.split(np.zeros(len(graphs)), labels):
        scaler = StandardScaler().fit(
            np.vstack([np.asarray(graphs[int(i)][0], dtype=float) for i in train])
        )
        train_batch = Batch.from_data_list([_to_data(graphs[int(i)], scaler) for i in train])
        test_batch = Batch.from_data_list([_to_data(graphs[int(i)], scaler) for i in test])
        train_labels = torch.tensor(labels[train], dtype=torch.float)
        torch.manual_seed(seed)
        model = GNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        loss_function = torch.nn.BCEWithLogitsLoss()
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = loss_function(model(train_batch), train_labels)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(model(test_batch)).numpy()
        aucs.append(float(roc_auc_score(labels[test], probabilities)))
    return float(np.mean(aucs))


def _run_leaders(result: dict, path: Path, seeds: Sequence[int]) -> None:
    from auditablebench.detection import PostDetection
    from auditablebench.graph_ad import nx_to_graph
    from agent_failure_detection import _cv

    if list(seeds) != list(range(5)):
        raise ValueError("the registered supervised feature-layer reference uses seeds 0 through 4")
    print("loading SWE-Gym supervised leaders", flush=True)
    task = PostDetection("swegym")
    task.setup()
    graphs = [nx_to_graph(graph) for graph in task.graphs]
    section = result["leaders"].setdefault("swegym", {})

    # One value per CV split seed, with the five held-out fold AUCs averaged within seed.
    structure_per_seed = _cv(task.layers["flatdep"], task.y).mean(axis=1)
    section["auditable (size+deps)"] = {
        "roc_auc": _summary(structure_per_seed),
        "seeds": {str(seed): {"roc_auc": float(value)}
                  for seed, value in zip(seeds, structure_per_seed)},
    }

    safeguard = section.setdefault("g-safeguard (sup GNN)", {"seeds": {}})
    for seed in seeds:
        key = str(seed)
        if key in safeguard["seeds"]:
            print(f"resume G-Safeguard seed {seed}", flush=True)
            continue
        print(f"fit G-Safeguard seed {seed}", flush=True)
        started = time.perf_counter()
        safeguard["seeds"][key] = {
            "roc_auc": _gsafeguard_seed_auc(graphs, task.y, seed),
            "seconds": float(time.perf_counter() - started),
        }
        _checkpoint(path, result)
    ordered = [safeguard["seeds"][str(seed)] for seed in seeds]
    safeguard["roc_auc"] = _summary([row["roc_auc"] for row in ordered])
    safeguard["seconds"] = _summary([row["seconds"] for row in ordered])
    _checkpoint(path, result)


def _print_summary(result: dict, seeds: Sequence[int]) -> None:
    def stats(summary: dict) -> str:
        return (f"{summary['mean']:.3f} +/- {summary['std']:.3f} "
                f"[{summary['min']:.3f}, {summary['max']:.3f}]")

    for corpus, corpus_result in result["detection"].items():
        print(f"\nDetection {corpus}: node-count AUC {corpus_result['node_count_auc']:.3f}")
        for detector, detector_result in corpus_result["detectors"].items():
            print(f"  {detector:22s} AUC {stats(detector_result['roc_auc'])}; "
                  f"rho {stats(detector_result['score_size_spearman'])}; "
                  f"exact-size AUC {stats(detector_result['exact_size_matched_auc'])}")
    for board, board_result in result["localization"].items():
        print(f"\nLocalization {board}:")
        for method, method_result in board_result.items():
            if method in ("runs", "nodes", "injection_seed", "setup_seconds"):
                continue
            if board == "gold" and method == "pygod (graph AD)":
                for pool in ("full_pool", "eligibility_matched"):
                    print(f"  {method} {pool}: " + ", ".join(
                        f"{metric} {stats(method_result[pool][metric])}" for metric in METRICS))
            else:
                print(f"  {method}: " + ", ".join(
                    f"{metric} {stats(method_result[metric])}" for metric in METRICS))
    if result["leaders"]:
        print("\nSWE-Gym supervised detection references:")
        for method, method_result in result["leaders"]["swegym"].items():
            print(f"  {method}: {stats(method_result['roc_auc'])}")

    expensive_seconds = []
    for corpus_result in result["detection"].values():
        for detector_result in corpus_result["detectors"].values():
            expensive_seconds.extend(detector_result["seconds"]["values"])
    for board_result in result["localization"].values():
        method_result = board_result.get("pygod (graph AD)")
        if method_result:
            expensive_seconds.extend(method_result["seconds"]["values"])
    safeguard = result.get("leaders", {}).get("swegym", {}).get("g-safeguard (sup GNN)")
    if safeguard:
        expensive_seconds.extend(safeguard["seconds"]["values"])
    print(f"\nRecorded expensive-fit time: {sum(expensive_seconds):.1f} seconds "
          f"over {len(expensive_seconds)} seed jobs ({len(seeds)} seeds requested).")


def main() -> int:
    args = _parse_args()
    seeds = tuple(args.seeds)
    if len(seeds) < 1 or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list of distinct integers")
    result = _load_results(args.output, seeds, resume=not args.no_resume)
    invocation_start = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()

    if "detection" in args.sections:
        _run_detection(result, args.output, seeds, args.corpora)
    if "whoandwhen" in args.sections:
        _run_whoandwhen(result, args.output, seeds)
    if "gold" in args.sections:
        _run_gold(result, args.output, seeds)
    if "leaders" in args.sections:
        _run_leaders(result, args.output, seeds)

    result["timing"]["invocations"].append({
        "started_utc": started_utc,
        "sections": list(args.sections),
        "corpora": list(args.corpora),
        "wall_seconds": float(time.perf_counter() - invocation_start),
    })
    _checkpoint(args.output, result)
    _print_summary(result, seeds)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
