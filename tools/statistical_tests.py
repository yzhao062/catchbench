"""Regenerate the paper's inferential ordering claims without an API key.

The comparison families below are the definition of multiplicity for the paper. Every raw
p-value in one family is adjusted together by Holm's step-down procedure, including the required
running-maximum monotonicity correction. Point-estimate ordering in a table is descriptive unless
it is represented by one of these families.

``localization_band_top1`` (28 tests)
    Every unordered Top-1 pair among the eight all-at-once judges from GPT-5.5 through
    Llama-3.3-70B. This tests the claim that the 0.333 to 0.452 band is unresolved.
``localization_gpt55_no_llm_top1`` (5 tests)
    GPT-5.5 against each no-LLM entrant returned by ``post_localization_methods``: random,
    auditable (blast), position, PyGOD, and exec-rank (sup.).
``localization_exec_position`` (3 tests)
    Exec-rank (sup.) against position on Top-1, Top-3, and reciprocal rank.
``localization_small_position_top1`` (2 tests)
    Position against Mistral-Small and Nova-Micro on Top-1.
``localization_protocol_top1`` (20 tests)
    For each of the ten judges having all three committed caches, all-at-once against step-by-step
    and all-at-once against binary-search on Top-1.
``post_detection_auc`` (7 tests)
    The seven POST ROC-AUC contrasts stated together in the detection discussion: dependency over
    size on each corpus; G-Safeguard against full on SWE-Gym; GUARDIAN against ECOD on SWE-Gym;
    auditable against ECOD and GUARDIAN on SWE-Gym; and auditable against full on tau-bench.
``live_swegym_25_auc`` (4 tests)
    At the prespecified first SWE-Gym prefix: auditable against size and ECOD, and full against
    auditable and ECOD.
``live_swegym_auditable_ecod_later_auc`` (3 tests)
    Auditable against ECOD at the remaining three SWE-Gym prefixes. The first prefix belongs to the
    preceding first-prefix family and is not counted twice.
``live_tau_auditable_ecod_auc`` (4 tests)
    Auditable against ECOD at each tau-bench prefix. This is the four-cell within-corpus curve claim.
``live_tau_threshold_auc`` (20 tests)
    Every nonrandom tau-bench method-by-prefix cell against the fixed 0.70 bar. Random is a declared
    floor, not a candidate for the paper's "best reported cell" claim. Adjusting all 20 cells makes
    the selected-best comparison reproducible.
``gold_localization_top1`` (3 tests)
    Full-pool max-span on stale-state against its analytic floor; full-pool has-dep on dropped
    grounding against its analytic floor in the inversion direction; and matched-pool PyGOD against
    max-span overall.
``gold_attribution_auc`` (2 tests)
    Max-span and edge-count against 0.5 on the paired Gold attribution construction.
``pre_rules_flag_all_f1`` (7 tests)
    Every rule-based PRE method against the pooled flag-all floor on capability-level micro-F1.
``pre_combined_judge_f1`` (1 test)
    The pooled combined scanner against the held-out Llama-3.3-70B judge on micro-F1.
``pre_source_best_flag_all_f1`` (6 tests)
    The best non-oracle PRE candidate against flag-all in each of the six sources. Candidate
    selection is repeated inside every bootstrap draw and tested with a max-centered null.
``pre_narrow_precision`` (3 tests)
    Each narrow PRE rule's precision against the pooled capability base rate.

Binary Top-1 and Top-3 outcomes use the exact conditional McNemar test on discordant runs. For a
five-seed method, the inferential binary call is its prespecified majority vote; the board's mean
over seed-specific metrics is reported separately. Reciprocal rank uses Wilcoxon's paired signed-rank
test, with the exact binomial sign test reported only as a diagnostic. Ordinary paired ROC-AUC
contrasts use paired DeLong, cross-checked by a stratified paired bootstrap. Gold attribution uses a
pair-cluster bootstrap and within-pair label-swap randomization. Tests against an analytic Gold floor
use the exact Poisson-binomial randomization distribution, not a sign test.

PRE micro-F1 and precision are ratios of capability counts. Their intervals and paired tests resample
whole configurations, stratified by source for pooled quantities, and recompute each ratio after
summing TP, FP, and FN inside every draw. Capabilities are never resampled independently.

The JSON is deterministic for fixed inputs and options. Typical use from the repository root::

    python tools/statistical_tests.py --output tools/statistical_tests_results.json

Use ``--print-schema`` to print the JSON Schema without loading a corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import special, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ALPHA = 0.05
Z975 = float(stats.norm.ppf(0.975))
CV_SEEDS = tuple(range(5))
PREFIXES = (0.25, 0.5, 0.75, 1.0)
METRICS = ("top1", "top3", "mrr")
PRE_SOURCES = ("crewai", "n8n", "mcp", "injecagent", "sweagent", "synthetic")
PRE_RULE_METHODS = (
    "flag_risky_perms",
    "owasp_excess_permissions",
    "owasp_excess_functionality",
    "owasp_privilege_escalation",
    "unrequested_high_impact",
    "sensitive_access",
    "owasp_asi_combined",
)
PRE_NARROW_RULES = (
    "owasp_privilege_escalation", "unrequested_high_impact", "sensitive_access",
)
PRE_JUDGE_METHOD = "llm_judge_needed(llama-3.3-70b)"
PRE_CANDIDATE_METHODS = PRE_RULE_METHODS + (PRE_JUDGE_METHOD,)
PRE_METHOD_ORDER = (
    "flag_all", "flag_none", *PRE_RULE_METHODS, "oracle_privilege_diff", PRE_JUDGE_METHOD,
)
PRE_SOURCE_BEST = {
    "crewai": PRE_JUDGE_METHOD,
    "n8n": "owasp_excess_functionality",
    "mcp": PRE_JUDGE_METHOD,
    "injecagent": PRE_JUDGE_METHOD,
    "sweagent": "owasp_excess_functionality",
    "synthetic": PRE_JUDGE_METHOD,
}
PRE_PRINTED_FLAG_ALL_F1 = {
    "overall": 0.601,
    "crewai": 0.388,
    "n8n": 0.154,
    "mcp": 0.654,
    "injecagent": 0.750,
    "sweagent": 0.574,
    "synthetic": 0.763,
}
PRE_EQUIVALENCE_MARGIN = 0.05

JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://auditablebench.example/statistical-tests.schema.json",
    "title": "AuditableBench ordering-claim audit",
    "type": "object",
    "required": [
        "schema_version", "settings", "comparison_families", "claims",
        "audit_corrections", "stated_claim_mismatches", "reported_quantities",
    ],
    "properties": {
        "schema_version": {"const": "1.1.0"},
        "settings": {"type": "object"},
        "comparison_families": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "size", "correction", "claim_ids"],
                "properties": {
                    "id": {"type": "string"},
                    "description": {"type": "string"},
                    "size": {"type": "integer", "minimum": 1},
                    "correction": {"const": "Holm step-down with monotonicity enforcement"},
                    "claim_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "id", "family", "label", "metric", "estimate", "interval",
                    "test", "variance_axes", "expected_verdict", "verdict",
                    "matches_stated_claim",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "family": {"type": "string"},
                    "label": {"type": "string"},
                    "metric": {"type": "string"},
                    "estimate": {"type": "object"},
                    "interval": {"type": "object"},
                    "test": {"type": "object"},
                    "variance_axes": {"type": "object"},
                    "expected_verdict": {
                        "enum": ["separates_as_stated", "does_not_separate"]
                    },
                    "verdict": {
                        "enum": [
                            "separates_as_stated", "separates_opposite_to_statement",
                            "does_not_separate",
                        ]
                    },
                    "matches_stated_claim": {"type": "boolean"},
                },
            },
        },
        "audit_corrections": {"type": "array", "items": {"type": "object"}},
        "stated_claim_mismatches": {"type": "array", "items": {"type": "string"}},
        "reported_quantities": {"type": "object"},
    },
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _rng_for(label: str, base_seed: int) -> np.random.Generator:
    digest = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(digest ^ int(base_seed))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm adjusted p-values, including the required cumulative maximum."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be a nonempty one-dimensional sequence in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * float(values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def exact_mcnemar(a: Sequence[bool], b: Sequence[bool]) -> dict[str, Any]:
    """Exact two-sided conditional McNemar test; ``b10`` means A succeeds and B fails."""
    left = np.asarray(a, dtype=bool)
    right = np.asarray(b, dtype=bool)
    if left.shape != right.shape:
        raise ValueError("paired McNemar outcomes must have the same shape")
    b10 = int(np.sum(left & ~right))
    b01 = int(np.sum(~left & right))
    p = 1.0 if b10 + b01 == 0 else float(stats.binomtest(b10, b10 + b01, 0.5).pvalue)
    return {"b10": b10, "b01": b01, "discordant": b10 + b01, "p": p}


def _paired_mean_interval(
    differences: Sequence[float], n_boot: int, rng: np.random.Generator
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    if len(values) == 0:
        raise ValueError("cannot bootstrap an empty vector")
    means = np.empty(n_boot, dtype=float)
    batch = max(1, min(1000, n_boot))
    for start in range(0, n_boot, batch):
        stop = min(n_boot, start + batch)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, (0.025, 0.975)))


def _axis_summary(values: Sequence[float], axis: str, statistic: str) -> dict[str, Any]:
    data = np.asarray(values, dtype=float)
    result: dict[str, Any] = {
        "axis": axis,
        "statistic": statistic,
        "values": data.tolist(),
        "mean": float(data.mean()),
        "sd_population": float(data.std(ddof=0)),
        "min": float(data.min()),
        "max": float(data.max()),
    }
    if len(data) > 1:
        half = float(stats.t.ppf(0.975, len(data) - 1) * data.std(ddof=1) / math.sqrt(len(data)))
        result["mean_t_interval_95"] = [float(data.mean() - half), float(data.mean() + half)]
    return result


def _midranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _auc_fast(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        raise ValueError("ROC-AUC requires both classes")
    ranks = _midranks(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def delong_covariance(labels: Sequence[int], score_vectors: Sequence[Sequence[float]]) -> tuple:
    """DeLong covariance from positive and negative placement values."""
    y = np.asarray(labels, dtype=int)
    scores = [np.asarray(score, dtype=float) for score in score_vectors]
    positives = [score[y == 1] for score in scores]
    negatives = [score[y == 0] for score in scores]
    n_pos, n_neg = len(positives[0]), len(negatives[0])
    v10 = np.zeros((n_pos, len(scores)), dtype=float)
    v01 = np.zeros((n_neg, len(scores)), dtype=float)
    aucs = np.zeros(len(scores), dtype=float)
    for column, (pos, neg) in enumerate(zip(positives, negatives)):
        placement = (pos[:, None] > neg[None, :]).astype(float)
        placement += 0.5 * (pos[:, None] == neg[None, :])
        v10[:, column] = placement.mean(axis=1)
        v01[:, column] = placement.mean(axis=0)
        aucs[column] = placement.mean()
    pos_cov = np.atleast_2d(np.cov(v10, rowvar=False, ddof=1))
    neg_cov = np.atleast_2d(np.cov(v01, rowvar=False, ddof=1))
    return aucs, pos_cov / n_pos + neg_cov / n_neg, n_pos, n_neg


def paired_delong(labels: Sequence[int], a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    aucs, covariance, n_pos, n_neg = delong_covariance(labels, [a, b])
    difference = float(aucs[0] - aucs[1])
    variance = float(covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1])
    variance = max(0.0, variance)
    se = math.sqrt(variance)
    if se == 0:
        z = 0.0 if difference == 0 else math.copysign(math.inf, difference)
        p = 1.0 if difference == 0 else 0.0
    else:
        z = difference / se
        p = float(2 * stats.norm.sf(abs(z)))
    return {
        "auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "difference": difference,
        "se": se, "z": z, "p": p,
        "interval_95": [difference - Z975 * se, difference + Z975 * se],
        "n_positive": n_pos, "n_negative": n_neg,
    }


def single_delong(labels: Sequence[int], scores: Sequence[float], null: float) -> dict[str, Any]:
    aucs, covariance, n_pos, n_neg = delong_covariance(labels, [scores])
    auc = float(aucs[0])
    se = math.sqrt(max(0.0, float(covariance[0, 0])))
    if se:
        z = (auc - null) / se
    elif auc == null:
        z = 0.0
    else:
        z = math.copysign(math.inf, auc - null)
    log_p_less = float(special.log_ndtr(z))
    p_less = math.exp(max(log_p_less, math.log(sys.float_info.min)))
    return {
        "auc": auc, "null": null, "difference": auc - null, "se": se, "z": z,
        "p_less": p_less, "log_p_less": log_p_less,
        "interval_95": [auc - null - Z975 * se, auc - null + Z975 * se],
        "n_positive": n_pos, "n_negative": n_neg,
    }


def _stratified_auc_bootstrap(
    labels: Sequence[int], a: Sequence[float], b: Sequence[float] | None,
    n_boot: int, rng: np.random.Generator,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    score_a = np.asarray(a, dtype=float)
    score_b = None if b is None else np.asarray(b, dtype=float)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    values = np.empty(n_boot, dtype=float)
    for draw in range(n_boot):
        indices = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ])
        boot_y = y[indices]
        value = _auc_fast(boot_y, score_a[indices])
        if score_b is not None:
            value -= _auc_fast(boot_y, score_b[indices])
        values[draw] = value
    interval = np.quantile(values, (0.025, 0.975))
    result = {
        "method": "stratified paired bootstrap" if b is not None else "stratified bootstrap",
        "statistic": "ROC-AUC difference" if b is not None else "ROC-AUC",
        "replicates": n_boot,
        "interval_95": [float(interval[0]), float(interval[1])],
    }
    if b is not None:
        lower = (int(np.sum(values <= 0)) + 1) / (n_boot + 1)
        upper = (int(np.sum(values >= 0)) + 1) / (n_boot + 1)
        result["two_sided_tail_p"] = min(1.0, 2 * min(lower, upper))
    return result


def poisson_binomial_tail(probabilities: Sequence[float], observed: float, alternative: str) -> float:
    """Exact tail under independent Bernoulli probabilities with a possibly fractional statistic."""
    probabilities = np.asarray(probabilities, dtype=float)
    distribution = np.zeros(len(probabilities) + 1, dtype=float)
    distribution[0] = 1.0
    for probability in probabilities:
        distribution[1:] = (
            distribution[1:] * (1 - probability) + distribution[:-1] * probability
        )
        distribution[0] *= 1 - probability
    if alternative == "greater":
        threshold = int(math.ceil(observed - 1e-12))
        return float(distribution[max(0, threshold):].sum())
    if alternative == "less":
        threshold = int(math.floor(observed + 1e-12))
        return float(distribution[:min(len(probabilities), threshold) + 1].sum())
    raise ValueError("alternative must be 'greater' or 'less'")


def _signflip_p(
    differences: Sequence[float], n_random: int, rng: np.random.Generator, alternative: str = "two-sided"
) -> tuple[float, int]:
    values = np.asarray(differences, dtype=float)
    values = values[np.abs(values) > 1e-12]
    if not len(values):
        return 1.0, 0
    observed = float(values.mean())
    extreme = 0
    batch = min(2000, n_random)
    for start in range(0, n_random, batch):
        count = min(batch, n_random - start)
        signs = rng.integers(0, 2, size=(count, len(values)), dtype=np.int8) * 2 - 1
        null = (signs * values).mean(axis=1)
        if alternative == "two-sided":
            extreme += int(np.sum(np.abs(null) >= abs(observed) - 1e-15))
        elif alternative == "greater":
            extreme += int(np.sum(null >= observed - 1e-15))
        else:
            extreme += int(np.sum(null <= observed + 1e-15))
    return (extreme + 1) / (n_random + 1), len(values)


def supervised_oof(
    features: np.ndarray, labels: np.ndarray, seeds: Sequence[int] = CV_SEEDS
) -> tuple[np.ndarray, np.ndarray]:
    """Capture OOF scores while exactly matching GRADE's five-fold logistic-CV recipe."""
    features = np.asarray(features)
    labels = np.asarray(labels)
    oof = np.zeros((len(seeds), len(labels)), dtype=float)
    fold_auc = np.zeros((len(seeds), 5), dtype=float)
    for seed_index, seed in enumerate(seeds):
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold_index, (train, test) in enumerate(folds.split(features, labels)):
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(solver="liblinear", class_weight="balanced"),
            )
            classifier.fit(features[train], labels[train])
            probabilities = classifier.predict_proba(features[test])[:, 1]
            oof[seed_index, test] = probabilities
            fold_auc[seed_index, fold_index] = roc_auc_score(labels[test], probabilities)
    return oof, fold_auc


def _ecod_scores(features: np.ndarray) -> np.ndarray:
    from pyod.models.ecod import ECOD

    standardized = StandardScaler().fit_transform(features)
    detector = ECOD()
    detector.fit(standardized)
    return np.asarray(detector.decision_scores_, dtype=float)


def _gsafeguard_oof(graphs_nx: Sequence, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Registered three-seed G-Safeguard CV, with held-out scores retained for paired testing."""
    import torch
    from torch_geometric.data import Batch
    from torch_geometric.nn import GCNConv, global_mean_pool

    from auditablebench.agent_detectors import _to_data
    from auditablebench.graph_ad import nx_to_graph

    graphs = [nx_to_graph(graph) for graph in graphs_nx]
    labels = np.asarray(labels)
    input_dim = np.asarray(graphs[0][0]).shape[1]

    class GNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = GCNConv(input_dim, 32)
            self.conv2 = GCNConv(32, 32)
            self.linear = torch.nn.Linear(32, 1)

        def forward(self, batch):
            hidden = torch.relu(self.conv1(batch.x, batch.edge_index))
            hidden = torch.relu(self.conv2(hidden, batch.edge_index))
            return self.linear(global_mean_pool(hidden, batch.batch)).squeeze(-1)

    oof = np.zeros((3, len(labels)), dtype=float)
    fold_auc = np.zeros((3, 5), dtype=float)
    for seed_index, seed in enumerate(range(3)):
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold_index, (train, test) in enumerate(folds.split(np.zeros(len(labels)), labels)):
            scaler = StandardScaler().fit(
                np.vstack([np.asarray(graphs[int(index)][0], dtype=float) for index in train])
            )
            train_batch = Batch.from_data_list([_to_data(graphs[int(index)], scaler) for index in train])
            test_batch = Batch.from_data_list([_to_data(graphs[int(index)], scaler) for index in test])
            train_labels = torch.tensor(labels[train], dtype=torch.float)
            torch.manual_seed(seed)
            model = GNN()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            loss_function = torch.nn.BCEWithLogitsLoss()
            model.train()
            for _ in range(60):
                optimizer.zero_grad()
                loss = loss_function(model(train_batch), train_labels)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                probabilities = torch.sigmoid(model(test_batch)).numpy()
            oof[seed_index, test] = probabilities
            fold_auc[seed_index, fold_index] = roc_auc_score(labels[test], probabilities)
    return oof, fold_auc


def _localization_metrics(scores: np.ndarray, groups: np.ndarray, mistake_row: dict) -> np.ndarray:
    rows_out = []
    for run_index in sorted(set(groups.tolist())):
        rows = np.where(groups == run_index)[0]
        order = rows[np.argsort(-scores[rows], kind="stable")]
        rank = int(np.where(order == mistake_row[run_index])[0][0]) + 1
        rows_out.append((rank == 1, rank <= 3, 1.0 / rank))
    return np.asarray(rows_out, dtype=float)


def _majority(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values.mean(axis=0) > 0.5


def _wilcoxon_tie_safe(differences: Sequence[float]):
    """Signed-rank test after restoring rational-rank ties obscured by binary roundoff."""
    rounded = np.around(np.asarray(differences, dtype=float), decimals=12)
    return stats.wilcoxon(rounded, alternative="two-sided", zero_method="wilcox")


def _base_claim(
    claim_id: str, family: str, label: str, metric: str, a_name: str, b_name: str,
    a_value: float, b_value: float, interval: Sequence[float], interval_method: str,
    test_name: str, statistic: dict[str, Any], p_raw: float,
    expected: str, direction: int, variance_axes: dict[str, Any],
    crosscheck: dict[str, Any] | None = None,
    interval_axis: str = "run-level sampling",
) -> dict[str, Any]:
    claim = {
        "id": claim_id,
        "family": family,
        "label": label,
        "metric": metric,
        "direction": direction,
        "estimate": {
            "a_name": a_name, "a": float(a_value), "b_name": b_name, "b": float(b_value),
            "difference_a_minus_b": float(a_value - b_value),
        },
        "interval": {
            "level": 0.95, "low": float(interval[0]), "high": float(interval[1]),
            "axis": interval_axis, "method": interval_method,
        },
        "test": {"name": test_name, "statistic": statistic, "p_raw": float(p_raw)},
        "variance_axes": variance_axes,
        "expected_verdict": expected,
    }
    if crosscheck is not None:
        claim["crosscheck"] = crosscheck
    return claim


def _pre_ratio(counts: np.ndarray, metric: str) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    if values.shape[-1] != 3:
        raise ValueError("PRE count arrays must end with TP, FP, FN")
    tp, fp, fn = values[..., 0], values[..., 1], values[..., 2]
    if metric in ("f1", "micro_f1"):
        numerator, denominator = 2 * tp, 2 * tp + fp + fn
    elif metric == "precision":
        numerator, denominator = tp, tp + fp
    elif metric == "recall":
        numerator, denominator = tp, tp + fn
    else:
        raise ValueError(f"unsupported PRE ratio metric: {metric}")
    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0
    )


def _pre_count_matrix(instances: Sequence, predictions: dict[str, set[str]]) -> np.ndarray:
    counts = np.zeros((len(instances), 3), dtype=np.int64)
    for index, instance in enumerate(instances):
        truth = set(instance.labels["excess_set"])
        predicted = set(predictions.get(instance.instance_id, set()))
        counts[index] = (
            len(predicted & truth), len(predicted - truth), len(truth - predicted)
        )
    return counts


def _pre_prediction_maps(task) -> dict[str, dict[str, set[str]]]:
    from auditablebench.pre import FlagAllMethod, FlagNoneMethod, pre_methods
    from auditablebench.pre_baselines import (
        FlagRiskyPermsMethod, LlmJudgeNeededMethod, PrivilegeDiffOracleMethod,
        _RISKY_PERMISSION_LEVELS,
    )
    from auditablebench.pre_static_scanner import _RuleScanner

    task.setup()
    methods = pre_methods()
    method_ids = tuple(method.method_id for method in methods)
    if method_ids != PRE_METHOD_ORDER:
        raise RuntimeError(
            f"PRE method registry changed: expected {PRE_METHOD_ORDER}, got {method_ids}"
        )
    view = task.method_view()
    declared = {
        row["instance_id"]: {capability["name"] for capability in row["declared_capabilities"]}
        for row in view
    }
    predictions: dict[str, dict[str, set[str]]] = {}
    for method in methods:
        if isinstance(method, FlagAllMethod):
            flagged = {instance_id: set(names) for instance_id, names in declared.items()}
        elif isinstance(method, FlagNoneMethod):
            flagged = {}
        elif isinstance(method, FlagRiskyPermsMethod):
            flagged = {
                row["instance_id"]: {
                    capability["name"]
                    for capability in row["declared_capabilities"]
                    if capability["permission_level"] in _RISKY_PERMISSION_LEVELS
                }
                for row in view
            }
        elif isinstance(method, _RuleScanner):
            flagged = {
                row["instance_id"]: method._rule(
                    {
                        "spec_tokens": row["spec_tokens"],
                        "spec_token_overrides": row["spec_token_overrides"],
                    },
                    row["declared_capabilities"],
                )
                for row in view
            }
        elif isinstance(method, PrivilegeDiffOracleMethod):
            flagged = {
                instance.instance_id: (
                    {capability["name"] for capability in instance.declared_capabilities}
                    - set(instance.minimal_reference)
                )
                for instance in task.instances
            }
        elif isinstance(method, LlmJudgeNeededMethod):
            with open(method.cache_path, encoding="utf-8") as handle:
                rows = json.load(handle)
            flagged = {}
            for instance_id, names in rows.items():
                if instance_id not in declared or not isinstance(names, list):
                    continue
                needed = {name for name in names if name in declared[instance_id]}
                flagged[instance_id] = declared[instance_id] - needed
        else:
            raise RuntimeError(f"unsupported PRE method type: {type(method).__name__}")
        predictions[method.method_id] = flagged
    return predictions


def _pre_configuration_bootstrap(
    a_counts: np.ndarray,
    b_counts: np.ndarray,
    strata: Sequence[np.ndarray],
    metric: str,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if a_counts.shape != b_counts.shape or a_counts.shape != (len(a_counts), 3):
        raise ValueError("paired PRE count matrices must both have shape (configurations, 3)")
    a_values = np.empty(n_boot, dtype=float)
    b_values = np.empty(n_boot, dtype=float)
    batch = max(1, min(500, n_boot))
    for start in range(0, n_boot, batch):
        stop = min(n_boot, start + batch)
        count = stop - start
        a_sum = np.zeros((count, 3), dtype=np.int64)
        b_sum = np.zeros((count, 3), dtype=np.int64)
        for stratum in strata:
            sampled = stratum[rng.integers(0, len(stratum), size=(count, len(stratum)))]
            a_sum += a_counts[sampled].sum(axis=1)
            b_sum += b_counts[sampled].sum(axis=1)
        a_values[start:stop] = _pre_ratio(a_sum, metric)
        b_values[start:stop] = _pre_ratio(b_sum, metric)
    return a_values, b_values, a_values - b_values


def _pre_bootstrap_p(values: np.ndarray, alternative: str) -> float:
    denominator = len(values) + 1
    lower = (int(np.sum(values <= 0)) + 1) / denominator
    if alternative == "greater":
        return lower
    if alternative == "two-sided":
        upper = (int(np.sum(values >= 0)) + 1) / denominator
        return min(1.0, 2 * min(lower, upper))
    raise ValueError("PRE bootstrap alternative must be 'greater' or 'two-sided'")


def _pre_variance_axes(instances: Sequence, source_labels: Sequence[str]) -> dict[str, Any]:
    source_counts = {
        source: int(sum(label == source for label in source_labels)) for source in PRE_SOURCES
    }
    return {
        "configuration_sampling": {
            "axis": "configuration clusters",
            "n_clusters": len(instances),
            "stratified_by_logical_source": len(set(source_labels)) > 1,
            "source_cluster_counts": source_counts,
            "statistic": "ratio difference recomputed after summing counts in each resample",
        },
        "capability_scoring": {
            "axis": "capabilities nested within configurations",
            "n_capabilities": int(sum(len(instance.declared_capabilities) for instance in instances)),
            "statistic": "micro score from pooled TP, FP, and FN",
        },
    }


def _pre_fixed_claim(
    args: argparse.Namespace,
    claim_id: str,
    family: str,
    label: str,
    metric: str,
    a_name: str,
    b_name: str,
    a_counts: np.ndarray,
    b_counts: np.ndarray,
    strata: Sequence[np.ndarray],
    expected: str,
    alternative: str,
    variance_axes: dict[str, Any],
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    bootstrap = _pre_configuration_bootstrap(
        a_counts, b_counts, strata, metric, args.bootstrap, _rng_for(claim_id, args.seed)
    )
    a_draws, b_draws, differences = bootstrap
    interval = np.quantile(differences, (0.025, 0.975))
    a_total = a_counts.sum(axis=0)
    b_total = b_counts.sum(axis=0)
    a_value = float(_pre_ratio(a_total, metric))
    b_value = float(_pre_ratio(b_total, metric))
    p_raw = _pre_bootstrap_p(differences, alternative)
    test_name = (
        f"{alternative} paired configuration-cluster percentile-bootstrap test"
    )
    claim = _base_claim(
        claim_id, family, label, metric, a_name, b_name, a_value, b_value, interval,
        "paired configuration-cluster percentile bootstrap of a ratio of sums", test_name,
        {
            "bootstrap_replicates": args.bootstrap,
            "plus_one_tail_correction": True,
            "monte_carlo_se": math.sqrt(p_raw * (1 - p_raw) / (args.bootstrap + 1)),
            "a_counts": {"tp": int(a_total[0]), "fp": int(a_total[1]), "fn": int(a_total[2])},
            "b_counts": {"tp": int(b_total[0]), "fp": int(b_total[1]), "fn": int(b_total[2])},
        },
        p_raw, expected, 1, variance_axes, interval_axis="configuration-cluster sampling",
    )
    return claim, (a_draws, b_draws, differences)


def _pre_score_summary(counts: np.ndarray) -> dict[str, Any]:
    total = counts.sum(axis=0)
    precision = float(_pre_ratio(total, "precision"))
    recall = float(_pre_ratio(total, "recall"))
    f1 = float(_pre_ratio(total, "f1"))
    return {
        "tp": int(total[0]), "fp": int(total[1]), "fn": int(total[2]),
        "precision": precision, "recall": recall, "f1": f1,
        "printed_3dp": {
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
        },
    }


def _pre_claims(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from auditablebench.pre import PreOverPrivilege, _pre_logical_source

    task = PreOverPrivilege()
    task.setup()
    instances = task.instances
    source_labels = np.asarray([_pre_logical_source(instance.source) for instance in instances])
    if set(source_labels.tolist()) != set(PRE_SOURCES):
        raise RuntimeError("PRE logical source membership changed")
    source_indices = {source: np.flatnonzero(source_labels == source) for source in PRE_SOURCES}
    pooled_strata = [source_indices[source] for source in PRE_SOURCES]
    predictions = _pre_prediction_maps(task)
    counts = {
        method: _pre_count_matrix(instances, flagged) for method, flagged in predictions.items()
    }
    pooled_axes = _pre_variance_axes(instances, source_labels.tolist())

    # Configurations the held-out judge actually judged. It abstains where its reply failed to
    # parse, so any family that includes it must run on this common support; otherwise the board
    # (which reports the judge on its own coverage) and these tests would answer different
    # questions about the same sentence.
    _parsed_ids = set(predictions[PRE_JUDGE_METHOD])
    parsed_index_set = {
        index for index, instance in enumerate(instances) if instance.instance_id in _parsed_ids
    }
    parsed_all_indices = np.asarray(sorted(parsed_index_set), dtype=int)
    # Strata re-expressed in the subset's own coordinates, so the paired bootstrap still resamples
    # configurations within source rather than across the whole corpus.
    _position = {index: position for position, index in enumerate(parsed_all_indices.tolist())}
    parsed_pooled_strata = [
        np.asarray([_position[i] for i in source_indices[source].tolist() if i in parsed_index_set],
                   dtype=int)
        for source in PRE_SOURCES
    ]

    claims: list[dict[str, Any]] = []
    for method in PRE_RULE_METHODS:
        claim, _ = _pre_fixed_claim(
            args,
            f"pre.rule.{method}.vs.flag_all",
            "pre_rules_flag_all_f1",
            f"PRE pooled: {method} vs flag_all",
            "micro_f1",
            method,
            "flag_all",
            counts[method],
            counts["flag_all"],
            pooled_strata,
            "separates_as_stated" if method == "owasp_asi_combined" else "does_not_separate",
            "greater",
            pooled_axes,
        )
        claims.append(claim)

    combined_claim, combined_bootstrap = _pre_fixed_claim(
        args,
        "pre.combined.vs.heldout_judge",
        "pre_combined_judge_f1",
        "PRE pooled: held-out judge vs combined scanner",
        "micro_f1",
        # Judge first, because that is the direction the section states: on the support both
        # methods cover, the held-out judge leads the best rule scanner.
        PRE_JUDGE_METHOD,
        "owasp_asi_combined",
        counts[PRE_JUDGE_METHOD][parsed_all_indices],
        counts["owasp_asi_combined"][parsed_all_indices],
        parsed_pooled_strata,
        "separates_as_stated",
        "two-sided",
        pooled_axes,
    )
    difference_draws = combined_bootstrap[2]
    interval_90 = np.quantile(difference_draws, (0.05, 0.95))
    lower_p = (
        int(np.sum(difference_draws <= -PRE_EQUIVALENCE_MARGIN)) + 1
    ) / (args.bootstrap + 1)
    upper_p = (
        int(np.sum(difference_draws >= PRE_EQUIVALENCE_MARGIN)) + 1
    ) / (args.bootstrap + 1)
    equivalence_p = max(lower_p, upper_p)
    combined_claim["equivalence_diagnostic"] = {
        "status": "demonstrated" if equivalence_p < ALPHA else "not_demonstrated",
        "absolute_f1_margin": PRE_EQUIVALENCE_MARGIN,
        "interval_90": [float(interval_90[0]), float(interval_90[1])],
        "method": "two one-sided configuration-cluster bootstrap tail checks",
        "p_lower_margin": lower_p,
        "p_upper_margin": upper_p,
        "p_tost": equivalence_p,
        "minimum_symmetric_margin_covering_interval_90": float(np.max(np.abs(interval_90))),
        "interpretation": (
            "A nonsignificant difference is not equivalence; equivalence requires both margin tests."
        ),
    }
    claims.append(combined_claim)

    for source in PRE_SOURCES:
        # The judge abstains on five configurations, and it is one of the candidates, so every
        # candidate is scored on the support all of them share. Comparing a method scored on 144
        # configurations against one scored on 143 is the defect the board's coverage column exists
        # to prevent, and it would reappear here silently.
        indices = np.asarray([i for i in source_indices[source] if i in parsed_index_set],
                             dtype=int)
        floor_counts = counts["flag_all"][indices]
        candidate_counts = np.stack([counts[method][indices] for method in PRE_CANDIDATE_METHODS])
        floor_value = float(_pre_ratio(floor_counts.sum(axis=0), "f1"))
        candidate_values = np.asarray([
            float(_pre_ratio(method_counts.sum(axis=0), "f1"))
            for method_counts in candidate_counts
        ])
        observed_differences = candidate_values - floor_value
        winner_index = int(np.argmax(observed_differences))
        winner = PRE_CANDIDATE_METHODS[winner_index]
        if winner != PRE_SOURCE_BEST[source]:
            raise RuntimeError(
                f"PRE source winner changed for {source}: expected {PRE_SOURCE_BEST[source]}, got {winner}"
            )

        selected_draws = np.empty(args.bootstrap, dtype=float)
        centered_max_draws = np.empty(args.bootstrap, dtype=float)
        rng = _rng_for(f"pre.source.{source}.best.vs.flag_all", args.seed)
        batch = max(1, min(250, args.bootstrap))
        local = np.arange(len(indices))
        for start in range(0, args.bootstrap, batch):
            stop = min(args.bootstrap, start + batch)
            sampled = local[rng.integers(0, len(local), size=(stop - start, len(local)))]
            floor_sums = floor_counts[sampled].sum(axis=1)
            candidate_sums = candidate_counts[:, sampled, :].sum(axis=2)
            draw_differences = (
                _pre_ratio(candidate_sums, "f1")
                - _pre_ratio(floor_sums, "f1")[None, :]
            )
            selected_draws[start:stop] = draw_differences.max(axis=0)
            centered_max_draws[start:stop] = (
                draw_differences - observed_differences[:, None]
            ).max(axis=0)
        observed = float(observed_differences[winner_index])
        p_raw = (
            int(np.sum(centered_max_draws >= observed)) + 1
        ) / (args.bootstrap + 1)
        interval = np.quantile(selected_draws, (0.025, 0.975))
        winner_total = candidate_counts[winner_index].sum(axis=0)
        floor_total = floor_counts.sum(axis=0)
        source_instances = [instances[index] for index in indices]
        source_axes = _pre_variance_axes(source_instances, [source] * len(indices))
        claim = _base_claim(
            f"pre.source.{source}.best.vs.flag_all",
            "pre_source_best_flag_all_f1",
            f"PRE {source}: best candidate ({winner}) vs flag_all",
            "micro_f1",
            winner,
            "flag_all",
            candidate_values[winner_index],
            floor_value,
            interval,
            "configuration-cluster percentile bootstrap with candidate reselection",
            "one-sided max-centered configuration-cluster bootstrap test",
            {
                "bootstrap_replicates": args.bootstrap,
                "plus_one_tail_correction": True,
                "selection_repeated_in_each_draw": True,
                "candidate_methods": list(PRE_CANDIDATE_METHODS),
                "winner": winner,
                "winner_counts": {
                    "tp": int(winner_total[0]), "fp": int(winner_total[1]),
                    "fn": int(winner_total[2]),
                },
                "floor_counts": {
                    "tp": int(floor_total[0]), "fp": int(floor_total[1]),
                    "fn": int(floor_total[2]),
                },
                "monte_carlo_se": math.sqrt(p_raw * (1 - p_raw) / (args.bootstrap + 1)),
            },
            p_raw,
            (
                "separates_as_stated"
                if source in ("crewai", "n8n", "injecagent", "synthetic")
                else "does_not_separate"
            ),
            1,
            source_axes,
            interval_axis="configuration-cluster sampling",
        )
        claims.append(claim)

    for method in PRE_NARROW_RULES:
        claim, bootstrap = _pre_fixed_claim(
            args,
            f"pre.precision.{method}.vs.base_rate",
            "pre_narrow_precision",
            f"PRE pooled: {method} precision vs capability base rate",
            "precision",
            method,
            "pooled capability base rate",
            counts[method],
            counts["flag_all"],
            pooled_strata,
            "separates_as_stated",
            "greater",
            pooled_axes,
        )
        precision_interval = np.quantile(bootstrap[0], (0.025, 0.975))
        total = counts[method].sum(axis=0)
        claim["precision_interval"] = {
            "level": 0.95,
            "low": float(precision_interval[0]),
            "high": float(precision_interval[1]),
            "axis": "configuration-cluster sampling",
            "method": "source-stratified configuration-cluster percentile bootstrap",
            "true_positive_capabilities": int(total[0]),
            "predicted_capabilities": int(total[0] + total[1]),
        }
        claims.append(claim)

    flag_identity = []
    identity_groups = [("overall", np.arange(len(instances)))] + [
        (source, source_indices[source]) for source in PRE_SOURCES
    ]
    for source, indices in identity_groups:
        selected = [instances[index] for index in indices]
        n_capabilities = sum(len(instance.declared_capabilities) for instance in selected)
        n_excess = sum(len(instance.labels["excess_set"]) for instance in selected)
        base_rate = n_excess / n_capabilities
        identity_f1 = 2 * base_rate / (1 + base_rate)
        direct_f1 = float(_pre_ratio(counts["flag_all"][indices].sum(axis=0), "f1"))
        printed_f1 = PRE_PRINTED_FLAG_ALL_F1[source]
        exact_match = math.isclose(identity_f1, direct_f1, rel_tol=0.0, abs_tol=1e-15)
        printed_match = round(identity_f1, 3) == printed_f1
        if not exact_match or not printed_match:
            raise RuntimeError(f"flag_all identity failed for {source}")
        flag_identity.append({
            "source": source,
            "n_configurations": len(indices),
            "n_capabilities": n_capabilities,
            "n_excess_capabilities": n_excess,
            "capability_base_rate": base_rate,
            "identity_f1": identity_f1,
            "direct_micro_f1": direct_f1,
            "printed_f1": printed_f1,
            "matches_direct_within_1e-15": exact_match,
            "matches_printed_at_3dp": printed_match,
        })

    base_rates = []
    for source in PRE_SOURCES:
        indices = source_indices[source]
        selected = [instances[index] for index in indices]
        n_capabilities = sum(len(instance.declared_capabilities) for instance in selected)
        n_excess = sum(len(instance.labels["excess_set"]) for instance in selected)
        n_positive_configurations = sum(bool(instance.labels["excess_set"]) for instance in selected)
        base_rates.append({
            "source": source,
            "capability_level": {
                "positive": n_excess,
                "total": n_capabilities,
                "rate": n_excess / n_capabilities,
            },
            "configuration_level": {
                "definition": "configuration has at least one excess capability",
                "positive": n_positive_configurations,
                "total": len(selected),
                "rate": n_positive_configurations / len(selected),
            },
        })

    parsed_ids = set(predictions[PRE_JUDGE_METHOD])
    unparsed_indices = np.asarray([
        index for index, instance in enumerate(instances) if instance.instance_id not in parsed_ids
    ])
    parsed_indices = np.asarray([
        index for index, instance in enumerate(instances) if instance.instance_id in parsed_ids
    ])
    if len(unparsed_indices) != 5:
        raise RuntimeError(f"held-out PRE judge parse-failure count changed: {len(unparsed_indices)}")
    judge_all = _pre_score_summary(counts[PRE_JUDGE_METHOD])
    judge_excluding = _pre_score_summary(counts[PRE_JUDGE_METHOD][parsed_indices])
    oracle_matches = sum(
        predictions["oracle_privilege_diff"][instance.instance_id]
        == set(instance.labels["excess_set"])
        for instance in instances
    )
    if oracle_matches != len(instances):
        raise RuntimeError("oracle_privilege_diff no longer restates every PRE label")
    reported = {
        "pre": {
            "scoring_unit": "capability",
            "resampling_unit": "configuration cluster",
            "micro_score_definition": "sum TP, FP, and FN over capabilities, then form the ratio",
            "flag_all_identity": {
                "formula": "F1 = 2p/(1+p), where p is the capability-level base rate",
                "reason": "flag_all precision equals p and recall equals 1",
                "all_seven_cells_match": True,
                "cells": flag_identity,
            },
            "base_rates_by_source": base_rates,
            "judge_parse_failures": {
                "method": PRE_JUDGE_METHOD,
                "parsed_configurations": len(parsed_indices),
                "unparsed_configurations": len(unparsed_indices),
                "unparsed": [
                    {
                        "instance_id": instances[index].instance_id,
                        "source": source_labels[index],
                        "declared_capabilities": len(instances[index].declared_capabilities),
                        "excess_capabilities": len(instances[index].labels["excess_set"]),
                    }
                    for index in unparsed_indices
                ],
                "scores_counting_unparsed_as_empty_predictions": judge_all,
                "scores_excluding_unparsed_configurations": judge_excluding,
                "change_when_excluded": {
                    "precision": judge_excluding["precision"] - judge_all["precision"],
                    "recall": judge_excluding["recall"] - judge_all["recall"],
                    "f1": judge_excluding["f1"] - judge_all["f1"],
                },
            },
            "oracle_consistency": {
                "definition": "declared capabilities minus minimal reference",
                "matches_excess_set": oracle_matches,
                "total_configurations": len(instances),
                "is_label_identity": oracle_matches == len(instances),
            },
            "declaration_order_leak": _declaration_order_leak(instances, source_labels),
        }
    }
    return claims, reported


def _declaration_order_leak(instances, source_labels) -> dict[str, Any]:
    """Score the rule "keep the first declared capability, flag every later one", per source.

    The harvester that built the injecagent split wrote each configuration as the user's own tool
    followed by the attacker's, so the label is recoverable from position alone. A rule that reads
    no name, no description, and no permission scores a perfect F1 there, above every method that
    reads the configuration. That is a property of the corpus rather than of the rule, and the paper
    says so; this records the number so a reader can check the claim against an artifact instead of
    against prose. It is an exact construction diagnostic on the released corpus rather than a
    population-level method comparison, so it carries no interval and joins no Holm family.
    Determinism is not the reason: resampling configurations would still give a spread.
    """
    per_source: dict[str, Any] = {}
    for source in PRE_SOURCES:
        tp = fp = fn = 0
        members = [instance for instance, label in zip(instances, source_labels.tolist())
                   if label == source]
        for instance in members:
            truth = set(instance.labels["excess_set"])
            # A declared capability is a record, and excess_set holds its name, so the rule keeps
            # the first record and flags the names of the rest.
            predicted = {capability["name"] for capability in instance.declared_capabilities[1:]}
            tp += len(predicted & truth)
            fp += len(predicted - truth)
            fn += len(truth - predicted)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_source[source] = {
            "configurations": len(members),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        }
    return {
        "rule": "flag every declared capability after the first",
        "reads": "declaration order only; no name, description, or permission",
        "per_source": per_source,
        "perfect_sources": sorted(s for s, v in per_source.items() if v["f1"] == 1.0),
    }


def _localization_claims(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    from auditablebench import _reuse  # noqa: F401  installs GRADE's experiment path

    from agent_failure_localization import _oof_scores, _step_matrix

    from auditablebench.graph_ad import full_context_edges, pygod_node_scores
    from auditablebench.llm_judge import _score_vector, load_cache, load_judge_runs, run_key
    from auditablebench.post import PostLocalization, _auditable_blast, post_localization_methods

    task = PostLocalization()
    task.setup()
    judge_runs = load_judge_runs()
    if len(judge_runs) != len(task.runs):
        raise RuntimeError("LLM-judge runs and PostLocalization runs are not aligned")

    judge_models = [
        "gpt-5.5", "claude-opus-4.8", "gpt-5.4", "deepseek-r1", "gemini",
        "qwen3-32b", "gpt-oss-20b", "llama-3.3-70b", "gemma-3-12b",
        "mistral-small", "nova-micro",
    ]
    entries: dict[str, np.ndarray] = {}

    def judge_metrics(model: str, protocol: str) -> np.ndarray:
        cache = load_cache(protocol, model)["predictions"]
        scores = np.concatenate([
            _score_vector(cache[run_key(run)], len(run["steps"])) for run in judge_runs
        ])
        return _localization_metrics(scores, task.groups, task.mistake_row)

    for model in judge_models:
        entries[model] = judge_metrics(model, "all_at_once")[None, :, :]

    random_rows = []
    position_rows = []
    exec_rows = []
    for seed in CV_SEEDS:
        random_rows.append(_localization_metrics(
            np.random.RandomState(seed).rand(len(task.y)), task.groups, task.mistake_row
        ))
        position_rows.append(_localization_metrics(
            _oof_scores(task.X, task.y, task.groups, 1, seed), task.groups, task.mistake_row
        ))
        exec_rows.append(_localization_metrics(
            _oof_scores(task.X, task.y, task.groups, 8, seed), task.groups, task.mistake_row
        ))
    entries["random"] = np.asarray(random_rows)
    entries["position"] = np.asarray(position_rows)
    entries["exec-rank (sup.)"] = np.asarray(exec_rows)
    blast = np.concatenate([_auditable_blast(run) for run in task.runs])
    entries["auditable (blast)"] = _localization_metrics(
        blast, task.groups, task.mistake_row
    )[None, :, :]
    graphs = [(_step_matrix(run), full_context_edges(len(run["steps"]))) for run in task.runs]
    pygod = np.concatenate(pygod_node_scores(graphs, seed=0))
    entries["pygod (graph AD)"] = _localization_metrics(
        pygod, task.groups, task.mistake_row
    )[None, :, :]

    claims: list[dict] = []

    def binary_claim(
        claim_id: str, family: str, a: str, b: str, metric_index: int,
        expected: str, label: str | None = None,
    ) -> None:
        a_rows, b_rows = entries[a][:, :, metric_index], entries[b][:, :, metric_index]
        a_test = _majority(a_rows) if len(a_rows) > 1 else a_rows[0] > 0.5
        b_test = _majority(b_rows) if len(b_rows) > 1 else b_rows[0] > 0.5
        mcnemar = exact_mcnemar(a_test, b_test)
        per_run_diff = a_rows.mean(axis=0) - b_rows.mean(axis=0)
        interval = _paired_mean_interval(
            per_run_diff, args.bootstrap, _rng_for(claim_id, args.seed)
        )
        axes: dict[str, Any] = {
            "run_sampling": {
                "axis": "paired runs", "n": len(per_run_diff),
                "statistic": "mean seed-averaged per-run metric difference",
            }
        }
        n_seed = max(len(a_rows), len(b_rows))
        if n_seed > 1:
            seed_diffs = [
                float(a_rows[index % len(a_rows)].mean() - b_rows[index % len(b_rows)].mean())
                for index in range(n_seed)
            ]
            axes["cv_or_initialization_seed"] = _axis_summary(
                seed_diffs, "method seed", "per-seed board-metric difference"
            )
        claims.append(_base_claim(
            claim_id, family, label or f"{a} vs {b}", METRICS[metric_index], a, b,
            float(a_rows.mean()), float(b_rows.mean()), interval, "paired percentile bootstrap",
            "exact conditional McNemar on majority-vote binary outcomes",
            {key: mcnemar[key] for key in ("b10", "b01", "discordant")}, mcnemar["p"],
            expected, 1, axes,
        ))

    band = judge_models[:8]
    for a, b in combinations(band, 2):
        binary_claim(
            f"loc.band.{a}.vs.{b}", "localization_band_top1", a, b, 0,
            "does_not_separate",
        )

    registered_no_llm = [method.method_id for method in post_localization_methods()]
    expected_no_llm = [
        "random", "auditable (blast)", "position", "pygod (graph AD)", "exec-rank (sup.)"
    ]
    if registered_no_llm != expected_no_llm:
        raise RuntimeError(
            f"localization family changed: expected {expected_no_llm}, got {registered_no_llm}"
        )
    for method in registered_no_llm:
        binary_claim(
            f"loc.gpt55.vs.{method}", "localization_gpt55_no_llm_top1", "gpt-5.5", method, 0,
            "separates_as_stated",
        )

    binary_claim(
        "loc.exec.vs.position.top1", "localization_exec_position", "exec-rank (sup.)",
        "position", 0, "does_not_separate",
    )
    binary_claim(
        "loc.exec.vs.position.top3", "localization_exec_position", "exec-rank (sup.)",
        "position", 1, "separates_as_stated",
    )
    exec_rr = entries["exec-rank (sup.)"][:, :, 2]
    position_rr = entries["position"][:, :, 2]
    rr_diff = exec_rr.mean(axis=0) - position_rr.mean(axis=0)
    wilcoxon = _wilcoxon_tie_safe(rr_diff)
    rr_interval = _paired_mean_interval(
        rr_diff, args.bootstrap, _rng_for("loc.exec.vs.position.mrr", args.seed)
    )
    pos_count = int(np.sum(rr_diff > 1e-12))
    neg_count = int(np.sum(rr_diff < -1e-12))
    sign_p = float(stats.binomtest(pos_count, pos_count + neg_count, 0.5).pvalue)
    seed_rr_diff = exec_rr.mean(axis=1) - position_rr.mean(axis=1)
    claims.append(_base_claim(
        "loc.exec.vs.position.mrr", "localization_exec_position", "exec-rank (sup.) vs position",
        "mrr", "exec-rank (sup.)", "position", float(exec_rr.mean()), float(position_rr.mean()),
        rr_interval, "paired percentile bootstrap", "Wilcoxon paired signed-rank",
        {"W": float(wilcoxon.statistic), "nonzero_pairs": pos_count + neg_count,
         "exact_sign_diagnostic": {"positive": pos_count, "negative": neg_count, "p": sign_p}},
        float(wilcoxon.pvalue), "separates_as_stated", 1,
        {
            "run_sampling": {"axis": "paired runs", "n": len(rr_diff),
                             "statistic": "mean reciprocal-rank difference"},
            "cv_seed": _axis_summary(seed_rr_diff, "cross-validation split seed",
                                      "per-seed MRR difference"),
        },
    ))

    for model in ("mistral-small", "nova-micro"):
        binary_claim(
            f"loc.position.vs.{model}", "localization_small_position_top1", "position", model, 0,
            "does_not_separate",
        )

    protocol_models = [model for model in judge_models if model != "gpt-oss-20b"]
    for model in protocol_models:
        for protocol in ("step_by_step", "binary_search"):
            protocol_key = f"{model}:{protocol}"
            entries[protocol_key] = judge_metrics(model, protocol)[None, :, :]
            expected = (
                "separates_as_stated"
                if model == "qwen3-32b" and protocol == "binary_search"
                else "does_not_separate"
            )
            binary_claim(
                f"loc.protocol.{model}.all.vs.{protocol}", "localization_protocol_top1",
                model, protocol_key, 0, expected,
                label=f"{model}: all-at-once vs {protocol.replace('_', '-')}",
            )

    gpt_rr = entries["gpt-5.5"][0, :, 2]
    exec_mean_rr = exec_rr.mean(axis=0)
    correction_rows = []
    for correction_id, label, differences in (
        ("exec_position_mrr", "exec-rank (sup.) vs position", rr_diff),
        ("gpt55_exec_mrr", "GPT-5.5 vs exec-rank (sup.)", gpt_rr - exec_mean_rr),
    ):
        positive = int(np.sum(differences > 1e-12))
        negative = int(np.sum(differences < -1e-12))
        exact_sign = float(stats.binomtest(positive, positive + negative, 0.5).pvalue)
        signed_rank = _wilcoxon_tie_safe(differences)
        correction_rows.append({
            "id": correction_id,
            "label": label,
            "issue": "A value called an exact sign test was a Wilcoxon signed-rank result.",
            "tie_handling": "Reciprocal-rank differences rounded to 12 decimals before ranking.",
            "exact_two_sided_sign_p": exact_sign,
            "sign_positive": positive,
            "sign_negative": negative,
            "wilcoxon_two_sided_p": float(signed_rank.pvalue),
            "wilcoxon_W": float(signed_rank.statistic),
        })
    return claims, correction_rows


def _auc_entry(oof: np.ndarray | None, fold_auc: np.ndarray | None, score: np.ndarray | None,
               labels: np.ndarray) -> dict[str, Any]:
    if oof is not None:
        primary = np.asarray(oof).mean(axis=0)
        board = float(np.asarray(fold_auc).mean())
        per_seed_board = np.asarray(fold_auc).mean(axis=1)
        per_seed_pooled = np.asarray([roc_auc_score(labels, row) for row in oof])
    else:
        primary = np.asarray(score, dtype=float)
        board = float(roc_auc_score(labels, primary))
        per_seed_board = None
        per_seed_pooled = None
    return {
        "primary": primary, "board": board, "per_seed_board": per_seed_board,
        "per_seed_pooled": per_seed_pooled,
    }


def _auc_claim(
    args: argparse.Namespace, claim_id: str, family: str, label: str, metric: str,
    a_name: str, b_name: str, a: dict[str, Any], b: dict[str, Any], labels: np.ndarray,
    expected: str,
) -> dict[str, Any]:
    result = paired_delong(labels, a["primary"], b["primary"])
    bootstrap = _stratified_auc_bootstrap(
        labels, a["primary"], b["primary"], args.bootstrap, _rng_for(claim_id, args.seed)
    )
    axes: dict[str, Any] = {
        "run_sampling": {
            "axis": "paired labeled runs", "n": len(labels),
            "statistic": "DeLong covariance of seed-averaged OOF or deterministic run scores",
        },
        "board_point_estimates": {
            "axis": "board scoring convention", "a": a["board"], "b": b["board"],
            "difference": a["board"] - b["board"],
        },
    }
    seed_count = 0
    seed_diff = None
    if a["per_seed_board"] is not None and b["per_seed_board"] is not None:
        seed_count = min(len(a["per_seed_board"]), len(b["per_seed_board"]))
        seed_diff = a["per_seed_board"][:seed_count] - b["per_seed_board"][:seed_count]
    elif a["per_seed_board"] is not None:
        seed_diff = a["per_seed_board"] - b["board"]
    elif b["per_seed_board"] is not None:
        seed_diff = a["board"] - b["per_seed_board"]
    if seed_diff is not None:
        axes["cv_seed"] = _axis_summary(
            seed_diff, "cross-validation split seed", "within-seed board AUC difference"
        )
        axes["cv_seed"]["positive_count"] = int(np.sum(seed_diff > 0))
    return _base_claim(
        claim_id, family, label, metric, a_name, b_name, result["auc_a"], result["auc_b"],
        result["interval_95"], "paired DeLong", "paired DeLong z test",
        {"z": result["z"], "se": result["se"], "n_positive": result["n_positive"],
         "n_negative": result["n_negative"]}, result["p"], expected, 1, axes, bootstrap,
    )


def _post_detection_claims(args: argparse.Namespace) -> list[dict]:
    from auditablebench.agent_detectors import guardian_run_scores
    from auditablebench.detection import PostDetection
    from auditablebench.graph_ad import nx_to_graph

    corpora: dict[str, tuple[np.ndarray, dict[str, dict[str, Any]]]] = {}
    for corpus in ("swegym", "tau"):
        task = PostDetection(corpus)
        task.setup()
        labels = np.asarray(task.y)
        entries: dict[str, dict[str, Any]] = {}
        for name, layer in (
            ("size (flat)", "flat"), ("auditable (size+deps)", "flatdep"), ("full", "full")
        ):
            oof, folds = supervised_oof(task.layers[layer], labels)
            entries[name] = _auc_entry(oof, folds, None, labels)
        ecod = _ecod_scores(task.layers["flat"])
        entries["pyod-flatten (ECOD)"] = _auc_entry(None, None, ecod, labels)
        if corpus == "swegym":
            graphs = [nx_to_graph(graph) for graph in task.graphs]
            guardian = guardian_run_scores(graphs)
            entries["guardian (recon-AE)"] = _auc_entry(None, None, guardian, labels)
            gs_oof, gs_folds = _gsafeguard_oof(task.graphs, labels)
            entries["g-safeguard (sup GNN)"] = _auc_entry(gs_oof, gs_folds, None, labels)
        corpora[corpus] = labels, entries

    definitions = [
        ("det.swe.auditable.vs.size", "swegym", "auditable (size+deps)", "size (flat)",
         "separates_as_stated"),
        ("det.tau.auditable.vs.size", "tau", "auditable (size+deps)", "size (flat)",
         "does_not_separate"),
        ("det.swe.gsafeguard.vs.full", "swegym", "g-safeguard (sup GNN)", "full",
         "does_not_separate"),
        ("det.swe.guardian.vs.ecod", "swegym", "guardian (recon-AE)",
         "pyod-flatten (ECOD)", "does_not_separate"),
        ("det.swe.auditable.vs.ecod", "swegym", "auditable (size+deps)",
         "pyod-flatten (ECOD)", "does_not_separate"),
        ("det.swe.auditable.vs.guardian", "swegym", "auditable (size+deps)",
         "guardian (recon-AE)", "does_not_separate"),
        ("det.tau.auditable.vs.full", "tau", "auditable (size+deps)", "full",
         "does_not_separate"),
    ]
    claims = []
    for claim_id, corpus, a_name, b_name, expected in definitions:
        labels, entries = corpora[corpus]
        claims.append(_auc_claim(
            args, claim_id, "post_detection_auc", f"{corpus}: {a_name} vs {b_name}",
            "roc_auc", a_name, b_name, entries[a_name], entries[b_name], labels, expected,
        ))
    return claims


def _live_claims(args: argparse.Namespace) -> list[dict]:
    from auditablebench.live import LiveStreaming, _mean_span, _prefix_steps

    all_entries: dict[str, tuple[np.ndarray, dict[float, dict[str, dict[str, Any]]]]] = {}
    for corpus in ("swegym", "tau"):
        task = LiveStreaming(corpus, prefixes=PREFIXES)
        task.setup()
        labels = np.asarray(task.y)
        by_prefix: dict[float, dict[str, dict[str, Any]]] = {}
        for prefix in PREFIXES:
            entries: dict[str, dict[str, Any]] = {}
            for name, layer in (
                ("size (flat)", "flat"),
                ("auditable (size+deps)", "flatdep"),
                ("full", "full"),
            ):
                oof, folds = supervised_oof(task.layers_at[prefix][layer], labels)
                entries[name] = _auc_entry(oof, folds, None, labels)
            ecod = _ecod_scores(task.layers_at[prefix]["flat"])
            entries["pyod (ECOD)"] = _auc_entry(None, None, ecod, labels)
            span = np.asarray([_mean_span(_prefix_steps(run, prefix)) for run in task.runs])
            entries["dep-span (online)"] = _auc_entry(None, None, span, labels)
            by_prefix[prefix] = entries
        all_entries[corpus] = labels, by_prefix

    claims: list[dict] = []
    labels, swe = all_entries["swegym"]
    claims.append(_auc_claim(
        args, "live.swe25.auditable.vs.size", "live_swegym_25_auc",
        "SWE-Gym 25%: auditable vs size", "roc_auc", "auditable (size+deps)", "size (flat)",
        swe[0.25]["auditable (size+deps)"], swe[0.25]["size (flat)"], labels,
        "separates_as_stated",
    ))
    for prefix in PREFIXES:
        claims.append(_auc_claim(
            args, f"live.swe.{int(prefix * 100)}.auditable.vs.ecod",
            ("live_swegym_25_auc" if prefix == 0.25
             else "live_swegym_auditable_ecod_later_auc"),
            f"SWE-Gym {int(prefix * 100)}%: auditable vs ECOD",
            "roc_auc", "auditable (size+deps)", "pyod (ECOD)",
            swe[prefix]["auditable (size+deps)"], swe[prefix]["pyod (ECOD)"], labels,
            "does_not_separate",
        ))
    for other in ("auditable (size+deps)", "pyod (ECOD)"):
        claims.append(_auc_claim(
            args, f"live.swe25.full.vs.{other}", "live_swegym_25_auc",
            f"SWE-Gym 25%: full vs {other}", "roc_auc", "full", other,
            swe[0.25]["full"], swe[0.25][other], labels, "separates_as_stated",
        ))

    labels, tau = all_entries["tau"]
    for prefix in PREFIXES:
        claims.append(_auc_claim(
            args, f"live.tau.{int(prefix * 100)}.auditable.vs.ecod",
            "live_tau_auditable_ecod_auc",
            f"tau-bench {int(prefix * 100)}%: auditable vs ECOD",
            "roc_auc", "auditable (size+deps)", "pyod (ECOD)",
            tau[prefix]["auditable (size+deps)"], tau[prefix]["pyod (ECOD)"], labels,
            "separates_as_stated",
        ))

    for prefix in PREFIXES:
        for method in (
            "size (flat)", "auditable (size+deps)", "full", "pyod (ECOD)",
            "dep-span (online)",
        ):
            claim_id = f"live.tau.bar.{int(prefix * 100)}.{method}"
            entry = tau[prefix][method]
            result = single_delong(labels, entry["primary"], 0.70)
            bootstrap = _stratified_auc_bootstrap(
                labels, entry["primary"], None, args.bootstrap, _rng_for(claim_id, args.seed)
            )
            axes: dict[str, Any] = {
                "run_sampling": {
                    "axis": "labeled runs", "n": len(labels),
                    "statistic": "single-curve DeLong variance",
                },
                "board_point_estimate": {"axis": "board scoring convention", "value": entry["board"]},
            }
            if entry["per_seed_board"] is not None:
                axes["cv_seed"] = _axis_summary(
                    entry["per_seed_board"] - 0.70, "cross-validation split seed",
                    "per-seed board AUC minus 0.70",
                )
            expected = (
                "does_not_separate"
                if prefix == 1.0 and method in ("auditable (size+deps)", "full")
                else "separates_as_stated"
            )
            claims.append(_base_claim(
                claim_id, "live_tau_threshold_auc",
                f"tau-bench {int(prefix * 100)}%: {method} vs 0.70", "roc_auc",
                method, "fixed bar", result["auc"], 0.70, result["interval_95"],
                "single-curve DeLong", "one-sided DeLong z test (less than 0.70)",
                {"z": result["z"], "se": result["se"], "log_p_raw": result["log_p_less"],
                 "n_positive": result["n_positive"], "n_negative": result["n_negative"]},
                result["p_less"], expected, -1, axes, bootstrap,
            ))
    return claims


def _gold_claims(args: argparse.Namespace) -> list[dict]:
    from auditablebench.gold import (
        GoldAttribution, GoldLocalization, _matched_pools, _ranks, _run_edges, _run_maxspan,
        gold_localization_methods,
    )
    from auditablebench.graph_ad import pygod_node_scores

    task = GoldLocalization(seed=0)
    task.setup()
    groups = sorted(set(task.groups.tolist()))
    stale = [group for group in groups if task.kinds[group] == "stale"]
    dropped = [group for group in groups if task.kinds[group] == "dropped"]
    full_sizes = {group: int(np.sum(task.groups == group)) for group in groups}
    pools = _matched_pools(task)
    methods = {method.method_id: method for method in gold_localization_methods() if hasattr(method, "scores")}
    max_ranks = _ranks(task, methods["max-span (control)"].scores(task))
    hasdep_ranks = _ranks(task, methods["has-dep (control)"].scores(task))

    claims: list[dict] = []

    max_values = np.asarray([max_ranks[group][0] for group in stale])
    max_floor = np.asarray([1.0 / full_sizes[group] for group in stale])
    max_diff = max_values - max_floor
    max_p = poisson_binomial_tail(max_floor, max_values.sum(), "greater")
    max_interval = _paired_mean_interval(
        max_diff, args.bootstrap, _rng_for("gold.maxspan.stale.floor", args.seed)
    )

    has_values = np.asarray([hasdep_ranks[group][0] for group in dropped])
    has_floor = np.asarray([1.0 / full_sizes[group] for group in dropped])
    has_diff = has_values - has_floor
    has_p = poisson_binomial_tail(has_floor, has_values.sum(), "less")
    has_interval = _paired_mean_interval(
        has_diff, args.bootstrap, _rng_for("gold.hasdep.dropped.floor", args.seed)
    )

    injection_max = []
    injection_has = []
    for seed in CV_SEEDS:
        seeded = GoldLocalization(seed=seed)
        seeded.setup()
        seeded_groups = sorted(set(seeded.groups.tolist()))
        seeded_stale = [group for group in seeded_groups if seeded.kinds[group] == "stale"]
        seeded_drop = [group for group in seeded_groups if seeded.kinds[group] == "dropped"]
        sizes = {group: int(np.sum(seeded.groups == group)) for group in seeded_groups}
        seeded_methods = {
            method.method_id: method for method in gold_localization_methods() if hasattr(method, "scores")
        }
        seeded_max = _ranks(seeded, seeded_methods["max-span (control)"].scores(seeded))
        seeded_has = _ranks(seeded, seeded_methods["has-dep (control)"].scores(seeded))
        injection_max.append(float(np.mean([
            seeded_max[group][0] - 1.0 / sizes[group] for group in seeded_stale
        ])))
        injection_has.append(float(np.mean([
            seeded_has[group][0] - 1.0 / sizes[group] for group in seeded_drop
        ])))

    claims.append(_base_claim(
        "gold.maxspan.stale.floor", "gold_localization_top1",
        "Gold full-pool max-span on stale-state vs analytic floor", "top1",
        "max-span (control)", "analytic random floor", max_values.mean(), max_floor.mean(),
        max_interval, "paired run bootstrap", "exact Poisson-binomial upper-tail randomization",
        {"observed_expected_hits": float(max_values.sum()),
         "null_expected_hits": float(max_floor.sum()), "n": len(max_values)},
        max_p, "separates_as_stated", 1,
        {
            "run_sampling": {"axis": "stale-state runs", "n": len(max_values),
                             "statistic": "mean Top-1 minus per-run analytic floor"},
            "injection_seed": _axis_summary(
                injection_max, "synthetic injection seed", "Top-1 minus analytic floor"
            ),
        },
    ))
    claims.append(_base_claim(
        "gold.hasdep.dropped.floor", "gold_localization_top1",
        "Gold full-pool has-dep on dropped grounding vs analytic floor", "top1",
        "has-dep (control)", "analytic random floor", has_values.mean(), has_floor.mean(),
        has_interval, "paired run bootstrap", "exact Poisson-binomial lower-tail randomization",
        {"observed_expected_hits": float(has_values.sum()),
         "null_expected_hits": float(has_floor.sum()), "n": len(has_values)},
        has_p, "separates_as_stated", -1,
        {
            "run_sampling": {"axis": "dropped-grounding runs", "n": len(has_values),
                             "statistic": "mean Top-1 minus per-run analytic floor"},
            "injection_seed": _axis_summary(
                injection_has, "synthetic injection seed", "Top-1 minus analytic floor"
            ),
        },
    ))

    matched_max = _ranks(task, methods["max-span (control)"].scores(task), pools)
    max_overall = np.asarray([matched_max[group][0] for group in groups])
    pygod_by_seed = []
    pygod_per_run = []
    for seed in CV_SEEDS:
        ranks = _ranks(task, pygod_node_scores(task.graphs, seed=seed), pools)
        values = np.asarray([ranks[group][0] for group in groups])
        pygod_per_run.append(values)
        pygod_by_seed.append(float(values.mean()))
    pygod_per_run = np.asarray(pygod_per_run)
    board_pygod = pygod_per_run[0]
    pygod_diff = board_pygod - max_overall
    pygod_interval = _paired_mean_interval(
        pygod_diff, args.bootstrap, _rng_for("gold.pygod.maxspan.matched", args.seed)
    )
    pygod_p, nonzero = _signflip_p(
        pygod_diff, args.randomizations,
        _rng_for("gold.pygod.maxspan.matched.randomization", args.seed), "two-sided",
    )
    claims.append(_base_claim(
        "gold.pygod.maxspan.matched", "gold_localization_top1",
        "Gold matched-pool PyGOD vs max-span overall", "top1", "pygod (graph AD)",
        "max-span (control)", board_pygod.mean(), max_overall.mean(), pygod_interval,
        "paired run bootstrap", "paired sign-flip randomization on the mean difference",
        {"nonzero_pairs": nonzero, "randomizations": args.randomizations,
         "monte_carlo_se": math.sqrt(pygod_p * (1 - pygod_p) / (args.randomizations + 1))},
        pygod_p,
        "does_not_separate", 1,
        {
            "run_sampling": {"axis": "paired Gold runs", "n": len(groups),
                             "statistic": "tie-aware Top-1 difference"},
            "detector_initialization_seed": _axis_summary(
                np.asarray(pygod_by_seed) - max_overall.mean(),
                "PyGOD initialization seed with injection seed fixed at 0",
                "matched overall Top-1 minus deterministic max-span",
            ),
        },
    ))

    attribution = GoldAttribution(seed=0)
    attribution.setup()
    n_pairs = attribution.n_pairs
    labels = np.asarray(attribution.y)
    attribute_scores = {
        "max-span (higher=stale)": np.asarray([_run_maxspan(run) for run in attribution.runs]),
        "edge-count (higher=stale)": np.asarray([_run_edges(run) for run in attribution.runs]),
    }
    injection_auc: dict[str, list[float]] = {key: [] for key in attribute_scores}
    for seed in CV_SEEDS:
        seeded = GoldAttribution(seed=seed)
        seeded.setup()
        for name, function in (
            ("max-span (higher=stale)", _run_maxspan),
            ("edge-count (higher=stale)", _run_edges),
        ):
            scores = np.asarray([function(run) for run in seeded.runs])
            injection_auc[name].append(float(roc_auc_score(seeded.y, scores)))

    for name, scores in attribute_scores.items():
        claim_id = f"gold.attribution.{name}"
        observed = float(roc_auc_score(labels, scores))
        rng = _rng_for(claim_id, args.seed)
        boot = np.empty(args.bootstrap, dtype=float)
        for draw in range(args.bootstrap):
            pair_indices = rng.integers(0, n_pairs, size=n_pairs)
            indices = np.concatenate([pair_indices, pair_indices + n_pairs])
            boot[draw] = roc_auc_score(labels[indices], scores[indices])
        interval = np.quantile(boot - 0.5, (0.025, 0.975))
        permutation_rng = _rng_for(claim_id + ".permutation", args.seed)
        pooled_ranks = _midranks(scores)
        positive_ranks = pooled_ranks[:n_pairs]
        negative_ranks = pooled_ranks[n_pairs:]
        rank_base = n_pairs * (n_pairs + 1) / 2.0
        denominator = float(n_pairs * n_pairs)
        extreme = 0
        batch = min(20_000, args.randomizations)
        for start in range(0, args.randomizations, batch):
            count = min(batch, args.randomizations - start)
            swap = permutation_rng.integers(0, 2, size=(count, n_pairs), dtype=np.int8).astype(bool)
            positive_sum = np.where(
                swap, negative_ranks[None, :], positive_ranks[None, :]
            ).sum(axis=1)
            permuted_auc = (positive_sum - rank_base) / denominator
            extreme += int(np.sum(permuted_auc >= observed - 1e-15))
        p = (extreme + 1) / (args.randomizations + 1)
        claims.append(_base_claim(
            claim_id, "gold_attribution_auc", f"Gold attribution {name} vs chance", "roc_auc",
            name, "chance", observed, 0.5, interval, "pair-cluster percentile bootstrap",
            "within-pair label-swap randomization (greater than 0.5)",
            {"n_pairs": n_pairs, "randomizations": args.randomizations,
             "monte_carlo_se": math.sqrt(p * (1 - p) / (args.randomizations + 1))}, p,
            "separates_as_stated", 1,
            {
                "run_sampling": {"axis": "underlying-run pair clusters", "n_clusters": n_pairs,
                                 "statistic": "ROC-AUC minus 0.5"},
                "injection_seed": _axis_summary(
                    np.asarray(injection_auc[name]) - 0.5, "paired-injection seed",
                    "ROC-AUC minus 0.5",
                ),
            },
        ))
    return claims


FAMILY_DESCRIPTIONS = {
    "localization_band_top1": "All 28 pairs in the eight-judge all-at-once Top-1 band.",
    "localization_gpt55_no_llm_top1": "GPT-5.5 against the five registered no-LLM entrants.",
    "localization_exec_position": "Exec-rank against position on Top-1, Top-3, and MRR.",
    "localization_small_position_top1": "Position against Mistral-Small and Nova-Micro.",
    "localization_protocol_top1": "All-at-once against two alternatives for ten cached judges.",
    "post_detection_auc": "Seven prespecified POST detection AUC contrasts.",
    "live_swegym_25_auc": "Four prespecified contrasts at the first SWE-Gym prefix.",
    "live_swegym_auditable_ecod_later_auc": (
        "Auditable against ECOD at the three later SWE-Gym prefixes."
    ),
    "live_tau_auditable_ecod_auc": "Auditable against ECOD at all four tau-bench prefixes.",
    "live_tau_threshold_auc": "Twenty nonrandom tau method-prefix cells against 0.70.",
    "gold_localization_top1": "Three primary Gold localization Top-1 claims.",
    "gold_attribution_auc": "Two Gold attribution features against chance.",
    "pre_rules_flag_all_f1": "Seven pooled rule-based PRE methods against flag-all.",
    "pre_combined_judge_f1": "Pooled combined PRE scanner against the held-out judge.",
    "pre_source_best_flag_all_f1": (
        "Selection-aware best non-oracle PRE candidate against flag-all in six sources."
    ),
    "pre_narrow_precision": "Three narrow PRE rules' precision against the capability base rate.",
}

EXPECTED_FAMILY_SIZES = {
    "localization_band_top1": 28,
    "localization_gpt55_no_llm_top1": 5,
    "localization_exec_position": 3,
    "localization_small_position_top1": 2,
    "localization_protocol_top1": 20,
    "post_detection_auc": 7,
    "live_swegym_25_auc": 4,
    "live_swegym_auditable_ecod_later_auc": 3,
    "live_tau_auditable_ecod_auc": 4,
    "live_tau_threshold_auc": 20,
    "gold_localization_top1": 3,
    "gold_attribution_auc": 2,
    "pre_rules_flag_all_f1": 7,
    "pre_combined_judge_f1": 1,
    "pre_source_best_flag_all_f1": 6,
    "pre_narrow_precision": 3,
}


def _finalize(claims: list[dict]) -> tuple[list[dict], list[dict]]:
    families = []
    for family_id in FAMILY_DESCRIPTIONS:
        members = [claim for claim in claims if claim["family"] == family_id]
        if not members:
            continue
        expected_size = EXPECTED_FAMILY_SIZES[family_id]
        if len(members) != expected_size:
            raise RuntimeError(
                f"family {family_id} has {len(members)} members, expected {expected_size}"
            )
        adjusted = holm_adjust([claim["test"]["p_raw"] for claim in members])
        for claim, value in zip(members, adjusted):
            claim["test"]["p_adjusted_holm"] = value
            significant = value < ALPHA
            difference = claim["estimate"]["difference_a_minus_b"]
            stated_direction = claim["direction"]
            correct_direction = difference * stated_direction > 0
            if significant and correct_direction:
                verdict = "separates_as_stated"
            elif significant:
                verdict = "separates_opposite_to_statement"
            else:
                verdict = "does_not_separate"
            claim["verdict"] = verdict
            claim["matches_stated_claim"] = verdict == claim["expected_verdict"]
            claim["one_line_verdict"] = (
                f"{claim['label']}: {verdict.replace('_', ' ')} "
                f"(Holm p={value:.6g})."
            )
        families.append({
            "id": family_id,
            "description": FAMILY_DESCRIPTIONS[family_id],
            "size": len(members),
            "correction": "Holm step-down with monotonicity enforcement",
            "claim_ids": [claim["id"] for claim in members],
        })
    return claims, families


def _holm_correction_audit(claims: list[dict]) -> dict[str, Any]:
    members = [claim for claim in claims if claim["family"] == "post_detection_auc"]
    raw = np.asarray([claim["test"]["p_raw"] for claim in members])
    order = np.argsort(raw, kind="stable")
    naive = np.empty(len(raw), dtype=float)
    for rank, index in enumerate(order):
        naive[index] = min(1.0, (len(raw) - rank) * raw[index])
    affected = []
    for index, claim in enumerate(members):
        corrected = claim["test"]["p_adjusted_holm"]
        if corrected > naive[index] + 1e-15:
            affected.append({
                "claim_id": claim["id"], "raw_p": raw[index],
                "nonmonotone_product": naive[index], "correct_holm_p": corrected,
            })
    return {
        "id": "holm_monotonicity",
        "issue": "A prior Holm calculation reported a rankwise product without the running maximum.",
        "family": "post_detection_auc",
        "family_size": len(members),
        "affected_values": affected,
    }


def _format_p(value: float) -> str:
    if value == 0:
        return "0"
    if value < 0.001:
        return f"{value:.3e}"
    return f"{value:.4f}"


def render_table(claims: Sequence[dict]) -> str:
    lines = [
        "| Family | Claim | Statistic (A - B) | Sampling 95% interval | Raw p | Holm p | Verdict |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for claim in claims:
        estimate = claim["estimate"]["difference_a_minus_b"]
        interval = claim["interval"]
        lines.append(
            f"| {claim['family']} | {claim['label']} | {estimate:+.4f} | "
            f"[{interval['low']:+.4f}, {interval['high']:+.4f}] | "
            f"{_format_p(claim['test']['p_raw'])} | "
            f"{_format_p(claim['test']['p_adjusted_holm'])} | "
            f"{claim['verdict'].replace('_', ' ')} |"
        )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "tools" / "statistical_tests_results.json",
        help="JSON destination (default: tools/statistical_tests_results.json)",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--randomizations", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--sections", nargs="+", choices=("localization", "detection", "live", "gold", "pre"),
        default=["localization", "detection", "live", "gold", "pre"],
    )
    parser.add_argument("--print-schema", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.print_schema:
        print(json.dumps(JSON_SCHEMA, indent=2, sort_keys=True))
        return 0
    if args.bootstrap < 100 or args.randomizations < 100:
        raise ValueError("bootstrap and randomizations must each be at least 100")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    random.seed(args.seed)
    np.random.seed(args.seed)

    claims: list[dict] = []
    corrections: list[dict] = []
    reported_quantities: dict[str, Any] = {}
    if "localization" in args.sections:
        section_claims, section_corrections = _localization_claims(args)
        claims.extend(section_claims)
        corrections.extend(section_corrections)
    if "detection" in args.sections:
        claims.extend(_post_detection_claims(args))
    if "live" in args.sections:
        claims.extend(_live_claims(args))
    if "gold" in args.sections:
        claims.extend(_gold_claims(args))
    if "pre" in args.sections:
        section_claims, section_reported = _pre_claims(args)
        claims.extend(section_claims)
        reported_quantities.update(section_reported)

    claims, families = _finalize(claims)
    if "detection" in args.sections:
        corrections.append(_holm_correction_audit(claims))
    mismatches = [claim["id"] for claim in claims if not claim["matches_stated_claim"]]
    from auditablebench.corpora import CORPUS_REVISIONS

    package_versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "torch", "torch-geometric", "pygod", "pyod"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "not installed"
    result = {
        "schema_version": "1.1.0",
        "settings": {
            "alpha": ALPHA,
            "bootstrap_replicates": args.bootstrap,
            "randomization_replicates": args.randomizations,
            "random_seed": args.seed,
            "sections": list(args.sections),
            "cv_seeds": list(CV_SEEDS),
            "no_api_key_required": True,
            "python": sys.version.split()[0],
            "package_versions": package_versions,
            "corpus_revisions": {
                corpus.name: {"repo_id": corpus.repo_id, "revision": corpus.revision}
                for corpus in CORPUS_REVISIONS
            },
        },
        "comparison_families": families,
        "claims": claims,
        "audit_corrections": corrections,
        "stated_claim_mismatches": mismatches,
        "reported_quantities": reported_quantities,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(render_table(claims))
    print(f"\nWrote {args.output}")
    if mismatches:
        print("Stated-claim mismatches: " + ", ".join(mismatches))
    else:
        print("Stated-claim mismatches: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
