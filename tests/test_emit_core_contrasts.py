"""Check the fixed-prediction estimand, task dependence, and both checker directions."""
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import emit_core_contrasts as ecc  # noqa: E402


@pytest.fixture(scope="module")
def record():
    return ecc.load()


@pytest.fixture(scope="module")
def rows(record):
    return ecc.diagnostics(record)


@pytest.fixture(scope="module")
def generated(record):
    return ecc.table(record)


def _cli(*args, paper_env=None):
    env = os.environ.copy()
    env.pop("CATCHBENCH_PAPER_DIR", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if paper_env is not None:
        env["CATCHBENCH_PAPER_DIR"] = str(paper_env)
    return subprocess.run([sys.executable, str(Path(ecc.__file__)), *map(str, args)],
                          capture_output=True, env=env, timeout=60)


def test_four_effects_are_aucs_of_mean_oof_scores(record, rows):
    assert [(r["corpus"], r["specification"]) for r in rows] == [
        ("swegym", "Linear"), ("tau", "Linear"),
        ("swegym", "Matched spline"), ("tau", "Matched spline")]
    for row in rows:
        corpus = record["corpora"][row["corpus"]]
        y = corpus["labels"]
        a = np.asarray(corpus["arms"][row["a_name"]]["oof_proba"])
        b = np.asarray(corpus["arms"][row["b_name"]]["oof_proba"])
        expected = roc_auc_score(y, a.mean(axis=0)) - roc_auc_score(y, b.mean(axis=0))
        assert row["point"] == pytest.approx(expected, rel=0, abs=1e-12)
        pooled = np.mean([roc_auc_score(y, aa) - roc_auc_score(y, bb)
                          for aa, bb in zip(a, b)])
        fold_mean = (corpus["arms"][row["a_name"]]["mean_roc_auc"]
                     - corpus["arms"][row["b_name"]]["mean_roc_auc"])
        assert abs(row["point"] - pooled) > 1e-5
        assert abs(row["point"] - fold_mean) > 1e-5


def test_swegym_intervals_reproduce_paired_placement_variance(record, rows):
    for row in (r for r in rows if r["corpus"] == "swegym"):
        corpus = record["corpora"]["swegym"]
        y = np.asarray(corpus["labels"])
        kernels = []
        for arm in (row["a_name"], row["b_name"]):
            scores = np.asarray(corpus["arms"][arm]["oof_proba"]).mean(axis=0)
            delta = scores[y == 1, None] - scores[None, y == 0]
            kernels.append((delta > 0) + 0.5 * (delta == 0))
        paired = kernels[0] - kernels[1]
        variance = (paired.mean(axis=1).var(ddof=1) / sum(y == 1)
                    + paired.mean(axis=0).var(ddof=1) / sum(y == 0))
        half = norm.ppf(0.975) * np.sqrt(variance)
        np.testing.assert_allclose([row["low"], row["high"]],
                                   paired.mean() + np.array([-half, half]),
                                   rtol=0, atol=1e-12)
        assert row["method"] == "paired run-level DeLong"
        assert "clustered_interval" not in row


def test_linear_rows_agree_with_existing_registry(rows):
    registry = json.loads(ecc.RECORD.with_name("statistical_tests_results.json").read_text())
    for row, claim_id in zip(rows[:2], ("det.swe.auditable.vs.size", "det.tau.auditable.vs.size")):
        claim = next(c for c in registry["claims"] if c["id"] == claim_id)
        assert row["point"] == pytest.approx(claim["estimate"]["difference_a_minus_b"],
                                             rel=0, abs=1e-12)
        np.testing.assert_allclose([row["low"], row["high"]],
                                   [claim["interval"]["low"], claim["interval"]["high"]],
                                   rtol=0, atol=1e-12)


def _weighted_auc(y, score_groups, weights):
    positive = np.bincount(score_groups, weights=weights * y)
    negative = np.bincount(score_groups, weights=weights * (1 - y))
    return np.sum(positive * (np.cumsum(negative) - negative / 2)) / (
        positive.sum() * negative.sum())


def test_tau_bootstrap_matches_independent_task_weight_calculation(record, rows):
    corpus = record["corpora"]["tau"]
    ids = corpus["run_identities"]
    y = np.asarray(corpus["labels"])
    domains = np.asarray(ids["domain"])
    cluster_names, row_cluster = np.unique(ids["task_cluster"], return_inverse=True)
    first_rows = [ids["task_cluster"].index(name) for name in cluster_names]
    strata = [np.flatnonzero(domains[first_rows] == d) for d in ("airline", "retail")]
    assert [len(s) for s in strata] == [50, 115]
    assert np.all(np.bincount(row_cluster) == 4)
    for row in (r for r in rows if r["corpus"] == "tau"):
        seed = int(hashlib.sha256(row["rng_label"].encode()).hexdigest()[:8], 16) ^ 20260907
        assert row["rng_seed"] == seed
        rng = np.random.default_rng(seed)
        groups = [np.unique(np.asarray(corpus["arms"][arm]["oof_proba"]).mean(axis=0),
                            return_inverse=True)[1] for arm in (row["a_name"], row["b_name"])]
        values = []
        for _ in range(10000):
            selected = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
            weights = np.bincount(selected, minlength=165)[row_cluster]
            if not 0 < weights @ y < weights.sum():
                continue
            values.append(_weighted_auc(y, groups[0], weights)
                          - _weighted_auc(y, groups[1], weights))
        expected = np.percentile(values, [2.5, 97.5], method="linear")
        np.testing.assert_allclose([row["low"], row["high"]], expected, rtol=0, atol=1e-12)
        boot = row["clustered_interval"]
        assert boot["replicates"] == boot["draws"] == 10000
        assert boot["usable_replicates"] == len(values)
        assert boot["discarded_single_class_draws"] == 10000 - len(values)
        assert boot["seed"] == 20260907
    assert rows[1]["low"] < 0 < rows[1]["high"]


def test_single_class_draws_are_discarded_and_counted(record):
    clustering = ecc._tau_clusters(record["corpora"]["tau"])
    y = np.zeros(660, dtype=int)
    positive_cluster = clustering["strata"]["airline"][0]
    y[clustering["blocks"][positive_cluster]] = 1
    rng = np.random.default_rng(17)
    usable = 0
    for _ in range(128):
        airline = rng.choice(clustering["strata"]["airline"], size=50)
        rng.choice(clustering["strata"]["retail"], size=115)
        usable += positive_cluster in airline
    result = ecc._task_clustered_auc_bootstrap(
        y, clustering, y.astype(float), 1.0 - y, None, 128, np.random.default_rng(17))
    assert 0 < usable < 128
    assert result["usable_replicates"] == usable
    assert result["discarded_single_class_draws"] == 128 - usable
    assert result["interval_95"] == [1.0, 1.0]


@pytest.mark.parametrize("field", ["scores", "labels", "seeds", "clusters"])
def test_changed_frozen_inputs_fail_loudly(record, field):
    changed = copy.deepcopy(record)
    corpus = changed["corpora"]["tau"]
    if field == "scores":
        corpus["arms"]["size (flat)"]["oof_proba"][0][0] += 0.01
    elif field == "labels":
        corpus["labels"][0] = 1 - corpus["labels"][0]
    elif field == "seeds":
        corpus["seeds"].reverse()
    else:
        corpus["run_identities"]["task_cluster"][0] = "another task"
    with pytest.raises(ValueError, match="cannot reproduce the released contrasts"):
        ecc.table(changed)


def test_emitted_block_never_contains_the_fold_mean_spline_increment(generated):
    assert "-0.010" not in generated
    assert "mean fold-score increments" in generated
    assert "five pooled per-seed AUC differences" in generated
    assert generated.count(r"\begin{table}") == generated.count(r"\end{table}") == 1
    assert r"\textbf" not in generated
    assert "Holm" not in generated
    assert "p =" not in generated


def test_cli_prints_deterministic_lf_bytes(generated):
    first, second = _cli(), _cli()
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == b""
    assert first.stdout == second.stdout == (generated + "\n").encode("utf-8")
    assert b"\r" not in first.stdout


@pytest.mark.parametrize("use_env", [False, True])
def test_check_accepts_exact_block(tmp_path, generated, use_env):
    (tmp_path / "09_appendix.tex").write_bytes((generated + "\n").encode("utf-8"))
    result = (_cli("--check", paper_env=tmp_path) if use_env
              else _cli("--check", "--paper", tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert b"paper is current: tab:core-contrasts" in result.stdout


@pytest.mark.parametrize("corruption, message", [
    ("cell", b"table differs"), ("missing_begin", b"appears 0 times"),
    ("missing_end", b"appears 0 times"), ("no_table", b"appears 0 times"),
    ("no_appendix", b"cannot read appendix"), ("duplicate", b"appears 2 times"),
    ("reversed", b"precedes its begin marker"), ("whitespace", b"table differs"),
    ("crlf", b"table differs"),
])
def test_check_rejects_stale_paper(tmp_path, generated, corruption, message):
    text = generated
    if corruption == "cell":
        text = text.replace("SWE-Gym & Linear & $+0.147$", "SWE-Gym & Linear & $+0.999$", 1)
    elif corruption == "missing_begin":
        text = text.replace(ecc._BEGIN, "")
    elif corruption == "missing_end":
        text = text.replace(ecc._END, "")
    elif corruption == "no_table":
        text = r"\section{An appendix without this table}"
    elif corruption == "duplicate":
        text += "\n" + ecc._BEGIN
    elif corruption == "reversed":
        text = ecc._END + "\n" + ecc._BEGIN
    elif corruption == "whitespace":
        text = text.replace(r"\centering", r"\centering ", 1)
    elif corruption == "crlf":
        text = text.replace("\n", "\r\n")
    if corruption != "no_appendix":
        (tmp_path / "09_appendix.tex").write_bytes(text.encode("utf-8"))
    result = _cli("--check", "--paper", tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert message in result.stdout
    if corruption == "cell":
        assert b"--- paper/09_appendix.tex" in result.stdout
        assert b"+0.999" in result.stdout and b"+0.147" in result.stdout


def test_check_requires_paper_path():
    result = _cli("--check")
    assert result.returncode == 2
    assert b"--check needs --paper <dir> or CATCHBENCH_PAPER_DIR" in result.stderr
