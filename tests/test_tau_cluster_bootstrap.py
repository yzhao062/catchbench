r"""The tau-bench task-clustered interval: its design invariants, and the guards that admit it.

tau-bench's 660 runs are 165 task instances attempted by four agent models each, so a run-level
interval counts four attempts at one task as four observations. These tests pin the replacement
construction rather than its output alone: that whole clusters travel together, that the domain
mixture is fixed, that the statistic is the one the cell prints, and that the emitter accepts an
asymmetric percentile interval while still rejecting a wrong one. The published anchor is checked
against the shipped record so a silent regeneration cannot quietly restore the narrow interval.

No corpus, GRADE checkout, or network is needed. The one real-data case reads the committed audit
arrays, which is also what the two independent implementations of this specification used.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import emit_live_prefix_table as elpt  # noqa: E402
import statistical_tests as st  # noqa: E402

AUDIT = ROOT / "tools" / "detection_audit_results.json"

# Round 25's task-clustered reading of the core tau contrast, reproduced independently twice before
# this code existed. The bar is the third decimal: at 10,000 draws the Monte Carlo spread on one
# endpoint is about 0.002, so equality is not the claim and would be the wrong claim.
ANCHOR_CONTRAST = (-0.005, 0.098)
ANCHOR_TOLERANCE = 0.0025


@pytest.fixture(scope="module")
def audit():
    return json.loads(AUDIT.read_text(encoding="utf-8"))["corpora"]["tau"]


@pytest.fixture(scope="module")
def tau_scores(audit):
    return {name: np.asarray(arm["oof_proba"], dtype=float).mean(axis=0)
            for name, arm in audit["arms"].items() if "oof_proba" in arm}


@pytest.fixture(scope="module")
def tau_clustering(audit):
    identities = audit["run_identities"]
    return st._tau_clustering(
        np.asarray(audit["labels"], dtype=int), identities["task_cluster"], identities["domain"]
    )


def _toy(n_clusters=24, per_cluster=4, seed=7):
    """A small task-by-model grid with a real cluster effect, so clustering has to matter."""
    rng = np.random.default_rng(seed)
    cluster_ids, strata, labels, score_a, score_b = [], [], [], [], []
    for index in range(n_clusters):
        difficulty = rng.normal()
        label = int(index % 2 == 0)
        for _ in range(per_cluster):
            cluster_ids.append(f"task-{index:03d}")
            strata.append("airline" if index < n_clusters // 3 else "retail")
            labels.append(label)
            score_a.append(difficulty + 0.9 * label + 0.05 * rng.normal())
            score_b.append(difficulty + 0.3 * label + 0.05 * rng.normal())
    return (np.asarray(labels), cluster_ids, strata,
            np.asarray(score_a), np.asarray(score_b))


def test_the_vectorized_rank_auc_equals_the_shipped_one_on_tie_heavy_input():
    # A cluster resample duplicates whole rows, so ties are the rule rather than an edge case.
    rng = np.random.default_rng(0)
    for _ in range(60):
        size = int(rng.integers(20, 300))
        values = rng.integers(0, 5, size=size).astype(float)
        assert np.array_equal(st._midranks(values), st._midranks_vectorized(values))
        labels = rng.integers(0, 2, size=size)
        if labels.min() == labels.max():
            continue
        assert st._auc_fast(labels, values) == st._auc_vectorized(labels, values)


def test_ragged_or_cross_stratum_clusters_are_refused():
    with pytest.raises(RuntimeError, match="not equal-sized"):
        st._cluster_blocks(["a", "a", "b"], ["retail"] * 3)
    with pytest.raises(RuntimeError, match="spans strata"):
        st._cluster_blocks(["a", "a"], ["retail", "airline"])


def test_the_clustering_record_states_the_frozen_design(tau_clustering):
    record = tau_clustering["record"]
    assert record["n_rows"] == 660
    assert record["n_clusters"] == 165
    assert record["rows_per_cluster"] == 4
    assert record["clusters_per_stratum"] == {"airline": 50, "retail": 115}
    assert record["seed"] == st.TAU_CLUSTER_SEED == 20260907
    assert record["draws"] == st.TAU_CLUSTER_DRAWS == 10_000


def test_every_draw_keeps_whole_clusters_and_the_domain_mixture():
    labels, cluster_ids, strata, score_a, score_b = _toy()
    clustering = st._tau_clustering(labels, cluster_ids, strata)
    blocks, chosen_strata = clustering["blocks"], clustering["strata"]
    domain = np.asarray(strata)
    rng = np.random.default_rng(11)
    for _ in range(200):
        chosen = np.concatenate([
            rows[rng.integers(0, len(rows), size=len(rows))] for rows in chosen_strata.values()
        ])
        indices = blocks[chosen].reshape(-1)
        assert indices.size == len(labels)
        # whole blocks, in cluster order, never a partial cluster or a reshuffled one
        assert np.array_equal(indices.reshape(-1, blocks.shape[1]), blocks[chosen])
        for name, rows in chosen_strata.items():
            assert int((domain[indices] == name).sum()) == len(rows) * blocks.shape[1]


def test_the_interval_is_reproducible_and_carries_its_own_draw_accounting(audit, tau_clustering,
                                                                         tau_scores):
    labels = np.asarray(audit["labels"], dtype=int)

    def run(seed):
        return st._task_clustered_auc_bootstrap(
            labels, tau_clustering, tau_scores["auditable (size+deps)"], tau_scores["size (flat)"],
            None, 400, st._rng_for("det.tau.auditable.vs.size", seed),
        )

    first, second = run(st.TAU_CLUSTER_SEED), run(st.TAU_CLUSTER_SEED)
    assert first["interval_95"] == second["interval_95"]
    assert first["usable_replicates"] + first["discarded_single_class_draws"] == 400
    assert first["method"] == st.TAU_CLUSTER_METHOD
    assert first["axis"] == st.TAU_CLUSTER_AXIS
    assert first["n_clusters"] == 165
    # A different seed must move the endpoints, and only inside Monte Carlo noise. At 400 draws
    # that noise is far wider than at the shipped 10,000, so the bound is loosened accordingly.
    other = run(st.TAU_CLUSTER_SEED + 1)
    assert other["interval_95"] != first["interval_95"]
    assert other["point"] == first["point"]
    for a, b in zip(first["interval_95"], other["interval_95"]):
        assert abs(a - b) < 0.02


def test_the_clustered_interval_reproduces_the_published_anchor(audit, tau_clustering, tau_scores):
    labels = np.asarray(audit["labels"], dtype=int)
    result = st._task_clustered_auc_bootstrap(
        labels, tau_clustering, tau_scores["auditable (size+deps)"], tau_scores["size (flat)"],
        None, st.TAU_CLUSTER_DRAWS, st._rng_for("det.tau.auditable.vs.size", st.TAU_CLUSTER_SEED),
    )
    low, high = result["interval_95"]
    assert result["point"] == pytest.approx(0.046294, abs=5e-6)
    assert low == pytest.approx(ANCHOR_CONTRAST[0], abs=ANCHOR_TOLERANCE)
    assert high == pytest.approx(ANCHOR_CONTRAST[1], abs=ANCHOR_TOLERANCE)
    # The finding, and the reason the fix is a correctness item: zero is inside.
    assert low < 0 < high
    # The run-level interval this replaces excluded zero and was narrower.
    run_level = st.paired_delong(labels, tau_scores["auditable (size+deps)"],
                                 tau_scores["size (flat)"])["interval_95"]
    assert run_level[0] > 0
    assert (high - low) > 1.3 * (run_level[1] - run_level[0])


def test_a_threshold_interval_is_stored_on_auc_minus_the_bar(audit, tau_clustering, tau_scores):
    labels = np.asarray(audit["labels"], dtype=int)
    shifted = st._task_clustered_auc_bootstrap(
        labels, tau_clustering, tau_scores["full"], None, 0.70, 600,
        st._rng_for("live.tau.bar.100.full", st.TAU_CLUSTER_SEED),
    )
    plain = st._task_clustered_auc_bootstrap(
        labels, tau_clustering, tau_scores["full"], None, None, 600,
        st._rng_for("live.tau.bar.100.full", st.TAU_CLUSTER_SEED),
    )
    assert shifted["point"] == pytest.approx(plain["point"] - 0.70, abs=1e-12)
    for moved, bare in zip(shifted["interval_95"], plain["interval_95"]):
        assert moved == pytest.approx(bare - 0.70, abs=1e-12)
    assert shifted["statistic"] == "ROC-AUC minus 0.7"


def test_a_difference_and_a_bar_cannot_be_asked_for_together(tau_clustering, tau_scores):
    with pytest.raises(ValueError, match="not on both"):
        st._task_clustered_auc_bootstrap(
            np.zeros(660, dtype=int), tau_clustering, tau_scores["full"], tau_scores["full"],
            0.70, 10, np.random.default_rng(0),
        )


def test_endpoint_proximity_reports_a_near_zero_endpoint_and_stays_quiet_otherwise():
    def claim(claim_id, low, high, b_name="fixed bar"):
        return {"id": claim_id, "clustered_interval": {"method": st.TAU_CLUSTER_METHOD},
                "interval": {"low": low, "high": high},
                "estimate": {"b_name": b_name}}

    audit = st._clustered_endpoint_proximity([
        claim("near.bar", -0.0004, 0.05),
        claim("clear", -0.04, 0.05),
        claim("near.zero", -0.06, 0.0011, b_name="full"),
        {"id": "unclustered", "interval": {"low": 0.0, "high": 0.0},
         "estimate": {"b_name": "fixed bar"}},
    ])
    assert audit["clustered_rows_scanned"] == 3
    assert [(row["claim_id"], row["endpoint"]) for row in audit["affected_values"]] == [
        ("near.bar", "low"), ("near.zero", "high")]
    assert audit["affected_values"][0]["reference"].startswith("0.70 bar")
    assert audit["affected_values"][1]["reference"] == "zero"


# --- the emitter's input guard ------------------------------------------------------------------

def _clustered_claim(claim_id, method, low, high, point=None):
    a = 0.66
    difference = a - 0.70
    return {
        "id": claim_id, "family": "live_tau_threshold_auc", "metric": "roc_auc",
        "estimate": {"a_name": method, "a": a, "b_name": "fixed bar", "b": 0.70,
                     "difference_a_minus_b": difference},
        "interval": {"level": 0.95, "low": low, "high": high,
                     "method": st.TAU_CLUSTER_METHOD, "axis": st.TAU_CLUSTER_AXIS},
        "clustered_interval": {
            "method": st.TAU_CLUSTER_METHOD, "axis": st.TAU_CLUSTER_AXIS,
            "replicates": 10_000, "usable_replicates": 10_000,
            "interval_95": [low, high], "point": difference if point is None else point,
            "n_rows": 660, "n_clusters": 165, "rows_per_cluster": 4,
            "clusters_per_stratum": {"airline": 50, "retail": 115},
            "seed": st.TAU_CLUSTER_SEED, "monte_carlo_noise_on_one_endpoint": 0.002,
        },
        "variance_axes": {"board_point_estimate": {"value": a},
                          "run_sampling": {"axis": "labeled runs", "n": 660}},
    }


def test_the_guard_accepts_an_asymmetric_clustered_interval():
    claim = _clustered_claim("live.tau.bar.100.full", "full", -0.098, 0.020)
    elpt._check_endpoints(claim["id"], "clustered", claim)
    # the same endpoints would fail as a DeLong interval, which is the point of branching
    with pytest.raises(SystemExit, match="inconsistent AUC-minus-bar"):
        elpt._check_endpoints(claim["id"], "delong", claim)


@pytest.mark.parametrize("damage,message", [
    ("no-record", "without its resampling record"),
    ("wrong-method", "names another construction"),
    ("bad-draws", "unusable draw count"),
    ("endpoints-disagree", "disagree with their resampling record"),
    ("other-estimate", "resampled around another cell's estimate"),
    ("excludes-point", "does not contain its own estimate"),
])
def test_the_guard_still_rejects_a_wrong_clustered_interval(damage, message):
    claim = _clustered_claim("live.tau.bar.100.full", "full", -0.098, 0.020)
    if damage == "no-record":
        del claim["clustered_interval"]
    elif damage == "wrong-method":
        claim["clustered_interval"]["axis"] = "run-level sampling"
    elif damage == "bad-draws":
        claim["clustered_interval"]["usable_replicates"] = 20_000
    elif damage == "endpoints-disagree":
        claim["clustered_interval"]["interval_95"] = [-0.098, 0.019]
    elif damage == "other-estimate":
        claim["clustered_interval"]["point"] = -0.02
    else:
        claim["interval"]["low"] = claim["interval"]["high"] = claim["estimate"][
            "difference_a_minus_b"]
        claim["clustered_interval"]["interval_95"] = [claim["interval"]["low"],
                                                      claim["interval"]["high"]]
    with pytest.raises(SystemExit, match=message):
        elpt._check_endpoints(claim["id"], "clustered", claim)


def test_an_unknown_construction_is_refused(records=None):
    record, board = _emitter_fixture()
    record["claims"][0]["interval"]["method"] = "jackknife"
    with pytest.raises(SystemExit, match="not a 95% run-level single-curve interval"):
        elpt.table(record, board)


def test_a_corpus_that_mixes_constructions_is_refused():
    record, board = _emitter_fixture()
    swe = [claim for claim in record["claims"] if claim["id"].startswith("live.swe.bar.")]
    swe[0]["interval"]["method"] = st.TAU_CLUSTER_METHOD
    swe[0]["interval"]["axis"] = st.TAU_CLUSTER_AXIS
    swe[0]["clustered_interval"] = _clustered_claim("x", "y", swe[0]["interval"]["low"],
                                                    swe[0]["interval"]["high"])["clustered_interval"]
    swe[0]["clustered_interval"]["interval_95"] = [swe[0]["interval"]["low"],
                                                   swe[0]["interval"]["high"]]
    swe[0]["clustered_interval"]["point"] = swe[0]["estimate"]["difference_a_minus_b"]
    with pytest.raises(SystemExit, match="mixes interval constructions"):
        elpt.table(record, board)


def _emitter_fixture():
    """Forty cells: SWE-Gym on DeLong, tau-bench on the clustered construction, as shipped."""
    claims, blocks = [], []
    for corpus_index, (corpus, family, _, _) in enumerate(elpt._CORPORA):
        lines = [elpt._BOARD_TITLE, "  method  25%  50%  75%  100%  t2d",
                 "  random  0.500  0.500  0.500  0.500  >100%"]
        for method_index, (method, _) in enumerate(elpt._METHODS[1:]):
            board_values = []
            for prefix_index, prefix in enumerate(elpt._PREFIXES):
                a = 0.640 + 0.01 * corpus_index + 0.002 * method_index + 0.001 * prefix_index
                board_values.append(f"{a:.3f}")
                difference = a - 0.70
                claim = {
                    "id": f"live.{corpus}.bar.{prefix}.{method}", "family": family,
                    "metric": "roc_auc",
                    "estimate": {"a_name": method, "a": a, "b_name": "fixed bar", "b": 0.70,
                                 "difference_a_minus_b": difference},
                    "variance_axes": {"board_point_estimate": {"value": a},
                                      "run_sampling": {"axis": "labeled runs",
                                                       "n": 376 + 284 * corpus_index}},
                }
                if corpus == "tau":
                    low, high = difference - 0.055, difference + 0.049
                    claim["interval"] = {"level": 0.95, "low": low, "high": high,
                                         "method": st.TAU_CLUSTER_METHOD,
                                         "axis": st.TAU_CLUSTER_AXIS}
                    claim["clustered_interval"] = _clustered_claim(
                        claim["id"], method, low, high, point=difference)["clustered_interval"]
                else:
                    claim["interval"] = {"level": 0.95, "low": difference - 0.04,
                                         "high": difference + 0.04,
                                         "method": "single-curve DeLong",
                                         "axis": "run-level sampling"}
                claims.append(claim)
            lines.append("  " + "  ".join([method, *board_values, ">100%"]))
        blocks.append("\n".join(lines))
    return {"claims": claims}, "\n\n".join(blocks) + "\n"


def test_each_caption_describes_its_own_construction_and_evidence():
    block = " ".join(elpt.table(*_emitter_fixture()).split())
    assert block.count("95\\% single-curve DeLong interval for P (run-level sampling axis)") == 1
    assert block.count(f"95\\% {st.TAU_CLUSTER_METHOD} interval for P "
                       f"({st.TAU_CLUSTER_AXIS} axis)") == 1
    assert block.count("376 runs of this corpus carry 376 distinct task identifiers") == 1
    assert block.count("660 runs of this corpus are 165 task instances, each attempted by four "
                       "agent models") == 1
    assert block.count("50 airline, 115 retail") == 1
    assert block.count("percentiles of 10000 such draws (seed 20260907)") == 1
    for forbidden in ("unresolved", "separates", "verdict", "Holm", "\t"):
        assert forbidden not in block


def test_a_record_name_that_latex_would_mangle_is_refused():
    with pytest.raises(SystemExit, match="not printable in a caption"):
        elpt._tex("task_clustered 95% bootstrap")
    assert elpt._tex("task-clustered stratified percentile bootstrap")


# --- the shipped record -------------------------------------------------------------------------

def test_every_shipped_tau_interval_is_clustered_and_swegym_is_untouched():
    record, _ = elpt.load()
    seen = {"tau": set(), "swe": set()}
    for claim in record["claims"]:
        if "threshold" not in claim["family"]:
            continue
        corpus = claim["id"].split(".")[1]
        seen[corpus].add((claim["interval"]["method"], claim["interval"]["axis"]))
    assert seen["swe"] == {("single-curve DeLong", "run-level sampling")}
    assert seen["tau"] == {(st.TAU_CLUSTER_METHOD, st.TAU_CLUSTER_AXIS)}


def test_the_shipped_tau_rows_keep_their_run_level_output_as_a_labelled_diagnostic():
    record, _ = elpt.load()
    tau_rows = [claim for claim in record["claims"]
                if claim["id"].startswith(("live.tau.", "det.tau."))
                and "clustered_interval" in claim]
    assert len(tau_rows) == 26
    for claim in tau_rows:
        diagnostic = claim["reproduction_diagnostic"]
        assert diagnostic["axis"] == "run-level sampling"
        assert diagnostic["role"] == "reproduction diagnostic under the independence assumption"
        assert claim["test"]["assumes_independent_runs"] is True
        # the printed interval is the clustered one, never the diagnostic
        assert [claim["interval"]["low"], claim["interval"]["high"]] != diagnostic["interval_95"]
        assert math.isfinite(diagnostic["p_raw"])


def test_the_shipped_core_contrast_carries_the_clustered_interval():
    record, _ = elpt.load()
    claim = next(c for c in record["claims"] if c["id"] == "det.tau.auditable.vs.size")
    assert claim["estimate"]["difference_a_minus_b"] == pytest.approx(0.046294, abs=5e-6)
    assert claim["interval"]["low"] == pytest.approx(ANCHOR_CONTRAST[0], abs=ANCHOR_TOLERANCE)
    assert claim["interval"]["high"] == pytest.approx(ANCHOR_CONTRAST[1], abs=ANCHOR_TOLERANCE)
    assert claim["reproduction_diagnostic"]["interval_95"][0] == pytest.approx(0.010461, abs=5e-6)


def test_the_shipped_direction_of_the_auditable_full_row_is_unchanged():
    # The one clustered row with no cross-check when it was chosen; the registry's direction is
    # auditable minus full and a sign flip would silently invert the printed claim.
    record, _ = elpt.load()
    claim = next(c for c in record["claims"] if c["id"] == "det.tau.auditable.vs.full")
    assert claim["estimate"]["a_name"] == "auditable (size+deps)"
    assert claim["estimate"]["b_name"] == "full"
    assert claim["estimate"]["difference_a_minus_b"] == pytest.approx(-0.002235, abs=5e-6)
    assert claim["interval"]["low"] < 0 < claim["interval"]["high"]
