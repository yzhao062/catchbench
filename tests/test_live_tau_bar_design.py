r"""The tau-bench bar design calculation: what the solve is, and what it refuses to be.

The calculation answers one question per committed ``live.tau.bar`` cell: holding that cell's observed
effect fixed, how many task clusters would a 95% interval need before it excluded zero. The tempting
neighbour of that question is post-hoc power, which fixes the sample instead and solves for the
effect the achieved interval would have excluded. Part B3 of the declaration forbids it by name, so
the first group of tests below pins the direction of the solve rather than its output: the answer
moves with the effect and with the SD, and does not move at all when the achieved interval is
replaced by nonsense.

The second group binds the shipped record to its sources. Every row's point, SD, cluster count, and
verdict must equal the committed registry row it came from, and every solved count must re-solve from
that row alone, so a regenerated record cannot quietly carry a number the registry does not support.

No corpus, GRADE checkout, or network is needed. The subsampling test builds its own task-by-model
grid, and the record tests read two committed JSON files.
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import live_tau_bar_design as design  # noqa: E402
import statistical_tests as st  # noqa: E402

RECORD = ROOT / "tools" / "live_tau_bar_design_results.json"
REGISTRY = ROOT / "tools" / "statistical_tests_results.json"


@pytest.fixture(scope="module")
def record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def registry():
    return {claim["id"]: claim
            for claim in json.loads(REGISTRY.read_text(encoding="utf-8"))["claims"]
            if claim["id"].startswith("live.tau.bar.")}


def _claim(point, sd, low, high, n_clusters=165):
    """A minimal committed-cell shape, enough for one design row."""
    return {
        "id": "live.tau.bar.100.full",
        "verdict": "does_not_separate" if low < 0 < high else "separates_as_stated",
        "estimate": {"a": 0.70 + point, "b": 0.70},
        "clustered_interval": {
            "point": point, "bootstrap_sd": sd, "interval_95": [low, high],
            "n_clusters": n_clusters, "n_rows": n_clusters * 4, "rows_per_cluster": 4,
            "clusters_per_stratum": {"airline": 50, "retail": 115},
        },
    }


# --- the direction of the solve -----------------------------------------------------------------

def test_the_solve_inverts_its_own_answer():
    # The defining property: at the returned cluster count, the half-width equals the observed
    # effect exactly. Anything else means the solve and the interval disagree about the same design.
    for exponent in (-0.5, -0.46, -0.55):
        for point, sd in ((-0.035, 0.028), (-0.15, 0.03), (0.02, 0.05)):
            solved = design.design_clusters(point, sd, 165, exponent)
            half_width = st.Z975 * sd * (solved / 165) ** exponent
            assert half_width == pytest.approx(abs(point), rel=1e-12)


def test_the_measured_solve_reduces_to_the_closed_form_at_minus_one_half():
    closed = 165 * (st.Z975 * 0.028331 / 0.035318) ** 2
    assert design.design_clusters(-0.035318, 0.028331, 165, -0.5) == pytest.approx(closed, rel=1e-12)


def test_a_shallower_measured_decay_asks_for_more_clusters_than_the_closed_form():
    # An SD that falls more slowly than 1/sqrt(n) needs a larger sample to reach the same half-width,
    # for a cell that has not separated yet. This is the whole reason the exponent is measured.
    at_half = design.design_clusters(-0.035318, 0.028331, 165, -0.5)
    shallower = design.design_clusters(-0.035318, 0.028331, 165, -0.46)
    assert shallower > at_half


def test_the_answer_moves_with_the_effect_and_the_spread_and_nothing_else():
    base = design.design_clusters(-0.04, 0.03, 165, -0.5)
    assert design.design_clusters(-0.08, 0.03, 165, -0.5) < base   # a larger effect resolves sooner
    assert design.design_clusters(-0.04, 0.06, 165, -0.5) > base   # a noisier cell resolves later
    assert design.design_clusters(0.04, 0.03, 165, -0.5) == pytest.approx(base)  # sign is irrelevant


def test_the_achieved_interval_never_reaches_the_answer():
    """The anti-post-hoc-power binding: widen the achieved interval and the design size does not move.

    A minimum-detectable-difference calculation reads the achieved interval and reports the effect it
    would have excluded. If any such reading had crept into the row, replacing the interval with a
    different one would change the emitted count. It does not, because the solve sees only the point,
    the SD, and the cluster count.
    """
    available = {"airline": 50, "retail": 115}
    spec = {"kind": "own", "exponent": -0.5, "source": "test"}
    honest = design.design_row(_claim(-0.0353, 0.0283, -0.0931, 0.0185), spec, available)
    widened = design.design_row(_claim(-0.0353, 0.0283, -0.9000, 0.9000), spec, available)
    assert honest["needed_closed_form"] == widened["needed_closed_form"]
    assert honest["needed_measured"] == widened["needed_measured"]


def test_a_flat_scaling_or_a_zero_effect_is_refused():
    with pytest.raises(ValueError, match="must fall with sample size"):
        design.design_clusters(-0.04, 0.03, 165, 0.0)
    with pytest.raises(ValueError, match="undefined at a zero point"):
        design.design_clusters(0.0, 0.03, 165, -0.5)
    with pytest.raises(ValueError, match="positive bootstrap SD"):
        design.design_clusters(-0.04, 0.0, 165, -0.5)


# --- the ladder and its subsamples ----------------------------------------------------------------

def test_every_split_sums_exactly_and_holds_the_domain_mixture():
    available = {"airline": 50, "retail": 115}
    share = 50 / 165
    for size in (*design.LADDER, 3, 7, 91, 412, 1000):
        split = design.apportion(size, available)
        assert sum(split.values()) == size
        assert abs(split["airline"] - share * size) <= 0.5 + 1e-9


def test_a_ladder_rung_cannot_ask_for_more_clusters_than_the_corpus_holds():
    available = {"airline": 50, "retail": 115}
    for size in design.LADDER:
        assert design.ladder_split(size, available) == design.apportion(size, available)
    with pytest.raises(RuntimeError, match="the corpus has"):
        design.ladder_split(200, available)


def test_a_subsample_keeps_whole_clusters_and_the_rung_mixture():
    # The score vector carries each row's own cluster index, so a subsample that split a task's four
    # attempts, or reordered rows against their labels, shows up as a block whose four scores differ.
    cluster_ids, strata, labels, scores = [], [], [], []
    for index in range(165):
        domain = "airline" if index < 50 else "retail"
        for _ in range(4):
            cluster_ids.append(f"task-{index:03d}")
            strata.append(domain)
            labels.append(int(index % 2))
            scores.append(float(index))
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    clustering = st._tau_clustering(labels, cluster_ids, strata)
    for size in design.LADDER:
        sub_labels, sub_scores, sub_clustering, split = design.subsample_clustering(
            labels, scores, clustering, size, np.random.default_rng(size)
        )
        detail = sub_clustering["record"]
        assert detail["n_clusters"] == size
        assert detail["rows_per_cluster"] == 4
        assert detail["n_rows"] == len(sub_labels) == len(sub_scores) == size * 4
        assert detail["clusters_per_stratum"] == split
        markers = sub_scores[sub_clustering["blocks"]]
        assert np.array_equal(markers.min(axis=1), markers.max(axis=1))  # whole clusters
        assert len(set(markers[:, 0])) == size                           # drawn without replacement
        assert np.array_equal(sub_labels, (sub_scores.astype(int) % 2))  # rows kept their labels
        airline = sum(1 for marker in markers[:, 0] if marker < 50)
        assert airline == split["airline"]


# --- the shipped record ---------------------------------------------------------------------------

def test_the_shipped_record_declares_the_ladder_this_module_still_carries(record):
    frozen = record["declaration"]["frozen_before_the_run"]
    assert tuple(frozen["ladder"]) == design.LADDER
    assert frozen["draws_per_subsample"] == design.SUBSAMPLE_DRAWS
    assert frozen["subsamples_per_rung"] == design.SUBSAMPLE_REPLICATES
    assert frozen["rng_base_seed"] == design.SCALING_SEED == st.TAU_CLUSTER_SEED
    assert frozen["rng_label_template"] == design.SCALING_LABEL
    assert tuple(frozen["cells"]) == tuple(cell for cell, _, _, _ in design.SCALING_CELLS)


def test_every_shipped_row_reproduces_from_its_committed_registry_row(record, registry):
    rows = record["design_table"]
    assert len(rows) == len(registry) == 20
    for row in rows:
        committed = registry[row["id"]]["clustered_interval"]
        assert row["point_auc_minus_bar"] == committed["point"]
        assert row["bootstrap_sd"] == committed["bootstrap_sd"]
        assert row["n_clusters"] == committed["n_clusters"]
        assert row["interval_95"] == committed["interval_95"]
        assert row["verdict"] == registry[row["id"]]["verdict"]
        assert row["separates_at_present_sample"] == (row["verdict"] != "does_not_separate")
        for key, exponent in (("needed_closed_form", -0.5),
                              ("needed_measured", row["needed_measured_exponent"])):
            solved = design.design_clusters(committed["point"], committed["bootstrap_sd"],
                                            committed["n_clusters"], exponent)
            assert row[key]["clusters_exact"] == pytest.approx(solved, rel=1e-12)
            assert row[key]["clusters"] == math.ceil(solved)
            assert row[key]["runs"] == row[key]["clusters"] * committed["rows_per_cluster"]


def test_the_two_unresolved_cells_are_the_ones_the_committed_verdicts_name(record, registry):
    withheld = sorted(claim_id.removeprefix("live.tau.bar.") for claim_id, claim in registry.items()
                      if claim["verdict"] == "does_not_separate")
    assert sorted(record["unresolved_cells"]) == withheld
    assert record["summary"]["not_separating_at_present_sample"] == len(withheld)
    assert record["summary"]["separating_at_present_sample"] == 20 - len(withheld)
    for cell, needed in record["summary"]["needed_for_unresolved_cells"].items():
        assert cell in withheld
        # a cell that has not separated must need more than the sample that failed to separate it
        assert needed["closed_form_clusters"] > record["summary"]["present_clusters"]
        assert needed["measured_clusters"] > record["summary"]["present_clusters"]


def test_the_record_names_which_extrapolation_produced_the_printed_number(record):
    check = record["scaling_check"]
    consistent = [cell["consistent_with_minus_half"] for cell in check["per_cell"]]
    assert check["all_four_consistent_with_minus_half"] == all(consistent)
    assert check["printed_figure_source"] == ("closed_form" if all(consistent) else
                                              "measured_scaling")
    for cell in check["per_cell"]:
        low, high = cell["exponent_interval_95"]
        assert low < cell["exponent"] < high
        assert cell["consistent_with_minus_half"] == (low <= -0.5 <= high)
    # a pooled exponent exists only when the declared agreement rule says the four agree
    pooling = check["pooling"]
    assert (pooling["pooled_exponent"] is not None) == pooling["cells_agree"]
    assert pooling["cells_agree"] == (pooling["every_pair_of_intervals_overlaps"]
                                      and pooling["cochran_q_p"] > pooling["homogeneity_alpha"])


def test_every_scaling_replicate_carries_the_declared_label_and_rung(record):
    for cell in record["scaling_check"]["per_cell"]:
        assert len(cell["replicates"]) == len(design.LADDER) * design.SUBSAMPLE_REPLICATES
        for row in cell["replicates"]:
            assert row["rng_label"] == design.SCALING_LABEL.format(
                cell=cell["cell"], n_clusters=row["n_clusters"], replicate=row["replicate"]
            )
            assert row["n_clusters"] in design.LADDER
            assert 0 <= row["replicate"] < design.SUBSAMPLE_REPLICATES
            assert sum(row["clusters_per_stratum"].values()) == row["n_clusters"]
            assert row["n_rows"] == row["n_clusters"] * 4
            assert row["usable_replicates"] + row["discarded_single_class_draws"] == (
                design.SUBSAMPLE_DRAWS)
            assert row["bootstrap_sd"] > 0


def test_the_replayed_arms_matched_their_committed_cells_within_the_declared_tolerance(record):
    assert len(record["replay_reproduction"]) == len(design.SCALING_CELLS)
    for row in record["replay_reproduction"]:
        assert row["point_abs_difference"] <= row["point_tolerance"] == design.POINT_TOLERANCE
        assert row["bootstrap_sd_abs_difference"] <= row["bootstrap_sd_tolerance"]
        assert row["bootstrap_sd_tolerance"] == design.BOOTSTRAP_SD_TOLERANCE


def test_the_record_states_the_two_quantities_it_refuses_to_compute(record):
    refused = record["not_computed"]
    assert "achieved_power" in refused and "minimum_detectable_difference" in refused
    assert "held fixed" in refused["direction_of_every_solve"]
    # and no emitted row offers one under another name
    text = json.dumps(record["design_table"])
    assert not re.search(r"detectable|achieved_power|post_hoc", text, flags=re.IGNORECASE)
