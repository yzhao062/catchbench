"""The transfer-diagnostics generator must reproduce the record, and its checker must catch drift.

Section 5.6's numbers were typed from a console session and nothing recomputed them, which is how
the Spearman correlation came to be quoted from the five-seed record inside a paragraph that says
twenty seeds. Each test here pins one part of the contract that stops that recurring: the arithmetic
matches the committed arrays, the plus-or-minus column reports the record's own spread rather than a
silently different convention, and every corruption of the printed block is a failure.

The fixtures build a minimal seed record rather than copying the shipped one, so a legitimate rerun
of the seed sweep cannot turn these red. The real pair is checked separately, behind an environment
variable, so the unit suite carries no hidden dependency on a sibling checkout.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import emit_transfer_table as ett  # noqa: E402


def _series(values):
    """A seed-record node. ``std`` is the population form, which is what the tool must report."""
    import numpy as np
    array = np.asarray(values, dtype=float)
    return {"values": list(values), "mean": float(array.mean()), "std": float(array.std()),
            "min": float(array.min()), "max": float(array.max())}


def _detector(scores, matched=None, spearman=None):
    node = {"roc_auc": _series(scores)}
    if matched is not None:
        node["exact_size_matched_auc"] = _series(matched)
        node["exact_size_match"] = {"run_count": 7, "stratum_count": 3, "pair_count": 11,
                                    "sizes": [1, 2, 3]}
    if spearman is not None:
        node["score_size_spearman"] = _series(spearman)
    return node


GAAN = [0.70, 0.75, 0.80, 0.85]
SUPERVISED = [0.90, 0.91, 0.92]


def _twenty():
    detectors = {
        "pygod (graph AD)": _detector([0.50, 0.60, 0.70, 0.80]),
        "pygod-anomalydae": _detector([0.40, 0.45, 0.50, 0.55]),
        "pygod-conad": _detector([0.55, 0.60, 0.65, 0.70]),
        "pygod-gaan": _detector(GAAN, matched=[0.60, 0.62, 0.64, 0.66],
                                spearman=[-0.70, -0.72, -0.74, -0.76]),
    }
    tau = {name: _detector([0.50, 0.51, 0.52, 0.53]) for name in detectors}
    return {"detection": {"swegym": {"detectors": detectors, "runs": 40},
                          "tau": {"detectors": tau, "runs": 60}}}


def _five():
    return {
        "leaders": {"swegym": {"g-safeguard (sup GNN)": {"roc_auc": _series(SUPERVISED)}}},
        "detection": {"swegym": {"detectors": {
            "pygod-gaan": _detector(GAAN, spearman=[-0.80, -0.82])}}},
    }


@pytest.fixture
def records():
    return _five(), _twenty()


@pytest.fixture
def paper(tmp_path, records):
    (tmp_path / "09_appendix.tex").write_text(
        "\\section{Full Board Values}\n\n" + ett.table(*records) + "\n\n\\section{Next}\n",
        encoding="utf-8")
    return tmp_path


# --- the arithmetic ------------------------------------------------------------------------------


def test_the_plus_or_minus_column_reports_the_records_own_spread(records):
    """The body prints the record's std; a sample std here would disagree in the third decimal."""
    import numpy as np
    block = ett.table(*records)
    population = np.asarray(GAAN).std()
    sample = np.asarray(GAAN).std(ddof=1)
    assert "%.3f" % population != "%.3f" % sample, "the fixture must distinguish the two"
    assert "GAAN & %.3f & %.3f &" % (np.mean(GAAN), population) in block
    assert "GAAN & %.3f & %.3f &" % (np.mean(GAAN), sample) not in block


def test_the_interval_uses_the_sample_form(records):
    """The interval is the ordinary t interval, so its half-width exceeds the record's std path."""
    import numpy as np
    from scipy import stats
    diagnostics = ett.diagnostics(*records)
    gaan = [row for row in diagnostics["detectors"]["swegym"] if row["name"] == "GAAN"][0]
    n = len(GAAN)
    half = stats.t.ppf(0.975, n - 1) * np.std(GAAN, ddof=1) / np.sqrt(n)
    assert gaan["low"] == pytest.approx(np.mean(GAAN) - half)
    assert gaan["high"] == pytest.approx(np.mean(GAAN) + half)


def test_welch_matches_scipy(records):
    from scipy import stats
    result = ett.welch(GAAN, SUPERVISED)
    reference = stats.ttest_ind(GAAN, SUPERVISED, equal_var=False)
    assert result["p"] == pytest.approx(float(reference.pvalue))
    assert result["df"] == pytest.approx(float(reference.df))
    assert result["difference"] == pytest.approx(
        sum(GAAN) / len(GAAN) - sum(SUPERVISED) / len(SUPERVISED))


def test_the_welch_interval_brackets_its_difference(records):
    result = ett.welch(GAAN, SUPERVISED)
    assert result["low"] < result["difference"] < result["high"]


def test_the_spearman_row_names_both_records(records):
    """The stale value this tool exists to prevent was the five-seed number in a twenty-seed line."""
    block = ett.table(*records)
    assert "$-0.730$ over 4 seeds (the five-seed record reads $-0.810$)" in block


def test_the_supervised_reference_comes_from_the_five_seed_record(records):
    diagnostics = ett.diagnostics(*records)
    assert diagnostics["supervised"]["n"] == len(SUPERVISED)


def test_no_generated_line_carries_a_literal_tab(records):
    assert "\t" not in ett.table(*records)


# --- refusing to guess ---------------------------------------------------------------------------


def test_a_missing_detector_is_a_hard_failure(records):
    five, twenty = records
    del twenty["detection"]["swegym"]["detectors"]["pygod-conad"]
    with pytest.raises(SystemExit):
        ett.table(five, twenty)


def test_an_empty_values_array_is_a_hard_failure(records):
    five, twenty = records
    twenty["detection"]["swegym"]["detectors"]["pygod-gaan"]["roc_auc"]["values"] = []
    with pytest.raises(SystemExit):
        ett.table(five, twenty)


def test_a_missing_supervised_record_is_a_hard_failure(records):
    five, twenty = records
    del five["leaders"]["swegym"]["g-safeguard (sup GNN)"]
    with pytest.raises(SystemExit):
        ett.table(five, twenty)


# --- checking the paper --------------------------------------------------------------------------


def test_a_matching_paper_passes(paper, records):
    assert ett.check(paper, ett.table(*records)) == 0


def test_an_altered_value_fails(paper, records):
    path = paper / "09_appendix.tex"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("0.775", "0.999", 1), encoding="utf-8")
    assert ett.check(paper, ett.table(*records)) == 1


def test_a_deleted_row_fails(paper, records):
    path = paper / "09_appendix.tex"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines()
             if not line.startswith(" & CONAD &")]
    path.write_text("\n".join(lines), encoding="utf-8")
    assert ett.check(paper, ett.table(*records)) == 1


def test_a_missing_marker_fails(paper, records):
    path = paper / "09_appendix.tex"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(ett._END, ""), encoding="utf-8")
    assert ett.check(paper, ett.table(*records)) == 1


def test_a_commented_out_marker_fails(paper, records):
    path = paper / "09_appendix.tex"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(ett._BEGIN, "% " + ett._BEGIN), encoding="utf-8")
    assert ett.check(paper, ett.table(*records)) == 1


def test_a_duplicated_marker_fails(paper, records):
    path = paper / "09_appendix.tex"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(ett._BEGIN, ett._BEGIN + "\n" + ett._BEGIN, 1), encoding="utf-8")
    assert ett.check(paper, ett.table(*records)) == 1


def test_a_missing_appendix_fails(tmp_path, records):
    assert ett.check(tmp_path, ett.table(*records)) == 1


# --- the shipped records -------------------------------------------------------------------------


def test_the_shipped_records_still_carry_every_key_the_tool_reads():
    """A rerun of the seed sweep that drops a key must fail here, not in the manuscript."""
    five, twenty = ett.load()
    block = ett.table(five, twenty)
    assert "GAAN" in block and "g-safeguard" in block


def test_the_shipped_twenty_seed_record_really_has_twenty_seeds():
    _, twenty = ett.load()
    values = twenty["detection"]["swegym"]["detectors"]["pygod-gaan"]["roc_auc"]["values"]
    assert len(values) == 20


def test_shipped_records_match_configured_paper():
    """Cross-repository closure, opt-in, using the same variable the CLI honors."""
    configured = os.environ.get("CATCHBENCH_PAPER_DIR")
    if not configured:
        pytest.skip("set CATCHBENCH_PAPER_DIR to run the cross-repository integration check")
    five, twenty = ett.load()
    assert ett.check(Path(configured), ett.table(five, twenty)) == 0
