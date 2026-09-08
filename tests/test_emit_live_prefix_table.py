"""Keep LIVE board scores and pooled-score intervals distinct, and fail on paper drift.

Synthetic records distinguish the run-level interval from both the board score and the CV-seed
interval. Paper fixtures exercise both generated blocks without requiring a manuscript checkout.
The shipped example checks the actual record; the optional paper check follows the transfer
emitter's CATCHBENCH_PAPER_DIR convention.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import emit_live_prefix_table as elpt  # noqa: E402


@pytest.fixture
def records():
    claims, blocks = [], []
    for corpus_index, (corpus, family, _, _) in enumerate(elpt._CORPORA):
        lines = [elpt._BOARD_TITLE, "  method  25%  50%  75%  100%  t2d",
                 "  random  0.500  0.500  0.500  0.500  >100%"]
        for method_index, (method, _) in enumerate(elpt._METHODS[1:]):
            board_values = []
            for prefix_index, prefix in enumerate(elpt._PREFIXES):
                a = 0.640 + 0.01 * corpus_index + 0.002 * method_index + 0.001 * prefix_index
                supervised = method in ("size (flat)", "auditable (size+deps)", "full")
                board_point = a - 0.004 if supervised else a
                board_values.append(f"{board_point:.3f}")
                # One labeled-run count per corpus, which the caption prints; the emitter rejects
                # a corpus whose cells disagree on it, so the fixture keeps them consistent.
                axes = {"board_point_estimate": {"value": board_point},
                        "run_sampling": {"axis": "labeled runs", "n": 376 + 284 * corpus_index}}
                if supervised:
                    axes["cv_seed"] = {
                        "mean_t_interval_95": [board_point - 0.70 - 0.001,
                                               board_point - 0.70 + 0.001],
                    }
                claims.append({
                    "id": f"live.{corpus}.bar.{prefix}.{method}", "family": family,
                    "metric": "roc_auc",
                    "estimate": {"a_name": method, "a": a, "b_name": "fixed bar", "b": 0.70,
                                 "difference_a_minus_b": a - 0.70},
                    "interval": {"axis": "run-level sampling", "method": "single-curve DeLong",
                                 "level": 0.95, "low": a - 0.70 - 0.04,
                                 "high": a - 0.70 + 0.04},
                    "variance_axes": axes,
                })
            lines.append("  " + "  ".join([method, *board_values, ">100%"]))
        blocks.append("\n".join(lines))
    return {"claims": claims}, "\n\n".join(blocks) + "\n"


@pytest.fixture
def paper(tmp_path, records):
    (tmp_path / "09_appendix.tex").write_text(
        "\\section{Before}\n" + elpt.table(*records) + "\n\\section{After}\n", encoding="utf-8")
    return tmp_path


def test_board_scores_keep_the_pooled_interval_and_run_sampling_axis(records):
    record, board = records
    board_point, interval_point, low, high = elpt.intervals(record)["swe", "size (flat)", 25]
    assert (board_point, interval_point, low, high) == pytest.approx(
        (0.636, 0.640, 0.600, 0.680))
    assert (low + high) / 2 == pytest.approx(record["claims"][0]["estimate"]["a"])
    assert interval_point == pytest.approx((low + high) / 2)
    assert board_point != pytest.approx((low + high) / 2)
    block = elpt.table(record, board)
    assert r"\shortstack{B: 0.636\\P: 0.640\\$[0.600, 0.680]$}" in block
    # B is the board score and P is the estimate the interval is for. The swapped cell is the
    # error worth pinning, and 0.640 is a legitimate board score elsewhere in the fixture, so
    # this compares the whole cell rather than the B field alone. The endpoints stay shifted
    # rather than recentred on either estimate.
    assert r"\shortstack{B: 0.640\\P: 0.636\\$[0.600, 0.680]$}" not in block
    assert r"P: 0.636\\$[0.600, 0.680]$" not in block
    assert "[0.596, 0.676]" not in block
    assert "[0.635, 0.637]" not in block


def test_claim_order_and_other_families_do_not_change_the_table(records):
    record, board = records
    expected = elpt.table(record, board)
    record["claims"].reverse()
    record["claims"].append({"family": "other_auc", "id": "irrelevant"})
    assert elpt.table(record, board) == expected
    # dep-span is unsupervised, so its board score and its interval estimate coincide.
    assert elpt.intervals(record)["tau", "dep-span (online)", 100] == pytest.approx(
        (0.661, 0.661, 0.621, 0.701))


def test_both_tables_keep_all_rows_columns_and_the_estimand_guard(records):
    block = elpt.table(*records)
    assert block.count(r"\shortstack{") == 40
    assert block.count(r"Method & 25\% & 50\% & 75\% & 100\% & t2d") == 2
    assert block.count("random & 0.500 & 0.500 & 0.500 & 0.500 & none") == 2
    for _, name in elpt._METHODS[1:]:
        assert block.splitlines().count(name) == 2
    caption_text = " ".join(block.split())
    for phrase in ("run-level sampling axis", "It is not a comparison",
                   "discards the pairing a paired test keeps",
                   "overlap is therefore not evidence that two entrants perform alike",
                   "labeled runs of this corpus", "seed-averaged out-of-fold",
                   "board point estimates only; no interval is recorded",
                   "B is mean fold AUC and P is the AUC of seed-averaged",
                   "an unadjusted 95\\% single-curve DeLong interval for P",
                   "no simultaneous",
                   "coverage across methods or prefixes",
                   "whose board ROC-AUC reaches",
                   f"absolute gap is at most {elpt.MAX_CENTRE_GAP:.3f}",
                   r"Appendix~\ref{app:board-values}"):
        assert caption_text.count(phrase) == 2
    for forbidden in ("unresolved", "separates", "verdict", "Holm", "p-value", r"\textbf", "\t"):
        assert forbidden not in block


@pytest.mark.parametrize("damage", ["missing", "duplicate", "unknown"])
def test_an_incomplete_or_ambiguous_cell_set_fails(records, damage):
    record, board = records
    if damage == "missing":
        record["claims"].pop()
    elif damage == "duplicate":
        record["claims"].append(record["claims"][0])
    else:
        record["claims"][0]["id"] = "live.swe.bar.25.random"
    with pytest.raises(SystemExit, match="threshold cell"):
        elpt.table(record, board)


@pytest.mark.parametrize("section,key,value", [
    ("estimate", "b", 0.71),
    ("estimate", "b_name", "another arm"),
    ("estimate", "a_name", "random"),
    ("interval", "axis", "cross-validation split seed"),
    ("interval", "method", "paired DeLong"),
    ("interval", "level", 0.90),
], ids=["bar", "comparator", "arm", "axis", "method", "level"])
def test_an_interval_with_a_different_estimand_fails(records, section, key, value):
    record, board = records
    record["claims"][0][section][key] = value
    with pytest.raises(SystemExit, match="not a 95% run-level single-curve interval"):
        elpt.table(record, board)


@pytest.mark.parametrize("damage", ["nonfinite", "reversed", "uncentered"])
def test_an_invalid_interval_fails(records, damage):
    record, board = records
    interval = record["claims"][0]["interval"]
    if damage == "nonfinite":
        interval["low"] = float("nan")
    elif damage == "reversed":
        interval["low"], interval["high"] = interval["high"], interval["low"]
    else:
        interval["high"] += 0.01
    with pytest.raises(SystemExit, match="inconsistent AUC-minus-bar"):
        elpt.table(record, board)


@pytest.mark.parametrize("point", [float("nan"), float("inf"), -0.001, 1.001],
                         ids=["nan", "infinite", "negative", "above-one"])
def test_an_invalid_board_point_fails(records, point):
    record, _ = records
    record["claims"][0]["variance_axes"]["board_point_estimate"]["value"] = point
    with pytest.raises(SystemExit, match="invalid board point estimate"):
        elpt.intervals(record)


@pytest.mark.parametrize("direction", [-1, 1], ids=["below", "above"])
@pytest.mark.parametrize("gap", [0.0079, 0.0081], ids=["inside", "outside"])
def test_the_centre_gap_bound_is_enforced(records, direction, gap):
    record, board = records
    claim = record["claims"][0]
    point = claim["estimate"]["a"] + direction * gap
    claim["variance_axes"]["board_point_estimate"]["value"] = point
    board = board.replace("  size (flat)  0.636 ", f"  size (flat)  {point:.3f} ", 1)
    elpt._board_rows(board, record)
    if gap > elpt.MAX_CENTRE_GAP:
        with pytest.raises(SystemExit, match="centre gap 0.008100 exceeds MAX_CENTRE_GAP=0.008"):
            elpt.table(record, board)
    else:
        assert (r"\shortstack{B: %.3f\\P: 0.640\\$[0.600, 0.680]$}" % point
                in elpt.table(record, board))


def test_the_caption_and_guard_share_the_bound(records, monkeypatch):
    monkeypatch.setattr(elpt, "MAX_CENTRE_GAP", 0.006)
    assert "absolute gap is at most 0.006" in " ".join(elpt.table(*records).split())
    monkeypatch.setattr(elpt, "MAX_CENTRE_GAP", 0.003)
    with pytest.raises(SystemExit, match="MAX_CENTRE_GAP=0.003"):
        elpt.table(*records)


def test_swapped_anonymous_board_blocks_fail(records):
    record, board = records
    with pytest.raises(SystemExit, match="board point estimate disagrees"):
        elpt.table(record, "\n\n".join(reversed(board.strip().split("\n\n"))) + "\n")


def test_a_missing_random_row_fails(records):
    record, board = records
    board = board.replace("  random  0.500  0.500  0.500  0.500  >100%\n", "", 1)
    with pytest.raises(SystemExit, match="unexpected LIVE prefix board methods"):
        elpt.table(record, board)


def test_a_matching_paper_passes_with_crlf_and_unrelated_text(paper, records, capsys):
    path = paper / "09_appendix.tex"
    text = path.read_text(encoding="utf-8").replace("Before", "Before 123")
    path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    assert elpt.check(paper, elpt.table(*records)) == 0
    assert "paper is current" in capsys.readouterr().out


@pytest.mark.parametrize("damage", ["swe-point", "tau-interval", "t2d", "caption", "deleted-line"])
def test_changed_content_fails_and_prints_the_delta(paper, records, capsys, damage):
    path = paper / "09_appendix.tex"
    text = path.read_text(encoding="utf-8")
    if damage == "swe-point":
        altered = text.replace(r"\shortstack{B: 0.636", r"\shortstack{B: 0.936", 1)
    elif damage == "tau-interval":
        start = text.index(elpt._markers("tab:live-stream-tau")[0])
        altered = text[:start] + text[start:].replace("[0.610, 0.690]", "[0.910, 0.690]", 1)
    elif damage == "t2d":
        altered = text.replace(" & none", r" & 25\%", 1)
    elif damage == "caption":
        altered = text.replace("run-level sampling", "cross-validation seed", 1)
    else:
        altered = text.replace("0.70 (``none'' if no prefix does). Prefix fractions are relative "
                               "to eventual trace length.}\n", "", 1)
    assert altered != text
    path.write_text(altered, encoding="utf-8")
    assert elpt.check(paper, elpt.table(*records)) == 1
    output = capsys.readouterr().out
    label = "tab:live-stream-tau" if damage == "tau-interval" else "tab:live-stream"
    assert f"--- 09_appendix.tex:{label}" in output
    assert f"+++ generated:{label}" in output
    assert "@@" in output


@pytest.mark.parametrize("damage", ["missing", "duplicate", "commented", "reversed", "overlap"])
def test_malformed_markers_fail(paper, records, capsys, damage):
    path = paper / "09_appendix.tex"
    text = path.read_text(encoding="utf-8")
    begin, end = elpt._markers("tab:live-stream-tau")
    if damage == "missing":
        text = text.replace(end, "")
    elif damage == "duplicate":
        text = text.replace(begin, begin + "\n" + begin)
    elif damage == "commented":
        text = text.replace(begin, "% " + begin)
    elif damage == "reversed":
        text = text.replace(end, "").replace(begin, end + "\n" + begin)
    else:
        swe_end = elpt._markers("tab:live-stream")[1]
        text = text.replace(swe_end + "\n", "").replace(end, swe_end + "\n" + end)
    path.write_text(text, encoding="utf-8")
    assert elpt.check(paper, elpt.table(*records)) == 1
    assert "STALE" in capsys.readouterr().out


def test_a_missing_appendix_fails(tmp_path, records, capsys):
    assert elpt.check(tmp_path, elpt.table(*records)) == 1
    assert "09_appendix.tex is missing" in capsys.readouterr().out


def test_the_shipped_records_shift_the_worked_example_and_preserve_t2d():
    record, board = elpt.load()
    cells = elpt.intervals(record)
    assert len(cells) == 40
    assert cells["swe", "size (flat)", 25] == pytest.approx(
        (0.6294377479973048, 0.6367558850158442, 0.5774042873280869, 0.6961074827036017),
        abs=1e-14, rel=0)
    rows = elpt._board_rows(board, record)
    for claim in record["claims"]:
        if "threshold" not in claim["family"]:
            continue
        _, corpus, _, prefix, method = claim["id"].split(".", 4)
        board_point, interval_point, low, high = cells[corpus, method, int(prefix)]
        assert board_point == claim["variance_axes"]["board_point_estimate"]["value"]
        assert interval_point == claim["estimate"]["a"]
        assert f"{board_point:.3f}" == rows[corpus][method][elpt._PREFIXES.index(int(prefix))]
        assert (low, high) == pytest.approx(
            (claim["interval"]["low"] + 0.70, claim["interval"]["high"] + 0.70),
            abs=1e-14, rel=0)
    block = elpt.table(record, board)
    assert "random & 0.483 & 0.483 & 0.483 & 0.483 & none" in block
    assert "random & 0.498 & 0.498 & 0.498 & 0.498 & none" in block
    assert block.splitlines().count(r"  & 25\% \\") == 3
    assert block.splitlines().count(r"  & none \\") == 7


def test_no_flags_print_both_complete_tables_without_a_paper(tmp_path):
    env = dict(os.environ)
    env.pop("CATCHBENCH_PAPER_DIR", None)
    done = subprocess.run([sys.executable, "-S", str(elpt.ROOT / "emit_live_prefix_table.py")],
                          cwd=tmp_path, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout == elpt.table(*elpt.load()) + "\n"
    assert done.stderr == ""


def test_check_cli_rejects_a_changed_board_score(tmp_path):
    generated = elpt.table(*elpt.load())
    altered = generated.replace(r"\shortstack{B: 0.629", r"\shortstack{B: 0.937", 1)
    assert altered != generated
    (tmp_path / "09_appendix.tex").write_text(altered, encoding="utf-8")
    done = subprocess.run([sys.executable, "-S", str(elpt.ROOT / "emit_live_prefix_table.py"),
                           "--check", "--paper", str(tmp_path)],
                          cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 1
    assert "STALE 09_appendix.tex: tab:live-stream differs" in done.stdout
    assert r"-  & \shortstack{B: 0.937\\P: 0.637\\$[0.577, 0.696]$}" in done.stdout
    assert r"+  & \shortstack{B: 0.629\\P: 0.637\\$[0.577, 0.696]$}" in done.stdout
    assert done.stderr == ""


@pytest.mark.parametrize("explicit", [False, True], ids=["environment", "explicit-overrides-env"])
def test_check_accepts_the_paper_option_or_environment(paper, records, monkeypatch, capsys, explicit):
    monkeypatch.setattr(elpt, "load", lambda: records)
    monkeypatch.setenv("CATCHBENCH_PAPER_DIR", str(paper / "absent" if explicit else paper))
    monkeypatch.setattr(sys, "argv", ["emit_live_prefix_table.py", "--check"]
                        + (["--paper", str(paper)] if explicit else []))
    assert elpt.main() == 0
    assert "paper is current" in capsys.readouterr().out


def test_check_requires_a_paper_location(monkeypatch, capsys):
    monkeypatch.delenv("CATCHBENCH_PAPER_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", ["emit_live_prefix_table.py", "--check"])
    with pytest.raises(SystemExit) as error:
        elpt.main()
    assert error.value.code == 2
    assert "--check needs --paper" in capsys.readouterr().err


def test_shipped_records_match_configured_paper():
    configured = os.environ.get("CATCHBENCH_PAPER_DIR")
    if not configured:
        pytest.skip("set CATCHBENCH_PAPER_DIR to run the cross-repository integration check")
    assert elpt.check(Path(configured), elpt.table(*elpt.load())) == 0
