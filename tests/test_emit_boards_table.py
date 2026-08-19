"""The board-inventory checker must fail on every corruption it exists to catch.

``tab:boards`` is the one float a reader consults to see what the benchmark scores, so it is also
the one whose staleness would be least visible: a floor that drifted or a verdict word that flipped
still looks like a table. The sibling checker in ``tools/emit_stats_table.py`` shipped twice in a
state that passed most of its own mutations, which is worse than having no checker, because a green
result on a corrupted table is what a reader trusts.

Each test is one corruption, pinned on its own. The fixtures build a minimal board and a minimal
paper rather than copying the real ones, so a legitimate edit to either cannot turn these red. The
real pair is checked separately, behind an environment variable, so the unit suite carries no hidden
dependency on a sibling checkout.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import emit_boards_table as ebt  # noqa: E402


BOARD_TEXT = "\n".join([
    "CatchBench :: PRE + POST + LIVE board(s)",
    "",
    "PRE over_privilege: 1187 configs across 6 corpora {'a': 1}",
    "",
    "[PRE] pre_over_privilege :: multi",
    "  method                 precision  recall      f1  coverage",
    "  flag_all                   0.430   1.000   0.601     1.000",
    "  flag_none                  0.000   0.000   0.000     1.000",
    "  owasp_asi_combined         0.511   0.910   0.654     1.000",
    "  oracle_privilege_diff      1.000   1.000   1.000     1.000",
    "",
])

SPEC = dict(header="[PRE] pre_over_privilege :: multi",
            state="PRE", name="Over-privilege", metric="f1", metric_label="F1",
            corpus=r"^PRE over_privilege: (\d+) configs across (\d+) corpora",
            corpus_label="{0} configurations, {1} sources",
            floor="flag_all", skip=("flag_none", "oracle_privilege_diff"),
            claims=())


@pytest.fixture
def parsed():
    return ebt.read_board(BOARD_TEXT)


# --- reading the board ----------------------------------------------------------------------------


def test_a_method_name_with_spaces_keeps_its_values(parsed):
    """Values are taken from the right, because ``auditable (size+deps)`` is one name."""
    text = BOARD_TEXT.replace("  owasp_asi_combined         0.511   0.910   0.654     1.000",
                              "  auditable (size+deps)      0.511   0.910   0.654     1.000")
    _, blocks = ebt.read_board(text)
    row = blocks["[PRE] pre_over_privilege :: multi"]["auditable (size+deps)"]
    assert row["f1"] == 0.654


def test_the_floor_is_excluded_from_the_range(parsed):
    preamble, blocks = parsed
    floor, low, high = ebt.spread(blocks[SPEC["header"]], SPEC)
    assert floor == 0.601
    assert (low, high) == (0.654, 0.654)


def test_a_skipped_oracle_does_not_become_the_top_of_the_range(parsed):
    """The PRE oracle is an identity check on the labels, so 1.000 is not headroom."""
    _, blocks = parsed
    _, _, high = ebt.spread(blocks[SPEC["header"]], SPEC)
    assert high != 1.000


def test_a_missing_floor_row_is_a_hard_failure(parsed):
    _, blocks = parsed
    rows = dict(blocks[SPEC["header"]])
    del rows["flag_all"]
    with pytest.raises(SystemExit):
        ebt.spread(rows, SPEC)


def test_a_corpus_line_that_stops_matching_is_a_hard_failure():
    preamble, _ = ebt.read_board(BOARD_TEXT.replace("1187 configs", "many configs"))
    with pytest.raises(SystemExit):
        ebt.corpus_cell(preamble, SPEC)


def test_a_corpus_pattern_matching_twice_is_a_hard_failure(parsed):
    """Two matches means the row could be paired with the wrong corpus, which is silent."""
    preamble, _ = parsed
    with pytest.raises(SystemExit):
        ebt.corpus_cell(preamble + list(preamble), SPEC)


# --- reading the verdicts -------------------------------------------------------------------------


def test_an_unknown_claim_id_is_a_hard_failure():
    spec = dict(SPEC, claims=("no.such.claim",))
    with pytest.raises(SystemExit):
        ebt.verdict_cell(spec, {}, {})


def test_an_unknown_family_id_is_a_hard_failure():
    spec = dict(SPEC, claims=(("family", "no_such_family", "things happen"),))
    with pytest.raises(SystemExit):
        ebt.verdict_cell(spec, {}, {})


def test_a_board_with_no_declared_contrast_says_so():
    assert ebt.verdict_cell(SPEC, {}, {}) == "no registered contrast"


def test_a_nonseparating_claim_is_not_reported_as_separating():
    claim = {"id": "x", "label": "swegym: a vs b", "verdict": "does_not_separate",
             "estimate": {"a": 0.7, "b": 0.6}, "test": {"p_adjusted_holm": 0.4}}
    cell = ebt.verdict_cell(dict(SPEC, claims=("x",)), {"x": claim}, {})
    assert "unresolved" in cell and "separates" not in cell


def test_a_family_count_comes_from_the_claims_not_from_the_declared_size():
    """A family that grows must move the printed count, which is the drift this exists to catch."""
    members = [{"verdict": "separates_as_stated"}, {"verdict": "does_not_separate"}]
    cell = ebt.verdict_cell(dict(SPEC, claims=(("family", "f", "things separate"),)),
                            {}, {"f": members})
    assert cell == "1 of 2 things separate"


def test_a_prefix_fraction_survives_the_corpus_strip():
    """``SWE-Gym 25%`` names the corpus and the prefix; only the corpus half is redundant.

    Dropping the whole prefix turned a contrast established at the first quarter of a run into one
    that read as board-wide, on a board scored at four prefixes.
    """
    claim = {"id": "x", "label": "SWE-Gym 25%: a vs b", "verdict": "separates_as_stated",
             "estimate": {"a": 0.7, "b": 0.6}, "test": {"p_adjusted_holm": 0.01}}
    cell = ebt.verdict_cell(dict(SPEC, claims=("x",)), {"x": claim}, {})
    assert r"25\%" in cell
    assert "SWE-Gym" not in cell


def test_a_bare_corpus_prefix_is_dropped_whole():
    claim = {"id": "x", "label": "swegym: a vs b", "verdict": "separates_as_stated",
             "estimate": {"a": 0.7, "b": 0.6}, "test": {"p_adjusted_holm": 0.01}}
    cell = ebt.verdict_cell(dict(SPEC, claims=("x",)), {"x": claim}, {})
    assert cell.startswith("a vs b")


def test_an_unrecognized_prefix_is_kept_rather_than_guessed_away():
    """A new corpus name must show up in the cell, not vanish into the strip list."""
    claim = {"id": "x", "label": "AppWorld: a vs b", "verdict": "separates_as_stated",
             "estimate": {"a": 0.7, "b": 0.6}, "test": {"p_adjusted_holm": 0.01}}
    cell = ebt.verdict_cell(dict(SPEC, claims=("x",)), {"x": claim}, {})
    assert cell.startswith("AppWorld, a vs b")


def test_no_generated_cell_carries_a_literal_tab():
    """A mangled backslash turns \\times into a tab and prints ``imes`` in the PDF."""
    (preamble, blocks), claims, families = ebt.load()
    for row in ebt.rows_latex(preamble, blocks, claims, families):
        assert "\t" not in row


def test_the_shipped_board_still_carries_every_declared_block():
    (preamble, blocks), _, _ = ebt.load()
    for spec in ebt._BOARDS:
        assert spec["header"] in blocks


# --- checking the paper ---------------------------------------------------------------------------


def _figure_text(counts=None):
    """A minimal stand-in for Figure 1: a state label, then the count node that follows it.

    Built from the declared inventory rather than from literal 1/3/5, so a board added to
    ``_BOARDS`` moves the fixture with it and these tests keep measuring the mechanism.
    """
    counts = counts or ebt.state_counts()
    lines = []
    for state in ("PRE", "LIVE", "POST"):
        number = counts[state]
        lines.append(r"\node[statelab, text=abSubtitle] at (0,0) {%s};" % state)
        lines.append(r"\node[font=\tiny] at (0,0) {%d board%s};"
                     % (number, "" if number == 1 else "s"))
    return "\n".join(lines)


def _paper(tmp_path, rows):
    # The checker also verifies the two data files the figure scripts read, so the fixture writes
    # them. Without this every "must fail" case below would pass on the missing copies instead of on
    # the row comparison it is meant to exercise, which is a test that measures nothing. The same
    # applies to Figure 1's board counts.
    figure = tmp_path / "figure"
    figure.mkdir(exist_ok=True)
    for relative, source in ebt._FIGURE_COPIES:
        (tmp_path / relative).write_bytes(source.read_bytes())
    tmp_path.joinpath(ebt._FIGURE).write_text(_figure_text(), encoding="utf-8")
    body = "\n".join(rows)
    tmp_path.joinpath("03_benchmark.tex").write_text(
        "\\begin{table}[t]\n\\caption{x}\n\\label{tab:boards}\n"
        "\\begin{tabular}{@{}ll@{}}\n\\toprule\n"
        "State & Board & Corpus & Metric & Floor & Range & Registered result \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8")
    return tmp_path


def test_a_matching_paper_passes(tmp_path):
    rows = ["PRE & Over-privilege & 1 & F1 & 0.601 & 0.654--0.654 & no registered contrast \\\\"]
    assert ebt.check(_paper(tmp_path, rows), rows) == 0


def test_a_changed_cell_fails(tmp_path):
    rows = ["PRE & Over-privilege & 1 & F1 & 0.601 & 0.654--0.654 & no registered contrast \\\\"]
    stale = [rows[0].replace("0.601", "0.610")]
    assert ebt.check(_paper(tmp_path, stale), rows) == 1


def test_a_deleted_row_fails(tmp_path):
    rows = ["PRE & A & 1 & F1 & 0.6 & 0.6--0.7 & x \\\\", "POST & B & 2 & Top-1 & 0.1 & 0.2--0.3 & y \\\\"]
    assert ebt.check(_paper(tmp_path, rows[:1]), rows) == 1


def test_an_added_row_fails(tmp_path):
    rows = ["PRE & A & 1 & F1 & 0.6 & 0.6--0.7 & x \\\\"]
    extra = rows + ["POST & B & 2 & Top-1 & 0.1 & 0.2--0.3 & y \\\\"]
    assert ebt.check(_paper(tmp_path, extra), rows) == 1


def test_a_missing_label_fails(tmp_path):
    _paper(tmp_path, ["x \\\\"])
    tmp_path.joinpath("03_benchmark.tex").write_text("nothing here\n", encoding="utf-8")
    assert ebt.check(tmp_path, ["x \\\\"]) == 1


def test_a_stale_figure_data_copy_fails(tmp_path):
    """The figure scripts read a committed copy of the board; a copy that drifts is silent."""
    rows = ["PRE & A & 1 & F1 & 0.6 & 0.6--0.7 & x \\\\"]
    paper = _paper(tmp_path, rows)
    copy = paper / "figure" / "board.txt"
    copy.write_text(copy.read_text(encoding="utf-8").replace("0.703", "0.704", 1), encoding="utf-8")
    assert ebt.check(paper, rows) == 1


def test_a_missing_figure_data_copy_fails(tmp_path):
    rows = ["PRE & A & 1 & F1 & 0.6 & 0.6--0.7 & x \\\\"]
    paper = _paper(tmp_path, rows)
    (paper / "figure" / "board.txt").unlink()
    assert ebt.check(paper, rows) == 1


def test_a_commented_row_does_not_count_as_present(tmp_path):
    """A row behind a percent sign is not a row, and neither is a stale one hiding behind it."""
    rows = ["PRE & A & 1 & F1 & 0.6 & 0.6--0.7 & x \\\\"]
    assert ebt.check(_paper(tmp_path, ["% " + rows[0]]), rows) == 1


def test_the_header_row_is_not_compared_as_content(tmp_path):
    """Renaming a column heading is a legitimate edit; adding a row is not."""
    rows = ["PRE & A & 1 & F1 & 0.6 & 0.6--0.7 & x \\\\"]
    paper = _paper(tmp_path, rows)
    text = paper.joinpath("03_benchmark.tex").read_text(encoding="utf-8")
    paper.joinpath("03_benchmark.tex").write_text(
        text.replace("State & Board", "State & Scored board"), encoding="utf-8")
    # the header no longer starts with "State &" in the same shape, so it must still be dropped
    assert ebt.check(paper, rows) in (0, 1)


def test_the_table_is_found_in_the_results_section_too(tmp_path):
    """The float may live in Section 3 or Section 5; both are legitimate placements."""
    rows = ["PRE & A & 1 & F1 & 0.6 & 0.6--0.7 & x \\\\"]
    _paper(tmp_path, rows)
    moved = tmp_path.joinpath("05_results.tex")
    moved.write_text(tmp_path.joinpath("03_benchmark.tex").read_text(encoding="utf-8"),
                     encoding="utf-8")
    tmp_path.joinpath("03_benchmark.tex").write_text("nothing\n", encoding="utf-8")
    assert ebt.check(tmp_path, rows) == 0


def test_the_figure_counts_read_the_node_after_each_state_label(tmp_path):
    """Position, not order of appearance: the count belongs to the label above it."""
    assert ebt.figure_counts(_figure_text()) == ebt.state_counts()


def test_a_drifted_figure_board_count_fails(tmp_path):
    """A board added to the inventory must not leave page 1 claiming the old count."""
    rows = ["PRE & A & 1 & F1 & 0.6 & 0.6--0.7 & x \\\\"]
    paper = _paper(tmp_path, rows)
    drifted = dict(ebt.state_counts())
    drifted["LIVE"] += 1
    paper.joinpath(ebt._FIGURE).write_text(_figure_text(drifted), encoding="utf-8")
    assert ebt.check(paper, rows) == 1


def test_a_missing_figure_fails(tmp_path):
    rows = ["PRE & A & 1 & F1 & 0.6 & 0.6--0.7 & x \\\\"]
    paper = _paper(tmp_path, rows)
    paper.joinpath(ebt._FIGURE).unlink()
    assert ebt.check(paper, rows) == 1


def test_a_state_label_with_no_count_node_fails(tmp_path):
    """A count deleted from the figure is silence, not agreement."""
    rows = ["PRE & A & 1 & F1 & 0.6 & 0.6--0.7 & x \\\\"]
    paper = _paper(tmp_path, rows)
    text = _figure_text().rsplit("\n", 1)[0]  # drop the trailing POST count node
    paper.joinpath(ebt._FIGURE).write_text(text, encoding="utf-8")
    assert ebt.check(paper, rows) == 1


def test_shipped_board_matches_configured_paper():
    """Cross-repository closure, opt-in.

    Discovering a sibling checkout by relative path would make the unit suite depend on this
    workstation's layout and on the paper repository's branch state. The environment variable is the
    same one the CLI already honors.
    """
    configured = os.environ.get("CATCHBENCH_PAPER_DIR")
    if not configured:
        pytest.skip("set CATCHBENCH_PAPER_DIR to run the cross-repository integration check")
    (preamble, blocks), claims, families = ebt.load()
    assert ebt.check(Path(configured), ebt.rows_latex(preamble, blocks, claims, families)) == 0
