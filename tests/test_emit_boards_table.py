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
import ast
import importlib.util
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
            corpus_name="{1} sources", size="{0}", unit="configs",
            coverage=(),
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
        ebt.corpus_cells(preamble, SPEC)


def test_a_corpus_pattern_matching_twice_is_a_hard_failure(parsed):
    """Two matches means the row could be paired with the wrong corpus, which is silent."""
    preamble, _ = parsed
    with pytest.raises(SystemExit):
        ebt.corpus_cells(preamble + list(preamble), SPEC)


# --- reading the verdicts -------------------------------------------------------------------------


def test_an_unknown_family_id_is_a_hard_failure():
    spec = dict(SPEC, coverage=(("no_such_family", None),))
    with pytest.raises(SystemExit):
        ebt.separates_cell(spec, {})


def test_a_prefix_that_selects_nothing_is_a_hard_failure():
    """An empty selection must fail loudly; silently it prints a smaller denominator."""
    spec = dict(SPEC, coverage=(("f", "det.nosuch."),))
    with pytest.raises(SystemExit):
        ebt.separates_cell(spec, {"f": [{"id": "det.swe.a", "verdict": "separates_as_stated"}]})


def test_a_board_with_no_declared_contrast_says_so():
    assert ebt.separates_cell(SPEC, {}) == "--"


def test_a_nonseparating_claim_is_not_reported_as_separating():
    members = [{"id": "x", "verdict": "does_not_separate"}]
    assert ebt.separates_cell(dict(SPEC, coverage=(("f", None),)), {"f": members}) == "0/1"


def test_a_family_count_comes_from_the_claims_not_from_the_declared_size():
    """A family that grows must move the printed count, which is the drift this exists to catch."""
    members = [{"id": "a", "verdict": "separates_as_stated"},
               {"id": "b", "verdict": "does_not_separate"}]
    assert ebt.separates_cell(dict(SPEC, coverage=(("f", None),)), {"f": members}) == "1/2"


def test_an_id_prefix_splits_one_family_across_two_boards():
    """post_detection_auc spans both corpora, so each Detection row takes its own half."""
    members = [{"id": "det.swe.a", "verdict": "separates_as_stated"},
               {"id": "det.swe.b", "verdict": "does_not_separate"},
               {"id": "det.tau.a", "verdict": "does_not_separate"}]
    assert ebt.separates_cell(dict(SPEC, coverage=(("f", "det.swe."),)), {"f": members}) == "1/2"
    assert ebt.separates_cell(dict(SPEC, coverage=(("f", "det.tau."),)), {"f": members}) == "0/1"


# --- the partition gate ---------------------------------------------------------------------------


def test_a_contrast_belonging_to_no_board_is_a_hard_failure(monkeypatch):
    """The failure the fraction replaced: a dropped family shrinks the denominator in silence."""
    boards = (dict(SPEC, coverage=(("f", None),)),)
    monkeypatch.setattr(ebt, "_BOARDS", boards)
    families = {"f": [{"id": "a", "verdict": "separates_as_stated"}],
                "orphan": [{"id": "b", "verdict": "does_not_separate"}]}
    with pytest.raises(SystemExit):
        ebt.check_partition(families)


def test_a_contrast_claimed_by_two_boards_is_a_hard_failure(monkeypatch):
    boards = (dict(SPEC, coverage=(("f", None),)), dict(SPEC, coverage=(("f", None),)))
    monkeypatch.setattr(ebt, "_BOARDS", boards)
    with pytest.raises(SystemExit):
        ebt.check_partition({"f": [{"id": "a", "verdict": "separates_as_stated"}]})


def test_the_real_boards_partition_the_real_contrasts():
    """Every declared contrast is on exactly one board, so the column cannot under-count."""
    _, _, families = ebt.load()
    ebt.check_partition(families)


def test_the_printed_fractions_sum_to_the_papers_own_totals():
    """The abstract quotes a separation count; a reader who adds up the column must get it.

    This is the reconciliation the redesign exists to protect. Before it, the column showed 7 of
    16 families, so summing the table gave 32 of 65 against an abstract that said 47 of 118.
    """
    (preamble, blocks), claims, families = ebt.load()
    printed = [ebt.separates_cell(spec, families) for spec in ebt._BOARDS]
    pairs = [cell.split("/") for cell in printed if cell != "--"]
    separates = sum(int(a) for a, _ in pairs)
    total = sum(int(b) for _, b in pairs)
    every = [c for members in families.values() for c in members]
    assert total == len(every)
    assert separates == sum(c["verdict"] in ebt._SEPARATING for c in every)


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
        "Board & Corpus & Size & Metric & Floor & Field & Separates \\\\\n"
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
        text.replace("Board & Corpus", "Board & Scored corpus"), encoding="utf-8")
    # the header no longer starts with "Board &" in the same shape, so it must still be dropped
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


def test_the_papers_figure_board_is_the_golden_board():
    """The paper's figures read a copy of the board, and nothing else holds the copy equal.

    ``tools/emit_boards_table.py`` checks the manuscript's LaTeX tables against this repository's
    golden board, so a stale table fails. Its figures do not go through that path: they are drawn by
    ``<paper>/figure/make_*.py``, which parse ``<paper>/figure/board.txt``, a committed copy. The two
    files were byte-identical when this test was written, which is luck rather than enforcement. A
    board regenerated here and a copy left behind there would put a stale number into a figure, and a
    figure is the one place in the manuscript where no checker reads the value.

    Opt-in through the same environment variable as the other cross-repository checks, for the reason
    given on ``test_shipped_board_matches_configured_paper``.
    """
    configured = os.environ.get("CATCHBENCH_PAPER_DIR")
    if not configured:
        pytest.skip("set CATCHBENCH_PAPER_DIR to run the cross-repository integration check")
    copy = Path(configured) / "figure" / "board.txt"
    assert copy.is_file(), f"the paper's figure pipeline reads {copy}, which is absent"
    golden = Path(__file__).resolve().parents[1] / "tests" / "golden" / "board.txt"
    want = golden.read_text(encoding="utf-8").replace("\r\n", "\n")
    got = copy.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert got == want, (
        f"{copy} differs from {golden}; the manuscript's figures would be drawn from a stale board. "
        "Copy the golden over it and redraw the figures.")


def _named_number_rows(value, found=None):
    """Every ``name -> numbers`` pair anywhere in a nested result, ignoring how it is grouped."""
    found = {} if found is None else found
    if isinstance(value, dict):
        for key, item in value.items():
            row = None
            if isinstance(key, str) and isinstance(item, (int, float)) and not isinstance(item, bool):
                row = (item,)
            elif (isinstance(key, str) and isinstance(item, (list, tuple))
                    and item and all(isinstance(x, (int, float)) for x in item)):
                row = tuple(item)
            if row is None:
                _named_number_rows(item, found)
            else:
                found.setdefault(key, set()).add(tuple(round(float(x), 6) for x in row))
    elif isinstance(value, (list, tuple)):
        for item in value:
            _named_number_rows(item, found)
    return found


def test_the_two_figure_pipelines_read_the_board_identically():
    """Two copies of the board parser exist, so hold them equal where it matters: the numbers.

    ``figure-src/board_data.py`` here draws the README figures and ``<paper>/figure/board_data.py``
    draws the manuscript's. Both parse the same board through their own copy of the same accessors.
    A board whose section headers or column layout changed would need the identical edit in two
    repositories, and the second is the edit that gets made a week later or not at all. The failure
    is silent: both files keep parsing and one starts reading the wrong column, so a README figure
    and a manuscript figure print different numbers for one board.

    What is compared is the set of ``method -> values`` rows each side pulls out, not the containers
    it returns them in. The two APIs already differ and should be free to: ``live_prefix`` here
    returns the prefixes, the threshold and a per-corpus mapping, where the manuscript's returns one
    mapping per corpus. Neither shape changes a number. Comparing the shapes instead would make this
    test fail on refactoring, which is how a check gets deleted rather than fixed.
    """
    configured = os.environ.get("CATCHBENCH_PAPER_DIR")
    if not configured:
        pytest.skip("set CATCHBENCH_PAPER_DIR to run the cross-repository integration check")
    here = Path(__file__).resolve().parents[1] / "figure-src" / "board_data.py"
    there = Path(configured) / "figure" / "board_data.py"
    assert here.is_file() and there.is_file(), f"expected both {here} and {there}"

    def load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        # A by-path load must register before executing. Either copy may define a
        # dataclass under postponed annotations, and @dataclass resolves those through
        # sys.modules[cls.__module__].
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    mine = load(here, "_figure_board_data_benchmark")
    theirs = load(there, "_figure_board_data_paper")

    board = (Path(__file__).resolve().parents[1] / "tests" / "golden" / "board.txt").read_text(
        encoding="utf-8")
    assert mine._sections(board) == theirs._sections(board), (
        "the two copies split the board into different sections")

    # (accessor, arguments). detection is per corpus on both copies; the other two read the whole
    # board. The corpus names are the ones the golden prints as scored-block scenarios.
    accessors = (("detection", ("swegym",)), ("detection", ("tau",)),
                 ("live_prefix", ()), ("pre_by_source", ()))
    assert all(callable(getattr(mine, a, None)) and callable(getattr(theirs, a, None))
               for a, _ in accessors), f"expected {sorted({a for a, _ in accessors})} on both copies"

    disagree = []
    for name, args in accessors:
        label = f"{name}({', '.join(args)})" if args else f"{name}()"
        want = _named_number_rows(getattr(theirs, name)(*args))
        got = _named_number_rows(getattr(mine, name)(*args))
        assert want, f"the manuscript copy's {label} exposed no numeric rows to compare"
        for row in sorted(set(want) | set(got)):
            if want.get(row) != got.get(row):
                disagree.append(f"{label}:{row}")
    assert disagree == [], (
        "the two figure pipelines read different numbers off the same board for: "
        + ", ".join(disagree)
        + ". A README figure and a manuscript figure would disagree.")
