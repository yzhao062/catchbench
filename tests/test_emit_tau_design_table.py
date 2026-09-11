r"""Hold the tau-bench design block to the record it came from, and to the claims it may not make.

Three failures are worth failing loudly for, and none of them is visible to a reader of the LaTeX.

The first is a number the record does not support. Every cell in this block is a design target for a
future study, and the printed form of one is a three-digit integer that looks equally plausible at
408, 434 or 999. There is no internal evidence in a table of twenty such integers that any of them
came from the solve. So the tests compare the printed cells against the record's own fields, and
against the full-precision comment lines the block carries beside them, and both directions have to
agree: a corrupted record fails, and a corrupted block fails.

The second is a claim the calculation is not entitled to make. Holding an observed point estimate
fixed and solving for the sample says what would resolve an effect of that size; it does not say the
effect is there, and it certainly does not say that \texttt{full} or \texttt{auditable} would clear
the 0.70 bar at 410 or 434 clusters. Part B3 of the declaration forbids the two neighbouring
quantities, achieved power and a minimum detectable difference, by name. The caption is required to
say that neither was computed, so a blanket ban on the words would ban the disclosure with them; the
guard counts occurrences against the record's own refusal sentences instead, and a word that appears
once more than those sentences account for fails.

The third is an unescaped percent sign, which this repository has already been bitten by once. A
caption kept ``95\%%`` after its ``%`` interpolation was removed, the second sign reached LaTeX, and
it commented out the rest of the generated line: no build warning, no --check failure because source
and output shared the error, and no test failure because every test read a source string. The guard
here reads the bytes the command line actually writes.
"""
from __future__ import annotations

import ast
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import emit_live_prefix_table as elpt  # noqa: E402
import emit_tau_design_table as etd  # noqa: E402

RECORD = TOOLS / "live_tau_bar_design_results.json"
FRAGMENT = TOOLS / "tau_design_table.tex"

_ROW_END = r" \\"

# The full-precision comment lines the block carries beside every printed cell. Each is anchored at
# both ends, so a line that grew or lost a field stops matching rather than matching loosely.
_CELL_COMMENT = re.compile(
    r"^% (.+?): id=(\S.*?); auc=(-?[\d.]+); point=(-?[\d.]+); sd=([\d.]+); "
    r"interval=\[(-?[\d.]+), (-?[\d.]+)\]; verdict=(\w+)$")
_MEASURED_COMMENT = re.compile(
    r"^% (.+?): measured clusters=(\d+); exact=([\d.]+); runs=(\d+); multiple=([\d.]+); "
    r"exponent=(-?[\d.]+); below_ladder=(True|False); exponent_source=(.*)$")
_CLOSED_COMMENT = re.compile(
    r"^% (.+?): closed-form clusters=(\d+); exact=([\d.]+); runs=(\d+); multiple=([\d.]+); "
    r"below_ladder=(True|False)$")
_LADDER_COMMENT = re.compile(
    r"^% ladder (.+?): exponent=(-?[\d.]+); interval=\[(-?[\d.]+), (-?[\d.]+)\]; se=([\d.]+); "
    r"df=(\d+); R2=([\d.]+); consistent_with_reference=(True|False); declared_because=(.*)$")

# Words that would turn a design target into a promise. "would clear the bar" is handled apart: the
# caption is required to carry it, in the negated form the record supplies and only there.
FORBIDDEN = ("significant", "significance", "adequately powered", "powered to detect",
             "sufficient sample", "will clear", "would reach the bar", "guarantee", "proves",
             "holm", "underpowered")

# Every occurrence of these in the block has to be accounted for by the record's own refusal
# sentences plus the labels the caption states them under. One more is a claim nobody declared.
ACCOUNTED = ("power", "detectable", "post-hoc", "post hoc")

# Twenty design counts printed side by side invite a sentence about which entrant needs fewer, and
# nothing in this block tests a difference between two entrants: the cells are marginal readings
# against a fixed bar over shared runs. Two rows landing on the same count invite the opposite
# sentence just as strongly, so equivalence is banned with superiority. "strongest" is deliberately
# absent: Part B5 uses it in the body to name which two cells did not separate, which is a
# description of the pair rather than a verdict read off this table. "on par" is absent for a duller
# reason: this is a substring scan, and it matches inside "the directi(on par)t B3 forbids".
COMPARATIVE = ("outperforms", "beats", "leads", "better than", "worse than", "wins", "superior",
               "top performer", "no difference", "equivalent", "indistinguishable")


@pytest.fixture(scope="module")
def record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def generated():
    return etd.build()


@pytest.fixture(scope="module")
def rendered():
    """The bytes the command line writes, so the percent guard reads output rather than source."""
    result = _cli()
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    return result.stdout


@pytest.fixture
def mutable(record):
    return copy.deepcopy(record)


def _cli(*args, paper_env=None):
    env = os.environ.copy()
    env.pop("CATCHBENCH_PAPER_DIR", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if paper_env is not None:
        env["CATCHBENCH_PAPER_DIR"] = str(paper_env)
    return subprocess.run([sys.executable, str(Path(etd.__file__)), *map(str, args)],
                          capture_output=True, env=env, timeout=600)


def _float_lines(generated: str, label: str) -> list[str]:
    """One generated float, located by its own label rather than by position in the block."""
    lines = generated.splitlines()
    marker = r"\label{" + label + "}"
    assert lines.count(marker) == 1, f"{label} appears {lines.count(marker)} times"
    index = lines.index(marker)
    start = max(i for i, line in enumerate(lines[:index]) if line == r"\begin{table}[t]")
    stop = min(i for i, line in enumerate(lines) if i > index and line == r"\end{table}")
    return lines[start:stop]


def _cells(line: str) -> list[str]:
    return line.removesuffix(_ROW_END).split(" & ")


def _design_rows(generated: str) -> list[list[str]]:
    return [_cells(line) for line in _float_lines(generated, "tab:tau-design")
            if line.endswith(_ROW_END) and re.fullmatch(r"\d+\\%", _cells(line)[0])]


def _scaling_rows(generated: str, record: dict) -> list[list[str]]:
    known = {cell["cell"] for cell in record["scaling_check"]["per_cell"]}
    return [_cells(line) for line in _float_lines(generated, "tab:tau-scaling")
            if line.endswith(_ROW_END) and _cells(line)[0] in known]


def _comments(generated: str, pattern: re.Pattern) -> dict[str, tuple]:
    found = {}
    for line in generated.splitlines():
        match = pattern.match(line)
        if match:
            assert match.group(1) not in found, f"duplicate comment line for {match.group(1)}"
            found[match.group(1)] = match.groups()[1:]
    return found


# --- the printed block against the record it came from --------------------------------------------

def test_every_printed_row_is_the_record_row_it_names(generated, record):
    """Same cells, same order, same numbers and the same verdict as the shipped record.

    Order is compared rather than looked up, because a table whose verdicts had slid one row against
    their entrants would still carry twenty correct numbers and twenty correct verdicts.
    """
    rows = _design_rows(generated)
    assert len(rows) == len(record["design_table"]) == record["summary"]["cells"]
    printed_names = dict(elpt._METHODS)
    for cells, row in zip(rows, record["design_table"]):
        prefix, method, point, sd, interval, verdict, measured, closed = cells
        assert prefix == f"{row['prefix_percent']}" + r"\%"
        assert method.removesuffix(etd._OWN_LADDER) == printed_names[row["method"]]
        assert (method.endswith(etd._OWN_LADDER)
                == (row["needed_measured_exponent_source"] == etd._OWN_LADDER_SOURCE))
        assert point == "$" + f"{row['point_auc_minus_bar']:.3f}" + "$"
        assert sd == f"{row['bootstrap_sd']:.3f}"
        low, high = row["interval_95"]
        assert interval == "$[" + f"{low:.3f}" + ", " + f"{high:.3f}" + "]$"
        assert verdict == etd._VERDICT_TEXT[row["verdict"]]
        for cell, key in ((measured, "needed_measured"), (closed, "needed_closed_form")):
            assert cell.removesuffix(etd._BELOW_LADDER) == str(row[key]["clusters"])
            assert cell.endswith(etd._BELOW_LADDER) == row[key]["below_measured_ladder"]


def test_every_printed_number_is_its_comment_line_value_rounded(generated, record):
    """The printed cell, the comment line beside it and the record must be one number.

    The comment lines are the only place a reader can re-solve a row from, so they are checked
    against the record at full precision and the printed cells are checked against them. Corrupting
    either one alone breaks the chain.
    """
    cells = _comments(generated, _CELL_COMMENT)
    measured = _comments(generated, _MEASURED_COMMENT)
    closed = _comments(generated, _CLOSED_COMMENT)
    rows = {row["cell"]: row for row in record["design_table"]}
    assert set(cells) == set(measured) == set(closed) == set(rows)
    for name, row in rows.items():
        claim_id, auc, point, sd, low, high, verdict = cells[name]
        assert claim_id == row["id"]
        assert auc == f"{row['auc']:.12f}"
        assert point == f"{row['point_auc_minus_bar']:.12f}"
        assert sd == f"{row['bootstrap_sd']:.12f}"
        assert [low, high] == [f"{value:.12f}" for value in row["interval_95"]]
        assert verdict == row["verdict"]
        # The measured line carries its exponent and that exponent's source after the below-ladder
        # flag and the closed-form line carries neither, so the flag sits at a different index in
        # the two patterns. Reading it from the end of both would read a source string as a flag.
        for fields, key, flag in ((measured[name], "needed_measured", 5),
                                  (closed[name], "needed_closed_form", 4)):
            needed = row[key]
            assert fields[0] == str(needed["clusters"])
            assert fields[1] == f"{needed['clusters_exact']:.12f}"
            assert fields[2] == str(needed["runs"])
            assert fields[3] == f"{needed['multiple_of_present_sample']:.12f}"
            assert fields[flag] == str(needed["below_measured_ladder"])
        assert measured[name][4] == f"{row['needed_measured_exponent']:.12f}"
        assert measured[name][6] == row["needed_measured_exponent_source"]
    # and the printed three-decimal cells are those same comment values rounded, not derived again
    for printed, row in zip(_design_rows(generated), record["design_table"]):
        exact = cells[row["cell"]]
        assert printed[2] == "$" + f"{float(exact[2]):.3f}" + "$"
        assert printed[3] == f"{float(exact[3]):.3f}"


def test_the_scaling_rows_are_the_four_ladders_the_record_carries(generated, record):
    """The exponent every design count rests on, printed where a reader can see it."""
    per_cell = record["scaling_check"]["per_cell"]
    rows = _scaling_rows(generated, record)
    assert len(rows) == len(per_cell)
    comments = _comments(generated, _LADDER_COMMENT)
    assert set(comments) == {cell["cell"] for cell in per_cell}
    for cells, cell in zip(rows, per_cell):
        name, because, exponent, interval, consistent = cells
        assert name == cell["cell"]
        assert because == cell["declared_because"]
        assert exponent == "$" + f"{cell['exponent']:.3f}" + "$"
        low, high = cell["exponent_interval_95"]
        assert interval == "$[" + f"{low:.3f}" + ", " + f"{high:.3f}" + "]$"
        assert consistent == ("yes" if cell["consistent_with_minus_half"] else "no")
        fields = comments[name]
        assert fields[0] == f"{cell['exponent']:.12f}"
        assert [fields[1], fields[2]] == [f"{value:.12f}" for value in cell["exponent_interval_95"]]
        assert fields[6] == str(cell["consistent_with_minus_half"])
        assert fields[7] == cell["declared_because"]


def test_the_daggered_rows_are_exactly_the_cells_whose_ladder_was_run(generated, record):
    """The dagger vouches for a fitted exponent; it may not appear on a cell with no ladder."""
    laddered = {cell["cell"] for cell in record["scaling_check"]["per_cell"]}
    marked = {row["cell"] for row, cells in zip(record["design_table"], _design_rows(generated))
              if cells[1].endswith(etd._OWN_LADDER)}
    assert marked == laddered == set(record["declaration"]["frozen_before_the_run"]["cells"])


def test_the_caption_counts_are_the_records_summary(generated, record):
    """Eighteen, twenty and two are the summary's, spelled; the present sample is the summary's too."""
    summary = record["summary"]
    prose = " ".join(generated.split())
    separating = etd._word(summary["separating_at_present_sample"]).capitalize()
    assert (f"{separating} of the {etd._word(summary['cells'])} cells already separate from the "
            f"0.70 bar at that sample and "
            f"{etd._word(summary['not_separating_at_present_sample'])} do not") in prose
    assert (f"The present sample is {summary['present_clusters']:,} task clusters and "
            f"{summary['present_rows']:,} runs, 50 airline and 115 retail.") in prose


def test_the_caption_states_the_design_target_for_each_unresolved_cell(generated, record):
    """The two cells the manuscript quotes, with the figures the record solved for them."""
    prose = " ".join(generated.split())
    targets = record["summary"]["needed_for_unresolved_cells"]
    assert sorted(targets) == sorted(record["unresolved_cells"])
    for cell, needed in targets.items():
        assert (f"{cell} at {needed['measured_clusters']:,} clusters "
                f"({needed['measured_runs']:,} runs, "
                f"{needed['multiple_of_present_sample']:.1f} times the present sample, "
                f"{needed['closed_form_clusters']:,} clusters under the closed form)") in prose


def test_the_ladder_and_its_frozen_inputs_are_the_declarations(generated, record):
    """Part B5 asks for the ladder beside the exponents; it is the one frozen before the run."""
    frozen = record["declaration"]["frozen_before_the_run"]
    prose = " ".join(generated.split())
    assert ("Ladder: " + etd._and_list([str(rung) for rung in frozen["ladder"]])
            + " clusters; " + frozen["stratification"] + "; "
            + etd._word(frozen["subsamples_per_rung"]) + " subsamples per rung and "
            + f"{frozen['draws_per_subsample']:,} bootstrap draws inside each subsample; base seed "
            + f"{frozen['rng_base_seed']}") in prose
    assert etd._texttt(frozen["rng_label_template"]) in generated
    assert f"% Ladder: {', '.join(str(rung) for rung in frozen['ladder'])} clusters" in generated


def test_no_committed_point_reaches_the_bar(record):
    """The premise of the discussion clause this block supplies a number for.

    ``06_discussion.tex`` says no tau-bench entrant's point estimate reaches its 0.70 bar at any
    prefix. Every design count in this table is a target for closing a gap that sentence asserts
    exists, so a record in which some cell had crossed the bar would leave the block answering a
    question the manuscript no longer asks.
    """
    above = [row["cell"] for row in record["design_table"] if row["auc"] >= row["bar"]]
    assert above == [], f"these cells reach the bar: {above}"


# --- the red direction: records this block refuses to print ---------------------------------------

def test_a_summary_that_disagrees_with_its_rows_is_refused(mutable):
    """The counts in the caption and the rows in the table have separate sources and must agree."""
    mutable["summary"]["separating_at_present_sample"] = 17
    with pytest.raises(ValueError, match="the summary says 17"):
        etd.table(mutable)


def test_a_summary_that_names_the_wrong_unresolved_cells_is_refused(mutable):
    mutable["unresolved_cells"] = ["25.full"]
    with pytest.raises(ValueError, match="as unresolved"):
        etd.table(mutable)


def test_rows_measured_at_different_samples_are_refused(mutable):
    """The caption prints one cluster count for the whole table."""
    mutable["design_table"][0]["n_clusters"] = 164
    with pytest.raises(ValueError, match="disagree on the sample"):
        etd.table(mutable)


def test_a_verdict_its_own_interval_contradicts_is_refused(mutable):
    """The verdict column and the interval column sit side by side and tell the same story."""
    mutable["design_table"][0]["verdict"] = "does_not_separate"
    mutable["design_table"][0]["separates_at_present_sample"] = False
    with pytest.raises(ValueError, match="contradicts the interval"):
        etd.table(mutable)


def test_an_unrecognised_verdict_is_refused(mutable):
    mutable["design_table"][0]["verdict"] = "inconclusive"
    with pytest.raises(ValueError, match="unrecognised verdict"):
        etd.table(mutable)


def test_a_row_claiming_an_exponent_its_ladder_did_not_fit_is_refused(mutable):
    """The dagger says a fitted exponent produced this count; it has to be that cell's own."""
    for row in mutable["design_table"]:
        if row["needed_measured_exponent_source"] == etd._OWN_LADDER_SOURCE:
            row["needed_measured_exponent"] = -0.5
            break
    with pytest.raises(ValueError, match="claims its own measured exponent"):
        etd.table(mutable)


def test_a_row_set_that_is_not_the_live_prefix_entrants_in_order_is_refused(mutable):
    mutable["design_table"][0], mutable["design_table"][1] = (mutable["design_table"][1],
                                                              mutable["design_table"][0])
    with pytest.raises(ValueError, match="are not the entrants of"):
        etd.table(mutable)


def test_a_cell_that_does_not_name_its_own_claim_is_refused(mutable):
    mutable["design_table"][0]["id"] = "live.tau.bar.25.full"
    with pytest.raises(ValueError, match="does not name its own claim id"):
        etd.table(mutable)


def test_a_ladder_that_is_not_the_declared_one_is_refused(mutable):
    mutable["scaling_check"]["per_cell"][0]["ladder"] = [40, 60, 80]
    with pytest.raises(ValueError, match="against the frozen"):
        etd.table(mutable)


def test_a_ladder_on_a_cell_the_declaration_did_not_freeze_is_refused(mutable):
    mutable["scaling_check"]["per_cell"][0]["cell"] = "50.full"
    with pytest.raises(ValueError, match="not the cells the declaration froze"):
        etd.table(mutable)


def test_a_pooled_exponent_without_the_agreement_that_licenses_it_is_refused(mutable):
    mutable["scaling_check"]["pooling"]["pooled_exponent"] = -0.49
    with pytest.raises(ValueError, match="without the agreement that licenses it"):
        etd.table(mutable)


def test_an_unrecognised_printed_figure_source_is_refused(mutable):
    mutable["scaling_check"]["printed_figure_source"] = "whichever"
    with pytest.raises(ValueError, match="unrecognised printed figure source"):
        etd.table(mutable)


def test_two_different_bracket_sources_are_refused(mutable):
    """The caption names one source for every row without a ladder; two would leave one unstated."""
    changed = 0
    for row in mutable["design_table"]:
        if row["needed_measured_exponent_source"] != etd._OWN_LADDER_SOURCE and changed == 0:
            row["needed_measured_exponent_source"] = "some other bracket"
            changed = 1
    with pytest.raises(ValueError, match="cite 2 different exponent sources"):
        etd.table(mutable)


# --- the branch Part B4 requires the caption to take ----------------------------------------------

def test_the_block_names_which_extrapolation_produced_the_printed_number(generated, record):
    assert record["scaling_check"]["printed_figure_source"] == "measured_scaling"
    prose = " ".join(generated.split())
    assert "the printed figure comes from the measured scaling" in prose
    assert "the closed form stands beside it as the idealized comparison" in prose
    assert "the printed figure is the closed-form extrapolation" not in prose


def test_the_closed_form_branch_of_the_caption_is_reachable_and_says_so(mutable):
    """B4 was written for a run where every exponent agreed with the reference; that branch works."""
    for cell in mutable["scaling_check"]["per_cell"]:
        cell["consistent_with_minus_half"] = True
    mutable["scaling_check"]["all_four_consistent_with_minus_half"] = True
    mutable["scaling_check"]["printed_figure_source"] = "closed_form"
    prose = " ".join(etd.table(mutable).split())
    assert "the printed figure is the closed-form extrapolation" in prose
    assert "comes from the measured scaling" not in prose


def test_the_pooled_branch_of_the_caption_prints_the_pooled_exponent(mutable):
    pooling = mutable["scaling_check"]["pooling"]
    pooling["cells_agree"] = True
    pooling["pooled_exponent"] = -0.4903
    pooling["pooled_exponent_interval_95"] = [-0.5012, -0.4794]
    prose = " ".join(etd.table(mutable).split())
    assert "The four therefore agree and the pooled exponent is -0.490 [-0.501, -0.479]." in prose
    assert "No pooled exponent is reported" not in prose


def test_the_withheld_pooling_reason_is_the_records_own(generated, record):
    reason = record["scaling_check"]["pooling"]["withheld_reason"]
    assert "No pooled exponent is reported: " + reason + "." in " ".join(generated.split())


# --- claims this block may not make ---------------------------------------------------------------

def test_the_block_promises_nothing_about_clearing_the_bar(generated, record):
    """A design target says what would resolve an effect of the size observed, and nothing else."""
    lowered = " ".join(generated.split()).lower()
    hits = [word for word in FORBIDDEN if word in lowered]
    assert hits == [], f"the block makes a claim it did not measure: {hits}"
    # the one permitted occurrence, in the negated form the record supplies and only there
    assert lowered.count("would clear the bar") == 1
    assert lowered.count("not a claim that a cell would clear the bar") == 1
    assert record["interpretation"]["design_target_only"] in " ".join(generated.split())


def test_the_block_compares_no_two_entrants_in_either_direction(generated):
    """A table of twenty design counts orders itself; the caption may not read that as a verdict."""
    prose = " ".join(generated.split()).lower()
    hits = [word for word in COMPARATIVE if word in prose]
    assert hits == [], f"the block sets entrants against each other: {hits}"


def test_the_two_refused_quantities_appear_only_as_refusals(generated, record):
    """Part B3 forbids both by name, and the caption has to say so, so the guard counts.

    A blanket ban would ban the disclosure. Counting against the record's own sentences plus the two
    labels they are stated under means a later edit that reaches for "power" one more time, in a
    sentence nobody declared, fails here.
    """
    lowered = " ".join(generated.split()).lower()
    accounted = " ".join([record["not_computed"][key] for key in etd._REFUSED_LABELS]
                         + list(etd._REFUSED_LABELS.values())).lower()
    for word in ACCOUNTED:
        assert lowered.count(word) == accounted.count(word), (
            f"{word!r} appears {lowered.count(word)} times in the block and "
            f"{accounted.count(word)} times in the sentences that account for it")
    assert "achieved power: never computed." in lowered
    assert "minimum detectable difference: never computed." in lowered


def test_every_required_record_sentence_reaches_the_caption_verbatim(generated, record):
    """The record and the manuscript may not carry different reasons for the same refusal."""
    prose = " ".join(generated.split())
    for key in ("design_target_only", "fixed_point_estimates", "normal_idealization",
                "exploratory"):
        assert record["interpretation"][key] in prose, f"the caption drops {key}"
    for key in etd._REFUSED_LABELS:
        assert record["not_computed"][key] in prose, f"the caption drops {key}"
    assert etd._prose(record["question"]) in prose
    assert record["scaling_check"]["assumption"] in prose


def test_the_long_notes_that_carry_markup_are_kept_in_the_comment_lines(generated, record):
    """Two record notes carry characters LaTeX would read as markup, so they stay unparsed."""
    check = record["scaling_check"]
    assert "% Shape of the relation: " + check["shape_of_the_relation"] in generated
    assert ("% Within-cell error: " + check["within_cell_error_is_probably_understated"]
            in generated)
    with pytest.raises(ValueError, match="LaTeX special characters"):
        etd._prose(check["shape_of_the_relation"])


# --- escaping and rendering -----------------------------------------------------------------------

def test_no_generated_percent_starts_a_latex_comment(rendered):
    r"""A bare % comments out the rest of its line, silently and invisibly.

    This is not hypothetical. A shortened caption elsewhere in this appendix kept ``95\%%`` from a
    string that had used % interpolation; the rewrite had no % operator, so both signs reached LaTeX
    and the second one ate the rest of the line. The build reported zero warnings, --check passed
    because source and output shared the error, and every test passed because they all read source
    strings. This one reads the bytes the command line writes.
    """
    offenders = []
    for number, line in enumerate(rendered.decode("utf-8").splitlines(), 1):
        if line.startswith("%"):
            continue  # a provenance comment line, which is meant to be a comment
        for index, char in enumerate(line):
            if char == "%" and (index == 0 or line[index - 1] != "\\"):
                offenders.append(f"line {number} col {index + 1}: {line[max(0, index - 40):]}")
    assert offenders == [], "unescaped % in the rendered block:\n  " + "\n  ".join(offenders)


def test_prose_escapes_a_percent_and_refuses_every_other_special():
    # The two sets are written out rather than derived from each other, because the difference
    # operator is banned in the emitter. They must still name the same characters.
    assert etd._PROSE_SPECIALS | {"%"} == etd._LATEX_SPECIALS
    assert "%" not in etd._PROSE_SPECIALS
    assert etd._prose("a 95% interval") == r"a 95\% interval"
    assert etd._prose("plain words") == "plain words"
    for markup in (r"a\b", "a{b}", "a$b", "a&b", "a#b", "a^2", "a_b", "a~b"):
        with pytest.raises(ValueError, match="LaTeX special characters"):
            etd._prose(markup)


def test_texttt_escapes_underscores_and_braces_and_refuses_the_rest():
    assert etd._texttt("live_tau_scaling.{cell}") == r"\texttt{live\_tau\_scaling.\{cell\}}"
    for bad in (r"a\b", "a$b", "a&b", "a#b", "a^b", "a~b", "a%b"):
        with pytest.raises(ValueError, match="not a printable identifier"):
            etd._texttt(bad)


def test_the_caption_refuses_a_roster_its_prose_cannot_describe(record):
    with pytest.raises(ValueError, match="outgrown the prose"):
        etd._word(record["summary"]["cells"] + len(etd._WORDS))
    assert etd._word(record["summary"]["cells"]) == "twenty"
    assert etd._word(record["summary"]["separating_at_present_sample"]) == "eighteen"
    assert etd._word(0) == "no"


def test_the_entrant_names_are_the_ones_the_live_prefix_table_prints(generated, record):
    """One entrant, one printed name, in both appendix tables. A rename moves both or fails here."""
    assert etd._METHODS is elpt._METHODS
    printed = dict(elpt._METHODS)
    methods = {row["method"] for row in record["design_table"]}
    assert methods == {method for method, _ in elpt._METHODS[1:]}
    assert "random" not in methods, "random has no bar cell and cannot be solved for"
    names = {cells[1].removesuffix(etd._OWN_LADDER) for cells in _design_rows(generated)}
    assert names == {printed[method] for method in methods}
    assert etd.LIVE_TAU_TABLE in generated


# --- the emitter owns no arithmetic ---------------------------------------------------------------

def test_the_emitter_owns_no_arithmetic_of_its_own():
    """The whole reason this tool exists rather than a hand-typed table.

    A second solve in this repository would return a plausible three-digit integer for every row and
    would be wrong in the manuscript rather than in a console line. Banning the operators is what
    makes that mechanical. It has a second effect worth having: with no ``%`` operator anywhere, the
    doubled percent sign that once ate a caption line cannot be left behind by a rewrite.
    """
    source = Path(etd.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned = (ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.Mod, ast.MatMult)
    offenders = sorted({type(node.op).__name__ for node in ast.walk(tree)
                        if isinstance(node, ast.BinOp) and isinstance(node.op, banned)})
    assert offenders == [], f"the emitter computes: {offenders}"
    for reachable in ("math", "np", "numpy", "statistics", "scipy", "design_clusters"):
        assert not hasattr(etd, reachable), f"emit_tau_design_table reaches {reachable}"
    numbers = {node.value for node in ast.walk(tree)
               if isinstance(node, ast.Constant)
               and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)}
    floats = sorted(value for value in numbers if isinstance(value, float))
    assert floats == [], f"no measured value is written down here; found {floats}"
    # every count and cluster figure the caption states is read, never typed
    for typed in (18, 20, 165, 660, 410, 434, 408, 412, 1640, 1736, 40, 70):
        assert typed not in numbers, f"{typed} is written down in the emitter"


# --- the command line and the paper check ---------------------------------------------------------

def test_cli_prints_deterministic_lf_bytes(generated, rendered):
    second = _cli()
    assert second.returncode == 0
    assert second.stderr == b""
    assert rendered == second.stdout == (generated + "\n").encode("utf-8")
    assert b"\r" not in rendered


def test_the_shipped_fragment_is_what_the_emitter_prints(rendered):
    """The .tex fragment in tools/ is a checked artifact, not a copy someone took once."""
    assert FRAGMENT.is_file(), f"{FRAGMENT.name} is missing; regenerate it from the emitter"
    assert FRAGMENT.read_bytes() == rendered


def test_a_recorded_windows_path_prints_as_a_file_name_off_windows():
    r"""The provenance line must name a file, on the platform CI runs as well as this one.

    The record stores absolute paths and the block prints only the last segment. Taking it with
    ``Path`` reads correctly here and fails on Linux, where a backslash is an ordinary character and
    ``PosixPath(...).name`` returns the whole string. That shipped: the emitter printed
    ``C:\Users\...\statistical_tests_results.json`` into the generated block on CI while the
    committed fragment carried the file name, so the two matched on Windows alone.

    The first assertion pins the defect rather than the fix. If POSIX semantics ever changed, the
    second assertion would pass whether or not the helper did anything, and this test would go
    quietly useless.
    """
    recorded = r"C:\Users\somebody\PycharmProjects\auditablebench\tools\statistical_tests_results.json"

    assert PurePosixPath(recorded).name == recorded, (
        "a Windows path no longer survives POSIX parsing intact, so this test no longer reproduces "
        "the condition it was written for")
    assert etd._recorded_basename(recorded) == "statistical_tests_results.json"

    assert etd._recorded_basename("/home/runner/work/catchbench/tools/x.json") == "x.json"
    assert etd._recorded_basename("x.json") == "x.json"
    with pytest.raises(ValueError):
        etd._recorded_basename("")


def test_the_provenance_line_names_no_absolute_path(generated):
    """Whatever the record stored, the block a reader sees carries no one's home directory."""
    for line in generated.splitlines():
        assert not re.search(r"[A-Za-z]:[\\/]", line), f"an absolute path reached the block: {line}"
        assert "/home/" not in line, f"an absolute path reached the block: {line}"


@pytest.mark.parametrize("use_env", [False, True])
def test_check_accepts_exact_block(tmp_path, generated, use_env):
    (tmp_path / "09_appendix.tex").write_bytes((generated + "\n").encode("utf-8"))
    result = (_cli("--check", paper_env=tmp_path) if use_env
              else _cli("--check", "--paper", tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert b"paper is current: tab:tau-design and tab:tau-scaling" in result.stdout


@pytest.mark.parametrize("corruption, message", [
    ("no_block", b"appears 0 times"), ("no_appendix", b"cannot read appendix"),
    ("cell", b"block differs"), ("caption", b"block differs"),
    ("missing_begin", b"appears 0 times"), ("missing_end", b"appears 0 times"),
    ("duplicate", b"appears 2 times"), ("reversed", b"precedes its begin marker"),
    ("whitespace", b"block differs"), ("crlf", b"block differs"),
])
def test_check_rejects_a_paper_that_does_not_carry_the_block(tmp_path, generated, corruption,
                                                             message):
    """The red direction. A checker that cannot fail proves nothing about the paper it passes.

    ``no_block`` and ``no_appendix`` are the state of the manuscript today, so they pin the exit code
    this tool returns before anything is spliced.
    """
    text = generated
    if corruption == "no_block":
        text = r"\section{An appendix that has never seen this table}"
    elif corruption == "cell":
        text = text.replace(r"& 434 & 412 \\", r"& 999 & 412 \\", 1)
    elif corruption == "caption":
        text = text.replace("are design targets for a future study",
                            "are guarantees for a future study", 1)
    elif corruption == "missing_begin":
        text = text.replace(etd._BEGIN, "")
    elif corruption == "missing_end":
        text = text.replace(etd._END, "")
    elif corruption == "duplicate":
        text += "\n" + etd._BEGIN
    elif corruption == "reversed":
        text = etd._END + "\n" + etd._BEGIN
    elif corruption == "whitespace":
        text = text.replace(r"\centering", r"\centering ", 1)
    elif corruption == "crlf":
        text = text.replace("\n", "\r\n")
    assert corruption in ("no_block", "no_appendix") or text != generated, "corruption did nothing"
    if corruption != "no_appendix":
        (tmp_path / "09_appendix.tex").write_bytes(text.encode("utf-8"))
    result = _cli("--check", "--paper", tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert message in result.stdout
    if corruption == "cell":
        assert b"--- paper/09_appendix.tex" in result.stdout
        assert b"999" in result.stdout and b"434" in result.stdout


def test_check_requires_paper_path():
    result = _cli("--check")
    assert result.returncode == 2
    assert b"--check needs --paper <dir> or CATCHBENCH_PAPER_DIR" in result.stderr


def test_a_broken_record_exits_one_with_a_readable_reason(tmp_path, monkeypatch, capsys):
    """main() reports rather than tracebacks, so a checker sees a reason instead of a stack."""
    monkeypatch.setattr(etd, "RECORD", tmp_path / "absent.json")
    monkeypatch.setattr(sys, "argv", ["emit_tau_design_table.py"])
    assert etd.main() == 1
    assert "ERROR: tau-bench bar design table:" in capsys.readouterr().err


# --- claims about the manuscript, which need the manuscript ---------------------------------------

def test_the_discussion_clause_this_block_answers_is_in_the_manuscript():
    """DISCUSSION_CLAUSE is prose about another file; bind it to that file."""
    paper_dir = os.environ.get("CATCHBENCH_PAPER_DIR")
    if not paper_dir:
        pytest.skip("no CATCHBENCH_PAPER_DIR")
    discussion = (Path(paper_dir) / "06_discussion.tex").read_text(encoding="utf-8")
    assert etd.DISCUSSION_CLAUSE in discussion, (
        f"the block supplies the number for {etd.DISCUSSION_CLAUSE!r}; the discussion does not say "
        f"it")


def test_the_live_prefix_table_both_captions_point_at_exists():
    """Both captions send a reader to the table that prints these cells; it has to be there."""
    paper_dir = os.environ.get("CATCHBENCH_PAPER_DIR")
    if not paper_dir:
        pytest.skip("no CATCHBENCH_PAPER_DIR")
    sources = " ".join(path.read_text(encoding="utf-8")
                       for path in sorted(Path(paper_dir).glob("*.tex")))
    assert r"\label{" + etd.LIVE_TAU_TABLE + "}" in sources, (
        f"both captions reference {etd.LIVE_TAU_TABLE}; the paper does not define it")
