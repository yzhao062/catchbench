r"""Emit the tau-bench bar design block: twenty design rows, and the ladder that licensed them.

``06_discussion.tex`` says that measuring the gap to the 0.70 warning bar more sharply "needs a
larger sample" and carries no number. ``tools/live_tau_bar_design.py`` computed one for each of the
twenty committed ``live.tau.bar`` cells and wrote ``tools/live_tau_bar_design_results.json``. Part B5
of ``research/catchbench-live-size-control-declaration-2026-09-10.md`` says what the appendix then
carries: the full twenty-row table with point, standard deviation, interval, verdict and needed
cluster count, plus the measured scaling exponent and the ladder. This file formats exactly that and
computes nothing.

**Nothing here is arithmetic.** Every point, spread, endpoint, verdict, cluster count, run count,
multiple and exponent is read out of the record at the precision the record stores it and printed
rounded. That is the point rather than a convenience. A design count is the solution of
``|point| = z * sd * (n / 165) ** exponent`` for ``n``, and a second implementation of that solve
would be wrong in a way a reader could not see: it would still return a plausible three-digit number
for every row, in the manuscript rather than in a console line. So the solve stays in
``live_tau_bar_design.design_clusters``, where ``tests/test_live_tau_bar_design.py`` holds it to its
own inversion, and this file is forbidden the operators that would let it recompute anything. The
test suite enforces that with an AST scan rather than by reading the code, so the ban survives an
edit made in good faith.

That ban has a second effect worth having. Banning ``%`` as an operator is what makes the escaped
percent signs in these captions safe. The failure this repository has already met once is a caption
that kept ``95\%%`` after its ``%``-interpolation was removed: both signs reached LaTeX, the second
commented out the rest of the generated line, the build reported no warning and every test passed
because they all read source strings. With no ``%`` operator anywhere in the module there is no
doubled sign to leave behind, and the test that matters checks the rendered bytes rather than a
source string.

Three sources feed the block and each is bound rather than restated.

The record supplies every number and, verbatim, every sentence the declaration requires the caption
to carry: what the quantity is, that the counts are design targets rather than a promise that a cell
would clear the bar, that the point estimates are held fixed, that the normal solve idealizes an
asymmetric percentile interval, and that neither achieved power nor a minimum detectable difference
was computed. Reading those out of the record rather than paraphrasing them is what stops the public
record and the manuscript from carrying different reasons for the same refusal.

``emit_live_prefix_table._METHODS`` supplies the printed entrant names, so an entrant reads the same
way in this table as in Table~\ref{tab:live-stream-tau}, which prints the very cells this table
solves. A rename there moves both tables or fails here.

The declaration block inside the record supplies the ladder, the draw counts, the seed and the RNG
label template, which are the frozen inputs Part B4 named before any subsample was drawn.

Usage::

    python tools/emit_tau_design_table.py
    python tools/emit_tau_design_table.py --check --paper <paper directory>

The default stdout is one UTF-8, LF-delimited region between sentinels, carrying two appendix floats.
--check also accepts CATCHBENCH_PAPER_DIR, compares bytes including whitespace and line endings, and
exits 1 with a readable delta on drift. No file is written, and nothing is spliced into the paper.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from emit_live_prefix_table import _METHODS  # noqa: E402

# Paths are joined with joinpath rather than with "/". The suite bans the division operator outright,
# so that no multiple, ratio or design count can be recomputed here, and a path join is the only
# place "/" would otherwise appear. Spelling it out costs a word and leaves the ban with no
# exceptions for a later reader to remember.
RECORD = ROOT.joinpath("live_tau_bar_design_results.json")

_APPENDIX = "09_appendix.tex"
_BEGIN = ("% BEGIN GENERATED tab:tau-design + tab:tau-scaling -- regenerate with: "
          "python tools/emit_tau_design_table.py")
_END = "% END GENERATED tab:tau-design + tab:tau-scaling"

# The appendix table that prints the twenty cells this one solves. Naming it in both captions is what
# lets a reader check a point estimate against the table it was published in, one page away, rather
# than take this table's word for it.
LIVE_TAU_TABLE = "tab:live-stream-tau"

# The clause in 06_discussion.tex this block supplies the number for. It sits here rather than inside
# a caption string because a literal inside a caption is invisible to a test: the same review pass
# that moved 72 entrants to 76 in the judge addendum's prose would move this one silently. A test in
# tests/test_emit_tau_design_table.py holds it to the manuscript.
DISCUSSION_CLAUSE = "measured scaling gives design targets of about 410 clusters"

# The two quantities Part B3 forbids, and the labels under which the caption states that neither was
# computed. The caption is required to say so, so a blanket ban on the words would ban the disclosure
# as well; the test counts occurrences against these labels plus the record's own sentences instead.
_REFUSED_LABELS = {
    "achieved_power": "Achieved power",
    "minimum_detectable_difference": "Minimum detectable difference",
}

# The committed verdict vocabulary, mapped to what a table cell prints. An unrecognised verdict is a
# record this file has never been told how to read, and printing it would print a word nobody chose.
_VERDICT_TEXT = {"separates_as_stated": "separates", "does_not_separate": "does not separate"}

# Row markers. Dagger: this cell had its own scaling ladder run, so its measured count uses its own
# exponent rather than the bracket. Asterisk: the solved count falls below the smallest rung the
# ladder reached, so it extrapolates outside the range where the scaling was checked.
_OWN_LADDER = r"$^{\dagger}$"
_BELOW_LADDER = r"$^{\ast}$"
_OWN_LADDER_SOURCE = "this cell's own measured exponent"

# Every character LaTeX would read as markup. A percent sign is the one of them these records
# genuinely carry, always inside "95%", and it is escaped on the way through: a bare one would
# comment out the rest of its line without a warning, which is the silent failure this module's
# docstring describes. Any other special means the record grew markup nobody chose for print, and
# refusing says so rather than guessing at an escape.
#
# The second set is spelled out rather than derived from the first, because the set-difference
# operator is banned here along with the arithmetic. A test holds the two to naming the same
# characters, so an edit to one that forgets the other fails rather than silently letting a special
# through.
_LATEX_SPECIALS = frozenset(r"\{}$&#^_~%")
_PROSE_SPECIALS = frozenset(r"\{}$&#^_~")
_ESCAPED_PERCENT = r"\%"

# What may reach a \texttt without being rewritten beyond its underscores and braces. The RNG label
# template and the tool paths are the strings that need it; anything else is refused for the same
# reason prose is.
_TEXTTT_ALLOWED = re.compile(r"[A-Za-z0-9 ._{}()*+:/-]+")


def _prose(text: str) -> str:
    r"""A record sentence, passed through to a caption with its percent signs escaped.

    The caption sentences that carry a claim about what this calculation is and is not are the
    record's own, word for word. Restating them here would put the reason a quantity was refused in
    two places, and the two would drift: the public record would say the calculation holds the point
    fixed while the manuscript said something adjacent to that.
    """
    intruders = sorted(set(text) & _PROSE_SPECIALS)
    if intruders:
        raise ValueError(f"a record sentence carries LaTeX special characters {intruders}; it "
                         f"cannot reach a caption unescaped: {text!r}")
    return text.replace("%", _ESCAPED_PERCENT)


def _texttt(text: str) -> str:
    r"""An identifier set in \texttt, with the underscores and braces LaTeX would otherwise eat."""
    if not _TEXTTT_ALLOWED.fullmatch(text):
        raise ValueError(f"not a printable identifier: {text!r}")
    for character in ("_", "{", "}"):
        text = text.replace(character, "\\" + character)
    return r"\texttt{" + text + "}"


def _and_list(parts: list[str]) -> str:
    """Join caption fragments the way a sentence does, so a two-stratum corpus reads as a sentence.

    The strata are read off the record rather than written here, so a corpus that grew a third domain
    would print three of them; joining with bare commas would then read as a list in one place and a
    sentence in another.
    """
    if not parts:
        raise ValueError("no fragments to list")
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
          "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
          "nineteen", "twenty")


def _word(count: int) -> str:
    """Spell a small count for caption prose, and fail rather than print a bare integer.

    Every count this caption states is derived from the rows in hand, so a record that grew or lost a
    cell moves the prose with the table. A roster this prose cannot describe stops the run: a caption
    that says eighteen of twenty over a table of nineteen rows is worse than no table.
    """
    if not 0 <= count < len(_WORDS):
        raise ValueError(f"no caption word for a count of {count}; the design table has outgrown "
                         f"the prose this caption was written for")
    return _WORDS[count]


def load() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _rows(record: dict) -> list[dict]:
    """The twenty design rows, in the order both this table and the record print them.

    Order is checked rather than imposed. The record lists the cells prefix-major and, inside a
    prefix, in the entrant order Table~\\ref{tab:live-stream-tau} prints; a row set that arrived in
    another order would still produce a plausible-looking table with its verdicts against the wrong
    entrants, and that is not a difference a reader could see.
    """
    rows = record["design_table"]
    methods = [method for method, _ in _METHODS[1:]]
    prefixes = sorted({row["prefix_percent"] for row in rows})
    wanted = [f"{prefix}.{method}" for prefix in prefixes for method in methods]
    found = [row["cell"] for row in rows]
    if found != wanted:
        raise ValueError(f"the record's cells are not the entrants of {LIVE_TAU_TABLE} in prefix "
                         f"order; got {found}, wanted {wanted}")
    for row in rows:
        if row["id"] != "live.tau.bar." + row["cell"]:
            raise ValueError(f"cell {row['cell']!r} does not name its own claim id {row['id']!r}")
        if row["verdict"] not in _VERDICT_TEXT:
            raise ValueError(f"{row['cell']}: unrecognised verdict {row['verdict']!r}")
        if row["separates_at_present_sample"] != (row["verdict"] != "does_not_separate"):
            raise ValueError(f"{row['cell']}: the separation flag contradicts the verdict "
                             f"{row['verdict']!r}")
        low, high = row["interval_95"]
        if (low < 0 < high) != (row["verdict"] == "does_not_separate"):
            raise ValueError(f"{row['cell']}: the verdict {row['verdict']!r} contradicts the "
                             f"interval [{low}, {high}] this table prints beside it")
    return rows


def _present_sample(record: dict, rows: list[dict]) -> dict:
    """The one sample every row was measured at, checked against the summary that states it.

    The caption prints the cluster and run counts once for the whole table, so a record whose rows
    disagreed with each other, or with its own summary, would be described by a caption true of some
    of it. Both directions are checked because they have separate sources: the rows come from the
    committed registry cells and the summary from the run that solved them.
    """
    summary = record["summary"]
    clusters = {row["n_clusters"] for row in rows}
    runs = {row["n_rows"] for row in rows}
    bars = {row["bar"] for row in rows}
    strata = {tuple(sorted(row["clusters_per_stratum"].items())) for row in rows}
    if len(clusters) != 1 or len(runs) != 1 or len(bars) != 1 or len(strata) != 1:
        raise ValueError(f"the rows disagree on the sample they were measured at: clusters "
                         f"{sorted(clusters)}, runs {sorted(runs)}, bars {sorted(bars)}")
    if clusters != {summary["present_clusters"]} or runs != {summary["present_rows"]}:
        raise ValueError(f"the rows sit at {sorted(clusters)} clusters and {sorted(runs)} runs; the "
                         f"summary says {summary['present_clusters']} and {summary['present_rows']}")
    if bars != {record["settings"]["bar"]}:
        raise ValueError(f"the rows are read against {sorted(bars)}; the settings say "
                         f"{record['settings']['bar']}")
    separating = sum(1 for row in rows if row["separates_at_present_sample"])
    withheld = sorted(row["cell"] for row in rows if not row["separates_at_present_sample"])
    if separating != summary["separating_at_present_sample"]:
        raise ValueError(f"{separating} rows separate at the present sample; the summary says "
                         f"{summary['separating_at_present_sample']}")
    if len(withheld) != summary["not_separating_at_present_sample"]:
        raise ValueError(f"{len(withheld)} rows do not separate; the summary says "
                         f"{summary['not_separating_at_present_sample']}")
    if withheld != sorted(record["unresolved_cells"]):
        raise ValueError(f"the rows that do not separate are {withheld}; the record names "
                         f"{sorted(record['unresolved_cells'])} as unresolved")
    if withheld != sorted(summary["needed_for_unresolved_cells"]):
        raise ValueError(f"the rows that do not separate are {withheld}; the summary carries design "
                         f"targets for {sorted(summary['needed_for_unresolved_cells'])}")
    if len(rows) != summary["cells"]:
        raise ValueError(f"{len(rows)} rows against a summary of {summary['cells']} cells")
    return {
        "clusters": summary["present_clusters"],
        "runs": summary["present_rows"],
        "bar": record["settings"]["bar"],
        "strata": dict(sorted(rows[0]["clusters_per_stratum"].items())),
        "separating": separating,
        "withheld": withheld,
    }


def _ladder(record: dict) -> dict:
    """The frozen ladder and the four measured exponents, each bound to the cell it was fitted on.

    The sixteen cells with no ladder of their own carry a bracket end rather than a fitted exponent,
    and the four that do carry their own. A row claiming its own exponent must therefore be one of
    the four and must carry that cell's fitted value; a mismatch would put a fitted number on a cell
    whose ladder was never run, which the printed dagger would then vouch for.
    """
    frozen = record["declaration"]["frozen_before_the_run"]
    check = record["scaling_check"]
    measured = check["per_cell"]
    if [cell["cell"] for cell in measured] != list(frozen["cells"]):
        raise ValueError(f"the ladders that were run, {[c['cell'] for c in measured]}, are not the "
                         f"cells the declaration froze, {list(frozen['cells'])}")
    for cell in measured:
        if list(cell["ladder"]) != list(frozen["ladder"]):
            raise ValueError(f"{cell['cell']}: ladder {cell['ladder']} against the frozen "
                             f"{frozen['ladder']}")
        low, high = cell["exponent_interval_95"]
        if not low < cell["exponent"] < high:
            raise ValueError(f"{cell['cell']}: exponent {cell['exponent']} outside its own interval "
                             f"[{low}, {high}]")
    if check["printed_figure_source"] not in ("closed_form", "measured_scaling"):
        raise ValueError(f"unrecognised printed figure source "
                         f"{check['printed_figure_source']!r}")
    pooling = check["pooling"]
    if (pooling["pooled_exponent"] is not None) != pooling["cells_agree"]:
        raise ValueError("a pooled exponent is reported without the agreement that licenses it, or "
                         "withheld despite it")
    return {"frozen": frozen, "check": check, "measured": measured, "pooling": pooling,
            "fixed": record["declaration"]["fixed_by_this_implementation"]}


def _exponent_of(cell: str, ladder: dict) -> float:
    for measured in ladder["measured"]:
        if measured["cell"] == cell:
            return measured["exponent"]
    raise ValueError(f"{cell}: no ladder was run for this cell")


def _needed(row: dict, key: str) -> str:
    """A solved cluster count, marked when it sits below the smallest rung the ladder reached."""
    needed = row[key]
    marker = _BELOW_LADDER if needed["below_measured_ladder"] else ""
    return str(needed["clusters"]) + marker


def _provenance(record: dict, rows: list[dict], sample: dict, ladder: dict) -> list[str]:
    """Comment lines binding each printed cell to the record's own full-precision value.

    The printed table rounds to three decimals and its counts are integers, so a reader who wants to
    re-solve a row, or a checker comparing two runs, has nothing to work from in the printed cells.
    These lines carry what the record stores. The two long notes on the shape of the fitted relation
    and on the within-cell error live here rather than in a caption because both carry characters
    LaTeX would read as markup, and a comment line is not parsed.
    """
    check, pooling = ladder["check"], ladder["pooling"]
    frozen = ladder["frozen"]
    inputs = record["inputs"]
    lines = [
        _BEGIN,
        f"% Source: tools/{RECORD.name}, schema {record['schema_version']}, generated by "
        f"{record['generated_by']}. This file computes nothing; every number below is read.",
        f"% Declaration: {record['declaration']['file']} Part {record['declaration']['part']}, "
        f"sha256 {inputs['declaration']['sha256']}.",
        f"% Registry: the twenty committed live.tau.bar cells of tools/"
        f"{Path(inputs['registry']['path']).name}, sha256 {inputs['registry']['sha256']}.",
        f"% Question: {record['question']}",
        f"% Corpus: {inputs['corpus']}",
        f"% Present sample: {sample['clusters']} task clusters, {sample['runs']} runs, "
        + ", ".join(f"{count} {name}" for name, count in sample["strata"].items())
        + f"; bar {sample['bar']}.",
        f"% Verdicts at the present sample: {sample['separating']} separate, "
        f"{len(sample['withheld'])} do not ({', '.join(sample['withheld'])}).",
        f"% Printed figure source: {check['printed_figure_source']}; reference exponent "
        f"{check['reference_exponent']}; assumption under test: {check['assumption']}.",
        f"% Ladder: {', '.join(str(rung) for rung in frozen['ladder'])} clusters; "
        f"{frozen['subsamples_per_rung']} subsamples per rung; {frozen['draws_per_subsample']} "
        f"draws per subsample; base seed {frozen['rng_base_seed']}; RNG label "
        f"{frozen['rng_label_template']}.",
        f"% Stratification: {frozen['stratification']}.",
        f"% Pooling: cells_agree={pooling['cells_agree']}; every_pair_overlaps="
        f"{pooling['every_pair_of_intervals_overlaps']}; Cochran Q={pooling['cochran_q']:.12f} on "
        f"{pooling['cochran_q_df']} df, p={pooling['cochran_q_p']:.12f}, alpha="
        f"{pooling['homogeneity_alpha']}; pooled_exponent={pooling['pooled_exponent']}.",
        f"% Pooling withheld because: {pooling['withheld_reason']}",
        f"% Shape of the relation: {check['shape_of_the_relation']}",
        f"% Within-cell error: {check['within_cell_error_is_probably_understated']}",
    ]
    for measured in ladder["measured"]:
        low, high = measured["exponent_interval_95"]
        fit = measured["rung_level_fit"]
        lines.append(
            f"% ladder {measured['cell']}: exponent={measured['exponent']:.12f}; "
            f"interval=[{low:.12f}, {high:.12f}]; se={measured['exponent_se']:.12f}; "
            f"df={measured['exponent_df']}; R2={fit['r_squared']:.12f}; "
            f"consistent_with_reference={measured['consistent_with_minus_half']}; "
            f"declared_because={measured['declared_because']}")
    for row in rows:
        low, high = row["interval_95"]
        measured, closed = row["needed_measured"], row["needed_closed_form"]
        lines.extend([
            f"% {row['cell']}: id={row['id']}; auc={row['auc']:.12f}; "
            f"point={row['point_auc_minus_bar']:.12f}; sd={row['bootstrap_sd']:.12f}; "
            f"interval=[{low:.12f}, {high:.12f}]; verdict={row['verdict']}",
            f"% {row['cell']}: measured clusters={measured['clusters']}; "
            f"exact={measured['clusters_exact']:.12f}; runs={measured['runs']}; "
            f"multiple={measured['multiple_of_present_sample']:.12f}; "
            f"exponent={row['needed_measured_exponent']:.12f}; "
            f"below_ladder={measured['below_measured_ladder']}; "
            f"exponent_source={row['needed_measured_exponent_source']}",
            f"% {row['cell']}: closed-form clusters={closed['clusters']}; "
            f"exact={closed['clusters_exact']:.12f}; runs={closed['runs']}; "
            f"multiple={closed['multiple_of_present_sample']:.12f}; "
            f"below_ladder={closed['below_measured_ladder']}",
        ])
    return lines


def _printed_source_sentence(ladder: dict) -> str:
    """Which extrapolation the manuscript is reading, stated in the branch the record is in.

    Part B4 requires the manuscript to say this either way, and the two branches are not each other's
    negation: under the closed form the measured scaling is the check that passed, and under the
    measured scaling the closed form is an idealized comparison beside a printed number that came
    from somewhere else. A caption written for one branch would be wrong rather than incomplete in
    the other, so the sentence is chosen here and both branches are exercised by the tests.
    """
    if ladder["check"]["printed_figure_source"] == "closed_form":
        return ("Measured is the number this table reads, and the closed form agrees with it: every "
                "measured exponent is consistent with the reference, so the printed figure is the "
                "closed-form extrapolation.")
    return ("Measured is the number this table reads. Not every measured exponent is consistent "
            "with the reference, so the printed figure comes from the measured scaling and the "
            "closed form stands beside it as the idealized comparison.")


def _pooling_sentences(ladder: dict) -> list[str]:
    """What the declared pooling rule returned, in whichever branch it landed."""
    pooling = ladder["pooling"]
    overlap = ("Every pair of exponent intervals overlaps"
               if pooling["every_pair_of_intervals_overlaps"]
               else "At least one pair of exponent intervals does not overlap")
    line = (overlap + f", and Cochran's Q is {pooling['cochran_q']:.2f} on "
            f"{pooling['cochran_q_df']} degrees of freedom "
            f"(p = {pooling['cochran_q_p']:.4f}, alpha {pooling['homogeneity_alpha']}).")
    if pooling["cells_agree"]:
        low, high = pooling["pooled_exponent_interval_95"]
        return [line, f"The four therefore agree and the pooled exponent is "
                      f"{pooling['pooled_exponent']:.3f} [{low:.3f}, {high:.3f}]."]
    return [line, "No pooled exponent is reported: " + _prose(pooling["withheld_reason"]) + "."]


def _design_caption(record: dict, rows: list[dict], sample: dict, ladder: dict) -> list[str]:
    """One clause per line, so a caption edit shows as a one-line delta in --check."""
    summary = record["summary"]
    interpretation = record["interpretation"]
    refused = record["not_computed"]
    bracket = sorted({row["needed_measured_exponent_source"] for row in rows
                      if row["needed_measured_exponent_source"] != _OWN_LADDER_SOURCE})
    if len(bracket) != 1:
        raise ValueError(f"the rows without a ladder of their own cite {len(bracket)} different "
                         f"exponent sources; the caption names one: {bracket}")
    targets = []
    for cell in sample["withheld"]:
        needed = summary["needed_for_unresolved_cells"][cell]
        targets.append(f"{cell} at {needed['measured_clusters']:,} clusters "
                       f"({needed['measured_runs']:,} runs, "
                       f"{needed['multiple_of_present_sample']:.1f} times the present sample, "
                       f"{needed['closed_form_clusters']:,} clusters under the closed form)")
    return [
        r"\caption{" + _prose(record["question"]),
        f"The present sample is {sample['clusters']:,} task clusters and {sample['runs']:,} runs, "
        + _and_list([f"{count} {name}" for name, count in sample["strata"].items()])
        + ".",
        _word(sample["separating"]).capitalize() + " of the " + _word(len(rows))
        + f" cells already separate from the {sample['bar']:.2f} bar at that sample and "
        + _word(len(sample["withheld"])) + " do not: " + "; ".join(targets) + ".",
        f"Point is the committed ROC-AUC minus the {sample['bar']:.2f} bar, SD the committed "
        r"clustered bootstrap standard deviation, and the interval and verdict the committed ones; "
        r"Table~\ref{" + LIVE_TAU_TABLE + "} prints the same cells on the same scale.",
        _printed_source_sentence(ladder),
        "Closed form assumes what the declaration set out to check, that "
        + _prose(ladder["check"]["assumption"]) + ".",
        r"A row marked " + _OWN_LADDER + " had its own scaling ladder run (Table~\\ref{tab:tau-"
        r"scaling}) and uses that cell's own exponent; every other row uses "
        + _prose(bracket[0]) + ".",
        r"A count marked " + _BELOW_LADDER + f" falls below {min(ladder['frozen']['ladder'])} "
        r"clusters, the smallest rung the ladder measured, so it extrapolates outside the range "
        r"where the scaling was checked.",
        _prose(interpretation["design_target_only"]),
        _prose(interpretation["fixed_point_estimates"]),
        _prose(interpretation["normal_idealization"]),
        _REFUSED_LABELS["achieved_power"] + ": " + _prose(refused["achieved_power"]),
        _REFUSED_LABELS["minimum_detectable_difference"] + ": "
        + _prose(refused["minimum_detectable_difference"]),
        _prose(interpretation["exploratory"]),
        r"Generated by " + _texttt("tools/emit_tau_design_table.py") + ".}",
    ]


def _scaling_caption(ladder: dict) -> list[str]:
    """The ladder Part B5 asks for beside the exponents it produced."""
    frozen, check = ladder["frozen"], ladder["check"]
    fixed = ladder["fixed"]
    return [
        r"\caption{The measured scaling of the clustered bootstrap standard deviation, on the "
        + _word(len(ladder["measured"])) + " cells the declaration named before any subsample was "
        r"drawn.",
        "The assumption under test is that " + _prose(check["assumption"]) + ", against a reference "
        f"exponent of {check['reference_exponent']}.",
        "Ladder: " + _and_list([str(rung) for rung in frozen["ladder"]]) + " clusters; "
        + _prose(frozen["stratification"]) + "; " + _word(frozen["subsamples_per_rung"])
        + f" subsamples per rung and {frozen['draws_per_subsample']:,} bootstrap draws inside each "
        f"subsample; base seed {frozen['rng_base_seed']}, RNG label "
        + _texttt(frozen["rng_label_template"]) + ".",
        "Each exponent is " + _prose(fixed["exponent_estimator"]) + ".",
        "Consistency rule, fixed before the fits: " + _prose(fixed["consistency_rule"]) + ".",
        "Pooling rule, fixed before the fits: " + _prose(fixed["pooling_rule"]) + ".",
        *_pooling_sentences(ladder),
        _prose(check["within_cell_error_is_probably_understated"]),
        "These are the rows Table~\\ref{tab:tau-design} marks " + _OWN_LADDER + ".",
        r"Generated by " + _texttt("tools/emit_tau_design_table.py") + ".}",
    ]


def _design_float(record: dict, rows: list[dict], sample: dict, ladder: dict) -> list[str]:
    """The twenty-row float: one committed reading and both extrapolations of it per cell."""
    printed = dict(_METHODS)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        *_design_caption(record, rows, sample, ladder),
        r"\label{tab:tau-design}",
        r"\begin{tabular}{@{}llrrclrr@{}}",
        r"\toprule",
        r"& & & & & & \multicolumn{2}{c}{Clusters needed} \\",
        r"\cmidrule(l){7-8}",
        r"Prefix & Method & Point & SD & 95\% interval & Verdict & Measured & Closed form \\",
        r"\midrule",
    ]
    previous = None
    for row in rows:
        if previous is not None and row["prefix_percent"] != previous:
            lines.append(r"\midrule")
        previous = row["prefix_percent"]
        own = _OWN_LADDER if row["needed_measured_exponent_source"] == _OWN_LADDER_SOURCE else ""
        low, high = row["interval_95"]
        lines.append(" & ".join([
            str(row["prefix_percent"]) + r"\%",
            printed[row["method"]] + own,
            "$" + f"{row['point_auc_minus_bar']:.3f}" + "$",
            f"{row['bootstrap_sd']:.3f}",
            "$[" + f"{low:.3f}" + ", " + f"{high:.3f}" + "]$",
            _VERDICT_TEXT[row["verdict"]],
            _needed(row, "needed_measured"),
            _needed(row, "needed_closed_form"),
        ]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return lines


def _scaling_float(ladder: dict) -> list[str]:
    """The four laddered cells, so the exponent every design count rests on is visible."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        *_scaling_caption(ladder),
        r"\label{tab:tau-scaling}",
        # The reason strings come from the declaration and are long on purpose, so the column
        # wraps rather than running the row off the measure. Presentation only: the strings
        # themselves are read from the record unchanged.
        r"\begin{tabular}{@{}lp{0.30\linewidth}rcc@{}}",
        r"\toprule",
        r"Cell & Declared because & Exponent & 95\% interval & Consistent \\",
        r"\midrule",
    ]
    for measured in ladder["measured"]:
        low, high = measured["exponent_interval_95"]
        lines.append(" & ".join([
            measured["cell"],
            measured["declared_because"],
            "$" + f"{measured['exponent']:.3f}" + "$",
            "$[" + f"{low:.3f}" + ", " + f"{high:.3f}" + "]$",
            "yes" if measured["consistent_with_minus_half"] else "no",
        ]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return lines


def table(record: dict) -> str:
    """Both appendix floats between one pair of sentinels, in the order the appendix reads them."""
    rows = _rows(record)
    sample = _present_sample(record, rows)
    ladder = _ladder(record)
    for row in rows:
        if row["needed_measured_exponent_source"] != _OWN_LADDER_SOURCE:
            continue
        fitted = _exponent_of(row["cell"], ladder)
        if row["needed_measured_exponent"] != fitted:
            raise ValueError(f"{row['cell']} claims its own measured exponent "
                             f"{row['needed_measured_exponent']} but its ladder fitted {fitted}")
    lines = _provenance(record, rows, sample, ladder)
    lines += _design_float(record, rows, sample, ladder)
    lines.append("")
    lines += _scaling_float(ladder)
    lines.append(_END)
    return "\n".join(lines)


def build() -> str:
    """The one path from record to block, so a caller cannot assemble a different table than --check.

    ``emit_addendum_table`` learned this the hard way: its test suite once called the formatter
    directly, skipped the step ``build`` performs, and confirmed a block against itself while the
    command line published something else.
    """
    return table(load())


def check(paper: Path, generated: str) -> int:
    try:
        raw = paper.joinpath(_APPENDIX).read_bytes()
    except OSError as exc:
        print(f"STALE {_APPENDIX}: cannot read appendix: {exc}")
        return 1
    markers = []
    for marker in (_BEGIN, _END):
        found = list(re.finditer(rb"(?m)^" + re.escape(marker.encode("utf-8")) + rb"\r?$", raw))
        if len(found) != 1:
            print(f"STALE {_APPENDIX}: tau-design marker {marker!r} appears "
                  f"{len(found)} times, expected once")
            return 1
        markers.append(found[0])
    begin, end = markers
    if end.start() < begin.start():
        print(f"STALE {_APPENDIX}: tau-design end marker precedes its begin marker")
        return 1
    actual = raw[begin.start():end.start() + len(_END)]
    wanted = generated.encode("utf-8")
    if actual != wanted:
        print(f"STALE {_APPENDIX}: tau-design block differs from the generated block")
        for index, (got, want) in enumerate(zip(actual.splitlines(keepends=True),
                                                wanted.splitlines(keepends=True)), 1):
            if got != want:
                print(f"first difference at block line {index}: paper={got!r}; generated={want!r}")
                break
        print("\n".join(difflib.unified_diff(
            actual.decode("utf-8", errors="replace").splitlines(), generated.splitlines(),
            fromfile=f"paper/{_APPENDIX}", tofile="generated/tab:tau-design", lineterm="")))
        print("Regenerate with: python tools/emit_tau_design_table.py")
        return 1
    print("paper is current: tab:tau-design and tab:tau-scaling (twenty design rows, four ladders)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="check the marked appendix block instead of printing it")
    parser.add_argument("--paper", default=os.environ.get("CATCHBENCH_PAPER_DIR"),
                        help="paper source directory (or set CATCHBENCH_PAPER_DIR)")
    args = parser.parse_args()
    if args.check and not args.paper:
        parser.error("--check needs --paper <dir> or CATCHBENCH_PAPER_DIR")
    try:
        generated = build()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"ERROR: tau-bench bar design table: {exc}", file=sys.stderr)
        return 1
    if args.check:
        return check(Path(args.paper), generated)
    sys.stdout.buffer.write((generated + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
