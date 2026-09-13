r"""Emit the judge addendum as its own appendix block, scored by tools/score_judge_addendum.py.

Every arm in ``score_judge_addendum.TABLE_COHORT`` over the same 126 Who&When runs and the same
``all_at_once`` protocol as the eleven published judges, plus two published caches pulled in so the
comparison reads in one place. The roster is explicit, and each member is read from the exact path
its protocol and label name: a cache written later must not join the historical table without a
decision, and a cache of another protocol must not stand in for one that is missing. Rows print the provider model
name and carry the route in the Channel column, so one model reached two ways reads as two rows of
one model. The two same-channel repeats are held out and reported in the caption as the
disagreement floor. Every number here is imported; ``rank_arms`` loads, scores and orders the arms,
and this file formats what it returns.

Nothing in this file computes Top-1, Top-3, or a hit count. That is the point rather than a
convenience. The declared outcome is ``rank == 1`` after ``np.argsort(-scores, kind="stable")`` over
the per-step vector, which is not the reading the generation script prints, and the two can disagree
on the 20 runs whose gold mistake is step 0. A second implementation would be wrong in a way that
still looks plausible, and it would be wrong in the paper rather than in a console line.

The interval is the one exception, and it is drawn here rather than taken from the scorer. The
scorer reports a Wilson interval, which is its own documented choice for a console ranking. The
appendix already prints two of these twelve arms in Table~\\ref{tab:protocol} under Procedure BIN, so
taking the scorer's would have put one score under two interval constructions about 640 source lines
apart. ``_bin_interval`` reuses ``emit_protocol_table.py``'s label rule and base seed, and the two
published arms therefore reproduce the cells that table already prints.

The block declares no registered contrast. It reports each arm's own score with a BIN interval on
that arm's own proportion, and it tests no difference between any two arms. That is the reporting
frame settled on 2026-09-06 by surveying comparable agent benchmarks, recorded in the scorer's
module docstring; the caption states it because the M4 completion test asks the printed block to
state it, so the sentence is a requirement rather than a hedge.

The addendum caches deliberately sit outside ``data/llm_judge/``, which the board globs. They
therefore print no board row, and the nine boards, the 72 entrants, and the 138 recorded comparisons
are unchanged by this table.

Mean reciprocal rank stays in the block's comment lines instead of the printed body. The appendix
supplies no interval on part of a table, an MRR interval is not among the quantities the addendum
declares, and printing a bare MRR column beside two interval columns would read as a complete table
with one column silently weaker. The value is still emitted, at full precision, where a checker can
read it.

Usage::

    python tools/emit_addendum_table.py
    python tools/emit_addendum_table.py --check --paper <paper directory>

The default stdout is one UTF-8, LF-delimited appendix table between sentinels. --check also accepts
CATCHBENCH_PAPER_DIR, compares bytes including whitespace and line endings, and exits 1 with a
readable delta on drift. No file is written, and nothing is spliced into the paper.
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from score_judge_addendum import (  # noqa: E402
    LATE_RUN_HISTORY, REPLICATE_LABELS, TABLE_COHORT, rank_arms,
)
from statistical_tests import _rng_for  # noqa: E402

# Procedure BIN, the per-arm interval the rest of this appendix uses. The label rule and the base
# seed are emit_protocol_table.py's, so the two published arms here reproduce the intervals
# tab:protocol already prints for them rather than carrying a second construction of the same cell.
# Top-3 has no cell in tab:protocol; it extends the same rule with its own metric suffix.
BASE_SEED = 20260907
DRAWS = 10000


def _bin_interval(model: str, metric: str, hits: int, n: int) -> tuple[float, float]:
    """The 2.5 and 97.5 percentiles of 10,000 binomial draws at hits/n, seeded from the label."""
    rng = _rng_for(f"per-arm/bin/whoandwhen/all_at_once/{model}/{metric}", BASE_SEED)
    draws = rng.binomial(n, hits / n, size=DRAWS) / n
    low, high = np.percentile(draws, [2.5, 97.5], method="linear")
    return float(low), float(high)


def _with_bin_intervals(arms: list[dict]) -> list[dict]:
    """Replace the scorer's own interval with BIN, leaving every score it computed untouched."""
    out = []
    for arm in arms:
        arm = dict(arm)
        for metric in ("top1", "top3"):
            low, high = _bin_interval(arm["label"], metric, arm[f"{metric}_hits"], arm["n"])
            arm[f"{metric}_lo"], arm[f"{metric}_hi"] = low, high
        out.append(arm)
    return out


_APPENDIX = "09_appendix.tex"

# Counts the caption states about the rest of the paper. They sit here rather than inside the caption
# strings because a literal inside a caption is invisible to a test: a review pass changed 72 to 76
# and eleven to fourteen in that prose and the suite stayed green on both. Each is bound to its source
# by a test in tests/test_emit_addendum_table.py, so a future change to the paper fails here.
PUBLISHED_JUDGES = 11        # all-at-once caches in data/llm_judge/
ARENA_ENTRANTS = 72          # the entrant count the paper prints
RECORDED_COMPARISONS = 138   # the size of the recorded-comparison registry
NULL_TOP_FLOOR = 20          # runs whose gold mistake is step 0, so a null top still wins Top-1
_BEGIN = ("% BEGIN GENERATED tab:judge-addendum -- regenerate with: "
          "python tools/emit_addendum_table.py")
_END = "% END GENERATED tab:judge-addendum"

# Labels set in \texttt wherever they occur in the imported disclosure prose.
_TEXTTT_LABELS = ("gpt-5.6-sol", "gpt-6-astra")

# Every character LaTeX would read as markup. The disclosure is prose written for a console, so it
# carries none of them today; the guard exists so that a later edit to the constant fails here
# rather than in a build log.
_LATEX_SPECIALS = frozenset("\\{}$&#^_~%")


def caption_disclosure(lines: tuple[str, ...] = LATE_RUN_HISTORY) -> str:
    r"""The scorer's console disclosure, rejoined as caption prose with its labels in \texttt.

    The console prints one line per element and every break falls at a word boundary, so a single
    space restores the paragraph. Reading the constant rather than restating it is what stops the
    public report and the manuscript from carrying different reasons for the same non-run, which is
    the failure Erratum 2 of the frozen declaration was written to prevent.
    """
    prose = " ".join(line.strip() for line in lines)
    intruders = sorted(set(prose) & _LATEX_SPECIALS)
    if intruders:
        raise ValueError("the not-run disclosure carries LaTeX special characters %r; escape them "
                         "before this text reaches a caption" % intruders)
    for label in _TEXTTT_LABELS:
        prose = prose.replace(label, r"\texttt{%s}" % label)
    return prose


def _provenance(arms: list[dict], n_runs: int) -> list[str]:
    """Comment lines binding each printed cell to the scorer's full-precision value."""
    n_addendum = sum(1 for arm in arms if arm["source"] == "addendum")
    n_published = sum(1 for arm in arms if arm["source"] == "published")
    lines = [
        _BEGIN,
        "%% Source: data/llm_judge_addendum/whoandwhen__all_at_once__<label>.json for the %s "
        "addendum arms, read by path;" % _word(n_addendum),
        "%% data/llm_judge/ through load_cache for the %s published arms, whose keys need the "
        "legacy map." % _word(n_published),
        "% Scored by tools/score_judge_addendum.py, which calls the _score_vector and "
        "_rank_metrics of src/catchbench/llm_judge.py,",
        "% the same two calls LLMJudgeLocalization.evaluate makes for the published board. This "
        "file computes no score of its own.",
        "% Top-1 outcome: rank == 1 after np.argsort(-scores, kind='stable') over the per-step "
        "vector. The console reading of the",
        "% generation script asks a different question and is not this quantity.",
        f"% Intervals: Procedure BIN on each arm's own proportion over n={n_runs}; "
        f"draws={DRAWS}; quantile=linear; RNG=PCG64; base_seed={BASE_SEED}; "
        "label=per-arm/bin/whoandwhen/all_at_once/<model>/<metric>; marginal, not simultaneous.",
        f"% Roster: {len(TABLE_COHORT)} cohort labels; held-out same-channel repeats: "
        f"{', '.join(REPLICATE_LABELS)}.",
    ]
    for arm in arms:
        lines.extend([
            "%% %s: source=%s; channel=%s; position=%d; above=%d; n=%d"
            % (arm["label"], arm["source"], arm["channel"], arm["seq"], arm["above"], arm["n"]),
            "%% %s: top1=%d/%d=%.12f; interval=[%.12f, %.12f]"
            % (arm["label"], arm["top1_hits"], arm["n"], arm["top1"],
               arm["top1_lo"], arm["top1_hi"]),
            "%% %s: top3=%d/%d=%.12f; interval=[%.12f, %.12f]; mrr=%.12f"
            % (arm["label"], arm["top3_hits"], arm["n"], arm["top3"],
               arm["top3_lo"], arm["top3_hi"], arm["mrr"]),
        ])
    return lines


def _word(count: int) -> str:
    """Spell a small count for caption prose, and fail rather than print a bare integer.

    The counts a reader meets in this caption are all derived from the arms in hand, so an added or
    removed cache moves the prose with the table instead of leaving a stale number behind it.
    """
    words = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
             "eleven", "twelve")
    if not 0 <= count < len(words):
        raise ValueError("no caption word for a count of %d; the addendum has outgrown the prose "
                         "this caption was written for" % count)
    return words[count]


def _caption(n_runs: int, arms: list[dict]) -> list[str]:
    """One clause per line, so a caption edit shows as a one-line delta in --check."""
    published = [arm["label"] for arm in arms if arm["source"] == "published"]
    addendum = [arm["label"] for arm in arms if arm["source"] == "addendum"]
    if len(published) != 2:
        raise ValueError("the caption names two published arms and this ranking carries %d (%s); "
                         "rewrite the sentence before changing the roster"
                         % (len(published), ", ".join(published) or "none"))
    # Two counts the addendum caches cannot supply. Eleven published judges: the arena roster of
    # Table~\ref{tab:protocol} and the abstract. Twenty runs whose gold mistake is step 0: the
    # null-top floor of research/catchbench-m4-declaration-v3.md, restated in the scorer docstring.
    return [
        r"\caption{Judge addendum on POST localization: the same %d Who\&When runs and "
        r"all-at-once protocol as the %s published judges."
        % (n_runs, _word(PUBLISHED_JUDGES)),
        r"%s addendum arms are shown with the published \texttt{%s} and \texttt{%s} references."
        % (_word(len(addendum)).capitalize(), published[0], published[1]),
        # One percent sign, not two: this string carries no % interpolation, so a doubled sign
        # reaches LaTeX and the second one comments out the rest of the generated line.
        r"Top-1 and Top-3 carry marginal 95\% Procedure BIN intervals, and the references' Top-1 "
        r"cells repeat Table~\ref{tab:protocol}.",
        r"This block declares no registered contrast and tests no difference between any two arms; "
        r"interval overlap establishes neither equivalence nor the absence of an improvement.",
        r"Rows run in descending Top-1 order and carry no rank: no contrast is declared here "
        r"and no difference between arms is tested.",
        r"A Top-1 hit means the gold step ranks first after a stable sort of the per-step scores, "
        r"as on the published board; a null output takes the all-zero ranking and can still take "
        r"credit on the %d runs labelled at step 0." % NULL_TOP_FLOOR,
        r"The addendum is excluded from the nine boards, the %d entrants and the %d recorded "
        r"comparisons." % (ARENA_ENTRANTS, RECORDED_COMPARISONS),
        r"Two arms were generated a second time on the same channel and are held out of this "
        r"table; those repeats name a different step on 9 and 6 of the runs, which is the floor "
        r"any reading across routes or generations has to clear.",
        caption_disclosure(),
        r"Addendum channels are reconstructed from declared configurations; "
        r"\texttt{not-recorded} marks an absent channel record.",
        r"Mean reciprocal rank is carried in this block's comment lines rather than printed, so "
        r"every printed score carries its interval.",
        r"Generated by \texttt{tools/emit\_addendum\_table.py}.}",
    ]


def _cell(point: float, low: float, high: float) -> str:
    r"""A score beside its interval, on one line.

    The stacked form of Table~\ref{tab:protocol} sets the point estimate on the line above its own
    row, because a ``\shortstack`` rests on its last line. There every numeric column is stacked and
    the row carries one short label, so the offset reads as a column habit. Here four of the seven
    columns are single-line text, and a stacked score would float half a row above the model it
    belongs to, with the top row's estimate landing against the header rule. One line per row costs
    horizontal space and removes that misreading.
    """
    return r"%.3f {\scriptsize $[%.3f, %.3f]$}" % (point, low, high)


def table(arms: list[dict], n_runs: int) -> str:
    """One appendix float: every scored arm in the roster, each under its provider model name.

    The same model appears once per route, distinguished by the Channel column rather than by the
    cache label, and the two same-channel repeats are held out; the caption carries their
    disagreement counts instead.
    """
    lines = _provenance(arms, n_runs)
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        *_caption(n_runs, arms),
        r"\label{tab:judge-addendum}",
        r"\begin{tabular}{@{}lllcc@{}}",
        r"\toprule",
        r"Model & Channel & Source & Top-1 & Top-3 \\",
        r"\midrule",
    ]
    for arm in arms:
        display = TABLE_COHORT.get(arm["label"], arm["label"])
        lines.append(r"\texttt{%s} & \texttt{%s} & %s & %s & %s \\"
                     % (display, arm["channel"], arm["source"],
                        _cell(arm["top1"], arm["top1_lo"], arm["top1_hi"]),
                        _cell(arm["top3"], arm["top3_lo"], arm["top3_hi"])))
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        _END,
    ]
    return "\n".join(lines)


def build() -> str:
    """The one path from caches to block, so a caller cannot assemble a different table than --check.

    An earlier version let the test suite call ``table(*rank_arms())`` directly. That skipped the
    interval substitution and produced a block with the scorer's Wilson bounds, which the tests then
    confirmed against themselves while the command line published something else.
    """
    arms, n_runs = rank_arms()
    return table(_with_bin_intervals(arms), n_runs)


def check(paper: Path, generated: str) -> int:
    try:
        raw = (paper / _APPENDIX).read_bytes()
    except OSError as exc:
        print(f"STALE {_APPENDIX}: cannot read appendix: {exc}")
        return 1
    markers = []
    for marker in (_BEGIN, _END):
        found = list(re.finditer(rb"(?m)^" + re.escape(marker.encode("utf-8")) + rb"\r?$", raw))
        if len(found) != 1:
            print(f"STALE {_APPENDIX}: judge-addendum marker {marker!r} appears "
                  f"{len(found)} times, expected once")
            return 1
        markers.append(found[0])
    begin, end = markers
    if end.start() < begin.start():
        print(f"STALE {_APPENDIX}: judge-addendum end marker precedes its begin marker")
        return 1
    actual = raw[begin.start():end.start() + len(_END)]
    wanted = generated.encode("utf-8")
    if actual != wanted:
        print(f"STALE {_APPENDIX}: judge-addendum table differs from the generated block")
        for index, (got, want) in enumerate(zip(actual.splitlines(keepends=True),
                                               wanted.splitlines(keepends=True)), 1):
            if got != want:
                print(f"first difference at block line {index}: paper={got!r}; generated={want!r}")
                break
        print("\n".join(difflib.unified_diff(
            actual.decode("utf-8", errors="replace").splitlines(), generated.splitlines(),
            fromfile=f"paper/{_APPENDIX}", tofile="generated/tab:judge-addendum", lineterm="")))
        print("Regenerate with: python tools/emit_addendum_table.py")
        return 1
    print("paper is current: tab:judge-addendum (%d scored arms in the roster)" % len(TABLE_COHORT))
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
        print(f"ERROR: judge addendum: {exc}", file=sys.stderr)
        return 1
    if args.check:
        return check(Path(args.paper), generated)
    sys.stdout.buffer.write((generated + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
