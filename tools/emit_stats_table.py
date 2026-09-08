"""Emit the paper's contrast matrix from the shipped results, so the two cannot drift.

The appendix prints every comparison this benchmark records. That table is generated here and
compared byte for byte by ``--check``, because it is the printed home of every claim: without it a
body cut can delete a claim's only printed estimate and interval while a checker stays green.

Usage::

    python tools/emit_stats_table.py             # the contrast matrix block for Appendix F
    python tools/emit_stats_table.py --contrasts # the same block, named explicitly
    python tools/emit_stats_table.py --check     # exit 1 if the paper is stale, printing the delta

``--check`` needs ``--paper <dir>`` or the ``CATCHBENCH_PAPER_DIR`` environment variable.

What this generator stopped printing, and why
---------------------------------------------

The table used to carry two more columns, a Holm-adjusted $p$ and a verdict word. Both are gone,
and no column replaces them.

A benchmark's product is a measurement. Whether a comparison reaches significance is not the
benchmark's claim to make, and a non-significant result is as much a finding as a significant one.
A family-wise correction exists to control error across a set of decisions; this paper makes no
decisions from these rows, so the correction served nothing and the verdict words were the
adjudication itself. ``tools/statistical_tests.py`` still computes both, and
``statistical_tests_results.json`` still records every raw and adjusted $p$-value. The record is a
record. What changed is what the manuscript prints. Deleting the ``p`` column also removed
``_p(0.0)``, which looped forever on an exact zero and sat behind a live path.

The family table went with them. Its only content was the family structure, which existed to define
the multiplicity that is no longer corrected, so ``tab:stat-families`` is deleted rather than reduced
and this module no longer checks the count sentences that table fed.

Grouping is what survived, and it is presentation rather than statistics. A reader navigates a long
and heterogeneous table by the group headers, which name each group's scope, metric, interval
construction, and sampling axis. Two constructions are now in play: tau-bench rows carry a
task-clustered stratified percentile bootstrap, because that corpus is a task-by-model grid whose
runs are not independent draws, while the rest carry their original run-level constructions. Where a
group mixes the two, the header says so and every row in it names its own construction, the same way
a group that mixes metrics names the metric on each row. A header that named one construction over
rows built two ways would be a false statement in the paper, which is why the mixture is handled
here rather than left to the reader.

What ``--check`` guards
-----------------------

The required-file check, the statistics-section bounds, and the hidden-content guard are deliberately
not folded into any per-file pattern loop. In the previous version they were incidental effects of
looping over the prose patterns, so removing the patterns would have made ``check()`` return 0 for an
empty paper directory and for a paper whose statistics section label had been deleted. A checker that
passes on a corrupted paper is worse than no checker, because the green result is what gets trusted.

The section's end marker used to be ``\\label{tab:stat-families}``. With that table deleted, the
generated block's own begin sentinel is the boundary: the statistics section is the prose between
``\\label{app:stats}`` and the matrix it introduces. Each marker must stand alone on its line, so a
commented-out label is a failure rather than a pass.

``03_benchmark.tex`` is no longer read. Every statement this generator pinned there was a count of
the deleted family table, and a required-file check with nothing behind it fails for a reason it
cannot name.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "statistical_tests_results.json"

_APPENDIX = "09_appendix.tex"

# --- the contrast matrix ------------------------------------------------------------------------
#
# The block is compared byte for byte instead of row by row: any edit inside the markers, including
# one that still parses as a table, is staleness.
_CONTRASTS_BEGIN = ("% BEGIN GENERATED tab:all-contrasts -- regenerate with: "
                    "python tools/emit_stats_table.py --contrasts")
_CONTRASTS_END = "% END GENERATED tab:all-contrasts"

# The statistics section introduces the matrix, so the matrix's own begin sentinel bounds it now
# that the family table it used to end at is gone. Both markers must stand alone on their line;
# see _required_marker.
_STATS_SECTION = (r"\label{app:stats}", _CONTRASTS_BEGIN)

# Anything outside this set in a claim label, a group id, or a method name is a character whose
# LaTeX meaning was not considered, so it fails review rather than being escaped by guesswork.
_LATEX_ESCAPES = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$"}
_UNHANDLED = re.compile(r"[\\^~{}]")

# Width bookkeeping. Dropping two columns freed 3.15cm of fixed width and 12pt of \tabcolsep, so the
# contrast column absorbs 8.87 - 5.30 = 3.57cm and the table's total width is unchanged. That is
# what keeps the 13.6cm group-header rule flush with the table rather than overhanging it.
_CONTRAST_COLUMN = "8.87cm"
_HEADER_RULE = "13.6cm"
_COLUMNS = 5

_CAPTION = (
    r"\caption{Every recorded comparison, by group. $A$ and $B$ are the two entrants named in the "
    r"contrast, in that order, and the interval on their difference is built under the construction "
    r"and sampling axis the group header states. These rows are measurements reported with their "
    r"uncertainty: no multiplicity correction is applied, no family-wise error control is claimed, "
    r"and the text draws no conclusion from whether an interval excludes zero. The unadjusted and "
    r"adjusted $p$-values stay in the released record, "
    r"\texttt{tools/statistical\_tests\_results.json}, and are not printed here. This table is the "
    r"printed home of every claim in Appendix~\ref{app:stats}; the body reports the subset it "
    r"argues from. Generated by \texttt{tools/emit\_stats\_table.py -{}-contrasts}, whose "
    r"\texttt{-{}-check} mode runs before a submission and fails when the paper falls behind the "
    r"shipped results.}\label{tab:all-contrasts}\\"
)


def load() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def inventory(data: dict) -> dict[str, int]:
    """What the generated block contains, for the CLI's own diagnostics."""
    return {"groups": len(data["comparison_families"]), "contrasts": len(data["claims"])}


def _tex(text: str) -> str:
    if _UNHANDLED.search(text):
        raise SystemExit(f"claim text needs escaping this generator does not do: {text!r}")
    return "".join(_LATEX_ESCAPES.get(character, character) for character in text)


def _construction(interval: dict) -> str:
    """The interval's construction and sampling axis, as one phrase."""
    return "%s, %s" % (_tex(interval["method"]), _tex(interval["axis"]))


def contrasts(data: dict) -> str:
    """The whole longtable float, markers included, ready to paste into the appendix."""
    grouped: dict[str, list[dict]] = {}
    for claim in data["claims"]:
        grouped.setdefault(claim["family"], []).append(claim)

    header = r"Contrast & $A$ & $B$ & $A-B$ & 95\% CI \\"
    rule = r"\multicolumn{%d}{@{}p{%s}@{}}" % (_COLUMNS, _HEADER_RULE)
    out = [
        _CONTRASTS_BEGIN,
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        # Fixed widths on the two text-shaped columns. Natural-width l and c columns overflowed
        # \textwidth by 234pt, because a wide interval has nowhere to wrap.
        r"\begin{longtable}{@{}p{%s}rrr"
        r">{\centering\arraybackslash}p{2.15cm}@{}}" % _CONTRAST_COLUMN,
        _CAPTION,
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        rule + r"{\emph{Continued from the previous page.}}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    printed = 0
    for family in data["comparison_families"]:
        claims = grouped.get(family["id"], [])
        if not claims:
            raise SystemExit(f"group {family['id']!r} declares {family['size']} contrasts "
                             f"and the results carry none")
        if len(claims) != family["size"]:
            raise SystemExit(f"group {family['id']!r} declares {family['size']} contrasts "
                             f"and the results carry {len(claims)}")
        # One metric per group in every group but localization_exec_position, which runs the same
        # pair on Top-1, Top-3, and MRR. Naming the metric in the header where it is constant keeps
        # most rows clean; where it is not, the metric goes on the row, because three rows reading
        # "exec-rank (sup.) vs position" with three different numbers name nothing. The interval
        # construction follows the same rule, and post_detection_auc is where it bites: its two
        # tau-bench rows are task-clustered and its five SWE-Gym rows are not.
        metrics = {claim["metric"] for claim in claims}
        constructions = {_construction(claim["interval"]) for claim in claims}
        shared_metric = (_tex(sorted(metrics)[0]) if len(metrics) == 1 else "metric on each row")
        shared_construction = (sorted(constructions)[0] if len(constructions) == 1
                               else "interval construction on each row")
        out.append(r"\addlinespace")
        # A bare "|" is an em dash in OT1, which is both wrong here and a dash the style forbids.
        out.append(rule + r"{\texttt{%s} $\cdot$ %s $\cdot$ %s $\cdot$ %s}\\"
                   % (_tex(family["id"]), _tex(family["description"]), shared_metric,
                      shared_construction))
        for claim in claims:
            estimate, interval = claim["estimate"], claim["interval"]
            name = _tex(claim["label"])
            notes = []
            if len(metrics) > 1:
                notes.append(_tex(claim["metric"]))
            if len(constructions) > 1:
                # Labels carry parentheses of their own, so this one says what it is. Without the
                # prefix, "auditable (size+deps) vs size (flat) (paired DeLong, run-level
                # sampling)" reads as a third part of the entrant name.
                notes.append("interval: %s" % _construction(interval))
            if notes:
                name = "%s (%s)" % (name, "; ".join(notes))
            out.append("%s & %.3f & %.3f & $%+.3f$ & $[%.3f, %.3f]$ \\\\"
                       % (name, estimate["a"], estimate["b"],
                          estimate["difference_a_minus_b"], interval["low"], interval["high"]))
            printed += 1
    if printed != len(data["claims"]):
        orphans = sorted(set(grouped) - {family["id"] for family in data["comparison_families"]})
        raise SystemExit(f"{len(data['claims']) - printed} contrast(s) have no printed home; "
                         f"the results carry groups the registry does not declare: "
                         f"{', '.join(orphans)}")
    out.append(r"\end{longtable}")
    out.append(r"}")
    out.append(_CONTRASTS_END)
    return "\n".join(out)


# --- checking the paper -------------------------------------------------------------------------


def _required_marker(text: str, marker: str, offset: int = 0) -> tuple[int, int]:
    """Find a marker that stands alone on its line, and say so when it does not.

    Requiring its own line is what makes a commented-out ``% \\label{app:stats}`` a failure rather
    than a pass. A substring search accepted it and then narrowed to a section that no longer began
    where the label said.
    """
    found = re.search(rf"(?m)^[ \t]*{re.escape(marker)}[ \t]*$", text[offset:])
    if not found:
        raise ValueError(f"missing marker {marker}")
    return offset + found.start(), offset + found.end()


def _section(text: str, start: str, end: str) -> str:
    """Return the text between two required markers, each alone on its line."""
    _, begin = _required_marker(text, start)
    stop, _ = _required_marker(text, end, begin)
    return text[begin:stop]


def _check_contrasts(appendix: str, generated: str) -> list[str]:
    """Compare the contrast block byte for byte, and say where the first difference is.

    Each marker must stand alone on its line, so a commented-out marker is a failure rather than a
    pass, and a second copy of either marker is reported rather than silently bounding the wrong
    span.
    """
    findings: list[str] = []
    spans = {}
    for marker in (_CONTRASTS_BEGIN, _CONTRASTS_END):
        found = re.findall(rf"(?m)^[ \t]*{re.escape(marker)}[ \t]*$", appendix)
        if len(found) != 1:
            findings.append(f"{_APPENDIX}: the contrast-matrix marker {marker[:46]}... appears "
                            f"{len(found)} times, expected once")
            return findings
        spans[marker] = appendix.index(marker)
    if spans[_CONTRASTS_END] < spans[_CONTRASTS_BEGIN]:
        findings.append(f"{_APPENDIX}: the contrast-matrix end marker precedes its begin marker")
        return findings
    actual = appendix[spans[_CONTRASTS_BEGIN]:spans[_CONTRASTS_END] + len(_CONTRASTS_END)]
    actual_lines = actual.replace("\r\n", "\n").splitlines()
    wanted_lines = generated.splitlines()
    if actual_lines == wanted_lines:
        return findings
    findings.append(f"{_APPENDIX}: the contrast matrix differs from the generated block "
                    f"({len(actual_lines)} lines in the paper, {len(wanted_lines)} generated)")
    for index, (got, want) in enumerate(zip(actual_lines, wanted_lines)):
        if got != want:
            findings.append(f"  first difference at block line {index + 1}: paper has {got!r}")
            findings.append(f"                                     generated {want!r}")
            break
    return findings


def check(data: dict, paper: Path) -> int:
    """Report every place the paper differs from the generated inventory.

    The guards below are flat rather than folded into a loop over prose patterns, because there are
    no prose patterns left. An earlier shape reached the required-file and section checks only as a
    side effect of iterating the patterns, which would have turned an empty paper directory into a
    pass the moment the last pattern was deleted.
    """
    stale: list[str] = []
    appendix_path = paper / _APPENDIX

    if not appendix_path.exists():
        stale.append(f"{_APPENDIX}: missing")
    else:
        whole = appendix_path.read_text(encoding="utf-8")
        try:
            section = _section(whole, *_STATS_SECTION)
        except ValueError as error:
            stale.append(f"{_APPENDIX}: {error}")
        else:
            # There is no legitimate use of a TeX conditional in this section, and parsing one
            # properly is out of scope, so its presence is reported rather than interpreted.
            if r"\iffalse" in section:
                stale.append(f"{_APPENDIX}: a TeX conditional hides part of the statistics section")
        stale.extend(_check_contrasts(whole, contrasts(data)))

    for line in stale:
        print(f"STALE {line}")
    if stale:
        print(f"\n{len(stale)} staleness finding(s). "
              "Regenerate with: python tools/emit_stats_table.py --contrasts")
        return 1
    count = inventory(data)
    print(f"paper is current: {count['contrasts']} contrasts in {count['groups']} groups")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit or check the appendix contrast matrix.")
    parser.add_argument("--check", action="store_true",
                        help="verify the paper's contrast matrix instead of printing it")
    parser.add_argument("--contrasts", action="store_true",
                        help="print the contrast matrix block (the default, named explicitly "
                             "because the paper and the README both invoke it this way)")
    parser.add_argument("--paper", default=os.environ.get("CATCHBENCH_PAPER_DIR"),
                        help="paper source directory (or set CATCHBENCH_PAPER_DIR)")
    args = parser.parse_args()
    data = load()
    if args.check:
        if not args.paper:
            parser.error("--check needs --paper <dir> or CATCHBENCH_PAPER_DIR")
        return check(data, Path(args.paper))
    print(contrasts(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
