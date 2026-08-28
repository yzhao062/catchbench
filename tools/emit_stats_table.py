"""Emit the paper's comparison-family table from the shipped results, so the two cannot drift.

The appendix table and the family list in ``statistical_tests.py`` were kept in sync by hand, which
lasted exactly one wave. The PRE families landed in the module and the paper kept printing twelve
families and 101 contrasts, so a reader following the abstract's 0.048 claim into Appendix
``app:stats`` would not find the family that produced it. The counts and the rows both come from
``statistical_tests_results.json`` now; regenerate and paste after any change to the family list.

Usage::

    python tools/emit_stats_table.py             # rows plus the counts sentence
    python tools/emit_stats_table.py --contrasts # the full contrast matrix block for Appendix F
    python tools/emit_stats_table.py --check     # exit 1 if the paper is stale, printing the delta

``--check`` needs ``--paper <dir>`` or the ``CATCHBENCH_PAPER_DIR`` environment variable.

``--contrasts`` prints the block between two sentinel comments, and ``--check`` compares that span
byte for byte. The family table alone pinned no number, so a body cut could delete a claim's only
printed estimate, interval, adjusted p, and verdict while this checker stayed green. The matrix is
what makes moving prose out of the body safe: every declared contrast has a printed home whether or
not the body still argues from it.

``--check`` compares the family rows exactly rather than by name, and treats a pattern it cannot
find as staleness rather than as nothing to check. The first version did neither, and a review
showed it certifying a wrong family size, a rewritten scope description, an inserted obsolete row,
a deleted count sentence, and a removed section label as current. A checker that passes on a
corrupted table is worse than no checker, because the green result is what gets trusted.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "statistical_tests_results.json"

# LaTeX-escaping is deliberately minimal: family ids are ASCII identifiers, so an underscore is the
# only character that needs it. Anything else appearing in an id should fail loudly during review
# rather than be silently mangled into a valid-looking row.
_SAFE_ID = re.compile(r"^[A-Za-z0-9_]+$")

# A claim separates whether or not it did so in the stated direction. Folding the opposite-direction
# verdict into the nonseparating count would hide exactly the result worth reading.
_SEPARATING = frozenset({"separates_as_stated", "separates_opposite_to_statement"})
_KNOWN_VERDICTS = _SEPARATING | {"does_not_separate"}

# The prose spells the family count, so the check accepts either form. A hardcoded map would report
# a correct paper as stale the first time the family list grows past it, and a checker that cries
# wolf gets switched off, so the words are generated. The range runs past any plausible family count
# and the hyphenated forms matter: a review found that "Twenty-one" was read as "one", because the
# count pattern stops at a word boundary.
_ONES = ("", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
         "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen",
         "Eighteen", "Nineteen")
_TENS = ("", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety")


def _spell(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + (f"-{_ONES[ones].lower()}" if ones else "")


_NUMBER_WORDS = {n: _spell(n) for n in range(1, 100)}

_APPENDIX = "09_appendix.tex"
_BENCHMARK = "03_benchmark.tex"
_STATS_SECTION = (r"\label{app:stats}", r"\label{tab:stat-families}")

# One wrapped sentence, never two paragraphs: horizontal whitespace, or a single line break.
_PROSE_GAP = r"(?:[ \t]+|[ \t]*\r?\n[ \t]*)"

_PATTERNS = {
    _BENCHMARK: (
        # Was "N comparison families are declared", which stopped being true when one family
        # arrived a wave after the rest. The clause moved; the count it pins did not. The gap
        # spans a line wrap because the sentence wraps, and no more than that: ``\s+`` also
        # spans a blank line, which would let the checker certify a count statement that a
        # paragraph break had split into two sentences the reader never sees as one.
        (rf"registry to ([\w-]+){_PROSE_GAP}comparison{_PROSE_GAP}families",
         "families", "families declared"),
        (r"all (\d+) reported contrasts", "total", "reported contrasts"),
    ),
    _APPENDIX: (
        (r"The ([\w-]+) families below", "families", "families below"),
        (r"Of the (\d+)\s*\n?\s*contrasts", "total", "contrast total"),
        (r"contrasts,\s*(\d+) separate after correction", "separating", "separating"),
        (r"separate after correction and (\d+) do not", "not_separating", "not separating"),
        # The Total row is pinned exactly by the table-body comparison below, which also
        # prints the first difference, so a separate pattern for it would only duplicate a
        # check and a diagnostic.
    ),
}

_TABLE_BEGIN = re.compile(r"(?m)^[ \t]*\\begin\{tabular\}\{[^\r\n]*\}[ \t]*$")
_TABLE_END = re.compile(r"(?m)^[ \t]*\\end\{tabular\}[ \t]*$")
# Layout lines carry no inventory, so they are dropped before comparison. Restyling the rules or
# renaming the column headings is a legitimate edit; adding a row is not.
_SCAFFOLDING = frozenset({r"\toprule", r"\midrule", r"\bottomrule"})
_HEADER_ROW = re.compile(r"^[A-Za-z][A-Za-z ]*&[A-Za-z ]*&[A-Za-z ]*\\\\$")


def load() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def counts(data: dict) -> dict[str, int]:
    claims = data["claims"]
    unknown = sorted({claim["verdict"] for claim in claims} - _KNOWN_VERDICTS)
    if unknown:
        raise SystemExit(f"unknown claim verdicts, refusing to count: {', '.join(unknown)}")
    separating = sum(claim["verdict"] in _SEPARATING for claim in claims)
    return {
        "families": len(data["comparison_families"]),
        "total": len(claims),
        "separating": separating,
        "not_separating": len(claims) - separating,
    }


def rows(data: dict) -> str:
    out = []
    for family in data["comparison_families"]:
        fid = family["id"]
        if not _SAFE_ID.match(fid):
            raise SystemExit(f"family id needs escaping beyond underscores: {fid!r}")
        escaped = fid.replace("_", r"\_")
        out.append(f"\\texttt{{{escaped}}} & {family['size']} & {family['description']} \\\\")
    return "\n".join(out)


def sentence(data: dict) -> str:
    n = counts(data)
    return (
        f"The {n['families']} families below are the paper's\n"
        f"definition of multiplicity: each family's raw $p$-values are adjusted together by Holm's "
        f"step-down\nprocedure with running-maximum monotonicity enforcement. A point-estimate "
        f"ordering that appears in a\ntable but not in a family is descriptive and is not claimed as "
        f"a result. Of the {n['total']}\ncontrasts, {n['separating']} separate after correction and "
        f"{n['not_separating']} do not."
    )


# --- the full contrast matrix -----------------------------------------------------------------
#
# The family table above pins how multiplicity is defined. It does not pin a single number, so a
# body cut can drop a claim's estimate, interval, adjusted p, and verdict and this checker stays
# green. That gap is why the paper needed a printed home for every contrast rather than for every
# family, and why the block below is compared byte for byte instead of row by row: any edit inside
# the markers, including one that still parses as a table, is staleness.
_CONTRASTS_BEGIN = ("% BEGIN GENERATED tab:all-contrasts -- regenerate with: "
                    "python tools/emit_stats_table.py --contrasts")
_CONTRASTS_END = "% END GENERATED tab:all-contrasts"

# Anything outside this set in a claim label or a method name is a character whose LaTeX meaning
# was not considered, so it fails review rather than being escaped by guesswork.
_LATEX_ESCAPES = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$"}
_UNHANDLED = re.compile(r"[\\^~{}]")


def _tex(text: str) -> str:
    if _UNHANDLED.search(text):
        raise SystemExit(f"claim text needs escaping this generator does not do: {text!r}")
    return "".join(_LATEX_ESCAPES.get(character, character) for character in text)


def _p(value: float) -> str:
    """Holm-adjusted p for a column already headed "Holm p", grouped so it cannot line-break."""
    if value >= 0.001:
        return r"$%.3f$" % value
    exponent = 0
    while value < 1:
        value *= 10
        exponent += 1
    return r"$%.1f{\times}10^{-%d}$" % (value, exponent)


_VERDICT_WORDS = {
    "separates_as_stated": "separates",
    "separates_opposite_to_statement": "separates, opposite",
    # Failure to reject, which is not evidence of equivalence. The paper says "unresolved"
    # everywhere else and this table must not invent a second word for the same verdict.
    "does_not_separate": "unresolved",
}


def contrasts(data: dict) -> str:
    """The whole longtable float, markers included, ready to paste into the appendix."""
    grouped: dict[str, list[dict]] = {}
    for claim in data["claims"]:
        grouped.setdefault(claim["family"], []).append(claim)

    header = (r"Contrast & $A$ & $B$ & $A-B$ & 95\% CI & Holm $p$ & Verdict \\")
    out = [
        _CONTRASTS_BEGIN,
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        # Fixed widths on the three text-shaped columns. Natural-width l and c columns overflowed
        # \textwidth by 234pt, because a long verdict word and a wide interval have nowhere to wrap.
        r"\begin{longtable}{@{}p{5.3cm}rrr"
        r">{\centering\arraybackslash}p{2.15cm}"
        r">{\raggedleft\arraybackslash}p{1.65cm}"
        r">{\raggedright\arraybackslash}p{1.5cm}@{}}",
        r"\caption{Every declared contrast, by family. $A$ and $B$ are the two entrants named in "
        r"the contrast, in that order, and the interval is on the difference at the level and axis "
        r"the family header states. \emph{Unresolved} is a failure to reject at the Holm-adjusted "
        r"level, which is not evidence that the two are equivalent. This table is the printed home "
        r"of every claim in Appendix~\ref{app:stats}; the body reports the subset it argues from. "
        r"Generated by \texttt{tools/emit\_stats\_table.py -{}-contrasts}, which fails CI when the "
        r"paper falls behind the shipped results.}"
        r"\label{tab:all-contrasts}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{7}{@{}p{13.6cm}@{}}{\emph{Continued from the previous page.}}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for family in data["comparison_families"]:
        claims = grouped.get(family["id"], [])
        if not claims:
            raise SystemExit(f"family {family['id']!r} declares {family['size']} contrasts "
                             f"and the results carry none")
        if len(claims) != family["size"]:
            raise SystemExit(f"family {family['id']!r} declares {family['size']} contrasts "
                             f"and the results carry {len(claims)}")
        first = claims[0]
        # One metric per family in every family but localization_exec_position, which runs the same
        # pair on Top-1, Top-3, and MRR. Naming the metric in the header where it is constant keeps
        # most rows clean; where it is not, the metric goes on the row, because three rows reading
        # "exec-rank (sup.) vs position" with three different numbers name nothing.
        metrics = {claim["metric"] for claim in claims}
        shared = _tex(sorted(metrics)[0]) if len(metrics) == 1 else "metric on each row"
        out.append(r"\addlinespace")
        # A bare "|" is an em dash in OT1, which is both wrong here and a dash the style forbids.
        out.append(r"\multicolumn{7}{@{}p{13.6cm}@{}}{\texttt{%s} $\cdot$ %s $\cdot$ %s $\cdot$ "
                   r"%s, %s}\\"
                   % (_tex(family["id"]), _tex(family["description"]), shared,
                      _tex(first["interval"]["method"]), _tex(first["interval"]["axis"])))
        for claim in claims:
            estimate, interval, test = claim["estimate"], claim["interval"], claim["test"]
            verdict = _VERDICT_WORDS.get(claim["verdict"])
            if verdict is None:
                raise SystemExit(f"unknown verdict {claim['verdict']!r} on {claim['id']!r}")
            name = _tex(claim["label"])
            if len(metrics) > 1:
                name = "%s (%s)" % (name, _tex(claim["metric"]))
            out.append("%s & %.3f & %.3f & $%+.3f$ & $[%.3f, %.3f]$ & %s & %s \\\\"
                       % (name, estimate["a"], estimate["b"],
                          estimate["difference_a_minus_b"], interval["low"], interval["high"],
                          _p(test["p_adjusted_holm"]), verdict))
    out.append(r"\end{longtable}")
    out.append(r"}")
    out.append(_CONTRASTS_END)
    return "\n".join(out)


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


def _table_body(text: str) -> list[str]:
    """Return the content lines inside the statistics tabular, scaffolding removed.

    Bounding the scan to the tabular is the point. The previous version collected every line in the
    section that looked like a generated row, so moving a row past ``\\end{tabular}`` left the
    collected list unchanged and the corrupted table passed. Layout lines are dropped rather than
    matched exactly, so restyling the rules or renaming the column headings does not raise a false
    alarm, while a duplicated total, a stray row, or a conditional wrapper survives and does.
    """
    begin = _TABLE_BEGIN.search(text)
    if not begin:
        raise ValueError("statistics tabular does not start")
    end = _TABLE_END.search(text, begin.end())
    if not end:
        raise ValueError("statistics tabular does not end")
    body = []
    for line in text[begin.end():end.start()].splitlines():
        stripped = line.strip()
        if stripped and stripped not in _SCAFFOLDING and not _HEADER_ROW.match(stripped):
            body.append(stripped)
    return body


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
    """Report every place the paper differs from the generated inventory."""
    want = counts(data)
    stale: list[str] = []
    appendix_text: str | None = None

    for name, patterns in _PATTERNS.items():
        path = paper / name
        if not path.exists():
            stale.append(f"{name}: missing")
            continue
        text = path.read_text(encoding="utf-8")
        if name == _APPENDIX:
            try:
                text = _section(text, *_STATS_SECTION)
            except ValueError as error:
                stale.append(f"{name}: {error}")
                continue
            appendix_text = text
        # A commented-out claim is not a claim. Counting it would let a stale sentence sit in the
        # file behind a percent sign and still satisfy the check.
        active = re.sub(r"(?m)(?<!\\)%.*$", "", text)
        for pattern, key, label in patterns:
            found = re.findall(pattern, active)
            if not found:
                stale.append(f"{name}: {label} statement is absent")
                continue
            if len(found) > 1:
                stale.append(f"{name}: {label} stated {len(found)} times, so one of them is stale")
                continue
            got = found[0]
            expected = {str(want[key]), _NUMBER_WORDS.get(want[key], "")}
            if got.lower() not in {form.lower() for form in expected if form}:
                stale.append(f"{name}: {label} says {got!r}, results say {want[key]}")

    # The contrast matrix is checked against the whole appendix file, not the bounded statistics
    # section, because it is a long float that may legitimately be placed after the family table.
    appendix_path = paper / _APPENDIX
    if appendix_path.exists():
        whole = appendix_path.read_text(encoding="utf-8")
        stale.extend(_check_contrasts(whole, contrasts(data)))

    if appendix_text is not None:
        # There is no legitimate use of a TeX conditional in this table, and parsing one properly is
        # out of scope, so its presence is reported rather than interpreted.
        if r"\iffalse" in appendix_text:
            stale.append(f"{_APPENDIX}: a TeX conditional hides part of the statistics section")
        try:
            actual = _table_body(appendix_text)
        except ValueError as error:
            stale.append(f"{_APPENDIX}: {error}")
        else:
            expected = rows(data).splitlines() + [f"Total & {want['total']} & \\\\"]
            if actual != expected:
                stale.append(f"{_APPENDIX}: the statistics table differs from the generated table "
                             f"({len(actual)} content lines in the paper, {len(expected)} generated)")
                for got, wanted in zip(actual, expected):
                    if got != wanted:
                        stale.append(f"  first difference: paper has {got!r}")
                        stale.append(f"                  generated {wanted!r}")
                        break

    for line in stale:
        print(f"STALE {line}")
    if stale:
        print(f"\n{len(stale)} staleness finding(s). "
              f"Regenerate with: python tools/emit_stats_table.py")
        return 1
    print(f"paper is current: {want['families']} families, {want['total']} contrasts, "
          f"{want['separating']} separating")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the paper's counts and family rows instead of printing them")
    parser.add_argument("--contrasts", action="store_true",
                        help="print the full contrast matrix block for the appendix")
    parser.add_argument("--paper", default=os.environ.get("CATCHBENCH_PAPER_DIR"),
                        help="paper source directory (or set CATCHBENCH_PAPER_DIR)")
    args = parser.parse_args()
    data = load()
    if args.contrasts:
        print(contrasts(data))
        return 0
    if args.check:
        if not args.paper:
            parser.error("--check needs --paper <dir> or CATCHBENCH_PAPER_DIR")
        return check(data, Path(args.paper))
    n = counts(data)
    print(f"% {n['families']} families, {n['total']} contrasts, {n['separating']} separating, "
          f"{n['not_separating']} not. Generated by tools/emit_stats_table.py.")
    print()
    print(sentence(data))
    print()
    print(rows(data))
    print(r"\midrule")
    print(f"Total & {n['total']} & \\\\")
    print()
    print(f"% {_BENCHMARK} prose: \"registry to "
          f"{_NUMBER_WORDS.get(n['families'], n['families'])} comparison families\" and "
          f"\"all {n['total']} reported contrasts\".")
    return 0


if __name__ == "__main__":
    sys.exit(main())
