"""Emit the paper's comparison-family table from the shipped results, so the two cannot drift.

The appendix table and the family list in ``statistical_tests.py`` were kept in sync by hand, which
lasted exactly one wave. The PRE families landed in the module and the paper kept printing twelve
families and 101 contrasts, so a reader following the abstract's 0.048 claim into Appendix
``app:stats`` would not find the family that produced it. The counts and the rows both come from
``statistical_tests_results.json`` now; regenerate and paste after any change to the family list.

Usage::

    python tools/emit_stats_table.py            # rows plus the counts sentence
    python tools/emit_stats_table.py --check    # exit 1 if the paper is stale, printing the delta

``--check`` needs ``--paper <dir>`` or the ``CATCHBENCH_PAPER_DIR`` environment variable.

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

_PATTERNS = {
    _BENCHMARK: (
        (r"([\w-]+)\s+comparison families are declared", "families", "families declared"),
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
    parser.add_argument("--paper", default=os.environ.get("CATCHBENCH_PAPER_DIR"),
                        help="paper source directory (or set CATCHBENCH_PAPER_DIR)")
    args = parser.parse_args()
    data = load()
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
    print(f"% {_BENCHMARK} prose: \"{_NUMBER_WORDS.get(n['families'], n['families'])} comparison "
          f"families are declared\" and \"all {n['total']} reported contrasts\".")
    return 0


if __name__ == "__main__":
    sys.exit(main())
