"""Run the scoring board and check the README against committed evidence.

``run.py`` produces the scoring board. Until now nothing checked that it still produced the
committed output: ``ci_smoke.py`` covers the PRE board only, and POST, LIVE, and Gold are in its skip
list because they need the GRADE bridge and three corpus downloads. A refactor could move a
published cell and no test would notice. The board comparison closes that gap. It is byte-exact
except for the explicitly named torch rows, where ``NEURAL_TOLERANCE`` documents and bounds a
cross-platform kernel difference.

The README half does three narrower things. It extracts every numeric pipe table from ``README.md``
and compares each cell exactly against the committed golden board. It scans every decimal numeral
outside pipe tables and fenced code blocks; a numeral must either equal a parsed golden cell at its
displayed precision or be claimed exactly once by ``PROSE_NUMBER_ALLOWLIST``, with a one-line
reason. Allowances and board claims use unique content substrings, not line numbers, so unrelated
insertions do not invalidate them. Finally, it scans prose paragraphs under ``The Boards`` for the
comparison spellings in ``COMPARATIVE_WORD``. Every paragraph it finds must be claimed by
exactly one ``BOARD_CLAIMS`` entry, which declares the measurements that paragraph states.

*What replaced the verdict licence.* Until this change, a paragraph was licensed by a registered
claim's ``verdict``: ``separates_as_stated`` licensed an ordering, and ``does_not_separate``
licensed only a paragraph carrying an explicit non-separation phrase. That gate encoded a
reporting policy this benchmark no longer follows. A benchmark reports measurements; whether a
comparison reaches significance is not the benchmark's claim to make, and a non-significant result
is as much a finding as a significant one. So the gate now reads the measurement, not the verdict.

A ``BoardClaim`` declares two kinds of measurement. A ``BoardOrdering`` names two golden board cells
and asserts which one the paragraph puts higher; the checker resolves both cells through the same
source binding ``TABLE_SPECS`` uses and fails when the printed values do not agree with the stated
order. A ``RecordedDifference`` names one contrast in ``tools/statistical_tests_results.json`` and
says whether the paragraph orders the contrast's two arms, or cites it without ordering them; the
checker reads that record's ``estimate`` and fails when the sign of ``difference_a_minus_b``
disagrees. It reads no ``verdict``, no ``p_raw``, and no ``p_adjusted_holm``, and it never asks
whether an interval excludes zero. A paragraph that orders nothing declares
``reports_no_ordering`` with its reason, in the shape ``NON_BOARD_TABLES`` uses for a table.

The rewritten README retains the Gold analytic-floor measurements and LIVE point-estimate
comparisons. Their bindings follow that prose. The removed LIVE figure family counts and SWE-Gym
threshold-side statements have no bindings; source comments record what those entries used to bind.

The comparison scan is a coverage gate over a registry, not natural-language understanding. It
catches a new paragraph that uses one of the registered comparison spellings, a removed entry, a
missing or duplicated claim ID, a stated order the board contradicts, a stated order the recorded
estimate contradicts, and a record whose own arithmetic disagrees with itself. It cannot catch a
comparison phrased without those spellings, inspect prose outside ``The Boards``, or prove that a
registered measurement is the one a sentence actually rests on. A new comparison added to an already
claimed paragraph can still escape it. Reviewers still have to read the mapping.

``check_adjudicating_numbers`` is the one lexical rule kept, and it is deliberately narrow: a
printed ``Holm`` or a printed p-value fails, anywhere outside a fenced code block. Those two
spellings have no other use in this README. The wider vocabulary a reader might expect here,
``separates``, ``resolves`` and ``unresolved``, is **not** gated, because this README spends all
three on ordinary English and on SWE-Gym run outcomes ("188 failed, 188 resolved"). A gate that
fires there would be noise. The policy binds the inference a sentence makes, and no regular
expression sees that.

Two rules govern the README check.

*No README-table tolerance.* A cell matches when its printed text matches. ``0.703`` and ``0.707``
are a failure, and so are ``0.71`` and ``0.710``. Prose can print fewer decimal places; there a
golden cell is rounded to the precision the prose chose before comparison.

*Fail closed.* Every numeric README table must be claimed exactly once by ``TABLE_SPECS`` (or by a
reasoned ``NON_BOARD_TABLES`` entry), and every non-board prose numeral must be claimed exactly once
by its registry. Every comparator paragraph in scope must be claimed by one ``BOARD_CLAIMS`` entry,
and that entry must name a measurement or record why the paragraph orders nothing. A table,
numeral, paragraph, allowance, board claim, golden source, or claim ID that matches nothing or
matches more than once is a failure, never a silent skip.

Usage::

    python tools/check_board.py               # run the board, then check the README
    python tools/check_board.py --readme-only # check README tables, prose, and claims (seconds)
    python tools/check_board.py --board-only  # run or compare only the expensive scoring board
    python tools/check_board.py --produced F  # compare a saved board, plus the README
    python tools/check_board.py --update      # regenerate the golden from the current code

The default and ``--update`` modes check both halves. ``--board-only`` exists for the slow board CI
workflow because the fast verify workflow checks the README on every push and pull request.

The board takes roughly nine minutes and needs the GRADE checkout bridge, the torch stack for the
PyGOD rows, and about 320 MB of corpora at their pinned revisions. The README half needs none of
that: it reads the README, golden board, and committed statistical results.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "golden" / "board.txt"
RUNNER = ROOT / "run.py"
README = ROOT / "README.md"
STATISTICAL_RESULTS = ROOT / "tools" / "statistical_tests_results.json"


def produce() -> str:
    """Run the board with a fixed environment and return its stdout."""
    env = dict(os.environ)
    # The board must not depend on hash ordering. Pinning the seed here keeps a local run and a CI
    # run comparable; if a future change makes the board seed-sensitive, that is itself a defect.
    env["PYTHONHASHSEED"] = "0"
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    done = subprocess.run([sys.executable, str(RUNNER)], capture_output=True, text=True,
                          cwd=str(ROOT), env=env)
    if done.returncode != 0:
        tail = "\n".join(done.stderr.strip().splitlines()[-15:])
        raise SystemExit(f"run.py exited {done.returncode}\n{tail}")
    return done.stdout


def normalize(text: str) -> list[str]:
    """Strip trailing whitespace and line endings, which differ by platform and carry no result."""
    return [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]


# Rows whose score comes from a torch model. Everything else on this board reproduces exactly across
# platforms, which is not an assumption: the first CI run of this comparison, on ubuntu-latest
# against a golden generated on Windows, moved these two lines and nothing else.
#
#     g-safeguard (sup GNN)   0.828 -> 0.829
#     pygod-anomalydae        0.490 -> 0.487
#
# A second, independent CI run produced the same two values, which is why this was first recorded as a
# stable difference between platforms rather than nondeterminism between runs. That reading is
# withdrawn. A scheduled run on 2026-09-08 produced pygod-anomalydae 0.485, a third value for the same
# row on the same commit, so the row does move between runs and not only between platforms. What that
# run also showed is how narrow the movement is: of 274 board lines it changed exactly one, and the
# other sixteen rows this tolerance covers, g-safeguard among them, reproduced byte for byte.
# The cause is still underneath torch: a different BLAS, a different reduction order, a different
# build, and now also a run-to-run path that is not pinned by the seed.
#
# The tolerance is 0.005, smaller than the seed variance the paper already reports for the larger of
# the two rows, 0.824 +/- 0.007 for g-safeguard over five joint split and initialization seeds. A
# difference this check now tolerates is one the paper already tells a reader to expect, and a
# difference large enough to move a claim still fails.
#
# Everything outside these rows stays byte-exact. Widening the tolerance to the whole board would
# leave it unable to see a real regression in the rule methods, which are exact by construction and
# are most of the board.
#
# The list names the two rows rather than three model families, and that narrowing is the point. The
# family prefixes ("guardian (", "g-safeguard (", "pygod") matched seventeen board lines carrying
# forty of the board's 653 values, while the rows ever observed to move are the two below, carrying
# four. Thirty-six values were being tolerated on no evidence at all, which is the same mistake the
# paragraph above warns about, made one level down. The paper states the exception as two torch-backed
# cells, so this is also what makes that sentence exactly true rather than an understatement.
#
# A row that starts moving later will fail this check, and that is the intended outcome: a red run
# saying "a new row moved" is the observation, and absorbing it silently is what this list is for
# avoiding. Add a row here only with a produced board showing it move, which every run now uploads.
TORCH_ROW_PREFIXES = ("g-safeguard (sup GNN)", "pygod-anomalydae")
NEURAL_TOLERANCE = 0.005

# The comparison runs on Decimal, not float, and the reason is a run this check rejected. A scheduled
# CI job produced pygod-anomalydae 0.485 against a golden 0.490, a difference of exactly the tolerance
# above. In binary floating point that subtraction is 0.0050000000000000044, which is greater than
# 0.005 by 4.3e-18, so a difference sitting on the documented bound failed the bound. The board prints
# decimals, so the arithmetic that decides whether two printed cells agree is done on decimals. The
# float form is kept as the published constant because it is what the prose quotes.
_NEURAL_TOLERANCE = Decimal(str(NEURAL_TOLERANCE))

_FLOAT = re.compile(r"-?\d+\.\d+")


def _is_torch_row(line: str) -> bool:
    return line.lstrip().startswith(TORCH_ROW_PREFIXES)


def within_neural_tolerance(want_line: str, got_line: str) -> bool:
    """True when two rows differ only in a torch model's score, by at most NEURAL_TOLERANCE.

    The label, the column layout, and the count of numbers all have to match. Only the values may
    move, and only on a row this board scores with a torch model.
    """
    if not (_is_torch_row(want_line) and _is_torch_row(got_line)):
        return False
    a, b = _FLOAT.findall(want_line), _FLOAT.findall(got_line)
    if not a or len(a) != len(b):
        return False
    if _FLOAT.sub("#", want_line) != _FLOAT.sub("#", got_line):
        return False
    return all(abs(Decimal(x) - Decimal(y)) <= _NEURAL_TOLERANCE for x, y in zip(a, b))


def reconcile_neural_rows(want: list[str], got: list[str]) -> tuple[list[str], list[str], int]:
    """Fold tolerated torch-row differences into agreement, and report how many were folded.

    Only same-length boards are reconciled. A board that gained or lost a line changed structurally,
    and that is never a float-kernel difference.
    """
    if len(want) != len(got):
        return want, got, 0
    folded = 0
    merged = list(want)
    for i, (a, b) in enumerate(zip(want, got)):
        if a != b and within_neural_tolerance(a, b):
            merged[i] = b
            folded += 1
    return merged, got, folded


# ---------------------------------------------------------------------------------------------
# Text normalization shared by both sides
# ---------------------------------------------------------------------------------------------

_MARKUP = re.compile(r"[*`]+")
_WS = re.compile(r"\s+")

# Cells that say "this quantity is not reported here" rather than carrying a value.
PLACEHOLDERS = frozenset({"", "-", "--", "---", "—", "–", "n/a", "na"})

# A cell counts as numeric for the "is this a board table" test when everything in it is number,
# once whitespace and the separators a board value is printed with are removed.
#
# Two earlier rules each let a real table through. Requiring the whole cell to be one bare number
# skipped any table of spreads. Accepting any whitespace-separated numeric token fixed the spaced
# form "0.824 +/- 0.007" and nothing else: the board also prints "0.309/0.414/0.402", which stayed
# invisible, while "RFC 6749" and "OWASP Top 10" became numeric and dragged reference tables in.
#
# Removing separators and then demanding that no letter survive handles both directions. The board's
# real shapes pass (0.452, 11%, 1187, .824, 1e-3, bracketed intervals, slash triples, spreads with or
# without spaces), and anything carrying a word does not: CWE-272, OWASP LLM06, v0.1.0, n/a.
_SEPARATORS = str.maketrans("", "", " \t/±()[],;+-")
# What may remain once separators are gone. A character test rather than a number grammar, because
# a slash triple collapses to "0.3090.4140.402", which is three dots and no valid number, yet is
# unmistakably board output. The question here is only "is this cell numbers", not "parse it".
_NUMERIC_CHARS = frozenset("0123456789.%eE")


def strip_markup(cell: str) -> str:
    """Drop the emphasis and code markers the README wraps values in, keeping the value."""
    return _MARKUP.sub("", cell).strip()


def norm_label(text: str) -> str:
    """Fold a label to its comparable form.

    The two sides write the same row different ways: the board prints ``flag_all`` and
    ``pygod (graph AD)``, the README prints ``flag-all`` and ``**PyGOD (graph AD)**``. Case, the
    emphasis markers, underscore-versus-hyphen, and runs of whitespace are all noise here. Anything
    beyond that is a real difference and stays.
    """
    text = strip_markup(text).lower().replace("_", "-")
    return _WS.sub(" ", text).strip().rstrip(".:").strip()


def is_placeholder(cell: str) -> bool:
    return strip_markup(cell).lower() in PLACEHOLDERS


def is_numeric(cell: str) -> bool:
    bare = strip_markup(cell).translate(_SEPARATORS)
    return any(c.isdigit() for c in bare) and all(c in _NUMERIC_CHARS for c in bare)


@dataclass(frozen=True)
class Problem:
    """One reason the README does not agree with the golden board."""

    kind: str
    message: str
    line: int | None = None

    def render(self) -> str:
        where = f"README.md:{self.line}" if self.line else "README.md"
        return f"{where}: [{self.kind}] {self.message}"


# ---------------------------------------------------------------------------------------------
# Golden board -> named blocks
# ---------------------------------------------------------------------------------------------

# The board is a fixed-width table, so two or more spaces usually separate cells. Usually, not
# always: a long row label can leave a single space before its first value ("auditable
# (dep-anomaly) 0.309/0.414/0.402"), and one header packs two names into one gap ("tpr@5fpr
# tpr@10fpr"). Splitting on runs of two spaces alone silently drops the rows below such a line,
# which is how the first version of this parser lost two Gold rows. So a row that does not split
# cleanly is retried from the right, and the retry is accepted only when every trailing cell looks
# like a printed value.
_SPLIT_CELLS = re.compile(r"\s{2,}")
_VALUEISH = re.compile(r"^[<>]?[-+]?\d[\d.,/%+\-]*$")


def _split_row(line: str, width: int) -> list[str] | None:
    """Split one board row into exactly ``width`` cells, or return None."""
    cells = _SPLIT_CELLS.split(line.strip())
    if len(cells) == width:
        return cells
    if width < 2:
        return None
    retry = line.strip().rsplit(None, width - 1)
    if len(retry) == width and all(_VALUEISH.match(c) for c in retry[1:]):
        return retry
    return None


@dataclass
class GoldenBlock:
    """One scored table in ``tests/golden/board.txt``."""

    key: str
    title: str
    line: int
    columns: tuple[str, ...]          # normalized, including the leading "method"
    rows: dict[str, tuple[str, ...]]  # normalized row label -> printed values
    order: tuple[str, ...]
    error: str | None = None


def _block_key(title: str) -> str:
    """Name a block by the stable head of its title.

    A prose title carries counts that move with the corpus ("82 stale + 106 dropped"), so keying on
    the whole line would make every spec stale the first time a corpus grows. Cutting at the first
    parenthesis or comma keeps the part that names the table.
    """
    head = title.split("(")[0].split(",")[0]
    return norm_label(head)


def parse_golden(text: str) -> tuple[dict[str, GoldenBlock], Counter]:
    """Index the golden board's tables by name.

    A table is a title line at column 0 whose next non-blank line is indented and starts with the
    cell ``method``. That structural signal is what makes a block a block; the titles themselves are
    inconsistent (some end in a colon, one ends in ``Top-1/Top-3/MRR``), so keying on their
    punctuation would miss one. Everything else in the board (the corpus header, the distributional
    check, the Reading notes) has no such header and is not indexed; a spec that names one of those
    fails to resolve, which is the intended outcome rather than a silent match.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    found: list[GoldenBlock] = []
    i = 0
    while i < len(lines):
        title = lines[i].rstrip()
        if not title or title.startswith(" "):
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines) or not lines[j].startswith(" "):
            i += 1
            continue
        header_cells = _SPLIT_CELLS.split(lines[j].strip())
        if norm_label(header_cells[0]) != "method":
            i += 1
            continue
        # A header whose own names are separated by a single space splits short. Take the width from
        # the first data row and re-split from the right, but only into cells that hold no space.
        first = lines[j + 1] if j + 1 < len(lines) else ""
        if first.startswith(" ") and first.strip():
            width = len(_SPLIT_CELLS.split(first.strip()))
            if width > len(header_cells):
                retry = lines[j].strip().rsplit(None, width - 1)
                if len(retry) == width:
                    header_cells = retry
        columns = tuple(norm_label(c) for c in header_cells)

        rows: dict[str, tuple[str, ...]] = {}
        order: list[str] = []
        error: str | None = None
        k = j + 1
        while k < len(lines):
            raw = lines[k]
            if not raw.strip() or not raw.startswith(" "):
                break
            cells = _split_row(raw, len(columns))
            if cells is None:
                break
            label = norm_label(cells[0])
            if label in rows:
                error = f"duplicate row {label!r}"
            rows[label] = tuple(cells[1:])
            order.append(label)
            k += 1
        if not rows:
            # Recording a block as unusable is honest; a spec that names it gets the reason instead
            # of an empty match.
            error = error or ("no data rows parsed; the header splits into "
                              f"{len(columns)} cell(s) and the rows do not agree")
        found.append(GoldenBlock(key=_block_key(title), title=title, line=i + 1, columns=columns,
                                 rows=rows, order=tuple(order), error=error))
        i = max(k, i + 1)

    counts = Counter(b.key for b in found)
    seen: Counter = Counter()
    index: dict[str, GoldenBlock] = {}
    for block in found:
        seen[block.key] += 1
        index[f"{block.key}#{seen[block.key]}"] = block
        if counts[block.key] == 1:
            index[block.key] = block
    return index, counts


# ---------------------------------------------------------------------------------------------
# README -> tables
# ---------------------------------------------------------------------------------------------

_DELIMITER_CELL = re.compile(r"^:?-+:?$")


@dataclass
class ReadmeTable:
    """One pipe table in the README, with the context that identifies it."""

    line: int                                     # 1-based line of the header row
    heading: str                                  # normalized nearest preceding heading
    lead_in: str                                  # normalized paragraph just above the table
    header: tuple[str, ...]                       # normalized header cells
    rows: tuple[tuple[int, str, tuple[str, ...]], ...]  # line, raw label, raw value cells

    @property
    def numeric(self) -> bool:
        return any(is_numeric(c) for _, label, cells in self.rows for c in (label,) + cells)


def _split_pipe_row(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [cell.strip() for cell in body.split("|")]


def parse_readme(text: str) -> tuple[list[ReadmeTable], list[Problem]]:
    """Pull every pipe table out of the README along with its heading and lead-in paragraph.

    The README is prose with tables in it, so a table is located by context rather than by position:
    the heading it sits under, the paragraph immediately above it, and its own column header. Any
    run of pipe lines that is not a well-formed table is reported, not skipped, because a table
    whose delimiter row or column count has been damaged is exactly the state in which a silent skip
    would read as a pass.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    tables: list[ReadmeTable] = []
    problems: list[Problem] = []
    heading = ""
    paragraph: list[str] = []
    previous_paragraph = ""
    in_fence = False
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            paragraph = []
            previous_paragraph = ""
            i += 1
            continue
        if in_fence:
            i += 1
            continue
        if stripped.startswith("#"):
            heading = norm_label(stripped.lstrip("#"))
            paragraph = []
            previous_paragraph = ""
            i += 1
            continue
        if not stripped:
            if paragraph:
                previous_paragraph = " ".join(paragraph)
                paragraph = []
            i += 1
            continue
        if not stripped.startswith("|"):
            paragraph.append(stripped)
            i += 1
            continue

        j = i
        while j < len(lines) and lines[j].strip().startswith("|"):
            j += 1
        run = lines[i:j]
        start = i + 1
        lead_in = norm_label(" ".join(paragraph) if paragraph else previous_paragraph)
        paragraph = []
        previous_paragraph = ""
        i = j

        if len(run) < 3:
            problems.append(Problem(
                "unparsed-table",
                f"{len(run)} pipe line(s) with no table body; a markdown table needs a header row, "
                "a delimiter row, and at least one data row",
                line=start))
            continue
        header = _split_pipe_row(run[0])
        delimiter = _split_pipe_row(run[1])
        if not delimiter or not all(_DELIMITER_CELL.match(c) for c in delimiter):
            problems.append(Problem("unparsed-table",
                                    "the second line of this table is not a delimiter row "
                                    f"({run[1].strip()!r})", line=start + 1))
            continue
        if len(delimiter) != len(header):
            problems.append(Problem("unparsed-table",
                                    f"header has {len(header)} column(s) and the delimiter row has "
                                    f"{len(delimiter)}", line=start + 1))
            continue
        rows: list[tuple[int, str, tuple[str, ...]]] = []
        ragged = False
        for offset, raw in enumerate(run[2:], start=2):
            cells = _split_pipe_row(raw)
            if len(cells) != len(header):
                problems.append(Problem("unparsed-table",
                                        f"row has {len(cells)} cell(s) and the header has "
                                        f"{len(header)}", line=start + offset))
                ragged = True
                continue
            rows.append((start + offset, cells[0], tuple(cells[1:])))
        if ragged or not rows:
            if not rows and not ragged:
                problems.append(Problem("unparsed-table", "table has no data rows", line=start))
            continue
        tables.append(ReadmeTable(line=start, heading=heading, lead_in=lead_in,
                                  header=tuple(norm_label(c) for c in header), rows=tuple(rows)))
    return tables, problems


# ---------------------------------------------------------------------------------------------
# The map from README tables to golden blocks
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """Where one README column's numbers come from."""

    block: str
    column: str
    part: int | None = None  # index into a "/"-joined cell, e.g. Top-1 of "0.309/0.414/0.402"


@dataclass(frozen=True)
class Row:
    """One README row and the golden row it copies."""

    golden: str
    sources: Mapping[str, Source] = field(default_factory=dict)   # per-column override
    absent: frozenset[str] = frozenset()                          # columns the golden does not have


@dataclass(frozen=True)
class TableSpec:
    """One README table, its anchor, and its mapping onto the golden board."""

    name: str
    header: tuple[str, ...]
    columns: Mapping[str, Source]
    rows: Mapping[str, Row]
    heading_contains: str = ""
    lead_in_contains: str = ""
    separators: frozenset[str] = frozenset()

    def matches(self, table: ReadmeTable) -> bool:
        return (table.header == self.header
                and self.heading_contains in table.heading
                and self.lead_in_contains in table.lead_in)


@dataclass(frozen=True)
class Exemption:
    """A README table that carries numbers but is deliberately not a board table."""

    name: str
    header: tuple[str, ...]
    reason: str
    heading_contains: str = ""
    lead_in_contains: str = ""

    def matches(self, table: ReadmeTable) -> bool:
        return (table.header == self.header
                and self.heading_contains in table.heading
                and self.lead_in_contains in table.lead_in)


_POST_LOC = "[post] post-localization :: whoandwhen"
_POST_DET_SWE = "[post] post-detection :: swegym"
_POST_DET_TAU = "[post] post-detection :: tau"
_GOLD_LOC = "[post] gold-localization :: swegym-gold"
_GOLD_BREAKDOWN = "gold per-fault breakdown"
_GOLD_MATCHED = "gold eligibility-matched control"
_PRE_MULTI = "[pre] pre-over-privilege :: multi"
_PRE_SOURCE = "[pre] pre-over-privilege :: f1 by source"


def _judge(model: str) -> Row:
    return Row(f"llm-judge all-at-once ({model})")


TABLE_SPECS: tuple[TableSpec, ...] = (
    TableSpec(
        name="POST localization (Who&When)",
        heading_contains="fault localization",
        header=("method", "top-1", "top-3", "mrr"),
        columns={"top-1": Source(_POST_LOC, "top1"),
                 "top-3": Source(_POST_LOC, "top3"),
                 "mrr": Source(_POST_LOC, "mrr")},
        separators=frozenset({"llm-judge panel (all-at-once)", "structural / baseline (no llm)"}),
        rows={
            "gpt-5.5": _judge("gpt-5.5"),
            "claude-opus-4.8": _judge("claude-opus-4.8"),
            "gpt-5.4": _judge("gpt-5.4"),
            "deepseek-r1": _judge("deepseek-r1"),
            "gemini": _judge("gemini"),
            "qwen3-32b": _judge("qwen3-32b"),
            "gpt-oss-20b": _judge("gpt-oss-20b"),
            "llama-3.3-70b": _judge("llama-3.3-70b"),
            "gemma-3-12b": _judge("gemma-3-12b"),
            "mistral-small": _judge("mistral-small"),
            "nova-micro": _judge("nova-micro"),
            "exec-rank (sup.)": Row("exec-rank (sup.)"),
            "auditable (blast share)": Row("auditable (blast)"),
            "position prior": Row("position"),
            "pygod (graph ad, dominant)": Row("pygod (graph AD)"),
            "random": Row("random"),
        },
    ),
    TableSpec(
        name="POST detection (SWE-Gym)",
        heading_contains="failure detection",
        lead_in_contains="swe-gym",
        header=("method", "roc-auc"),
        columns={"roc-auc": Source(_POST_DET_SWE, "roc-auc")},
        rows={
            "random": Row("random"),
            "size (flat)": Row("size (flat)"),
            "pyod-flatten (ecod)": Row("pyod-flatten (ECOD)"),
            "pygod-dominant (graph ad)": Row("pygod (graph AD)"),
            "guardian (recon-ae)": Row("guardian (recon-AE)"),
            "auditable (size+deps)": Row("auditable (size+deps)"),
            "full (reference)": Row("full"),
            "g-safeguard (supervised gnn)": Row("g-safeguard (sup GNN)"),
        },
    ),
    TableSpec(
        name="POST detection (tau-bench)",
        heading_contains="failure detection",
        lead_in_contains="tau-bench",
        header=("method", "roc-auc"),
        columns={"roc-auc": Source(_POST_DET_TAU, "roc-auc")},
        rows={
            "random": Row("random"),
            "size (flat)": Row("size (flat)"),
            "pyod-flatten (ecod)": Row("pyod-flatten (ECOD)"),
            "pygod-dominant (graph ad)": Row("pygod (graph AD)"),
            "guardian (recon-ae)": Row("guardian (recon-AE)"),
            "auditable (size+deps)": Row("auditable (size+deps)"),
            "full (reference)": Row("full"),
            "g-safeguard (supervised gnn)": Row("g-safeguard (sup GNN)"),
        },
    ),
    TableSpec(
        name="Gold localization, full pool",
        heading_contains="injected faults",
        header=("method", "overall top-1", "stale-state top-1", "dropped-grounding top-1"),
        columns={"overall top-1": Source(_GOLD_BREAKDOWN, "overall", part=0),
                 "stale-state top-1": Source(_GOLD_BREAKDOWN, "stale-state", part=0),
                 "dropped-grounding top-1": Source(_GOLD_BREAKDOWN, "dropped-grounding", part=0)},
        rows={
            # The breakdown block has no random row; its overall Top-1 comes from the scored Gold
            # localization board, and the per-fault split is genuinely not reported for it.
            "random (seed-averaged)": Row("random",
                                          sources={"overall top-1": Source(_GOLD_LOC, "top1")},
                                          absent=frozenset({"stale-state top-1",
                                                            "dropped-grounding top-1"})),
            "position (leak check)": Row("position"),
            "degree (leak check)": Row("degree"),
            "has-dep (control)": Row("has-dep (control)"),
            "max-span (control)": Row("max-span (control)"),
            "auditable (dep-anomaly)": Row("auditable (dep-anomaly)"),
            "pygod (graph ad)": Row("pygod (graph AD)"),
        },
    ),
    TableSpec(
        name="Gold localization, eligible pool",
        heading_contains="injected faults",
        header=("method (eligible pool)", "overall top-1", "stale-state top-1",
                "dropped-grounding top-1"),
        columns={"overall top-1": Source(_GOLD_MATCHED, "overall", part=0),
                 "stale-state top-1": Source(_GOLD_MATCHED, "stale-state", part=0),
                 "dropped-grounding top-1": Source(_GOLD_MATCHED, "dropped-grounding", part=0)},
        rows={
            "random (matched floor)": Row("random (matched)"),
            "position": Row("position"),
            "degree": Row("degree"),
            "has-dep": Row("has-dep (control)"),
            "max-span": Row("max-span (control)"),
            "auditable (dep-anomaly)": Row("auditable (dep-anomaly)"),
            "pygod (graph ad)": Row("pygod (graph AD)"),
        },
    ),
    TableSpec(
        name="PRE over-privilege, pooled",
        heading_contains="over-privilege audit",
        header=("method", "precision", "recall", "f1", "coverage"),
        columns={"precision": Source(_PRE_MULTI, "precision"),
                 "recall": Source(_PRE_MULTI, "recall"),
                 "f1": Source(_PRE_MULTI, "f1"),
                 "coverage": Source(_PRE_MULTI, "coverage")},
        rows={
            "flag-all (floor)": Row("flag_all"),
            "flag-none (floor)": Row("flag_none"),
            "risky-permission scan": Row("flag_risky_perms"),
            "owasp-excess-permissions": Row("owasp_excess_permissions"),
            "owasp-excess-functionality": Row("owasp_excess_functionality"),
            "owasp-privilege-escalation": Row("owasp_privilege_escalation"),
            "unrequested-high-impact": Row("unrequested_high_impact"),
            "sensitive-access": Row("sensitive_access"),
            "owasp-asi-combined": Row("owasp_asi_combined"),
            "llm judge, held out (llama-3.3-70b)": Row("llm_judge_needed(llama-3.3-70b)"),
            "oracle (declared minus minimal)": Row("oracle_privilege_diff"),
        },
    ),
    TableSpec(
        name="PRE over-privilege, F1 by source",
        heading_contains="over-privilege audit",
        header=("method", "crewai", "n8n", "mcp", "injecagent", "sweagent", "synthetic"),
        columns={name: Source(_PRE_SOURCE, name)
                 for name in ("crewai", "n8n", "mcp", "injecagent", "sweagent", "synthetic")},
        rows={
            "risky-permission scan": Row("flag_risky_perms"),
            "owasp-asi-combined": Row("owasp_asi_combined"),
            "llm judge, held out": Row("llm_judge_needed(llama-3.3-70b)"),
            "oracle": Row("oracle_privilege_diff"),
        },
    ),
)

# Numeric README tables that deliberately do not come from the board. Empty today. An entry here is
# a recorded decision with a reason, which is the difference between an exemption and a blind spot.
NON_BOARD_TABLES: tuple[Exemption, ...] = ()


# ---------------------------------------------------------------------------------------------
# README prose -> board values, reasoned allowances, and statistical claim licences
# ---------------------------------------------------------------------------------------------

# Dates and versions are kept whole before the ordinary decimal alternatives are tried. The word
# boundaries deliberately admit numerals after punctuation (Top-1, GPT-5.5, LLM06:2025) while
# excluding digits embedded directly in a word. Those identifier and version occurrences are still
# prose numerals; the allowlist records why they are not board cells.
_PROSE_NUMBER = re.compile(
    r"(?<![A-Za-z0-9%])(?:\d{4}-\d{2}-\d{2}|\d+(?:\.\d+){2,}|\d+\.\d+|\d+)"
    r"(?:[eE][+-]?\d+)?%?(?![A-Za-z0-9])"
)
_PLAIN_DECIMAL = re.compile(r"^\d+\.\d+%?$")
_FENCE = re.compile(r"^(`{3,}|~{3,})")
_URL_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")
_HTML_IMAGE = re.compile(r"^<img\b.*>\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class ProseNumber:
    """One decimal numeral outside a pipe table and fenced code block."""

    value: str
    line: int
    column: int
    ordinal: int  # zero-based among equal values on the same line


@dataclass(frozen=True)
class ProseNumberAllowance:
    """Non-board prose numerals near one unique content substring, with their reason."""

    name: str
    context_contains: str
    values: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class ProseParagraph:
    """One prose paragraph in the board discussion."""

    start_line: int
    end_line: int
    text: str


# What a paragraph may say about the two arms of a recorded contrast. The vocabulary is about
# order, not about evidence strength: a benchmark reports which cell is higher and by how much, and
# says so or declines to. ``NO_ORDERING`` is the honest registration for a pair the paragraph cites
# without ranking, and it is what a near-tie should carry.
A_ABOVE_B = "a_above_b"
B_ABOVE_A = "b_above_a"
NO_ORDERING = "no_ordering"
_STATES = frozenset({A_ABOVE_B, B_ABOVE_A, NO_ORDERING})


@dataclass(frozen=True)
class Cell:
    """One golden board cell, named the way a :class:`TableSpec` names one."""

    block: str
    row: str
    column: str
    part: int | None = None  # index into a "/"-joined cell, e.g. Top-1 of "0.309/0.414/0.402"

    def render(self) -> str:
        part = "" if self.part is None else f" part {self.part}"
        return f"{self.block} / {self.row} / {self.column}{part}"


@dataclass(frozen=True)
class BoardOrdering:
    """Two golden cells a README paragraph puts in an order, highest first."""

    above: Cell
    below: Cell


@dataclass(frozen=True)
class RecordedDifference:
    """One recorded contrast a README paragraph cites, and the order it states over the two arms.

    ``states`` is :data:`A_ABOVE_B`, :data:`B_ABOVE_A`, or :data:`NO_ORDERING`. The checker compares
    it against the sign of the record's ``estimate.difference_a_minus_b`` and reads nothing else
    from the record. The record's p-values and verdict stay where they are; this gate does not
    consult them, so a changed verdict cannot turn a stated ordering green or red.
    """

    claim_id: str
    states: str = NO_ORDERING


@dataclass(frozen=True)
class BoardClaim:
    """The measurements one uniquely anchored README paragraph states.

    An entry must declare at least one ordering or one recorded difference, or say in
    ``reports_no_ordering`` why the paragraph orders nothing. An entry that declares neither is a
    failure rather than a silent pass, for the same reason a numeric table with no spec is.
    """

    name: str
    paragraph_contains: str
    orderings: tuple[BoardOrdering, ...] = ()
    differences: tuple[RecordedDifference, ...] = ()
    reports_no_ordering: str = ""


def _logical_markdown_line(line: str) -> str:
    """Strip indentation and Markdown quote leaders for structural parsing."""
    logical = line.lstrip()
    while logical.startswith(">"):
        logical = logical[1:].lstrip()
    return logical


def scan_prose_numbers(text: str) -> tuple[list[ProseNumber], list[str]]:
    """Return prose numerals and the normalized source lines they came from."""
    lines = text.replace("\r\n", "\n").split("\n")
    numbers: list[ProseNumber] = []
    fence_char = ""
    fence_width = 0
    for line_number, line in enumerate(lines, start=1):
        logical = _logical_markdown_line(line)
        marker = _FENCE.match(logical)
        if marker:
            token = marker.group(1)
            if not fence_char:
                fence_char, fence_width = token[0], len(token)
            elif token[0] == fence_char and len(token) >= fence_width:
                fence_char, fence_width = "", 0
            continue
        if fence_char or logical.startswith("|") or _HTML_IMAGE.fullmatch(logical):
            continue
        seen: Counter[str] = Counter()
        # Mask URL percent escapes without changing columns. Otherwise ``%20`` next to ``3.12``
        # becomes the invented numeral ``203.12``.
        numeric_line = _URL_ESCAPE.sub("   ", line)
        for match in _PROSE_NUMBER.finditer(numeric_line):
            value = match.group(0)
            numbers.append(ProseNumber(value=value, line=line_number, column=match.start() + 1,
                                       ordinal=seen[value]))
            seen[value] += 1
    return numbers, lines


def _scan_prose_contexts(text: str) -> list[ProseParagraph]:
    """Return non-table, non-code content blocks with whitespace normalized.

    These blocks give numeral allowances a content neighborhood that survives line-number changes
    and ordinary paragraph reflow. Blank lines, headings, tables, and code fences are boundaries.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    contexts: list[ProseParagraph] = []
    parts: list[str] = []
    start_line = 0
    fence_char = ""
    fence_width = 0

    def flush(end_line: int) -> None:
        nonlocal parts, start_line
        if parts:
            contexts.append(ProseParagraph(start_line, end_line, " ".join(parts)))
        parts, start_line = [], 0

    for line_number, line in enumerate(lines, start=1):
        logical = _logical_markdown_line(line)
        marker = _FENCE.match(logical)
        if marker:
            flush(line_number - 1)
            token = marker.group(1)
            if not fence_char:
                fence_char, fence_width = token[0], len(token)
            elif token[0] == fence_char and len(token) >= fence_width:
                fence_char, fence_width = "", 0
            continue
        if fence_char:
            continue
        if not logical or logical.startswith("|") or _HTML_IMAGE.fullmatch(logical):
            flush(line_number - 1)
            continue
        if logical.startswith("#"):
            flush(line_number - 1)
            contexts.append(ProseParagraph(line_number, line_number, logical))
            continue
        if not parts:
            start_line = line_number
        parts.append(logical)
    flush(len(lines))
    return contexts


def _golden_cell_values(blocks: Mapping[str, GoldenBlock]) -> tuple[str, ...]:
    """Extract decimal components from parsed golden cells, excluding titles, labels, and notes."""
    values: list[str] = []
    seen_blocks: set[int] = set()
    for block in blocks.values():
        if id(block) in seen_blocks:
            continue
        seen_blocks.add(id(block))
        for row in block.rows.values():
            for cell in row:
                values.extend(match.group(0) for match in _PROSE_NUMBER.finditer(cell))
    return tuple(values)


def _matches_golden_precision(value: str, golden_values: Sequence[str]) -> bool:
    """Whether a prose value is a golden cell printed or rounded to the prose precision."""
    if value in golden_values:
        return True
    if not _PLAIN_DECIMAL.fullmatch(value):
        return False
    percent = value.endswith("%")
    raw = value[:-1] if percent else value
    places = len(raw.rsplit(".", 1)[1])
    quantum = Decimal(1).scaleb(-places)
    try:
        wanted = Decimal(raw)
    except InvalidOperation:
        return False
    for candidate in golden_values:
        if bool(candidate.endswith("%")) != percent or not _PLAIN_DECIMAL.fullmatch(candidate):
            continue
        candidate_raw = candidate[:-1] if percent else candidate
        try:
            rounded = Decimal(candidate_raw).quantize(quantum, rounding=ROUND_HALF_EVEN)
        except InvalidOperation:
            continue
        if rounded == wanted:
            return True
    return False


def check_prose_numbers(readme_text: str, blocks: Mapping[str, GoldenBlock],
                        allowances: Sequence[ProseNumberAllowance]) -> tuple[list[Problem], int, int,
                                                                             int]:
    """Check every prose numeral. Returns problems, seen, board-backed, and allowed counts."""
    numbers, lines = scan_prose_numbers(readme_text)
    contexts = _scan_prose_contexts(readme_text)
    golden_values = _golden_cell_values(blocks)
    problems: list[Problem] = []
    claims: dict[tuple[int, int], list[str]] = {}
    seen_names: set[str] = set()

    by_context_value: dict[tuple[int, str], list[ProseNumber]] = {}
    for number in numbers:
        containing = [context for context in contexts
                      if context.start_line <= number.line <= context.end_line]
        if len(containing) == 1:
            by_context_value.setdefault((id(containing[0]), number.value), []).append(number)

    for allowance in allowances:
        if allowance.name in seen_names:
            problems.append(Problem("allowance-duplicate",
                                    f"prose-number allowance name {allowance.name!r} appears twice"))
            continue
        seen_names.add(allowance.name)
        if not allowance.reason.strip() or "\n" in allowance.reason:
            problems.append(Problem("allowance-reason",
                                    f"{allowance.name}: reason must be one non-empty line"))
            continue
        if not allowance.values:
            problems.append(Problem("allowance-empty",
                                    f"{allowance.name}: allowance claims no values"))
            continue
        if not allowance.context_contains.strip() or "\n" in allowance.context_contains:
            problems.append(Problem(
                "allowance-anchor",
                f"{allowance.name}: content anchor must be one non-empty line"))
            continue
        matched = [(context, context.text.count(allowance.context_contains))
                   for context in contexts if allowance.context_contains in context.text]
        match_count = sum(count for _, count in matched)
        if match_count != 1:
            first_line = matched[0][0].start_line if matched else None
            problems.append(Problem("allowance-unmatched",
                                    f"{allowance.name}: content anchor "
                                    f"{allowance.context_contains!r} occurs {match_count} times; "
                                    "it must occur exactly once", line=first_line))
            continue
        context = matched[0][0]
        anchor_lines = [line_number for line_number in range(context.start_line,
                                                              context.end_line + 1)
                        if allowance.context_contains
                        in _logical_markdown_line(lines[line_number - 1])]
        anchor_line = anchor_lines[0] if len(anchor_lines) == 1 else context.start_line
        requested: Counter[str] = Counter()
        for value in allowance.values:
            occurrence = requested[value]
            requested[value] += 1
            candidates = sorted(
                by_context_value.get((id(context), value), []),
                key=lambda number: (abs(number.line - anchor_line), number.line, number.column),
            )
            if occurrence >= len(candidates):
                problems.append(Problem(
                    "allowance-unmatched",
                    f"{allowance.name}: expected occurrence {occurrence + 1} of {value!r} near "
                    f"content anchor {allowance.context_contains!r}", line=context.start_line))
                continue
            number = candidates[occurrence]
            claims.setdefault((number.line, number.column), []).append(allowance.name)

    board_backed = 0
    allowed = 0
    for number in numbers:
        claimers = claims.get((number.line, number.column), [])
        board_match = _matches_golden_precision(number.value, golden_values)
        if board_match:
            board_backed += 1
            if claimers:
                problems.append(Problem(
                    "number-double-claimed",
                    f"prose numeral {number.value!r} is a golden value and is also claimed by "
                    f"{', '.join(claimers)}", line=number.line))
            continue
        if not claimers:
            problems.append(Problem(
                "unclaimed-number",
                f"prose numeral {number.value!r} is not a golden board cell at the printed "
                "precision and no PROSE_NUMBER_ALLOWLIST entry claims it",
                line=number.line))
        elif len(claimers) > 1:
            problems.append(Problem(
                "number-double-claimed",
                f"prose numeral {number.value!r} is claimed by {', '.join(claimers)}",
                line=number.line))
        else:
            allowed += 1
    return problems, len(numbers), board_backed, allowed


# A deliberately small lexical surface. The phrases around above/below/under avoid treating
# navigational prose such as "the table below" as a statistical comparison. Expanding this regex is
# a policy change and needs corresponding content licences.
COMPARATIVE_WORD = re.compile(
    r"\b(?:beat(?:s|en)?|best|better|clear(?:s|ed)?|exceed(?:s|ed)?|higher|highest|lead|"
    r"lower|lowest|opposite|outperform(?:s|ed)?|strongest|worse|worst)\b"
    r"|\b(?:above|below|under)\s+(?:chance|the|a|an|\d)\b"
    r"|\b(?:sit(?:s)?|land(?:s)?|stay(?:s)?|fall(?:s)?)\s+(?:above|below|under)\b"
    r"|\b(?:tie(?:s|d)?)\s+(?:at|the|with)\b"
    r"|\b(?:range(?:s|d)?|score(?:s|d)?|span(?:s|ned)?)\s+from\b"
    r"|\btop of (?:the|that|its)\b",
    re.IGNORECASE,
)

# Phrases that discuss a pair without ordering it. They are trigger vocabulary for the paragraph
# scan and nothing more: they license no claim and satisfy no rule. Their job is that a paragraph
# which touches a comparison only in the negative is still accounted for by ``BOARD_CLAIMS`` rather
# than falling outside the scan. Keep the tuple literal, because every addition widens what the
# coverage gate has to account for.
NON_ORDERING_PHRASES = (
    r"do not separate",
    r"does not separate",
    r"fail to separate",
    r"fails to separate",
    r"failing to separate",
    r"declines to separate",
    r"do not resolve",
    r"does not resolve",
    r"does not improve on",
    r"do not improve on",
    r"not as a result",
    r"not a result",
    r"next to",
    r"none of the reported methods reaches",
    r"unresolved",
    r"tie(?:s|d)?\s+(?:at|the|with)",
)
NON_ORDERING_WORD = re.compile(
    r"\b(?:" + "|".join(NON_ORDERING_PHRASES) + r")\b",
    re.IGNORECASE,
)

# The whole lexical surface of the reporting policy, and it is two spellings wide on purpose.
#
# The policy is about inference: prose reports observed scores, differences, and explicitly labelled
# marginal intervals, and does not read a reliable advantage out of a threshold crossing. No regular
# expression sees an inference. What a regular expression can see is the apparatus a reader is
# invited to do the inference with, and in this README that apparatus is exactly two things: the
# name of the family-wise correction, and a printed p-value.
#
# Nothing wider is gated. ``separates``, ``resolves`` and ``unresolved`` all carry ordinary meanings
# in this file -- "separate tracks", "the pillars are separate", "188 failed, 188 resolved" as
# SWE-Gym run outcomes, "unresolved terms" in the licensing section, "pip resolves whatever torch is
# newest". A gate over those would fire on ten sentences that are not comparisons at all, and a gate
# that cries wolf is switched off within a week. This one has no false positive in the file it
# guards.
ADJUDICATION_MARKER = re.compile(r"\bHolm\b|\bp\s*[=<>]\s*\.?\d", re.IGNORECASE)


def scan_comparative_paragraphs(text: str) -> list[ProseParagraph]:
    """Find lexical comparison paragraphs inside the README's ``The Boards`` section."""
    lines = text.replace("\r\n", "\n").split("\n")
    paragraphs: list[ProseParagraph] = []
    paragraph: list[str] = []
    start_line = 0
    in_boards = False
    fence_char = ""
    fence_width = 0

    def flush(end_line: int) -> None:
        nonlocal paragraph, start_line
        if paragraph:
            prose = " ".join(part.strip() for part in paragraph)
            if COMPARATIVE_WORD.search(prose) or NON_ORDERING_WORD.search(prose):
                paragraphs.append(ProseParagraph(start_line, end_line, prose))
        paragraph, start_line = [], 0

    for line_number, line in enumerate(lines, start=1):
        logical = _logical_markdown_line(line)
        marker = _FENCE.match(logical)
        if marker:
            flush(line_number - 1)
            token = marker.group(1)
            if not fence_char:
                fence_char, fence_width = token[0], len(token)
            elif token[0] == fence_char and len(token) >= fence_width:
                fence_char, fence_width = "", 0
            continue
        if fence_char:
            continue
        if logical.startswith("## ") and not logical.startswith("### "):
            flush(line_number - 1)
            in_boards = norm_label(logical[3:]) == "the boards"
            continue
        if not in_boards:
            continue
        if (not logical or logical.startswith("#") or logical.startswith("|")
                or _HTML_IMAGE.fullmatch(logical)
                or (re.match(r"^(?:[-*+] |\d+[.)] )", logical) and paragraph)):
            flush(line_number - 1)
            if (not logical or logical.startswith(("#", "|"))
                    or _HTML_IMAGE.fullmatch(logical)):
                continue
        if not paragraph:
            start_line = line_number
        paragraph.append(logical)
    flush(len(lines))
    return paragraphs


def _claim_index(data: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], Counter[str]]:
    claims = data.get("claims")
    if not isinstance(claims, list):
        return {}, Counter()
    counts: Counter[str] = Counter(
        claim.get("id") for claim in claims
        if isinstance(claim, Mapping) and isinstance(claim.get("id"), str)
    )
    index = {
        claim["id"]: claim for claim in claims
        if isinstance(claim, Mapping) and isinstance(claim.get("id"), str)
        and counts[claim["id"]] == 1
    }
    return index, counts


def _cell_number(text: str) -> tuple[Decimal, bool] | None:
    """Parse the leading printed value out of a golden cell, with its percent flag.

    Board cells are not bare numbers. They carry spreads (``0.824+/-0.007``), realized rates in
    parentheses (``0.024 (6.1%)``), and slash triples that ``Cell.part`` has already split. Taking
    the first numeral is what every other consumer of these cells does, and the percent flag travels
    with it so a fraction and a percentage can never be ordered against each other by accident.
    """
    match = _PROSE_NUMBER.search(text.strip())
    if match is None:
        return None
    token = match.group(0)
    percent = token.endswith("%")
    try:
        return Decimal(token[:-1] if percent else token), percent
    except InvalidOperation:
        return None


def _check_ordering(name: str, ordering: BoardOrdering, blocks: Mapping[str, GoldenBlock],
                    counts: Counter, line: int | None) -> list[Problem]:
    """Compare one stated ordering against the two golden cells it names."""
    values: list[tuple[Decimal, bool]] = []
    for cell in (ordering.above, ordering.below):
        source = Source(cell.block, cell.column, cell.part)
        printed, why = _golden_value(blocks, counts, source, cell.row)
        if printed is None:
            return [Problem("board-claim-unresolved",
                            f"{name}: {cell.render()}: {why}", line=line)]
        parsed = _cell_number(printed)
        if parsed is None:
            return [Problem("board-claim-unreadable",
                            f"{name}: {cell.render()} prints {printed!r}, which carries no value "
                            "this ordering can compare", line=line)]
        values.append(parsed)
    (high, high_percent), (low, low_percent) = values
    if high_percent != low_percent:
        return [Problem("board-claim-unreadable",
                        f"{name}: {ordering.above.render()} and {ordering.below.render()} are "
                        "printed on different scales, so ordering them compares nothing",
                        line=line)]
    if high > low:
        return []
    relation = "equals" if high == low else "is below"
    return [Problem("ordering-contradicted",
                    f"{name}: the paragraph puts {ordering.above.render()} above "
                    f"{ordering.below.render()}, and the board {relation} it ({high} against "
                    f"{low})", line=line)]


def _check_difference(name: str, difference: RecordedDifference,
                      claims: Mapping[str, Mapping[str, Any]], claim_counts: Counter[str],
                      line: int | None) -> list[Problem]:
    """Compare one stated ordering against the recorded contrast's own estimate."""
    claim_id = difference.claim_id
    if difference.states not in _STATES:
        return [Problem("board-claim-states",
                        f"{name}: {claim_id} states {difference.states!r}; it must be one of "
                        f"{', '.join(sorted(_STATES))}", line=line)]
    if claim_counts.get(claim_id, 0) > 1:
        return [Problem("claim-duplicate",
                        f"{name}: claim ID {claim_id!r} occurs {claim_counts[claim_id]} times",
                        line=line)]
    claim = claims.get(claim_id)
    if claim is None:
        return [Problem("claim-missing", f"{name}: no statistical claim has ID {claim_id!r}",
                        line=line)]
    estimate = claim.get("estimate")
    parts: list[float] = []
    if isinstance(estimate, Mapping):
        for field_name in ("a", "b", "difference_a_minus_b"):
            value = estimate.get(field_name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                parts.append(float(value))
    if len(parts) != 3:
        return [Problem("measurement-missing",
                        f"{name}: {claim_id} carries no numeric estimate with 'a', 'b', and "
                        "'difference_a_minus_b', so it measures nothing this paragraph can state",
                        line=line)]
    a, b, recorded = parts
    if abs((a - b) - recorded) > 1e-9:
        return [Problem("measurement-inconsistent",
                        f"{name}: {claim_id} records a={a!r} and b={b!r}, whose difference is "
                        f"{a - b!r}, and reports difference_a_minus_b={recorded!r}", line=line)]
    if difference.states == NO_ORDERING:
        return []
    wanted_above = difference.states == A_ABOVE_B
    if (recorded > 0) == wanted_above and recorded != 0:
        return []
    high, low = ((estimate.get("a_name"), estimate.get("b_name")) if wanted_above
                 else (estimate.get("b_name"), estimate.get("a_name")))
    return [Problem("ordering-contradicted",
                    f"{name}: the paragraph puts {high!r} above {low!r}, and {claim_id} records "
                    f"difference_a_minus_b={recorded!r}", line=line)]


def check_adjudicating_numbers(readme_text: str) -> list[Problem]:
    """Report every printed correction name and p-value outside a fenced code block.

    This is the whole lexical half of the reporting policy. See :data:`ADJUDICATION_MARKER` for why
    it is two spellings wide and for what deliberately stays ungated.
    """
    problems: list[Problem] = []
    fence_char = ""
    fence_width = 0
    for line_number, line in enumerate(readme_text.replace("\r\n", "\n").split("\n"), start=1):
        logical = _logical_markdown_line(line)
        marker = _FENCE.match(logical)
        if marker:
            token = marker.group(1)
            if not fence_char:
                fence_char, fence_width = token[0], len(token)
            elif token[0] == fence_char and len(token) >= fence_width:
                fence_char, fence_width = "", 0
            continue
        if fence_char:
            continue
        for hit in ADJUDICATION_MARKER.finditer(line):
            problems.append(Problem(
                "adjudicating-number",
                f"{hit.group(0)!r} prints a correction name or a p-value. This benchmark reports "
                "the measurement and leaves every p-value in "
                "tools/statistical_tests_results.json", line=line_number))
    return problems


def check_board_claims(readme_text: str, blocks: Mapping[str, GoldenBlock], counts: Counter,
                       statistical_data: Mapping[str, Any],
                       board_claims: Sequence[BoardClaim]) -> tuple[list[Problem], int, int]:
    """Validate each comparative paragraph's measurements. Returns problems, seen, and claimed.

    A stated board ordering is checked against the two golden cells it names. A stated ordering over
    a recorded contrast is checked against the sign of that record's ``difference_a_minus_b``. No
    verdict, p-value, or interval endpoint is read, so a comparison that reaches significance and
    one that does not are checked exactly alike.
    """
    paragraphs = scan_comparative_paragraphs(readme_text)
    claims, claim_counts = _claim_index(statistical_data)
    problems: list[Problem] = []
    claimed: dict[int, list[str]] = {}
    seen_names: set[str] = set()

    if not isinstance(statistical_data.get("claims"), list):
        problems.append(Problem("claim-schema",
                                "statistical results must contain a list named 'claims'"))

    for entry in board_claims:
        if entry.name in seen_names:
            problems.append(Problem("board-claim-duplicate",
                                    f"board claim name {entry.name!r} appears twice"))
            continue
        seen_names.add(entry.name)
        if not entry.orderings and not entry.differences:
            reason = entry.reports_no_ordering
            if not reason.strip() or "\n" in reason:
                problems.append(Problem(
                    "board-claim-empty",
                    f"{entry.name}: declares no ordering and no recorded difference. Name the "
                    "measurements the paragraph states, or record in reports_no_ordering, on one "
                    "line, why it orders nothing."))
                continue
        if not entry.paragraph_contains.strip() or "\n" in entry.paragraph_contains:
            problems.append(Problem(
                "board-claim-anchor",
                f"{entry.name}: content anchor must be one non-empty line"))
            continue
        matched = [(paragraph, paragraph.text.count(entry.paragraph_contains))
                   for paragraph in paragraphs
                   if entry.paragraph_contains in paragraph.text]
        match_count = sum(count for _, count in matched)
        if match_count != 1:
            first_line = matched[0][0].start_line if matched else None
            problems.append(Problem(
                "board-claim-unmatched",
                f"{entry.name}: content anchor {entry.paragraph_contains!r} occurs "
                f"{match_count} times in comparative paragraphs; it must occur exactly once",
                line=first_line))
            continue
        paragraph = matched[0][0]
        claimed.setdefault(id(paragraph), []).append(entry.name)
        for ordering in entry.orderings:
            problems.extend(_check_ordering(entry.name, ordering, blocks, counts,
                                            paragraph.start_line))
        for difference in entry.differences:
            problems.extend(_check_difference(entry.name, difference, claims, claim_counts,
                                              paragraph.start_line))

    for paragraph in paragraphs:
        claimers = claimed.get(id(paragraph), [])
        if not claimers:
            word = COMPARATIVE_WORD.search(paragraph.text) or NON_ORDERING_WORD.search(
                paragraph.text)
            problems.append(Problem(
                "unaccounted-comparison",
                f"paragraph {paragraph.start_line}-{paragraph.end_line} contains comparative "
                f"word {word.group(0)!r} but no BOARD_CLAIMS content anchor claims it",
                line=paragraph.start_line))
        elif len(claimers) > 1:
            problems.append(Problem(
                "comparison-double-claimed",
                f"paragraph {paragraph.start_line}-{paragraph.end_line} is claimed by "
                f"{', '.join(claimers)}", line=paragraph.start_line))
    return problems, len(paragraphs), len(claimed)


# Populated below from the current README. Each allowance claims the exact occurrences named in its
# value tuple near one unique content anchor. Entries are grouped only when they share one reason.
def _declared_test_count() -> tuple[str, ...]:
    """The contract-test count the README badge must display, read from the manifest.

    Pinning the number here as a literal made every added test fail this checker until someone
    edited two files, which is how an allowance becomes a nuisance and then gets deleted. The value
    the badge must show is already decided by ``tests/expected_tests.txt``, and
    ``tools/check_test_report.py`` fails when the badge disagrees with it. Reading it here keeps this
    checker fail-closed on the same fact without a second copy of the number: a badge showing
    anything other than the declared count is not claimed by this allowance and fails as an
    unclaimed numeral.
    """

    manifest = ROOT / "tests" / "expected_tests.txt"
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    declared = sum(1 for line in lines
                   if re.match(r"^(always|needs-\S+)\s+\S", line))
    return (str(declared),) if declared else ()


PROSE_NUMBER_ALLOWLIST: tuple[ProseNumberAllowance, ...] = (
    ProseNumberAllowance("python-badge", "badge/python-", ("3.10", "3.12"),
                         "Supported Python versions displayed by the badge."),
    ProseNumberAllowance("test-badge", "badge/tests-", _declared_test_count(),
                         "Contract-test count displayed by the badge, read from the manifest."),
    ProseNumberAllowance("release-current", "Release 0.1.1", ("0.1.1",),
                         "Package release version."),
    ProseNumberAllowance("release-previous", "0.1.0 wheel", ("0.1.0",),
                         "Earlier package release version."),
    ProseNumberAllowance("quickstart-config-count", "eleven scored rows over 1187",
                         ("1187",),
                         "Count of committed PRE configurations."),
    ProseNumberAllowance("download-size", "The POST, LIVE, and Gold boards need",
                         ("320",),
                         "Approximate corpus download size in megabytes."),
    ProseNumberAllowance("rjudge-arxiv", "[R-Judge]", ("2401.10019",),
                         "arXiv paper identifier."),
    ProseNumberAllowance("asb-arxiv", "[Agent Security Bench]", ("2410.02644",),
                         "arXiv paper identifier."),
    ProseNumberAllowance("catchbench-arxiv-badge", "badge/status-arXiv",
                         ("2608.22808", "2608.22808"),
                         "CatchBench's own arXiv identifier, shown by the badge and its link."),
    ProseNumberAllowance("catchbench-arxiv-paper",
                         "[CatchBench: When Can an Agent Failure Be Caught?]",
                         ("2608.22808", "2608.22808"),
                         "CatchBench's own arXiv identifier, repeated in the Paper section."),
    ProseNumberAllowance("synthetic-record-count", "56 authored", ("56",),
                         "Count of CatchBench-authored synthetic records."),
    ProseNumberAllowance(
        "licence-counts-a", "committed PRE records",
        ("1187", "663", "626", "24", "2.0", "9", "3.0"),
        "Record and licence-version counts in the data licensing audit."),
    ProseNumberAllowance(
        "licence-counts-b", "remaining 524",
        ("2", "4.0", "1", "3.0", "1", "1.1", "524"),
        "Record counts and licence versions in the data licensing audit."),
    ProseNumberAllowance("licence-audit-date", "were checked", ("524", "2026-08-21"),
                         "Audited record count and audit date."),
    ProseNumberAllowance("licence-recheck-count", "found that 106", ("106",),
                         "Count found by the licence re-check."),
    ProseNumberAllowance("noassertion-count", "All 524", ("524",),
                         "Count of records marked NOASSERTION."),
    ProseNumberAllowance("restricted-licence-versions", "records stay in the release",
                         ("3.0", "3.0", "4.0", "1.1", "1.1", "11", "1.1"),
                         "Licence version identifiers."),
    ProseNumberAllowance("licence-link-fragment", "why-the-gpl-30-and-cc-by-40",
                         ("30", "40"),
                         "Markdown fragment encodes licence version identifiers."),
    ProseNumberAllowance("localization-corpus-counts", "126 failed runs",
                         ("126", "1099", "11%"),
                         "Who&When run, step, and fault-rate counts."),
    ProseNumberAllowance("headline-localization-counts",
                         "Eight of eleven LLM judges score between",
                         ("1", "126"),
                         "Top-1 metric index and Who&When run count in the headline summary."),
    ProseNumberAllowance("judge-panel-count", "11-model panel", ("11",),
                         "Count of judge models in the panel."),
    ProseNumberAllowance("localization-band-run-count", "band from 0.333",
                         ("126",), "Who&When run count supporting the band disclosure."),
    ProseNumberAllowance("gpt-model-version", "GPT-5.5 has the highest", ("5.5",),
                         "Model version in a method name."),
    ProseNumberAllowance("localization-metric-name", "Top-1 score here",
                         ("1", "1", "3"), "Metric indices repeated in prose."),
    ProseNumberAllowance("localization-metric-name-random", "has the only displayed Top-1", ("1",),
                         "Metric index in prose."),
    ProseNumberAllowance("swegym-run-counts", "SWE-Gym, 376 runs",
                         ("376", "188", "188"),
                         "SWE-Gym total, failed, and resolved run counts."),
    ProseNumberAllowance("tau-run-counts", "tau-bench, 660 runs",
                         ("660", "363", "297"),
                         "tau-bench total, failed, and resolved run counts."),
    ProseNumberAllowance("detection-differences", "corpora (+",
                         ("0.142", "0.046"),
                         "Paired ROC-AUC point-estimate differences, each rounded from the exact "
                         "difference rather than taken between the displayed cells. Subtracting "
                         "the two displayed SWE-Gym cells gives 0.141; the exact difference is "
                         "0.141625. Section 5 of the paper prints the same 0.142."),
    # The four Holm-adjusted p-value allowances that stood here are gone. Under this benchmark's
    # reporting policy no printed p-value is a board number the README may carry with a reason: the
    # p-values live in tools/statistical_tests_results.json. The matching README prose has been
    # removed; reintroducing it fails both the numeral and adjudication checks.
    ProseNumberAllowance("gsafeguard-seed-mean", "cross-validation seeds", ("0.824",),
                         "Five-seed mean reported in the golden reading notes."),
    ProseNumberAllowance("dominant-publication-year", "Ding et al.", ("2019",),
                         "Publication year."),
    ProseNumberAllowance("guardian-citation", "GUARDIAN (Zhou et al.",
                         ("2025", "2505.19234"),
                         "Publication year and arXiv identifier."),
    ProseNumberAllowance("gsafeguard-citation", "G-Safeguard (Wang et al.",
                         ("2025", "2502.11127"),
                         "Publication year and arXiv identifier."),
    ProseNumberAllowance("gold-run-count", "There are 188 clean SWE-Gym runs", ("188",),
                         "Count of clean SWE-Gym runs used for injection."),
    ProseNumberAllowance("gold-fault-counts", "fault each: 82 stale-state", ("82", "106"),
                         "Counts of injected faults by mechanism."),
    ProseNumberAllowance("eligible-candidate-count", "mean of 7.4", ("7.4",),
                         "Mean eligible candidate count from the golden board heading."),
    ProseNumberAllowance("gold-dropped-analytic-floor",
                         "`has-dep` scores 0.005 on dropped grounding", ("0.035",),
                         "Analytic random floor for dropped grounding. It comes from the recorded "
                         "gold.hasdep.dropped.floor contrast, not from a displayed board cell."),
    ProseNumberAllowance("gold-mechanism-dropped-analytic-floor",
                         "dropped-grounding column carries no comparable margin", ("0.035",),
                         "The mechanism summary repeats the analytic random floor from "
                         "gold.hasdep.dropped.floor, which is not a displayed board cell."),
    ProseNumberAllowance("gold-seed-mean", "0.795 +/-", ("0.795",),
                         "Selection-controlled five-seed mean in the golden reading notes."),
    ProseNumberAllowance("artifact-target-counts-a", "uniquely ranks all",
                         ("82", "106"), "Counts of injected targets by mechanism."),
    ProseNumberAllowance("artifact-target-counts-b", "flagging 0 of 188",
                         ("1", "0", "188"),
                         "Metric index and clean-run diagnostic counts."),
    ProseNumberAllowance("edge-count-shifts-a", "valid dependency-edge count",
                         ("9.2", "7.9"), "Mean dependency-edge counts in the shift diagnostic."),
    ProseNumberAllowance("edge-count-shifts-b", "run-level shift", ("6.9",),
                         "Mean dependency-edge count after injection."),
    ProseNumberAllowance("span-shift", "board's span line",
                         ("188", "8.6", "9.4", "53", "82"),
                         "Mean span values and affected-run counts in the shift diagnostic."),
    ProseNumberAllowance("attribution-pair-count", "166 paired", ("166",),
                         "Count of paired cause-attribution runs."),
    ProseNumberAllowance("attribution-seed-mean", "0.671 +/-", ("0.671",),
                         "Five-seed cause-attribution mean in the golden reading notes."),
    ProseNumberAllowance("online-target-rate", "displayed 5%", ("5%",),
                         "Target false-positive-rate threshold."),
    ProseNumberAllowance(
        "swegym-bar-family", "The 20 SWE-Gym cells read against the 0.70 bar",
        ("20",),
        "Count of SWE-Gym method-prefix cells whose exploratory intervals the paper reports.",
    ),
    # swegym-bar-family also bound 75%, 50%, and 75% labels in statements about sides of the bar.
    # Those statements were removed; the rewritten paragraph retains only the exploratory count.
    # live-figure-bar-families bound "Each corpus registers all 20 nonrandom method-prefix cells"
    # and its 20, 75%, and 18 numerals. The caption now describes curves and individual intervals;
    # it no longer reports a family size, above-bar prefixes, or a tau separating count.
    ProseNumberAllowance(
        "live-stale-descriptive-design", "displayed cells from 82 paired runs",
        ("82", "410"),
        "Paired-run count and the explicitly rejected pseudo-replication count.",
    ),
    ProseNumberAllowance("pre-config-count", "It runs over 1187 configurations", ("1187",),
                         "Count of PRE configurations."),
    ProseNumberAllowance("label-kappa", "overall Cohen's kappa", ("0.666",),
                         "Inter-rater agreement statistic."),
    ProseNumberAllowance("owasp-edition", "LLM06:2025", ("2025",),
                         "OWASP category edition year."),
    ProseNumberAllowance("coverage-total", "share of the 1187", ("1187",),
                         "Total PRE configuration count."),
    ProseNumberAllowance("coverage-abstentions", "held-out judge abstains on 5",
                         ("5", "1182"), "Abstention and evaluable-configuration counts."),
    ProseNumberAllowance("judge-evaluable-count", "1182 configs", ("1182",),
                         "Count of configurations evaluated by the held-out judge."),
    ProseNumberAllowance("rule-prediction-counts", "three rules make",
                         ("37", "59", "676"), "Prediction counts for the three named rules."),
    ProseNumberAllowance("rule-board-count", "1187-config board", ("1187",),
                         "Total PRE configuration count."),
    ProseNumberAllowance("judge-model-versions", "other judges",
                         ("5.5", "3.3"), "Model versions in judge names."),
    ProseNumberAllowance("source-label-kappa", "Label origin per column", ("0.666",),
                         "Inter-rater agreement statistic."),
    ProseNumberAllowance("injecagent-shortcut-counts", "all 340 injecagent configurations",
                         ("340", "510", "0.106"),
                         "Corpus construction diagnostic counts and cross-source score."),
    ProseNumberAllowance("mcp-oracle-f1", "where the mean excess ratio is", ("0.084",),
                         "Per-source oracle F1 not emitted as a mapped scoring-board cell."),
    ProseNumberAllowance("judge-answer-count", "all 1187", ("1187",),
                         "Count of held-out judge replies."),
    ProseNumberAllowance("judge-coverage-counts", "cover 1182 of 1187",
                         ("1182", "1187"), "Evaluable and total configuration counts."),
    ProseNumberAllowance("abstention-source-counts", "n8n is scored on",
                         ("215", "219", "143", "144"),
                         "Evaluable and total counts for the two affected sources."),
    ProseNumberAllowance("large-server-counts", "622-capability",
                         ("622", "337", "2893"),
                         "Capability and excess-label counts for the cited server."),
    ProseNumberAllowance("task-list-localization-metrics", "Fault localization** (POST",
                         ("1", "3"), "Metric indices in the task list."),
    ProseNumberAllowance("task-list-gold-metrics", "Gold fault localization** (POST",
                         ("1", "3"), "Metric indices in the task list."),
    ProseNumberAllowance("task-list-pre-count", "Over-privilege audit** (PRE", ("1187",),
                         "Count of PRE configurations."),
    ProseNumberAllowance("full-board-download-size", "Every board, including POST", ("320",),
                         "Approximate corpus download size in megabytes."),
    # Two counts now, because the release holds 43 Who&When caches: the 31 the arena scores and the
    # 12 addendum caches, which ship for reproducibility but sit outside the globbed directory. Ten
    # of the twelve enter the addendum table and two are held-out same-channel repeats. The
    # unqualified "31" was true of the arena and false of the release.
    ProseNumberAllowance("retained-cache-count", "31 published Who&When", ("31", "12"),
                         "Counts of retained Who&When judge caches, published and addendum."),
)


# The eight-model Who&When band. The paragraph cites the pair set and ranks nothing inside it, so
# every member is registered with NO_ORDERING: the binding is to the record, and the ordering check
# has nothing to check. Deleting one of these contrasts from the record still turns the README red,
# which is the source binding this registry exists for.
_LOCALIZATION_BAND_CLAIMS = (
    "loc.band.gpt-5.5.vs.claude-opus-4.8",
    "loc.band.gpt-5.5.vs.gpt-5.4",
    "loc.band.gpt-5.5.vs.deepseek-r1",
    "loc.band.gpt-5.5.vs.gemini",
    "loc.band.gpt-5.5.vs.qwen3-32b",
    "loc.band.gpt-5.5.vs.gpt-oss-20b",
    "loc.band.gpt-5.5.vs.llama-3.3-70b",
    "loc.band.claude-opus-4.8.vs.gpt-5.4",
    "loc.band.claude-opus-4.8.vs.deepseek-r1",
    "loc.band.claude-opus-4.8.vs.gemini",
    "loc.band.claude-opus-4.8.vs.qwen3-32b",
    "loc.band.claude-opus-4.8.vs.gpt-oss-20b",
    "loc.band.claude-opus-4.8.vs.llama-3.3-70b",
    "loc.band.gpt-5.4.vs.deepseek-r1",
    "loc.band.gpt-5.4.vs.gemini",
    "loc.band.gpt-5.4.vs.qwen3-32b",
    "loc.band.gpt-5.4.vs.gpt-oss-20b",
    "loc.band.gpt-5.4.vs.llama-3.3-70b",
    "loc.band.deepseek-r1.vs.gemini",
    "loc.band.deepseek-r1.vs.qwen3-32b",
    "loc.band.deepseek-r1.vs.gpt-oss-20b",
    "loc.band.deepseek-r1.vs.llama-3.3-70b",
    "loc.band.gemini.vs.qwen3-32b",
    "loc.band.gemini.vs.gpt-oss-20b",
    "loc.band.gemini.vs.llama-3.3-70b",
    "loc.band.qwen3-32b.vs.gpt-oss-20b",
    "loc.band.qwen3-32b.vs.llama-3.3-70b",
    "loc.band.gpt-oss-20b.vs.llama-3.3-70b",
)

_GOLD_ATTRIB = "[post] gold-attribution :: swegym-gold"
# The prefix board prints one block per corpus under the same title, so the golden index numbers
# them. SWE-Gym is first in the board and tau-bench second.
_LIVE_PREFIX_SWE = "live streaming early-warning#1"
_LIVE_PREFIX_TAU = "live streaming early-warning#2"

_LIVE_METHODS = ("size (flat)", "auditable (size+deps)", "full", "pyod (ECOD)",
                 "dep-span (online)")
_LIVE_PREFIXES = ("25", "50", "75", "100")

# Every tau-bench prefix cell sits below the fixed 0.70 bar, which is what the paragraph says and
# what the board prints: the highest tau cell anywhere on that board is full at 100%, 0.665. Each
# bar contrast records the cell as ``a`` and 0.70 as ``b``, so "below the bar" is B_ABOVE_A.
_TAU_BELOW_BAR = tuple(
    RecordedDifference(f"live.tau.bar.{prefix}.{method}", B_ABOVE_A)
    for prefix in _LIVE_PREFIXES for method in _LIVE_METHODS
)

# _SWE_BAR_SIDES bound full above the bar at all four prefixes and dep-span below it at 25%, 50%,
# and 75%. The rewritten LIVE paragraph no longer states those sides, so those bindings are gone.
# It retains the SWE-Gym 25% method comparison and the tau-bench below-bar measurements.


def _loc(row: str, column: str = "top1") -> Cell:
    return Cell(_POST_LOC, row, column)


def _det_swe(row: str) -> Cell:
    return Cell(_POST_DET_SWE, row, "roc-auc")


def _det_tau(row: str) -> Cell:
    return Cell(_POST_DET_TAU, row, "roc-auc")


def _matched(row: str, column: str) -> Cell:
    """One Top-1 cell of the Gold eligibility-matched control, whose cells are Top-1/Top-3/MRR."""
    return Cell(_GOLD_MATCHED, row, column, part=0)


def _prefix(block: str, row: str, prefix: str) -> Cell:
    return Cell(block, row, f"{prefix}%")


# Each entry names the measurements one README paragraph states. Anchors are chosen from the
# descriptive half of a paragraph rather than from a sentence about a test outcome. A rewrite that
# changes a bound measurement must reconcile its entry even when the new prose remains descriptive.
BOARD_CLAIMS: tuple[BoardClaim, ...] = (
    BoardClaim(
        "Who&When localization board",
        "GPT-5.5 has the highest Top-1 score here",
        orderings=(
            # "the highest Top-1 score here at 0.452", against the runner-up that would displace it.
            BoardOrdering(_loc("llm-judge all-at-once (gpt-5.5)"),
                          _loc("llm-judge all-at-once (claude-opus-4.8)")),
            # "Mistral-Small and Nova-Micro fall below the position prior on point estimate".
            BoardOrdering(_loc("position"), _loc("llm-judge all-at-once (mistral-small)")),
            BoardOrdering(_loc("position"), _loc("llm-judge all-at-once (nova-micro)")),
            # "localizes beyond the prior on Top-3", and the Top-1 margin the same sentence names.
            BoardOrdering(_loc("exec-rank (sup.)", "top3"), _loc("position", "top3")),
            BoardOrdering(_loc("exec-rank (sup.)"), _loc("position")),
            # "the only displayed Top-1 point estimate below chance, 0.048 against the 0.119 floor".
            BoardOrdering(_loc("random"), _loc("pygod (graph AD)")),
        ),
        differences=tuple(RecordedDifference(claim_id) for claim_id in _LOCALIZATION_BAND_CLAIMS)
        + (
            RecordedDifference("loc.exec.vs.position.top1", A_ABOVE_B),
            RecordedDifference("loc.exec.vs.position.top3", A_ABOVE_B),
            RecordedDifference("loc.exec.vs.position.mrr"),
            RecordedDifference("loc.position.vs.mistral-small", A_ABOVE_B),
            RecordedDifference("loc.position.vs.nova-micro", A_ABOVE_B),
            RecordedDifference("loc.gpt55.vs.random", A_ABOVE_B),
            RecordedDifference("loc.gpt55.vs.pygod (graph AD)", A_ABOVE_B),
            RecordedDifference("loc.gpt55.vs.position", A_ABOVE_B),
            RecordedDifference("loc.gpt55.vs.exec-rank (sup.)", A_ABOVE_B),
            # The blast share ties the prior at displayed precision, so the paragraph ranks neither.
            RecordedDifference("loc.gpt55.vs.auditable (blast)", A_ABOVE_B),
        ),
    ),
    BoardClaim(
        "headline detection boards",
        "size-normalized dependency block scores above the size-and-counts baseline",
        orderings=(
            BoardOrdering(_det_swe("auditable (size+deps)"), _det_swe("size (flat)")),
            BoardOrdering(_det_tau("auditable (size+deps)"), _det_tau("size (flat)")),
            # "PyOD ECOD exceeds the linear size model (0.765 over 0.663)", and the dependency
            # method "scores higher again at 0.804".
            BoardOrdering(_det_swe("pyod-flatten (ECOD)"), _det_swe("size (flat)")),
            BoardOrdering(_det_swe("auditable (size+deps)"), _det_swe("pyod-flatten (ECOD)")),
        ),
        differences=(
            RecordedDifference("det.swe.auditable.vs.size", A_ABOVE_B),
            RecordedDifference("det.tau.auditable.vs.size", A_ABOVE_B),
            # The structural block and the full reference tie at displayed precision on tau-bench,
            # and the recorded difference is -0.002. Ordering them either way is what this
            # registration refuses.
            RecordedDifference("det.tau.auditable.vs.full"),
            RecordedDifference("det.swe.auditable.vs.ecod", A_ABOVE_B),
        ),
    ),
    BoardClaim(
        "unsupervised arena spans",
        "wider unsupervised arena behind the headline table",
        orderings=(
            # "the tabular detectors span 0.319 to 0.625, all below the 0.663 size baseline": the
            # top of that span is the cell that would break the statement.
            BoardOrdering(_det_swe("size (flat)"), _det_swe("pyod-copod")),
            # "two of its members clear that baseline: CONAD at 0.750 and GAAN at 0.850".
            BoardOrdering(_det_swe("pygod-conad"), _det_swe("size (flat)")),
            BoardOrdering(_det_swe("pygod-gaan"), _det_swe("size (flat)")),
            # "On tau-bench both families stay below the 0.619 size baseline", again at the top of
            # each span.
            BoardOrdering(_det_tau("size (flat)"), _det_tau("pyod-copod")),
            BoardOrdering(_det_tau("size (flat)"), _det_tau("pygod-conad")),
        ),
        # GUARDIAN "scores 0.767 next to ECOD at 0.765", which prints both and ranks neither.
        differences=(RecordedDifference("det.swe.guardian.vs.ecod"),),
    ),
    BoardClaim(
        "SWE-Gym graph maximum",
        "GAAN's 0.850 is a single-seed number",
        orderings=(
            # "DOMINANT lands under the random floor on Who&When localization".
            BoardOrdering(_loc("random"), _loc("pygod (graph AD)")),
            # "every PyGOD entry stays below the size baseline on tau-bench", at the family top.
            BoardOrdering(_det_tau("size (flat)"), _det_tau("pygod-conad")),
            # "the highest supervised SWE-Gym point estimate at 0.828".
            BoardOrdering(_det_swe("g-safeguard (sup GNN)"), _det_swe("full")),
        ),
        differences=(
            RecordedDifference("det.swe.auditable.vs.ecod"),
            RecordedDifference("det.swe.auditable.vs.guardian"),
        ),
    ),
    BoardClaim(
        "detection baselines and lineage",
        "The graph-AD baselines are ports of published methods",
        orderings=(
            # "Its 0.828 is the highest displayed value on the SWE-Gym table".
            BoardOrdering(_det_swe("g-safeguard (sup GNN)"), _det_swe("full")),
        ),
        # The paragraph cites the paired comparison with the full-feature reference and orders
        # neither arm through it; the displayed maximum above carries the ordering it does state.
        differences=(RecordedDifference("det.swe.gsafeguard.vs.full"),),
    ),
    BoardClaim(
        "Gold displayed floor comparisons",
        "In the full pool, max-span scores 0.703 on stale-state",
        orderings=(
            # "max-span displays 0.805 against the displayed 0.350 floor", "degree displays 0.394".
            BoardOrdering(_matched("max-span (control)", "stale-state"),
                          _matched("random (matched)", "stale-state")),
            BoardOrdering(_matched("degree", "stale-state"),
                          _matched("random (matched)", "stale-state")),
            # "For dropped grounding, position displays 0.321 against 0.277."
            BoardOrdering(_matched("position", "dropped-grounding"),
                          _matched("random (matched)", "dropped-grounding")),
            # "PyGOD displays ... 0.404 overall", above the max-span cell beside it.
            BoardOrdering(_matched("pygod (graph AD)", "overall"),
                          _matched("max-span (control)", "overall")),
        ),
        differences=(
            RecordedDifference("gold.maxspan.stale.floor", A_ABOVE_B),
            # has-dep sits below the dropped-grounding analytic floor, so the record's ``b`` is the
            # higher arm. A registration that said A_ABOVE_B here would fail on the sign.
            RecordedDifference("gold.hasdep.dropped.floor", B_ABOVE_A),
            RecordedDifference("gold.pygod.maxspan.matched"),
        ),
    ),
    BoardClaim(
        "Gold measured localization mechanism",
        "Stale-state max-span scores 0.703 against its 0.029 analytic floor",
        orderings=(
            # The graph detector is the highest dropped-grounding cell; degree is the runner-up.
            BoardOrdering(Cell(_GOLD_BREAKDOWN, "pygod (graph AD)", "dropped-grounding", part=0),
                          Cell(_GOLD_BREAKDOWN, "degree", "dropped-grounding", part=0)),
        ),
        differences=(
            RecordedDifference("gold.maxspan.stale.floor", A_ABOVE_B),
            RecordedDifference("gold.hasdep.dropped.floor", B_ABOVE_A),
        ),
    ),
    BoardClaim(
        "Gold leakage controls",
        "Leakage check, two levels",
        orderings=(
            # "position, degree, and `has-dep` display 0.000, 0.045, and 0.078 overall against the
            # 0.032 random floor", with the latter two named as above it.
            BoardOrdering(Cell(_GOLD_LOC, "has-dep (control)", "top1"),
                          Cell(_GOLD_LOC, "random", "top1")),
            BoardOrdering(Cell(_GOLD_LOC, "degree", "top1"),
                          Cell(_GOLD_LOC, "random", "top1")),
            BoardOrdering(Cell(_GOLD_LOC, "random", "top1"),
                          Cell(_GOLD_LOC, "position", "top1")),
            # "`has-dep` equals the matched floor at 0.350, degree scores 0.394, and the
            # dependency-span detector scores 0.805".
            BoardOrdering(_matched("degree", "stale-state"),
                          _matched("has-dep (control)", "stale-state")),
            BoardOrdering(_matched("max-span (control)", "stale-state"),
                          _matched("has-dep (control)", "stale-state")),
        ),
        # The stale-state and dropped-grounding analytic-floor differences moved to "One measured
        # localization mechanism". This paragraph now reports overall and eligible-pool controls.
    ),
    BoardClaim(
        "Gold cause attribution",
        "the two injections leave opposite traces",
        orderings=(
            BoardOrdering(Cell(_GOLD_ATTRIB, "max-span (higher=stale)", "roc-auc"),
                          Cell(_GOLD_ATTRIB, "random", "roc-auc")),
            BoardOrdering(Cell(_GOLD_ATTRIB, "edge-count (higher=stale)", "roc-auc"),
                          Cell(_GOLD_ATTRIB, "random", "roc-auc")),
        ),
        differences=(
            RecordedDifference("gold.attribution.max-span (higher=stale)", A_ABOVE_B),
            RecordedDifference("gold.attribution.edge-count (higher=stale)", A_ABOVE_B),
        ),
    ),
    BoardClaim(
        "LIVE early warning",
        "dependency-structure block scores 0.74",
        orderings=(
            # The dependency block's 0.74 is above the flat baseline's 0.63 at the 25% prefix.
            BoardOrdering(_prefix(_LIVE_PREFIX_SWE, "auditable (size+deps)", "25"),
                          _prefix(_LIVE_PREFIX_SWE, "size (flat)", "25")),
        ),
        differences=(RecordedDifference("live.swe25.auditable.vs.size", A_ABOVE_B),)
        + _TAU_BELOW_BAR,
    ),
    BoardClaim(
        "LIVE prefix figure",
        "Drawn by `figure-src/board_live_prefix.py`",
        orderings=(
            # "the lowest curve at the shortest prefix on SWE-Gym", against the reference it is
            # below there.
            BoardOrdering(_prefix(_LIVE_PREFIX_SWE, "random", "25"),
                          _prefix(_LIVE_PREFIX_SWE, "dep-span (online)", "25")),
            # "on tau-bench it sits just above the random reference there".
            BoardOrdering(_prefix(_LIVE_PREFIX_TAU, "dep-span (online)", "25"),
                          _prefix(_LIVE_PREFIX_TAU, "random", "25")),
        ),
    ),
    BoardClaim(
        "PRE pooled board",
        "combined OWASP/CWE scanner has the highest displayed rule-based F1",
        orderings=(
            # "the highest displayed rule-based F1 (0.654 ...)", against the next rule down.
            BoardOrdering(Cell(_PRE_MULTI, "owasp_asi_combined", "f1"),
                          Cell(_PRE_MULTI, "owasp_excess_functionality", "f1")),
            # "Three rules display a higher precision cell than the judge's 0.594".
            BoardOrdering(Cell(_PRE_MULTI, "owasp_privilege_escalation", "precision"),
                          Cell(_PRE_MULTI, "llm_judge_needed(llama-3.3-70b)", "precision")),
            BoardOrdering(Cell(_PRE_MULTI, "sensitive_access", "precision"),
                          Cell(_PRE_MULTI, "llm_judge_needed(llama-3.3-70b)", "precision")),
            BoardOrdering(Cell(_PRE_MULTI, "unrequested_high_impact", "precision"),
                          Cell(_PRE_MULTI, "llm_judge_needed(llama-3.3-70b)", "precision")),
        ),
        differences=(
            # "0.654 ... against 0.695 F1 for the held-out LLM judge", whose record carries the
            # judge as ``a``.
            RecordedDifference("pre.combined.vs.heldout_judge", A_ABOVE_B),
            # These three test a rule against the pooled base rate. The paragraph prints the three
            # precision cells and says outright that they are not a ranking against the judge, so
            # they are bound to the record and ordered by nothing.
            RecordedDifference("pre.precision.owasp_privilege_escalation.vs.base_rate"),
            RecordedDifference("pre.precision.unrequested_high_impact.vs.base_rate"),
            RecordedDifference("pre.precision.sensitive_access.vs.base_rate"),
        ),
    ),
    BoardClaim(
        "PRE per-source figure",
        "Drawn by `figure-src/board_pre_source.py`",
        orderings=(
            # "The floor itself moves by nearly a factor of five across the six sources": the two
            # ends of that move.
            BoardOrdering(Cell(_PRE_SOURCE, "flag_all", "mcp"),
                          Cell(_PRE_SOURCE, "flag_all", "n8n")),
        ),
        # Each row pairs one source's best method with that source's floor. The caption states no
        # ordering for those pairs, and it is right not to: on sweagent the best method's F1 is
        # 0.00006 below the floor, so a blanket "best beats floor" registration would be false.
        differences=tuple(
            RecordedDifference(f"pre.source.{source}.best.vs.flag_all")
            for source in ("crewai", "n8n", "mcp", "injecagent", "sweagent", "synthetic")
        ),
    ),
    BoardClaim(
        "PRE declaration-order shortcut",
        "The injecagent column carries a shortcut",
        orderings=(
            # "above the 0.990 held-out judge and above every other method in the table": the judge
            # is the highest of those, so this is the cell the statement rests on.
            BoardOrdering(Cell(_PRE_SOURCE, "llm_judge_needed(llama-3.3-70b)", "injecagent"),
                          Cell(_PRE_SOURCE, "owasp_asi_combined", "injecagent")),
        ),
    ),
    BoardClaim(
        "PRE n8n displayed range",
        "Using a different scoring judge avoids direct label reuse",
        orderings=(
            # "the non-oracle methods in the table score from 0.095 to 0.411 F1": the two ends.
            BoardOrdering(Cell(_PRE_SOURCE, "owasp_asi_combined", "n8n"),
                          Cell(_PRE_SOURCE, "flag_risky_perms", "n8n")),
        ),
    ),
)


# ---------------------------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------------------------


def _resolve_block(blocks: Mapping[str, GoldenBlock], counts: Counter,
                   key: str) -> tuple[GoldenBlock | None, str | None]:
    block = blocks.get(key)
    if block is None:
        if counts.get(key, 0) > 1:
            return None, (f"golden block {key!r} appears {counts[key]} times; name the occurrence "
                          f"({key}#1 .. {key}#{counts[key]})")
        return None, f"no golden block named {key!r}"
    if block.error:
        return None, f"golden block {key!r} did not parse: {block.error}"
    return block, None


def _golden_value(blocks: Mapping[str, GoldenBlock], counts: Counter, source: Source,
                  golden_row: str) -> tuple[str | None, str | None]:
    """Return the printed golden value for one cell, or the reason there is none."""
    block, why = _resolve_block(blocks, counts, source.block)
    if block is None:
        return None, why
    label = norm_label(golden_row)
    if label not in block.rows:
        return None, f"golden block {source.block!r} has no row {label!r}"
    column = norm_label(source.column)
    if column not in block.columns:
        return None, (f"golden block {source.block!r} has no column {column!r} "
                      f"(columns: {', '.join(block.columns)})")
    value = block.rows[label][block.columns.index(column) - 1]
    if source.part is not None:
        parts = value.split("/")
        if source.part >= len(parts):
            return None, (f"golden cell {value!r} has {len(parts)} part(s); the spec asks for "
                          f"part {source.part}")
        value = parts[source.part]
    return value, None


def _check_table(spec: TableSpec, table: ReadmeTable, blocks: Mapping[str, GoldenBlock],
                 counts: Counter) -> tuple[list[Problem], int]:
    """Compare one README table against the golden. Returns (problems, cells compared)."""
    problems: list[Problem] = []
    compared = 0
    columns = table.header[1:]
    for column in columns:
        if column not in spec.columns:
            problems.append(Problem("spec-gap", f"{spec.name}: column {column!r} is not mapped to a "
                                                "golden column", line=table.line))
    for label in sorted(spec.separators & set(spec.rows)):
        problems.append(Problem("spec-gap",
                                f"{spec.name}: row {label!r} is declared both as a section divider "
                                "and as a row with golden values; it cannot be both",
                                line=table.line))
    if problems:
        return problems, compared

    seen: dict[str, tuple[int, str, tuple[str, ...]]] = {}
    for line, raw_label, cells in table.rows:
        label = norm_label(raw_label)
        if label in seen:
            problems.append(Problem("duplicate-row",
                                    f"{spec.name}: row {label!r} appears twice", line=line))
            continue
        seen[label] = (line, raw_label, cells)

    for label in sorted(spec.separators):
        if label not in seen:
            problems.append(Problem("row-missing",
                                    f"{spec.name}: separator row {label!r} is declared in the spec "
                                    "but not in the README table", line=table.line))
            continue
        line, _, cells = seen.pop(label)
        filled = [c for c in cells if not is_placeholder(c)]
        if filled:
            problems.append(Problem("separator-has-values",
                                    f"{spec.name}: row {label!r} is declared as a section divider "
                                    f"but carries values {filled}", line=line))

    for label in sorted(set(spec.rows) - set(seen)):
        problems.append(Problem("row-missing",
                                f"{spec.name}: the spec declares row {label!r} and the README table "
                                "does not have it", line=table.line))
    for label in sorted(set(seen) - set(spec.rows)):
        line = seen[label][0]
        if all(is_placeholder(c) for c in seen[label][2]):
            problems.append(Problem("undeclared-separator",
                                    f"{spec.name}: row {label!r} has no values; if it is a section "
                                    "divider add it to this spec's separators", line=line))
        else:
            problems.append(Problem("row-undeclared",
                                    f"{spec.name}: row {label!r} is not declared in the spec, so "
                                    "there is no golden row to compare it against", line=line))

    for label in sorted(set(spec.rows) & set(seen)):
        row = spec.rows[label]
        line, _, cells = seen[label]
        for column, cell in zip(columns, cells):
            source = row.sources.get(column, spec.columns[column])
            value, why = _golden_value(blocks, counts, source, row.golden)
            if column in row.absent:
                if value is not None:
                    problems.append(Problem("stale-spec",
                                            f"{spec.name}: the spec says {label!r} has no golden "
                                            f"{column!r}, but the board reports {value!r}",
                                            line=line))
                elif not is_placeholder(cell):
                    problems.append(Problem("unsupported-value",
                                            f"{spec.name}: {label!r} {column!r} reads "
                                            f"{strip_markup(cell)!r} and the board has no such "
                                            f"cell ({why})", line=line))
                continue
            if value is None:
                problems.append(Problem("unresolved", f"{spec.name}: {label!r} {column!r}: {why}",
                                        line=line))
                continue
            compared += 1
            printed = strip_markup(cell)
            # A placeholder where the board has a value is a mismatch like any other; reporting it
            # in the same shape keeps the "README says X, board says Y" line uniform.
            if printed != value:
                problems.append(Problem("mismatch",
                                        f"{spec.name}: {label!r} {column!r} README {printed!r} != "
                                        f"golden {value!r}", line=line))
    return problems, compared


@dataclass
class Result:
    """What one README check found, and how much of the README it actually looked at."""

    problems: list[Problem]
    tables_seen: int = 0        # pipe tables parsed out of the README
    tables_numeric: int = 0     # of those, the ones carrying at least one number
    tables_claimed: int = 0     # of those, the ones a spec or an exemption claimed
    cells_compared: int = 0     # README cells compared against a golden cell
    prose_numbers_seen: int = 0
    prose_numbers_board: int = 0
    prose_numbers_allowed: int = 0
    comparative_paragraphs: int = 0
    comparative_claimed: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems


def check_readme_detailed(readme_text: str, golden_text: str,
                          specs: Sequence[TableSpec] | None = None,
                          exemptions: Sequence[Exemption] | None = None,
                          number_allowances: Sequence[ProseNumberAllowance] | None = None,
                          board_claims: Sequence[BoardClaim] | None = None,
                          statistical_data: Mapping[str, Any] | None = None) -> Result:
    """Check README tables, prose numerals, and comparative claims against committed evidence.

    Returns one :class:`Problem` per disagreement, plus the counts that say how much was looked at.
    An empty problem list means every numeric table was claimed by a spec, every declared row was
    found, and every mapped cell printed exactly what the board printed. The counts are reported
    because "no problems" and "nothing checked" are the same output otherwise.

    The registries are read at call time rather than bound as defaults, so a test can point the
    shipped entry points at fixture registries and exercise the same path CI runs.
    """
    specs = TABLE_SPECS if specs is None else specs
    exemptions = NON_BOARD_TABLES if exemptions is None else exemptions
    number_allowances = (PROSE_NUMBER_ALLOWLIST if number_allowances is None
                         else number_allowances)
    board_claims = BOARD_CLAIMS if board_claims is None else board_claims
    if statistical_data is None:
        statistical_data = json.loads(STATISTICAL_RESULTS.read_text(encoding="utf-8"))
    blocks, counts = parse_golden(golden_text)
    tables, problems = parse_readme(readme_text)
    result = Result(problems=problems, tables_seen=len(tables),
                    tables_numeric=sum(1 for t in tables if t.numeric))

    claimed: dict[int, str] = {}
    for spec in specs:
        matched = [t for t in tables if spec.matches(t)]
        if not matched:
            problems.append(Problem("spec-unmatched",
                                    f"{spec.name}: no README table has header "
                                    f"{list(spec.header)} under a heading containing "
                                    f"{spec.heading_contains!r}"
                                    + (f" after a paragraph containing "
                                       f"{spec.lead_in_contains!r}" if spec.lead_in_contains
                                       else "")))
            continue
        if len(matched) > 1:
            problems.append(Problem("spec-ambiguous",
                                    f"{spec.name}: matches {len(matched)} README tables (lines "
                                    f"{', '.join(str(t.line) for t in matched)}); tighten its "
                                    "heading or lead-in"))
            continue
        table = matched[0]
        if id(table) in claimed:
            problems.append(Problem("double-claim",
                                    f"the table at this line is claimed by both "
                                    f"{claimed[id(table)]!r} and {spec.name!r}", line=table.line))
            continue
        claimed[id(table)] = spec.name
        found, compared = _check_table(spec, table, blocks, counts)
        problems.extend(found)
        result.cells_compared += compared

    for exemption in exemptions:
        matched = [t for t in tables if exemption.matches(t)]
        if len(matched) != 1:
            problems.append(Problem("exemption-unmatched",
                                    f"{exemption.name}: the non-board exemption matches "
                                    f"{len(matched)} README tables; it should match exactly one"))
            continue
        table = matched[0]
        if id(table) in claimed:
            problems.append(Problem("double-claim",
                                    f"the table at this line is claimed by both "
                                    f"{claimed[id(table)]!r} and the exemption "
                                    f"{exemption.name!r}", line=table.line))
            continue
        claimed[id(table)] = f"exempt: {exemption.name}"

    for table in tables:
        if id(table) not in claimed and table.numeric:
            problems.append(Problem("unclaimed-table",
                                    "this table carries numbers and no spec in TABLE_SPECS claims "
                                    f"it (header {list(table.header)}, heading "
                                    f"{table.heading!r}). Map it to a golden block, or record it "
                                    "in NON_BOARD_TABLES with a reason.", line=table.line))
    result.tables_claimed = len(claimed)

    number_problems, seen, board_backed, allowed = check_prose_numbers(
        readme_text, blocks, number_allowances)
    problems.extend(number_problems)
    result.prose_numbers_seen = seen
    result.prose_numbers_board = board_backed
    result.prose_numbers_allowed = allowed

    claim_problems, comparative_seen, comparative_claimed = check_board_claims(
        readme_text, blocks, counts, statistical_data, board_claims)
    problems.extend(claim_problems)
    result.comparative_paragraphs = comparative_seen
    result.comparative_claimed = comparative_claimed

    problems.extend(check_adjudicating_numbers(readme_text))
    return result


def check_readme(readme_text: str, golden_text: str,
                 specs: Sequence[TableSpec] | None = None,
                 exemptions: Sequence[Exemption] | None = None,
                 number_allowances: Sequence[ProseNumberAllowance] | None = None,
                 board_claims: Sequence[BoardClaim] | None = None,
                 statistical_data: Mapping[str, Any] | None = None) -> list[Problem]:
    """The problem list alone, for callers that do not need the counts."""
    return check_readme_detailed(readme_text, golden_text, specs, exemptions, number_allowances,
                                 board_claims, statistical_data).problems


def readme_report(result: Result) -> str:
    if result.ok:
        return (f"README matches the golden board: {result.cells_compared} cell(s) across "
                f"{result.tables_claimed} table(s); {result.tables_numeric} numeric table(s) in "
                f"README.md, all claimed; {result.prose_numbers_seen} prose numeral(s), "
                f"{result.prose_numbers_board} board-backed and "
                f"{result.prose_numbers_allowed} allowed; "
                f"{result.comparative_paragraphs} comparative paragraph(s), all bound to "
                "recorded measurements")
    lines = [f"README DRIFT: {len(result.problems)} problem(s)", ""]
    lines += [p.render() for p in sorted(result.problems, key=lambda p: (p.line or 0, p.kind))]
    lines += ["",
              f"({result.cells_compared} cell(s) compared, across {result.tables_claimed} of "
              f"{result.tables_numeric} numeric table(s). A low count next to a long problem list "
              "means whole tables went unchecked.)",
              f"Prose numerals: {result.prose_numbers_seen} seen, "
              f"{result.prose_numbers_board} board-backed at printed precision, "
              f"{result.prose_numbers_allowed} allowed with reasons.",
              f"Comparative paragraphs: {result.comparative_paragraphs} seen, "
              f"{result.comparative_claimed} claimed by a BOARD_CLAIMS content anchor."]
    return "\n".join(lines)


def run_readme_check(readme_path: Path, golden_path: Path,
                     statistical_path: Path = STATISTICAL_RESULTS) -> tuple[int, str]:
    """Check one README against the golden and statistical results. Return code and report."""
    if not readme_path.exists():
        return 1, f"no README at {readme_path}"
    if not golden_path.exists():
        return 1, f"no golden at {golden_path}; create it with --update"
    if not statistical_path.exists():
        return 1, f"no statistical results at {statistical_path}"
    try:
        statistical_data = json.loads(statistical_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return 1, f"invalid statistical results at {statistical_path}: {exc}"
    if not isinstance(statistical_data, Mapping):
        return 1, f"invalid statistical results at {statistical_path}: top level is not an object"
    result = check_readme_detailed(
        readme_path.read_text(encoding="utf-8"), golden_path.read_text(encoding="utf-8"),
        statistical_data=statistical_data)
    return (0 if result.ok else 1), readme_report(result)


def main() -> int:
    # Raw, because the docstring's Usage block is a list of commands and argparse's default
    # reflow runs them into one paragraph.
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update", action="store_true",
                        help="regenerate the golden instead of comparing against it")
    parser.add_argument("--produced", type=Path,
                        help="compare a saved board instead of running one; the failure artifact "
                             "from CI can be checked this way without spending nine minutes")
    parser.add_argument("--readme-only", action="store_true",
                        help="check README.md against the committed golden and skip the board run")
    parser.add_argument("--board-only", action="store_true",
                        help="run or compare only the scoring board; the fast CI job checks README")
    parser.add_argument("--readme", type=Path, default=README,
                        help="README to check (default: the one in this checkout)")
    parser.add_argument("--golden", type=Path, default=GOLDEN,
                        help="golden board to compare against (default: tests/golden/board.txt)")
    parser.add_argument("--claims", type=Path, default=STATISTICAL_RESULTS,
                        help="statistical claim results (default: "
                             "tools/statistical_tests_results.json)")
    args = parser.parse_args()

    if args.readme_only and args.board_only:
        parser.error("--readme-only and --board-only are mutually exclusive")
    if args.update and args.board_only:
        parser.error("--update must also report whether README copies became stale")

    golden_path = args.golden

    if args.readme_only:
        code, report = run_readme_check(args.readme, golden_path, args.claims)
        print(report)
        return code

    produced = args.produced.read_text(encoding="utf-8") if args.produced else produce()

    board_code = 0
    if args.update:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        previous = golden_path.read_text(encoding="utf-8") if golden_path.exists() else ""
        golden_path.write_text(produced, encoding="utf-8")
        changed = sum(1 for line in difflib.unified_diff(normalize(previous), normalize(produced))
                      if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))
        print(f"wrote {golden_path} ({len(normalize(produced))} lines, {changed} changed)")
    elif not golden_path.exists():
        print(f"no golden at {golden_path}; create it with --update", file=sys.stderr)
        return 1
    else:
        want = normalize(golden_path.read_text(encoding="utf-8"))
        got = normalize(produced)
        want, got, folded = reconcile_neural_rows(want, got)
        if want == got:
            scored = sum(1 for line in got if line.startswith("["))
            tolerated = ("" if not folded else
                         f"; {folded} torch row(s) within {NEURAL_TOLERANCE} of the golden")
            print(f"board matches the golden: {len(got)} lines, {scored} scored blocks{tolerated}")
        else:
            diff = list(difflib.unified_diff(want, got, fromfile="golden", tofile="run.py",
                                             lineterm=""))
            changed = sum(1 for line in diff if line.startswith(("+", "-"))
                          and not line.startswith(("+++", "---")))
            print(f"BOARD DRIFT: {changed} changed line(s)\n")
            for line in diff[:80]:
                print(line)
            if len(diff) > 80:
                print(f"... {len(diff) - 80} more diff lines")
            print("\nIf this change is intended, regenerate with: "
                  "python tools/check_board.py --update\n"
                  "and put the diff in the commit message. Check the corpus revisions in the board "
                  "header first: a moved corpus produces drift that is not a code change.")
            board_code = 1

    if args.board_only:
        return board_code

    readme_code, report = run_readme_check(args.readme, golden_path, args.claims)
    print()
    print(report)
    if readme_code and args.update:
        print("\nThe golden was written. The README still disagrees with it, so this run exits "
              "non-zero: a regenerated board and its README copies belong in the same commit.")
    return max(board_code, readme_code)


if __name__ == "__main__":
    sys.exit(main())
