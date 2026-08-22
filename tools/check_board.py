"""Run the scoring board and compare it against the committed golden output.

Every number the paper reports comes from ``run.py``. Until now nothing checked that ``run.py`` still
produces those numbers: ``ci_smoke.py`` covers the PRE board only, and POST, LIVE, and Gold are in
its skip list because they need the GRADE bridge and three corpus downloads. A refactor could move a
published cell and no test would notice. That is the gap this closes.

The comparison is exact, on the whole printed board. A label rename fails it, which is intended: a
board line changing is either a result moving or a deliberate edit, and both should be looked at by
a person. ``--update`` regenerates the golden, and the diff it produces belongs in the commit
message.

The board is only half of what ships. ``README.md`` carries hand-typed copies of the same tables,
and those copies drift: a PyGOD localization row sat at three times its scored value, and the PRE
judge row was wrong on recall and F1, while the board itself was correct the whole time.
CONTRIBUTING.md already says "Do not type numbers directly into the README", and nothing enforced
it. The second half of this checker does: it extracts every numeric table from ``README.md`` and
compares each cell, at the printed precision, against the committed golden board.

Two rules govern that second half.

*No tolerance.* A cell matches when its printed text matches. ``0.703`` and ``0.707`` are a failure,
and so are ``0.71`` and ``0.710``. A tolerance would hide exactly the small movements the check
exists to catch.

*Fail closed.* Every numeric README table must be claimed by exactly one entry in ``TABLE_SPECS``
(or, deliberately, by ``NON_BOARD_TABLES``). Every row of a claimed table must be declared with the
golden row it copies. A table that cannot be parsed, a table no spec claims, a spec that claims no
table or two, an undeclared row, a row declared but absent, a golden block or column a spec names
but the golden does not have: each is a failure with a message, never a silent skip. A checker that
quietly matches nothing is worse than no checker, because it reads as coverage. So a table added to
the README in the future fails this check the moment it contains one numeric cell, and keeps failing
until someone either maps it to a golden block in ``TABLE_SPECS`` or records in ``NON_BOARD_TABLES``,
with a reason, that it is not a board table.

Usage::

    python tools/check_board.py               # run the board, compare both halves, exit 1 on any drift
    python tools/check_board.py --readme-only # compare README against the committed golden (seconds)
    python tools/check_board.py --produced F  # compare a saved board, plus the README
    python tools/check_board.py --update      # regenerate the golden from the current code

Every mode checks the README, ``--update`` included: regenerating the golden is exactly when the
README goes stale, so that run reports the drift and exits non-zero after writing the new golden.
The CI board job runs this file with no arguments, so it now gates both halves without a workflow
change.

The board takes roughly nine minutes and needs the GRADE checkout bridge, the torch stack for the
PyGOD rows, and about 320 MB of corpora at their pinned revisions. Those revisions are printed in
the board's own header, so a golden mismatch caused by a moved corpus is self-diagnosing. The README
half needs none of that: it reads two files.
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "golden" / "board.txt"
RUNNER = ROOT / "run.py"
README = ROOT / "README.md"


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
# A second, independent CI run produced the same two values, so this is a stable difference between
# platforms rather than nondeterminism between runs. The cause is underneath torch: a different BLAS,
# a different reduction order, a different build.
#
# The tolerance is 0.005, smaller than the seed variance the paper already reports for the larger of
# the two rows, 0.824 +/- 0.007 for g-safeguard over five joint split and initialization seeds. A
# difference this check now tolerates is one the paper already tells a reader to expect, and a
# difference large enough to move a claim still fails.
#
# Everything outside these rows stays byte-exact. Widening the tolerance to the whole board would
# leave it unable to see a real regression in the rule methods, which are exact by construction and
# are most of the board.
TORCH_ROW_PREFIXES = ("guardian (", "g-safeguard (", "pygod")
NEURAL_TOLERANCE = 0.005

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
    return all(abs(float(x) - float(y)) <= NEURAL_TOLERANCE for x, y in zip(a, b))


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

    @property
    def ok(self) -> bool:
        return not self.problems


def check_readme_detailed(readme_text: str, golden_text: str,
                          specs: Sequence[TableSpec] | None = None,
                          exemptions: Sequence[Exemption] | None = None) -> Result:
    """Compare every numeric table in the README against the golden board.

    Returns one :class:`Problem` per disagreement, plus the counts that say how much was looked at.
    An empty problem list means every numeric table was claimed by a spec, every declared row was
    found, and every mapped cell printed exactly what the board printed. The counts are reported
    because "no problems" and "nothing checked" are the same output otherwise.

    The two registries are read at call time rather than bound as defaults, so a test can point the
    shipped entry points at a fixture registry and exercise the same path CI runs.
    """
    specs = TABLE_SPECS if specs is None else specs
    exemptions = NON_BOARD_TABLES if exemptions is None else exemptions
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
    return result


def check_readme(readme_text: str, golden_text: str,
                 specs: Sequence[TableSpec] | None = None,
                 exemptions: Sequence[Exemption] | None = None) -> list[Problem]:
    """The problem list alone, for callers that do not need the counts."""
    return check_readme_detailed(readme_text, golden_text, specs, exemptions).problems


def readme_report(result: Result) -> str:
    if result.ok:
        return (f"README matches the golden board: {result.cells_compared} cell(s) across "
                f"{result.tables_claimed} table(s); {result.tables_numeric} numeric table(s) in "
                f"README.md, all claimed")
    lines = [f"README DRIFT: {len(result.problems)} problem(s)", ""]
    lines += [p.render() for p in sorted(result.problems, key=lambda p: (p.line or 0, p.kind))]
    lines += ["",
              f"({result.cells_compared} cell(s) compared, across {result.tables_claimed} of "
              f"{result.tables_numeric} numeric table(s). A low count next to a long problem list "
              "means whole tables went unchecked.)",
              "Every number in README.md must be a copy of a cell the board printed. Regenerate the "
              "board with `python run.py`, copy the values, and do not round them: this check "
              "compares the printed text, with no tolerance."]
    return "\n".join(lines)


def run_readme_check(readme_path: Path, golden_path: Path) -> tuple[int, str]:
    """Check one README against one golden file. Returns (exit code, report)."""
    if not readme_path.exists():
        return 1, f"no README at {readme_path}"
    if not golden_path.exists():
        return 1, f"no golden at {golden_path}; create it with --update"
    result = check_readme_detailed(readme_path.read_text(encoding="utf-8"),
                                   golden_path.read_text(encoding="utf-8"))
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
    parser.add_argument("--readme", type=Path, default=README,
                        help="README to check (default: the one in this checkout)")
    parser.add_argument("--golden", type=Path, default=GOLDEN,
                        help="golden board to compare against (default: tests/golden/board.txt)")
    args = parser.parse_args()

    golden_path = args.golden

    if args.readme_only:
        code, report = run_readme_check(args.readme, golden_path)
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

    readme_code, report = run_readme_check(args.readme, golden_path)
    print()
    print(report)
    if readme_code and args.update:
        print("\nThe golden was written. The README still disagrees with it, so this run exits "
              "non-zero: a regenerated board and its README copies belong in the same commit.")
    return max(board_code, readme_code)


if __name__ == "__main__":
    sys.exit(main())
