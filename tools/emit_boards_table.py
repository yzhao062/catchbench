"""Emit the paper's board-inventory table from the shipped board, so the two cannot drift.

The paper prints eleven tables and a reader has no single place to see what the benchmark actually
scores. This generates that place: one row per scored block, with the corpus it runs on, the metric,
the trivial floor, the spread of the entrants, and the verdict the registered tests support.

The Separates column is the reason this is generated rather than typed. A hand-written summary
table is where a paper's numbers rot first, because nothing recomputes it: the README carried a
PyGOD row at three times its scored value for weeks, and only a checker caught it. That column also
has to stay complete, not merely current, so ``check_partition`` requires the boards to partition
the declared contrasts exactly; the fractions therefore sum to the total the abstract quotes. Everything here is read from
``tests/golden/board.txt`` and ``tools/statistical_tests_results.json``. The only hand-authored
content is the mapping from a board to the claim ids that speak for it, and even that is validated:
an unknown claim id, a missing board, or a corpus line that stops matching is a hard failure.

Usage::

    python tools/emit_boards_table.py            # print the LaTeX table body
    python tools/emit_boards_table.py --check    # exit 1 if the paper is stale, printing the delta

``--check`` needs ``--paper <dir>`` or the ``CATCHBENCH_PAPER_DIR`` environment variable.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOARD = ROOT / "tests" / "golden" / "board.txt"
RESULTS = ROOT / "tools" / "statistical_tests_results.json"

_RESULTS_SECTION = "05_results.tex"
_BENCHMARK = "03_benchmark.tex"

# A board's identity is its printed header. Matching on that rather than on position means a
# reordered runner fails loudly instead of silently pairing a corpus with the wrong board.
#
# floor      the entrant that answers "what does doing the trivial thing score", by its board name
# skip       rows that are not entrants: the floor itself, and any oracle that is an identity check
# claims     claim ids from statistical_tests_results.json, or ("family", <id>) for a whole family
_BOARDS = (
    dict(header="[PRE] pre_over_privilege :: multi",
         state="PRE", name="Over-privilege", metric="f1", metric_label="F1",
         corpus=r"^PRE over_privilege: (\d+) configs across (\d+) corpora",
         corpus_name="{1} sources", size="{0}", unit="configs",
         floor="flag_all", skip=("flag_none", "oracle_privilege_diff"),
         coverage=(("pre_combined_judge_f1", None), ("pre_source_best_flag_all_f1", None),
                   ("pre_rules_flag_all_f1", None), ("pre_narrow_precision", None))),
    dict(header="[LIVE] live_streaming :: swegym",
         state="LIVE", name="Early warning", metric="prefix_auc", metric_label="ROC-AUC",
         corpus=r"^swegym: (\d+) runs \(>=4 steps",
         corpus_name="SWE-Gym", size="{0}", unit="runs",
         floor="random", skip=(),
         coverage=(("live_swegym_25_auc", None),
                   ("live_swegym_auditable_ecod_later_auc", None))),
    dict(header="[LIVE] live_streaming :: tau",
         state="LIVE", name="Early warning", metric="prefix_auc", metric_label="ROC-AUC",
         corpus=r"^tau: (\d+) runs \(>=4 steps",
         corpus_name="tau-bench", size="{0}", unit="runs",
         floor="random", skip=(),
         coverage=(("live_tau_threshold_auc", None), ("live_tau_auditable_ecod_auc", None))),
    dict(header="[LIVE] live_stale_state :: swegym-gold",
         state="LIVE", name="Online stale$^{\\dagger}$", metric="tpr@5fpr",
         metric_label="TPR@5\\%FPR",
         corpus=r"^swegym-gold: (\d+) stale-state injections",
         corpus_name="Gold", size="{0}", unit="injections",
         floor="random", skip=(), gold=True,
         coverage=()),
    dict(header="[POST] post_localization :: whoandwhen",
         state="POST", name="Localization", metric="top1", metric_label="Top-1",
         corpus=r"^Who&When: (\d+) failed runs, (\d+) steps",
         corpus_name="Who\\&When", size="{0}", unit="runs",
         floor="random", skip=(),
         coverage=(("localization_gpt55_no_llm_top1", None), ("localization_band_top1", None),
                   ("localization_exec_position", None), ("localization_protocol_top1", None),
                   ("localization_small_position_top1", None))),
    dict(header="[POST] post_detection :: swegym",
         state="POST", name="Detection", metric="roc_auc", metric_label="ROC-AUC",
         corpus=r"^swegym: (\d+) runs \((\d+) failed",
         corpus_name="SWE-Gym", size="{0}", unit="runs",
         floor="random", skip=(),
         coverage=(("post_detection_auc", "det.swe."),)),
    dict(header="[POST] post_detection :: tau",
         state="POST", name="Detection", metric="roc_auc", metric_label="ROC-AUC",
         corpus=r"^tau: (\d+) runs \((\d+) failed",
         corpus_name="tau-bench", size="{0}", unit="runs",
         floor="random", skip=(),
         coverage=(("post_detection_auc", "det.tau."),)),
    dict(header="[POST] gold_localization :: swegym-gold",
         state="POST", name="Localization$^{\\dagger}$", metric="top1", metric_label="Top-1",
         corpus=r"^swegym-gold: (\d+) clean SWE-Gym runs",
         corpus_name="Gold", size="{0}", unit="runs",
         floor="random", skip=(), gold=True,
         coverage=(("gold_localization_top1", None),)),
    dict(header="[POST] gold_attribution :: swegym-gold",
         state="POST", name="Cause attribution$^{\\dagger}$", metric="roc_auc",
         metric_label="ROC-AUC",
         corpus=r"^swegym-gold: (\d+) runs affording both faults",
         corpus_name="Gold", size="{0}", unit="paired runs",
         floor="random", skip=(), gold=True,
         coverage=(("gold_attribution_auc", None),)),
)

# The six-word gloss each state's banded group row carries in the paper. Hand-written scaffolding
# there (a \multicolumn row has no "&", so table_body drops it and --check does not compare it),
# kept here so the two cannot describe the states differently.
_STATE_GLOSS = {
    "PRE": "before the run, plan and harness only",
    "LIVE": "during the run, a growing prefix",
    "POST": "after the run, the complete trace",
}

_SEPARATING = frozenset({"separates_as_stated", "separates_opposite_to_statement"})

_TABLE_BEGIN = re.compile(r"(?m)^[ \t]*\\begin\{tabular\}\{[^\r\n]*\}[ \t]*$")
_TABLE_END = re.compile(r"(?m)^[ \t]*\\end\{tabular\}[ \t]*$")
_SCAFFOLDING = frozenset({r"\toprule", r"\midrule", r"\bottomrule"})


# --------------------------------------------------------------------------- reading the board

def read_board(text: str) -> tuple[list[str], dict[str, dict[str, dict[str, float]]]]:
    """Split board.txt into its preamble description lines and one table per block header."""
    preamble: list[str] = []
    blocks: dict[str, dict[str, dict[str, float]]] = {}
    current: str | None = None
    columns: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("[PRE]") or line.startswith("[POST]") or line.startswith("[LIVE]"):
            current = line.strip()
            blocks[current] = {}
            columns = []
            continue
        if current is None:
            if line.strip():
                preamble.append(line.strip())
            continue
        if not line.startswith("  "):
            current = None
            continue
        cells = line.split()
        if not columns:
            columns = cells[1:]
            continue
        # A method name can contain spaces, so the values are taken from the right.
        values = cells[-len(columns):]
        try:
            numbers = [float(v) for v in values]
        except ValueError:
            continue
        name = " ".join(cells[:len(cells) - len(columns)])
        blocks[current][name] = dict(zip(columns, numbers))
    return preamble, blocks


def _thousands(text: str) -> str:
    """LaTeX thousands separator, so a right-aligned count still aligns inside a tinted row."""
    return "{:,}".format(int(text)).replace(",", "{,}")


def corpus_cells(preamble: list[str], spec: dict) -> tuple[str, str, str]:
    """The corpus name, its size as a right-alignable numeral, and the unit noun."""
    pattern = re.compile(spec["corpus"])
    hits = [m for m in (pattern.search(line) for line in preamble) if m]
    if len(hits) != 1:
        raise SystemExit("board.txt: corpus pattern %r matched %d description lines for %r"
                         % (spec["corpus"], len(hits), spec["header"]))
    groups = hits[0].groups()
    return (spec["corpus_name"].format(*groups),
            _thousands(spec["size"].format(*groups)),
            spec["unit"])


def spread(rows: dict[str, dict[str, float]], spec: dict) -> tuple[float, float, float]:
    metric = spec["metric"]
    if spec["floor"] not in rows:
        raise SystemExit("board.txt: %r has no %r row" % (spec["header"], spec["floor"]))
    floor = rows[spec["floor"]][metric]
    entrants = [v[metric] for name, v in rows.items()
                if name != spec["floor"] and name not in spec["skip"]]
    if not entrants:
        raise SystemExit("board.txt: %r has no scored entrant" % spec["header"])
    return floor, min(entrants), max(entrants)


# ------------------------------------------------------------------------- reading the verdicts

def covered(spec: dict, families: dict) -> list[dict]:
    """Every declared contrast this board speaks for.

    A board usually owns whole families. ``post_detection_auc`` is the exception: one declared
    family spans both corpora, so the two Detection rows each take an id prefix out of it. The
    prefix is matched, never inferred, and an empty selection is a hard failure rather than an
    empty fraction.
    """
    out: list[dict] = []
    for fid, prefix in spec["coverage"]:
        members = families.get(fid)
        if members is None:
            raise SystemExit("unknown comparison family: %s" % fid)
        chosen = [c for c in members if prefix is None or c["id"].startswith(prefix)]
        if not chosen:
            raise SystemExit("family %s has no claim under prefix %r (board %r)"
                             % (fid, prefix, spec["header"]))
        out.extend(chosen)
    return out


def separates_cell(spec: dict, families: dict) -> str:
    """How many of the board's declared contrasts the Holm-adjusted tests separate."""
    members = covered(spec, families)
    if not members:
        return "--"
    return "%d/%d" % (sum(c["verdict"] in _SEPARATING for c in members), len(members))


def check_partition(families: dict) -> None:
    """The boards must partition the declared contrasts: each one covered exactly once.

    Without this the Separates column can shrink silently. It is the failure the column replaced:
    the prose it supersedes reported 7 of 16 families, and as prose that read as a selection. A
    fraction does not, so the denominators have to be the real ones.
    """
    everything = {c["id"] for members in families.values() for c in members}
    seen: dict[str, str] = {}
    for spec in _BOARDS:
        for claim in covered(spec, families):
            first = seen.get(claim["id"])
            if first is not None:
                raise SystemExit("claim %s is claimed by both %r and %r"
                                 % (claim["id"], first, spec["header"]))
            seen[claim["id"]] = spec["header"]
    missing = sorted(everything - set(seen))
    if missing:
        raise SystemExit("%d declared contrast(s) belong to no board, so the Separates column "
                         "would under-count: %s" % (len(missing), ", ".join(missing[:6])))


def rows_latex(preamble: list[str], blocks: dict, claims: dict, families: dict) -> list[str]:
    check_partition(families)
    out = []
    for spec in _BOARDS:
        rows = blocks.get(spec["header"])
        if rows is None:
            raise SystemExit("board.txt has no block %r" % spec["header"])
        floor, low, high = spread(rows, spec)
        name, size, unit = corpus_cells(preamble, spec)
        tint = r"\rowcolor{abMint!35}" if spec.get("gold") else ""
        out.append("%s%s & %s & %s & %s & %s & %.3f & %.3f--%.3f & %s \\\\"
                   % (tint, spec["name"], name, size, unit, spec["metric_label"],
                      floor, low, high, separates_cell(spec, families)))
    return out


def load():
    board = BOARD.read_text(encoding="utf-8")
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    claims = {c["id"]: c for c in data["claims"]}
    families: dict[str, list] = {}
    for c in data["claims"]:
        families.setdefault(c["family"], []).append(c)
    for fam in data["comparison_families"]:
        families.setdefault(fam["id"], [])
    return read_board(board), claims, families


def table_body(text: str, label: str) -> list[str]:
    """The content rows of the tabular carrying ``label``, scaffolding and header dropped."""
    anchor = text.find(label)
    if anchor == -1:
        raise ValueError("%s is absent" % label)
    # the tabular may sit before or after the \label inside the float, so search the whole float
    start = text.rfind(r"\begin{table", 0, anchor)
    stop = text.find(r"\end{table", anchor)
    if start == -1 or stop == -1:
        raise ValueError("%s is not inside a table float" % label)
    float_text = text[start:stop]
    begin = _TABLE_BEGIN.search(float_text)
    end = _TABLE_END.search(float_text, begin.end()) if begin else None
    if not begin or not end:
        raise ValueError("%s has no tabular" % label)
    body = []
    for line in float_text[begin.end():end.start()].splitlines():
        s = line.strip()
        if s and s not in _SCAFFOLDING and not s.startswith("%") and "&" in s:
            body.append(s)
    return body


# The figure scripts read committed copies of these two files, because the paper repository is an
# Overleaf submodule that cannot import from a sibling checkout at build time. A copy is a drift
# source, so it is checked here rather than trusted.
_FIGURE_COPIES = (
    ("figure/board.txt", BOARD),
    ("figure/statistical_tests_results.json", RESULTS),
)

# Figure 1 prints a board count under each state label. It is hand-drawn TikZ, so nothing
# recomputes it, and adding a board here would leave the figure quietly claiming the old count
# on page 1 while Table 2 carried the new one.
_FIGURE = "fig_lifecycle.tex"
_STATE_LABEL = re.compile(r"statelab[^{}]*\{(PRE|LIVE|POST)\}")
_BOARD_COUNT = re.compile(r"\{(\d+) boards?\}")


def state_counts() -> dict[str, int]:
    """How many scored blocks each information state carries, from the declared inventory."""
    counts: dict[str, int] = {}
    for spec in _BOARDS:
        counts[spec["state"]] = counts.get(spec["state"], 0) + 1
    return counts


def _fmt_counts(counts: dict[str, int]) -> str:
    named = ", ".join("%s %d" % (state, counts[state])
                      for state in ("PRE", "LIVE", "POST") if state in counts)
    return named or "none"


def figure_counts(text: str) -> dict[str, int]:
    """The board count each state label claims, taken from the count node that follows it."""
    labels = [(m.start(), m.group(1)) for m in _STATE_LABEL.finditer(text)]
    counts = [(m.start(), int(m.group(1))) for m in _BOARD_COUNT.finditer(text)]
    found: dict[str, int] = {}
    for position, state in labels:
        if state in found:
            raise ValueError("%s labels %s twice" % (_FIGURE, state))
        following = [n for p, n in counts if p > position]
        if not following:
            raise ValueError("%s gives no board count for %s" % (_FIGURE, state))
        found[state] = following[0]
    return found


def check(paper: Path, generated: list[str]) -> int:
    stale: list[str] = []

    figure = paper / _FIGURE
    if not figure.exists():
        stale.append("%s is missing; Figure 1's board counts cannot be checked" % _FIGURE)
    else:
        expected = state_counts()
        try:
            printed = figure_counts(figure.read_text(encoding="utf-8"))
        except ValueError as error:
            stale.append(str(error))
        else:
            if printed != expected:
                stale.append("%s prints board counts %s; the board has %s"
                             % (_FIGURE, _fmt_counts(printed), _fmt_counts(expected)))

    for relative, source in _FIGURE_COPIES:
        copy = paper / relative
        if not copy.exists():
            stale.append("%s is missing; the figure scripts have no data" % relative)
        elif copy.read_bytes() != source.read_bytes():
            stale.append("%s differs from %s; regenerate the figures after copying it"
                         % (relative, source.name))
    path = paper / _RESULTS_SECTION
    alt = paper / _BENCHMARK
    text = ""
    for candidate in (alt, path):
        if candidate.exists() and r"\label{tab:boards}" in candidate.read_text(encoding="utf-8"):
            text = candidate.read_text(encoding="utf-8")
            break
    if not text:
        stale.append("tab:boards is in neither %s nor %s" % (_BENCHMARK, _RESULTS_SECTION))
    else:
        try:
            actual = table_body(text, r"\label{tab:boards}")
        except ValueError as error:
            stale.append(str(error))
        else:
            # the header row is the one whose first cell is the literal column name
            actual = [r for r in actual if not r.startswith("Board &")]
            if actual != generated:
                stale.append("the board table differs from the generated table "
                             "(%d content rows in the paper, %d generated)"
                             % (len(actual), len(generated)))
                for got, want in zip(actual, generated):
                    if got != want:
                        stale.append("  first difference: paper has %r" % got)
                        stale.append("                  generated %r" % want)
                        break
    for line in stale:
        print("STALE %s" % line)
    if stale:
        print("\n%d staleness finding(s). Regenerate with: python tools/emit_boards_table.py"
              % len(stale))
        return 1
    print("paper is current: %d scored boards (Figure 1: %s)"
          % (len(generated), _fmt_counts(state_counts())))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the paper's board table instead of printing it")
    parser.add_argument("--paper", default=os.environ.get("CATCHBENCH_PAPER_DIR"),
                        help="paper source directory (or set CATCHBENCH_PAPER_DIR)")
    args = parser.parse_args()
    (preamble, blocks), claims, families = load()
    generated = rows_latex(preamble, blocks, claims, families)
    if args.check:
        if not args.paper:
            parser.error("--check needs --paper <dir> or CATCHBENCH_PAPER_DIR")
        return check(Path(args.paper), generated)
    print("%% %d scored boards. Generated by tools/emit_boards_table.py; do not hand-edit."
          % len(generated))
    for row in generated:
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
