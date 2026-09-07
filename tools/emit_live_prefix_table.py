"""Emit both LIVE prefix tables from committed scores and per-arm intervals.

The appendix printed prefix scores without their sampling uncertainty. The 40 nonrandom cells
already have single-curve DeLong intervals in ``tools/statistical_tests_results.json``, recorded
on AUC minus the fixed 0.70 bar. Adding that constant to both endpoints recovers each arm's own
interval without fitting, resampling, or testing another comparison.

The displayed score is ``variance_axes.board_point_estimate.value``, matching the LIVE board.
For supervised methods it is mean fold AUC, whereas the interval is centred on ``estimate.a``,
the AUC of seed-averaged out-of-fold scores. Keep those fields distinct: shift the endpoints by
the fixed bar only, without recentering them on the displayed score. The captions disclose and
the emitter enforces their maximum gap. Per-arm intervals cannot establish differences or
equality between arms.

The paper also prints a random reference and a time-to-detection column. The statistical record
has no LIVE random interval. Preserve those board cells from ``tests/golden/board.txt`` and state
that random has point estimates only. Thus the two tables have 48 prefix cells but only 40
recorded intervals. Validate the anonymous board blocks against the record's board estimates
before assigning them to corpora. No corpus, GRADE checkout, or numerical package is needed.

Usage::

    python tools/emit_live_prefix_table.py            # print both complete LaTeX blocks
    python tools/emit_live_prefix_table.py --check    # exit 1 on drift and print a unified diff

``--check`` needs ``--paper <dir>`` or ``CATCHBENCH_PAPER_DIR``. It compares both sentinel-comment
blocks in ``09_appendix.tex`` exactly, allowing only platform line-ending differences. It reads
the manuscript without writing it.
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "statistical_tests_results.json"
BOARD = ROOT.parent / "tests" / "golden" / "board.txt"

_APPENDIX = "09_appendix.tex"
_PREFIXES = (25, 50, 75, 100)
_CORPORA = (
    ("swe", "live_swegym_threshold_auc", "SWE-Gym", "tab:live-stream"),
    ("tau", "live_tau_threshold_auc", "tau-bench", "tab:live-stream-tau"),
)
_METHODS = (
    ("random", "random"),
    ("size (flat)", "size (flat)"),
    ("auditable (size+deps)", r"\texttt{auditable} (size+deps)"),
    ("full", "full"),
    ("pyod (ECOD)", "PyOD (ECOD, unsup.)"),
    ("dep-span (online)", "dep-span (online)"),
)
_BOARD_TITLE = (
    "LIVE streaming early-warning (ROC-AUC by prefix; t2d = earliest prefix with AUC>=0.70):"
)


def load() -> tuple[dict, str]:
    return (json.loads(RESULTS.read_text(encoding="utf-8")), BOARD.read_text(encoding="utf-8"))


def _markers(label: str) -> tuple[str, str]:
    return ("% BEGIN GENERATED " + label + " -- regenerate with: "
            "python tools/emit_live_prefix_table.py", "% END GENERATED " + label)


# Shared by the input guard and caption so estimator drift cannot leave a stale bound.
MAX_CENTRE_GAP = 0.008


def intervals(record: dict) -> dict:
    """Validate all 40 cells; return board scores with shifted single-curve interval endpoints."""
    expected = {
        f"live.{corpus}.bar.{prefix}.{method}": (corpus, family, method, prefix)
        for corpus, family, _, _ in _CORPORA
        for method, _ in _METHODS[1:]
        for prefix in _PREFIXES
    }
    cells = {}
    for claim in record["claims"]:
        if "threshold" not in claim["family"]:
            continue
        claim_id = claim["id"]
        if claim_id not in expected or claim_id in cells:
            raise SystemExit(f"unexpected or duplicate threshold cell: {claim_id}")
        corpus, family, method, prefix = expected[claim_id]
        estimate, interval = claim["estimate"], claim["interval"]
        if (claim["family"] != family or claim["metric"] != "roc_auc"
                or estimate["a_name"] != method or estimate["b_name"] != "fixed bar"
                or estimate["b"] != 0.70 or interval["level"] != 0.95
                or interval["method"] != "single-curve DeLong"
                or interval["axis"] != "run-level sampling"):
            raise SystemExit(f"not a 95% run-level single-curve interval against 0.70: {claim_id}")
        a, b = estimate["a"], estimate["b"]
        low, high = interval["low"], interval["high"]
        difference = estimate["difference_a_minus_b"]
        if (not all(math.isfinite(v) for v in (a, b, low, high, difference))
                or not 0 <= a <= 1 or not low <= difference <= high
                or not math.isclose(difference, a - b, rel_tol=0, abs_tol=1e-12)
                or not math.isclose((low + high) / 2, difference, rel_tol=0, abs_tol=1e-12)):
            raise SystemExit(f"inconsistent AUC-minus-bar interval: {claim_id}")
        board = claim["variance_axes"]["board_point_estimate"]["value"]
        if not math.isfinite(board) or not 0 <= board <= 1:
            raise SystemExit(f"invalid board point estimate: {claim_id}")
        gap = abs(board - a)
        if gap > MAX_CENTRE_GAP:
            raise SystemExit(f"board/pooled-score centre gap {gap:.6f} exceeds "
                             f"MAX_CENTRE_GAP={MAX_CENTRE_GAP:.3f}: {claim_id} "
                             f"(board={board:.6f}, pooled={a:.6f})")
        cells[claim_id] = (board, a, low + b, high + b)
    missing = expected.keys() - cells.keys()
    if missing:
        raise SystemExit("missing threshold cells: " + ", ".join(sorted(missing)))
    return {(corpus, method, prefix): cells[claim_id]
            for claim_id, (corpus, _, method, prefix) in expected.items()}


def _board_rows(text: str, record: dict) -> dict:
    """Read the existing random and t2d cells, checking corpus identity against the JSON."""
    blocks = re.findall(r"(?m)^" + re.escape(_BOARD_TITLE) + r"\r?\n((?:[ \t]+[^\n]+\n)+)",
                        text + "\n")
    if len(blocks) != len(_CORPORA):
        raise SystemExit(f"expected two LIVE prefix board blocks, found {len(blocks)}")
    claims = {claim["id"]: claim for claim in record["claims"]}
    rows = {}
    for (corpus, _, _, _), block in zip(_CORPORA, blocks):
        lines = block.splitlines()
        if lines[0].split() != ["method", "25%", "50%", "75%", "100%", "t2d"]:
            raise SystemExit(f"unexpected LIVE prefix board columns: {corpus}")
        entries = {}
        for line in lines[1:]:
            parts = line.strip().rsplit(None, 5)
            if len(parts) != 6 or parts[0] in entries:
                raise SystemExit(f"malformed or duplicate LIVE prefix board row: {line}")
            method, *values = parts
            if (not all(re.fullmatch(r"0\.\d{3}|1\.000", value) for value in values[:4])
                    or values[4] not in ("25%", "50%", "75%", "100%", ">100%")):
                raise SystemExit(f"invalid LIVE prefix board cells: {line}")
            entries[method] = values
        if set(entries) != {method for method, _ in _METHODS}:
            raise SystemExit(f"unexpected LIVE prefix board methods: {corpus}")
        for method, _ in _METHODS[1:]:
            for index, prefix in enumerate(_PREFIXES):
                claim_id = f"live.{corpus}.bar.{prefix}.{method}"
                value = claims[claim_id]["variance_axes"]["board_point_estimate"]["value"]
                if entries[method][index] != f"{value:.3f}":
                    raise SystemExit(f"board point estimate disagrees with statistical record: "
                                     f"{claim_id}")
        rows[corpus] = entries
    return rows


def table(record: dict, board: str) -> str:
    """Both complete floats, with the original six columns and six method rows preserved."""
    cells = intervals(record)
    rows = _board_rows(board, record)
    blocks = []
    for corpus, _, printed, label in _CORPORA:
        begin, end = _markers(label)
        counts = {
            claim["variance_axes"]["run_sampling"]["n"]
            for claim in record["claims"]
            if claim["id"].startswith(f"live.{corpus}.bar.")
        }
        if len(counts) != 1:
            raise SystemExit(f"LIVE {corpus} cells disagree on the labeled-run count: "
                             f"{sorted(counts)}. The caption prints one number for the corpus.")
        run_count = counts.pop()
        lines = [
            begin,
            r"\begin{table}[t]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\caption{LIVE streaming early warning on " + printed + ". Each nonrandom cell prints",
            r"the board ROC-AUC (B), the interval's point estimate (P), and an unadjusted",
            r"95\% single-curve DeLong interval for P (run-level sampling axis). For supervised",
            r"size, auditable, and full, B is mean fold AUC and P is the AUC of seed-averaged",
            r"out-of-fold scores; their absolute gap is at most " + f"{MAX_CENTRE_GAP:.3f}" + r" here.",
            r"For ECOD and dep-span, B and P agree at the displayed precision. See",
            r"Appendix~\ref{app:board-values} for the same convention gap on the POST board.",
            r"Each interval is one entrant's own, over the " + str(run_count) + r" labeled runs of this",
            r"corpus. It is not a comparison: the entrants share those runs, so reading two intervals",
            r"against each other discards the pairing a paired test keeps, and overlap is therefore not",
            r"evidence that two entrants perform alike. These intervals do not encode the separate",
            r"multiplicity-adjusted conclusions about the 0.70 bar reported in the text.",
            r"PyOD (ECOD) is batch-unsupervised; dep-span is",
            r"strict per-run online.",
            r"Random is a label-independent reference with board point estimates only; no interval",
            r"is recorded. Time to detection (t2d) is the earliest prefix whose board ROC-AUC reaches",
            r"0.70 (``none'' if no prefix does). Prefix fractions are relative to eventual trace length.}",
            r"\label{" + label + "}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Method & 25\% & 50\% & 75\% & 100\% & t2d \\",
            r"\midrule",
        ]
        for method, name in _METHODS:
            if method == "pyod (ECOD)":
                lines.append(r"\midrule")
            values = rows[corpus][method]
            t2d = "none" if values[4] == ">100%" else values[4].replace("%", r"\%")
            if method == "random":
                lines.append(" & ".join([name, *values[:4], t2d]) + r" \\")
                continue
            lines.append(name)
            for prefix in _PREFIXES:
                board_point, interval_point, low, high = cells[corpus, method, prefix]
                lines.append(
                    r"  & \shortstack{B: %.3f\\P: %.3f\\$[%.3f, %.3f]$}"
                    % (board_point, interval_point, low, high)
                )
            lines.append("  & " + t2d + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", end]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def check(paper: Path, generated: str) -> int:
    appendix = paper / _APPENDIX
    if not appendix.is_file():
        print(f"STALE {_APPENDIX} is missing")
        return 1
    text = appendix.read_text(encoding="utf-8")
    stale = 0
    spans = []
    for _, _, _, label in _CORPORA:
        begin, end = _markers(label)
        matches = [list(re.finditer(rf"(?m)^[ \t]*{re.escape(marker)}[ \t]*$", text))
                   for marker in (begin, end)]
        if any(len(found) != 1 for found in matches):
            print(f"STALE {_APPENDIX}: {label} has {len(matches[0])} begin and "
                  f"{len(matches[1])} end markers, expected one of each")
            stale += 1
            continue
        first, last = matches[0][0], matches[1][0]
        if last.start() < first.start():
            print(f"STALE {_APPENDIX}: {label} end marker precedes its begin marker")
            stale += 1
            continue
        spans.append((first.start(), last.end()))
        actual = text[first.start():last.end()].splitlines()
        wanted = generated[generated.index(begin):generated.index(end) + len(end)].splitlines()
        if actual != wanted:
            print(f"STALE {_APPENDIX}: {label} differs from the generated block")
            print("\n".join(difflib.unified_diff(actual, wanted,
                                               fromfile=f"{_APPENDIX}:{label}",
                                               tofile=f"generated:{label}", lineterm="")))
            stale += 1
    if len(spans) == 2 and max(start for start, _ in spans) < min(stop for _, stop in spans):
        print(f"STALE {_APPENDIX}: LIVE prefix generated blocks overlap")
        stale += 1
    if stale:
        print(f"\n{stale} staleness finding(s). Regenerate with: "
              "python tools/emit_live_prefix_table.py")
        return 1
    print("paper is current: LIVE prefix tables (40 per-arm intervals; random points only)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify both paper tables instead of printing them")
    parser.add_argument("--paper", default=os.environ.get("CATCHBENCH_PAPER_DIR"),
                        help="paper source directory (or set CATCHBENCH_PAPER_DIR)")
    args = parser.parse_args()
    if args.check and not args.paper:
        parser.error("--check needs --paper <dir> or CATCHBENCH_PAPER_DIR")
    try:
        generated = table(*load())
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise SystemExit(f"invalid LIVE prefix input: {error}") from error
    if args.check:
        return check(Path(args.paper), generated)
    print(generated)
    return 0


if __name__ == "__main__":
    sys.exit(main())
