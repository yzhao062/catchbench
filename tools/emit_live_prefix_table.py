"""Emit both LIVE prefix tables from committed scores and per-arm intervals.

The appendix printed prefix scores without their sampling uncertainty. The 40 nonrandom cells
already have per-arm intervals in ``tools/statistical_tests_results.json``, recorded on AUC minus
the fixed 0.70 bar. Adding that constant to both endpoints recovers each arm's own interval
without fitting, resampling, or testing another comparison.

The two corpora build that interval differently and the tables say so. SWE-Gym keeps a run-level
single-curve DeLong interval, which is symmetric about the estimate. tau-bench's 660 runs are 165
task instances each attempted by four agent models, so a run-level interval treats repeated
attempts at one task as separate observations; its cells carry a task-clustered stratified
percentile bootstrap instead, which is asymmetric. The emitter reads the construction off each
corpus's own cells rather than hard-coding one, checks the invariants that construction actually
has, and prints the matching caption sentence. A cell whose recorded method is neither, or whose
endpoints fail that method's invariants, is rejected.

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

# The two interval constructions a cell may carry, keyed by the (method, axis) pair the statistical
# record stores. Nothing else is accepted: an unrecognised pair is a record this emitter has never
# been told how to check, and printing it would be printing an interval whose invariants nobody
# verified. "delong" is symmetric about the estimate by construction; "clustered" is a percentile
# bootstrap and is not, which is why the symmetry check is per construction rather than global.
_DELONG = ("single-curve DeLong", "run-level sampling")
_CLUSTERED = ("task-clustered stratified percentile bootstrap", "task-cluster resampling")
_CONSTRUCTIONS = {_DELONG: "delong", _CLUSTERED: "clustered"}


def _check_endpoints(claim_id: str, kind: str, claim: dict) -> None:
    """The invariants of one construction, checked against that construction and not another.

    Both kinds must contain their own estimate and sit on the AUC-minus-bar scale. Beyond that they
    diverge. A DeLong interval is estimate plus or minus a multiple of one standard error, so its
    midpoint is the estimate and a midpoint that has drifted means the endpoints were shifted,
    recentred, or copied from another cell. A percentile interval has no such identity, so the
    equivalent check is against the record that produced it: the resampled statistic must be this
    cell's own estimate, the draw counts must be coherent, and the stored endpoints must be the ones
    the bootstrap returned. That is what stops a clustered cell from silently carrying another arm's
    interval, which is the failure the symmetry check used to catch for free.
    """
    estimate, interval = claim["estimate"], claim["interval"]
    a, b = estimate["a"], estimate["b"]
    low, high = interval["low"], interval["high"]
    difference = estimate["difference_a_minus_b"]
    if (not all(math.isfinite(v) for v in (a, b, low, high, difference))
            or not 0 <= a <= 1 or not low <= difference <= high
            or not math.isclose(difference, a - b, rel_tol=0, abs_tol=1e-12)):
        raise SystemExit(f"inconsistent AUC-minus-bar interval: {claim_id}")
    if kind == "delong":
        if not math.isclose((low + high) / 2, difference, rel_tol=0, abs_tol=1e-12):
            raise SystemExit(f"inconsistent AUC-minus-bar interval: {claim_id}")
        return
    clustered = claim.get("clustered_interval")
    if not isinstance(clustered, dict):
        raise SystemExit(f"clustered interval without its resampling record: {claim_id}")
    replicates = clustered.get("replicates")
    usable = clustered.get("usable_replicates")
    if (clustered.get("method"), clustered.get("axis")) != _CLUSTERED:
        raise SystemExit(f"clustered resampling record names another construction: {claim_id}")
    if (not isinstance(replicates, int) or not isinstance(usable, int)
            or not 2 <= usable <= replicates):
        raise SystemExit(f"clustered interval has an unusable draw count: {claim_id}")
    stored = clustered.get("interval_95")
    if (not isinstance(stored, list) or len(stored) != 2
            or not math.isclose(stored[0], low, rel_tol=0, abs_tol=1e-12)
            or not math.isclose(stored[1], high, rel_tol=0, abs_tol=1e-12)):
        raise SystemExit(f"clustered endpoints disagree with their resampling record: {claim_id}")
    if not math.isclose(clustered.get("point", math.nan), difference, rel_tol=0, abs_tol=1e-9):
        raise SystemExit(f"clustered interval was resampled around another cell's estimate: "
                         f"{claim_id}")
    if not low < difference < high:
        raise SystemExit(f"clustered interval does not contain its own estimate: {claim_id}")


def _threshold_cells(record: dict) -> tuple[dict, dict]:
    """Validate all 40 cells; return their shifted endpoints and each corpus's construction."""
    expected = {
        f"live.{corpus}.bar.{prefix}.{method}": (corpus, family, method, prefix)
        for corpus, family, _, _ in _CORPORA
        for method, _ in _METHODS[1:]
        for prefix in _PREFIXES
    }
    cells = {}
    kinds: dict[str, dict] = {}
    for claim in record["claims"]:
        if "threshold" not in claim["family"]:
            continue
        claim_id = claim["id"]
        if claim_id not in expected or claim_id in cells:
            raise SystemExit(f"unexpected or duplicate threshold cell: {claim_id}")
        corpus, family, method, prefix = expected[claim_id]
        estimate, interval = claim["estimate"], claim["interval"]
        kind = _CONSTRUCTIONS.get((interval.get("method"), interval.get("axis")))
        if (claim["family"] != family or claim["metric"] != "roc_auc"
                or estimate["a_name"] != method or estimate["b_name"] != "fixed bar"
                or estimate["b"] != 0.70 or interval["level"] != 0.95 or kind is None):
            raise SystemExit(f"not a 95% run-level single-curve interval, nor a task-clustered "
                             f"percentile interval, against 0.70: {claim_id}")
        _check_endpoints(claim_id, kind, claim)
        seen = kinds.setdefault(corpus, {"kind": kind, "claim_id": claim_id})
        if seen["kind"] != kind:
            raise SystemExit(f"corpus {corpus} mixes interval constructions: "
                             f"{seen['claim_id']} is {seen['kind']} and {claim_id} is {kind}. "
                             f"One caption cannot describe two constructions.")
        a, b = estimate["a"], estimate["b"]
        board = claim["variance_axes"]["board_point_estimate"]["value"]
        if not math.isfinite(board) or not 0 <= board <= 1:
            raise SystemExit(f"invalid board point estimate: {claim_id}")
        gap = abs(board - a)
        if gap > MAX_CENTRE_GAP:
            raise SystemExit(f"board/pooled-score centre gap {gap:.6f} exceeds "
                             f"MAX_CENTRE_GAP={MAX_CENTRE_GAP:.3f}: {claim_id} "
                             f"(board={board:.6f}, pooled={a:.6f})")
        cells[claim_id] = (board, a, interval["low"] + b, interval["high"] + b)
        if kind == "clustered":
            kinds[corpus]["clustered"] = claim["clustered_interval"]
    missing = expected.keys() - cells.keys()
    if missing:
        raise SystemExit("missing threshold cells: " + ", ".join(sorted(missing)))
    return ({(corpus, method, prefix): cells[claim_id]
             for claim_id, (corpus, _, method, prefix) in expected.items()}, kinds)


def intervals(record: dict) -> dict:
    """Validate all 40 cells; return board scores with shifted per-arm interval endpoints."""
    return _threshold_cells(record)[0]


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


_NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def _tex(text: str) -> str:
    """Pass a record string through to the caption, refusing anything LaTeX would mangle.

    Method and axis names reach the printed caption verbatim, so a name carrying an underscore or a
    percent sign would silently become a subscript or eat the rest of the line. Escaping it would
    hide that a statistical record grew a name nobody chose for print; refusing says so.
    """
    if not re.fullmatch(r"[A-Za-z0-9 ()\-.,]+", str(text)):
        raise SystemExit(f"record string is not printable in a caption unescaped: {text!r}")
    return str(text)


def _interval_lines(kind: dict) -> list[str]:
    """How this corpus's interval was built, read off the record rather than fixed here.

    The two corpora no longer share a construction, and a caption naming one would be wrong about
    the other. A DeLong cell keeps, to the character, the line the appendix already carries, so a
    staleness check on the SWE-Gym block reports only what this change actually moved. A clustered
    cell names the method and axis the record itself stores.
    """
    if kind["kind"] == "delong":
        return [r"95\% single-curve DeLong interval for P (run-level sampling axis). For supervised"]
    clustered = kind["clustered"]
    return [
        r"95\% " + _tex(clustered["method"]) + r" interval for P",
        r"(" + _tex(clustered["axis"]) + r" axis). For supervised",
    ]


def _dependence_lines(kind: dict, run_count: int) -> list[str]:
    """The evidence for that choice, which differs by corpus and is owed to a reader.

    tau-bench is a task-by-model grid, so its runs are not independent draws and the interval
    resamples tasks. SWE-Gym's runs carry one distinct task identifier each, so resampling runs and
    resampling tasks are the same draw and its interval is left alone.
    """
    if kind["kind"] != "clustered":
        return [
            r"The " + str(int(run_count)) + r" runs of this corpus carry "
            + str(int(run_count)) + r" distinct task",
            r"identifiers, so resampling runs and resampling tasks are the same draw here.",
        ]
    clustered = kind["clustered"]
    attempts = _NUMBER_WORDS[int(clustered["rows_per_cluster"])]
    listed = ", ".join(f"{count} {name}"
                       for name, count in sorted(clustered["clusters_per_stratum"].items()))
    return [
        r"The " + str(int(clustered["n_rows"])) + r" runs of this corpus are "
        + str(int(clustered["n_clusters"])) + r" task instances, each",
        r"attempted by " + attempts + r" agent models. Counting those attempts as independent "
        r"observations",
        r"gives an interval narrower than the data support, so each draw resamples task instances",
        r"with replacement within domain (" + _tex(listed) + r"), carrying all " + attempts
        + r" attempts at a",
        r"drawn task together. The interval is the 2.5 and 97.5 percentiles of "
        + f"{int(clustered['replicates'])}" + r" such draws",
        r"(seed " + str(int(clustered["seed"])) + r"), so it is asymmetric about P and its last "
        r"printed digit carries",
        r"Monte Carlo noise of about "
        + f"{float(clustered['monte_carlo_noise_on_one_endpoint']):.3f}" + r".",
    ]


def table(record: dict, board: str) -> str:
    """Both complete floats, with the original six columns and six method rows preserved."""
    cells, kinds = _threshold_cells(record)
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
            *_interval_lines(kinds[corpus]),
            r"size, auditable, and full, B is mean fold AUC and P is the AUC of seed-averaged",
            r"out-of-fold scores; their absolute gap is at most " + f"{MAX_CENTRE_GAP:.3f}" + r" here.",
            r"For ECOD and dep-span, B and P agree at the displayed precision. See",
            r"Appendix~\ref{app:board-values} for the same convention gap on the POST board.",
            r"Each interval is one entrant's own, over the " + str(run_count) + r" labeled runs of this",
            r"corpus. It is not a comparison: the entrants share those runs, so reading two intervals",
            r"against each other discards the pairing a paired test keeps, and overlap is therefore not",
            r"evidence that two entrants perform alike. Comparing P's interval with the fixed 0.70",
            r"bar gives a marginal, unadjusted reading for that cell. We claim no simultaneous",
            r"coverage across methods or prefixes.",
            *_dependence_lines(kinds[corpus], run_count),
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
