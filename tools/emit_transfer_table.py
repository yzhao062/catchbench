"""Emit the paper's transfer diagnostics from the committed seed records, so the two cannot drift.

Section 5.6 of the paper reports the unsupervised transfer arena: how far each off-the-shelf
detector's score moves across initialization seeds, how GAAN compares with the supervised reference,
and what survives once run size is controlled for. Every one of those numbers was computed once, read
off a console, and typed into the manuscript. Nothing recomputed them, and one had already gone
stale: the paragraph says twenty seeds and printed the five-seed Spearman correlation.

The inputs are two committed records written by ``tools/pygod_seed_stability.py``:

* ``pygod_seed_stability_results.json``, five seeds, and the only file carrying the supervised
  graph network's own seed values.
* ``pygod_seed_stability_results_20seeds.json``, twenty seeds, which is what Section 5.6 reports.

Everything else here is derived arithmetic on those arrays, so this tool needs no corpus, no GRADE
checkout, and no torch. That is the point: the numbers become reproducible from the artifact rather
than from a console session nobody kept.

Usage::

    python tools/emit_transfer_table.py            # print the LaTeX block for the appendix
    python tools/emit_transfer_table.py --check    # exit 1 if the paper is stale, printing the delta

``--check`` needs ``--paper <dir>`` or the ``CATCHBENCH_PAPER_DIR`` environment variable, and
compares the block between its two sentinel comments byte for byte, the same contract
``emit_stats_table.py --contrasts`` uses.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
FIVE = ROOT / "pygod_seed_stability_results.json"
TWENTY = ROOT / "pygod_seed_stability_results_20seeds.json"

_APPENDIX = "09_appendix.tex"
_BEGIN = ("% BEGIN GENERATED tab:transfer -- regenerate with: "
          "python tools/emit_transfer_table.py")
_END = "% END GENERATED tab:transfer"

# The four PyGOD detectors Section 5.6 reports, by the key the seed record uses.
_DETECTORS = (
    ("pygod (graph AD)", "DOMINANT"),
    ("pygod-anomalydae", "AnomalyDAE"),
    ("pygod-conad", "CONAD"),
    ("pygod-gaan", "GAAN"),
)
_SUPERVISED = "g-safeguard (sup GNN)"
_LEVEL = 0.95


def load() -> tuple[dict, dict]:
    return (json.loads(FIVE.read_text(encoding="utf-8")),
            json.loads(TWENTY.read_text(encoding="utf-8")))


def _values(node: dict, *path: str) -> list[float]:
    """Walk to a values array, failing loudly rather than returning a default.

    A missing key here means the seed record changed shape, which must stop the run: silently
    substituting an empty list would print a table of zeros that still looks like a table.
    """
    for key in path:
        if key not in node:
            raise SystemExit("seed record has no %s (looking for %s)" % (key, " / ".join(path)))
        node = node[key]
    if "values" not in node or not node["values"]:
        raise SystemExit("no values array at %s" % " / ".join(path))
    return [float(v) for v in node["values"]]


def _interval(sample: list[float], record_sd: float) -> tuple[float, float, float, float]:
    """Mean, the record's own spread, and the two-sided t interval on the mean.

    Two standard deviations are in play and the paper already prints both. Its plus-or-minus values
    are the ``std`` field the seed record writes, which is the population form over the seeds drawn.
    Its intervals are the ordinary t interval, which needs the sample form. Reporting the record's
    value keeps the table agreeing with both the manuscript and the shipped artifact, and the caption
    names the difference rather than leaving a reader to find it in the third decimal.
    """
    array = np.asarray(sample, dtype=float)
    mean = float(array.mean())
    sample_sd = float(array.std(ddof=1))
    half = float(stats.t.ppf(0.5 + _LEVEL / 2, len(array) - 1)) * sample_sd / np.sqrt(len(array))
    return mean, float(record_sd), mean - half, mean + half


def welch(a: list[float], b: list[float]) -> dict:
    """Welch's unequal-variance comparison of two seed samples, with its interval and p."""
    first, second = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    result = stats.ttest_ind(first, second, equal_var=False)
    difference = float(first.mean() - second.mean())
    standard_error = float(np.sqrt(first.var(ddof=1) / first.size
                                   + second.var(ddof=1) / second.size))
    half = float(stats.t.ppf(0.5 + _LEVEL / 2, float(result.df))) * standard_error
    return {"difference": difference, "standard_error": standard_error,
            "df": float(result.df), "p": float(result.pvalue),
            "low": difference - half, "high": difference + half,
            "n_a": int(first.size), "n_b": int(second.size)}


def diagnostics(five: dict, twenty: dict) -> dict:
    """Every quantity Section 5.6 prints, derived from the two committed seed records."""
    out: dict = {"detectors": {}, "supervised": {}, "size_control": {}, "welch": {}}

    for corpus in ("swegym", "tau"):
        rows = []
        for key, name in _DETECTORS:
            # _values raises on a missing key, so read it first and take the record's own
            # spread from the same node afterwards.
            sample = _values(twenty["detection"][corpus]["detectors"], key, "roc_auc")
            node = twenty["detection"][corpus]["detectors"][key]["roc_auc"]
            if "std" not in node:
                raise SystemExit("seed record has no std for %s on %s" % (key, corpus))
            mean, sd, low, high = _interval(sample, node["std"])
            rows.append({"name": name, "mean": mean, "sd": sd, "low": low, "high": high,
                         "n": len(sample)})
        out["detectors"][corpus] = rows

    # The supervised graph network's own seeds live only in the five-seed record, because the
    # twenty-seed sweep re-ran the unsupervised family alone.
    supervised = _values(five["leaders"]["swegym"], _SUPERVISED, "roc_auc")
    mean, sd, low, high = _interval(
        supervised, five["leaders"]["swegym"][_SUPERVISED]["roc_auc"]["std"])
    out["supervised"] = {"name": _SUPERVISED, "mean": mean, "sd": sd, "low": low, "high": high,
                         "n": len(supervised)}

    gaan = twenty["detection"]["swegym"]["detectors"]["pygod-gaan"]
    out["welch"] = welch(_values(twenty["detection"]["swegym"]["detectors"], "pygod-gaan",
                                 "roc_auc"), supervised)

    matched = _values(twenty["detection"]["swegym"]["detectors"], "pygod-gaan",
                      "exact_size_matched_auc")
    mean, sd, low, high = _interval(matched, gaan["exact_size_matched_auc"]["std"])
    match = gaan["exact_size_match"]
    out["size_control"] = {
        "spearman": float(gaan["score_size_spearman"]["mean"]),
        "spearman_five": float(five["detection"]["swegym"]["detectors"]["pygod-gaan"]
                               ["score_size_spearman"]["mean"]),
        "runs": int(match["run_count"]),
        "strata": int(match["stratum_count"]),
        "pairs": int(match["pair_count"]),
        "auc_mean": mean, "auc_sd": sd, "auc_low": low, "auc_high": high, "auc_n": len(matched),
        "total_runs": int(twenty["detection"]["swegym"]["runs"]),
    }
    return out


def _p(value: float) -> str:
    if value >= 0.001:
        return r"$%.3f$" % value
    exponent = 0
    while value < 1:
        value *= 10
        exponent += 1
    return r"$%.1f{\times}10^{-%d}$" % (value, exponent)


def table(five: dict, twenty: dict) -> str:
    """The whole float, markers included, ready to paste into the appendix."""
    d = diagnostics(five, twenty)
    seeds = d["detectors"]["swegym"][0]["n"]
    control, welch_result = d["size_control"], d["welch"]

    lines = [
        _BEGIN,
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Transfer diagnostics for Section~\ref{sec:res-transfer}, derived from the two "
        r"committed seed records. The upper block is each unsupervised detector's ROC-AUC over "
        r"%d initialization seeds, with the interval on its mean. The supervised reference varies "
        r"on a different axis and over five seeds, so its row is separated. The lower block is the "
        r"run-size control on GAAN, and the Welch comparison that places it below the supervised "
        r"reference. SD is the seed record's own \texttt{std} field, which is the population form "
        r"over the seeds drawn, and is what the body prints after a plus-or-minus; the interval is "
        r"the ordinary t interval and therefore uses the sample form, so its half-width is slightly "
        r"wider than SD alone implies. Generated by \texttt{tools/emit\_transfer\_table.py}, which reads only "
        r"\texttt{tools/pygod\_seed\_stability\_results\{,\_20seeds\}.json}; its \texttt{-{}-check} mode "
        r"runs before a submission and fails when this table falls behind them.}" % seeds,
        r"\label{tab:transfer}",
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"Corpus & Entrant & Mean & SD & 95\% CI on the mean \\",
        r"\midrule",
    ]
    for corpus, printed in (("swegym", "SWE-Gym"), ("tau", "tau-bench")):
        for index, row in enumerate(d["detectors"][corpus]):
            lines.append(r"%s & %s & %.3f & %.3f & $[%.3f, %.3f]$ \\"
                         % (printed if index == 0 else "", row["name"], row["mean"], row["sd"],
                            row["low"], row["high"]))
        lines.append(r"\addlinespace")
    supervised = d["supervised"]
    lines.append(r"SWE-Gym & %s, %d seeds & %.3f & %.3f & $[%.3f, %.3f]$ \\"
                 % (supervised["name"].replace("_", r"\_"), supervised["n"], supervised["mean"],
                    supervised["sd"], supervised["low"], supervised["high"]))
    lines += [
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\emph{Run-size control on GAAN, SWE-Gym}} \\",
        r"Quantity & \multicolumn{4}{l}{Value} \\",
        r"Spearman of run score with node count & \multicolumn{4}{l}{$%.3f$ over %d seeds "
        r"(the five-seed record reads $%.3f$)} \\"
        % (control["spearman"], seeds, control["spearman_five"]),
        r"Runs in strata holding both outcomes & \multicolumn{4}{l}{%d of %d, in %d node-count "
        r"strata, giving %d positive-negative pairs} \\"
        % (control["runs"], control["total_runs"], control["strata"], control["pairs"]),
        r"Within-size ROC-AUC on that pair set & \multicolumn{4}{l}{%.3f $\pm$ %.3f over %d seeds, "
        r"95\%% CI $[%.3f, %.3f]$} \\"
        % (control["auc_mean"], control["auc_sd"], control["auc_n"], control["auc_low"],
           control["auc_high"]),
        r"\addlinespace",
        r"\multicolumn{5}{@{}l}{\emph{GAAN against the supervised graph network, Welch on the seed "
        r"samples}} \\",
        r"Difference in means & \multicolumn{4}{l}{$%+.3f$, 95\%% CI $[%.3f, %.3f]$, "
        r"$\nu = %.1f$, $p = %s$} \\"
        % (welch_result["difference"], welch_result["low"], welch_result["high"],
           welch_result["df"], _p(welch_result["p"]).strip("$")),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        _END,
    ]
    block = "\n".join(lines)
    if "\t" in block:
        raise SystemExit("a literal tab reached the generated block")
    return block


def check(paper: Path, generated: str) -> int:
    appendix = paper / _APPENDIX
    stale: list[str] = []
    if not appendix.exists():
        stale.append("%s is missing" % _APPENDIX)
    else:
        text = appendix.read_text(encoding="utf-8")
        for marker in (_BEGIN, _END):
            found = re.findall(rf"(?m)^[ \t]*{re.escape(marker)}[ \t]*$", text)
            if len(found) != 1:
                stale.append("%s: the transfer marker %s... appears %d times, expected once"
                             % (_APPENDIX, marker[:44], len(found)))
                break
        else:
            start, stop = text.index(_BEGIN), text.index(_END) + len(_END)
            if stop < start:
                stale.append("%s: the transfer end marker precedes its begin marker" % _APPENDIX)
            else:
                actual = text[start:stop].replace("\r\n", "\n").splitlines()
                wanted = generated.splitlines()
                if actual != wanted:
                    stale.append("%s: the transfer table differs from the generated block "
                                 "(%d lines in the paper, %d generated)"
                                 % (_APPENDIX, len(actual), len(wanted)))
                    for index, (got, want) in enumerate(zip(actual, wanted)):
                        if got != want:
                            stale.append("  first difference at block line %d: paper has %r"
                                         % (index + 1, got))
                            stale.append("                                     generated %r" % want)
                            break
    for line in stale:
        print("STALE %s" % line)
    if stale:
        print("\n%d staleness finding(s). Regenerate with: "
              "python tools/emit_transfer_table.py" % len(stale))
        return 1
    print("paper is current: transfer diagnostics over %d seeds" % len(
        _values(load()[1]["detection"]["swegym"]["detectors"], "pygod-gaan", "roc_auc")))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the paper's transfer table instead of printing it")
    parser.add_argument("--paper", default=os.environ.get("CATCHBENCH_PAPER_DIR"),
                        help="paper source directory (or set CATCHBENCH_PAPER_DIR)")
    args = parser.parse_args()
    five, twenty = load()
    generated = table(five, twenty)
    if args.check:
        if not args.paper:
            parser.error("--check needs --paper <dir> or CATCHBENCH_PAPER_DIR")
        return check(Path(args.paper), generated)
    print(generated)
    return 0


if __name__ == "__main__":
    sys.exit(main())
