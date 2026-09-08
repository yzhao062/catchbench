"""Emit both PRE tables from committed configuration records and fixed predictions.

The six ``data/pre/<source>.json`` files retain configuration identities, sources,
capabilities, labels, and scanner inputs. The held-out predictions are in
``data/pre/llm_judge_method/llama-3.3-70b.json``. Three existing marginal precision
intervals come from ``tools/statistical_tests_results.json`` and keep their seed.
Everything else is computed with the released prediction and count functions,
without refitting or API calls. Thus the artifact reproduces the numbers without
depending on a console session or inverting a rounded paper cell.

The 2026-09-07 per-arm determination fixes the support, source order, zero
denominator convention, and RNG labels. Each method and metric resets its stream.
The existing bootstrap helper also fixes the draw order (batches of 500, then
sources in the declared order). Full-precision quantities and provenance travel
with each table as JSON comments; this emitter writes no result file.

Usage::

    python tools/emit_pre_tables.py
    python tools/emit_pre_tables.py --check --paper <paper-directory>

The default is two independently delimited LaTeX blocks on stdout. ``--check``
also accepts ``CATCHBENCH_PAPER_DIR`` and compares each block's UTF-8 bytes,
including whitespace and line endings. Required-file and section-boundary checks
run independently of the per-block loop.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import statistical_tests as stats
from catchbench.pre import PreOverPrivilege, _pre_logical_source, pre_instance_from_dict

BASE_SEED = 20260907
DRAWS = 10_000
SOURCES = ("crewai", "n8n", "mcp", "injecagent", "sweagent", "synthetic")
SOURCE_COUNTS = (298, 219, 144, 340, 130, 56)
PARSED_COUNTS = (298, 215, 143, 340, 130, 56)
METRICS = ("precision", "recall", "f1")
JUDGE = stats.PRE_JUDGE_METHOD
ORACLE = "oracle_privilege_diff"
SOURCE_METHODS = stats.PRE_METHOD_ORDER
MAIN_METHODS = (*SOURCE_METHODS[:9], JUDGE, ORACLE)
LABELS = ("tab:pre-source", "tab:pre-main")
APPENDIX = "09_appendix.tex"
BOARD_START = r"\label{app:board-values}"
MAIN_START = r"\subsection{Additional PRE Board Values}"
BOARD_END = r"\subsection{Additional LIVE Prefix Results}"
NAMES = dict(zip(SOURCE_METHODS, (
    "Flag all", "Flag none", "Risky permissions", "Excess permissions",
    "Excess functionality", "Privilege escalation", "Unrequested impact",
    "Sensitive access", "ASI combined", "Oracle", "Held-out LLM judge",
)))


def markers(label: str) -> tuple[str, str]:
    return (f"% BEGIN GENERATED {label} -- regenerate with: python tools/emit_pre_tables.py",
            f"% END GENERATED {label}")


def load() -> dict:
    """Read the exact six sources, retaining each file's original record order."""
    instances, files = [], {}
    for source, expected in zip(SOURCES, SOURCE_COUNTS):
        name = f"data/pre/{source}.json"
        raw = (ROOT / name).read_bytes()
        rows = json.loads(raw)
        if len(rows) != expected:
            raise ValueError(f"{name}: expected {expected} configurations, got {len(rows)}")
        for row in rows:
            if _pre_logical_source(row["source"]) != source:
                raise ValueError(f"{name}: unexpected source {row['source']!r}")
            instances.append(pre_instance_from_dict(row))
        files[name] = hashlib.sha256(raw).hexdigest()
    if len({row.instance_id for row in instances}) != sum(SOURCE_COUNTS):
        raise ValueError("PRE configuration instance_id values are not unique")
    for name in ("data/pre/llm_judge_method/llama-3.3-70b.json",
                 "tools/statistical_tests_results.json"):
        files[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    task = PreOverPrivilege(instances)
    predictions = stats._pre_prediction_maps(task)
    parsed_counts = tuple(sum(row.instance_id in predictions[JUDGE]
                              for row in instances if _pre_logical_source(row.source) == source)
                          for source in SOURCES)
    if parsed_counts != PARSED_COUNTS:
        raise ValueError(f"judge parsed support changed: {parsed_counts}, expected {PARSED_COUNTS}")
    if any(predictions[ORACLE][row.instance_id] != set(row.labels["excess_set"])
           for row in instances):
        raise ValueError("oracle no longer equals the label construction identity")
    record = json.loads((ROOT / "tools/statistical_tests_results.json").read_bytes())
    provenance = {
        "source_files_sha256": files,
        "corpus_revision": "sha256:" + hashlib.sha256(json.dumps(
            {name: digest for name, digest in files.items() if name.startswith("data/pre/")},
            sort_keys=True).encode()).hexdigest(),
        "identity_selectors": ["data/pre/<source>.json::$[*].instance_id",
                               "data/pre/<source>.json::$[*].source"],
        "input_selectors": ["$[*].labels.excess_set", "$[*].declared_capabilities[*].name",
                            "$[*].declared_capabilities[*].permission_level", "$[*].spec_tokens",
                            "$[*].spec_token_overrides", "$[*].minimal_reference"],
        "judge_selector": "data/pre/llm_judge_method/llama-3.3-70b.json::$[instance_id]",
        "source_order": list(SOURCES), "order_within_source": "original JSON array order",
        "prediction_function": "tools/statistical_tests.py::_pre_prediction_maps",
        "score_fold_seeds": [], "refit": False,
    }
    return {"instances": instances, "predictions": predictions,
            "record": record, "provenance": provenance}


def support(data: dict, method: str, source: str) -> tuple[list, list[np.ndarray]]:
    rows = [row for row in data["instances"]
            if (source == "overall" or _pre_logical_source(row.source) == source)
            and (method != JUDGE or row.instance_id in data["predictions"][method])]
    strata = [np.array([i for i, row in enumerate(rows)
                        if _pre_logical_source(row.source) == name], dtype=np.int64)
              for name in (SOURCES if source == "overall" else (source,))]
    return rows, strata


def quantities(data: dict) -> dict:
    """Compute each unique cell once, reusing overall F1 across the two tables."""
    out = {}
    for method in SOURCE_METHODS:
        for source in (*SOURCES, "overall"):
            rows, strata = support(data, method, source)
            counts = stats._pre_count_matrix(rows, data["predictions"][method])
            for metric in (METRICS if source == "overall" else ("f1",)):
                key = (method, source, metric)
                point = float(stats._pre_ratio(counts.sum(axis=0), metric))
                q = {"point": point, "metric": metric, "method": method, "source": source,
                     "support": {"configurations": len(rows),
                                 "kind": "parsed" if method == JUDGE else "all",
                                 "source_counts": dict(zip(
                                     SOURCES if source == "overall" else (source,),
                                     map(len, strata)))}}
                out[key] = q
                if method == ORACLE:
                    q.update(interval=None, exemption="construction identity")
                    continue
                label = f"per-arm/pre/{source}/{q['support']['kind']}"
                seed, draws = BASE_SEED, DRAWS
                recorded = (source == "overall" and metric == "precision"
                            and method in stats.PRE_NARROW_RULES)
                if recorded:
                    label = f"pre.precision.{method}.vs.base_rate"
                    matches = [(i, c) for i, c in enumerate(data["record"]["claims"])
                               if c["id"] == label]
                    if len(matches) != 1:
                        raise ValueError(f"expected exactly one recorded interval for {label}")
                    index, claim = matches[0]
                    if not np.isclose(point, claim["estimate"]["a"], rtol=0, atol=1e-14):
                        raise ValueError(f"{label}: recorded precision disagrees with configuration counts")
                    interval = dict(claim["precision_interval"])
                    seed = data["record"]["settings"]["random_seed"]
                    draws = data["record"]["settings"]["bootstrap_replicates"]
                    q["record_selector"] = (
                        f"tools/statistical_tests_results.json::$.claims[{index}].precision_interval")
                rng = stats._rng_for(label, seed)
                rng_seed = int(rng.bit_generator.seed_seq.entropy)
                if not recorded:
                    values, _, _ = stats._pre_configuration_bootstrap(
                        counts, counts, strata, metric, DRAWS, rng)
                    low, high = np.quantile(values, (0.025, 0.975), method="linear")
                    interval = {"low": float(low), "high": float(high), "level": 0.95,
                                "method": "source-stratified configuration-cluster percentile bootstrap",
                                "axis": "configuration-cluster sampling"}
                interval.update(draws=draws, usable_draws=draws, discarded_draws=0,
                                seed=seed, rng_label=label, rng_seed=rng_seed,
                                rng_engine="PCG64", quantile_method="linear",
                                zero_denominator=0,
                                empirical_degenerate=interval["low"] == interval["high"])
                q["interval"] = interval
    return out


def _json_comment(name: str, value: dict) -> str:
    return "% " + name + " " + json.dumps(value, sort_keys=True, separators=(",", ":"))


def _cell(q: dict) -> str:
    point = f"{q['point']:.3f}"
    interval = q["interval"]
    if interval is None:
        return r"\shortstack{%s$^{\mathrm{I}}$\\{\tiny identity}}" % point
    return (r"\shortstack{%s\\{\fontsize{6}{7}\selectfont $[%.3f,%.3f]$}}"
            % (point, interval["low"], interval["high"]))


def _caption(label: str) -> str:
    common = (
        r"Each bracket is a marginal 95\% percentile interval on that one cell, "
        r"source-stratified over configuration clusters, conditional on the released labels "
        r"and fixed predictions. Each draw carries a configuration's capabilities together "
        r"and pools TP, FP, and FN before computing the metric. "
        r"A source column resamples only that source; overall preserves the six source counts. "
        r"These marginal intervals support no comparison between two cells because they "
        r"cannot recover their pairing; overlap establishes neither equality nor simultaneous coverage. "
        r"New intervals use 10,000 draws, linear quantiles, PCG64, and base seed 20260907. "
        r"Zero denominators give zero. Equal endpoints, including flag-none's $[0,0]$, "
        r"are empirical degenerate intervals. $^{\mathrm{I}}$ marks a construction identity "
        r"with no performance interval: declared minus minimal equals the released excess labels. "
        r"The held-out judge is Llama-3.3-70B, scored on 1182 parsed configurations: "
        r"298 crewai, 215 n8n, 143 mcp, 340 injecagent, 130 sweagent, and 56 synthetic. "
        r"Other rows use all 1187 (219 n8n and 144 mcp). "
        r"The overall column mixes four label processes: joint LLM labels for crewai, n8n, "
        r"and mcp; roster labels for injecagent; declared-minus-used labels for sweagent; "
        r"and injected labels for synthetic. "
    )
    if label == LABELS[0]:
        intro = r"PRE capability-level micro-F1 by configuration source. "
        extra = (r"The judge abstains on five configurations, including an MCP server with "
                 r"622 capabilities and 337 excess labels. ")
    else:
        intro = r"PRE static coverage and capability-level micro precision, recall, and F1. "
        extra = (r"The privilege-escalation, unrequested-impact, and sensitive-access precision "
                 r"intervals retain their committed endpoints, 10,000 draws, and base seed 20260817. "
                 r"Standards identifiers name public categories. Scanner inputs are derived task "
                 r"or role tokens and declared capabilities. ")
    return (r"\caption{" + intro + common + extra
            + r"Generated by \texttt{tools/emit\_pre\_tables.py}; full-precision quantities "
              r"and input hashes accompany the source as comments.}")


def _mapping() -> list[str]:
    rows = (
        ("Excessive permissions / least privilege", "owasp_excess_permissions",
         "OWASP LLM06:2025; CWE-272, CWE-250"),
        ("Excessive functionality", "owasp_excess_functionality", "OWASP LLM06:2025"),
        ("Privilege compromise / escalation", "owasp_privilege_escalation", "CWE-269; OWASP ASI"),
        ("Excessive autonomy (approximation)", "unrequested_high_impact", "OWASP LLM06:2025"),
        ("Sensitive-access exposure surface", "sensitive_access", "OWASP LLM02:2025"),
    )
    width = r"\dimexpr(\linewidth-8pt)/3\relax"
    lines = [r"\begin{tabular}{@{}*{3}{>{\raggedright\arraybackslash}p{" + width + r"}}@{}}",
             r"\toprule", r"Public category & Scanner rule & Standard reference \\", r"\midrule"]
    for category, rule, reference in rows:
        name = rule.replace("_", r"\_\allowbreak{}")
        lines.append(category + r" & \texttt{" + name + "} & " + reference + r" \\")
    return [*lines, r"\bottomrule", r"\end{tabular}", r"\par\medskip"]


def tables(data: dict, values: dict | None = None) -> dict[str, str]:
    values = quantities(data) if values is None else values
    blocks = {}
    for label in LABELS:
        source_table = label == LABELS[0]
        columns = [(source, "f1") for source in (*SOURCES, "overall")] if source_table else [
            ("overall", metric) for metric in METRICS]
        methods = SOURCE_METHODS if source_table else MAIN_METHODS
        begin, end = markers(label)
        lines = [begin, _json_comment("PRE provenance", data["provenance"]),
                 r"\begin{table*}[t]", r"\centering", r"\scriptsize",
                 r"\setlength{\tabcolsep}{2pt}", r"\renewcommand{\arraystretch}{1.15}"]
        if not source_table:
            lines.extend(_mapping())
        name_width = "2.35cm" if source_table else "4.2cm"
        n = len(columns)
        width = rf"\dimexpr(\linewidth-{name_width}-{4*n}pt)/{n}\relax"
        lines.extend([
            r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{" + name_width
            + r"}*{" + str(n) + r"}{>{\centering\arraybackslash}p{" + width + r"}}@{}}",
            r"\toprule",
            "Method & " + " & ".join((*SOURCES, "overall") if source_table
                                     else ("Precision", "Recall", "F1")) + r" \\",
            r"\midrule",
        ])
        for method in methods:
            for source, metric in columns:
                lines.append(_json_comment("PRE quantity", values[method, source, metric]))
            lines.append(NAMES[method] + " & " + " & ".join(
                _cell(values[method, source, metric]) for source, metric in columns) + r" \\")
            lines.append(r"\addlinespace[2pt]")
        lines.extend([r"\bottomrule", r"\end{tabular}", _caption(label),
                      r"\label{" + label + "}", r"\end{table*}", end])
        blocks[label] = "\n".join(lines)
    return blocks


def _marker_span(text: str, marker: str) -> tuple[int, int]:
    matches = list(re.finditer(rf"(?m)^[ \t]*{re.escape(marker)}[ \t]*\r?$", text))
    if len(matches) != 1:
        raise ValueError(f"marker {marker!r} appears {len(matches)} times, expected once")
    match = matches[0]
    return match.start(), match.end()


def check(paper: Path, generated: dict[str, str]) -> int:
    findings = []
    path = paper / APPENDIX
    if not path.is_file():
        findings.append(f"{APPENDIX}: missing required file")
    else:
        whole = path.read_bytes().decode("utf-8")
        bounds = None
        try:
            board, main, end = (_marker_span(whole, marker)[0]
                                for marker in (BOARD_START, MAIN_START, BOARD_END))
            if not board < main < end:
                raise ValueError("PRE section boundaries are out of order")
            bounds = ((board, main), (main, end))
            if r"\iffalse" in whole[board:end]:
                raise ValueError("a TeX conditional hides part of the PRE board section")
        except ValueError as error:
            findings.append(f"{APPENDIX}: {error}")
        for i, label in enumerate(LABELS):
            try:
                start, _ = _marker_span(whole, markers(label)[0])
                stop_start, stop = _marker_span(whole, markers(label)[1])
                if stop_start < start:
                    raise ValueError(f"{label}: end marker precedes begin marker")
                if bounds is not None and not bounds[i][0] < start < stop < bounds[i][1]:
                    raise ValueError(f"{label}: block is outside its required PRE section")
            except ValueError as error:
                findings.append(f"{APPENDIX}: {error}")
                continue
            actual, wanted = whole[start:stop], generated[label]
            if actual != wanted:
                delta = "\n".join(difflib.unified_diff(
                    actual.splitlines(), wanted.splitlines(),
                    fromfile=f"paper/{label}", tofile=f"generated/{label}", lineterm="", n=1))
                if not delta:
                    delta = "line-ending bytes differ (generated blocks use LF)"
                findings.append(f"{APPENDIX}: {label} differs from generated block\n{delta}")
    for finding in findings:
        print("STALE " + finding)
    if findings:
        print(f"{len(findings)} staleness finding(s). Regenerate with: python tools/emit_pre_tables.py")
        return 1
    print("paper is current: tab:pre-source and tab:pre-main (100 intervals, 10 construction identities)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check both generated PRE blocks")
    parser.add_argument("--paper", default=os.environ.get("CATCHBENCH_PAPER_DIR"),
                        help="paper directory (or set CATCHBENCH_PAPER_DIR)")
    args = parser.parse_args(argv)
    if args.check and not args.paper:
        parser.error("--check needs --paper <dir> or CATCHBENCH_PAPER_DIR")
    generated = tables(load())
    if args.check:
        return check(Path(args.paper), generated)
    sys.stdout.buffer.write(("\n\n".join(generated.values()) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
