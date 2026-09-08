"""Emit tab:protocol from the committed statistical_tests_results.json record.

The record preserves exact marginal Top-1 scalars and n=126 for all ten judges
under three elicitation protocols. Multiplying each scalar by n must recover an
integer hit count; rounded paper cells and paired difference intervals are never
inputs. Procedure BIN draws 10,000 Binomial(n, k/n) counts and takes the linear
2.5 and 97.5 percentiles of their proportions. These marginal intervals are
reproducible offline from the artifact rather than from a console session: no
corpus, GRADE checkout, model call, or assignment of hits to runs is required.

The September 7 per-arm determination freezes base seed 20260907 and the shared
statistical_tests._rng_for rule. Labels identify protocol, method, and metric,
so the all-at-once cells use precisely the streams tab:loc would use. That table
prints no intervals today, because other cells in it need a regeneration run and
the appendix does not mark part of a table. If it ever gains them, the shared
label rule means the duplicated cells will already agree.

Usage::

    python tools/emit_protocol_table.py
    python tools/emit_protocol_table.py --check --paper <paper directory>

Printing emits a UTF-8, LF-delimited LaTeX block between sentinel comments.
--check also accepts CATCHBENCH_PAPER_DIR. It compares the marked block byte
for byte, including whitespace and line endings, and exits 1 with a readable
delta when the paper is stale. Two-line cells keep the three protocols readable
at the paper's page width without scaling the table.
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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from statistical_tests import _rng_for  # noqa: E402

RECORD = Path(__file__).with_name("statistical_tests_results.json")
BASE_SEED = 20260907
DRAWS = 10000
N = 126
PROTOCOLS = ("all_at_once", "step_by_step", "binary_search")
MODELS = (
    ("gpt-5.5", "GPT-5.5"),
    ("claude-opus-4.8", "Claude-Opus-4.8"),
    ("gpt-5.4", "GPT-5.4"),
    ("deepseek-r1", "DeepSeek-R1"),
    ("gemini", "Gemini"),
    ("qwen3-32b", "Qwen3-32B"),
    ("llama-3.3-70b", "Llama-3.3-70B"),
    ("gemma-3-12b", "Gemma-3-12B"),
    ("mistral-small", "Mistral-Small"),
    ("nova-micro", "Nova-Micro"),
)
_APPENDIX = "09_appendix.tex"
_BEGIN = ("% BEGIN GENERATED tab:protocol -- regenerate with: "
          "python tools/emit_protocol_table.py")
_END = "% END GENERATED tab:protocol"


def load() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _count(point: float, selector: str) -> int:
    if not math.isfinite(point) or not 0 <= point <= 1:
        raise ValueError(f"{selector}: Top-1 must be a finite proportion, got {point!r}")
    scaled = point * N
    k = round(scaled)
    if not math.isclose(scaled, k, rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"{selector}: Top-1 * {N} = {scaled!r} is not an integer; "
                         "the cell and record disagree")
    return k


def cells(record: dict) -> list[dict]:
    """Derive all 30 cells, retaining their exact selectors and RNG metadata."""
    out = []
    for model, name in MODELS:
        claims = {}
        for protocol in PROTOCOLS[1:]:
            claim_id = f"loc.protocol.{model}.all.vs.{protocol}"
            matches = [(i, c) for i, c in enumerate(record["claims"])
                       if c["id"] == claim_id]
            if len(matches) != 1:
                raise ValueError(f"expected one claim {claim_id}, found {len(matches)}")
            index, claim = matches[0]
            if (claim["family"] != "localization_protocol_top1"
                    or claim["metric"] != "top1"
                    or claim["estimate"]["a_name"] != model
                    or claim["estimate"]["b_name"] != f"{model}:{protocol}"):
                raise ValueError(f"{claim_id}: incompatible metric or arm identity")
            n = claim["variance_axes"]["run_sampling"]["n"]
            if n != N:
                raise ValueError(f"{claim_id}: expected n={N}, got {n!r}")
            for arm in ("a", "b"):
                _count(float(claim["estimate"][arm]), f"$.claims[{index}].estimate.{arm}")
            claims[protocol] = (index, claim)
        first = claims["step_by_step"][1]["estimate"]["a"]
        second = claims["binary_search"][1]["estimate"]["a"]
        if not math.isclose(first, second, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{model}: duplicate all-at-once scalars disagree")
        for protocol in PROTOCOLS:
            index, claim = claims["step_by_step" if protocol == "all_at_once" else protocol]
            arm = "a" if protocol == "all_at_once" else "b"
            selector = f"$.claims[{index}].estimate.{arm}"
            point = float(claim["estimate"][arm])
            k = _count(point, selector)
            label = f"per-arm/bin/whoandwhen/{protocol}/{model}/top1"
            rng = _rng_for(label, BASE_SEED)
            rng_seed = int(rng.bit_generator.seed_seq.entropy)
            draws = rng.binomial(N, k / N, size=DRAWS) / N
            low, high = np.percentile(draws, [2.5, 97.5], method="linear")
            out.append({
                "model": model, "name": name, "protocol": protocol,
                "point": point, "k": k, "n": N, "selector": selector,
                "n_selector": f"$.claims[{index}].variance_axes.run_sampling.n",
                "low": float(low), "high": float(high),
                "rng_label": label, "rng_seed": rng_seed,
            })
    return out


def table(record: dict) -> str:
    """The complete float, including markers and source/RNG comments."""
    values = cells(record)
    lines = [
        _BEGIN,
        "% Source: tools/statistical_tests_results.json; selectors below are zero-based.",
        "% BIN: marginal Top-1; unit=run; n=126; level=0.95; draws=10000; "
        "usable=10000; discarded=0; quantile=linear; RNG=PCG64; base_seed=20260907.",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{LLM-judge elicitation protocols on Who\&When (126 runs, Top-1), "
        r"adapted from Who\&When \citep{zhang2025whoandwhen}.",
        r"Each cell gives its hit rate above a marginal 95\% interval over the 126 runs, "
        r"conditional on the recorded predictions.",
        r"Procedure BIN uses 10,000 binomial draws with success probability $k/126$, "
        r"then the 2.5 and 97.5 percentiles of the sampled hit rates (linear quantiles).",
        r"The generator uses NumPy PCG64, base seed 20260907, and the released "
        r"SHA-256 label rule for each protocol, model, and metric.",
        r"These marginal intervals support no comparison between two cells: the runs "
        r"are shared, and this construction cannot recover their pairing.",
        r"Appendix~\ref{app:stats} carries each model's two protocol differences with "
        r"their paired intervals.",
        r"All protocols normalize whitespace. All-at-once shows every step in one call "
        r"with a 1500-character content cap per step; its point estimates repeat the Top-1 "
        r"column of Table~\ref{tab:loc}, which prints no interval of its own.",
        r"Step-by-step reveals growing prefixes (1200 characters per visible step); "
        r"binary-search retains every step index while halving the suspect interval "
        r"(900 characters per step).",
        r"Structural baselines appear in Table~\ref{tab:loc}. GPT-oss-20B is omitted "
        r"because Bedrock capacity was unavailable during regeneration of the two alternatives.",
        r"Generated offline from the committed marginal scalars by "
        r"\texttt{tools/emit\_protocol\_table.py}; \texttt{-{}-check} detects paper drift.}",
        r"\label{tab:protocol}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Model & All-at-once & Step-by-step & Binary-search \\",
        r"\midrule",
    ]
    for offset in range(0, len(values), 3):
        row = values[offset:offset + 3]
        for cell in row:
            lines.append(f"% {cell['selector']}; n={cell['n_selector']}; "
                         f"k={cell['k']}; rng_label={cell['rng_label']}; "
                         f"rng_seed={cell['rng_seed']}")
        rendered = [r"\shortstack{%.3f\\{\footnotesize $[%.3f, %.3f]$}}"
                    % (cell["point"], cell["low"], cell["high"]) for cell in row]
        lines.append(row[0]["name"] + " & " + " & ".join(rendered) + r" \\")
        if offset < len(values) - 3:
            lines.append(r"\addlinespace[3pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", _END]
    return "\n".join(lines)


def check(paper: Path, generated: str) -> int:
    appendix = paper / _APPENDIX
    try:
        raw = appendix.read_bytes()
    except OSError as exc:
        print(f"STALE {_APPENDIX}: cannot read appendix: {exc}")
        return 1
    markers = []
    for marker in (_BEGIN, _END):
        found = list(re.finditer(rb"(?m)^" + re.escape(marker.encode("utf-8")) + rb"\r?$", raw))
        if len(found) != 1:
            print(f"STALE {_APPENDIX}: protocol marker {marker!r} appears "
                  f"{len(found)} times, expected once")
            return 1
        markers.append(found[0])
    begin, end = markers
    if end.start() < begin.start():
        print(f"STALE {_APPENDIX}: protocol end marker precedes its begin marker")
        return 1
    actual = raw[begin.start():end.start() + len(_END)]
    wanted = generated.encode("utf-8")
    if actual != wanted:
        print(f"STALE {_APPENDIX}: protocol table differs from the generated block")
        for index, (got, want) in enumerate(zip(actual.splitlines(keepends=True),
                                               wanted.splitlines(keepends=True)), 1):
            if got != want:
                print(f"first difference at block line {index}: paper={got!r}; generated={want!r}")
                break
        delta = difflib.unified_diff(
            actual.decode("utf-8", errors="replace").splitlines(), generated.splitlines(),
            fromfile=f"paper/{_APPENDIX}", tofile="generated/tab:protocol", lineterm="")
        print("\n".join(delta))
        print("Regenerate with: python tools/emit_protocol_table.py")
        return 1
    print("paper is current: tab:protocol (30 marginal Top-1 intervals, n=126)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="check the marked paper block instead of printing it")
    parser.add_argument("--paper", default=os.environ.get("CATCHBENCH_PAPER_DIR"),
                        help="paper source directory (or set CATCHBENCH_PAPER_DIR)")
    args = parser.parse_args()
    if args.check and not args.paper:
        parser.error("--check needs --paper <dir> or CATCHBENCH_PAPER_DIR")
    try:
        generated = table(load())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: protocol record: {exc}", file=sys.stderr)
        return 1
    if args.check:
        return check(Path(args.paper), generated)
    sys.stdout.buffer.write((generated + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
