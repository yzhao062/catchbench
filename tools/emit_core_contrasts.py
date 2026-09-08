"""Emit the four structural contrasts from detection_audit_results.json.

Each arm is the mean of its five saved out-of-fold score vectors. The effect is
ROC-AUC(A) minus ROC-AUC(B), computed after averaging scores, with no fitting,
corpus loading, or fold redraw. Digests bind the selected inputs to the released
record so an input change fails loudly instead of moving a frozen measurement.
SWE-Gym retains paired run-level DeLong intervals. The recovered identity table
catchbench-run-identity-table-2026-09-06.json verified 376 distinct tasks and
row-aligned labels; it is provenance, not an additional runtime dependency.

tau-bench uses the frozen domain-stratified task bootstrap: 50 airline and 115
retail clusters, all four model rows together, 10,000 draws, single-class draws
discarded, and linear 2.5/97.5 percentiles. The shared statistical_tests helpers
perform only scoring arithmetic here. The linear contrast reuses its registry
RNG label; the matched spline has its own label and adds no registry row.
Source selectors, full-precision results, seeds, and usable counts are emitted
as comments. Run-level diagnostics remain available through diagnostics().

Usage::

    python tools/emit_core_contrasts.py
    python tools/emit_core_contrasts.py --check --paper <paper directory>

The default stdout is one UTF-8, LF-delimited appendix table between sentinels.
--check also accepts CATCHBENCH_PAPER_DIR, compares bytes including whitespace
and line endings, and exits 1 with a readable delta on drift. No file is written.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from statistical_tests import (  # noqa: E402
    _rng_for, _task_clustered_auc_bootstrap, _tau_clustering, paired_delong,
)

RECORD = Path(__file__).with_name("detection_audit_results.json")
BASE_SEED = 20260907
DRAWS = 10000
ARMS = (
    ("Linear", "auditable (size+deps)", "size (flat)", "auditable.vs.size"),
    ("Matched spline", "size-spline + linear-deps", "size (spline)",
     "size_spline_linear_deps.vs.size_spline"),
)
_INPUT_DIGESTS = {
    "swegym": "7f88d94b051f6c4d686b500ace925b81ebbb1e59f0bcf4fc1ac8d2c53c0863f6",
    "tau": "7877884793e3d6c62d4fcbe20212e39b9e18aa9b3276531f10663da4d2255eb0",
}
_APPENDIX = "09_appendix.tex"
_BEGIN = ("% BEGIN GENERATED tab:core-contrasts -- regenerate with: "
          "python tools/emit_core_contrasts.py")
_END = "% END GENERATED tab:core-contrasts"


def load() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _inputs(corpus: str, record: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    payload = {k: record[k] for k in
               ("labels", "seeds", "n_runs", "n_failed", "run_identities")}
    payload["scores"] = {arm: record["arms"][arm]["oof_proba"]
                         for _, a, b, _ in ARMS for arm in (a, b)}
    digest = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if digest != _INPUT_DIGESTS[corpus]:
        raise ValueError(f"{corpus}: frozen prediction/identity inputs changed "
                         f"(SHA-256 {digest}); cannot reproduce the released contrasts")
    y = np.asarray(record["labels"], dtype=int)
    n = {"swegym": 376, "tau": 660}[corpus]
    if y.shape != (n,) or set(y) != {0, 1} or record["seeds"] != list(range(5)):
        raise ValueError(f"{corpus}: expected {n} binary labels and five OOF seeds")
    scores = {}
    for arm, values in payload["scores"].items():
        array = np.asarray(values, dtype=float)
        if array.shape != (5, n) or not np.isfinite(array).all():
            raise ValueError(f"{corpus}/{arm}: expected five finite OOF score vectors")
        scores[arm] = array.mean(axis=0)
    return y, scores


def _tau_clusters(record: dict) -> dict:
    identities = record["run_identities"]
    if not identities["available"]:
        raise ValueError("tau: task identities are unavailable")
    for key in ("task_cluster", "domain", "model", "run_key"):
        if len(identities[key]) != 660:
            raise ValueError(f"tau: expected 660 aligned {key} entries")
    clustering = _tau_clustering(
        record["labels"], identities["task_cluster"], identities["domain"])
    blocks, strata = clustering["blocks"], clustering["strata"]
    if blocks.shape != (165, 4) or {k: len(v) for k, v in strata.items()} != {
            "airline": 50, "retail": 115}:
        raise ValueError("tau: expected 165 four-row tasks, 50 airline and 115 retail")
    roster = set(identities["model"])
    if (len(roster) != 4 or len(set(identities["run_key"])) != 660
            or any({identities["model"][i] for i in block} != roster for block in blocks)):
        raise ValueError("tau: each task must contain each of the four models exactly once")
    if (clustering["record"]["seed"] != BASE_SEED
            or clustering["record"]["draws"] != DRAWS):
        raise ValueError("tau: shared bootstrap constants differ from the frozen specification")
    return clustering


def diagnostics(record: dict) -> list[dict]:
    """Derive four ordered rows, keeping diagnostic DeLong output off the printed table."""
    inputs = {c: _inputs(c, record["corpora"][c]) for c in ("swegym", "tau")}
    clustering = _tau_clusters(record["corpora"]["tau"])
    rows = []
    for specification, a, b, suffix in ARMS:
        for corpus, name in (("swegym", "SWE-Gym"), ("tau", "tau-bench")):
            y, scores = inputs[corpus]
            result = paired_delong(y, scores[a], scores[b])
            row = {
                "corpus": corpus, "name": name, "specification": specification,
                "a_name": a, "b_name": b,
                "a_selector": f"$.corpora.{corpus}.arms[{json.dumps(a)}].oof_proba",
                "b_selector": f"$.corpora.{corpus}.arms[{json.dumps(b)}].oof_proba",
                "labels_selector": f"$.corpora.{corpus}.labels",
                "auc_a": result["auc_a"], "auc_b": result["auc_b"],
                "point": result["difference"],
                "low": result["interval_95"][0], "high": result["interval_95"][1],
                "method": "paired run-level DeLong", "n_runs": len(y),
                "reproduction_diagnostic": {
                    "role": "run-level diagnostic under the independence assumption",
                    **result,
                },
            }
            if corpus == "tau":
                label = f"det.tau.{suffix}"
                rng = _rng_for(label, BASE_SEED)
                rng_seed = int(rng.bit_generator.seed_seq.entropy)
                boot = _task_clustered_auc_bootstrap(
                    y, clustering, scores[a], scores[b], None, DRAWS, rng)
                if not np.isclose(boot["point"], row["point"], rtol=0, atol=1e-12):
                    raise ValueError(f"{label}: clustered scoring changed the point estimate")
                row.update({
                    "low": boot["interval_95"][0], "high": boot["interval_95"][1],
                    "method": boot["method"], "clustered_interval": boot,
                    "rng_label": label, "rng_seed": rng_seed,
                })
            rows.append(row)
    return rows


def table(record: dict) -> str:
    """One appendix float, including source and bootstrap metadata comments."""
    rows = diagnostics(record)
    lines = [
        _BEGIN,
        "% Source: tools/detection_audit_results.json; scores averaged over seeds 0,1,2,3,4.",
        "% SWE-Gym identity evidence: catchbench-run-identity-table-2026-09-06.json; "
        "376 rows, 376 distinct task ids; labels verified row for row.",
        *(f"% Frozen {c} input SHA-256: {digest}" for c, digest in _INPUT_DIGESTS.items()),
        "% tau: unit=task_cluster; strata=domain; airline=50; retail=115; rows_per_task=4; "
        f"draws={DRAWS}; percentiles=2.5,97.5; quantile=linear; RNG=PCG64; base_seed={BASE_SEED}.",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Four exploratory structural contrasts on the released fixed-prediction "
        r"scoring functional: ROC-AUC of the mean of the five out-of-fold score vectors "
        r"for arm A, minus the same for arm B.",
        r"This differs from the mean fold-score increments in the existing specification "
        r"comparison; it also differs from averaging five pooled per-seed AUC differences.",
        r"Linear compares \texttt{auditable (size+deps)} with \texttt{size (flat)}; "
        r"matched spline compares size-spline plus linear-dependency terms with size-only spline.",
        r"SWE-Gym retains paired run-level DeLong intervals: the recovered identity table "
        r"verifies 376 runs with 376 distinct task ids.",
        r"For tau-bench, 660 runs are 165 tasks attempted by four agent models. "
        r"Its task-clustered bootstrap resamples 50 airline and 115 retail tasks with "
        r"replacement within domain, carrying all four model rows, labels, and both arms' "
        r"scores together. Intervals are the 2.5 and 97.5 percentiles of 10,000 draws; "
        r"single-class draws are discarded.",
        r"Usable draws: linear %d; matched spline %d. NumPy PCG64 uses base seed %d "
        r"with the recorded SHA-256 contrast labels. The last interval digit carries "
        r"Monte Carlo noise of about 0.001."
        % (rows[1]["clustered_interval"]["usable_replicates"],
           rows[3]["clustered_interval"]["usable_replicates"], BASE_SEED),
        r"Intervals condition on the released fitted predictions and fixed model roster; "
        r"they omit training, tuning, and fold-redraw uncertainty and do not establish "
        r"performance on unseen tasks, models, or domains.",
        r"Generated by \texttt{tools/emit\_core\_contrasts.py}.}",
        r"\label{tab:core-contrasts}",
        r"\begin{tabular}{@{}llrr@{}}",
        r"\toprule",
        r"Corpus & Specification & AUC difference & 95\% interval \\",
        r"\midrule",
    ]
    for row in rows:
        lines.extend([
            f"% A: {row['a_selector']}; B: {row['b_selector']}; labels: {row['labels_selector']}",
            f"% {row['name']} / {row['specification']}: auc_a={row['auc_a']:.12f}; "
            f"auc_b={row['auc_b']:.12f}; effect={row['point']:+.12f}; "
            f"interval=[{row['low']:+.12f}, {row['high']:+.12f}]; method={row['method']}",
        ])
        if row["corpus"] == "tau":
            boot = row["clustered_interval"]
            lines.append(f"% rng_label={row['rng_label']}; rng_seed={row['rng_seed']}; "
                         f"usable={boot['usable_replicates']}; "
                         f"discarded={boot['discarded_single_class_draws']}")
        lines.append(r"%s & %s & $%+.3f$ & $[%+.3f, %+.3f]$ \\"
                     % (row["name"], row["specification"], row["point"],
                        row["low"], row["high"]))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", _END]
    return "\n".join(lines)


def check(paper: Path, generated: str) -> int:
    try:
        raw = (paper / _APPENDIX).read_bytes()
    except OSError as exc:
        print(f"STALE {_APPENDIX}: cannot read appendix: {exc}")
        return 1
    markers = []
    for marker in (_BEGIN, _END):
        found = list(re.finditer(rb"(?m)^" + re.escape(marker.encode("utf-8")) + rb"\r?$", raw))
        if len(found) != 1:
            print(f"STALE {_APPENDIX}: core-contrasts marker {marker!r} appears "
                  f"{len(found)} times, expected once")
            return 1
        markers.append(found[0])
    begin, end = markers
    if end.start() < begin.start():
        print(f"STALE {_APPENDIX}: core-contrasts end marker precedes its begin marker")
        return 1
    actual = raw[begin.start():end.start() + len(_END)]
    wanted = generated.encode("utf-8")
    if actual != wanted:
        print(f"STALE {_APPENDIX}: core-contrasts table differs from the generated block")
        for index, (got, want) in enumerate(zip(actual.splitlines(keepends=True),
                                               wanted.splitlines(keepends=True)), 1):
            if got != want:
                print(f"first difference at block line {index}: paper={got!r}; generated={want!r}")
                break
        print("\n".join(difflib.unified_diff(
            actual.decode("utf-8", errors="replace").splitlines(), generated.splitlines(),
            fromfile=f"paper/{_APPENDIX}", tofile="generated/tab:core-contrasts", lineterm="")))
        print("Regenerate with: python tools/emit_core_contrasts.py")
        return 1
    print("paper is current: tab:core-contrasts (four structural effects and intervals)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="check the marked appendix block instead of printing it")
    parser.add_argument("--paper", default=os.environ.get("CATCHBENCH_PAPER_DIR"),
                        help="paper source directory (or set CATCHBENCH_PAPER_DIR)")
    args = parser.parse_args()
    if args.check and not args.paper:
        parser.error("--check needs --paper <dir> or CATCHBENCH_PAPER_DIR")
    try:
        generated = table(load())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"ERROR: core-contrasts record: {exc}", file=sys.stderr)
        return 1
    if args.check:
        return check(Path(args.paper), generated)
    sys.stdout.buffer.write((generated + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
