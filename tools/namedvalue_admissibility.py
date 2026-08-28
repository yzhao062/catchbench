"""Gold v2 fixed-margin diagnostic on the named-value substrate.

The scored board owns the evaluator so the command-line task, this standalone tool, and the tie
contract tests cannot drift. The diagnostic checks two fixed criteria across five injection seeds:

  process controls: the injection-site Top-1 minus matched floor confidence interval must fit
                    inside [-0.05, +0.05], and the clean-versus-injected run-AUC interval must fit
                    inside [0.45, 0.55].
  semantic oracles: reported separately as checks that the programmed fault can be recognized.

Passing these observed margins does not pass the full no-artifact-leakage bar. Positive-control
power is demonstrated for only three of six controls on Top-1 and for no run-AUC path.

Run:  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python tools/namedvalue_admissibility.py
"""
import os
import statistics as st
import sys
from collections import Counter, defaultdict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from catchbench import namedvalue as nv  # noqa: E402

T975_4 = 2.776
SEEDS = nv.GOLD_V2_SEEDS
TOP1_MARGIN = nv.GOLD_V2_TOP1_MARGIN
AUC_BAND = nv.GOLD_V2_AUC_BAND
KIND = os.environ.get("NV_KIND", "dropped")

# Public aliases retained for the control-power tool and contract tests. The implementations live
# with the scored task, so those checks exercise the same evaluator that run.py uses.
corpus_stats = nv.corpus_stats
ctl_format_outlier = nv._ctl_format_outlier
ctl_schema_shape = nv._ctl_schema_shape
ctl_position_prior = nv._ctl_position_prior
ctl_field_prior = nv._ctl_field_prior
ctl_tool_prior = nv._ctl_tool_prior
ctl_edit_distance = nv._ctl_edit_distance
orc_superseded = nv._orc_superseded
orc_provenance = nv._orc_provenance
eligible_pool = nv._eligible_pool
pools_for = nv._pools_for
_tie_aware_top1 = nv._tie_aware_top1
top1_and_floor = nv._top1_and_floor
run_auc = nv._run_auc

CONTROLS = list(nv.GOLD_V2_CONTROLS)
ORACLES = list(nv.GOLD_V2_ORACLES)


def ci(values):
    if len(values) < 2:
        return (values[0], 0.0) if values else (0.0, 0.0)
    return st.mean(values), T975_4 * st.stdev(values) / (len(values) ** 0.5)


def main():
    print("loading tau-bench and building named-value graphs...", flush=True)
    graphs = nv.load_graphs()
    stats = corpus_stats(graphs)
    print(f"named-value substrate: {len(graphs)} runs\n", flush=True)

    per_seed = defaultdict(lambda: {"gap": [], "auc": []})
    kinds = Counter()
    sizes = []
    for seed in SEEDS:
        pairs = nv.build_single_kind_corpus(graphs, KIND, seed=seed)
        pools = pools_for(pairs)
        sizes.append(len(pairs))
        print(f"  seed {seed}: {len(pairs)} pairs built", flush=True)
        if seed == SEEDS[0]:
            kinds.update(pair.label.kind for pair in pairs)
        for name, score_fn in CONTROLS + ORACLES:
            top1, floor = top1_and_floor(pairs, score_fn, stats, pools)
            print(f"    {name} scored", flush=True)
            per_seed[name]["gap"].append(top1 - floor)
            per_seed[name]["auc"].append(run_auc(pairs, score_fn, stats))

    print(f"paired corpus: {sizes[0]} pairs (seed {SEEDS[0]}); "
          f"kinds {dict(kinds)}; sizes across seeds {sizes}\n")
    print(f"FIXED-THRESHOLD criterion: Top-1 gap CI inside "
          f"[{-TOP1_MARGIN:+.2f}, {TOP1_MARGIN:+.2f}] AND run-level AUC CI inside "
          f"[{AUC_BAND[0]:.2f}, {AUC_BAND[1]:.2f}], {len(SEEDS)} seeds\n")

    print(f"{'control (process artifact)':28s}{'Top-1 - floor':>22s}{'run AUC':>20s}  verdict")
    all_pass = True
    for name, _score_fn in CONTROLS:
        gap_mean, gap_half = ci(per_seed[name]["gap"])
        auc_mean, auc_half = ci(per_seed[name]["auc"])
        passes = (
            abs(gap_mean) + gap_half <= TOP1_MARGIN
            and AUC_BAND[0] <= auc_mean - auc_half
            and auc_mean + auc_half <= AUC_BAND[1]
        )
        all_pass &= passes
        print(f"  {name:26s}{f'{gap_mean:+.3f}+/-{gap_half:.3f}':>22s}"
              f"{f'{auc_mean:.3f}+/-{auc_half:.3f}':>20s}  "
              f"{'PASS' if passes else 'FAIL'}")

    print()
    print(f"{'oracle (fault-definitional)':28s}{'Top-1 - floor':>22s}{'run AUC':>20s}  role")
    for name, _score_fn in ORACLES:
        gap_mean, gap_half = ci(per_seed[name]["gap"])
        auc_mean, auc_half = ci(per_seed[name]["auc"])
        role = "matching oracle" if name == "provenance" else "other-fault oracle"
        print(f"  {name:26s}{f'{gap_mean:+.3f}+/-{gap_half:.3f}':>22s}"
              f"{f'{auc_mean:.3f}+/-{auc_half:.3f}':>20s}  {role}")

    print()
    if all_pass:
        print("FIXED-MARGIN PASS - every observed process-control interval is inside its declared "
              "band.")
        print("FULL NO-ARTIFACT-LEAKAGE BAR UNDETERMINED - only three of six controls have a "
              "Top-1 power check, and no run-AUC path has one.")
    else:
        print("FIXED-MARGIN FAIL - at least one process control separates the classes; the "
              "injector or substrate needs work before this board is evidence.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
