"""Run the CatchBench seed board(s) and print the leaderboard.

    python run.py

PRE, POST, and LIVE pillars run here. PRE: an over-privilege harness board flagging over-granted
capabilities across six config corpora, with a per-source F1 breakdown. POST: a
localization board (Who&When, human labels), detection boards
(SWE-Gym and tau-bench outcome labels), and a Gold board (faults injected into clean SWE-Gym runs,
with injection-site labels: the benchmark's own data contribution). Gold v2 scores the named-value
substrate over tau-bench as a separate artifact diagnostic. LIVE: streaming early-warning
(SWE-Gym and tau-bench prefixes) and online stale-state detection over the Gold injection. The POST
and LIVE boards download their corpora from the
Hugging Face Hub on first run; the PyGOD baseline needs torch + pygod + a pyg-lib / torch-sparse
backend.
"""
import argparse

from catchbench.core import RunPipeline  # noqa: E402
from catchbench.corpora import (  # noqa: E402
    revision_header,
    verify_corpus_heads,
    verify_pinned_fetches,
)
from catchbench.pre import PreOverPrivilege, pre_methods, pre_source_breakdown  # noqa: E402

# The POST, LIVE, and Gold modules bind the GRADE checkout bridge at import time. Importing them here
# would make even the PRE board unrunnable without GRADE, a torch stack, and 320 MB of corpora, which
# is what forced a newcomer's first command to be the nine-minute one. PRE scores offline from
# committed records in well under a second, so those imports live inside the branch that needs them.


def _pre_board() -> None:
    """Score the PRE board alone: offline, no GRADE bridge, no download, no model call.

    This is the fast path a reader should meet first. It reproduces the PRE block of the full board
    exactly, because it builds the same task with the same methods; the only thing it skips is the
    corpus preflight, which PRE does not use.
    """
    task = PreOverPrivilege()
    methods = pre_methods()
    print("CatchBench :: PRE board (offline; no GRADE bridge, no corpus download)")
    print()
    print(task.corpus_line())
    rows = RunPipeline([task], methods).run()
    print(RunPipeline.leaderboard(rows))
    print(pre_source_breakdown(task, methods))
    print(
        "\nReading: flag_all is the floor a method must beat to earn its false alarms, and the "
        "per-source columns are the result. The pooled row mixes four label processes, so read it "
        "last. Run 'catchbench' (or 'python run.py' from a checkout) for the POST, LIVE, and "
        "Gold boards; those need the GRADE checkout bridge and download their corpora on first "
        "use."
    )


def _gold_v2_board() -> None:
    """Score the named-value Gold v2 board and its five-seed artifact diagnostic alone."""
    from catchbench.namedvalue import (
        GoldNamedValue,
        gold_v2_breakdown,
        gold_v2_methods,
    )

    revisions = verify_corpus_heads(names={"tau-bench"})
    task = GoldNamedValue()
    methods = gold_v2_methods()
    corpus_line = task.corpus_line()
    verify_pinned_fetches(names={"tau-bench"})

    print("CatchBench :: Gold v2 named-value board")
    print(revision_header(revisions))
    print()
    print(corpus_line)
    rows = RunPipeline([task], methods).run()
    print(RunPipeline.leaderboard(rows))
    print(gold_v2_breakdown(task, methods))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CatchBench boards.")
    parser.add_argument(
        "--task", choices=["all", "pre", "gold-v2"], default="all",
        help="'pre' scores the offline PRE board only, in about a second, with no GRADE checkout "
             "and no corpus download. 'gold-v2' scores the tau-bench named-value board only. "
             "'all' (the default) scores every board.")
    args = parser.parse_args()
    if args.task == "pre":
        return _pre_board()
    if args.task == "gold-v2":
        return _gold_v2_board()

    from catchbench.detection import PostDetection, post_detection_methods
    from catchbench.gold import (
        GoldAttribution,
        GoldLocalization,
        gold_attribution_methods,
        gold_attribution_robustness,
        gold_breakdown,
        gold_localization_methods,
        gold_matched_breakdown,
        gold_report,
        gold_seed_robustness,
    )
    from catchbench.live import (
        LiveStaleState,
        LiveStreaming,
        live_breakdown,
        live_stale_breakdown,
        live_stale_methods,
        live_stale_robustness,
        live_streaming_methods,
    )
    from catchbench.llm_judge import discovered_llm_judge_methods  # cached LLM-judge panel
    from catchbench.namedvalue import (
        GoldNamedValue,
        gold_v2_breakdown,
        gold_v2_methods,
    )
    from catchbench.post import PostLocalization, post_localization_methods
    from catchbench.pyod_extra import pyod_extra_methods  # more PyOD tabular detectors
    from catchbench.pygod_extra import pygod_extra_methods  # more PyGOD graph detectors

    revisions = verify_corpus_heads()
    tasks = [PreOverPrivilege(), PostLocalization(), PostDetection("swegym"), PostDetection("tau"),
             GoldLocalization(), GoldAttribution(), GoldNamedValue(), LiveStreaming("swegym"),
             LiveStreaming("tau"), LiveStaleState()]
    methods = (pre_methods() + post_localization_methods() + discovered_llm_judge_methods()
               + post_detection_methods()
               + pyod_extra_methods() + pygod_extra_methods()
               + gold_localization_methods() + gold_attribution_methods()
               + gold_v2_methods() + live_streaming_methods() + live_stale_methods())

    corpus_lines = [task.corpus_line() for task in tasks if hasattr(task, "corpus_line")]
    verify_pinned_fetches()

    print("CatchBench :: PRE + POST + LIVE board(s)")
    print(revision_header(revisions))
    print()
    for line in corpus_lines:
        print(line)

    rows = RunPipeline(tasks, methods).run()
    print(RunPipeline.leaderboard(rows))

    pre = next((t for t in tasks if isinstance(t, PreOverPrivilege)), None)
    if pre is not None:  # PRE headline is the per-source F1, since the pooled row mixes label sources
        print(pre_source_breakdown(pre, pre_methods()))

    gold = next((t for t in tasks if isinstance(t, GoldLocalization)), None)
    if gold is not None:  # Gold's headline is the per-fault breakdown, not the aggregate row
        print(gold_breakdown(gold, gold_localization_methods()))
        print(gold_matched_breakdown(gold, gold_localization_methods()))
        print()
        print(gold_report(gold))
        print(gold_seed_robustness(gold_localization_methods()))  # stability across injection seeds

    if any(isinstance(t, GoldAttribution) for t in tasks):  # cause-attribution stability across seeds
        print(gold_attribution_robustness(gold_attribution_methods()))

    gold_v2 = next((t for t in tasks if isinstance(t, GoldNamedValue)), None)
    if gold_v2 is not None:
        print(gold_v2_breakdown(gold_v2, gold_v2_methods()))

    for live in (t for t in tasks if isinstance(t, LiveStreaming)):  # early-warning curve per corpus
        print(live_breakdown(live, live_streaming_methods()))

    stale = next((t for t in tasks if isinstance(t, LiveStaleState)), None)
    if stale is not None:  # online detection: TPR with the realized FPR beside it
        print(live_stale_breakdown(stale, live_stale_methods()))
        print(live_stale_robustness(live_stale_methods()))  # stability across injection seeds

    print(
        "\nReading:"
        "\n- Localization (Who&When): the all-at-once LLM-judge panel spans 0.127 to 0.452 "
        "Top-1. Eight models form a band from 0.333 upward that 126 runs do not separate, "
        "so the ordering within that band is unresolved. Mistral-Small and Nova-Micro have "
        "lower displayed point estimates than the 0.159 position prior, but the registered "
        "tests leave both comparisons unresolved. Among methods that use no LLM, auditable's "
        "blast share coincides with position because Who&When assumes full-context dependencies, "
        "and GRADE's supervised exec-rank method separates from position on Top-3 while its "
        "Top-1 comparison remains unresolved. A long-range gold-edge corpus is the next data lever."
        "\n- Detection (SWE-Gym, tau-bench): the question is whether the dependency structure "
        "predicts failure beyond run size and counts; compare 'auditable (size+deps)' against "
        "'size (flat)', which reads size and event counts. "
        "The registered contrast separates on SWE-Gym; on tau-bench the point estimates run in the "
        "same direction, but the paired test leaves that pair unresolved."
        "\n- Unsupervised AD arena (PyOD flat vs PyGOD graph): after the batching repair (see "
        "graph_ad.flat_disconnected), the single-seed PyGOD family spans 0.547 to 0.850 on SWE-Gym "
        "and 0.490 to 0.552 on tau-bench, and DOMINANT stays below the position prior on Who&When "
        "localization. The SWE-Gym maximum does NOT establish an ordering: its five-seed range "
        "overlaps the supervised references, and scoring only between runs of exactly equal node "
        "count establishes no beyond-size advantage on the matchable subset "
        "(tools/pygod_seed_stability.py). No off-the-shelf detector "
        "establishes a task-relevant board lead, and neither does the task-aware structural method "
        "against the better ones: on SWE-Gym its paired tests against ECOD and against GUARDIAN "
        "both fail to separate (Holm p=0.404 and p=0.376), and failing to separate is not evidence "
        "that they are equal. G-Safeguard appears here as the supervised graph comparator (0.828 "
        "displayed, 0.824 +/- 0.007 over five cross-validation seeds)."
        "\n- Gold (injected dependency faults): READ AS MECHANISM DIAGNOSTICS, per fault kind. "
        "The registered full-pool family establishes max-span above the stale-state analytic floor "
        "(0.703 vs 0.029) and has-dep below the dropped-grounding analytic floor (0.005 vs 0.035). "
        "It declares no other method-versus-floor contrast in either pool. The matched-pool stale "
        "cells display has-dep 0.350, degree 0.394, and max-span 0.805 against the 0.350 floor, but "
        "those displayed comparisons are untested. A broken-predecessor baseline uniquely ranks all "
        "82 stale-state and all 106 dropped-grounding targets Top-1 while flagging 0 of 188 clean "
        "runs (tools/gold_artifact_diagnostic.py), so the file-level substrate fails the no-artifact-"
        "leakage bar and its scores remain mechanism evidence. Gold v2 scores the named-value "
        "substrate separately, but its no-artifact-leakage item remains undetermined because "
        "positive-control power coverage is incomplete. See "
        "gold_breakdown, gold_matched_breakdown, gold_report, and gold_seed_robustness below."
        "\n- Gold attribution (cause): given a faulty run, is the cause stale-state or "
        "dropped-grounding? Paired design (the same run is injected both ways, so the label is the "
        "fault, not the run, no eligibility leak). The two faults leave opposite traces: a stale read "
        "lengthens the max dependency span (ROC-AUC 0.675 for stale), dropped grounding removes an edge "
        "(edge-count 0.566), against a 0.498 random floor. The structure separates the two causes, each "
        "feature keyed to one mechanism, completing the POST localization / prediction / attribution "
        "triad."
        "\n- Gold v2 (named-value dropped grounding): every score is a displayed diagnostic cell; "
        "the registry currently declares no Gold v2 contrast. The six process controls meet the "
        "fixed Top-1 and run-AUC margins, but this is not a no-artifact-leakage PASS. Positive-control "
        "power covers only three controls on Top-1 and no run-AUC path, so the bar item remains "
        "undetermined. The stale-state arm is not scored because tau-bench affords only 16 sites in "
        "6 runs. See gold_v2_breakdown below."
        "\n- LIVE streaming (early warning): can a method separate failing from resolved runs before "
        "the trace is complete? On SWE-Gym, the registered 25% contrast separates the dependency-"
        "structure block from the flat size-and-counts baseline. The 20-cell SWE-Gym bar family "
        "is exploratory: it was added after these scores were examined and needs fresh data to "
        "confirm. It is two-sided and resolves 9 cells. It places full above 0.70 at every prefix "
        "and auditable above it at 75% and 100%, and it places the online dep-span scalar below "
        "the bar at 25%, 50%, and 75%. Auditable at the two early prefixes, all four ECOD cells, "
        "and the rest are unresolved. Random is an untested reference. On "
        "tau-bench, the same two-sided family establishes all five nonrandom entrants below "
        "0.70 at 25%, 50%, and 75%; at 100%, size, ECOD, and dep-span remain below the bar while "
        "full and auditable are unresolved. The strict per-run dependency-span scalar displays 0.36 "
        "at the first SWE-Gym prefix and is length-confounded. The 100% column is the POST-style "
        "detection check on the LIVE-filtered population; the current SWE-Gym and tau-bench LIVE "
        "and POST populations coincide rather than doing so by construction. See live_breakdown below."
        "\n- LIVE stale-state (online detection): the SAME Gold stale-state injection, but detected "
        "online at a fixed false-positive rate instead of localized post-hoc. At realized false-positive "
        "rates of ~6% and ~11%, the causal span z-score displays true-positive rates of ~6% and ~11%. "
        "Raw span displays ~12% and ~16%. Across five injection seeds, the corresponding means are "
        "0.054 and 0.124 for the z-score and 0.098 and 0.151 for raw span. At the displayed 5% target, "
        "the dependency-count control and z-score each display 0.061, while raw span displays 0.122. "
        "No contrast is declared among these methods, so the cells are point estimates only. They "
        "support no claim about the effect of per-run normalization. These are displayed cells from "
        "82 paired runs. The same clean runs calibrate and report each empirical threshold, and the "
        "five injection seeds reuse those runs rather than supplying 410 independent observations. "
        "No method ordering is registered. The ~0.703 Gold value scores post-hoc "
        "within-run localization, a different decision, so it is context rather than a cross-state "
        "effect estimate."
    )


if __name__ == "__main__":
    main()
