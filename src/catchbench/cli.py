"""Run the CatchBench seed board(s) and print the leaderboard.

    python run.py

PRE, POST, and LIVE pillars run here. PRE: an over-privilege harness board flagging over-granted
capabilities across six config corpora, with a per-source F1 breakdown. POST: a
localization board (Who&When, human labels), detection boards
(SWE-Gym and tau-bench outcome labels), and a Gold board (faults injected into clean SWE-Gym runs,
with injection-site labels: the benchmark's own data contribution). LIVE: streaming early-warning
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CatchBench boards.")
    parser.add_argument(
        "--task", choices=["all", "pre"], default="all",
        help="'pre' scores the offline PRE board only, in about a second, with no GRADE checkout "
             "and no corpus download. 'all' (the default) scores every board and is unchanged.")
    args = parser.parse_args()
    if args.task == "pre":
        return _pre_board()

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
    from catchbench.post import PostLocalization, post_localization_methods
    from catchbench.pyod_extra import pyod_extra_methods  # more PyOD tabular detectors
    from catchbench.pygod_extra import pygod_extra_methods  # more PyGOD graph detectors

    revisions = verify_corpus_heads()
    tasks = [PreOverPrivilege(), PostLocalization(), PostDetection("swegym"), PostDetection("tau"),
             GoldLocalization(), GoldAttribution(), LiveStreaming("swegym"), LiveStreaming("tau"),
             LiveStaleState()]
    methods = (pre_methods() + post_localization_methods() + discovered_llm_judge_methods()
               + post_detection_methods()
               + pyod_extra_methods() + pygod_extra_methods()
               + gold_localization_methods() + gold_attribution_methods()
               + live_streaming_methods() + live_stale_methods())

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

    for live in (t for t in tasks if isinstance(t, LiveStreaming)):  # early-warning curve per corpus
        print(live_breakdown(live, live_streaming_methods()))

    stale = next((t for t in tasks if isinstance(t, LiveStaleState)), None)
    if stale is not None:  # online detection: TPR with the realized FPR beside it
        print(live_stale_breakdown(stale, live_stale_methods()))
        print(live_stale_robustness(live_stale_methods()))  # stability across injection seeds

    print(
        "\nReading:"
        "\n- Localization (Who&When): the LLM-judge panel is the strongest post-hoc localizer here "
        "(GPT-5.5 0.452 Top-1 from the committed all-at-once cache), the expected result with the "
        "full trace in hand. The panel spans 0.127 to 0.452, but eight of the models sit in one "
        "band from 0.333 up that 126 runs do not separate, so read the band rather than the "
        "ordering inside it; only the smallest models are distinguishable, and they do not improve "
        "on the position prior. Among methods that use no LLM, position is the honest floor, "
        "auditable's blast coincides with it because Who&When assumes full-context dependencies, "
        "and GRADE's supervised exec-rank method localizes beyond the prior on Top-3 (its Top-1 "
        "margin over position does not resolve at this corpus size). A long-range gold-edge "
        "corpus is the next data lever."
        "\n- Detection (SWE-Gym, tau-bench): the question is whether the dependency structure "
        "predicts failure beyond run size; compare 'auditable (size+deps)' against 'size (flat)'. "
        "The lift holds in the same direction on both corpora (large on SWE-Gym, modest on tau)."
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
        "Stale-state (redirect a dependency to an earlier superseded event on the SAME file) is found "
        "by the dependency-span detector (dep-anomaly keyed to max-span, ~0.703 Top-1 stale); "
        "dropped-grounding is not localized by the span/count baselines (~0 Top-1). dep-anomaly is "
        "keyed to the stale-span mechanism (shown beside the max-span control). Leak check, two "
        "levels (gold_matched_breakdown): ranking within the injector's eligible pool holds selection "
        "fixed (stale has-dep 0.350 = floor 0.350, degree 0.394 just above) while the dependency-span "
        "signal clears it (stale max-span 0.805 vs floor 0.350), stable across 5 injection seeds "
        "(gold_seed_robustness). But a broken-predecessor baseline separates BOTH fault kinds "
        "perfectly on this file-level substrate (82/82 stale + 106/106 dropped unique Top-1, 0/188 "
        "clean; tools/gold_artifact_diagnostic.py), so the substrate fails the no-artifact-leakage "
        "bar and Gold scores are mechanism evidence pending a named-value substrate. See "
        "gold_breakdown / gold_matched_breakdown / gold_report / gold_seed_robustness below."
        "\n- Gold attribution (cause): given a faulty run, is the cause stale-state or "
        "dropped-grounding? Paired design (the same run is injected both ways, so the label is the "
        "fault, not the run, no eligibility leak). The two faults leave opposite traces: a stale read "
        "lengthens the max dependency span (ROC-AUC 0.675 for stale), dropped grounding removes an edge "
        "(edge-count 0.566), against a 0.498 random floor. The structure separates the two causes, each "
        "feature keyed to one mechanism, completing the POST localization / prediction / attribution "
        "triad."
        "\n- LIVE streaming (early warning): can a method separate failing from resolved runs before "
        "the trace is complete? Three settings. SUPERVISED CV on the prefix feature layers: on SWE-Gym "
        "the dependency-structure block clears ROC-AUC 0.74 at the 25% prefix (t2d 25%) while run size "
        "never does; on tau-bench it is weak and late, the same domain split the detection board shows. "
        "BATCH-UNSUPERVISED: an off-the-shelf ECOD over the run population's prefix flat vectors also "
        "fires early on SWE-Gym (0.76 at 25%, t2d 25%), so early warning is available without labels. "
        "STRICT PER-RUN ONLINE: a one-scalar mean dependency span from a run's own prefix is "
        "length-confounded and does NOT (0.36 at 25%). So the early signal comes from supervised "
        "structure or batch-unsupervised flat prefix vectors, not from a single online scalar. The "
        "100% column is the POST-style detection check on the LIVE-filtered population. LIVE keeps "
        "runs of >=4 steps and POST keeps >=2, but the current SWE-Gym and tau-bench populations "
        "coincide, so the two agree today rather than by construction. See live_breakdown below."
        "\n- LIVE stale-state (online detection): the SAME Gold stale-state injection, but detected "
        "online at a fixed false-positive rate instead of localized post-hoc. It is HARD: at a realized "
        "~6% FPR the causal span z-score catches only ~6% of stale reads (~11% at ~11% FPR), and the "
        "raw span does a little better (~12% / ~16%), both far below the ~0.703 WITHIN-run localization "
        "on Gold. Spotting one superseded same-file read online, against clean runs' natural long-range "
        "dependencies and without false-alarming, is an open challenge; per-run normalization does not "
        "help here (the raw span edges out the z-score). The hardness is stable across 5 injection "
        "seeds (live_stale_robustness)."
    )


if __name__ == "__main__":
    main()
