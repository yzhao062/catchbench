"""Run the AuditableBench seed board(s) and print the leaderboard.

    python run.py

POST and LIVE pillars run here. POST: a localization board (Who&When, human labels), detection boards
(SWE-Gym and tau-bench outcome labels), and a Gold board (faults injected into clean SWE-Gym runs,
with injection-site labels: the benchmark's own data contribution). LIVE: streaming early-warning
(SWE-Gym and tau-bench prefixes) and online stale-state detection over the Gold injection. The PRE
pillar plugs into the same RunPipeline as it lands. The boards download their corpora from the
Hugging Face Hub on first run; the PyGOD baseline needs torch + pygod + a pyg-lib / torch-sparse
backend.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from auditablebench.core import RunPipeline  # noqa: E402
from auditablebench.detection import PostDetection, post_detection_methods  # noqa: E402
from auditablebench.gold import (  # noqa: E402
    GoldLocalization,
    gold_breakdown,
    gold_localization_methods,
    gold_matched_breakdown,
    gold_report,
    gold_seed_robustness,
)
from auditablebench.live import (  # noqa: E402
    LiveStaleState,
    LiveStreaming,
    live_breakdown,
    live_stale_breakdown,
    live_stale_methods,
    live_stale_robustness,
    live_streaming_methods,
)
from auditablebench.post import PostLocalization, post_localization_methods  # noqa: E402


def main() -> None:
    tasks = [PostLocalization(), PostDetection("swegym"), PostDetection("tau"), GoldLocalization(),
             LiveStreaming("swegym"), LiveStreaming("tau"), LiveStaleState()]
    methods = (post_localization_methods() + post_detection_methods()
               + gold_localization_methods() + live_streaming_methods() + live_stale_methods())

    print("AuditableBench :: POST + LIVE board(s)\n")
    for task in tasks:
        if hasattr(task, "corpus_line"):
            print(task.corpus_line())

    rows = RunPipeline(tasks, methods).run()
    print(RunPipeline.leaderboard(rows))

    gold = next((t for t in tasks if isinstance(t, GoldLocalization)), None)
    if gold is not None:  # Gold's headline is the per-fault breakdown, not the aggregate row
        print(gold_breakdown(gold, gold_localization_methods()))
        print(gold_matched_breakdown(gold, gold_localization_methods()))
        print()
        print(gold_report(gold))
        print(gold_seed_robustness(gold_localization_methods()))  # stability across injection seeds

    for live in (t for t in tasks if isinstance(t, LiveStreaming)):  # early-warning curve per corpus
        print(live_breakdown(live, live_streaming_methods()))

    stale = next((t for t in tasks if isinstance(t, LiveStaleState)), None)
    if stale is not None:  # online detection: TPR with the realized FPR beside it
        print(live_stale_breakdown(stale, live_stale_methods()))
        print(live_stale_robustness(live_stale_methods()))  # stability across injection seeds

    print(
        "\nReading:"
        "\n- Localization (Who&When): position is the honest floor; auditable's blast coincides "
        "with it because Who&When assumes full-context dependencies, and GRADE's supervised "
        "execution-structure ranker localizes beyond the prior. A long-range gold-edge corpus is "
        "the next data lever."
        "\n- Detection (SWE-Gym, tau-bench): the question is whether the dependency structure "
        "predicts failure beyond run size; compare 'auditable (structure)' against 'size (flat)'. "
        "The lift holds in the same direction on both corpora (large on SWE-Gym, modest on tau)."
        "\n- Unsupervised AD arena (PyOD flat vs PyGOD graph): off-the-shelf graph anomaly "
        "detection (PyGOD/DOMINANT, the GUARDIAN reconstruction-AD family) sits near or below "
        "random on detection and below the position prior on localization. Reading the typed graph "
        "is not enough; the task-aware structural features are what work. (G-Safeguard is a "
        "supervised attack detector, for the future fault-injection scenarios, not these boards.)"
        "\n- Gold (injected dependency faults): READ PER FAULT KIND, not in aggregate. Stale-state "
        "(redirect a dependency to an earlier superseded event on the SAME file) is found by the "
        "dependency-span detector (dep-anomaly keyed to max-span, ~0.71 Top-1 stale); dropped-grounding "
        "is NOT localized by any baseline yet (~0 Top-1), an open problem. dep-anomaly is keyed to the "
        "stale-span mechanism (shown beside the max-span control). Leak check (gold_matched_breakdown): "
        "ranking only within the injector's eligible pool, the eligibility baselines sit at the matched "
        "random floor (stale has-dep 0.350 = floor 0.350, degree 0.394 just above) while the "
        "dependency-span signal clears it by far (stale max-span 0.805 vs floor 0.350), so the "
        "stale-state lift is leakage-controlled rather than a selection artifact. The stale-state "
        "detection and the leak-controlled max-span signal are stable across 5 injection seeds "
        "(gold_seed_robustness). See gold_breakdown / gold_matched_breakdown / gold_report / "
        "gold_seed_robustness below."
        "\n- LIVE streaming (early warning): can a method separate failing from resolved runs before "
        "the trace is complete? Three settings. SUPERVISED CV on the prefix feature layers: on SWE-Gym "
        "the dependency-structure block clears ROC-AUC 0.74 at the 25% prefix (t2d 25%) while run size "
        "never does; on tau-bench it is weak and late, the same domain split the detection board shows. "
        "BATCH-UNSUPERVISED: an off-the-shelf ECOD over the run population's prefix flat vectors also "
        "fires early on SWE-Gym (0.76 at 25%, t2d 25%), so early warning is available without labels. "
        "STRICT PER-RUN ONLINE: a one-scalar mean dependency span from a run's own prefix is "
        "length-confounded and does NOT (0.36 at 25%). So the early signal comes from supervised "
        "structure or batch-unsupervised flat prefix vectors, not from a single online scalar. The "
        "100% column is the POST-style detection check on the LIVE-filtered population (exact on "
        "SWE-Gym; off by one run on tau). See live_breakdown below."
        "\n- LIVE stale-state (online detection): the SAME Gold stale-state injection, but detected "
        "online at a fixed false-positive rate instead of localized post-hoc. It is HARD: at a realized "
        "~6% FPR the causal span z-score catches only ~6% of stale reads (~11% at ~11% FPR), and the "
        "raw span does a little better (~12% / ~16%), both far below the ~0.71 WITHIN-run localization "
        "on Gold. Spotting one superseded same-file read online, against clean runs' natural long-range "
        "dependencies and without false-alarming, is an open challenge; per-run normalization does not "
        "help here (the raw span edges out the z-score). The hardness is stable across 5 injection "
        "seeds (live_stale_robustness)."
    )


if __name__ == "__main__":
    main()
