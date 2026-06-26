"""Run the AuditableBench seed board(s) and print the leaderboard.

    python run.py

v1 ships the POST pillar: a localization board (Who&When, human labels), detection boards (SWE-Gym
and tau-bench outcome labels), and a Gold board (faults injected into clean SWE-Gym runs, with
injection-site labels: the benchmark's own data contribution). LIVE and PRE pillars plug into the
same RunPipeline as they land. The boards download their corpora from the Hugging Face Hub on first
run; the PyGOD baseline needs torch + pygod + a pyg-lib / torch-sparse backend.
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
)
from auditablebench.live import (  # noqa: E402
    LiveStaleState,
    LiveStreaming,
    live_breakdown,
    live_stale_methods,
    live_streaming_methods,
)
from auditablebench.post import PostLocalization, post_localization_methods  # noqa: E402


def main() -> None:
    tasks = [PostLocalization(), PostDetection("swegym"), PostDetection("tau"), GoldLocalization(),
             LiveStreaming("swegym"), LiveStreaming("tau"), LiveStaleState()]
    methods = (post_localization_methods() + post_detection_methods()
               + gold_localization_methods() + live_streaming_methods() + live_stale_methods())

    print("AuditableBench :: POST v1 board(s)\n")
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

    for live in (t for t in tasks if isinstance(t, LiveStreaming)):  # early-warning curve per corpus
        print(live_breakdown(live, live_streaming_methods()))

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
        "is found by the dependency-span detector (dep-anomaly keyed to max-span, ~0.67 Top-1 stale); "
        "dropped-grounding is NOT localized by any baseline yet (~0 Top-1), an open problem. "
        "dep-anomaly is keyed to the stale-span mechanism (shown beside the max-span control). Leak "
        "check (gold_matched_breakdown): ranking only within the injector's eligible pool, has-dep and "
        "degree fall to the matched random floor (stale has-dep 0.238 = floor 0.238) while the "
        "dependency-span signal survives (stale max-span 0.676 vs floor 0.238), so the stale-state "
        "lift is leakage-controlled rather than a selection artifact. See gold_breakdown / "
        "gold_matched_breakdown / gold_report below."
        "\n- LIVE streaming (early warning): can a method flag a failing run from a PREFIX, not the "
        "finished trace? On SWE-Gym the dependency-structure signal clears a useful bar (ROC-AUC 0.74) "
        "at the first 25% (time-to-detection 25%) while run size never clears it; on tau-bench the "
        "structure is a weak LATE signal (peaks ~0.61 at 75%, no early lift), the same domain split "
        "the detection board shows. The 100% column reproduces the POST detection board on each "
        "corpus. See live_breakdown below."
        "\n- LIVE stale-state (online detection): the SAME Gold stale-state injection, but detected "
        "online at a fixed false-positive rate instead of localized post-hoc. It is HARD: the causal "
        "span signal catches only ~11% of stale reads at 5% FPR (~27% at 10%), far below the 0.67 "
        "WITHIN-run localization on Gold. Spotting one stale read against clean runs' natural "
        "long-range dependencies, without false-alarming, is an open challenge; per-run normalization "
        "barely helps (raw-span ~= z-score)."
    )


if __name__ == "__main__":
    main()
