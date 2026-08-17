"""Report which Who&When split the localization board actually scores, and why.

The board scores 126 runs. The cached corpus holds 184. The difference is one whole split, and the
mechanism that drops it is a key spelling: `Algorithm-Generated` tasks carry `is_correct` while
`Hand-Crafted` tasks carry `is_corrected`, and the loader filters on the former. That is an accident
in mechanism, so it deserves a check rather than silence.

What the exclusion rests on is narrower than the missing field alone. Hand-Crafted trajectories omit
the per-step agent `name`, but GRADE's converter falls back to `role`, so the structural feature
vector is still defined: over all 2993 Hand-Crafted steps seven of the eight execution features vary
and only `is_tool` is constant. The splits differ in schema and in trace length instead, and the
length gap alone moves the random Top-1 floor by about as much as the gaps this board resolves. This
script measures composition, the key-spelling difference, `name` availability, and the per-split
floor. It does NOT establish that Hand-Crafted is unscorable; scoring one split is a board-scope
choice, and pooling them would need a validated cross-split conversion first.

Exits 0 when the composition matches what the paper states, and 1 when it has drifted, so a corpus
update that changes the scored population is visible rather than silent::

    python tools/whoandwhen_split_report.py
"""
import glob
import json
import os
import sys

import numpy as np

# What the paper reports. A corpus refresh that moves these should fail loudly.
EXPECTED_SCORED = 126
EXPECTED_EXCLUDED = 58


def main() -> int:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))
    from auditablebench import _reuse  # noqa: F401  side effect: puts GRADE on sys.path
    import agent_graph_characterization as ww

    ww._ensure_corpus()
    cache = ww.CACHE
    paths = sorted(glob.glob(os.path.join(cache, "**", "*.json"), recursive=True))
    if not paths:
        print(f"no cached Who&When corpus under {cache}; run the localization board once first")
        return 1

    splits: dict[str, list] = {}
    for path in paths:
        rel = os.path.relpath(path, cache).replace("\\", "/")
        split = rel.split("/")[1] if rel.count("/") >= 2 else rel.split("/")[0]
        try:
            splits.setdefault(split, []).append(json.load(open(path, encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue

    scored_total, excluded_total = 0, 0
    print(f"cached files: {len(paths)}\n")
    for split, tasks in sorted(splits.items()):
        # The loader's own filter, reproduced rather than imported, so a change there shows up here.
        passes = sum(str(t.get("is_correct", "")).lower() == "false" for t in tasks)
        keys = sorted({k for t in tasks for k in ("is_correct", "is_corrected") if k in t})
        steps = [len(t.get("history", []) or []) for t in tasks]
        named = sum(bool(h.get("name")) for t in tasks for h in (t.get("history", []) or []))
        eligible = np.array([n for n in steps if n >= 3], dtype=float)
        floor = float(np.mean(1.0 / eligible)) if eligible.size else float("nan")

        scored_total += passes
        excluded_total += len(tasks) - passes
        print(f"--- {split}: {len(tasks)} tasks, {passes} pass the loader filter")
        print(f"    correctness key spelled: {keys}")
        print(f"    steps per run: median {int(np.median(steps))}, max {max(steps)}")
        print(f"    steps carrying an agent `name`: {named}/{sum(steps)}")
        print(f"    random Top-1 floor on this split alone: {floor:.3f}")

    print(f"\nscored {scored_total}, excluded {excluded_total}")
    if scored_total == EXPECTED_SCORED and excluded_total == EXPECTED_EXCLUDED:
        print(f"matches the reported composition ({EXPECTED_SCORED} scored, "
              f"{EXPECTED_EXCLUDED} excluded)")
        return 0
    print(f"DRIFT: the paper reports {EXPECTED_SCORED} scored and {EXPECTED_EXCLUDED} excluded. "
          "The scored population has changed; update the paper or pin the corpus revision.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
