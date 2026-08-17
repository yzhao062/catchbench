"""Broken-predecessor artifact diagnostic for the Gold file-level substrate.

Measures the construction leakage that makes both Gold fault kinds inadmissible as
artifact-controlled benchmark evidence on the file-level substrate (Defensibility Bar item 2). The
clean substrate builds ``deps = [last_on_file[f]]``: every file event with a prior same-file event
carries exactly its immediate same-file predecessor. Both injections break that invariant at the
injected step (stale-state redirects the edge to an older same-file event; dropped-grounding removes
it), and nothing else in a run does. So a detector that knows the construction, not the fault, flags
"file event whose immediate same-file predecessor is missing from its deps" and separates injected
from clean perfectly.

Documented result (2026-07-19, seeds 0-4): the marker uniquely ranks all 82 stale-state and all 106
dropped-grounding targets Top-1 in every seed, and flags 0 of 188 paired clean runs. The
eligibility-matched control in ``gold_matched_breakdown`` holds eligibility fixed, but not
degree, and cannot
control this marker, so Gold scores on this substrate are mechanism diagnostics; artifact-controlled
evidence waits on a named-value substrate. Injection-site labels stay detector-independent (Bar
item 4 passes).

Run:  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python tools/gold_artifact_diagnostic.py
Exit code 0 iff the documented perfect separation reproduces (a substrate change flips it).
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from auditablebench import gold  # noqa: E402


def artifact_positions(steps):
    """Positions violating the clean-substrate invariant: a file event whose immediate same-file
    predecessor is absent from its deps. Impossible in a clean ``deps=[last_on_file[f]]`` run."""
    last_on_file = {}
    hits = []
    for i, s in enumerate(steps):
        f = s.get("file")
        if f:
            if f in last_on_file and last_on_file[f] not in s.get("deps", ()):
                hits.append(i)
            last_on_file[f] = s["idx"]
    return hits


def main(seeds=(0, 1, 2, 3, 4)):
    runs = gold._load_clean_runs()
    ok = True
    print("Gold broken-predecessor artifact diagnostic (per seed: unique-Top-1 / targets):")
    for seed in seeds:
        injected, clean, targets, kinds = gold._inject(runs, seed=seed)
        unique = {"stale": 0, "dropped": 0}
        total = {"stale": 0, "dropped": 0}
        for steps, target, kind in zip(injected, targets, kinds):
            total[kind] += 1
            if artifact_positions(steps) == [target]:
                unique[kind] += 1
        clean_flagged = sum(1 for steps in clean if artifact_positions(steps))
        print(f"  seed {seed}: stale {unique['stale']}/{total['stale']}, "
              f"dropped {unique['dropped']}/{total['dropped']}, "
              f"clean flagged {clean_flagged}/{len(clean)}")
        if (unique != total or clean_flagged
                or total != {"stale": 82, "dropped": 106} or len(clean) != 188):
            ok = False
    verdict = ("REPRODUCED: both fault kinds separate perfectly on the file-level substrate "
               "(Bar item 2 fails; scores are mechanism diagnostics)" if ok else
               "NOT REPRODUCED: the substrate or injector changed; re-derive the documented "
               "denominators before quoting them")
    print(verdict)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
