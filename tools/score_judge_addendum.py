r"""Rank the judge-addendum caches, which sit outside the directory the board globs.

The addendum caches live in ``data/llm_judge_addendum/`` rather than ``data/llm_judge/``, so
``LLMJudgeLocalization`` cannot reach them: it resolves its cache through ``cache_path``, which is
inside the globbed directory, and a cache inside that directory would enter the published arena block
and move the entrant count. The declaration at ``research/catchbench-m4-declaration-v3.md`` in the
notes repository makes the move the mechanism for keeping the addendum out of the arena.

This reads them by path and scores them through the same ``_score_vector`` and ``_rank_metrics``
calls ``LLMJudgeLocalization.evaluate`` makes at ``llm_judge.py:414-419``, so an addendum Top-1 is
computed by the code that computed the eleven published ones.

The declaration's single definition of the outcome, restated here because the other reading is easy
to reach for: the binary Top-1 outcome for a run is ``rank == 1``, where rank is the one-based
position of the gold mistake step after ``np.argsort(-scores, kind="stable")`` over the per-step
vector. It is NOT ``top == mistake``. The two agree on every run whose prediction names an in-range
step, because ``_score_vector`` forces that step above every ranking score, and they disagree on the
runs whose top is null: a null top wins Top-1 whenever the gold mistake is step 0, which is true of
20 of the 126 runs. ``run_llm_judge_panel.py`` prints the wrong reading in its per-pass summary, so
that printed figure is not the scored quantity and must not be reported as one.

Reporting frame, changed 2026-09-06
-----------------------------------
This used to close by pointing at a declared family of contrasts and a Holm correction. It no longer
does, and the change was settled by reading comparable benchmarks rather than by preference. The
survey is at ``research/catchbench-peer-benchmark-survey-2026-09-06.md`` in the notes repository; it
was audited and corrected, and the corrected figures are the ones below.

Across thirteen papers, comparative testing appears in five and multiplicity correction in two. In the
agent-evaluation subset the counts are two of eight and **zero of eight**. So the claim that carries
this decision is the narrow one: **no comparable agent benchmark reports a corrected registry of
pairwise verdicts across its leaderboard.** The broader claim that comparable agent work never tests
differences is false, and must not be repeated: AgentEval (arXiv:2604.23581) reports paired bootstrap
tests with a confidence interval on every metric at n=150, on a structure-against-flat comparison.

The judge addendum is a ranking and is reported as one. A small number of paired tests are kept for
the paper's core detection claims, which are not scored here. **Manuscript migration is future work at
the time of writing**; this tool produces what the addendum reports, not what the paper currently
says.

Two things are kept apart here, because they decide opposite questions. **Uncertainty on one arm's
own score** is a Wilson interval on that arm's own proportion over the 126 runs, and it is reported.
**A test of the difference between two arms** is not computed and not reported.

What the two printed columns actually mean
------------------------------------------
``#`` is the position by Top-1 point estimate, with arms holding the same Top-1 count given the same
position. That is what a rank-only benchmark publishes.

``above`` is one plus the number of arms whose whole 95 percent interval lies above this arm's. It is
an endpoint-count statistic and nothing more.

**``above`` does not define groups of equal arms, and an earlier version of this file wrongly said it
did.** Interval overlap is not transitive, so "overlapping arms share a value" is false as a general
statement and is false on this very output: ``llama-3.1-70b`` overlaps ``gpt-5.5``, ``gpt-6-astra``,
``llama-3.3-70b`` and ``llama-3-70b``, yet takes a different ``above`` value from all four, because it
is the only arm that fails to overlap ``claude-opus-5``. Read a value of 1 as "no arm sits entirely
above this one", never as "these arms are tied" or "these arms are equal".

Chatbot Arena (arXiv:2403.04132, Section 5) computes the same endpoint count. Its guarantee is that
when its joint confidence set covers the true score vector, no arm's reported position exceeds its
true position, with the failure probability controlled at alpha. That guarantee comes from the joint
construction and is asymptotic. **Nothing here carries it.** These are marginal binomial intervals; six
of them do not give simultaneous 95 percent coverage.

What comparing endpoints costs, stated precisely
------------------------------------------------
Every arm is scored on the same 126 runs, so the arms are paired and comparing two marginal intervals
throws away information a direct paired comparison would use. That is a statement about discarded
information, not about a guaranteed direction: for a paired difference
``Var(A - B) = Var(A) + Var(B) - 2 Cov(A, B)``, and sharing runs does not by itself fix the sign of
that covariance. On this data the observed cross-arm covariances are positive, so pairing is
informative here, but that was measured rather than assumed.

Overlap therefore proves neither equality nor the absence of a meaningful difference. At n=126 the
endpoint count puts a Top-1 gap of 0.15, ``claude-opus-5`` at 0.4762 against ``llama-3-70b`` at
0.3254, at the same value. That is a property of marginal intervals at this sample size, and it is
not evidence that the two judges perform alike.

Usage:
    python tools/score_judge_addendum.py
    python tools/score_judge_addendum.py --reference gpt-5.5 --also llama-3.3-70b
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from catchbench.llm_judge import (  # noqa: E402
    _METRIC_NAMES,
    _rank_metrics,
    _score_vector,
    load_cache,
    load_judge_runs,
    run_key,
)
from catchbench.post import PostLocalization  # noqa: E402

ADDENDUM_DIR = ROOT / "data" / "llm_judge_addendum"

# Written by tools/emit_addendum_sidecars.py beside each cache, sharing its name prefix. Kept here as
# a constant so the two tools cannot disagree about what a sidecar is called.
SIDECAR_SUFFIX = ".sidecar.json"

# Two-sided normal quantile at 95 percent, inlined so this file needs no scipy. The Wilson interval
# below is the only place it is used.
_Z95 = 1.959963984540054

# The published Llama at the same 70B scale, pulled into the ranking by default so the generation
# series 3 -> 3.1 -> 3.3 is readable in one table. It is the only published label that completes a
# series the addendum starts; every other published judge stays in the arena block where it belongs.
_DEFAULT_ALSO = ("llama-3.3-70b",)


# Historical generation is documented in tools/llm_judge_codex.py, lines 1 to 7.
# Current adapter routing does not establish how a published cache was generated.
_RECORDED_PUBLISHED_CHANNELS = {"gpt-5.5": "codex-cli"}


def generation_channel(label: str, source: str) -> str:
    """Read the recorded channel, retaining the sidecars' declared-configuration scope."""
    if source == "published":
        return _RECORDED_PUBLISHED_CHANNELS.get(label, "not-recorded")
    path = ADDENDUM_DIR / f"whoandwhen__all_at_once__{label}{SIDECAR_SUFFIX}"
    if not path.exists():
        return "not-recorded"
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    configuration = sidecar.get("provider_declared_configuration") or {}
    return configuration.get("channel") or "not-recorded"


def addendum_predictions(path: pathlib.Path) -> dict:
    """The predictions of an addendum cache, read by path.

    Addendum caches are written natively with content-address keys, so they need no mapping. The
    thirty-one published caches are the opposite: every one of their 126 keys is the legacy
    order-prefixed form, and ``load_cache`` applies ``legacy_run_keys.json`` on read. Reading a
    published cache with this function raises KeyError on the first lookup, which is the intended
    behaviour: the two conventions coexist and neither is converted into the other.
    """
    return json.loads(path.read_text(encoding="utf-8"))["predictions"]


def score_cache(preds: dict, runs, task, name: str) -> dict[str, float]:
    """Top-1, Top-3 and MRR for one prediction set, by the same calls evaluate makes."""
    missing = [run_key(r) for r in runs if run_key(r) not in preds]
    if missing:
        raise SystemExit("%s is missing %d of %d run keys" % (name, len(missing), len(runs)))
    chunks = [_score_vector(preds[run_key(r)], len(r["steps"])) for r in runs]
    metrics = _rank_metrics(np.concatenate(chunks), task.groups, task.mistake_row)
    return dict(zip(_METRIC_NAMES, (float(v) for v in metrics)))


def per_run_ranks(preds: dict, runs) -> np.ndarray:
    """The one-based rank of the gold mistake step in each run, under the declared definition.

    The point estimates come from ``_rank_metrics``, which is the published code path. This exists
    only to give the intervals something to be computed from.

    The guard in ``collect`` compares **aggregate** quantities, the Top-1 and Top-3 hit counts and
    the MRR, and that is the whole of its contract. It does not compare run by run, so it cannot
    detect a permutation of ranks across runs that leaves those aggregates unchanged, and an earlier
    version of this docstring claimed more than that. What the guard does establish is the thing the
    intervals need: the proportion an interval is drawn around is the proportion the board reports.
    """
    out = np.empty(len(runs), dtype=int)
    for i, run in enumerate(runs):
        scores = _score_vector(preds[run_key(run)], len(run["steps"]))
        order = np.argsort(-scores, kind="stable")
        out[i] = int(np.where(order == run["mistake"])[0][0]) + 1
    return out


def wilson(successes: int, n: int, z: float = _Z95) -> tuple[float, float]:
    """Wilson score interval on one arm's own proportion.

    Wilson rather than the normal approximation because Top-1 sits near 0.3 to 0.5 on 126 runs and
    the normal interval misbehaves toward the ends of the range. This is uncertainty on a single
    arm's score. It is not a comparison, and two of these intervals must not be differenced.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def endpoint_rank(arms: list[dict], key: str = "top1") -> None:
    """One plus the count of arms lying entirely above each arm, by interval endpoints. Mutates.

    This is a descriptive count, not a partition into equal groups. Interval overlap is not
    transitive, so two arms sharing a value here may each overlap different sets of other arms, and
    two arms with different values here may overlap each other. The module docstring gives the
    counterexample that occurs in this tool's own output. Never present a shared value as a tie, as
    equality, or as a failure to reject anything.
    """
    for arm in arms:
        arm["above"] = 1 + sum(1 for other in arms if other[key + "_lo"] > arm[key + "_hi"])


def position_by_score(arms: list[dict], key: str = "top1") -> None:
    """Standard competition position on the point estimate, so equal scores share a position.

    Without this, two arms with an identical Top-1 count take different positions purely from the
    order they were loaded in, which reads as a performance ordering that the numbers do not
    support. ``gpt-5.5`` and ``gpt-6-astra`` both score 57 of 126 on this data, so the case is live
    rather than hypothetical.
    """
    for index, arm in enumerate(arms):
        ties_above = sum(1 for other in arms if other[key] > arm[key])
        arm["seq"] = 1 + ties_above


def collect(label: str, preds: dict, runs, task, source: str) -> dict:
    """Point estimates from the published code path, plus intervals from the per-run ranks."""
    metrics = score_cache(preds, runs, task, label)
    ranks = per_run_ranks(preds, runs)
    n = len(runs)
    top1_hits = int(np.sum(ranks == 1))
    top3_hits = int(np.sum(ranks <= 3))

    # The aggregate guard per_run_ranks describes. MRR is included because it reads every rank
    # rather than only whether a rank is at most 1 or 3, so it catches drift the two hit counts
    # miss. It still compares aggregates, so a rank permutation preserving all three passes.
    for name, derived, published in (("top1", top1_hits / n, metrics["top1"]),
                                     ("top3", top3_hits / n, metrics["top3"]),
                                     ("mrr", float(np.mean(1.0 / ranks)), metrics["mrr"])):
        if abs(derived - published) > 1e-9:
            raise SystemExit(
                "%s: per-run %s %.12f disagrees with _rank_metrics %.12f; the interval would not "
                "belong to the reported number" % (label, name, derived, published))

    top1_lo, top1_hi = wilson(top1_hits, n)
    top3_lo, top3_hi = wilson(top3_hits, n)
    return {
        "label": label,
        "source": source,
        "channel": generation_channel(label, source),
        "top1": metrics["top1"], "top1_lo": top1_lo, "top1_hi": top1_hi,
        "top3": metrics["top3"], "top3_lo": top3_lo, "top3_hi": top3_hi,
        "mrr": metrics["mrr"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Rank the judge addendum with per-arm uncertainty.")
    ap.add_argument("--reference", default="gpt-5.5",
                    help="published label the addendum is ranked beside; read only, never written")
    ap.add_argument("--also", nargs="*", default=list(_DEFAULT_ALSO),
                    help="further published labels to include in the ranking")
    args = ap.parse_args()

    if not ADDENDUM_DIR.is_dir():
        raise SystemExit("no %s; nothing to score" % ADDENDUM_DIR)
    # Sidecars share the cache prefix and would otherwise be globbed as caches, then fail on their
    # missing "predictions" member. Filtered before the emptiness check so a directory holding only
    # sidecars reports "no caches" rather than a KeyError.
    caches = sorted(p for p in ADDENDUM_DIR.glob("whoandwhen__*__*.json")
                    if not p.name.endswith(SIDECAR_SUFFIX))
    if not caches:
        raise SystemExit("no addendum prediction caches in %s" % ADDENDUM_DIR)

    task = PostLocalization()
    task.setup()
    runs = load_judge_runs()

    arms = []
    # load_cache, not a raw read: these are published caches and their keys need the map.
    for label in [args.reference] + [a for a in args.also if a != args.reference]:
        cache = load_cache("all_at_once", label)
        if cache is None:
            raise SystemExit("no published cache for %r" % label)
        arms.append(collect(label, cache["predictions"], runs, task, "published"))
    for path in caches:
        label = path.stem.split("__")[-1]
        arms.append(collect(label, addendum_predictions(path), runs, task, "addendum"))

    arms.sort(key=lambda a: -a["top1"])
    position_by_score(arms, "top1")
    endpoint_rank(arms, "top1")

    print("Judge addendum over %d runs, ordered by Top-1. Intervals are Wilson at 95 percent on"
          % len(runs))
    print("each arm's own score, marginal and not simultaneous across arms.")
    print("  `#`     position by Top-1 point estimate; equal scores share a position.")
    print("  `above` 1 + the number of arms whose whole interval lies above this arm's. A count,")
    print("          NOT a tie group: interval overlap is not transitive. See the docstring.\n")
    print("  %-3s %-6s %-18s %-16s %-10s %8s %-18s %8s %-18s %8s"
          % ("#", "above", "model", "channel", "source", "Top-1", "95% interval",
             "Top-3", "95% interval", "MRR"))
    for arm in arms:
        print("  %-3d %-6d %-18s %-16s %-10s %8.4f [%.4f, %.4f] %8.4f [%.4f, %.4f] %8.4f"
              % (arm["seq"], arm["above"], arm["label"], arm["channel"], arm["source"],
                 arm["top1"], arm["top1_lo"], arm["top1_hi"],
                 arm["top3"], arm["top3_lo"], arm["top3_hi"], arm["mrr"]))
    print("  %-3s %-6s %-18s %-16s %-10s %8s %-18s %8s %-18s %8s"
          % ("-", "-", "gpt-5.6-sol", "not-run", "addendum", "-", "-", "-", "-", "-"))

    # Frozen declaration, research/catchbench-m4-declaration-v3.md, Erratum 2,
    # lines 727 to 735 and 787 to 792. This was a choice, not model unavailability.
    print("\n  gpt-5.6-sol was declared but not run. No subscription CLI served it, and its only")
    print("  route was the NAIRR gateway, whose remote plain HTTP endpoint the adapter refuses.")
    print("  The required SSH forward was not set up, because gpt-6-astra was preferred as the")
    print("  more advanced model in the same vendor family, before either OpenAI score existed.")
    print("  This was a configuration decision, not model unavailability. No score exists for it.")

    print("\n  Channel records: addendum channels come from provider_declared_configuration in")
    print("  data/llm_judge_addendum/*.sidecar.json. These are reconstructed configurations;")
    print("  no invocation channel was retained. gpt-5.5's Codex CLI generation is documented")
    print("  in tools/llm_judge_codex.py. not-recorded means the published cache or addendum")
    print("  sidecar supplies no channel and this report has no documented generation channel.")
    print("  source means published or addendum; it does not name a generation channel.")

    widest = max(a["top1"] for a in arms) - min(a["top1"] for a in arms)
    print("\n  %d arms across a Top-1 spread of %.4f. Top-1 is rank == 1 after the stable sort, not"
          % (len(arms), widest))
    print("  top == mistake. No difference between two arms is tested here and none is claimed.")
    print("  Overlapping intervals are not evidence that two arms perform alike.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
