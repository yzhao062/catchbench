r"""The declared seven-contrast addendum family, corrected as a whole. RETIRED FROM REPORTING.

Retired 2026-09-06, kept runnable on purpose
--------------------------------------------
Nothing this file computes is reported in the paper any more. The reporting frame moved from
separation verdicts to a ranking with per-arm uncertainty, and the move was settled by reading
comparable benchmarks rather than by preference. The survey is at
``research/catchbench-peer-benchmark-survey-2026-09-06.md`` in the notes repository, and its corrected
figures are: across thirteen papers, comparative testing in five and multiplicity correction in two;
in the agent-evaluation subset, two of eight and **zero of eight**.

The claim that retires this file is therefore narrow. **No comparable agent benchmark reports a
corrected registry of pairwise verdicts across its leaderboard.** It is not that comparable work never
tests differences, which is false. ``tools/score_judge_addendum.py`` is the successor and produces
what the addendum reports; migrating the manuscript is future work at the time of writing.

The file stays, and it stays executable, for two reasons. The family was declared in advance of any
score, so the record of what it would have concluded is part of the honest account of the addendum
rather than something to erase after the fact; running it reproduces that conclusion, which was that
**none of the seven contrasts separates**, the closest being ``llama-3.1-70b`` against ``gpt-5.5`` at
raw p 0.007787 and Holm p 0.054512. And a checker or computation that disappears without a note reads
to a later reader as one that was failing. This note is that record.

Do not restore its output to the paper without first extending the survey with a comparable benchmark
that reports per-pair testing. That is the standing rule, not a preference.

Original description follows.

The declaration at ``research/catchbench-m4-declaration-v3.md`` in the notes repository fixes the
family, the claim ids, the test and the correction rule before any score existed. Erratum 2 took it
from eight contrasts to seven when ``gpt-5.6-sol`` was not run. This computes exactly those seven and
nothing else: adding a contrast after seeing the scores would change the correction every other
contrast faces.

The test is exact two-sided McNemar on the discordant pairs, matching ``exact_mcnemar`` at
``tools/statistical_tests.py:258-267``. The correction is step-down Holm, matching ``holm_adjust`` at
``:243-255`` with its running maximum. A contrast separates when its adjusted p is strictly below
``ALPHA = 0.05``, matching ``significant = value < ALPHA`` at ``:1909``; an adjusted p of exactly 0.05
does not separate.

The family is exploratory. It was declared in advance, but it was declared after the eleven published
judges had been seen, so it is not a recovered preregistration and is never merged into the
eight-judge band.

Usage:
    python tools/score_judge_addendum_family.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.stats import binomtest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from catchbench.llm_judge import _score_vector, load_cache, load_judge_runs, run_key  # noqa: E402

ADDENDUM = ROOT / "data" / "llm_judge_addendum"

# (claim id, arm a, arm b). Declared before any score; see the Decision Rules section of v3.
CONTRASTS = [
    ("add.judge.claude-opus-5.vs.gpt-5.5", "claude-opus-5", "gpt-5.5"),
    ("add.judge.gpt-6-astra.vs.gpt-5.5", "gpt-6-astra", "gpt-5.5"),
    ("add.judge.llama-3-70b.vs.gpt-5.5", "llama-3-70b", "gpt-5.5"),
    ("add.judge.llama-3.1-70b.vs.gpt-5.5", "llama-3.1-70b", "gpt-5.5"),
    ("add.judge.llama-3-70b.vs.llama-3.1-70b", "llama-3-70b", "llama-3.1-70b"),
    ("add.judge.llama-3-70b.vs.llama-3.3-70b", "llama-3-70b", "llama-3.3-70b"),
    ("add.judge.llama-3.1-70b.vs.llama-3.3-70b", "llama-3.1-70b", "llama-3.3-70b"),
]

ALPHA = 0.05


def predictions(label: str) -> dict:
    """Addendum caches are read by path and are natively content-addressed. Published caches go
    through load_cache, which applies legacy_run_keys.json. Using the wrong route raises KeyError
    rather than failing quietly, which is the intent: the two key conventions never convert."""
    path = ADDENDUM / ("whoandwhen__all_at_once__%s.json" % label)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["predictions"]
    cache = load_cache("all_at_once", label)
    if cache is None:
        raise SystemExit("no cache for %r in either location" % label)
    return cache["predictions"]


def per_run_top1(label: str, runs) -> np.ndarray:
    """rank == 1 after the stable sort. Not top == mistake: the two disagree on the runs whose top
    is null, and a null wins Top-1 whenever the gold mistake is step 0."""
    preds = predictions(label)
    out = np.empty(len(runs), dtype=float)
    for i, run in enumerate(runs):
        scores = _score_vector(preds[run_key(run)], len(run["steps"]))
        order = np.argsort(-scores, kind="stable")
        out[i] = float(int(np.where(order == run["mistake"])[0][0]) + 1 == 1)
    return out


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min((len(pvalues) - rank) * pvalues[idx], 1.0))
        adjusted[idx] = running
    return adjusted


def main() -> int:
    runs = load_judge_runs()
    top1 = {}
    for _, a, b in CONTRASTS:
        for label in (a, b):
            if label not in top1:
                top1[label] = per_run_top1(label, runs)

    rows, raw = [], []
    for claim_id, a, b in CONTRASTS:
        wins = int(np.sum((top1[a] == 1) & (top1[b] == 0)))
        losses = int(np.sum((top1[a] == 0) & (top1[b] == 1)))
        n = wins + losses
        p = float(binomtest(wins, n, 0.5, alternative="two-sided").pvalue) if n else 1.0
        rows.append((claim_id, a, b, top1[a].mean() - top1[b].mean(), wins, losses, n))
        raw.append(p)
    adjusted = holm(raw)

    print("Addendum family, %d contrasts, exact two-sided McNemar, step-down Holm at alpha %.2f\n"
          % (len(CONTRASTS), ALPHA))
    print("  %-34s %8s %5s %5s %5s %11s %11s  %s"
          % ("contrast", "a-b", "w", "l", "disc", "raw p", "Holm p", "verdict"))
    for (claim_id, a, b, diff, w, l, n), p, adj in zip(rows, raw, adjusted):
        print("  %-34s %+8.4f %5d %5d %5d %11.6f %11.6f  %s"
              % ("%s vs %s" % (a, b), diff, w, l, n, p, adj,
                 "separates" if adj < ALPHA else "unresolved"))

    sep = sum(1 for a in adjusted if a < ALPHA)
    print("\n  %d of %d separate after correction." % (sep, len(CONTRASTS)))
    print("  Claim ids, as declared:")
    for (claim_id, _, _, _, _, _, _) in rows:
        print("    %s" % claim_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
