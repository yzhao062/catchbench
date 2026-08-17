"""LIVE pillar (real-time): streaming early-warning.

Predict whether a run will fail from a growing PREFIX of its steps, not from the finished trace. The
POST detection board asks "did it fail" in hindsight; LIVE asks "can you tell early," from the first
quarter or half of the run. Scored by ROC-AUC at each prefix fraction (the early-warning curve) and
time-to-detection (the earliest prefix that clears a useful bar).

The board has three method families. (1) Supervised: seed-averaged 5-fold stratified cross-validation
on the prefix feature layers, the same machinery as POST detection, measuring whether the dependency
STRUCTURE is early predictive of failure. (2) Batch-unsupervised: ECOD over the run population's prefix
flat vectors, no labels. (3) Strict per-run online: a single scalar from a run's own prefix, no labels
and no other runs (the closest stand-in for a running detector). The supervised pipeline reuses GRADE's
verified code unchanged: ``build_graph`` over the first ``k`` steps, the nested feature layers, and the
cross-validation. The
full prefix (100%) runs that machinery on the whole trace, so the 100% column reproduces the POST
detection board wherever the LIVE ``>=4``-step filter leaves the POST population unchanged. LIVE and
POST retain different minimum-step criteria (``>=4`` against ``>=2``), but the current SWE-Gym and
tau-bench populations both coincide, so the columns agree today rather than by construction. It is a
built-in validity check, not a separate number to reconcile.

  - random              : floor (~0.5 at every prefix).
  - size (flat)          : prefix size and counts only, the trivial baseline.
  - auditable (structure): size plus the size-normalized dependency block (the structural signal
                          ``auditable`` surfaces over the growing prefix). Reading it EARLIER than
                          size is the early-warning claim.
  - full                 : flat plus execution plus raw dependency features (reference).

prefix-AUC (the leaderboard metric) is the mean ROC-AUC across prefix fractions; the per-prefix curve
and time-to-detection are in ``live_breakdown``.
"""
from __future__ import annotations

from typing import List, Mapping, Sequence

from auditablebench import _reuse  # noqa: F401  side effect: sets sys.path for grade + auditable

import numpy as np  # noqa: E402

from grade import build_graph  # noqa: E402  GRADE's canonical graph builder
import agent_graph_swegym as swegym  # noqa: E402  SWE-Gym step loader (resolved / unresolved)
import agent_graph_tau_bench as tau  # noqa: E402  tau-bench step loader (db_match outcome)
from agent_failure_detection import _cv, _layer_matrices  # noqa: E402  shared detection eval

from auditablebench.gold import _inject, _load_clean_runs  # noqa: E402  reuse the stale-state injection

_PREFIXES = (0.25, 0.5, 0.75, 1.0)
_DETECT_BAR = 0.70  # ROC-AUC a prefix must clear to count as "detected" for time-to-detection


def _prefix_steps(steps: list, frac: float) -> list:
    """The first ``frac`` of a run, at least two steps (``build_graph`` needs an edge to form)."""
    return steps[: max(2, int(np.ceil(frac * len(steps))))]


_MIN_STEPS = 4  # a run needs enough steps for an early-versus-late prefix distinction


def _load_runs(corpus: str):
    """Per-run (steps, outcome label) for a LIVE corpus; label 1 = failed. Mirrors the POST detection
    loaders (so the 100% prefix matches that board) but keeps the raw step lists so a prefix can be
    taken at each fraction."""
    runs, labels = [], []
    if corpus == "swegym":
        for rec in swegym.load_runs():  # balanced resolved / unresolved, already projected to steps
            steps = rec["steps"]
            if len(steps) >= _MIN_STEPS:
                runs.append(steps)
                labels.append(0 if rec["resolved"] else 1)
    elif corpus == "tau":
        for rec in tau.load_runs(tau._ensure_files()):  # tau-bench retail + airline (MIT)
            msgs = rec.get("messages")
            er = rec.get("eval_result") or {}
            if not isinstance(msgs, list) or len(msgs) < 3 or "db_match" not in er:
                continue
            steps = tau.to_steps(msgs, rec.get("model_path", "model"))
            if len(steps) >= _MIN_STEPS:
                runs.append(steps)
                labels.append(0 if bool(er["db_match"]) else 1)  # db_match = consequential writes landed
    else:
        raise ValueError(f"unsupported LIVE corpus: {corpus!r}")
    return runs, labels


class LiveStreaming:
    """LIVE / streaming Task: predict run failure from a growing prefix of the trace."""

    task_id = "live_streaming"
    pillar = "LIVE"
    granularity = "run"

    def __init__(self, corpus: str = "swegym", prefixes: Sequence[float] = _PREFIXES) -> None:
        if corpus not in ("swegym", "tau"):
            raise ValueError(f"unsupported LIVE corpus: {corpus!r}")
        self.corpus = corpus
        self.dataset = corpus
        self.prefixes = tuple(prefixes)
        self._loaded = False

    def setup(self) -> None:
        if self._loaded:
            return
        runs, labels = _load_runs(self.corpus)
        self.y = np.array(labels)
        self.runs = runs  # raw step lists, so an online method can score each run from its own prefix
        # One {layer: matrix} bundle per prefix fraction; rows aligned with self.y.
        self.layers_at = {
            p: _layer_matrices([build_graph(_prefix_steps(s, p), dependency="explicit",
                                            shared_resource=False) for s in runs])
            for p in self.prefixes
        }
        self._loaded = True

    def corpus_line(self) -> str:
        self.setup()
        n, n_fail = len(self.y), int(self.y.sum())
        pcts = ", ".join(f"{int(p * 100)}%" for p in self.prefixes)
        return (f"{self.dataset}: {n} runs (>={_MIN_STEPS} steps; {n_fail} failed, "
                f"{n - n_fail} resolved), streaming prefixes [{pcts}], run-level outcome labels.")


class _PrefixLayerAUC:
    """A LIVE method: one GRADE feature layer scored as prefix-AUC (mean ROC-AUC across prefixes)."""

    def __init__(self, method_id: str, layer: str) -> None:
        self.method_id = method_id
        self.layer = layer
        self.supports = {"live_streaming"}

    def auc_curve(self, task: LiveStreaming) -> dict:
        task.setup()
        return {p: float(_cv(task.layers_at[p][self.layer], task.y).mean()) for p in task.prefixes}

    def evaluate(self, task: LiveStreaming) -> Mapping[str, float]:
        return {"prefix_auc": float(np.mean(list(self.auc_curve(task).values())))}


class RandomLive:
    """The prefix-AUC floor: random run scores at every prefix (averages to ~0.5)."""

    method_id = "random"
    supports = {"live_streaming"}

    def auc_curve(self, task: LiveStreaming) -> dict:
        from sklearn.metrics import roc_auc_score

        task.setup()
        out = {}
        for p in task.prefixes:
            rng = np.random.RandomState(0)
            out[p] = float(np.mean([roc_auc_score(task.y, rng.rand(len(task.y))) for _ in range(5)]))
        return out

    def evaluate(self, task: LiveStreaming) -> Mapping[str, float]:
        return {"prefix_auc": float(np.mean(list(self.auc_curve(task).values())))}


class _PrefixECOD:
    """Unsupervised online baseline: PyOD (ECOD) on the prefix flat features at each prefix, the online
    sibling of the POST detection ECOD baseline. Batch-unsupervised (no failure labels, but fit over
    the run population at that prefix), so it tests whether an off-the-shelf tabular detector gives
    early warning from prefix size and counts alone."""

    method_id = "pyod (ECOD)"
    supports = {"live_streaming"}

    def auc_curve(self, task: LiveStreaming) -> dict:
        from pyod.models.ecod import ECOD
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        task.setup()
        out = {}
        for p in task.prefixes:
            features = StandardScaler().fit_transform(task.layers_at[p]["flat"])
            detector = ECOD()
            detector.fit(features)
            out[p] = float(roc_auc_score(task.y, detector.decision_scores_))
        return out

    def evaluate(self, task: LiveStreaming) -> Mapping[str, float]:
        return {"prefix_auc": float(np.mean(list(self.auc_curve(task).values())))}


class _OnlineScore:
    """A truly online baseline: score each run from its OWN prefix by a raw structural statistic, with
    no labels, no training, and no other runs. ROC-AUC against failure at each prefix measures whether
    an unsupervised signal a running detector could compute gives early warning, the online complement
    to the supervised-CV feature layers."""

    def __init__(self, method_id: str, run_score_fn) -> None:
        self.method_id = method_id
        self._fn = run_score_fn
        self.supports = {"live_streaming"}

    def auc_curve(self, task: LiveStreaming) -> dict:
        from sklearn.metrics import roc_auc_score

        task.setup()
        out = {}
        for p in task.prefixes:
            scores = np.array([self._fn(_prefix_steps(s, p)) for s in task.runs])
            out[p] = float(roc_auc_score(task.y, scores))
        return out

    def evaluate(self, task: LiveStreaming) -> Mapping[str, float]:
        return {"prefix_auc": float(np.mean(list(self.auc_curve(task).values())))}


def _mean_span(steps: list) -> float:
    """Mean per-step dependency span over a prefix, the raw online signal (length-confounded, reported
    next to the size baseline so the confound is visible)."""
    series = _span_series(steps)
    return float(series.mean()) if len(series) else 0.0


def live_streaming_methods() -> list:
    """The LIVE streaming board: a random floor; the supervised feature layers (size, structure, full)
    scored by cross-validation; and two unsupervised online baselines (off-the-shelf ECOD on the
    prefix, and a raw per-run dependency-span signal a running detector could compute without labels)."""
    return [
        RandomLive(),
        _PrefixLayerAUC("size (flat)", "flat"),
        _PrefixLayerAUC("auditable (structure)", "flatdep"),
        _PrefixLayerAUC("full", "full"),
        _PrefixECOD(),
        _OnlineScore("dep-span (online)", _mean_span),
    ]


def live_breakdown(task: LiveStreaming, methods: list) -> str:
    """The early-warning view: ROC-AUC by prefix plus time-to-detection. Read the curve left to right,
    because the LIVE question is whether a method separates failing runs from a SHORT prefix, not only
    from the finished run (the 100% column, the POST-style detection check on the LIVE population)."""
    task.setup()
    ps = task.prefixes
    header = (f"  {'method':24s}" + "".join(f"{str(int(p * 100)) + '%':>8s}" for p in ps)
              + f"{'t2d':>8s}")
    lines = [f"\nLIVE streaming early-warning (ROC-AUC by prefix; "
             f"t2d = earliest prefix with AUC>={_DETECT_BAR:.2f}):", header]
    for m in methods:
        if not hasattr(m, "auc_curve"):
            continue
        curve = m.auc_curve(task)
        cells = "".join(f"{curve[p]:>8.3f}" for p in ps)
        t2d = next((f"{int(p * 100)}%" for p in ps if curve[p] >= _DETECT_BAR), ">100%")
        lines.append(f"  {m.method_id:24s}{cells}{t2d:>8s}")
    return "\n".join(lines)


# --- LIVE stale-state: online detection of the Gold stale-state injection -------------------------

_FPR_LEVELS = (0.05, 0.10)  # target clean-run FPRs the run-level peak threshold aims at (n is small,
#                             so the realized rate is reported beside the TPR, not assumed exact)


def _span_series(steps: list) -> np.ndarray:
    """Per-step max dependency span (how far back a step's farthest dependency reaches), the online
    signal a stale read inflates."""
    idx_of = {s["idx"]: i for i, s in enumerate(steps)}
    out = np.zeros(len(steps))
    for i, s in enumerate(steps):
        dep_pos = [idx_of[d] for d in s.get("deps", ()) if d in idx_of and idx_of[d] < i]
        out[i] = float(i - min(dep_pos)) if dep_pos else 0.0
    return out


def _count_series(steps: list) -> np.ndarray:
    """Per-step dependency count: a NEGATIVE control. The stale-state injection redirects one edge
    without changing any step's edge count, so the count series is identical clean versus injected and
    an online detector on it sits exactly at the false-positive floor. That the span detector lifts
    while this one does not shows the signal is the span, not generic structural change."""
    idx_of = {s["idx"]: i for i, s in enumerate(steps)}
    out = np.zeros(len(steps))
    for i, s in enumerate(steps):
        out[i] = float(sum(1 for d in s.get("deps", ()) if d in idx_of and idx_of[d] < i))
    return out


def _online_z(series: np.ndarray) -> np.ndarray:
    """Causal z-score: each step versus the prefix STRICTLY BEFORE it, so the score at step t uses
    only steps < t (an online detector cannot see the future). Steps without enough history score 0
    and cannot trip the detector."""
    z = np.zeros(len(series))
    for t in range(len(series)):
        prior = series[:t]
        if len(prior) >= 2 and prior.std() > 1e-9:
            z[t] = (series[t] - prior.mean()) / prior.std()
    return z


class LiveStaleState:
    """LIVE / stale-state Task: detect the Gold stale-state injection ONLINE, with false-positive
    control.

    Same injection as Gold (redirect one dependency to an earlier superseded event on the same file,
    inflating that step's dependency span), scored as FPR-calibrated detection rather than post-hoc
    localization. A
    causal per-step score (step t sees only steps < t, so an online detector could compute it) is
    reduced to the run's peak; the threshold is calibrated on the paired CLEAN runs to a fixed
    false-positive rate; a run is flagged when its peak crosses it. Because the injection only adds a
    spike to an otherwise identical clean run, TPR above the FPR isolates the injected signal. The
    span z-score is keyed to the stale-state mechanism (the same caveat as Gold's dep-anomaly), and
    the dependencies are inferred."""

    task_id = "live_stale_state"
    pillar = "LIVE"
    granularity = "step"
    dataset = "swegym-gold"

    def __init__(self, seed: int = 0, fprs: Sequence[float] = _FPR_LEVELS) -> None:
        self.seed = seed
        self.fprs = tuple(fprs)
        self._loaded = False

    def setup(self) -> None:
        if self._loaded:
            return
        injected, clean, targets, kinds = _inject(_load_clean_runs(), seed=self.seed)
        keep = [i for i, k in enumerate(kinds) if k == "stale"]  # the stale-state half only
        self.injected = [injected[i] for i in keep]
        self.clean = [clean[i] for i in keep]
        self.targets = [targets[i] for i in keep]
        self._loaded = True

    def corpus_line(self) -> str:
        self.setup()
        levels = ", ".join(f"{int(f * 100)}%" for f in self.fprs)
        return (f"{self.dataset}: {len(self.injected)} stale-state injections over real SWE-Gym runs "
                f"(paired clean controls), online detection at FPR [{levels}].")


class _OnlineDetector:
    """A LIVE stale-state method: a per-step score function, thresholded at each clean-run FPR."""

    def __init__(self, method_id: str, score_fn) -> None:
        self.method_id = method_id
        self._score_fn = score_fn
        self.supports = {"live_stale_state"}

    def detail(self, task: "LiveStaleState") -> Mapping[float, tuple]:
        """Per target FPR: (TPR on injected runs, REALIZED FPR on the paired clean runs). The realized
        rate is reported because n is small, so the empirical FPR cannot match the target exactly."""
        task.setup()
        clean_max = np.array([self._score_fn(s).max() if len(s) else 0.0 for s in task.clean])
        inj_max = np.array([self._score_fn(s).max() if len(s) else 0.0 for s in task.injected])
        out = {}
        for fpr in task.fprs:
            tau = float(np.quantile(clean_max, 1.0 - fpr))  # run-level peak threshold over clean runs
            out[fpr] = (float(np.mean(inj_max > tau)), float(np.mean(clean_max > tau)))  # paired
        return out

    def evaluate(self, task: "LiveStaleState") -> Mapping[str, float]:
        return {f"tpr@{int(fpr * 100)}fpr": tpr for fpr, (tpr, _) in self.detail(task).items()}


class RandomOnline:
    """The TPR floor: a random per-step score, so TPR ~ FPR (no signal)."""

    method_id = "random"
    supports = {"live_stale_state"}

    @staticmethod
    def _det() -> _OnlineDetector:
        rng = np.random.RandomState(0)  # fresh seed per call so detail / evaluate see the same stream
        return _OnlineDetector("random", lambda s: rng.rand(len(s)))

    def detail(self, task: "LiveStaleState") -> Mapping[float, tuple]:
        return self._det().detail(task)

    def evaluate(self, task: "LiveStaleState") -> Mapping[str, float]:
        return self._det().evaluate(task)


def live_stale_methods() -> list:
    """Online stale-state detectors: a random floor, the raw-span baseline (run-length confounded),
    and the causal span z-score (per-run normalized, the signal auditable surfaces over the stream)."""
    return [
        RandomOnline(),
        _OnlineDetector("dep-count (control)", lambda s: _online_z(_count_series(s))),
        _OnlineDetector("raw-span", _span_series),
        _OnlineDetector("auditable (span z-score)", lambda s: _online_z(_span_series(s))),
    ]


def live_stale_breakdown(task: LiveStaleState, methods: list) -> str:
    """Per-method TPR at each target FPR, with the REALIZED clean-flag rate beside it. On a paired set
    this small the empirical FPR cannot hit the target exactly, so it is reported, not assumed."""
    task.setup()
    fprs = task.fprs
    header = ("  " + f"{'method':26s}"
              + "".join(f"{'tpr@' + str(int(f * 100)) + '% (real fpr)':>22s}" for f in fprs))
    lines = [f"\nLIVE stale-state online detection (n={len(task.injected)} paired runs; TPR at target "
             f"FPR, realized clean-flag rate in parens):", header]
    for m in methods:
        if not hasattr(m, "detail"):
            continue
        d = m.detail(task)
        cells = "".join(f"{tpr:.3f} ({rf * 100:.1f}%)".rjust(22) for tpr, rf in (d[f] for f in fprs))
        lines.append(f"  {m.method_id:26s}{cells}")
    return "\n".join(lines)


def live_stale_robustness(methods: list, seeds=(0, 1, 2, 3, 4)) -> str:
    """Stability of the online stale-state TPR across injection seeds (the synthetic-injection
    robustness question for the LIVE board): per method, TPR at each target FPR as mean +/- std over
    seeds. Clean runs are cached in gold.py, so this re-injects K times without reloading the corpus."""
    seeds = tuple(seeds)
    keep = [m for m in methods if hasattr(m, "detail")]
    acc = {m.method_id: {f: [] for f in _FPR_LEVELS} for m in keep}
    for seed in seeds:
        task = LiveStaleState(seed=seed)
        task.setup()
        for m in keep:
            d = m.detail(task)
            for f in _FPR_LEVELS:
                acc[m.method_id][f].append(d[f][0])  # TPR at target FPR f
    header = ("  " + f"{'method':26s}"
              + "".join(f"{'tpr@' + str(int(f * 100)) + '%':>20s}" for f in _FPR_LEVELS))
    lines = [f"\nLIVE stale-state seed robustness ({len(seeds)} injection seeds, TPR mean +/- std):",
             header]
    for m in keep:
        cells = "".join(
            f"{np.mean(acc[m.method_id][f]):.3f}+/-{np.std(acc[m.method_id][f]):.3f}".rjust(20)
            for f in _FPR_LEVELS)
        lines.append(f"  {m.method_id:26s}{cells}")
    return "\n".join(lines)
