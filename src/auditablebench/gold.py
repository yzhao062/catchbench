"""AuditableBench-Gold: grounded fault injection over real agent run-graphs.

The contribution here is the injection itself, so it is built to hold up (BOND's discipline: the
synthesis must be defensible, not a convenient artifact) and reported honestly, including where it
does not yet clear the bar.

Faults are documented agent failure modes, realized on the inferred dependency layer:
  - stale-state read  : redirect one dependency from its true source to an EARLIER (superseded)
                        prior step. Signature: an unusually long dependency span.
  - dropped grounding : remove one required dependency, so the step acts ungrounded. Intended
                        signature: an unusually low dependency count.

What the current slice actually shows (report it per fault kind, not just in aggregate):
  - Stale-state is detectable. A dependency-span detector localizes it well above the floor.
  - Dropped-grounding is NOT yet localized by any baseline here; removing one edge among many leaves
    no signal the current detectors catch. Honest open problem, not a win to average away.
  - The dependency-aware detector (dep-anomaly) is essentially raw max-span, i.e. KEYED to the
    stale-state mechanism. It is reported as a mechanism check beside the raw max-span and low-count
    controls, not as a general detector.

Leakage and distribution checks, reported as first-class results (see ``gold_report`` /
``gold_breakdown``):
  - position does not lift above the seed-averaged random floor (no position artifact). In the full
    candidate pool degree sits above that floor, but only because the injector targets steps that
    have dependencies: a has-dep eligibility baseline scores almost the same, so the lift is target
    SELECTION, a construction leak. ``gold_matched_breakdown`` controls for it by ranking only within
    the injector's eligible pool, where has-dep and degree fall to the matched floor and the
    dependency-span signal is what survives (for stale-state). That matched comparison is the
    leakage control, reported beside the full-pool board rather than asserted.
  - Stale-state preserves run-level edge count; dropped-grounding removes one edge; stale-state also
    shifts the run-level max-span distribution by construction. All are reported paired, clean
    versus injected.
  - Labels are the injection site, correct by construction and independent of any detector.
  - Caveat: SWE-Gym dependencies are INFERRED, so a redirected edge is a dependency-misattribution
    proxy for a true stale read. The airtight upgrade is a named-value corpus; the degree-matched
    selection control and a dropped-grounding detector are the nearer fixes.
"""
from __future__ import annotations

from typing import List, Mapping, Tuple

from auditablebench import _reuse  # noqa: F401  side effect: sets sys.path for grade + auditable

import numpy as np  # noqa: E402

import agent_graph_swegym as swegym  # noqa: E402  SWE-Gym step loader (resolved / unresolved)
from agent_failure_localization import _rank_metrics  # noqa: E402  shared ranking metrics

from auditablebench.graph_ad import pygod_node_scores  # noqa: E402

_METRIC_NAMES = ("top1", "top3", "mrr")
Graph = Tuple[np.ndarray, np.ndarray]  # (node features [n, d], edge_index [2, m])


def _load_clean_runs() -> List[list]:
    """Resolved (clean) SWE-Gym runs as step lists, keeping only runs with deps to corrupt."""
    runs = []
    for rec in swegym.load_runs():
        if not rec.get("resolved"):
            continue
        steps = rec["steps"]
        if len(steps) < 4 or not any(s.get("deps") for s in steps):
            continue
        runs.append(steps)
    return runs


def _step_graph(steps: list) -> Graph:
    """A step-only graph (nodes = steps in order, edges = current-to-prior dependency edges) plus
    per-step features [position, is_tool, dep_count, max_span], from the ``deps`` lists. Edge
    direction is step -> dependency, matching GRADE's ``depends_on``."""
    idx_of = {s["idx"]: i for i, s in enumerate(steps)}
    n = len(steps)
    feats, src, dst = [], [], []
    for i, s in enumerate(steps):
        dep_pos = [idx_of[d] for d in s.get("deps", ()) if d in idx_of and idx_of[d] < i]
        feats.append([
            i / max(1, n - 1),                                  # position
            1.0 if s.get("kind") == "tool_call" else 0.0,       # is_tool
            float(len(dep_pos)),                                # dependency count
            float(i - min(dep_pos)) if dep_pos else 0.0,        # max dependency span
        ])
        for dp in dep_pos:
            src.append(i)       # step i depends_on dp: edge i -> dp (current -> prior)
            dst.append(dp)
    edges = np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)
    return np.array(feats, dtype=float), edges


def _inject(runs: List[list], seed: int = 0):
    """Inject one grounded fault per run; return (injected step lists, paired clean step lists,
    injected step position per run, fault kind per run). A run that affords neither fault is skipped,
    and its clean copy is dropped too, so the clean and injected lists stay paired run for run."""
    rng = np.random.RandomState(seed)
    injected, clean, targets, kinds = [], [], [], []
    for ri, original in enumerate(runs):
        steps = [dict(s, deps=list(s.get("deps", ()))) for s in original]  # copy with mutable deps
        idx_of = {s["idx"]: i for i, s in enumerate(steps)}
        hit = None
        for stale in ([True, False] if ri % 2 == 0 else [False, True]):  # intended kind, then other
            positions = list(range(len(steps)))
            rng.shuffle(positions)
            for p in positions:
                deps = [d for d in steps[p]["deps"] if d in idx_of and idx_of[d] < p]
                if stale:  # redirect a dependency to an earlier (superseded) prior step
                    cand = [d for d in deps if idx_of[d] >= 1]
                    if not cand:
                        continue
                    d = int(rng.choice(cand))
                    new_d = steps[int(rng.randint(0, idx_of[d]))]["idx"]  # an earlier step
                    if new_d == d or new_d in steps[p]["deps"]:
                        continue
                    steps[p]["deps"] = [new_d if x == d else x for x in steps[p]["deps"]]
                    hit = (p, "stale")
                    break
                else:  # drop a required dependency
                    if not deps:
                        continue
                    d = int(rng.choice(deps))
                    steps[p]["deps"] = [x for x in steps[p]["deps"] if x != d]
                    hit = (p, "dropped")
                    break
            if hit:
                break
        if hit is None:
            continue
        injected.append(steps)
        clean.append(original)
        targets.append(hit[0])
        kinds.append(hit[1])
    return injected, clean, targets, kinds


class GoldLocalization:
    """POST / Gold Task: localize the grounded injected fault in a real run (injection-site label)."""

    task_id = "gold_localization"
    pillar = "POST"
    granularity = "step"
    dataset = "swegym-gold"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._loaded = False

    def setup(self) -> None:
        if self._loaded:
            return
        clean = _load_clean_runs()
        self.steps, self.clean_steps, self.targets, self.kinds = _inject(clean, seed=self.seed)
        self.graphs = [_step_graph(s) for s in self.steps]  # injected step graphs
        groups, injected_row, offset = [], {}, 0
        for gi, (X, _) in enumerate(self.graphs):
            n = len(X)
            groups.extend([gi] * n)
            injected_row[gi] = offset + self.targets[gi]
            offset += n
        self.groups = np.array(groups)
        self.injected_row = injected_row
        self._loaded = True

    def corpus_line(self) -> str:
        self.setup()
        n_stale = sum(1 for k in self.kinds if k == "stale")
        return (f"{self.dataset}: {len(self.graphs)} clean SWE-Gym runs, one injected fault each "
                f"({n_stale} stale-state, {len(self.kinds) - n_stale} dropped-grounding), "
                f"injection-site labels (deps INFERRED, characterized as a proxy).")


def _scored(task: GoldLocalization, per_graph: List[np.ndarray]) -> Mapping[str, float]:
    metrics = _rank_metrics(np.concatenate(per_graph), task.groups, task.injected_row)
    return dict(zip(_METRIC_NAMES, (float(v) for v in metrics)))


def _zmax(*cols: np.ndarray) -> np.ndarray:
    """Per-node anomaly: the largest absolute within-run z-score across the given feature columns."""
    out = np.zeros(len(cols[0]))
    for col in cols:
        sd = col.std()
        z = np.abs((col - col.mean()) / sd) if sd > 1e-9 else np.zeros_like(col)
        out = np.maximum(out, z)
    return out


class _ScoreMethod:
    """A Gold method defined by a per-graph score function; ``evaluate`` ranks the injected node."""

    def __init__(self, method_id: str, score_fn) -> None:
        self.method_id = method_id
        self._score_fn = score_fn
        self.supports = {"gold_localization"}

    def scores(self, task: GoldLocalization) -> List[np.ndarray]:
        task.setup()
        return self._score_fn(task)

    def evaluate(self, task: GoldLocalization) -> Mapping[str, float]:
        return _scored(task, self.scores(task))


def _degrees(task: GoldLocalization) -> List[np.ndarray]:
    out = []
    for X, edges in task.graphs:
        deg = np.zeros(len(X))
        for s, d in edges.T:
            deg[int(s)] += 1
            deg[int(d)] += 1
        out.append(deg)
    return out


class GoldRandom:
    """Seed-averaged random floor; a single seed is too noisy at this run-length scale."""

    method_id = "random"
    supports = {"gold_localization"}

    def scores(self, task: GoldLocalization) -> List[np.ndarray]:
        task.setup()
        rng = np.random.RandomState(0)
        return [rng.rand(len(X)) for X, _ in task.graphs]

    def evaluate(self, task: GoldLocalization) -> Mapping[str, float]:
        task.setup()
        rows = []
        for seed in range(50):
            rng = np.random.RandomState(seed)
            draw = [rng.rand(len(X)) for X, _ in task.graphs]
            rows.append(_rank_metrics(np.concatenate(draw), task.groups, task.injected_row))
        mean = np.mean(rows, axis=0)
        return dict(zip(_METRIC_NAMES, (float(v) for v in mean)))


class GoldPyGOD:
    """PyGOD graph-AD (DOMINANT) on the step graph: a general detector not keyed to the injection."""

    method_id = "pygod (graph AD)"
    supports = {"gold_localization"}

    def scores(self, task: GoldLocalization) -> List[np.ndarray]:
        task.setup()
        return pygod_node_scores(task.graphs)

    def evaluate(self, task: GoldLocalization) -> Mapping[str, float]:
        return _scored(task, self.scores(task))


def gold_localization_methods() -> list:
    """The Gold board: random floor, two leak-ablation baselines, two keyed controls, the
    dependency-aware detector, and a general graph-AD detector."""
    return [
        GoldRandom(),
        _ScoreMethod("position", lambda t: [1.0 - X[:, 0] for X, _ in t.graphs]),
        _ScoreMethod("degree", _degrees),
        _ScoreMethod("has-dep (control)", lambda t: [(X[:, 2] > 0).astype(float) for X, _ in t.graphs]),
        _ScoreMethod("max-span (control)", lambda t: [X[:, 3] for X, _ in t.graphs]),
        _ScoreMethod("auditable (dep-anomaly)", lambda t: [_zmax(X[:, 2], X[:, 3]) for X, _ in t.graphs]),
        GoldPyGOD(),
    ]


def _ranks(task: GoldLocalization, per_graph: List[np.ndarray]) -> dict:
    """Per-run 1-based rank of the injected node."""
    scores = np.concatenate(per_graph)
    out = {}
    for gi in sorted(set(task.groups.tolist())):
        rows = np.where(task.groups == gi)[0]
        order = rows[np.argsort(-scores[rows], kind="stable")]
        out[gi] = int(np.where(order == task.injected_row[gi])[0][0]) + 1
    return out


def _kind_summary(ranks: dict, subset: list) -> tuple:
    rs = np.array([ranks[gi] for gi in subset], dtype=float)
    if not len(rs):
        return (float("nan"),) * 3
    return float(np.mean(rs == 1)), float(np.mean(rs <= 3)), float(np.mean(1.0 / rs))


def _row(label: str, o: tuple, s: tuple, d: tuple) -> str:
    """One breakdown row: label, then overall / stale-state / dropped-grounding Top-1/Top-3/MRR
    triples, aligned to the header columns."""
    return (
        f"  {label:24s}"
        f"{o[0]:.3f}/{o[1]:.3f}/{o[2]:.3f}".rjust(20)
        + f"{s[0]:.3f}/{s[1]:.3f}/{s[2]:.3f}".rjust(20)
        + f"{d[0]:.3f}/{d[1]:.3f}/{d[2]:.3f}".rjust(22))


def _eligible_positions(clean_steps: list, kind: str) -> List[int]:
    """Within-run step positions the injector could have targeted for ``kind``, read off the CLEAN
    structure (the a-priori eligible set, nothing a detector observes). dropped-grounding needs a
    prior dependency to remove; stale-state needs a prior dependency it can redirect earlier (a dep
    at position >= 1, so an earlier slot exists). This is a necessary condition for selection, so the
    true injected step is always inside the returned set."""
    idx_of = {s["idx"]: i for i, s in enumerate(clean_steps)}
    elig = []
    for i, s in enumerate(clean_steps):
        dep_pos = [idx_of[d] for d in s.get("deps", ()) if d in idx_of and idx_of[d] < i]
        ok = (len(dep_pos) >= 1) if kind == "dropped" else any(dp >= 1 for dp in dep_pos)
        if ok:
            elig.append(i)
    return elig


def _matched_pools(task: GoldLocalization) -> dict:
    """Per run, the selection-matched candidate pool (eligible within-run positions). The injected
    step is eligible by construction; union it in defensively so a pool can never exclude its own
    label."""
    pools = {}
    for gi in sorted(set(task.groups.tolist())):
        pos = _eligible_positions(task.clean_steps[gi], task.kinds[gi])
        if task.targets[gi] not in pos:
            pos = sorted(set(pos) | {task.targets[gi]})
        pools[gi] = pos
    return pools


def _tie_aware(pool_scores: np.ndarray, inj_local: int) -> tuple:
    """Exact expected Top-1 / Top-3 / reciprocal-rank for the injected step under uniform random
    tie-breaking. A constant-score baseline (has-dep over an all-eligible pool) therefore lands on
    the pool's random floor instead of winning on stable-sort order."""
    s = pool_scores
    si = s[inj_local]
    g = int(np.sum(s > si))               # steps that strictly outscore the injected one
    t = int(np.sum(s == si))              # steps tied with it (>= 1, includes itself)
    ranks = np.arange(g + 1, g + t + 1)   # the injected step occupies one of these uniformly
    return float(np.mean(ranks == 1)), float(np.mean(ranks <= 3)), float(np.mean(1.0 / ranks))


def _matched_floor(pools: dict, subset: list) -> tuple:
    """Analytic random floor inside the matched pools: a uniform rank over k eligible steps gives
    Top-1 1/k, Top-3 min(3,k)/k, MRR H_k/k, averaged over the subset of runs."""
    if not subset:
        return (float("nan"),) * 3
    t1 = np.mean([1.0 / len(pools[gi]) for gi in subset])
    t3 = np.mean([min(3, len(pools[gi])) / len(pools[gi]) for gi in subset])
    mr = np.mean([sum(1.0 / r for r in range(1, len(pools[gi]) + 1)) / len(pools[gi]) for gi in subset])
    return float(t1), float(t3), float(mr)


def _matched_metrics(task: GoldLocalization, per_graph: List[np.ndarray], pools: dict,
                     subset: list) -> tuple:
    """Tie-aware Top-1 / Top-3 / MRR for one method, ranking only within each run's matched pool."""
    if not subset:
        return (float("nan"),) * 3
    scores = np.concatenate(per_graph)
    acc = np.zeros(3)
    for gi in subset:
        rows = np.where(task.groups == gi)[0]
        pool_rows = rows[pools[gi]]
        acc += _tie_aware(scores[pool_rows], pools[gi].index(task.targets[gi]))
    return tuple(float(v) for v in acc / len(subset))


def gold_breakdown(task: GoldLocalization, methods: list) -> str:
    """Per-fault-kind Top-1 / Top-3 / MRR for every method with a per-graph score function. This is
    the headline view for Gold: the aggregate hides that stale-state and dropped-grounding behave
    completely differently."""
    task.setup()
    all_gi = sorted(set(task.groups.tolist()))
    stale = [gi for gi in all_gi if task.kinds[gi] == "stale"]
    dropped = [gi for gi in all_gi if task.kinds[gi] == "dropped"]
    lines = [f"\nGold per-fault breakdown (Top-1/Top-3/MRR), {len(stale)} stale + {len(dropped)} dropped:",
             f"  {'method':24s}{'overall':>20s}{'stale-state':>20s}{'dropped-grounding':>22s}"]
    for m in methods:
        if not hasattr(m, "scores") or m.method_id == "random":
            continue  # random's seed-averaged floor is on the board; its per-seed row would mislead
        ranks = _ranks(task, m.scores(task))
        lines.append(_row(m.method_id, _kind_summary(ranks, all_gi),
                          _kind_summary(ranks, stale), _kind_summary(ranks, dropped)))
    return "\n".join(lines)


def gold_matched_breakdown(task: GoldLocalization, methods: list) -> str:
    """Degree-matched (selection-matched) leakage control: re-rank each method only among the steps
    the injector could have targeted for that run's fault kind. The full-pool board carries a
    construction leak (the injected step always has a dependency, so detect-the-eligible baselines
    lift for free). Inside the matched pool the eligibility / degree artifact is held constant, so
    has-dep and degree fall to the matched random floor; a genuine dependency signal is what
    survives. Reported per fault kind, tie-aware so constant-score baselines do not win on sort
    order (see ``_tie_aware``)."""
    task.setup()
    pools = _matched_pools(task)
    all_gi = sorted(set(task.groups.tolist()))
    stale = [gi for gi in all_gi if task.kinds[gi] == "stale"]
    dropped = [gi for gi in all_gi if task.kinds[gi] == "dropped"]
    pool_sz = float(np.mean([len(pools[gi]) for gi in all_gi]))
    lines = [
        f"\nGold degree-matched control (rank within the injector's eligible pool only, "
        f"mean {pool_sz:.1f} candidates/run, tie-aware): Top-1/Top-3/MRR",
        f"  {'method':24s}{'overall':>20s}{'stale-state':>20s}{'dropped-grounding':>22s}",
        _row("random (matched)", _matched_floor(pools, all_gi),
             _matched_floor(pools, stale), _matched_floor(pools, dropped)),
    ]
    for m in methods:
        if not hasattr(m, "scores") or m.method_id == "random":
            continue  # the matched floor above is the reference; a per-seed random row would mislead
        sc = m.scores(task)
        lines.append(_row(m.method_id,
                          _matched_metrics(task, sc, pools, all_gi),
                          _matched_metrics(task, sc, pools, stale),
                          _matched_metrics(task, sc, pools, dropped)))
    return "\n".join(lines)


def gold_report(task: GoldLocalization) -> str:
    """Distributional-validity check: paired clean-versus-injected run-level statistics. Stale-state
    preserves edge count but shifts max-span by construction; dropped-grounding removes one valid
    dependency edge by construction. Both shifts are reported rather than hidden."""
    task.setup()

    def stats(step_lists):
        edges, spans = [], []
        for s in step_lists:
            X, e = _step_graph(s)
            edges.append(int(e.shape[1]))
            spans.append(float(X[:, 3].max()) if len(X) else 0.0)
        return np.array(edges, dtype=float), np.array(spans, dtype=float)

    ce, cs = stats(task.clean_steps)
    ie, isp = stats(task.steps)  # paired, same runs, valid-edge count both sides
    stale = np.array([k == "stale" for k in task.kinds])
    dropped = np.array([k == "dropped" for k in task.kinds])
    shifted = int(np.sum(isp > cs))

    def edge_line(name, mask):
        return (f"  valid dep-edges ({name}): mean {ce[mask].mean():.1f} -> {ie[mask].mean():.1f} "
                f"(delta {(ie[mask] - ce[mask]).mean():+.1f})")

    return ("Gold distributional check (paired clean -> injected, run-level):\n"
            f"{edge_line('all', np.ones(len(task.kinds), dtype=bool))}\n"
            f"{edge_line('stale-state', stale)}\n"
            f"{edge_line('dropped-grounding', dropped)}\n"
            f"  max dep-span      : mean {cs.mean():.1f} -> {isp.mean():.1f}, "
            f"p95 {np.percentile(cs, 95):.1f} -> {np.percentile(isp, 95):.1f} "
            f"({shifted}/{len(cs)} runs increased)\n"
            "  Edge count is unchanged for stale-state and decreases by one for dropped-grounding; "
            "stale-state also lengthens max-span by construction. These run-level shifts are "
            "reported, not hidden; localization still requires finding the step.")
