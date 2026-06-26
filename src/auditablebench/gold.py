"""AuditableBench-Gold: grounded fault injection over real agent run-graphs.

The contribution here is the injection itself, so it is built to hold up (BOND's discipline: the
synthesis must be defensible, not a convenient artifact) and reported honestly, including where it
does not yet clear the bar.

Faults are documented agent failure modes, realized on the inferred dependency layer:
  - stale-state read  : redirect one dependency from the latest event on a file to an EARLIER
                        (superseded) event on the SAME file. Signature: an unusually long span.
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


_CLEAN_CACHE: List[list] | None = None


def _load_clean_runs() -> List[list]:
    """Resolved (clean) SWE-Gym runs as step lists, keeping only runs with deps to corrupt. Cached so
    a multi-seed sweep re-injects without reloading the corpus; ``_inject`` copies each run before
    mutating, so the cached lists stay clean."""
    global _CLEAN_CACHE
    if _CLEAN_CACHE is None:
        runs = []
        for rec in swegym.load_runs():
            if not rec.get("resolved"):
                continue
            steps = rec["steps"]
            if len(steps) < 4 or not any(s.get("deps") for s in steps):
                continue
            runs.append(steps)
        _CLEAN_CACHE = runs
    return _CLEAN_CACHE


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
                if stale:  # redirect a dependency to an EARLIER event on the SAME file/resource
                    target_resource = steps[p].get("file")  # the file this tool step touches
                    cand = []  # (true dep idx, earlier same-file idxs that supersede it)
                    for d in deps:
                        d_pos = idx_of[d]
                        if target_resource is None or steps[d_pos].get("file") != target_resource:
                            continue  # only redirect a genuine same-file dependency
                        prior_same = [s["idx"] for s in steps[:d_pos]
                                      if s.get("file") == target_resource and s["idx"] not in steps[p]["deps"]]
                        if prior_same:
                            cand.append((d, prior_same))
                    if not cand:
                        continue  # no superseded same-file version exists; try the other fault
                    d, prior_same = cand[int(rng.randint(0, len(cand)))]
                    new_d = int(rng.choice(prior_same))  # an older version of the same state (stale read)
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


def _has_stale_target(steps: list, idx_of: dict, i: int) -> bool:
    """True if step ``i`` has a same-file dependency with an EARLIER superseded same-file event to
    redirect to: the exact precondition ``_inject`` uses to place a stale-state fault. Kept in lockstep
    with the injection so the matched pool is the injector's real eligible set, not an approximation."""
    s = steps[i]
    target_resource = s.get("file")
    if target_resource is None:
        return False
    for d in s.get("deps", ()):
        if d not in idx_of or idx_of[d] >= i:
            continue
        d_pos = idx_of[d]
        if steps[d_pos].get("file") != target_resource:
            continue
        if any(t.get("file") == target_resource and t["idx"] not in s.get("deps", ())
               for t in steps[:d_pos]):
            return True
    return False


def _eligible_positions(clean_steps: list, kind: str) -> List[int]:
    """Within-run step positions the injector could have targeted for ``kind``, read off the CLEAN
    structure (the a-priori eligible set, nothing a detector observes). dropped-grounding needs a
    prior dependency to remove; stale-state needs a same-file dependency with an earlier superseded
    same-file event to redirect to (mirroring ``_inject`` via ``_has_stale_target``). This is a
    necessary condition for selection, so the true injected step is always inside the returned set."""
    idx_of = {s["idx"]: i for i, s in enumerate(clean_steps)}
    elig = []
    for i, s in enumerate(clean_steps):
        dep_pos = [idx_of[d] for d in s.get("deps", ()) if d in idx_of and idx_of[d] < i]
        ok = (len(dep_pos) >= 1) if kind == "dropped" else _has_stale_target(clean_steps, idx_of, i)
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


def _redirect_stale(steps: list, idx_of: dict, p: int, rng) -> bool:
    """Apply the same-file stale redirect at position ``p`` (the same operation ``_inject`` uses for a
    stale-state fault); return True if a redirect was made. Mutates ``steps`` in place."""
    target_resource = steps[p].get("file")
    for d in [x for x in steps[p]["deps"] if x in idx_of and idx_of[x] < p]:
        d_pos = idx_of[d]
        if target_resource is None or steps[d_pos].get("file") != target_resource:
            continue
        prior_same = [s["idx"] for s in steps[:d_pos]
                      if s.get("file") == target_resource and s["idx"] not in steps[p]["deps"]]
        if prior_same:
            new_d = int(rng.choice(prior_same))
            steps[p]["deps"] = [new_d if x == d else x for x in steps[p]["deps"]]
            return True
    return False


def _inject_paired(runs: List[list], seed: int = 0):
    """Cause-attribution substrate: for each run that affords BOTH faults, return a stale-injected copy
    and a dropped-injected copy of the SAME run. Because the two classes are drawn from identical runs,
    a classifier cannot separate them on run identity, only on the fault signature, so there is no
    eligibility leak (the run-level analogue of the localization degree-matched control)."""
    rng = np.random.RandomState(seed)
    stale_steps, dropped_steps = [], []
    for original in runs:
        idx_of = {s["idx"]: i for i, s in enumerate(original)}
        stale_pos = [i for i in range(len(original)) if _has_stale_target(original, idx_of, i)]
        drop_pos = [i for i in range(len(original))
                    if any(d in idx_of and idx_of[d] < i for d in original[i].get("deps", ()))]
        if not stale_pos or not drop_pos:
            continue  # need both faults from the same run
        stale_copy = [dict(s, deps=list(s.get("deps", ()))) for s in original]
        if not _redirect_stale(stale_copy, idx_of, int(rng.choice(stale_pos)), rng):
            continue
        drop_copy = [dict(s, deps=list(s.get("deps", ()))) for s in original]
        dp = int(rng.choice(drop_pos))
        removable = [d for d in drop_copy[dp]["deps"] if d in idx_of and idx_of[d] < dp]
        drop_copy[dp]["deps"] = [x for x in drop_copy[dp]["deps"] if x != int(rng.choice(removable))]
        stale_steps.append(stale_copy)
        dropped_steps.append(drop_copy)
    return stale_steps, dropped_steps


def _run_maxspan(steps: list) -> float:
    X, _ = _step_graph(steps)
    return float(X[:, 3].max()) if len(X) else 0.0


def _run_edges(steps: list) -> float:
    _, e = _step_graph(steps)
    return float(e.shape[1])


class GoldAttribution:
    """POST / attribution Task: given a faulty run, attribute the CAUSE (stale-state vs
    dropped-grounding), scored as ROC-AUC for stale = 1. Paired design (``_inject_paired``): each run
    affording both faults contributes a stale copy and a dropped copy, so the label is the fault, not
    the run. The third POST task after localization and prediction."""

    task_id = "gold_attribution"
    pillar = "POST"
    granularity = "run"
    dataset = "swegym-gold"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._loaded = False

    def setup(self) -> None:
        if self._loaded:
            return
        stale_steps, dropped_steps = _inject_paired(_load_clean_runs(), seed=self.seed)
        self.runs = stale_steps + dropped_steps
        self.y = np.array([1] * len(stale_steps) + [0] * len(dropped_steps))  # 1 = stale-state
        self.n_pairs = len(stale_steps)
        self._loaded = True

    def corpus_line(self) -> str:
        self.setup()
        return (f"{self.dataset}: {self.n_pairs} runs affording both faults, each injected with a "
                f"stale-state and a dropped-grounding copy (paired), cause attribution by ROC-AUC.")


class _AttribScore:
    """An attribution method: a run-level score, ROC-AUC for separating stale (1) from dropped (0)."""

    def __init__(self, method_id: str, score_fn) -> None:
        self.method_id = method_id
        self._fn = score_fn
        self.supports = {"gold_attribution"}

    def evaluate(self, task: GoldAttribution) -> Mapping[str, float]:
        from sklearn.metrics import roc_auc_score

        task.setup()
        scores = np.array([self._fn(s) for s in task.runs])
        return {"roc_auc": float(roc_auc_score(task.y, scores))}


class _AttribRandom:
    """The attribution ROC-AUC floor (seed-averaged random scores)."""

    method_id = "random"
    supports = {"gold_attribution"}

    def evaluate(self, task: GoldAttribution) -> Mapping[str, float]:
        from sklearn.metrics import roc_auc_score

        task.setup()
        aucs = [roc_auc_score(task.y, np.random.RandomState(s).rand(len(task.y))) for s in range(5)]
        return {"roc_auc": float(np.mean(aucs))}


def gold_attribution_methods() -> list:
    """The attribution board: a random floor and the two mechanism-keyed run-level signals. A
    stale-state read lengthens the run's max dependency span; dropped grounding removes one edge. Each
    feature is keyed to one fault (the same caveat as the localization detectors), so they read the two
    faults as opposite, separable structural traces rather than detect one in the wild."""
    return [
        _AttribRandom(),
        _AttribScore("max-span (higher=stale)", _run_maxspan),
        _AttribScore("edge-count (higher=stale)", _run_edges),
    ]


def gold_attribution_robustness(methods: list, seeds=(0, 1, 2, 3, 4)) -> str:
    """Stability of the cause-attribution ROC-AUC across paired-injection seeds, matching the other
    Gold boards' robustness reporting so a reader does not have to trust a single draw."""
    from sklearn.metrics import roc_auc_score

    seeds = tuple(seeds)
    keep = [m for m in methods if hasattr(m, "_fn")]  # the scored features (the random floor is ~0.5)
    acc = {m.method_id: [] for m in keep}
    for seed in seeds:
        task = GoldAttribution(seed=seed)
        task.setup()
        for m in keep:
            scores = np.array([m._fn(s) for s in task.runs])
            acc[m.method_id].append(float(roc_auc_score(task.y, scores)))
    lines = [f"\nGold attribution seed robustness ({len(seeds)} paired-injection seeds, "
             f"ROC-AUC mean +/- std):"]
    for m in keep:
        mn, sd = _mean_std(acc[m.method_id])
        lines.append(f"  {m.method_id:28s}{mn:.3f}+/-{sd:.3f}")
    return "\n".join(lines)


def _mean_std(vals) -> Tuple[float, float]:
    arr = np.array(vals, dtype=float)
    return float(arr.mean()), float(arr.std())


def gold_seed_robustness(methods: list, seeds=(0, 1, 2, 3, 4)) -> str:
    """Stability of the Gold headline across injection seeds: the robustness question a synthetic
    injection has to answer. Per scoring method, stale-state and dropped-grounding Top-1 as mean +/-
    std over seeds; plus the matched-control stale max-span versus its floor, so the leak-controlled
    signal is shown stable, not a single-draw artifact. Clean runs are cached, so this re-injects K
    times without reloading the corpus."""
    seeds = tuple(seeds)
    keep = [m for m in methods if hasattr(m, "scores") and m.method_id != "random"]
    stale_t1 = {m.method_id: [] for m in keep}
    drop_t1 = {m.method_id: [] for m in keep}
    matched_span, matched_floor = [], []
    for seed in seeds:
        task = GoldLocalization(seed=seed)
        task.setup()
        all_gi = sorted(set(task.groups.tolist()))
        stale = [gi for gi in all_gi if task.kinds[gi] == "stale"]
        dropped = [gi for gi in all_gi if task.kinds[gi] == "dropped"]
        for m in keep:
            ranks = _ranks(task, m.scores(task))
            stale_t1[m.method_id].append(_kind_summary(ranks, stale)[0])
            drop_t1[m.method_id].append(_kind_summary(ranks, dropped)[0])
        pools = _matched_pools(task)
        span_m = next(m for m in keep if m.method_id == "max-span (control)")
        matched_span.append(_matched_metrics(task, span_m.scores(task), pools, stale)[0])
        matched_floor.append(_matched_floor(pools, stale)[0])
    lines = [f"\nGold seed robustness ({len(seeds)} injection seeds, Top-1 mean +/- std):",
             f"  {'method':24s}{'stale-state':>20s}{'dropped-grounding':>22s}"]
    for m in keep:
        sm, ss = _mean_std(stale_t1[m.method_id])
        dm, ds = _mean_std(drop_t1[m.method_id])
        lines.append(f"  {m.method_id:24s}{f'{sm:.3f}+/-{ss:.3f}':>20s}{f'{dm:.3f}+/-{ds:.3f}':>22s}")
    spm, sps = _mean_std(matched_span)
    flm, fls = _mean_std(matched_floor)
    lines.append(f"  matched stale max-span {spm:.3f}+/-{sps:.3f} vs floor {flm:.3f}+/-{fls:.3f} "
                 f"(leak-controlled signal, stable across seeds)")
    return "\n".join(lines)
