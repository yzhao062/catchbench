"""POST pillar (forensics): the localization scenario, the v1 seed board.

Given a failed multi-agent run, rank its steps by how likely each is the fault, and score the
ranking against Who&When's human ``mistake_step`` (Top-1, Top-3, MRR). Trusted labels;
``auditable`` is one baseline among several, with no obligation to win.

Methods on the board:
  - random              : a floor (rank steps at random; seed-averaged).
  - auditable (blast)    : ``auditable``'s public structural signal, the blast share
                          (``downstream_reach`` over the dependency graph) built with
                          ``auditable``'s own kernel. On this full-context corpus every step
                          depends on all prior steps, so blast is monotone in position and this
                          baseline coincides with the position prior. That is the honest result,
                          and it is why a long-range gold-edge corpus is the planned next lever.
  - position            : the early-fault prior (faults cluster early), one monotone feature.
  - exec-rank (sup.)     : GRADE's supervised execution-feature ranker (agent activity and
                          handoff centrality), the published reference.

This reuses GRADE's verified Who&When loader and localization eval for the reference methods,
and adds the ``auditable`` baseline through ``auditable``'s public kernel, all behind the
``Task`` / ``Method`` interfaces in ``core.py``.
"""
from __future__ import annotations

from functools import cached_property
from typing import Mapping

from catchbench import _reuse  # noqa: F401  side effect: sets sys.path for grade + auditable

import numpy as np  # noqa: E402

from agent_failure_localization import (  # noqa: E402  GRADE's verified Who&When eval
    load_failed_runs,
    _pool,
    _rank_metrics,
    _seed_metrics,
    _step_matrix,
)
from auditable.graph import downstream_reach  # noqa: E402  auditable's public kernel
from auditable.graph.session import DependencyEdge, SessionGraph, Step  # noqa: E402
from catchbench.llm_judge import load_judge_runs  # noqa: E402  the judges' own Who&When records

_METRIC_NAMES = ("top1", "top3", "mrr")


class PostLocalization:
    """POST / localization Task over Who&When failed runs with human ``mistake_step`` labels.

    After ``setup`` the task view is ``runs`` (typed steps and the run's dependency graph), ``X``
    (structural features), ``y`` / ``groups`` / ``mistake_row`` (the labels and the pooling arrays a
    metric needs), and ``step_texts`` for a method that reads the trace itself rather than its
    structure.
    """

    task_id = "post_localization"
    pillar = "POST"
    granularity = "step"
    dataset = "whoandwhen"

    def __init__(self) -> None:
        self._loaded = False

    def setup(self) -> None:
        if self._loaded:
            return
        self.runs = load_failed_runs()
        self.X, self.y, self.groups, self.mistake_row = _pool(self.runs)
        self._loaded = True

    @cached_property
    def step_texts(self) -> tuple[tuple[str, ...], ...]:
        """The raw text of every step, aligned to ``runs``: ``step_texts[i][j]`` is what the agent
        said or the tool returned at ``runs[i]["steps"][j]``.

        The LLM-judge rows lead this board, and they reach the trace through their own loader. A
        view with structural features alone would leave an outside text-reading localizer, the
        obvious entrant here, unable to read the input those baselines read. These are the judges'
        own strings: ``llm_judge.load_judge_runs`` returns the same 126 Who&When records at the same
        pinned revision, filtered and ordered to match ``runs``. That loader is memoized, so this
        adds no corpus, no download, and no second parse on a board run.

        Step content only. A Who&When history entry holds ``content``, ``role``, and ``name``. The
        annotations (``mistake_step``, ``mistake_agent``, ``mistake_reason``, ``ground_truth``) are
        task-level fields that stay in the loader and are unreachable from what is returned here.

        Built on first read rather than in ``setup``, so a structural method never pays for it. The
        1099 strings run about 1.5 MB across the 126 runs, and they are the loader's own string
        objects, so this holds references to them rather than copies.
        """
        self.setup()
        judge_runs = load_judge_runs()
        aligned = len(judge_runs) == len(self.runs) and all(
            len(judged["steps"]) == len(scored["steps"])
            # One text per step. The step lists can agree while the text list is short, and that
            # would shift every text after the gap onto the wrong step.
            and len(judged["texts"]) == len(judged["steps"])
            # The decisive-step label is the only field both loaders carry, so it is the only
            # identity available. It separates two runs whose step signatures coincide, which the
            # signature check alone does not: swapping two same-shaped runs passes that check and
            # attaches each run's words to the other. Two runs identical in signature AND in
            # decisive step would still pass, and closing that needs a shared record key, which
            # load_failed_runs does not currently expose.
            and judged["mistake"] == scored["mistake"]
            and all(a["idx"] == b["idx"] and a["agent"] == b["agent"]
                    for a, b in zip(judged["steps"], scored["steps"]))
            for judged, scored in zip(judge_runs, self.runs)
        )
        if not aligned:
            raise RuntimeError(
                "Who&When text records do not line up with the scored runs, so step text would be "
                "attributed to the wrong steps. load_judge_runs and load_failed_runs must select "
                "the same records in the same order."
            )
        return tuple(tuple(run["texts"]) for run in judge_runs)

    def corpus_line(self) -> str:
        self.setup()
        n_steps = sum(len(r["steps"]) for r in self.runs)
        return (f"Who&When: {len(self.runs)} failed runs, {n_steps} steps "
                f"({len(self.runs) / n_steps:.0%} faults), human mistake_step labels.")


class _SeedModel:
    """A reference method scored through GRADE's seed-averaged grouped CV (random / position /
    structure). ``model`` is the GRADE model key; the metric is averaged over GRADE's seeds."""

    def __init__(self, method_id: str, model: str) -> None:
        self.method_id = method_id
        self.model = model
        self.supports = {"post_localization"}

    def evaluate(self, task: PostLocalization) -> Mapping[str, float]:
        task.setup()
        per_seed = _seed_metrics(task.X, task.y, task.groups, task.mistake_row, self.model)
        mean = per_seed.mean(axis=0)  # (5 seeds, 3 metrics) -> (3,)
        return dict(zip(_METRIC_NAMES, (float(v) for v in mean)))


def _auditable_blast(run) -> np.ndarray:
    """Per-step blast share from ``auditable``'s kernel: build the run's graph with
    ``auditable``'s own ``SessionGraph`` (full-context dependency edges, as the corpus assumes)
    and read ``downstream_reach`` per step."""
    steps = [
        Step(
            idx=s["idx"],
            kind=s["kind"],
            agent=s["agent"],
            deps=[DependencyEdge(src_idx=j) for j in range(s["idx"])],
        )
        for s in run["steps"]
    ]
    graph = SessionGraph.from_steps(steps).to_networkx()
    return np.array([downstream_reach(graph, s["idx"]) for s in run["steps"]], dtype=float)


class AuditableBlast:
    """``auditable``'s public structural signal as a localization baseline (deterministic)."""

    method_id = "auditable (blast)"
    supports = {"post_localization"}

    def evaluate(self, task: PostLocalization) -> Mapping[str, float]:
        task.setup()
        scores = np.concatenate([_auditable_blast(r) for r in task.runs])
        metrics = _rank_metrics(scores, task.groups, task.mistake_row)
        return dict(zip(_METRIC_NAMES, (float(v) for v in metrics)))


class PyGODLocalization:
    """PyGOD graph-AD as a localization baseline: a DOMINANT autoencoder scores each step's
    anomaly from its features and the full-context dependency graph, and the most anomalous step is
    the predicted fault. Unsupervised and genuinely different from the position prior, the blast
    share, and the supervised ranker. Deterministic given the seed (CPU training)."""

    method_id = "pygod (graph AD)"
    supports = {"post_localization"}

    def evaluate(self, task: PostLocalization) -> Mapping[str, float]:
        task.setup()
        from catchbench.graph_ad import full_context_edges, pygod_node_scores

        graphs = [(_step_matrix(run), full_context_edges(len(run["steps"]))) for run in task.runs]
        per_run = pygod_node_scores(graphs)
        scores = np.concatenate(per_run)  # pooled order: runs in order, steps within run in order
        metrics = _rank_metrics(scores, task.groups, task.mistake_row)
        return dict(zip(_METRIC_NAMES, (float(v) for v in metrics)))


def post_localization_methods() -> list:
    """The seed board's baseline set, in display order."""
    return [
        _SeedModel("random", "random"),
        AuditableBlast(),
        _SeedModel("position", "position"),
        PyGODLocalization(),
        _SeedModel("exec-rank (sup.)", "structure"),
    ]
