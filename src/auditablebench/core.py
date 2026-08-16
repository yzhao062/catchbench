"""The five-layer plug-in contract (core interfaces).

One scenario is one ``Task``; a ``Method`` evaluates a ``Task``; ``RunPipeline`` runs every
valid ``(Task, Method)`` pair and emits leaderboard rows. The POST localization seed runs
through these interfaces, and LIVE and PRE Tasks plug into the same contract.

Under this contract, a ``Method`` sees the whole ``Task`` and returns its metric dict.
Supervised baselines prevent in-sample scoring by running out-of-sample cross-validation inside
``evaluate``: grouped by run for step-level localization and stratified over runs for run-level
detection, where each run is one row.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Task(Protocol):
    """One benchmark scenario: its data, its labels, and its metric.

    ``pillar`` is "PRE", "LIVE", or "POST"; ``granularity`` is "run", "step", "edge", or
    "plan". ``setup`` loads the corpus and is idempotent.
    """

    task_id: str
    pillar: str
    granularity: str
    dataset: str

    def setup(self) -> None: ...


@runtime_checkable
class Method(Protocol):
    """A leaderboard entry. ``supports`` lists the ``task_id``s it can run."""

    method_id: str
    supports: set

    def evaluate(self, task: Task) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class ResultRow:
    """One method's score on one task."""

    pillar: str
    task: str
    dataset: str
    method: str
    metrics: Mapping[str, float]


class RunPipeline:
    """Run every valid ``(Task, Method)`` pair and collect leaderboard rows."""

    def __init__(self, tasks: Sequence[Task], methods: Sequence[Method]) -> None:
        self.tasks = list(tasks)
        self.methods = list(methods)

    def run(self) -> list[ResultRow]:
        rows: list[ResultRow] = []
        for task in self.tasks:
            for method in self.methods:
                if task.task_id not in method.supports:
                    continue  # only valid (Task, Method) pairs run
                rows.append(
                    ResultRow(
                        task.pillar, task.task_id, getattr(task, "dataset", ""),
                        method.method_id, method.evaluate(task),
                    )
                )
        return rows

    @staticmethod
    def leaderboard(rows: Sequence[ResultRow]) -> str:
        """A text leaderboard, one block per task, methods in input order."""
        by_task: dict[tuple[str, str, str], list[ResultRow]] = defaultdict(list)
        for row in rows:
            by_task[(row.pillar, row.task, row.dataset)].append(row)
        out: list[str] = []
        for (pillar, task, dataset), task_rows in by_task.items():
            metric_keys = list(task_rows[0].metrics.keys())
            label = f"{task} :: {dataset}" if dataset else task
            mw = max([len("method")] + [len(row.method) for row in task_rows])  # fit the widest id
            out.append(f"\n[{pillar}] {label}")
            out.append(f"  {'method':{mw}s}" + "".join(f"{k:>10s}" for k in metric_keys))
            for row in task_rows:
                out.append(
                    f"  {row.method:{mw}s}"
                    + "".join(f"{row.metrics[k]:>10.3f}" for k in metric_keys)
                )
        return "\n".join(out)
