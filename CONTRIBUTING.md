# Contributing to AuditableBench

AuditableBench welcomes new methods and corpora. A contribution must keep the board fair,
deterministic, and reproducible from the public repository.

## Adding a method

A leaderboard entry is one small class with three parts:

- `method_id`: a stable, unique name for the leaderboard row.
- `supports`: the set of exact task IDs the method can run on.
- `evaluate(task)`: a function that returns a dictionary from metric names to floats.

Call `task.setup()` before reading the task view. Do not form predictions from evaluation labels. A
supervised method may read labels only through an out-of-sample split in which every scored item is
held out from training.

This runnable example is the `RandomDetection` baseline from
`src/auditablebench/detection.py`, with its required imports:

```python
from typing import Mapping

import numpy as np

from auditablebench.detection import PostDetection


class RandomDetection:
    """The ROC-AUC floor: random run scores (averages to ~0.5)."""

    method_id = "random"
    supports = {"post_detection"}

    def evaluate(self, task: PostDetection) -> Mapping[str, float]:
        from sklearn.metrics import roc_auc_score

        task.setup()
        rng = np.random.RandomState(0)
        aucs = [roc_auc_score(task.y, rng.rand(len(task.y))) for _ in range(5)]
        return {"roc_auc": float(np.mean(aucs))}
```

Add the class to the relevant source module, then return an instance from that board's method
factory. `run.py` collects those factories. `RunPipeline` skips a task unless its ID appears in the
method's `supports` set.

The shipped task IDs and scores are:

| Task ID | Question and score |
|---|---|
| `post_localization` | Rank failed-run steps against Who&When's human `mistake_step`: Top-1, Top-3, and MRR. |
| `post_detection` | Predict failed versus resolved runs on SWE-Gym and tau-bench: ROC-AUC. |
| `gold_attribution` | Separate stale-state from dropped-grounding in paired Gold injections: ROC-AUC. |
| `gold_localization` | Rank steps against the Gold injection site: Top-1, Top-3, and MRR. |
| `live_streaming` | Predict failure from growing prefixes: Prefix-AUC and time to detection. |
| `live_stale_state` | Detect a stale-state injection online: true-positive rate at fixed false-positive budgets. |
| `pre_over_privilege` | Flag granted capabilities that the task does not need: precision, recall, and F1. |

The planned missing-guardrail audit has no shipped task ID. Do not place a planned name in
`supports`. Some detailed reports read a board-specific method such as `auc_curve` or `detail` in
addition to `evaluate`; follow the existing methods for that board.

## Adding a corpus or adapter

Put upstream file access and conversion in a loader or adapter, then expose it through a `Task` with
`task_id`, `pillar`, `granularity`, `dataset`, and an idempotent `setup()` method. A new corpus must
supply stable record IDs and ordering, source and license information, the trusted task-specific
labels, and the normalized representation required by the task. Graph tasks need step or run nodes,
typed dependency edges, and any aligned feature layers or grouping arrays used by their metric. PRE
corpora need normalized capability declarations and excess-set labels.

Only the adapter reads raw traces. A method must read the normalized task view, not open or parse raw
corpus files. A graph method written against that view must run on every corpus attached to the same
task without corpus-specific branches.

Every corpus must have a clear redistribution position. Document its source, license, conversion,
and any limits on redistribution.

## Determinism and model APIs

`python run.py` must score the complete board without an API call or credentials. If a method calls a
model API, run that call only in an opt-in cache-generation path. Commit its predictions under
`data/`, and make `evaluate` read that file. Follow the shipped formats and readers in
`data/llm_judge/` and `data/pre/llm_judge_method/`.

Seed local randomness. Stable code is not enough when predictions come from a remote model; the
committed prediction file is the scored artifact.

## Fixed corpus revisions

Corpora download from the Hugging Face Hub, and an upstream dataset can be rebuilt at the same name.
Every corpus loader must pin an immutable dataset revision. Do not load from `main`, `latest`, or an
unpinned default. The tau-bench loader is pinned to `382e57d`; new corpus adapters must follow the
same rule.

## Reporting a result honestly

Generate all reported numbers by running:

```bash
python run.py
```

Do not type numbers directly into the README or paper. A result must come from the runner with the
submitted code and committed caches. Report per-source results. If a board combines labels made by
more than one label process, also report each label process separately rather than only a pooled
score.

## Tests

Run the full suite under both hash seeds:

```bash
PYTHONHASHSEED=0 pytest tests
PYTHONHASHSEED=1 pytest tests
```

Both runs must pass. The second seed matters because scanner rules iterate over sets.

## Changes that will be declined

The project will decline:

- A method that reads evaluation labels to form its predictions.
- A corpus with no clear redistribution position or no fixed dataset revision.
- A result that cannot be regenerated from committed code and prediction caches.
