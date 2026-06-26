# AuditableBench

The benchmark for finding and attributing agent failures over real agent traces.

Given a multi-agent run, two questions decide whether you can trust it: did it fail, and which
step broke it. AuditableBench scores both against labels the field already accepts, so a method
earns its place on the board instead of grading its own homework. Agent auditing has the tools
and the methods; it has not had its own benchmark. This is that benchmark, in the spirit of
ADBench for tabular anomaly detection and BOND for graph anomaly detection.

## The Dataset Is the Asset

A benchmark is worth what its data is worth, because every method runs on the same data. The
value here is a growing collection of agent traces typed into one dependency graph and paired
with trusted labels: human fault attribution from Who&When, and run-level outcomes from SWE-Gym
and tau-bench. New methods plug into a fixed `Task` and are scored the same way as every other
entry. The library [`auditable`](https://github.com/yzhao062/auditable) is what people build on;
this benchmark is the public arena that keeps it honest and turns the data flywheel.

## The Auditable Ecosystem

| Cell | Role | Asset |
|---|---|---|
| Tool | the SDK people build on | [`auditable`](https://github.com/yzhao062/auditable) |
| Evidence | the benchmark methods compete on | **AuditableBench** (this repo) |
| Knowledge | the curated reading list | `awesome-auditable-ai` |
| Method | the research engine the family cites | GRADE |

## Paper

The benchmark is written up alongside the code: *AuditableBench: A Benchmark for Auditing Agent
Failures Across the PRE / LIVE / POST Lifecycle* (ICLR 2027 target, in progress). Every board in this
README is a table in the paper, generated from `run.py` so the two cannot drift. The preprint link
goes here on release.

## Posture: Branded Name, Neutral Content

A benchmark named after one of its own entrants reads as self-serving. AuditableBench takes the
branded name and pays the credibility back in the content:

- **The labels are trusted.** Localization scores against Who&When's human-verified attribution;
  detection scores against SWE-Gym and tau-bench outcome labels. The arena does not invent its
  own ground truth.
- **`auditable` is one baseline, not the referee.** It sits on the board next to a random floor,
  a run-size baseline, PyOD on flattened features (the field's standard shallow tabular detector),
  a supervised reference, and a full-feature ceiling. It wins or loses in public, and it has no
  obligation to win.

## Pillars: PRE, LIVE, POST

AuditableBench is organized along the three windows of an agent run, mirroring `auditable`:

- **PRE** (before the run): audit the harness and the plan. Over-privileged tools, missing
  guardrails. *(planned)*
- **LIVE** (during the run): catch failure early from a streaming prefix of the trace.
  Time-to-detection, early-warning AUC. *(planned)*
- **POST** (after the run): forensics on the finished trace. Did it fail, and which step broke it.
  **This is what v1 ships.**

## The v1 Boards

POST ships three task families (localization, detection, and Gold). Run `python run.py` to reproduce
every number below.

### Fault Localization on Who&When

Rank the steps of a failed run by how likely each is the fault, scored against the human
`mistake_step`. 126 failed runs, 1099 steps, 11% of steps are faults.

| Method | Top-1 | Top-3 | MRR |
|---|---|---|---|
| random | 0.119 | 0.346 | 0.324 |
| `auditable` (blast share) | 0.159 | 0.516 | 0.407 |
| position prior | 0.159 | 0.516 | 0.407 |
| PyGOD (graph AD, DOMINANT) | 0.151 | 0.492 | 0.394 |
| structure (supervised) | **0.211** | **0.614** | **0.454** |

How to read it. The early-fault prior (faults cluster early in a run) is the honest floor, well
above random. `auditable`'s public structural signal, the blast share (how much of the run rests
on a step, via `downstream_reach` over the dependency graph), lands exactly on that prior here.
That is expected: this corpus assumes every step depends on all prior steps, so blast is monotone
in position and the two rankings coincide. An off-the-shelf graph anomaly detector (PyGOD's
DOMINANT, scoring each step by graph-autoencoder reconstruction error) beats random but trails the
position prior: structural anomaly is not the same as fault, so naive graph-AD does not localize
the mistake on its own. The supervised execution-structure ranker (agent activity and handoff
centrality) is what localizes beyond the prior. To separate a dependency
signal from raw position the benchmark needs traces where the two diverge, runs with real
long-range dependencies rather than a full-context assumption. That gold-edge corpus is the
planned next dataset.

### Failure Detection on SWE-Gym and tau-bench

Predict whether a run failed, scored by ROC-AUC. The keystone question: does the dependency
structure of a run predict failure beyond its raw size? Compare `auditable (structure)` against
`size (flat)`.

SWE-Gym, 376 runs (188 failed, 188 resolved):

| Method | ROC-AUC |
|---|---|
| random | 0.483 |
| size (flat) | 0.663 |
| PyOD-flatten (ECOD) | 0.765 |
| PyGOD (graph AD, DOMINANT) | 0.487 |
| `auditable` (structure) | 0.804 |
| full (reference) | **0.819** |

tau-bench (MIT), 660 runs (380 failed, 280 resolved):

| Method | ROC-AUC |
|---|---|
| random | 0.501 |
| size (flat) | 0.583 |
| PyOD-flatten (ECOD) | 0.571 |
| PyGOD (graph AD, DOMINANT) | 0.507 |
| `auditable` (structure) | 0.614 |
| full (reference) | **0.627** |

How to read it. The size-normalized dependency block beats the size-only baseline on both
corpora (+0.141 on SWE-Gym, +0.031 on tau-bench), so the structural signal predicts failure
beyond run length, and it does so in the same direction across two independent domains. The
magnitude is domain-dependent: strong on SWE-Gym, modest on tau-bench, which is the benchmark
doing its job of telling the two domains apart rather than rewarding one trick. PyOD (ECOD)
beating the linear size model on SWE-Gym (0.765 over 0.663) is its own signal: failure is
non-monotone in run length there (both very short and very long runs fail), which a tail-based
detector catches and a linear one misses. The dependency structure still adds on top of it.

The two unsupervised detectors split sharply. PyOD on flat size features is moderate, and strong on
SWE-Gym. PyGOD's graph autoencoder on structure-only node features sits near or below random on both
corpora, and its weak correlation with failure even flips sign between them. Reading the typed graph
with an off-the-shelf graph anomaly detector does not, by itself, find failures; the task-aware
dependency features do. That gap is the benchmark's point: the structure has to be used, not merely
be present.

### Baselines and Lineage

The graph-AD baseline is PyGOD's DOMINANT (Ding et al., 2019, *Deep Anomaly Detection on Attributed
Networks*, SDM), an unsupervised graph autoencoder that scores nodes by reconstruction error. That
is the same mechanism as GUARDIAN (arXiv:2505.19234), which safeguards multi-agent collaborations
with a reconstruction-error temporal graph autoencoder, so this board represents that family rather
than reimplementing it. G-Safeguard (Wang et al., 2025, arXiv:2502.11127) is a different design: a
supervised GNN trained to detect injected adversarial agents on the multi-agent utterance graph. Its
threat model is attack, not task failure, so it joins the benchmark with the fault-injection
scenarios (where there is an attack to detect), not these outcome-label boards.

### Gold: Injected Faults on Real Runs

The boards above borrow labels (human attribution, run outcomes). Gold is the benchmark's own data
contribution: plant a known fault in a real run and ask whether a method points to it. 188 clean
SWE-Gym runs, one injected fault each, half stale-state and half dropped-grounding; the label is the
injection site. Read the board PER FAULT KIND, because the two fault types behave completely
differently and the aggregate hides it.

| Method | overall Top-1 | stale-state Top-1 | dropped-grounding Top-1 |
|---|---|---|---|
| random (seed-averaged) | 0.034 | -- | -- |
| position (leak check) | 0.000 | 0.000 | 0.000 |
| degree (leak check) | 0.106 | 0.181 | 0.032 |
| has-dep (control) | 0.096 | 0.191 | 0.000 |
| max-span (control) | 0.335 | 0.670 | 0.000 |
| `auditable` (dep-anomaly) | 0.335 | **0.670** | 0.000 |
| PyGOD (graph AD) | 0.000 | 0.000 | 0.000 |

The injector can only target a step that has a dependency, so the injected step always carries one.
That makes the full-pool table leak-prone: any detect-the-eligible baseline lifts for free. The
degree-matched control re-ranks every method within only the steps the injector could have picked for
that run's fault kind (mean 8.5 candidates per run), holding eligibility and degree constant. The
random floor rises accordingly, and ties are broken in expectation so a constant-score baseline lands
on the floor instead of winning on sort order.

| Method (degree-matched pool) | overall Top-1 | stale-state Top-1 | dropped-grounding Top-1 |
|---|---|---|---|
| random (matched floor) | 0.232 | 0.238 | 0.225 |
| position | 0.218 | 0.191 | 0.245 |
| degree | 0.168 | 0.268 | 0.067 |
| has-dep | 0.135 | 0.238 | 0.032 |
| max-span | 0.354 | 0.676 | 0.032 |
| `auditable` (dep-anomaly) | 0.354 | **0.676** | 0.032 |
| PyGOD (graph AD) | 0.235 | 0.247 | 0.223 |

How to read it. Stale-state faults are detectable: in the full pool a dependency-span detector
localizes them at 0.670 Top-1, far above the 0.034 random floor. The degree-matched control shows
that lift is the fault, not an artifact: `has-dep` falls to exactly the matched floor (0.238 stale, a
constant score over an all-eligible pool), `degree` drops to the floor too (0.268 stale, 0.168
overall), and the dependency-span signal survives at 0.676 stale against a 0.238 floor. The
eligibility lift was target selection; the span signal is the fault signature. Dropped-grounding is a
different story: no baseline localizes it in either pool (about the floor), because removing one
dependency among many leaves no signal the current detectors catch, an honest open problem rather
than a result to average away. One framing caution remains: the dependency-aware detector is
essentially the raw `max-span` control, so it is keyed to the stale-state mechanism and sits next to
that control to make the keying visible, not framed as a general detector. A generic graph
autoencoder (PyGOD) sits at the matched floor for both fault types.

### The Injection: Where It Holds Up, and Where It Does Not Yet

The injection is the contribution, so its checks are reported as first-class results, including the
open ones, following BOND's discipline that the synthesis must be defensible:

- **Grounded faults, one strong half.** Stale-state (redirect a dependency to a superseded earlier
  step) is localized at 0.670 Top-1. Dropped-grounding (remove a required dependency) is realized but
  not yet localized by any baseline; it needs a better detector or a stronger substrate.
- **Leakage check, controlled for stale-state.** `position` stays at the random floor. In the full
  pool `degree` and a `has-dep` eligibility baseline both lift above it, but that is target selection:
  the injector only corrupts steps that have dependencies. Ranking within the degree-matched eligible
  pool removes the artifact. `has-dep` lands on the matched floor (0.238 stale) and `degree` falls to
  it (0.268 stale, 0.168 overall), while the dependency-span detector holds at 0.676 stale against a
  0.238 floor. The stale-state lift survives the control, so it is the fault signature rather than the
  construction. dropped-grounding stays at the floor in both pools, which is the open detection
  problem, not a leak.
- **Distributional validity, reported honestly.** Stale-state preserves the valid dependency-edge
  count, but dropped-grounding removes exactly one valid dependency edge, so the aggregate mean shifts
  from 8.5 to 8.0; treat edge count as a reported run-level shift for the dropped half, not as matched.
  Stale-state lengthens the run-level max dependency span by construction (mean 8.6 to 11.1, 63 of 188
  runs increased). These shifts are reported, not hidden, and localization still requires finding the
  step.
- **Labels are the injection site**, correct by construction and independent of any detector.
- **Characterized caveat.** SWE-Gym dependencies are inferred, not gold value-flow, so a redirected
  edge is a dependency-misattribution proxy for a true stale read. The nearer fix is a
  dropped-grounding detector; the airtight substrate is a named-value corpus where writes and reads
  are explicit, with a human-audited validation slice (Cohen's kappa).

## Tasks

- **Fault localization** (step level, shipped): name the faulting step. Top-1, Top-3, MRR
  against Who&When human attribution.
- **Failure detection** (run level, shipped): predict whether a run fails. ROC-AUC against
  SWE-Gym and tau-bench resolved / unresolved outcomes.
- **Gold fault injection** (step level, shipped): plant a known fault in a real run and localize
  it. Top-1, Top-3, MRR against injection-site labels; the benchmark's own data contribution.
- **Live early warning** (run level, planned): flag a failing run from a streaming prefix.
- **Harness and plan audit** (PRE, planned): flag over-privilege and missing guardrails.

## Run It

```bash
python run.py
```

The runner reuses GRADE's verified Who&When and SWE-Gym / tau-bench loaders and evaluations for
the reference methods, and computes the `auditable` baseline through `auditable`'s own public
kernel (`SessionGraph` plus `downstream_reach`). SWE-Gym and tau-bench download from the Hugging
Face Hub on first run. A packaged build will vendor the loaders and depend on `auditable` as a
normal dependency. The PyOD and PyGOD baselines add `pyod` and `pygod` (the latter needs `torch`,
`torch_geometric`, and a `pyg-lib` / `torch-sparse` backend); they are the only heavy dependencies
and load only when those baselines run. The GRADE loaders also need the dataset stack
(`huggingface_hub`, `pyarrow`, `datasets`): install GRADE's own `[experiments]` extra
(`pip install -e <grade-checkout>[experiments]`), or the `auditablebench[dev-seed]` mirror.

## How a Method Plugs In

One scenario is one `Task`; a `Method` scores a `Task`; `RunPipeline` runs every valid
`(Task, Method)` pair and prints the leaderboard. A new entry is one small class:

```python
class MyDetector:
    method_id = "my-detector"
    supports = {"post_detection"}          # the task_ids it runs on

    def evaluate(self, task):
        task.setup()                        # loads the corpus once
        scores = my_model(task.layers["flat"])
        return {"roc_auc": roc_auc(task.y, scores)}
```

The same `Task` feeds every method, so the comparison is apples to apples and the dataset, not
the method, is the fixed point. See `src/auditablebench/core.py` for the contract and
`detection.py` / `post.py` for the shipped baselines.

## Status and Release

v1 is a local working build of the POST pillar: localization, detection, and the Gold injection
board. The repository goes public together with the benchmark paper, so the framing and the
leaderboard arrive at once and the angle is established in print before it is copied. That is a release-timing choice, not a secrecy one: nothing here is
withheld. GRADE is public on arXiv, `auditable`'s kernel is public, and the labels are public.

The benchmark's own data contribution, the Gold board, now ships its first slice: faults injected
into real runs with injection-site labels (the standard move when real labels are scarce; BOND
injects anomalies into real graphs for the same reason), with the stale-state half leakage-controlled
by a degree-matched eligible-pool ranking. The airtight upgrade is a named-value corpus where a stale
read is unambiguous, plus a dropped-grounding detector and a human-audited validation slice, which
together grow the dataset past the public corpora it starts from.
