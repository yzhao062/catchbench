# AuditableBench

The benchmark for finding and attributing agent failures over real agent traces.

A run is audited at one of three moments, and what you can even ask is fixed by what you can see at
each: before it runs you have only the plan and harness (is it over-privileged?); while it runs you
have a growing prefix (is it about to fail?); after it runs you have the whole trace (which step broke
it, did it fail, what kind of fault was it). AuditableBench is built around those three information
states, and scores each question against labels the field already accepts, so a method earns its place
on the board instead of grading its own homework. Agent auditing has the tools and the methods; it has
not had its own benchmark. This is that benchmark, in the spirit of ADBench for tabular anomaly
detection and BOND for graph anomaly detection.

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

## Why PRE / LIVE / POST

The pillars are not three convenient buckets; they are the three information states a run passes
through, and the evidence available at each fixes which audit is possible. A method built for one
state cannot read another's evidence, so the pillars are separate tracks, not interchangeable views of
one dataset.

- **PRE** (only the plan and harness exist): the audit is static, for over-privilege and missing
  guardrails. **Over-privilege shipped; missing-guardrail planned.**
- **LIVE** (a growing prefix is visible, the outcome is not): the audit is predictive and runs under a
  false-alarm budget, for early warning from a streaming prefix and online detection. **Shipped.**
- **POST** (the complete trace and outcome are in hand): the audit is forensic, answering which step
  failed, whether it failed, and what kind of fault it was. **Shipped.**

Within a pillar, each board is the specific question an auditor asks at that state, paired with the
label that answers it and the metric the question implies (ranking questions use Top-k / MRR; a
yes-or-no question uses ROC-AUC; an online question uses true-positive rate at a fixed false-positive
budget, because a live detector is only useful if it does not flood the operator with false alarms).

## The Boards

POST answers its three forensic questions (which step, did it fail, what kind) plus the Gold injection
board; LIVE answers its two real-time questions (early warning from a prefix, online detection); and PRE
answers the deploy-gate question (is the declared harness over-privileged). Run `python run.py` to
reproduce every number. The POST localization, detection, and Gold boards and the PRE over-privilege
board are shown in full below; the cause-attribution board and the two LIVE boards are summarized at the
end and detailed in the paper.

### Fault Localization on Who&When

Rank the steps of a failed run by how likely each is the fault, scored against the human
`mistake_step`. 126 failed runs, 1099 steps, 11% of steps are faults.

| Method | Top-1 | Top-3 | MRR |
|---|---|---|---|
| **LLM-judge panel (all-at-once)** | | | |
| GPT-5.5 | **0.452** | 0.667 | **0.618** |
| Claude-Opus-4.8 | 0.421 | 0.698 | 0.605 |
| GPT-5.4 | 0.413 | 0.714 | 0.601 |
| DeepSeek-R1 | 0.405 | **0.754** | 0.606 |
| Gemini | 0.357 | 0.722 | 0.572 |
| Qwen3-32B | 0.349 | 0.659 | 0.541 |
| GPT-oss-20B | 0.333 | 0.595 | 0.521 |
| Llama-3.3-70B | 0.333 | 0.579 | 0.515 |
| Gemma-3-12B | 0.206 | 0.524 | 0.427 |
| Mistral-Small | 0.135 | 0.421 | 0.363 |
| Nova-Micro | 0.127 | 0.397 | 0.342 |
| **Structural / baseline (no LLM)** | | | |
| structure (supervised) | 0.211 | 0.614 | 0.454 |
| `auditable` (blast share) | 0.159 | 0.516 | 0.407 |
| position prior | 0.159 | 0.516 | 0.407 |
| PyGOD (graph AD, DOMINANT) | 0.151 | 0.492 | 0.394 |
| random | 0.119 | 0.346 | 0.324 |

How to read it. The field-standard control is to ask a strong LLM directly: show it the failed
trace and have it name the decisive step. We benchmark that as an 11-model panel, frontier models
through the NAIRR gateway and open-weights plus small proprietary models through AWS Bedrock, each prompted once (Who&When's
all-at-once protocol); predictions are cached and committed, so the board scores them with no API
call. A frontier LLM judge is the strongest localizer here (GPT-5.5 at 0.452), which is expected
with the full trace in hand. The capability gradient is clean: frontier judges at 0.41 to 0.45,
strong open models (Llama, Qwen) at 0.33 to 0.35, down to small models that fall below the trivial
position prior (Mistral, Nova near 0.13), so "just ask an LLM" is only as good as the LLM. Among
methods that use no LLM, the supervised execution-structure ranker beats the position prior cheaply
and deterministically; `auditable`'s blast share lands on the prior because this corpus assumes every
step depends on all prior steps, and an off-the-shelf graph-AD (PyGOD DOMINANT) trails it. The
structural methods are not built to win this board, and a third-party method topping it is the
benchmark working as intended. Their value is that they need no model call and they apply in the LIVE
and online settings where a post-hoc full-trace judge cannot run. (Separating a dependency signal
from raw position needs traces where the two diverge: real long-range dependencies rather than a
full-context assumption, the planned gold-edge corpus.)

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
| PyGOD-DOMINANT (graph AD) | 0.487 |
| GUARDIAN (recon-AE) | 0.767 |
| `auditable` (structure) | 0.804 |
| full (reference) | 0.819 |
| G-Safeguard (supervised GNN) | **0.828** |

tau-bench (MIT), 660 runs (380 failed, 280 resolved):

| Method | ROC-AUC |
|---|---|
| random | 0.501 |
| size (flat) | 0.583 |
| PyOD-flatten (ECOD) | 0.571 |
| PyGOD-DOMINANT (graph AD) | 0.507 |
| GUARDIAN (recon-AE) | 0.555 |
| `auditable` (structure) | 0.614 |
| full (reference) | **0.627** |
| G-Safeguard (supervised GNN) | 0.578 |

How to read it. The size-normalized dependency block beats the size-only baseline on both
corpora (+0.141 on SWE-Gym, +0.031 on tau-bench), so the structural signal predicts failure
beyond run length, and it does so in the same direction across two independent domains. The
magnitude is domain-dependent: strong on SWE-Gym, modest on tau-bench, which is the benchmark
doing its job of telling the two domains apart rather than rewarding one trick. PyOD (ECOD)
beating the linear size model on SWE-Gym (0.765 over 0.663) is its own signal: failure is
non-monotone in run length there (both very short and very long runs fail), which a tail-based
detector catches and a linear one misses. The dependency structure still adds on top of it.

The unsupervised anomaly detectors split sharply from the task-aware ones. AuditableBench runs a
wider arena behind the headline table: the PyOD tabular family (Isolation Forest, KNN, LOF, COPOD,
HBOS) and the PyGOD graph family (DOMINANT, CONAD, AnomalyDAE, GAAN). On SWE-Gym the tabular
detectors span ROC-AUC 0.32 to 0.63 and the graph detectors cluster near 0.49, all below the size
baseline; on tau-bench they sit near the floor. GUARDIAN, the agent-specific reconstruction
autoencoder, reaches the ECOD level (0.767) but no higher. Reading the typed graph with an
off-the-shelf detector does not, by itself, find failures; the task-aware dependency features do, and
a supervised network trained on them (G-Safeguard) tops SWE-Gym. That gap is the benchmark's point:
the structure has to be used, not merely be present.

### Baselines and Lineage

The graph-AD baselines are ports of published methods onto the dependency graph, in the ADBench /
BOND tradition of running a method on the benchmark's representation rather than gesturing at it.
PyGOD's DOMINANT (Ding et al., 2019, *Deep Anomaly Detection on Attributed Networks*, SDM) is an
unsupervised graph autoencoder that scores nodes by reconstruction error. GUARDIAN (Zhou et al.,
2025, arXiv:2505.19234), which safeguards multi-agent collaborations with a reconstruction-error
temporal graph autoencoder, is implemented here as a directed-GCN attribute-reconstruction
autoencoder over the per-run graph; the explicit adjacency-reconstruction term and the
information-bottleneck compression are simplified, as the code documents. G-Safeguard (Wang et al.,
2025, arXiv:2502.11127) is a supervised GNN; its original threat model is injected adversarial agents
on the utterance graph, and here it is a supervised graph-classification GNN over the dependency
graph, the benchmark's first learned-over-the-graph detector. It tops the SWE-Gym detection board at
0.828, a third-party method leading one domain, which is the benchmark working as intended.

### Gold: Injected Faults on Real Runs

The boards above borrow labels (human attribution, run outcomes). Gold is the benchmark's own data
contribution: plant a known fault in a real run and ask whether a method points to it. 188 clean
SWE-Gym runs, one injected fault each, 82 stale-state and 106 dropped-grounding (a run affords a
stale-state fault only when it has an earlier same-file read to redirect to); the label is the
injection site. The numbers below are a representative seed and are stable across five injection seeds
(`run.py` prints the mean and standard deviation). Read the board PER FAULT KIND, because the two fault
types behave completely differently and the aggregate hides it.

| Method | overall Top-1 | stale-state Top-1 | dropped-grounding Top-1 |
|---|---|---|---|
| random (seed-averaged) | 0.032 | -- | -- |
| position (leak check) | 0.000 | 0.000 | 0.000 |
| degree (leak check) | 0.016 | 0.000 | 0.028 |
| has-dep (control) | 0.000 | 0.000 | 0.000 |
| max-span (control) | 0.309 | 0.707 | 0.000 |
| `auditable` (dep-anomaly) | 0.309 | **0.707** | 0.000 |
| PyGOD (graph AD) | 0.000 | 0.000 | 0.000 |

The injector can only target a step that has a dependency, so the injected step always carries one.
That makes the full-pool table leak-prone: any detect-the-eligible baseline lifts for free. The
degree-matched control re-ranks every method within only the steps the injector could have picked for
that run's fault kind (mean 7.4 candidates per run, mirroring the injector's precondition exactly),
holding eligibility and degree constant. The
random floor rises accordingly, and ties are broken in expectation so a constant-score baseline lands
on the floor instead of winning on sort order.

| Method (degree-matched pool) | overall Top-1 | stale-state Top-1 | dropped-grounding Top-1 |
|---|---|---|---|
| random (matched floor) | 0.308 | 0.350 | 0.277 |
| position | 0.330 | 0.341 | 0.321 |
| degree | 0.225 | 0.394 | 0.095 |
| has-dep | 0.195 | 0.350 | 0.075 |
| max-span | 0.394 | **0.805** | 0.075 |
| `auditable` (dep-anomaly) | 0.391 | 0.799 | 0.075 |
| PyGOD (graph AD) | 0.311 | 0.359 | 0.274 |

How to read it. Stale-state faults are detectable: in the full pool a dependency-span detector
localizes them at 0.707 Top-1, far above the 0.032 random floor. The degree-matched control shows
that lift is the fault, not an artifact: `has-dep` falls to exactly the matched floor (0.350 stale, a
constant score over an all-eligible pool), `degree` sits below the floor overall (0.225 against 0.308)
with only a small stale-state residual (0.394 against 0.350) that a Monte-Carlo check reads as seed
noise, and the dependency-span signal clears the floor by far (0.805 stale against 0.350). The
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

- **Grounded faults, one strong half.** Stale-state (redirect a dependency to an earlier superseded
  event on the same file) is localized at 0.707 Top-1, and 0.671 +/- 0.022 across five injection seeds.
  Dropped-grounding (remove a required dependency) is realized but not yet localized by any baseline; it
  needs a better detector or a stronger substrate.
- **Leakage check, controlled for stale-state.** `position` stays at the random floor. In the full
  pool `degree` and a `has-dep` eligibility baseline both lift above it, but that is target selection:
  the injector only corrupts steps that have dependencies. Ranking within the degree-matched eligible
  pool removes the artifact. `has-dep` lands exactly on the matched floor (0.350 stale) and `degree`
  sits below it overall (0.225 against 0.308) with a small stale residual (0.394) that a Monte-Carlo
  check reads as seed noise, while the dependency-span detector clears the floor by far (0.805 stale
  against 0.350, and 0.795 +/- 0.020 across seeds). The stale-state lift survives the control, so it is
  the fault signature rather than the construction. dropped-grounding stays at the floor in both pools,
  which is the open detection problem, not a leak.
- **Distributional validity, reported honestly.** Stale-state preserves the valid dependency-edge
  count (mean 9.2, unchanged); dropped-grounding removes exactly one valid dependency edge (mean 7.9 to
  6.9), so treat edge count as a reported run-level shift for the dropped half, not as matched.
  Stale-state lengthens the run-level max dependency span by construction (mean 8.6 to 9.4, 53 of 188
  runs increased). These shifts are reported, not hidden, and localization still requires finding the
  step.
- **Labels are the injection site**, correct by construction and independent of any detector.
- **Characterized caveat.** SWE-Gym dependencies are inferred, not gold value-flow, so a redirected
  edge is a dependency-misattribution proxy for a true stale read. The nearer fix is a
  dropped-grounding detector; the airtight substrate is a named-value corpus where writes and reads
  are explicit, with a human-audited validation slice (Cohen's kappa).

### Cause Attribution and the LIVE Boards

Three more boards complete the POST and LIVE pillars; the paper carries the full tables.

**Cause attribution (POST, what kind).** Given a faulty run, is it stale-state or dropped-grounding?
The substrate is paired, the same run injected both ways, so the label is the fault, not the run, with
no eligibility leak (the run-level analogue of the degree-matched control). On 166 paired runs the two
faults leave opposite traces: a stale read lengthens the max dependency span (ROC-AUC 0.675, and
0.671 +/- 0.005 across seeds), dropped grounding removes an edge (edge-count 0.566), against a 0.498
floor. Each feature is keyed to one mechanism, so this measures whether the structure carries
cause-discriminative information, not a general attribution model.

**LIVE streaming early warning (can you tell early).** Can a method separate failing from resolved
runs from a growing prefix? On SWE-Gym the dependency-structure block clears ROC-AUC 0.74 at the 25%
prefix (time-to-detection 25%) while run size never does; an unsupervised ECOD on the prefix also
fires early (0.76), but a raw per-run span signal does not (0.36, length-confounded). The early signal
needs the full prefix features, supervised or unsupervised, not a single online scalar. On tau-bench
every method is weak and late, the same domain split as detection.

**LIVE online stale-state detection (catch it live).** The same Gold stale-state injection, detected
online at a fixed false-positive rate instead of localized post-hoc. It is hard: the causal span
z-score catches about 6% of stale reads at a realized 6% false-positive rate (0.054 +/- 0.012 across
seeds), far below the 0.71 within-run localization. A dep-count control sits exactly at the floor,
confirming the span, not generic structural change, is the signal. Spotting one stale read online
without false-alarming is an open problem.

### Over-Privilege Audit on Declared Harnesses (PRE)

Before a run, the only evidence is what the agent declares: its task or role, and the capabilities its
harness grants. The PRE board asks whether a method can flag the granted capabilities the task does not
need. It runs over 1187 configurations from six corpora (crewai, n8n, mcp, injecagent, sweagent, and a
synthetic set), each declared capability labeled needed or excess, scored by precision, recall, and F1
over the flagged-excess set.

Most agent-auditing tools ship a static over-privilege check, so the board's static baseline is built
to at least the accepted coverage, not one hand-rolled rule. Each rule maps to a standard
over-privilege category from OWASP LLM06:2025 Excessive Agency, the OWASP Agentic Security Initiative
(ASI), and the CWE privilege family:

| Standard category | Scanner rule | Reference |
|---|---|---|
| Excessive permissions / least privilege | `owasp_excess_permissions` | OWASP LLM06; CWE-272, CWE-250 |
| Excessive functionality | `owasp_excess_functionality` | OWASP LLM06 |
| Privilege compromise / escalation | `owasp_privilege_escalation` | CWE-269; OWASP ASI (Privilege Compromise) |
| Excessive autonomy (approximation) | `unrequested_high_impact` | OWASP LLM06 (autonomy driver) |
| Sensitive-access exposure surface | `sensitive_access` | OWASP LLM02 (risk surface) |

`owasp_asi_combined` is their union. Three standard concerns are named but stay out of static
single-config scope, and the board says so rather than overclaiming: a full excessive-autonomy check
needs an approval-gate field the schema does not carry, so `unrequested_high_impact` is an
approximation (a high-impact action the task never asks for); full ASI Tool Misuse needs declared
operation, scope, and allowlist controls the schema does not express, so `sensitive_access` is a
narrower LLM02 exposure heuristic; and a deprecated or duplicate extension needs deployment history.
The board scores each rule and the union, so coverage is visible rather than asserted:

| Method | Precision | Recall | F1 |
|---|---|---|---|
| flag-all (floor) | 0.430 | 1.000 | 0.601 |
| flag-none (floor) | 0.000 | 0.000 | 0.000 |
| risky-permission scan | 0.418 | 0.564 | 0.480 |
| `owasp_excess_permissions` | 0.504 | 0.506 | 0.505 |
| `owasp_excess_functionality` | 0.538 | 0.796 | 0.642 |
| `owasp_privilege_escalation` | 0.811 | 0.010 | 0.020 |
| `unrequested_high_impact` | 0.633 | 0.148 | 0.240 |
| `sensitive_access` | 0.763 | 0.016 | 0.030 |
| `owasp_asi_combined` | 0.511 | 0.910 | **0.654** |
| LLM judge, held out (Llama-3.3-70B) | 0.594 | 0.741 | **0.659** |
| oracle (declared minus minimal) | 1.000 | 1.000 | 1.000 |

How to read it. The combined OWASP/CWE scanner is the strongest rule-based method (0.654 F1, 0.910
recall), close behind the held-out LLM judge, because it flags unnecessary read and unknown
capabilities (excessive functionality applies at every permission level), not just risky permission
levels. The three narrow rules make few predictions each (privilege escalation 37, sensitive access
59, unrequested high-impact 676 over the full 1187-config board), so their low overall recall shows
they cover small slices of the aggregate excess set. The corpus labels one undifferentiated excess
set with no per-category annotation, so it cannot say how prevalent each standard category is, and
these rules' precision is measured over those small-to-moderate samples rather than a category-level
ground truth. Above every rule sits the held-out LLM judge (0.659), the PRE
analogue of the LLM-judge panel topping POST localization, because a capable model reading the stated
purpose is a strong over-privilege detector. Held out is the load-bearing phrase: the crewai, n8n, and
mcp labels were made by two other judges (GPT-5.5 and Claude), so scoring a third model that made none
of them keeps the baseline from grading its own homework. The scanners are keyword-based and therefore
language-brittle: a task spec in a language the keyword lists do not cover falls back to the read-only
floor and over-flags, a real property of static analysis that the board reports per source rather than
hides.

Read the board per source, because the four label processes differ and a pooled F1 hides it (the
per-rule per-source numbers print from `run.py`):

| Method | crewai | n8n | mcp | injecagent | sweagent | synthetic |
|---|---|---|---|---|---|---|
| risky-permission scan | 0.326 | 0.095 | 0.575 | 0.827 | 0.025 | 0.803 |
| `owasp_asi_combined` | 0.448 | 0.411 | 0.644 | 0.961 | 0.570 | 0.842 |
| LLM judge, held out | 0.518 | 0.357 | 0.662 | 0.990 | 0.467 | 0.972 |
| oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Label origin per column: crewai, n8n, and mcp carry cross-vendor LLM-judge labels (Cohen's kappa 0.666,
`data/pre/LABEL_QUALITY.md`); injecagent is a roster relabel; sweagent is declared-minus-used; the
synthetic set is injection. The held-out judge lands near the top on injecagent (0.990) and synthetic
(0.972), where the labels are constructed, and only moderate on the judge-labeled corpora (0.36 to
0.66). That gap is the reason to hold the judge out: a same-judge method would inherit perfect recall
and a mechanically inflated F1 by construction, so the moderate agreement here is real disagreement on
a subjective call, not a leak. Where excess is rare (n8n, mean excess ratio 0.084) every method
struggles, because a single over-flag sinks precision. The held-out judge parsed 1182 of the 1187
configs; the 5 it could not parse are scored as flagging nothing, which counts against its recall.

## The Full Task List

Each task is one auditor question at one information state, with the label that answers it, not a
list chosen for coverage. Shipped in v1:

- **Fault localization** (POST, *which step*): rank the steps of a failed run. Top-1 / Top-3 / MRR
  against Who&When human attribution.
- **Failure detection** (POST, *did it fail*): predict run failure. ROC-AUC against SWE-Gym and
  tau-bench resolved / unresolved outcomes.
- **Cause attribution** (POST, *what kind*): tell stale-state from dropped-grounding on a paired Gold
  injection (the same run injected both ways). ROC-AUC.
- **Gold fault localization** (POST, the data contribution): plant a known fault in a real run and
  localize it. Top-1 / Top-3 / MRR against injection-site labels.
- **Streaming early warning** (LIVE, *can you tell early*): flag a failing run from a growing prefix.
  Prefix-AUC and time to detection.
- **Online stale-state detection** (LIVE, *catch it live*): detect the Gold stale-state injection
  online. True-positive rate at a fixed false-positive budget.
- **Over-privilege audit** (PRE, *is the declared harness safe*): flag granted capabilities the task
  does not need, against an OWASP / CWE static-scanner set (the standard-coverage baseline) and a
  held-out LLM judge. Precision / recall / F1 over 1187 configs from six corpora.

Planned:

- **Missing-guardrail plan audit** (PRE, *is the plan safe*): flag removed or weakened guardrails in a
  declared plan.

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

v1 is a local working build of the POST and LIVE pillars plus the PRE over-privilege board: POST
localization, detection, cause attribution, and the Gold injection board; LIVE streaming early-warning
and online stale-state detection; and PRE over-privilege audit across six config corpora. The repository
goes public together with the benchmark paper, so the framing and the leaderboard arrive at once and the
angle is established in print before it is copied. That is a
release-timing choice, not a secrecy one: nothing here is withheld. GRADE is public on arXiv,
`auditable`'s kernel is public, and the labels are public.

The benchmark's own data contribution, the Gold board, now ships its first slice: faults injected
into real runs with injection-site labels (the standard move when real labels are scarce; BOND
injects anomalies into real graphs for the same reason), with the stale-state half leakage-controlled
by a degree-matched eligible-pool ranking. The airtight upgrade is a named-value corpus where a stale
read is unambiguous, plus a dropped-grounding detector and a human-audited validation slice, which
together grow the dataset past the public corpora it starts from.
