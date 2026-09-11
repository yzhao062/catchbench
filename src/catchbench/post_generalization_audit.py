r"""POST generalization audit: the endpoint size control, read on two further corpora.

Contribution (2) rests on two corpora. SWE-Gym's structural increment changes sign once the size
reference is allowed to bend, ``+0.142`` under linear models against ``-0.010`` under a fixed cubic
spline over the same four size columns, and tau-bench's does not. Part A of the LIVE declaration
extended that control to the prefixes of those same two corpora and found the SWE-Gym increment near
zero at every one of them. Every cell of that finding still comes from the same two corpora. This
module runs the same declared contrast on two more: OpenHands / SWE-rebench and ScienceWorld.

**The arms are ``detection``'s own objects, not prefix-shaped or corpus-shaped copies.** The number
the manuscript cites at the endpoint is what ``_MixedSplineAUC`` produces, and the whole point of a
generalization audit is that a new corpus is the same arm read on different rows. ``audit_arms``
therefore returns the four objects ``live_size_audit.live_size_audit_arms`` returns, built from
``detection``'s factories, so the spline configuration, the column split and the classifier reach
both new corpora from one place. **Only the four flat size and count columns receive the spline**;
the four normalized dependency columns pass through unchanged, which is what makes the last arm a
matched control rather than a different model. Splining all eight ``flatdep`` columns is a different
control and a different number, ``-0.016325`` against ``-0.005404`` on the committed endpoint
records, so the distinction does not round away.

**What is new here is the fold path, and that is the only thing that is new.**
``detection._cv_estimator_detail`` builds ``StratifiedKFold(n_splits=5, shuffle=True,
random_state=seed)`` inside its own loop, so it cannot be handed a grouped partition. OpenHands needs
one: 600 runs carry 574 distinct ``instance_id``, 26 of which are attempted twice, and a row split
would put one attempt at an issue in training while the other is scored. ``_cv_estimator_on_splits``
is ``_cv_estimator_detail``'s body with the partition handed in rather than constructed, and
everything else about the contract is kept: the same scorer object, the same per-(seed, fold)
ROC-AUC, the same five out-of-fold vectors, the same training-only pipeline fits, the same
``best_params_`` capture. Section D of the declaration requires that the two paths agree exactly on
ScienceWorld, where ordinary stratified folds and a saved-index path produce the same partition by
construction, and ``check_saved_split_parity`` is that check. It is not optional: it is the only
evidence that moving the partition out of the fold loop did not move the fitting.

**Neither corpus is registered anywhere.** ``detection._LOADERS`` is untouched, no board roster gains
a row, ``live.live_streaming_methods`` is unchanged, and the nine boards, the 72 entrants and the
138-record registry are what they were. The loaders below are local to this module for that reason,
and they call GRADE's shipped ``load_openhands`` and ``load_scienceworld`` rather than reimplementing
them, so the graphs and labels this audit scores are the ones GRADE's own eval would score.

Those two loaders drop the row identities, exactly as the SWE-Gym loader drops ``instance_id``.
``_openhands_identities`` and ``_scienceworld_identities`` recover them by replaying the same
iteration under the same filters and then verifying the replay against the scored labels and the
``n_steps`` column, row for row, which is the pattern ``detection._tau_run_identities`` already uses.
A misaligned identifier is worse than none, so both raise rather than return an unverified one.

The analysis is declared in ``research/catchbench-post-generalization-declaration-2026-09-11.md``,
frozen at 2026-09-11T01:05:49Z before any arm was fitted on either corpus. ``DECLARATION`` names it,
and the constants below are its frozen quantities written down in code so a later run cannot quietly
choose a different one.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Mapping, Sequence

from catchbench import _reuse  # noqa: F401  side effect: sets sys.path for grade + auditable

import numpy as np  # noqa: E402

from sklearn.metrics import get_scorer, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold  # noqa: E402

from agent_failure_detection import SEEDS  # noqa: E402  GRADE's split seeds, unchanged

from catchbench.detection import (  # noqa: E402
    _EstimatorAUC,
    _MixedSplineAUC,
    _cv_estimator_detail,
    _estimator_specification,
    _linear_estimator,
    _spline_estimator,
)


DECLARATION = "research/catchbench-post-generalization-declaration-2026-09-11.md"

# Section C, measured on 2026-09-11 before the declaration was written and frozen into it. A
# different count is a blocker rather than something to accommodate: the declaration says
# "Preflight must reproduce every one of those numbers exactly", and a moved population would mean
# the audit is not the audit that was declared.
#
# ``instance_id_multiplicity`` is keyed by string because a JSON round trip does that to an integer
# key anyway, and the record is what a reader checks this against.
DECLARED_POPULATION = {
    "openhands": {
        "n_runs": 600,
        "n_solved": 288,
        "n_failed": 312,
        "n_distinct_group": 574,
        "group_multiplicity": {"1": 548, "2": 26},
        "n_missing_group": 0,
    },
    "scienceworld": {
        "n_runs": 128,
        "n_solved": 64,
        "n_failed": 64,
    },
}

# Section C's POST eligibility rule, written down because the LIVE audit's is different and the two
# modules sit beside each other. POST keeps a run with at least two parsed steps; LIVE's four-step
# filter is not applied here.
MIN_STEPS = 2

# Section D. OpenHands is grouped by issue so that every attempt at an issue stays together in
# training and validation; ScienceWorld has no repeated unit to group and takes the row split the
# POST boards already use. The protocol is keyed by corpus rather than inferred from whether groups
# exist, so a corpus whose identities later become recoverable does not silently change protocol.
FOLD_PROTOCOL = {
    "openhands": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed), "
                 "grouped by instance_id",
    "scienceworld": "StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)",
}
GROUPED_CORPORA = ("openhands",)
N_SPLITS = 5

# Section D fixes the numerical tolerance before the first fit, so a comparison that fails cannot be
# rescued by widening it afterwards. The value and the two-level split are Part A's, unchanged,
# because the comparisons are the same kind of comparison:
#
#   PARITY_ATOL applies to a float score compared against a score the other code path produced on
#   the same rows in the same environment. The expected difference is exactly zero; the allowance
#   covers only the last bits of a float64. A difference above it means the new fold path changed
#   the fitting, which is a finding rather than a tolerance to loosen.
#
#   Structure carries NO tolerance. Fold index arrays, group memberships, column boundaries, feature
#   matrices and labels are compared with exact equality, because an approximate match there would
#   mean the rows or the splits are not the same rows or splits.
PARITY_ATOL = 1e-12

# Section B's four arms, keyed by the method id the committed POST records already use, with
# section B's role names beside them so the artifact says which arm is which control.
ARM_ROLES = {
    "size (flat)": "linear size reference",
    "auditable (size+deps)": "linear structural reference",
    "size (spline)": "fixed flexible size reference",
    "size-spline + linear-deps": "matched structural control",
}

# Section E's two contrasts, high arm minus low arm. ``L`` is the linear dependency increment, the
# quantity the POST board publishes; ``D`` is the matched-spline increment, the same question asked
# once the size reference is allowed to bend. ``S = L - D`` is the attenuation between them.
LINEAR_CONTRAST = ("auditable (size+deps)", "size (flat)")
MATCHED_CONTRAST = ("size-spline + linear-deps", "size (spline)")
ESTIMANDS = ("L", "D", "S")

# Section E nominates the primary quantity before any of the six exists. Recorded here so the
# nomination travels with the code that produces the number.
PRIMARY_ESTIMAND = ("openhands", "S")

# Section F freezes the RNG labels literally. Written down now, before any fit, because a label
# chosen after a bootstrap has been seen is not a frozen label.
RNG_LABELS = {(corpus, estimand): f"post_generalization.{corpus}.{estimand}"
              for corpus in DECLARED_POPULATION
              for estimand in ESTIMANDS}
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_BASE_SEED = 20260907

# Section F's resampling unit, which differs by corpus for the reason the fold protocol does.
RESAMPLING_UNIT = {"openhands": "issue group", "scienceworld": "row"}

# Section G's magnitude, and section G's own words about what it is.
SUBSTANTIAL_EFFECT = 0.03

# Section G's four branch names, in the order the declaration's table lists them.
BRANCHES = ("substantial-attenuation", "mixed-informative", "unresolved-or-modest",
            "stable-positive-both")


def audit_arms() -> list:
    """The four declared arms, in table order, as ``detection``'s own arm objects.

    Identical to ``live_size_audit.live_size_audit_arms``, and deliberately so: section B says "the
    four arms of Part A, unchanged, reusing the same classes rather than reimplementing them". The
    two modules build the list separately rather than importing one from the other, because the two
    declarations are independent documents and a later amendment to one must not silently move the
    other's arms. What they share is ``detection``'s factories, which is where the spline
    configuration and the classifier actually live.

    The first two reproduce the board's linear reading on each new corpus and give ``L``. The last
    two are the controlled pair and give ``D``. No other arm is fitted: all-column splines, boosted
    trees and nested knot selection are outside this declaration.
    """
    return [
        _EstimatorAUC("size (flat)", "flat", _linear_estimator),
        _EstimatorAUC("auditable (size+deps)", "flatdep", _linear_estimator),
        _EstimatorAUC("size (spline)", "flat", _spline_estimator),
        _MixedSplineAUC(),
    ]


# --- the two corpora, loaded through GRADE and re-identified beside it ---------------------------


def _openhands_identities(labels: np.ndarray, step_counts: np.ndarray) -> dict:
    """The ``instance_id`` of every scored OpenHands row, replayed and checked to align.

    ``agent_failure_detection.load_openhands`` iterates ``agent_graph_openhands.load_runs``, keeps
    the runs with at least two parsed steps, and returns graphs and labels, so the identifier is
    dropped rather than absent: ``load_runs`` reads ``instance_id`` out of the parquet and puts it in
    the record it yields. Replaying the same iteration under the same filter recovers it in the same
    order, which is the route ``detection._tau_run_identities`` takes for tau-bench.

    The alignment is verified, not assumed. The replayed labels must equal the scored labels and the
    replayed step counts must equal the ``n_steps`` column of the flat matrix, both exactly and row
    for row. Section D groups the folds by this identifier, so a silently misaligned one would mean
    the grouping is grouping the wrong rows, and the audit would look correct while being wrong.
    """
    import agent_graph_openhands as openhands

    records = []
    for record in openhands.load_runs(openhands.N_RUNS):
        steps = openhands.to_steps(record["traj"])
        if len(steps) < MIN_STEPS:
            continue
        records.append((record, len(steps)))

    replayed = np.array([0 if record["resolved"] else 1 for record, _ in records])
    if not np.array_equal(replayed, labels):
        raise AssertionError(
            f"openhands: the replayed labels no longer match the scored labels ({len(replayed)} "
            f"replayed against {len(labels)} scored); the issue identifiers cannot be trusted to "
            "line up with the feature matrices, and the grouped folds are built from them"
        )
    replayed_steps = np.array([float(count) for _, count in records])
    if not np.array_equal(replayed_steps, step_counts):
        raise AssertionError(
            "openhands: the replayed step counts do not match the n_steps column, so the replay is "
            "not in the order the feature matrices are in"
        )

    groups = [str(record["instance_id"]) for record, _ in records]
    per_group: dict[str, int] = {}
    for group in groups:
        per_group[group] = per_group.get(group, 0) + 1
    multiplicity: dict[str, int] = {}
    for count in per_group.values():
        multiplicity[str(count)] = multiplicity.get(str(count), 0) + 1
    return {
        "available": True,
        "source": "agent_graph_openhands.load_runs(N_RUNS), replayed under the one filter "
                  "agent_failure_detection.load_openhands applies: len(steps) >= 2",
        "alignment_verified": ["labels equal the scored labels row for row",
                               "replayed step counts equal the n_steps column row for row"],
        "group_key": groups,
        "group_key_field": "instance_id",
        "n_missing_group": sum(1 for group in groups if not group or group == "None"),
        "n_distinct_group": len(per_group),
        "group_multiplicity": dict(sorted(multiplicity.items())),
        "note": "the rows are not independent draws. 26 issues are attempted twice, so a row-level "
                "split would train on one attempt at an issue and score the other. Section D groups "
                "the folds by this key for that reason.",
    }


def _scienceworld_identities(labels: np.ndarray, step_counts: np.ndarray) -> dict:
    """The task identity and source file of every scored ScienceWorld row, replayed and checked.

    ``agent_graph_scienceworld.load_runs`` reads every sciworld JSON under the repository, balances
    solved against failed within each source file, and returns ``{label, steps, reward}``, so
    ``info.task_name``, ``info.var_num`` and the source file name are all dropped. The file name is
    dropped twice over: ``load_runs`` selects ``max(by_src.values(), ...)``, which discards the key
    of the winning entry, so even the record's own structure no longer says which file it came from.

    Section C makes the recorded task-family mixture the thing section I's limitation is stated
    against, so this replays the raw JSON and recovers all three. The selection is reproduced rather
    than read, including ``max``'s first-wins tie rule over the sorted file order, and is then
    verified against the scored labels and step counts row for row, which is the same standard the
    tau-bench and OpenHands replays are held to.

    The Hugging Face repository is named ``webshop_expert_trajectories``. GRADE documents at
    ``agent_graph_scienceworld.py:1-18`` that the name is wrong and that the sciworld files under it
    are ScienceWorld. The repository name is recorded here so the disclosure travels with the data.
    """
    import json

    import agent_graph_scienceworld as scienceworld
    from huggingface_hub import hf_hub_download

    by_source: dict[str, dict[int, list]] = {}
    for name in scienceworld._sciworld_files():
        if "webshop_gpt" in name.lower():
            continue  # genuine WebShop dumps at the repository root, skipped by the loader too
        path = hf_hub_download(scienceworld.REPO, name, repo_type="dataset")
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            data = data.get("data") or list(data.values())
        slot = by_source.setdefault(name, {0: [], 1: []})
        for record in data:
            if not isinstance(record, dict):
                continue
            info = record.get("info") or {}
            reward = info.get("reward", record.get("reward"))
            if reward is None:
                continue
            label = 0 if int(reward) >= 100 else 1
            steps = scienceworld.to_steps(record)
            if len(steps) < MIN_STEPS:
                continue
            slot[label].append({
                "label": label, "n_steps": len(steps), "reward": int(reward), "source": name,
                "task_name": info.get("task_name"), "var_num": info.get("var_num"),
                "record_id": record.get("id"), "llm_name": info.get("llm_name"),
                "agent_arch": info.get("agent_arch"),
            })

    # ``max`` returns the first maximal element, and ``by_source`` is in the sorted file order
    # ``_sciworld_files`` returns, so this reproduces the loader's choice including its tie rule.
    chosen = max(by_source, key=lambda name: min(len(by_source[name][0]), len(by_source[name][1])))
    slot = by_source[chosen]
    keep = min(len(slot[0]), len(slot[1]), scienceworld.N_PER_CLASS)
    replay = slot[0][:keep] + slot[1][:keep]

    replayed = np.array([row["label"] for row in replay])
    if not np.array_equal(replayed, labels):
        raise AssertionError(
            f"scienceworld: the replayed labels no longer match the scored labels ({len(replayed)} "
            f"replayed against {len(labels)} scored); the task identities cannot be trusted to line "
            "up with the feature matrices"
        )
    replayed_steps = np.array([float(row["n_steps"]) for row in replay])
    if not np.array_equal(replayed_steps, step_counts):
        raise AssertionError(
            "scienceworld: the replayed step counts do not match the n_steps column, so the replay "
            "is not in the order the feature matrices are in"
        )

    pairs = [(row["task_name"], row["var_num"]) for row in replay]
    per_name: dict[str, int] = {}
    for name, _ in pairs:
        per_name[str(name)] = per_name.get(str(name), 0) + 1
    return {
        "available": True,
        "source": "agent_graph_scienceworld raw JSON, replayed under the two filters "
                  "agent_graph_scienceworld.load_runs applies (info.reward present, len(steps) >= "
                  "2) and its within-source balance and single-source selection",
        "alignment_verified": ["labels equal the scored labels row for row",
                               "replayed step counts equal the n_steps column row for row"],
        "repository": scienceworld.REPO,
        "repository_name_is_wrong": "GRADE documents at agent_graph_scienceworld.py:1-18 that the "
                                    "repository is named webshop_expert_trajectories and that the "
                                    "sciworld files under it are ScienceWorld. The manuscript names "
                                    "the corpus ScienceWorld and discloses the repository name.",
        "source_file": chosen,
        "source_files_considered": sorted(by_source),
        "class_sizes_per_source": {name: {"solved": len(entry[0]), "failed": len(entry[1])}
                                   for name, entry in sorted(by_source.items())},
        "task_name": [str(row["task_name"]) for row in replay],
        "var_num": [row["var_num"] for row in replay],
        "record_id": [str(row["record_id"]) for row in replay],
        "n_distinct_task_var_pairs": len(set(pairs)),
        "n_distinct_task_names": len(per_name),
        "runs_per_task_name": dict(sorted(per_name.items())),
        "llm_name": sorted({str(row["llm_name"]) for row in replay}),
        "agent_arch": sorted({str(row["agent_arch"]) for row in replay}),
        "note": "section I states its limitation against this task-family mixture. A disagreement "
                "with the reported mixture changes the limitation rather than being accommodated.",
    }


def swegym_issue_overlap(openhands_groups: Sequence[str]) -> dict:
    """Section C's third measurement: which issues the scored SWE-Gym population shares with these.

    The overlap matters because contribution (2)'s existing evidence is SWE-Gym, and OpenHands runs
    the same scaffold on a different issue collection. It is not reachable through GRADE's loader:
    ``agent_graph_swegym.load_runs`` selects ``[messages, resolved, run_id]`` out of the parquet, so
    ``instance_id`` is never read, and the records it returns carry only ``resolved`` and ``steps``.
    ``detection._swegym_run_identities`` says exactly that and names the two routes past it. This is
    the second of them: an out-of-tree replay that adds ``instance_id`` to the requested columns and
    otherwise reproduces the loader's shard order, its ``len(steps) >= 2`` filter, its per-``run_id``
    class balance, its ``balanced >= N_PER_CLASS`` break and its single-``run_id`` selection.

    That route re-implements the selection rather than reading it, so its alignment has to be
    re-established against the shipped rows every time the loader moves. It is, here: the replayed
    ``resolved`` sequence and the replayed step counts must equal ``load_runs``'s own, exactly and
    row for row, and a mismatch raises rather than returning an overlap nothing supports.

    **No row is dropped for an overlap.** Section C says the audit refits within each corpus, so a
    shared issue does not train across the old and new populations. Any nonzero overlap is disclosed
    in the appendix, and section I forbids a disjoint-issue-set claim unless this measures zero.
    """
    import json

    import agent_graph_swegym as swegym
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    by_run: dict[object, dict[int, list]] = {}
    shards_read = []
    for shard in swegym._shards():
        path = hf_hub_download(swegym.REPO, shard, repo_type="dataset")
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        trajectory = swegym._traj_col(names)
        columns = [name for name in (trajectory, "resolved", "run_id") if name in names]
        if "instance_id" not in names:
            raise AssertionError(
                f"{swegym.REPO}/{shard} has no instance_id column, so the overlap cannot be "
                "measured through the raw shard"
            )
        shards_read.append(shard)
        for batch in parquet.iter_batches(batch_size=64, columns=columns + ["instance_id"]):
            rows = batch.to_pydict()
            for index in range(len(rows[trajectory])):
                raw = rows[trajectory][index]
                if isinstance(raw, str):
                    raw = json.loads(raw)
                steps = swegym.to_steps(raw)
                if len(steps) < MIN_STEPS:
                    continue
                resolved = int(rows["resolved"][index]) if "resolved" in rows else 0
                run_id = rows["run_id"][index] if "run_id" in rows else "all"
                slot = by_run.setdefault(run_id, {0: [], 1: []})
                slot[resolved ^ 1].append({"resolved": resolved, "n_steps": len(steps),
                                           "instance_id": str(rows["instance_id"][index])})
        balanced = sum(min(len(entry[0]), len(entry[1])) for entry in by_run.values())
        if balanced >= swegym.N_PER_CLASS:
            break

    chosen = max(by_run, key=lambda run_id: min(len(by_run[run_id][0]), len(by_run[run_id][1])))
    slot = by_run[chosen]
    keep = min(len(slot[0]), len(slot[1]))
    replay = slot[0][:keep] + slot[1][:keep]

    scored = swegym.load_runs()
    if [row["resolved"] for row in replay] != [row["resolved"] for row in scored]:
        raise AssertionError(
            f"the SWE-Gym replay returns {len(replay)} rows whose resolved sequence is not "
            f"load_runs's own over {len(scored)} rows; the recovered instance ids do not line up "
            "with the scored population and the overlap would be measured against the wrong rows"
        )
    if [row["n_steps"] for row in replay] != [len(row["steps"]) for row in scored]:
        raise AssertionError(
            "the SWE-Gym replay's step counts are not load_runs's own, so the replay is not in the "
            "order the scored rows are in"
        )

    scored_ids = [row["instance_id"] for row in replay]
    shared = sorted(set(scored_ids) & set(str(group) for group in openhands_groups))
    return {
        "measured_through": "an out-of-tree replay of agent_graph_swegym.load_runs with "
                            "instance_id added to the requested parquet columns",
        "why_not_the_loader": "agent_graph_swegym.load_runs selects only [messages, resolved, "
                              "run_id], so the SWE-bench instance_id column is never read and the "
                              "records it returns carry only 'resolved' and 'steps'",
        "alignment_verified": ["the replayed resolved sequence equals load_runs's own row for row",
                               "the replayed step counts equal load_runs's own row for row"],
        "repository": swegym.REPO,
        "shards_read": shards_read,
        "run_id_selected": str(chosen),
        "n_swegym_rows": len(scored_ids),
        "n_distinct_swegym_issues": len(set(scored_ids)),
        "n_distinct_openhands_issues": len(set(str(group) for group in openhands_groups)),
        "n_shared_issues": len(shared),
        "shared_issues": shared,
        "no_row_is_dropped": "section C: the audit refits within each corpus, so a shared issue does "
                             "not train across the old and new populations and no row is dropped "
                             "for it. Any nonzero overlap is disclosed in the appendix.",
        "claim_gate": "section I forbids a disjoint-issue-set claim unless this measures zero",
    }


class PostGeneralizationTask:
    """One of the two new corpora, in the shape ``detection``'s arms and checks expect.

    ``detection``'s arms read ``task.layers[layer]`` and ``task.y`` and call ``task.setup()``, and
    ``_MixedSplineAUC.columns`` additionally reads ``task.layers["flat"]`` to find the size width.
    This class provides exactly that surface, so the arms are REUSED rather than reimplemented and
    the column check runs against this corpus's own two matrices inside the class the endpoint
    number came from.

    It deliberately is NOT a ``PostDetection`` subclass and deliberately does not appear in
    ``detection._LOADERS``. Section C keeps both corpora off every roster, and a registered loader is
    reachable from the CLI and from the board, which is the thing being kept from happening.

    ``groups`` is the extra field ``PostDetection`` has no equivalent of. It is the grouping key
    section D partitions by, and it is ``None`` on a corpus with no repeated unit, so a caller that
    forgets which protocol a corpus takes cannot build a grouped split out of nothing.
    """

    task_id = "post_detection"
    pillar = "POST"
    granularity = "run"

    def __init__(self, corpus: str) -> None:
        if corpus not in DECLARED_POPULATION:
            raise ValueError(f"unsupported generalization corpus: {corpus!r}")
        self.corpus = corpus
        self.dataset = corpus
        self._loaded = False

    def setup(self) -> None:
        if self._loaded:
            return
        from agent_failure_detection import (
            _layer_matrices, load_openhands, load_scienceworld,
        )
        from grade.features import feature_vector

        loader = {"openhands": load_openhands, "scienceworld": load_scienceworld}[self.corpus]
        graphs, labels = loader()
        self.graphs = graphs
        self.y = np.array(labels)
        self.layers = _layer_matrices(graphs)
        names = feature_vector(graphs[0], layer="flat")[0]
        self.step_counts = self.layers["flat"][:, names.index("n_steps")]
        self.identities = (_openhands_identities if self.corpus == "openhands"
                           else _scienceworld_identities)(self.y, self.step_counts)
        self.groups = (np.array(self.identities["group_key"])
                       if self.corpus in GROUPED_CORPORA else None)
        self._loaded = True

    def corpus_line(self) -> str:
        self.setup()
        n_runs, n_failed = len(self.y), int(self.y.sum())
        return (f"{self.dataset}: {n_runs} runs ({n_failed} failed, {n_runs - n_failed} solved), "
                f"run-level outcome labels, {FOLD_PROTOCOL[self.corpus]}.")


# --- the saved-split fold path -------------------------------------------------------------------


def fold_splits(task: PostGeneralizationTask) -> dict[int, list[tuple[np.ndarray, np.ndarray]]]:
    """Section D's partition: the (train, test) index arrays every arm on this corpus is scored on.

    Built once and handed to every arm, which is what makes the four arms comparable at all. Both
    splitters read only ``len(X)``, ``y`` and (where given) ``groups``, so an identical partition
    would come back whichever feature matrix was passed; that is scikit-learn's behaviour rather
    than something this function arranges, which is exactly why ``check_fold_protocol`` verifies it
    on the real matrices instead of citing it.
    """
    task.setup()
    labels = np.asarray(task.y)
    placeholder = np.zeros(len(labels))
    splits: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for seed in SEEDS:
        if task.groups is None:
            splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            folds = splitter.split(placeholder, labels)
        else:
            splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            folds = splitter.split(placeholder, labels, task.groups)
        splits[int(seed)] = [(train.copy(), test.copy()) for train, test in folds]
    return splits


def fold_membership(splits: Mapping[int, Sequence[tuple[np.ndarray, np.ndarray]]],
                    n_rows: int) -> list[list[int]]:
    """Per seed, the fold each row was held out in: the pairing an interval needs, stored compactly.

    Twenty-five index arrays say the same thing, and reconstructing the pairs from one fold index per
    row survives a JSON round trip without a reader having to trust that the array order was kept.
    """
    membership = []
    for seed in sorted(splits):
        row = np.full(n_rows, -1, dtype=int)
        for index, (_, test) in enumerate(splits[seed]):
            row[test] = index
        if int(row.min()) < 0:
            raise AssertionError(f"seed {seed}: {int((row < 0).sum())} rows were never held out")
        membership.append([int(value) for value in row])
    return membership


def _cv_estimator_on_splits(factory, X, y, splits, inspect=None):
    """``detection._cv_estimator_detail`` with the partition handed in rather than constructed.

    That one difference is the whole reason this function exists. ``_cv_estimator_detail`` builds
    ``StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`` inside its own loop, so a
    grouped partition cannot reach it, and OpenHands needs one. Everything else is kept identical on
    purpose, line for line: the scorer is the same object ``get_scorer("roc_auc")`` returns rather
    than a hand-rolled AUC, the fit is still ``factory(seed).fit(X[train], y[train])`` so a
    pipeline's knots and moments remain training-fold quantities, the out-of-fold probability is
    still ``predict_proba(X[test])[:, 1]``, and a nested arm's ``best_params_`` is still captured.

    ``check_saved_split_parity`` is what turns "kept identical" from a claim into a measurement: on
    ScienceWorld the two paths receive the same partition by construction, so any difference between
    their outputs is a difference this function introduced.

    ``inspect``, when given, is called as ``inspect(seed, fold, train, fitted)`` after each fit and
    before it is scored, so a fold-locality verification reaches every fitted estimator without
    refitting one. It is handed no test rows and nothing it does can reach ``fold_auc`` or ``oof``.
    """
    seeds = sorted(splits)
    if seeds != [int(seed) for seed in SEEDS]:
        raise AssertionError(f"the saved splits carry seeds {seeds}, not GRADE's {list(SEEDS)}")
    scorer = get_scorer("roc_auc")
    fold_auc = np.zeros((len(seeds), N_SPLITS))
    oof = np.zeros((len(seeds), len(y)))
    selected = []
    for row, seed in enumerate(seeds):
        folds = splits[seed]
        if len(folds) != N_SPLITS:
            raise AssertionError(f"seed {seed} carries {len(folds)} folds, not {N_SPLITS}")
        for fold, (train, test) in enumerate(folds):
            fitted = factory(seed).fit(X[train], y[train])
            if inspect is not None:
                inspect(int(seed), int(fold), train, fitted)
            fold_auc[row, fold] = scorer(fitted, X[test], y[test])
            oof[row, test] = fitted.predict_proba(X[test])[:, 1]
            chosen = getattr(fitted, "best_params_", None)
            if chosen is not None:
                selected.append({"seed": int(seed), "fold": int(fold),
                                 **{str(k): v for k, v in chosen.items()}})
    return fold_auc, oof, selected


def score_arm(arm, task: PostGeneralizationTask, splits, inspect=None) -> dict:
    """Score one arm on one corpus, keeping everything a fold score alone throws away.

    Section E requires the five out-of-fold vectors, the fold scores and the fold membership to be
    preserved, because the declared estimands are ROC-AUCs of seed-averaged out-of-fold
    probabilities and section F's intervals are paired on those vectors. A mean fold score discards
    both the vectors and the pairing, which is why the mean of the 25 fold AUCs rides along here as
    a separate reproduction quantity under a name that says which it is.
    """
    task.setup()
    labels = np.asarray(task.y)
    matrix = task.layers[arm.layer]
    factory = arm.estimator_factory(task)
    fold_auc, oof, selected = _cv_estimator_on_splits(factory, matrix, labels, splits,
                                                      inspect=inspect)
    specification = _estimator_specification(factory)
    return {
        "role": ARM_ROLES.get(arm.method_id),
        "layer": arm.layer,
        "board_entrant": False,
        "estimator": specification["estimator"],
        "params": specification["params_at_seed_0"],
        "params_vary_with_seed": specification["varies_with_seed"],
        "n_columns": int(matrix.shape[1]),
        # The declared estimand's input: the ROC-AUC of the five-seed-averaged out-of-fold vector.
        "seed_averaged_oof_roc_auc": float(roc_auc_score(labels, oof.mean(axis=0))),
        "per_seed_oof_roc_auc": [float(roc_auc_score(labels, row)) for row in oof],
        # The separate reproduction quantity: the mean of the 25 fold AUCs, which is what a board
        # prints. It never receives an interval computed for the quantity above.
        "mean_fold_roc_auc": float(fold_auc.mean()),
        "seed_mean_fold_roc_auc": [float(value) for value in fold_auc.mean(axis=1)],
        "fold_roc_auc": [[float(value) for value in row] for row in fold_auc],
        "oof_proba": [[float(value) for value in row] for row in oof],
        "selected_hyperparameters": selected,
        "nonfinite_fold_scores": int(np.count_nonzero(~np.isfinite(fold_auc))),
        "nonfinite_oof_values": int(np.count_nonzero(~np.isfinite(oof))),
    }


def _effects(scored: Mapping[str, dict]) -> dict:
    """Section E's three estimands on one corpus, on both scales.

    ``L`` and ``D`` are reported on the seed-averaged out-of-fold quantity, which is the estimand,
    and on the mean of fold AUCs, which is what a board prints. Reporting one without the other is
    what lets a reader take the second for the first, which is the mistake Part A's A5 was written
    against and the same two scales are in play here.
    """
    effects = {}
    for name, (high, low) in (("L", LINEAR_CONTRAST), ("D", MATCHED_CONTRAST)):
        if high not in scored or low not in scored:
            continue
        effects[name] = {
            "estimand": name,
            "high": high,
            "low": low,
            "seed_averaged_oof_difference": (scored[high]["seed_averaged_oof_roc_auc"]
                                             - scored[low]["seed_averaged_oof_roc_auc"]),
            "mean_fold_difference": (scored[high]["mean_fold_roc_auc"]
                                     - scored[low]["mean_fold_roc_auc"]),
        }
    if "L" in effects and "D" in effects:
        effects["S"] = {
            "estimand": "S",
            "definition": "L - D",
            "seed_averaged_oof_difference": (effects["L"]["seed_averaged_oof_difference"]
                                             - effects["D"]["seed_averaged_oof_difference"]),
            "mean_fold_difference": (effects["L"]["mean_fold_difference"]
                                     - effects["D"]["mean_fold_difference"]),
        }
    return effects


def _matrix_hash(matrix: np.ndarray) -> str:
    """The sha256 of a feature matrix's bytes, with its shape and dtype, so a record pins its input.

    Section E requires source and feature hashes. Hashing ``tobytes()`` alone would collide two
    matrices that share a buffer and differ in shape, so both travel in the digest.
    """
    digest = sha256()
    digest.update(str((matrix.shape, matrix.dtype.str)).encode("utf-8"))
    digest.update(np.ascontiguousarray(matrix).tobytes())
    return digest.hexdigest()


def post_generalization_audit(task: PostGeneralizationTask, splits=None, arms: list | None = None,
                              inspect=None) -> dict:
    """Score the four declared arms on one corpus and return everything section E preserves.

    ``inspect``, when given, is called as ``inspect(arm, seed, fold, train, fitted)`` after every
    fit, so a verification reaches all 100 fitted estimators of a corpus without refitting one.

    Writes no file and prints nothing. Section H's result file and its appendix table are the
    driver's business, as they are for the LIVE audit.
    """
    task.setup()
    labels = np.asarray(task.y)
    arms = list(audit_arms() if arms is None else arms)
    splits = fold_splits(task) if splits is None else splits

    scored: dict[str, dict] = {}
    for arm in arms:
        watcher = None
        if inspect is not None:
            def watcher(seed, fold, train, fitted, arm=arm):
                inspect(arm, seed, fold, train, fitted)
        scored[arm.method_id] = score_arm(arm, task, splits, inspect=watcher)

    return {
        "corpus": task.dataset,
        "task_id": task.task_id,
        "declaration": DECLARATION,
        "n_runs": int(len(labels)),
        "n_failed": int(labels.sum()),
        "labels": [int(value) for value in labels],
        "seeds": [int(seed) for seed in SEEDS],
        "fold_protocol": FOLD_PROTOCOL[task.corpus],
        "fold_membership": fold_membership(splits, len(labels)),
        "run_identities": task.identities,
        "resampling_unit": RESAMPLING_UNIT[task.corpus],
        "arm_roles": dict(ARM_ROLES),
        "rng_labels": {estimand: RNG_LABELS[(task.corpus, estimand)] for estimand in ESTIMANDS},
        "feature_hashes": {layer: _matrix_hash(matrix)
                           for layer, matrix in sorted(task.layers.items())},
        "label_hash": _matrix_hash(labels),
        "arms": scored,
        "effects": _effects(scored),
    }


# --- section G's reading, fixed before any of the six quantities exists ---------------------------


def corpus_reading(estimates: Mapping[str, float],
                   intervals: Mapping[str, Sequence[float]]) -> dict:
    """What section G's two named conditions say about one corpus.

    ``estimates`` and ``intervals`` are keyed by ``L``, ``D`` and ``S``. Section G defines:

      - **substantial attenuation**: ``S >= 0.03`` with its interval excluding zero, alongside
        evidence of a positive ``L``.
      - **informative stable-positive boundary**: ``D >= 0.03`` with its interval excluding zero,
        AND the interval for ``S`` contained within ``[-0.03, +0.03]``.

    One phrase in the first condition is not given a test by the declaration: "alongside evidence of
    a positive ``L``". Section F gives ``L`` an interval of its own, and the same support-then-size
    order the rest of section G uses reads that as a positive point with an interval excluding zero.
    That is the reading fixed here, before any of the six quantities existed, and it is recorded as a
    choice this implementation fixed rather than one the declaration made.

    Section G's third bullet is a field rather than a third condition, because the declaration gives
    it no name of its own: an interval for ``S`` that contains both zero and 0.03 is unresolved, and
    is reported as unresolved rather than as evidence of stability.
    """
    def excludes_zero(name: str) -> bool:
        low, high = intervals[name]
        return bool(low > 0.0 or high < 0.0)

    s_low, s_high = intervals["S"]
    positive_linear = bool(estimates["L"] > 0.0 and excludes_zero("L"))
    attenuation = bool(estimates["S"] >= SUBSTANTIAL_EFFECT and excludes_zero("S")
                       and positive_linear)
    boundary = bool(estimates["D"] >= SUBSTANTIAL_EFFECT and excludes_zero("D")
                    and -SUBSTANTIAL_EFFECT <= s_low and s_high <= SUBSTANTIAL_EFFECT)
    return {
        "substantial_attenuation": attenuation,
        "informative_stable_positive_boundary": boundary,
        "evidence_of_a_positive_linear_increment": positive_linear,
        "s_interval_contains_zero_and_the_threshold": bool(
            s_low <= 0.0 <= s_high and s_low <= SUBSTANTIAL_EFFECT <= s_high),
        "unresolved": not attenuation and not boundary,
        "substantial_threshold": SUBSTANTIAL_EFFECT,
        "threshold_is": "a declared editorial magnitude about what would be a worthwhile result for "
                        "this paper, stated as such in section G; not a deployment threshold and "
                        "not a validated minimum useful effect",
    }


def classify_branch(readings: Mapping[str, Mapping[str, bool]]) -> dict:
    """Section G's branch, from both corpora's readings. One of four, recorded as one of four.

    Written before the batch, which is the only property that makes it a rule rather than a reading.
    ``readings`` is keyed by corpus and each value is a ``corpus_reading``.

    **Two rows of section G's table overlap as written**, and the overlap is resolved here rather
    than after the numbers appear. ``substantial-attenuation`` reads "one corpus shows substantial
    attenuation; the other gives an informative replication or boundary", and ``mixed-informative``
    reads "one corpus shows substantial attenuation; the other gives an informative stable-positive
    boundary". The boundary case satisfies both descriptions, so a literal first-match reading of the
    table order would leave ``mixed-informative`` unreachable. The reading fixed here sends the
    boundary case to the row that names it specifically and leaves the replication case to the first
    row, which makes the four rows exhaustive and mutually exclusive and is the only reading under
    which every declared branch can occur. "An informative replication" is read as the other corpus
    also showing substantial attenuation.

    ``stable-positive-both`` is section G's fourth row, "both corpora keep substantial positive
    increments with little size-control sensitivity", which is both corpora reaching the informative
    stable-positive boundary. Everything else is ``unresolved-or-modest``, which section G defines as
    "anything smaller or less precise than the two above".
    """
    if sorted(readings) != sorted(DECLARED_POPULATION):
        raise AssertionError(
            f"section G's branch requires both corpora; got {sorted(readings)}. No result at one "
            "corpus is promoted to stand for both."
        )
    attenuating = {name for name, reading in readings.items()
                   if reading["substantial_attenuation"]}
    bounding = {name for name, reading in readings.items()
                if reading["informative_stable_positive_boundary"]}
    if len(attenuating) == 2:
        branch = "substantial-attenuation"
    elif len(attenuating) == 1 and bounding - attenuating:
        branch = "mixed-informative"
    elif len(bounding) == 2:
        branch = "stable-positive-both"
    else:
        branch = "unresolved-or-modest"
    return {
        "branch": branch,
        "corpora_showing_substantial_attenuation": sorted(attenuating),
        "corpora_giving_an_informative_stable_positive_boundary": sorted(bounding),
        "per_corpus": {name: dict(reading) for name, reading in sorted(readings.items())},
        "branches": list(BRANCHES),
        "rule": "section G, with the overlap between its first two rows resolved toward the more "
                "specific row: both corpora attenuating is substantial-attenuation, one attenuating "
                "beside a stable-positive boundary is mixed-informative, both bounding is "
                "stable-positive-both, anything else is unresolved-or-modest",
        "no_promotion": "section G: no result at one corpus is promoted to stand for both",
    }


# --- execution validity: what has to hold before any score is read -------------------------------
#
# These establish that the populations and the partitions are the ones the declaration describes.
# They raise on failure rather than returning a verdict, because a corpus whose rows or folds moved
# is not a weaker result, it is a different analysis, and section J's stopping rule does not admit
# repairing one into the other. Each returns what it established, so the record carries the evidence
# and not only the fact that something passed.


def check_population(task: PostGeneralizationTask) -> dict:
    """Section C's counts, reproduced exactly or blocked.

    Section C says "Preflight must reproduce every one of those numbers exactly", and blocks if it
    cannot. Every number it fixes is compared, including the ones only OpenHands has: the distinct
    issue count, the multiplicity histogram and the missing-identifier count, because section D's
    grouping is built from that identifier and a moved histogram would mean it is grouping
    differently.
    """
    task.setup()
    labels = np.asarray(task.y)
    declared = DECLARED_POPULATION[task.corpus]
    observed = {
        "n_runs": int(len(labels)),
        "n_failed": int(labels.sum()),
        "n_solved": int(len(labels) - labels.sum()),
    }
    if task.corpus in GROUPED_CORPORA:
        observed["n_distinct_group"] = int(task.identities["n_distinct_group"])
        observed["group_multiplicity"] = dict(task.identities["group_multiplicity"])
        observed["n_missing_group"] = int(task.identities["n_missing_group"])
    disagreeing = {key: {"observed": observed[key], "declared": value}
                   for key, value in declared.items() if observed.get(key) != value}
    if disagreeing:
        raise AssertionError(
            f"{task.dataset}: the population is not the one section C declares: {disagreeing}. "
            "Section C makes this a blocker rather than something to accommodate."
        )
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise AssertionError(f"{task.dataset}: labels are not the two-class failure labels")
    return {
        "corpus": task.corpus,
        "declared": dict(declared),
        "observed": observed,
        "eligibility": f"POST only: at least {MIN_STEPS} parsed steps. The LIVE four-step filter is "
                       "not applied. Label 1 means failure.",
        "label_1_means": "failure",
        "identities": task.identities,
    }


def check_column_boundaries(task: PostGeneralizationTask) -> dict:
    """The matched arm's size / dependency split, through the arm's own check.

    ``_MixedSplineAUC.columns`` reads the size width off this corpus's ``flat`` matrix and refuses to
    proceed unless the flat block is still the leading columns of ``flatdep``. Calling it here is the
    check: a changed feature order raises inside the class the endpoint number came from rather than
    producing a spline over the wrong block. The widths are returned so the record shows the boundary
    was 4 and 8 rather than merely that nothing raised.
    """
    task.setup()
    mixed = _MixedSplineAUC()
    size, deps = mixed.columns(task)
    flat = task.layers["flat"]
    flatdep = task.layers[mixed.layer]
    if size != list(range(flat.shape[1])) or deps != list(range(flat.shape[1], flatdep.shape[1])):
        raise AssertionError(f"{task.dataset}: the matched arm's column split is not contiguous")
    if not np.array_equal(flatdep[:, size], flat):
        raise AssertionError(f"{task.dataset}: the splined block is not the flat matrix")
    return {
        "checked_by": "catchbench.detection._MixedSplineAUC.columns",
        "establishes": "the splined block is exactly this corpus's flat matrix and the passthrough "
                       "block is the remaining flatdep columns, so the dependency columns enter "
                       "linearly and only the four size and count columns are splined",
        "splined_columns": size,
        "passthrough_columns": deps,
        "flat_shape": [int(value) for value in flat.shape],
        "flatdep_shape": [int(value) for value in flatdep.shape],
    }


def check_fold_protocol(task: PostGeneralizationTask, splits) -> dict:
    """Section D's assertions about the partition, all of them, before any arm reads it.

    Four separate facts, and none of them follows from another.

    Coverage: five seeds of five folds, and every row held out exactly once per seed. A partition
    that dropped a row would leave a zero in that row's out-of-fold vector, which scores as a
    confident prediction rather than as a gap.

    Both classes in every fold: section D says preflight asserts this and blocks if one does not
    hold. A single-class validation fold has no ROC-AUC, and a grouped splitter can produce one,
    because it is balancing class proportions across whole groups rather than across rows.

    Groups intact: on a grouped corpus no issue may appear in a fold's training rows and its test
    rows at once. That is the property the grouping exists for, and it is checked rather than
    assumed from the splitter's name.

    One partition for all four arms: both splitters read only ``len(X)``, ``y`` and ``groups``, so
    an identical partition comes back whichever feature matrix is passed. That is scikit-learn's
    behaviour rather than something this code arranges, so it is verified here on the real matrices,
    layer by layer, by exact array comparison against the saved reference.
    """
    task.setup()
    labels = np.asarray(task.y)
    seeds = sorted(splits)
    if seeds != [int(seed) for seed in SEEDS]:
        raise AssertionError(f"{task.dataset}: the splits carry seeds {seeds}, not {list(SEEDS)}")

    per_seed = []
    for seed in seeds:
        folds = splits[seed]
        if len(folds) != N_SPLITS:
            raise AssertionError(f"{task.dataset} seed {seed}: {len(folds)} folds, not {N_SPLITS}")
        held_out = np.zeros(len(labels), dtype=int)
        rows = []
        for index, (train, test) in enumerate(folds):
            held_out[test] += 1
            failed = int(labels[test].sum())
            solved = int(len(test) - failed)
            if failed == 0 or solved == 0:
                raise AssertionError(
                    f"{task.dataset} seed {seed} fold {index}: {len(test)} held-out rows carry "
                    f"{failed} failed and {solved} solved, so one class is missing and the fold has "
                    "no ROC-AUC. Section D makes this a blocker."
                )
            if task.groups is not None:
                shared = set(task.groups[train]) & set(task.groups[test])
                if shared:
                    raise AssertionError(
                        f"{task.dataset} seed {seed} fold {index}: {len(shared)} group(s) appear in "
                        "both the training and the held-out rows, so an attempt at an issue is "
                        "scored by a model trained on another attempt at the same issue"
                    )
            rows.append({"n_test": int(len(test)), "n_train": int(len(train)),
                         "n_failed": failed, "n_solved": solved,
                         "n_test_groups": (None if task.groups is None
                                           else len(set(task.groups[test])))})
        if not np.array_equal(held_out, np.ones(len(labels), dtype=int)):
            raise AssertionError(
                f"{task.dataset} seed {seed}: {int((held_out != 1).sum())} rows are not held out "
                "exactly once, so the out-of-fold vector is not an out-of-fold vector"
            )
        per_seed.append({"seed": int(seed), "folds": rows})

    compared = 0
    for layer in sorted({arm.layer for arm in audit_arms()}):
        matrix = task.layers[layer]
        if matrix.shape[0] != len(labels):
            raise AssertionError(f"{task.dataset}/{layer}: row count is not the corpus size")
        for seed in seeds:
            if task.groups is None:
                splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
                folds = splitter.split(matrix, labels)
            else:
                splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
                folds = splitter.split(matrix, labels, task.groups)
            for index, (train, test) in enumerate(folds):
                want_train, want_test = splits[seed][index]
                if not (np.array_equal(train, want_train) and np.array_equal(test, want_test)):
                    raise AssertionError(
                        f"{task.dataset}/{layer} seed {seed} fold {index}: the split differs from "
                        "the saved reference split, so the four arms are not scored on the same "
                        "folds"
                    )
                compared += 1

    minimum = min(fold["n_failed"] for seed in per_seed for fold in seed["folds"])
    minimum = min(minimum, min(fold["n_solved"] for seed in per_seed for fold in seed["folds"]))
    return {
        "protocol": FOLD_PROTOCOL[task.corpus],
        "grouped_by": "instance_id" if task.groups is not None else None,
        "establishes": "five seeds of five folds; every row held out exactly once per seed; both "
                       "classes present in all 25 folds; no group split across a fold boundary; and "
                       "all four arms scored on one identical partition on every feature layer",
        "n_seeds": len(seeds),
        "n_splits": N_SPLITS,
        "splits_compared": compared,
        "layers_compared": sorted({arm.layer for arm in audit_arms()}),
        "minimum_class_count_in_any_fold": int(minimum),
        "per_seed": per_seed,
    }


def check_saved_split_parity(task: PostGeneralizationTask, splits, arms: list | None = None,
                             tolerance: float = PARITY_ATOL) -> dict:
    """Section D's parity check: the saved-index path reproduces ``_cv_estimator_detail`` exactly.

    Only defensible where the two partitions agree by construction, which is the ungrouped corpus:
    ``fold_splits`` there builds ``StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`` and
    hands it to ``_cv_estimator_on_splits``, and ``_cv_estimator_detail`` builds the same splitter
    inside its own loop. Any difference between the two outputs is therefore a difference the new
    fold path introduced, which is the one thing this check is for. Running it on a grouped corpus
    would compare two different partitions and measure nothing.

    Every arm is checked, not a representative one. The two linear arms and the two spline-bearing
    arms exercise different pipeline shapes, and a ``ColumnTransformer`` in the matched arm is the
    place where a column-order difference between two paths would surface.

    The partition itself is compared first, with exact equality and no tolerance, because comparing
    scores from two different partitions would be comparing two different analyses. Only the scores
    are compared against ``tolerance``, which section D fixed before the first fit.

    Returns the comparison rather than the scores. The four ScienceWorld arm scores are three of the
    six quantities section E defines, and section E nominates the primary before any of them exists;
    a preflight that printed them would make three exist early for no gain, since the parity question
    is entirely answered by whether two paths agree. What travels out is how far apart they were.
    """
    task.setup()
    if task.groups is not None:
        raise AssertionError(
            f"{task.dataset} is grouped, so _cv_estimator_detail's stratified folds are not this "
            "corpus's partition and the two paths would be compared on different splits"
        )
    labels = np.asarray(task.y)
    arms = list(audit_arms() if arms is None else arms)

    reference = {}
    for seed in sorted(splits):
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        reference[seed] = [(train.copy(), test.copy())
                           for train, test in splitter.split(np.zeros(len(labels)), labels)]
        for index, (train, test) in enumerate(reference[seed]):
            saved_train, saved_test = splits[seed][index]
            if not (np.array_equal(train, saved_train) and np.array_equal(test, saved_test)):
                raise AssertionError(
                    f"{task.dataset} seed {seed} fold {index}: the saved split is not the split "
                    "_cv_estimator_detail builds, so the two paths cannot be compared"
                )

    per_arm = {}
    failures = []
    for arm in arms:
        matrix = task.layers[arm.layer]
        factory = arm.estimator_factory(task)
        ours_fold, ours_oof, ours_selected = _cv_estimator_on_splits(factory, matrix, labels, splits)
        theirs_fold, theirs_oof, theirs_selected = _cv_estimator_detail(factory, matrix, labels)
        fold_gap = float(np.abs(ours_fold - theirs_fold).max())
        oof_gap = float(np.abs(ours_oof - theirs_oof).max())
        agrees = bool(fold_gap <= tolerance and oof_gap <= tolerance
                      and ours_selected == theirs_selected)
        per_arm[arm.method_id] = {
            "layer": arm.layer,
            "fits_each_path": int(ours_fold.size),
            "max_absolute_fold_auc_difference": fold_gap,
            "max_absolute_oof_difference": oof_gap,
            "bit_identical_fold_auc": bool(np.array_equal(ours_fold, theirs_fold)),
            "bit_identical_oof": bool(np.array_equal(ours_oof, theirs_oof)),
            "selected_hyperparameters_equal": ours_selected == theirs_selected,
            "within_tolerance": agrees,
        }
        if not agrees:
            failures.append(
                f"{task.dataset} {arm.method_id}: the saved-index path differs from "
                f"_cv_estimator_detail by {fold_gap:.3e} on fold AUC and {oof_gap:.3e} on the "
                "out-of-fold probabilities"
            )
    return {
        "corpus": task.corpus,
        "compared_against": "catchbench.detection._cv_estimator_detail",
        "tolerance": tolerance,
        "arms_compared": sorted(per_arm),
        "max_absolute_difference": max(
            max(row["max_absolute_fold_auc_difference"], row["max_absolute_oof_difference"])
            for row in per_arm.values()),
        "arms_disagreeing": len(failures),
        "establishes": "on the corpus where an ordinary stratified partition and the saved-index "
                       "partition agree by construction, the two fitting paths produce the same "
                       "per-(seed, fold) ROC-AUC and the same out-of-fold probabilities, so moving "
                       "the partition out of the fold loop did not change the fitting",
        "scores_withheld": "the four arm scores are three of section E's six quantities; the parity "
                           "question is answered by how far apart the two paths are, so the "
                           "differences travel and the scores do not",
        "per_arm": per_arm,
        "failures": failures,
    }
