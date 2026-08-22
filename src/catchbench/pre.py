"""PRE pillar: over-privileged harness configuration audit."""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Mapping

from ._paths import data_dir


_PERM_LEVELS = {"read", "write", "execute", "network", "admin", "unknown"}
_SPEC_OVERRIDE_KEYS = {
    "permission_levels",
    "functionality_capabilities",
    "privilege_bases",
    "high_impact_bases",
    "sensitive_bases",
}


@dataclass
class PreInstance:
    instance_id: str
    source: str
    provenance: dict
    spec_tokens: list[str]
    spec_token_overrides: dict[str, list[str]]
    declared_capabilities: list[dict]
    minimal_reference: list[str]
    labels: dict


def validate_pre_instance(o: PreInstance) -> None:
    assert isinstance(o.instance_id, str) and o.instance_id, "instance_id"
    assert o.source, "source"
    for k in ("repo", "commit", "path", "license"):
        assert k in o.provenance, f"provenance.{k}"
    assert o.spec_tokens == sorted(set(o.spec_tokens)), "spec_tokens must be sorted and unique"
    assert all(isinstance(token, str) and token for token in o.spec_tokens), "invalid spec token"
    assert set(o.spec_token_overrides) <= _SPEC_OVERRIDE_KEYS, "invalid spec_token_overrides key"
    for key, values in o.spec_token_overrides.items():
        assert values == sorted(set(values)), f"spec_token_overrides.{key} must be sorted and unique"
        assert all(isinstance(value, str) and value for value in values), f"invalid override {key}"
    names = {c["name"] for c in o.declared_capabilities}
    assert set(o.spec_token_overrides.get("functionality_capabilities", [])) <= names
    for c in o.declared_capabilities:
        assert c["permission_level"] in _PERM_LEVELS, f"perm {c['permission_level']}"
    assert set(o.minimal_reference) <= names, "minimal_reference not subset of declared"
    assert set(o.labels["excess_set"]) <= names, "excess_set not subset of declared"
    assert o.labels["label_source"] in {
        "declared_minus_used", "roster_relabel", "llm_judge", "synthetic_inject"
    }, "label_source"


def pre_score(flagged: dict, instances: list[PreInstance],
              evaluable: set[str] | None = None) -> dict[str, float]:
    """Micro-averaged precision, recall, and F1 over declared capabilities, with coverage.

    ``evaluable`` names the configurations the method actually judged. Anything outside it is an
    abstention: it leaves the denominator instead of counting as "flagged nothing". Scoring an
    unanswered configuration as a silent negative measures the harness that failed to read the
    answer rather than the method. That distinction is not hypothetical here. The held-out judge
    answered all 1187 configurations, but five of its replies named a capability whose spelling did
    not match the declared roster, and a strict all-or-nothing parser discarded those five whole
    judgments. One of them, a 622-capability MCP server, carries 337 excess labels, which is 11.7%
    of the corpus positive class, so the discard alone moved the pooled comparison.

    ``coverage`` is returned on every row, including the rows that answer everything, so a method
    scored on a smaller denominator can never be compared silently against one scored on the full
    corpus. Head-to-head claims should be made on a common evaluable set, not on these cells.
    """
    scored = [i for i in instances if evaluable is None or i.instance_id in evaluable]
    tp = fp = fn = 0
    for inst in scored:
        gt = set(inst.labels["excess_set"])
        pred = set(flagged.get(inst.instance_id, set()))
        tp += len(pred & gt)
        fp += len(pred - gt)
        fn += len(gt - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    cov = len(scored) / len(instances) if instances else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
            "coverage": round(cov, 3)}


def _fixture_instances() -> list[PreInstance]:
    return [
        PreInstance(
            instance_id="fixture-0001",
            source="fixture",
            provenance={"repo": "in-repo", "commit": "n/a", "path": "fixture", "license": "MIT"},
            spec_tokens=["read"],
            spec_token_overrides={},
            declared_capabilities=[
                {"name": "read_file", "type": "fs", "permission_level": "read"},
                {"name": "write_file", "type": "fs", "permission_level": "write"},
                {"name": "http_get", "type": "net", "permission_level": "network"},
            ],
            minimal_reference=["read_file"],
            labels={"excess_set": ["write_file", "http_get"], "label_source": "synthetic_inject"},
        )
    ]


def pre_instance_from_dict(d: dict) -> PreInstance:
    return PreInstance(
        instance_id=d["instance_id"],
        source=d["source"],
        provenance=d["provenance"],
        spec_tokens=d["spec_tokens"],
        spec_token_overrides=d.get("spec_token_overrides", {}),
        declared_capabilities=d["declared_capabilities"],
        minimal_reference=d["minimal_reference"],
        labels=d["labels"],
    )


_DATA_DIR = str(data_dir("pre"))


def _load_data_dir() -> list[PreInstance]:
    """Load every committed data/pre/*.json (each a list of PreInstance dicts).

    Keeps the board deterministic and zero-API on replay: harvesters write their
    normalized instances here offline, the board only reads them.
    """
    out: list[PreInstance] = []
    for p in sorted(glob.glob(os.path.join(_DATA_DIR, "*.json"))):
        with open(p, encoding="utf-8") as f:
            rows = json.load(f)
        # Every file matched here is loaded as a corpus, so a note or a manifest dropped into this
        # directory is read as one. Saying which file and what it is beats the TypeError that a
        # dict produces four frames down.
        if not isinstance(rows, list):
            raise ValueError(
                "%s holds a %s, and every data/pre/*.json is loaded as a list of PRE records; "
                "keep non-corpus files outside this directory"
                % (os.path.basename(p), type(rows).__name__)
            )
        for r in rows:
            out.append(pre_instance_from_dict(r))
    return out


class PreOverPrivilege:
    """PRE / over-privilege Task over harness capability declarations."""

    task_id = "pre_over_privilege"
    pillar = "PRE"
    granularity = "config"
    dataset = "multi"

    def __init__(self, instances: list[PreInstance] | None = None) -> None:
        self._seed = instances
        self._loaded = False

    def setup(self) -> None:
        if self._loaded:
            return
        if self._seed is not None:
            self.instances = self._seed
        else:
            loaded = _load_data_dir()
            self.instances = loaded if loaded else _fixture_instances()
        for o in self.instances:
            validate_pre_instance(o)
        self._loaded = True

    def corpus_line(self) -> str:
        self.setup()
        bys = {}
        for o in self.instances:
            g = _pre_logical_source(o.source)
            bys[g] = bys.get(g, 0) + 1
        return f"PRE over_privilege: {len(self.instances)} configs across {len(bys)} corpora {dict(bys)}"

    def method_view(self) -> list[dict]:
        self.setup()
        return [
            {
                "instance_id": o.instance_id,
                "source": o.source,
                "spec_tokens": o.spec_tokens,
                "spec_token_overrides": o.spec_token_overrides,
                "declared_capabilities": o.declared_capabilities,
            }
            for o in self.instances
        ]


class FlagAllMethod:
    method_id = "flag_all"
    supports = {"pre_over_privilege"}

    def evaluate(self, task: PreOverPrivilege) -> Mapping[str, float]:
        task.setup()
        flagged = {o.instance_id: {c["name"] for c in o.declared_capabilities} for o in task.instances}
        return pre_score(flagged, task.instances)


class FlagNoneMethod:
    method_id = "flag_none"
    supports = {"pre_over_privilege"}

    def evaluate(self, task: PreOverPrivilege) -> Mapping[str, float]:
        task.setup()
        return pre_score({}, task.instances)


def pre_methods() -> list:
    from .pre_baselines import pre_baseline_methods

    return [FlagAllMethod(), FlagNoneMethod()] + pre_baseline_methods()


# Logical source -> the label process that produced its excess_set, so the breakdown can flag that
# a pooled F1 mixes labels of different provenance (llm_judge is judge opinion; the others are
# roster / usage / injection derived). The four synth_* buckets share one injection process.
_PRE_LABEL_SOURCE = {
    "crewai": "llm_judge", "n8n": "llm_judge", "mcp": "llm_judge",
    "injecagent": "roster_relabel", "sweagent": "declared_minus_used", "synthetic": "synthetic_inject",
}
_PRE_SOURCE_ORDER = ["crewai", "n8n", "mcp", "injecagent", "sweagent", "synthetic"]


def _pre_logical_source(source: str) -> str:
    return "synthetic" if source.startswith("synth") else source


def pre_source_breakdown(task: "PreOverPrivilege", methods: list) -> str:
    """Per-source F1 for each PRE method, the analogue of POST's per-corpus tables.

    The pooled board scores one P/R/F1 over configs whose labels come from four different
    processes (llm_judge, roster_relabel, declared_minus_used, synthetic_inject). That pooled
    number is dominated by the largest sources and hides that the labels differ in reliability, so
    this breakdown reports F1 per source. Each method is re-scored on a seeded sub-task per source
    through the same evaluate() path, so no method interface changes.
    """
    task.setup()
    groups: dict[str, list[PreInstance]] = {}
    for o in task.instances:
        groups.setdefault(_pre_logical_source(o.source), []).append(o)
    cols = [g for g in _PRE_SOURCE_ORDER if g in groups] + [
        g for g in sorted(groups) if g not in _PRE_SOURCE_ORDER
    ]
    subtasks = {g: PreOverPrivilege(instances=insts) for g, insts in groups.items()}

    scored = [m for m in methods if "pre_over_privilege" in getattr(m, "supports", set())]
    mw = max([len("method")] + [len(m.method_id) for m in scored])  # fit the widest method id
    header = f"  {'method':{mw}s}" + "".join(f"{g:>13s}" for g in cols) + f"{'overall':>13s}"
    lines = ["\n[PRE] pre_over_privilege :: F1 by source", header]
    abstentions: list[str] = []
    for m in scored:
        per_source = {g: m.evaluate(subtasks[g]) for g in cols}
        cells = "".join(f"{per_source[g]['f1']:>13.3f}" for g in cols)
        overall = m.evaluate(task)
        lines.append(f"  {m.method_id:{mw}s}{cells}{overall['f1']:>13.3f}")
        # A cell scored on fewer configs than the column holds is not comparable to one that is not,
        # so name it here rather than letting the F1 stand alone.
        thin = [f"{g} {round(per_source[g]['coverage'] * len(groups[g]))}/{len(groups[g])}"
                for g in cols if per_source[g]["coverage"] < 1.0]
        if thin:
            abstentions.append(f"{m.method_id}: {', '.join(thin)}")

    legend = ", ".join(f"{g} n={len(groups[g])} ({_PRE_LABEL_SOURCE.get(g, '?')})" for g in cols)
    lines.append(f"  label source per column: {legend}")
    lines.append("  pooled F1 mixes these label sources; read per source, not just overall.")
    for note in abstentions:
        lines.append(f"  abstained (scored on fewer configs, not comparable cell to cell): {note}")
    return "\n".join(lines)
