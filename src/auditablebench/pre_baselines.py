"""Rule-based PRE scanner baselines.

The naive risky-permission flag and the privilege-diff oracle live here; the comprehensive
OWASP / CWE static scanner set (the standard-coverage baseline) lives in ``pre_static_scanner`` and
is spliced into ``pre_baseline_methods`` below.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Mapping

from .pre import PreOverPrivilege, pre_score
from .pre_static_scanner import owasp_scanner_methods


_RISKY_PERMISSION_LEVELS = {"write", "execute", "network", "admin"}
_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "pre",
    "llm_judge_method",
)


class FlagRiskyPermsMethod:
    method_id = "flag_risky_perms"
    supports = {"pre_over_privilege"}

    def evaluate(self, task: PreOverPrivilege) -> Mapping[str, float]:
        view = task.method_view()
        flagged = {
            o["instance_id"]: {
                c["name"]
                for c in o["declared_capabilities"]
                if c["permission_level"] in _RISKY_PERMISSION_LEVELS
            }
            for o in view
        }
        return pre_score(flagged, task.instances)


class PrivilegeDiffOracleMethod:
    method_id = "oracle_privilege_diff"
    supports = {"pre_over_privilege"}

    def evaluate(self, task: PreOverPrivilege) -> Mapping[str, float]:
        task.setup()
        flagged = {
            o.instance_id: {c["name"] for c in o.declared_capabilities} - set(o.minimal_reference)
            for o in task.instances
        }
        return pre_score(flagged, task.instances)


class LlmJudgeNeededMethod:
    """A held-out LLM judge as a method: reads ONE model's needed-cache.

    ``_llm_judge_methods`` builds one instance per cache file, so several judges appear as
    separate board rows instead of a silent union (which would blend distinct models into one
    ensemble and break the per-model attribution). The cache holds each judge's NEEDED capability
    names; the excess prediction is declared - needed. The method reads only the held-out judge
    cache, never the label-making votes (gpt_judge_votes / claude_judge_votes), so a held-out
    model that made none of the labels stays non-circular.
    """

    supports = {"pre_over_privilege"}

    def __init__(self, cache_path: str) -> None:
        self.cache_path = cache_path
        model = os.path.splitext(os.path.basename(cache_path))[0]
        self.method_id = f"llm_judge_needed({model})"

    def evaluate(self, task: PreOverPrivilege) -> Mapping[str, float]:
        view = task.method_view()
        declared_by_id = {
            o["instance_id"]: {c["name"] for c in o["declared_capabilities"]}
            for o in view
        }
        with open(self.cache_path, encoding="utf-8") as f:
            rows = json.load(f)
        flagged = {}
        for instance_id, names in rows.items():
            if instance_id not in declared_by_id or not isinstance(names, list):
                continue
            needed = {name for name in names if name in declared_by_id[instance_id]}
            flagged[instance_id] = declared_by_id[instance_id] - needed
        # A configuration missing from the cache is one whose reply did not parse, so this judge
        # never issued a judgment on it. It abstains instead of being recorded as flagging nothing,
        # and pre_score reports what that abstention costs in coverage.
        return pre_score(flagged, task.instances, evaluable=set(flagged))


def _llm_judge_methods() -> list:
    """One LlmJudgeNeededMethod per committed judge cache in _CACHE_DIR (separate rows, no union)."""
    return [LlmJudgeNeededMethod(p) for p in sorted(glob.glob(os.path.join(_CACHE_DIR, "*.json")))]


def pre_baseline_methods() -> list:
    # flag-risky (naive floor) -> the OWASP/CWE scanner set (standard-coverage) -> oracle (ceiling)
    # -> held-out LLM judge(s).
    return (
        [FlagRiskyPermsMethod()]
        + owasp_scanner_methods()
        + [PrivilegeDiffOracleMethod()]
        + _llm_judge_methods()
    )
