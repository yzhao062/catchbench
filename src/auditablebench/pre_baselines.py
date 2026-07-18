"""Rule-based PRE scanner baselines."""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Mapping

from .pre import PreOverPrivilege, pre_score


_RISKY_PERMISSION_LEVELS = {"write", "execute", "network", "admin"}
_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "pre",
    "llm_judge_method",
)
_KIND_KEYWORDS = {
    "read": ("read", "summarize", "inspect", "list", "report"),
    "write": ("write", "save", "edit", "create", "update", "delete"),
    "network": ("fetch", "http", "url", "api", "web", "download"),
    "execute": ("run", "execute", "shell", "bash", "test", "compile"),
}


def _matches_keyword(text: str, keyword: str) -> bool:
    return re.search(rf"\b{re.escape(keyword)}\w*\b", text) is not None


def _allowed_permission_levels(task_or_role_spec: str) -> set[str]:
    text = task_or_role_spec.lower()
    allowed = {"read", "unknown"}
    for permission_level, keywords in _KIND_KEYWORDS.items():
        if any(_matches_keyword(text, keyword) for keyword in keywords):
            allowed.add(permission_level)
    return allowed


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


class OwaspExcessiveAgencyMethod:
    method_id = "owasp_excessive_agency"
    supports = {"pre_over_privilege"}

    def evaluate(self, task: PreOverPrivilege) -> Mapping[str, float]:
        view = task.method_view()
        flagged = {}
        for o in view:
            allowed = _allowed_permission_levels(o["task_or_role_spec"])
            excess = {
                c["name"]
                for c in o["declared_capabilities"]
                if c["permission_level"] not in allowed
            }
            flagged[o["instance_id"]] = excess
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
        return pre_score(flagged, task.instances)


def _llm_judge_methods() -> list:
    """One LlmJudgeNeededMethod per committed judge cache in _CACHE_DIR (separate rows, no union)."""
    return [LlmJudgeNeededMethod(p) for p in sorted(glob.glob(os.path.join(_CACHE_DIR, "*.json")))]


def pre_baseline_methods() -> list:
    return [
        FlagRiskyPermsMethod(),
        OwaspExcessiveAgencyMethod(),
        PrivilegeDiffOracleMethod(),
    ] + _llm_judge_methods()
