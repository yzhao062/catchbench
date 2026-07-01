"""Harvest InjecAgent test cases into PRE over-privilege instances."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


REPO_URL = "https://github.com/uiuc-kang-lab/InjecAgent.git"
REPO_SLUG = "uiuc-kang-lab/InjecAgent"
LICENSE = "MIT"
TEST_CASE_FILES = (
    "data/test_cases_dh_base.json",
    "data/test_cases_ds_base.json",
    "data/test_cases_dh_enhanced.json",
    "data/test_cases_ds_enhanced.json",
)

TOOLKIT_TYPES = {
    "Amazon": "ecommerce",
    "AugustSmartLock": "smart_lock",
    "BankManager": "banking",
    "Binance": "crypto_exchange",
    "CiscoUmbrella": "network_security",
    "DeepfakeGenerator": "media_generation",
    "Dropbox": "cloud_storage",
    "EmergencyDispatchSystem": "emergency_dispatch",
    "EpicFHIR": "health_records",
    "EthereumManager": "blockchain",
    "EvernoteManager": "notes",
    "Expedia": "travel",
    "FacebookManager": "social_media",
    "FedExShipManager": "shipping",
    "GitHub": "code_hosting",
    "Gmail": "email",
    "GoogleCalendar": "calendar",
    "GoogleHome": "smart_home",
    "GoogleMap": "maps",
    "GoogleSearch": "web_search",
    "IndoorRobot": "robotics",
    "InvestmentManager": "investing",
    "NortonIdentitySafe": "identity_safe",
    "Shopify": "commerce",
    "Slack": "chat",
    "Spokeo": "people_search",
    "TDAmeritrade": "brokerage",
    "Teladoc": "healthcare",
    "Terminal": "shell",
    "The23andMe": "genetic_data",
    "Todoist": "tasks",
    "TrafficControl": "traffic_control",
    "Twilio": "sms",
    "TwitterManager": "social_media",
    "Venmo": "payments",
    "WebBrowser": "browser",
}

NETWORK_TOOL_NAMES = {
    "SendEmail",
    "SendInvitations",
    "SendMessage",
    "SendSms",
    "ShareData",
    "SharePassword",
}
NETWORK_FULL_NAMES = {
    "WebBrowserClickElement",
    "WebBrowserGetElementsDetails",
    "WebBrowserInputText",
    "WebBrowserNavigateTo",
    "WebBrowserRetrieveElements",
}
WRITE_WORDS = {
    "add",
    "apply",
    "auto",
    "book",
    "call",
    "cancel",
    "clear",
    "compile",
    "consult",
    "control",
    "create",
    "delete",
    "deposit",
    "deploy",
    "disable",
    "draft",
    "edit",
    "enable",
    "export",
    "fill",
    "follow",
    "generate",
    "give",
    "grant",
    "handle",
    "hand",
    "import",
    "join",
    "leave",
    "like",
    "lock",
    "make",
    "manage",
    "manipulate",
    "modify",
    "move",
    "order",
    "pay",
    "place",
    "post",
    "redirect",
    "remove",
    "reply",
    "request",
    "retweet",
    "revoke",
    "schedule",
    "set",
    "share",
    "sign",
    "store",
    "trade",
    "transfer",
    "unblock",
    "unfollow",
    "unlike",
    "unlock",
    "update",
    "upload",
    "withdraw",
}
READ_WORDS = {
    "access",
    "analyze",
    "check",
    "download",
    "estimate",
    "find",
    "get",
    "list",
    "look",
    "monitor",
    "predict",
    "read",
    "retrieve",
    "reverse",
    "search",
    "track",
    "verify",
    "view",
}


def run_git(args: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def words(name: str) -> list[str]:
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", name)
    return [p.lower() for p in parts]


def snake_case(name: str) -> str:
    return "_".join(words(name)) or "unknown"


def permission_level(full_name: str, meta: dict[str, str]) -> str:
    toolkit = meta["toolkit"]
    tool_name = meta["tool_name"]
    tool_words = set(words(tool_name))
    summary = meta["summary"].lower()

    if toolkit == "Terminal" or "execute a terminal command" in summary:
        return "execute"
    if full_name in NETWORK_FULL_NAMES:
        return "network"
    if toolkit == "GoogleSearch" and full_name != "GoogleSearchGetSearchHistory":
        return "network"
    if tool_name in NETWORK_TOOL_NAMES:
        return "network"
    if "send an email" in summary or "send an sms" in summary or "send a message" in summary:
        return "network"

    if tool_words & WRITE_WORDS:
        return "write"
    if tool_words & READ_WORDS:
        return "read"
    return "unknown"


def load_tool_defs(source_dir: Path) -> dict[str, dict[str, str]]:
    path = source_dir / "data" / "tools.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, str]] = {}
    for toolkit in rows:
        toolkit_name = toolkit["name_for_model"]
        for tool in toolkit["tools"]:
            full_name = f"{toolkit_name}{tool['name']}"
            out[full_name] = {
                "toolkit": toolkit_name,
                "tool_name": tool["name"],
                "summary": tool.get("summary", ""),
            }
    return out


def capability(name: str, tool_defs: dict[str, dict[str, str]]) -> dict[str, str]:
    meta = tool_defs.get(name)
    if meta is None:
        meta = {"toolkit": "", "tool_name": name, "summary": ""}
    toolkit = meta["toolkit"]
    return {
        "name": name,
        "type": TOOLKIT_TYPES.get(toolkit, snake_case(toolkit)),
        "permission_level": permission_level(name, meta),
    }


def ordered_unique(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def load_unique_cases(source_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates: list[dict[str, Any]] = []
    seen_rosters: set[tuple[str, ...]] = set()
    stats = {"raw_cases": 0, "overlap_skipped": 0, "duplicate_rosters": 0}

    for rel_path in TEST_CASE_FILES:
        rows = json.loads((source_dir / rel_path).read_text(encoding="utf-8"))
        stats["raw_cases"] += len(rows)
        for row in rows:
            user_tool = row["User Tool"]
            attacker_tools = list(row["Attacker Tools"])
            if user_tool in attacker_tools:
                stats["overlap_skipped"] += 1
                continue
            roster_key = tuple(sorted(ordered_unique([user_tool, *attacker_tools])))
            if roster_key in seen_rosters:
                stats["duplicate_rosters"] += 1
                continue
            seen_rosters.add(roster_key)
            candidates.append(
                {
                    "source_path": rel_path,
                    "user_tool": user_tool,
                    "attacker_tools": attacker_tools,
                    "attack_type": row["Attack Type"],
                    "task": row["User Instruction"],
                    "roster_key": roster_key,
                }
            )
    stats["unique_rosters"] = len(candidates)
    return candidates, stats


def select_balanced(candidates: list[dict[str, Any]], per_user_limit: int) -> list[dict[str, Any]]:
    if per_user_limit <= 0:
        return sorted(candidates, key=lambda x: (x["user_tool"], x["attack_type"], x["roster_key"]))

    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_user[candidate["user_tool"]].append(candidate)

    selected: list[dict[str, Any]] = []
    for user_tool in sorted(by_user):
        by_attack_type: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        rows = sorted(by_user[user_tool], key=lambda x: (x["attack_type"], x["roster_key"]))
        for row in rows:
            by_attack_type[row["attack_type"]].append(row)

        user_selected: list[dict[str, Any]] = []
        attack_types = sorted(by_attack_type)
        while len(user_selected) < per_user_limit:
            progressed = False
            for attack_type in attack_types:
                queue = by_attack_type[attack_type]
                if not queue:
                    continue
                user_selected.append(queue.popleft())
                progressed = True
                if len(user_selected) >= per_user_limit:
                    break
            if not progressed:
                break
        selected.extend(user_selected)
    return selected


def build_instances(
    candidates: list[dict[str, Any]], tool_defs: dict[str, dict[str, str]], commit: str
) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for index, row in enumerate(candidates, start=1):
        declared_names = ordered_unique([row["user_tool"], *row["attacker_tools"]])
        instances.append(
            {
                "instance_id": f"injecagent-{index:04d}",
                "source": "injecagent",
                "provenance": {
                    "repo": REPO_SLUG,
                    "commit": commit,
                    "path": row["source_path"],
                    "license": LICENSE,
                },
                "task_or_role_spec": row["task"],
                "declared_capabilities": [capability(name, tool_defs) for name in declared_names],
                "minimal_reference": [row["user_tool"]],
                "labels": {
                    "excess_set": row["attacker_tools"],
                    "label_source": "roster_relabel",
                    "attack_type": row["attack_type"],
                },
            }
        )
    return instances


def source_dir_from_args(args: argparse.Namespace) -> tuple[Path, str, tempfile.TemporaryDirectory[str] | None]:
    if args.source_dir:
        source_dir = Path(args.source_dir).resolve()
        commit = args.commit or run_git(["rev-parse", "HEAD"], cwd=source_dir)
        return source_dir, commit, None

    temp = tempfile.TemporaryDirectory(prefix="injecagent-src-")
    source_dir = Path(temp.name) / "InjecAgent"
    run_git(["clone", "--depth", "1", REPO_URL, str(source_dir)])
    commit = run_git(["rev-parse", "HEAD"], cwd=source_dir)
    return source_dir, commit, temp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", help="Existing InjecAgent checkout. If omitted, clone from GitHub.")
    parser.add_argument("--commit", help="Commit hash to use when --source-dir is not a git checkout.")
    parser.add_argument("--per-user-limit", type=int, default=20)
    parser.add_argument("--output", default="data/pre/injecagent.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir, commit, temp = source_dir_from_args(args)
    try:
        tool_defs = load_tool_defs(source_dir)
        candidates, stats = load_unique_cases(source_dir)
        selected = select_balanced(candidates, args.per_user_limit)
        instances = build_instances(selected, tool_defs, commit)

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(instances, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

        mean_excess = sum(len(row["attacker_tools"]) for row in selected) / len(selected)
        print(f"wrote={output}")
        print(f"commit={commit}")
        print(f"raw_cases={stats['raw_cases']}")
        print(f"overlap_skipped={stats['overlap_skipped']}")
        print(f"duplicate_rosters={stats['duplicate_rosters']}")
        print(f"unique_rosters={stats['unique_rosters']}")
        print(f"written={len(instances)}")
        print(f"dedup_ratio_raw_to_written={stats['raw_cases'] / len(instances):.2f}")
        print(f"mean_excess_size={mean_excess:.2f}")
    finally:
        if temp is not None:
            temp.cleanup()


if __name__ == "__main__":
    main()
