"""Harvest normalized PRE instances from public SWE-agent trajectories."""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "pre" / "sweagent.json"

BUCKET_BASE_URL = "https://swe-bench-submissions.s3.amazonaws.com"
SUBMISSION = "20240728_sweagent_gpt4o"
S3_PREFIX = f"lite/{SUBMISSION}/trajs"
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

SAMPLE_INSTANCE_IDS = [
    line.strip()
    for line in """
astropy__astropy-12907
astropy__astropy-14182
astropy__astropy-14365
astropy__astropy-14995
astropy__astropy-6938
astropy__astropy-7746
django__django-10914
django__django-10924
django__django-11001
django__django-11019
django__django-11039
django__django-11049
django__django-11099
django__django-11133
django__django-11179
django__django-11283
django__django-11422
django__django-11564
django__django-11583
django__django-11620
django__django-11630
django__django-11742
django__django-11797
django__django-11815
django__django-11848
django__django-11905
django__django-11910
django__django-11964
django__django-11999
django__django-12113
django__django-12125
django__django-12184
django__django-12284
django__django-12286
django__django-12308
django__django-12453
django__django-12470
django__django-12497
django__django-12589
django__django-12700
django__django-12708
django__django-12747
django__django-12856
django__django-12908
django__django-12915
django__django-12983
django__django-13028
django__django-13033
django__django-13158
django__django-13220
django__django-13230
django__django-13265
django__django-13315
django__django-13321
django__django-13401
django__django-13447
django__django-13448
django__django-13551
django__django-13590
django__django-13658
django__django-13660
django__django-13710
django__django-13757
django__django-13768
django__django-13925
django__django-13933
django__django-13964
django__django-14016
django__django-14017
django__django-14155
django__django-14238
django__django-14382
django__django-14411
django__django-14534
django__django-14580
django__django-14608
django__django-14667
django__django-14672
django__django-14730
django__django-14752
django__django-14787
django__django-14855
django__django-14915
django__django-14997
django__django-14999
django__django-15061
django__django-15202
django__django-15213
django__django-15252
django__django-15320
django__django-15347
django__django-15388
django__django-15400
django__django-15498
django__django-15695
django__django-15738
django__django-15781
django__django-15789
django__django-15790
django__django-15814
django__django-15819
django__django-15851
django__django-15902
django__django-15996
django__django-16041
django__django-16046
django__django-16139
django__django-16229
django__django-16255
django__django-16379
django__django-16400
django__django-16408
django__django-16527
django__django-16595
django__django-16816
django__django-16820
django__django-16873
django__django-16910
django__django-17051
django__django-17087
matplotlib__matplotlib-22711
mwaskom__seaborn-2848
mwaskom__seaborn-3010
mwaskom__seaborn-3190
mwaskom__seaborn-3407
pallets__flask-4045
pallets__flask-4992
pallets__flask-5063
psf__requests-1963
psf__requests-2148
""".splitlines()
    if line.strip()
]

READ_COMMANDS = {
    "open",
    "goto",
    "scroll",
    "scroll_down",
    "scroll_up",
    "search_dir",
    "search_file",
    "find_file",
}
WRITE_COMMANDS = {"edit", "create"}
# `submit` finalizes the answer; it grants no shell/code execution, so it is
# `unknown` (matching the synthetic SWE-agent toolset), not `execute`.
EXECUTE_COMMANDS = {"python", "bash"}


class ParseError(ValueError):
    pass


def permission_level(name: str) -> str:
    if name in READ_COMMANDS:
        return "read"
    if name in WRITE_COMMANDS:
        return "write"
    if name in EXECUTE_COMMANDS:
        return "execute"
    return "unknown"


def capability_type(name: str) -> str:
    if name in READ_COMMANDS:
        return "navigation"
    if name in WRITE_COMMANDS:
        return "editor"
    if name in EXECUTE_COMMANDS:
        return "execution"
    return "command"


def capability(name: str) -> dict[str, str]:
    return {
        "name": name,
        "type": capability_type(name),
        "permission_level": permission_level(name),
    }


def fetch_traj(instance_id: str) -> dict[str, Any]:
    url = f"{BUCKET_BASE_URL}/{S3_PREFIX}/{instance_id}.traj"
    proc = subprocess.run(
        ["curl.exe", "-sSL", "-A", BROWSER_UA, url],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    return json.loads(proc.stdout)


def extract_declared(system_content: str) -> list[str]:
    if "COMMANDS:" not in system_content:
        raise ParseError("missing COMMANDS block")

    block = system_content.split("COMMANDS:", 1)[1]
    stops = [idx for marker in ("\n\nPlease note", "\nRESPONSE", "\nFORMAT") if (idx := block.find(marker)) >= 0]
    if stops:
        block = block[: min(stops)]

    names: list[str] = []
    for line in block.splitlines():
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*):\s*$", line)
        if match:
            names.append(match.group(1))

    lower_system = system_content.lower()
    if "typical bash commands" in lower_system or "bash commands" in lower_system:
        names.append("bash")

    names = list(dict.fromkeys(names))
    if not names:
        raise ParseError("empty declared command set")
    return names


def first_action_token(action: Any) -> str | None:
    if not isinstance(action, str):
        return None
    for line in action.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped.split(maxsplit=1)[0]
    return None


def extract_used(history: list[dict[str, Any]], declared_names: list[str]) -> set[str]:
    declared = set(declared_names)
    used: set[str] = set()
    for msg in history:
        if msg.get("role") != "assistant":
            continue
        token = first_action_token(msg.get("action"))
        if token is None:
            continue
        if token in declared:
            used.add(token)
        elif token in {"python", "python3"}:
            used.add("python")
        else:
            used.add("bash")
    return used


def extract_problem_statement(history: list[dict[str, Any]], instance_id: str) -> str:
    for msg in history:
        if msg.get("role") != "user" or msg.get("is_demo"):
            continue
        content = msg.get("content")
        if not isinstance(content, str) or "ISSUE:" not in content:
            continue
        issue = content.split("ISSUE:", 1)[1]
        if "\nINSTRUCTIONS:" in issue:
            issue = issue.split("\nINSTRUCTIONS:", 1)[0]
        issue = issue.strip()
        if issue:
            return issue
    return instance_id


def parse_instance(instance_id: str, traj: dict[str, Any]) -> dict[str, Any]:
    history = traj.get("history")
    if not isinstance(history, list) or not history:
        raise ParseError("missing history")
    if not isinstance(history[0], dict) or history[0].get("role") != "system":
        raise ParseError("missing system message")

    system_content = history[0].get("content")
    if not isinstance(system_content, str):
        raise ParseError("system content is not text")

    declared_names = extract_declared(system_content)
    used = extract_used(history, declared_names)
    for inferred in ("bash", "python"):
        if inferred in used and inferred not in declared_names:
            declared_names.append(inferred)

    declared_set = set(declared_names)
    used &= declared_set

    return {
        "instance_id": instance_id,
        "source": "sweagent",
        "provenance": {
            "repo": "swe-bench/experiments",
            "commit": SUBMISSION,
            "path": f"{instance_id}.traj",
            "license": "unverified",
        },
        "task_or_role_spec": extract_problem_statement(history, instance_id),
        "declared_capabilities": [capability(name) for name in declared_names],
        "minimal_reference": [name for name in declared_names if name in used],
        "labels": {
            "excess_set": [name for name in declared_names if name not in used],
            "label_source": "declared_minus_used",
        },
    }


def stats(rows: list[dict[str, Any]], attempted: int) -> dict[str, float]:
    declared_sizes = [len(row["declared_capabilities"]) for row in rows]
    used_sizes = [len(row["minimal_reference"]) for row in rows]
    excess_sizes = [len(row["labels"]["excess_set"]) for row in rows]
    return {
        "attempted": attempted,
        "parsed": len(rows),
        "parse_rate": round(len(rows) / attempted, 4) if attempted else 0.0,
        "mean_declared": round(statistics.mean(declared_sizes), 3) if rows else 0.0,
        "mean_used": round(statistics.mean(used_sizes), 3) if rows else 0.0,
        "mean_excess": round(statistics.mean(excess_sizes), 3) if rows else 0.0,
    }


def harvest(out_path: Path) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    for i, instance_id in enumerate(SAMPLE_INSTANCE_IDS, start=1):
        try:
            traj = fetch_traj(instance_id)
            rows.append(parse_instance(instance_id, traj))
        except Exception as exc:
            print(f"skip {instance_id}: {exc}", file=sys.stderr)
        else:
            print(f"parsed {i}/{len(SAMPLE_INSTANCE_IDS)} {instance_id}", file=sys.stderr)

    if len(rows) < 100:
        raise RuntimeError(f"only parsed {len(rows)} clean instances")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows, stats(rows, len(SAMPLE_INSTANCE_IDS))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    rows, summary = harvest(args.out)
    print(json.dumps(summary, sort_keys=True))
    print(f"wrote {len(rows)} instances to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
