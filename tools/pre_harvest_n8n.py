"""Harvest unlabeled PRE staging rows from public n8n workflow templates."""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from pre_spec_features import add_retain_prose_argument, deidentify_rows, write_local_prose


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "pre_staging" / "n8n.json"

SEARCH_ENDPOINT = "https://api.n8n.io/api/templates/search"
DETAIL_ENDPOINT = "https://api.n8n.io/api/templates/workflows/{template_id}"
AI_AGENT_NODE = "@n8n/n8n-nodes-langchain.agent"
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

# Discovered from:
# https://api.n8n.io/api/templates/search?rows=100&page=N&nodes=%40n8n%2Fn8n-nodes-langchain.agent
TEMPLATE_IDS = [
    11366, 6270, 8573, 5819, 7639, 5035, 8500, 6281, 7756, 5148, 5110, 5962, 15369, 9867, 4827, 5338,
    13503, 9437, 12462, 9200, 6287, 4968, 5678, 11204, 9383, 8428, 7156, 5010, 13526, 4966, 4722, 14167,
    5948, 12325, 8093, 15686, 9814, 6841, 6524, 5626, 13409, 9429, 8597, 8591, 5796, 13958, 9626, 4600,
    9277, 5523, 3066, 5398, 4484, 9801, 7467, 5799, 5817, 4723, 8237, 5857, 7422, 5832, 4557, 5541,
    13919, 5938, 5881, 4721, 14008, 5808, 4912, 5296, 7455, 5828, 12299, 11290, 16234, 15687, 5163, 7004,
    3586, 15933, 5953, 3859, 10290, 10111, 7154, 8721, 4767, 5159, 3770, 3050, 13791, 9999, 8604, 6542,
    4967, 5611, 15484, 8210, 7449, 4949, 5751, 5741, 5374, 4975, 6771, 6531, 5128, 3790, 16419, 5024,
    3443, 9986, 5552, 8654, 3250, 6290, 6538, 5734, 3514, 5413, 5139, 4366, 5553, 7957, 4696, 5002,
    2465, 6993, 6532, 12110, 5821, 4247, 5286, 4526, 8326, 5842, 5789, 5011, 4365, 3905, 10132, 15051,
    10214, 9375, 4877, 4868, 5294, 5202, 6239, 5908, 5607, 5305, 3900, 3940, 3291, 3135, 4733, 10157,
    9026, 5368, 2982, 13270, 13269, 13250, 10665, 9562, 8724, 8448, 6387, 6018, 5897, 5303, 2753, 4674,
    3657, 2859, 2786, 9546, 8090, 5405, 4875, 3986, 3379, 5074, 4930, 4740, 4739, 4402, 13271, 13015,
    7979, 5469, 4459, 4376, 3942, 3224, 13572, 10150, 10123, 9764, 9438, 8562, 6020, 5749, 4588, 9100,
    5862, 5820, 5757, 4689, 4478, 4399, 5656, 5617, 5435, 4678, 4057, 7215, 6537, 4494, 3656, 2872,
    15918, 14316, 9544, 5682, 5375, 4754, 4735, 4630, 4371, 16251, 14372, 13553, 13357, 12600, 12599, 12497,
    12346, 11600, 10924, 10268, 10282, 10281, 10119, 10045, 9592, 9558, 9402, 8008, 7157, 6153, 5947, 5942,
    5918, 5913, 5866, 5595, 5508, 5012, 4766, 2950, 9577, 8662, 6332, 5454, 4501, 3617, 5926, 5070,
    5042, 4910, 4794, 4102, 3425, 3336, 1954, 9270, 7625, 5369, 4589, 3798, 3694, 16439, 10666, 7671,
    7643, 7470, 6844, 5291, 5111, 3751, 3501, 14321, 13787, 13516, 11829, 10142, 9455, 9212, 6500, 6330,
    6031, 5951, 5923, 5871, 5622, 5521, 5292, 4891, 4755, 4086, 3189, 3100, 3025, 14449, 13088, 10199,
    9851, 10096, 5829, 5687, 5614, 5463, 4783, 4641, 4551, 4083, 3314, 2703, 2621, 13182, 13172, 9435,
    6906, 5924, 5858, 5784, 5597, 5511, 5164, 5096, 4889, 4638, 4233, 3906, 3473, 3303, 2845, 2466,
    13089, 13080, 10230, 9850, 7666, 7544, 7168, 6827, 5843, 5807, 5431, 5130, 4879, 4774, 4694, 4368,
    3686, 2986, 2864, 2783, 2777, 12289, 10256, 9545, 5478, 5259, 5100, 4847, 4483, 4452, 4237, 3997,
    3959, 3879, 3713, 3350, 3348, 3082, 3078, 2752, 2606, 2462, 16513, 16508, 16459, 16152, 15692, 16052,
    16049, 16039, 15888, 14805, 14080, 14267, 14254, 14168, 14165, 13759, 13676, 13542, 13528, 13453, 13367, 12644,
    12286, 12348, 12347, 11605, 11807, 11619, 11409, 11190, 11152, 11058, 10156, 11566, 11279, 11128, 11495, 11479,
    11362, 11368, 11370, 10440, 11083, 10863, 10837, 10815, 10772, 10625, 10538, 10786, 10722, 10681, 10626, 10624,
    10536, 10465, 10455, 9310, 8752, 10621, 10457, 10569, 10525, 10522, 10435, 10379, 10330, 10327, 10352, 10316,
    10312, 10310, 10292, 10288, 10287, 10441, 10314, 10293, 10033, 10252, 10248, 10240, 10246, 10242, 10241, 10239,
    10148, 10103, 9983, 10086, 10009, 9981, 9834, 9754, 9861, 9607, 9807, 9798, 9803, 9802, 9800, 9799,
    9797, 8877, 9636, 9628, 9561, 9552, 9543, 9473, 9211, 9494, 9486, 9400, 7653, 7632, 7405, 7155,
    7143, 5993, 5946, 5944, 5914, 5874, 5863, 5845, 5657, 5504, 5370, 5022, 4741, 4717, 4553, 4092,
    3891, 3765, 3670, 2978, 2862, 2749, 16175, 10630, 12971, 9048, 6914, 5619, 5473, 5383, 5309, 5160,
    5103, 4556, 4474, 4414, 3908, 3896, 3804, 3433, 3131, 3089, 3053, 2883, 2682, 14263, 13853, 13893,
    10900, 8607, 8592, 8585, 5666, 5723, 5492, 5475, 5461, 5421, 5039, 5014, 4873, 4872, 4841, 4684,
    4373, 4288, 4043, 3799, 3772, 3672, 3592, 3585, 3535, 3192, 3178, 3151, 2956, 2846, 2718, 2557,
    13704, 13185, 11138, 10578, 7672, 7401, 7288, 5468,
]

OUTPUT_KEYS = {
    "instance_id",
    "source",
    "provenance",
    "task_or_role_spec",
    "declared_capabilities",
}
PERM_LEVELS = {"read", "write", "execute", "network", "admin", "unknown"}

WRITE_WORDS = {
    "add",
    "append",
    "archive",
    "book",
    "cancel",
    "clear",
    "copy",
    "create",
    "delete",
    "draft",
    "edit",
    "insert",
    "invite",
    "move",
    "post",
    "publish",
    "remove",
    "reply",
    "send",
    "set",
    "share",
    "submit",
    "transfer",
    "update",
    "upload",
    "write",
}
READ_WORDS = {
    "analyze",
    "check",
    "download",
    "fetch",
    "find",
    "get",
    "list",
    "lookup",
    "monitor",
    "read",
    "retrieve",
    "search",
    "select",
    "summarize",
}
EXECUTE_WORDS = {"code", "execute", "run", "script", "workflow"}
NETWORK_WORDS = {
    "api",
    "brave",
    "browser",
    "calendar",
    "discord",
    "drive",
    "dropbox",
    "facebook",
    "firecrawl",
    "gmail",
    "google",
    "hubspot",
    "http",
    "instagram",
    "jira",
    "linkedin",
    "mail",
    "mcp",
    "notion",
    "reddit",
    "rss",
    "serp",
    "slack",
    "telegram",
    "twitter",
    "web",
    "whatsapp",
    "wikipedia",
    "wolfram",
}


class ParseError(ValueError):
    pass


def fetch_json(url: str) -> dict[str, Any]:
    proc = subprocess.run(
        ["curl.exe", "-sSL", "-A", BROWSER_UA, url],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    return json.loads(proc.stdout)


def discover_template_ids(pages: int, rows: int) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    for page in range(1, pages + 1):
        url = f"{SEARCH_ENDPOINT}?rows={rows}&page={page}&nodes=%40n8n%2Fn8n-nodes-langchain.agent"
        payload = fetch_json(url)
        for workflow in payload.get("workflows", []):
            template_id = int(workflow["id"])
            if template_id not in seen:
                ids.append(template_id)
                seen.add(template_id)
    return ids


def fetch_template(template_id: int) -> dict[str, Any]:
    payload = fetch_json(DETAIL_ENDPOINT.format(template_id=template_id))
    workflow = payload.get("workflow")
    if not isinstance(workflow, dict) or workflow.get("status") != "published":
        raise ParseError("missing published workflow")
    return workflow


def words(text: str) -> list[str]:
    return [
        w.lower()
        for w in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", text)
        if re.search(r"[A-Za-z]", w)
    ]


def normalize_type(node_type: str) -> str:
    raw = node_type.rsplit(".", 1)[-1]
    raw = re.sub(r"Tool$", "", raw)
    raw = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", raw).lower()
    return raw or "unknown"


def flatten_strings(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            out.append(str(k))
            out.extend(flatten_strings(v))
    elif isinstance(value, list):
        for item in value:
            out.extend(flatten_strings(item))
    return out


def permission_level(node: dict[str, Any]) -> str:
    node_type = str(node.get("type", ""))
    lower_type = node_type.lower()
    type_words = set(words(node_type))
    param_text = " ".join(flatten_strings(node.get("parameters", {})))
    combined_words = set(words(f"{node.get('name', '')} {node_type} {param_text}"))

    if "http" in lower_type or "api" in type_words:
        return "network"
    if type_words & EXECUTE_WORDS or "code" in lower_type:
        return "execute"
    if combined_words & WRITE_WORDS:
        return "write"
    if "api" in lower_type or combined_words & NETWORK_WORDS:
        return "network"
    if combined_words & READ_WORDS:
        return "read"
    return "unknown"


def get_connections(workflow_json: dict[str, Any]) -> dict[str, Any]:
    connections = workflow_json.get("connections", {})
    if not isinstance(connections, dict):
        return {}
    return connections


def tool_node_names_for_agents(workflow_json: dict[str, Any], agent_names: set[str]) -> list[str]:
    tools: list[str] = []
    for source_name, connection_types in get_connections(workflow_json).items():
        if not isinstance(connection_types, dict):
            continue
        for connection_type, outputs in connection_types.items():
            if not isinstance(outputs, list):
                continue
            for output in outputs:
                if not isinstance(output, list):
                    continue
                for edge in output:
                    if not isinstance(edge, dict):
                        continue
                    if edge.get("node") in agent_names and (
                        connection_type == "ai_tool" or edge.get("type") == "ai_tool"
                    ):
                        tools.append(source_name)
    return ordered_unique(tools)


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def uniquify_capability_names(capabilities: list[dict[str, str]]) -> list[dict[str, str]]:
    counts: dict[str, int] = {}
    out: list[dict[str, str]] = []
    for cap in capabilities:
        name = cap["name"]
        counts[name] = counts.get(name, 0) + 1
        if counts[name] > 1:
            cap = {**cap, "name": f"{name} #{counts[name]}"}
        out.append(cap)
    return out


def find_system_messages(value: Any) -> list[str]:
    messages: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "systemMessage" and isinstance(child, str) and child.strip():
                messages.append(normalize_text(child))
            else:
                messages.extend(find_system_messages(child))
    elif isinstance(value, list):
        for child in value:
            messages.extend(find_system_messages(child))
    return messages


def normalize_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_task_spec(template: dict[str, Any], agents: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    name = template.get("name")
    description = template.get("description")
    if isinstance(name, str) and name.strip():
        parts.append(normalize_text(name))
    if isinstance(description, str) and description.strip():
        parts.append(normalize_text(description))

    system_messages: list[str] = []
    for agent in agents:
        system_messages.extend(find_system_messages(agent.get("parameters", {})))
    for idx, message in enumerate(ordered_unique(system_messages), start=1):
        parts.append(f"AI Agent systemMessage {idx}:\n{message}")

    return "\n\n".join(parts)


def parse_template(template_id: int, template: dict[str, Any]) -> dict[str, Any] | None:
    workflow_json = template.get("workflow")
    if not isinstance(workflow_json, dict):
        raise ParseError("missing workflow JSON")

    nodes = workflow_json.get("nodes")
    if not isinstance(nodes, list):
        raise ParseError("missing nodes")
    by_name = {str(node.get("name")): node for node in nodes if isinstance(node, dict)}
    agents = [node for node in nodes if isinstance(node, dict) and node.get("type") == AI_AGENT_NODE]
    if not agents:
        return None

    tool_names = tool_node_names_for_agents(workflow_json, {str(agent.get("name")) for agent in agents})
    if not tool_names:
        return None

    capabilities: list[dict[str, str]] = []
    for name in tool_names:
        node = by_name.get(name)
        if node is None:
            continue
        node_type = str(node.get("type", "unknown"))
        capabilities.append(
            {
                "name": str(node.get("name") or node_type),
                "type": normalize_type(node_type),
                "permission_level": permission_level(node),
            }
        )

    if not capabilities:
        return None

    return {
        "instance_id": f"n8n-{template_id}",
        "source": "n8n",
        "provenance": {
            "repo": "n8n.io/workflows",
            "commit": str(template_id),
            "path": str(template_id),
            "license": "unverified",
        },
        "task_or_role_spec": build_task_spec(template, agents),
        "declared_capabilities": uniquify_capability_names(capabilities),
    }


def validate_row(row: dict[str, Any]) -> None:
    assert set(row) == OUTPUT_KEYS, set(row)
    assert isinstance(row["instance_id"], str) and row["instance_id"]
    assert row["source"] == "n8n"
    assert set(row["provenance"]) == {"repo", "commit", "path", "license"}
    assert isinstance(row["task_or_role_spec"], str) and row["task_or_role_spec"]
    assert isinstance(row["declared_capabilities"], list)
    for cap in row["declared_capabilities"]:
        assert set(cap) == {"name", "type", "permission_level"}, cap
        assert isinstance(cap["name"], str) and cap["name"]
        assert isinstance(cap["type"], str) and cap["type"]
        assert cap["permission_level"] in PERM_LEVELS, cap


def harvest(template_ids: list[int], target_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    stats = {"attempted": 0, "errors": 0, "empty_tool_rosters": 0, "duplicates": 0}
    for template_id in template_ids:
        if template_id in seen:
            stats["duplicates"] += 1
            continue
        seen.add(template_id)
        stats["attempted"] += 1
        try:
            template = fetch_template(template_id)
            row = parse_template(template_id, template)
        except Exception as exc:
            stats["errors"] += 1
            print(f"skip {template_id}: {exc}", file=sys.stderr)
            continue
        if row is None:
            stats["empty_tool_rosters"] += 1
            print(f"skip {template_id}: no AI Agent tool wiring", file=sys.stderr)
            continue
        validate_row(row)
        rows.append(row)
        print(f"parsed {len(rows)}/{target_count} n8n-{template_id}", file=sys.stderr)
        if len(rows) >= target_count:
            break

    if len(rows) < 150:
        raise RuntimeError(f"only harvested {len(rows)} rows")

    stats["written"] = len(rows)
    stats["mean_tools"] = round(
        statistics.mean(len(row["declared_capabilities"]) for row in rows), 3
    )
    return rows, stats


def write_rows(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument("--target-count", type=int, default=220)
    parser.add_argument("--discover-pages", type=int, help="Print discovered AI Agent template IDs and exit.")
    parser.add_argument("--discover-rows", type=int, default=100)
    add_retain_prose_argument(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.discover_pages:
        ids = discover_template_ids(args.discover_pages, args.discover_rows)
        print(json.dumps(ids))
        return 0

    raw_rows, stats = harvest(TEMPLATE_IDS, args.target_count)
    write_local_prose(raw_rows, args.retain_prose)
    rows = deidentify_rows(raw_rows)
    write_rows(rows, args.output)
    print(json.dumps(stats, sort_keys=True))
    print(f"wrote {len(rows)} instances to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
