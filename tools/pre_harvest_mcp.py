"""Harvest unlabeled PRE staging instances from real MCP server tool manifests."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import re
import statistics
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pre_spec_features import (
    add_retain_prose_argument,
    deidentify_rows,
    write_json_rows,
    write_local_prose,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "pre_staging" / "mcp.json"
REGISTRY_URL = "https://registry.modelcontextprotocol.io/v0/servers"
USER_AGENT = "catchbench-mcp-harvester/0.1"
PROTOCOL_VERSION = "2025-06-18"
RAW_STAGING_KEYS = {
    "instance_id",
    "source",
    "provenance",
    "task_or_role_spec",
    "declared_capabilities",
}
DERIVED_STAGING_KEYS = (RAW_STAGING_KEYS - {"task_or_role_spec"}) | {
    "spec_tokens",
    "spec_token_overrides",
}
PROVENANCE_KEYS = {"repo", "commit", "path", "license"}
CAPABILITY_KEYS = {"name", "type", "permission_level"}
PERMISSION_LEVELS = {"read", "write", "execute", "network", "admin", "unknown"}

# Explicit source set: latest active registry entries with resolvable GitHub
# provenance and live unauthenticated streamable-HTTP tools/list responses.
SOURCE_SERVER_NAMES = (
    "ai.boolsai/scan",
    "ai.boolsai/signals",
    "ai.compeller/compel",
    "ac.tandem/docs-mcp",
    "ai.borealhost/mcp",
    "ai.bowmark/bowmark",
    "ai.com.mcp/linkedin",
    "ai.com.mcp/registry",
    "ai.com.mcp/lenny-rachitsky-podcast",
    "ai.com.mcp/petstore",
    "ai.com.mcp/strava",
    "ai.dreamlit/mcp",
    "ai.exa/exa",
    "ai.helixar/mcp",
    "ai.byteask/embedded-docs",
    "ai.justpublish/just-publish",
    "ai.keenable/web-search",
    "ai.inxy/seo-audit",
    "ai.law.mcp/lawyer-search",
    "ai.example4/xmp4",
    "ai.presentations/presentations-ai",
    "ai.mrmarket/mrmarket-mcp",
    "ai.preclick/preclick-mcp",
    "ai.pyrimid/pyrimid",
    "ai.sitepulsar/mcp",
    "ai.joinmultiplayer/gpu",
    "ai.masnavi/masnavi",
    "ai.nefesh/human-state",
    "ai.proxygate/mcp",
    "ai.tensorfeed/mcp-server",
    "ai.tunnelmind/sigil",
    "ai.roopslaw/legalsearch",
    "ai.sugra/api-mcp",
    "ai.thinkneo/control-plane",
    "ai.trydock/dock",
    "ai.unulu/unulu",
    "ai.urlcheck/urlcheck-mcp",
    "app.cnvs/whiteboard",
    "app.flaim/mcp",
    "app.himalayas/mcp",
    "app.humantaste/taste-mcp",
    "app.pulltrader/seller-economics",
    "ai.tunnelmind/scry",
    "app.ganty/mcp-server",
    "app.toolsnap/toolsnap-mcp",
    "art.travel/mcp",
    "bid.scope/aec",
    "bid.scope/claims",
    "bid.scope/legal",
    "bot.myagi/openagent-registry",
    "bot.yeehaw/events",
    "ch.fedlex-connector/fedlex-connector",
    "app.evlek/mcp-server",
    "ca.swiftsign/mcp",
    "app.nausika/mcp",
    "cloud.dchub/mcp-server",
    "co.ainumbers/tools",
    "co.launchtrust/launchtrust",
    "app.cardog/mcp",
    "ca.netgrant/canadian-grants",
    "co.bizverify/mcp",
    "ai.xpoz/social-insights",
    "bio.atlarium/habitat-database",
    "cn.whylingxi/insurance",
    "co.heista/api",
    "co.tempguru/event-staffing",
    "app.zooza/mcp-server",
    "ch.entscheidsuche/mcp",
    "com.agent-tune/agenttune",
    "com.appendix/appendix",
    "com.adbutler/mcp-server",
    "com.advocatemcp/advocate",
    "com.apple-rag/mcp-server",
    "com.babyblueviper/invinoveritas",
    "br.com.brasilnfe/fiscal",
    "com.arzbin/market-data",
    "com.bankruptcyobserver/mcp",
    "com.cityparity/cityparity",
    "com.anots/directory",
    "com.changethisfile/mcp",
    "com.checkrecall/vehicle-recalls",
    "com.clauxel.agentdataboundary/agentdataboundary-mcp",
    "com.clauxel.browserspendguard/browserspendguard-mcp",
    "com.clauxel.codexrunledger/codexrunledger-mcp",
    "com.cloudflare.mcp/mcp",
    "com.commonlands/optics-mcp",
    "com.100hires/100hires",
    "com.bikefuchs/bikefuchs",
    "com.blackveilsecurity/dns",
    "com.daedalmap/county-map",
    "com.daedalmap/earthquakes",
    "com.daedalmap/hurricanes",
    "com.daedalmap/tsunamis",
    "com.craftedtrust/mcp-shield",
    "com.crosswire-api/crosswire-polymarket-kalshi-arbitrage",
    "com.daedalmap/boundaries",
    "com.daedalmap/currency",
    "com.daedalmap/floods",
    "com.daedalmap/geocoding",
    "com.daedalmap/population",
    "com.daedalmap/reverse-geocoding",
    "com.daedalmap/tornadoes",
    "com.daedalmap/un_sdg",
    "com.daedalmap/volcanoes",
    "com.daedalmap/wildfires",
    "com.daedalmap/world_factbook",
    "com.defaultverifier/settlement-witness",
    "com.boostermage/mtg-prices",
    "com.compoid/mcp-server",
    "com.blockscout/mcp-server",
    "com.contrastcyber/api",
    "com.decision-anchor/da",
    "com.feode/feode-mcp",
    "com.geiant/mcp-agentcore",
    "com.gettreatmenthelp/gettreatmenthelp",
    "com.eveoy/mcp",
    "com.geiant/mcp-perception",
    "com.filingfirehose/firehose",
    "com.gotfreefax/mcp",
    "com.gribstream/mcp",
    "com.hemmabo/hemmabo-mcp-server",
    "com.hydrata/hydrata-mcp-server",
    "com.htagai/htag-docs",
    "com.joinsandwich/directory",
    "com.kaicalls/kaicalls",
    "com.lifescored/mcp",
    "com.fiatdock/fiatdock-mcp",
    "com.joinplexa/plexa",
    "com.kettlelogic/mcp-kettlelogic",
    "com.getvaletparking/valet-parking-directory",
    "com.hauntapi/haunt",
    "com.ifthenpay/payments",
    "com.linklyhq/linkly",
    "com.kapruka/kapruka-mcp",
    "com.loopxxi/loop-mcp",
    "com.lowtoxgear/scanner",
    "com.lowtoxgear/storefront",
    "com.kernelcad/kernelcad",
    "com.local-mcp/local-mcp",
    "com.luthersystems.insideout/mcp",
    "com.makometrics/mako-metrics",
    "com.memxus/memxus",
    "com.mux/mcp",
    "com.nursinghomedatabase/mcp",
    "com.olympus-bets/olympus-bets-analytics",
    "com.microsoft/microsoft-learn-mcp",
    "com.moltravel/travel",
    "com.nicheangle/niche",
    "com.metricspot/seo-mcp",
    "com.onchaindiligence/compliance",
)


@dataclass(frozen=True)
class ServerSpec:
    index: int
    name: str
    title: str
    description: str
    version: str
    repo_url: str
    repo_path: str
    remote_url: str


class HarvestError(RuntimeError):
    pass


def request_json(url: str, timeout: int = 30) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def is_latest_active(item: dict[str, Any]) -> bool:
    meta = item.get("_meta", {}).get("io.modelcontextprotocol.registry/official", {})
    return bool(meta.get("isLatest") and meta.get("status") == "active")


def streamable_http_url(server: dict[str, Any]) -> str | None:
    for remote in server.get("remotes") or []:
        if remote.get("type") != "streamable-http":
            continue
        if any(header.get("isRequired") for header in remote.get("headers") or []):
            continue
        url = remote.get("url")
        if isinstance(url, str) and url:
            return url
    return None


def github_repo_url(server: dict[str, Any]) -> str | None:
    repo = server.get("repository") or {}
    url = repo.get("url")
    if isinstance(url, str) and "github.com" in url.lower():
        return url
    return None


def resolve_server_specs(target_names: tuple[str, ...]) -> list[ServerSpec]:
    wanted = set(target_names)
    found: dict[str, dict[str, Any]] = {}
    cursor: str | None = None
    pages = 0

    while wanted - found.keys():
        url = f"{REGISTRY_URL}?limit=100"
        if cursor is not None:
            url += "&cursor=" + urllib.parse.quote(cursor)
        data = request_json(url)
        pages += 1

        for item in data.get("servers", []):
            server = item.get("server") or {}
            name = server.get("name")
            if name in wanted and is_latest_active(item):
                found[name] = server

        cursor = data.get("metadata", {}).get("nextCursor")
        if not cursor:
            break

    missing = [name for name in target_names if name not in found]
    if missing:
        raise HarvestError(f"missing registry entries: {missing}")

    specs: list[ServerSpec] = []
    for index, name in enumerate(target_names, start=1):
        server = found[name]
        repo_url = github_repo_url(server)
        remote_url = streamable_http_url(server)
        if repo_url is None or remote_url is None:
            raise HarvestError(f"target lost required repo or remote: {name}")
        repo = server.get("repository") or {}
        specs.append(
            ServerSpec(
                index=index,
                name=name,
                title=str(server.get("title") or name),
                description=clean_text(str(server.get("description") or "")),
                version=str(server.get("version") or ""),
                repo_url=repo_url,
                repo_path=str(repo.get("subfolder") or "."),
                remote_url=remote_url,
            )
        )
    print(f"resolved {len(specs)} registry specs from {pages} pages", file=sys.stderr)
    return specs


def parse_mcp_response(body: bytes, content_type: str) -> dict[str, Any] | None:
    text = body.decode("utf-8", errors="replace")
    if "text/event-stream" in content_type or text.lstrip().startswith(("event:", "data:")):
        events: list[str] = []
        data_lines: list[str] = []
        for line in text.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
            elif not line.strip() and data_lines:
                events.append("\n".join(data_lines))
                data_lines = []
        if data_lines:
            events.append("\n".join(data_lines))
        for event in events:
            try:
                obj = json.loads(event)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and ("result" in obj or "error" in obj):
                return obj
        return None

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def mcp_post(url: str, payload: dict[str, Any], session_id: str | None = None) -> tuple[dict[str, Any] | None, str | None]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": PROTOCOL_VERSION,
        "User-Agent": USER_AGENT,
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        body = response.read(5_000_000)
        next_session = (
            response.headers.get("Mcp-Session-Id")
            or response.headers.get("mcp-session-id")
            or session_id
        )
        obj = parse_mcp_response(body, response.headers.get("Content-Type", ""))
    return obj, next_session


def list_tools(spec: ServerSpec) -> tuple[ServerSpec, list[dict[str, Any]], str | None]:
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "catchbench-mcp-harvester", "version": "0.1"},
        },
    }
    try:
        init_response, session_id = mcp_post(spec.remote_url, initialize)
        if not init_response or init_response.get("error"):
            return spec, [], f"initialize failed: {init_response}"

        try:
            mcp_post(
                spec.remote_url,
                {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
                session_id,
            )
        except Exception:
            pass

        tools_response, _ = mcp_post(
            spec.remote_url,
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            session_id,
        )
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        return spec, [], f"{type(exc).__name__}: {exc}"

    if not tools_response or tools_response.get("error"):
        return spec, [], f"tools/list failed: {tools_response}"
    tools = tools_response.get("result", {}).get("tools")
    if not isinstance(tools, list):
        return spec, [], "tools/list had no result.tools list"
    if not tools:
        return spec, [], "tools/list returned no tools"
    return spec, [tool for tool in tools if isinstance(tool, dict)], None


def github_slug(repo_url: str) -> str | None:
    match = re.search(r"github\.com[:/]+([^/\s]+)/([^/\s#?]+)", repo_url)
    if not match:
        return None
    owner = match.group(1)
    repo = match.group(2).removesuffix(".git")
    return f"{owner}/{repo}"


def run_command(args: list[str], timeout: int = 45) -> str:
    completed = subprocess.run(
        args,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return completed.stdout.strip()


def github_repo_meta(repo_url: str, cache: dict[str, tuple[str, str]]) -> tuple[str, str]:
    slug = github_slug(repo_url)
    if slug is None:
        return "n/a", "unverified"
    if slug in cache:
        return cache[slug]

    commit = "n/a"
    license_id = "unverified"
    try:
        info = json.loads(run_command(["gh", "api", f"repos/{slug}"]))
        default_branch = info.get("default_branch") or "HEAD"
        license_info = info.get("license") or {}
        spdx = license_info.get("spdx_id")
        license_id = spdx or license_info.get("name") or "NOASSERTION"
        if license_id == "NOASSERTION":
            license_id = license_info.get("name") or "NOASSERTION"
        commit_obj = json.loads(run_command(["gh", "api", f"repos/{slug}/commits/{default_branch}"]))
        commit = str(commit_obj.get("sha") or "n/a")
    except Exception:
        try:
            ls_remote = run_command(["git", "ls-remote", repo_url, "HEAD"])
            commit = ls_remote.split()[0] if ls_remote else "n/a"
        except Exception:
            commit = "n/a"
    cache[slug] = (commit, license_id)
    return cache[slug]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def schema_text(tool: dict[str, Any]) -> str:
    try:
        return json.dumps(tool.get("inputSchema") or {}, sort_keys=True)
    except TypeError:
        return ""


def tool_text(tool: dict[str, Any]) -> str:
    return clean_text(
        " ".join(
            [
                str(tool.get("name") or ""),
                str(tool.get("description") or ""),
                schema_text(tool)[:2000],
            ]
        )
    ).lower()


EXECUTE_RE = re.compile(r"\b(shell|command|terminal|bash|exec|execute|script|python|node|code[_ -]?run|run[_ -]?code)\b")
NETWORK_RE = re.compile(
    r"\b(url|uri|http|https|fetch|web|website|browser|browse|crawl|scrape|search|endpoint|api|request)\b"
)


def permission_level(tool: dict[str, Any]) -> str:
    annotations = tool.get("annotations")
    if not isinstance(annotations, dict):
        annotations = {}
    text = tool_text(tool)

    if EXECUTE_RE.search(text):
        return "execute"
    if annotations.get("destructiveHint") is True:
        return "write"
    if NETWORK_RE.search(text):
        return "network"
    if annotations.get("readOnlyHint") is True:
        return "read"
    if annotations.get("openWorldHint") is True:
        return "network"
    return "unknown"


def capability_type(tool: dict[str, Any]) -> str:
    text = tool_text(tool)
    if EXECUTE_RE.search(text):
        return "execution"
    if NETWORK_RE.search(text):
        return "network"
    if re.search(r"\b(file|document|pdf|docx|csv|json|image|video|audio)\b", text):
        return "file"
    if re.search(r"\b(sql|database|query|table|record)\b", text):
        return "database"
    if re.search(r"\b(email|message|chat|slack|sms|notification)\b", text):
        return "messaging"
    if re.search(r"\b(order|payment|invoice|billing|commerce|purchase)\b", text):
        return "commerce"
    if re.search(r"\b(ticket|issue|task|calendar|schedule|project)\b", text):
        return "workflow"
    return "mcp_tool"


def normalize_capabilities(tools: list[dict[str, Any]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    capabilities: list[dict[str, str]] = []
    for tool in tools:
        name = clean_text(str(tool.get("name") or ""))
        if not name or name in seen:
            continue
        seen.add(name)
        capabilities.append(
            {
                "name": name,
                "type": capability_type(tool),
                "permission_level": permission_level(tool),
            }
        )
    return capabilities


def annotation_coverage(raw_tools_by_name: dict[str, list[dict[str, Any]]]) -> float:
    total = 0
    annotated = 0
    for tools in raw_tools_by_name.values():
        for tool in tools:
            total += 1
            if isinstance(tool.get("annotations"), dict) and tool["annotations"]:
                annotated += 1
    return annotated / total if total else 0.0


def instance_id(spec: ServerSpec) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", spec.name.lower()).strip("-")
    return f"mcp-{spec.index:04d}-{slug}"


def build_instance(
    spec: ServerSpec,
    tools: list[dict[str, Any]],
    repo_meta_cache: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    commit, license_id = github_repo_meta(spec.repo_url, repo_meta_cache)
    capabilities = normalize_capabilities(tools)
    if not capabilities:
        raise HarvestError(f"{spec.name} has no named tools")
    return {
        "instance_id": instance_id(spec),
        "source": "mcp",
        "provenance": {
            "repo": spec.repo_url,
            "commit": commit,
            "path": spec.repo_path,
            "license": license_id,
        },
        "task_or_role_spec": clean_text(f"{spec.title}: {spec.description}").rstrip(":"),
        "declared_capabilities": capabilities,
    }


def validate_rows(rows: list[dict[str, Any]]) -> None:
    instance_ids: set[str] = set()
    for row in rows:
        assert set(row) in (RAW_STAGING_KEYS, DERIVED_STAGING_KEYS), f"keys {row.keys()}"
        assert row["source"] == "mcp", row["source"]
        assert isinstance(row["instance_id"], str) and row["instance_id"], "instance_id"
        assert row["instance_id"] not in instance_ids, f"duplicate {row['instance_id']}"
        instance_ids.add(row["instance_id"])
        assert set(row["provenance"]) == PROVENANCE_KEYS, f"provenance keys {row['provenance'].keys()}"
        for key in PROVENANCE_KEYS:
            assert isinstance(row["provenance"][key], str) and row["provenance"][key], f"provenance.{key}"
        if "task_or_role_spec" in row:
            assert isinstance(row["task_or_role_spec"], str) and row["task_or_role_spec"], "task_or_role_spec"
        else:
            assert row["spec_tokens"] == sorted(set(row["spec_tokens"])), "spec_tokens"
            assert isinstance(row["spec_token_overrides"], dict), "spec_token_overrides"
        capabilities = row["declared_capabilities"]
        assert isinstance(capabilities, list) and capabilities, "declared_capabilities"
        names: set[str] = set()
        for capability in capabilities:
            assert set(capability) == CAPABILITY_KEYS, f"capability keys {capability.keys()}"
            assert isinstance(capability["name"], str) and capability["name"], "capability.name"
            assert capability["name"] not in names, f"duplicate capability {capability['name']}"
            names.add(capability["name"])
            assert isinstance(capability["type"], str) and capability["type"], "capability.type"
            assert capability["permission_level"] in PERMISSION_LEVELS, capability["permission_level"]


def harvest(max_workers: int, min_servers: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs = resolve_server_specs(SOURCE_SERVER_NAMES)
    raw_tools_by_name: dict[str, list[dict[str, Any]]] = {}
    failures: list[str] = []

    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(list_tools, spec): spec for spec in specs}
        for future in futures.as_completed(future_map):
            spec, tools, error = future.result()
            if error is not None:
                failures.append(f"{spec.name}: {error}")
                print(f"skip {spec.name}: {error}", file=sys.stderr)
                continue
            raw_tools_by_name[spec.name] = tools
            print(f"tools {spec.name}: {len(tools)}", file=sys.stderr)

    rows: list[dict[str, Any]] = []
    repo_meta_cache: dict[str, tuple[str, str]] = {}
    for spec in specs:
        tools = raw_tools_by_name.get(spec.name)
        if not tools:
            continue
        rows.append(build_instance(spec, tools, repo_meta_cache))

    if len(rows) < min_servers:
        raise HarvestError(f"only harvested {len(rows)} servers, below min {min_servers}")
    validate_rows(rows)

    tool_counts = [len(row["declared_capabilities"]) for row in rows]
    summary = {
        "servers": len(rows),
        "mean_tools_per_server": round(statistics.mean(tool_counts), 3),
        "annotation_coverage": round(annotation_coverage(raw_tools_by_name), 4),
        "failed_sources": len(failures),
    }
    return rows, summary


def validate_file(path: Path) -> tuple[int, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(rows, list), "top-level JSON must be a list"
    validate_rows(rows)
    sample = rows[0] if rows else {}
    return len(rows), sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--min-servers", type=int, default=100)
    parser.add_argument("--validate-only", action="store_true")
    add_retain_prose_argument(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.validate_only:
        count, sample = validate_file(args.output)
        print(f"count={count}")
        if sample:
            print(f"sample.instance_id={sample['instance_id']}")
            print(f"sample.source={sample['source']}")
            print(
                "sample.provenance="
                + json.dumps(sample["provenance"], sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            )
            print(f"sample.capabilities={len(sample['declared_capabilities'])}")
        return 0

    raw_rows, summary = harvest(args.max_workers, args.min_servers)
    write_local_prose(raw_rows, args.retain_prose)
    rows = deidentify_rows(raw_rows)
    write_json_rows(rows, args.output)
    print(json.dumps(summary, sort_keys=True))
    print(f"wrote {len(rows)} servers to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
