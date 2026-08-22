"""Generate the machine-readable inventory of committed derived data.

The manifest is data-derived: every JSON artifact under ``data/`` must match a known schema and
source class before any output is written. This makes a newly added or malformed artifact a hard
failure instead of silently producing an incomplete inventory.

Two properties matter for a licensing record and are enforced rather than assumed. An artifact
that holds a subset of one source set reports the declarations of the records it actually
contains, and an instance ID with no committed source record aborts the run instead of borrowing
a distribution it has no provenance for. Upstream references are typed, so a mutable template ID
or submission label is never presented as a pinned revision.

Usage:

    python tools/emit_asset_manifest.py --check
    python tools/emit_asset_manifest.py --update
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "ASSET_MANIFEST.json"
PRE_SOURCES = ("crewai", "injecagent", "mcp", "n8n", "sweagent", "synthetic")
WHO_AND_WHEN = "whoandwhen"

# How the recorded upstream identifier behaves. A git commit or a Hugging Face dataset revision
# addresses one immutable object; an n8n template ID and a SWE-bench submission label address a
# mutable location whose content can change under the same name. Calling the second kind a
# "pinned revision" would overstate what the record proves, so the schema names the two apart.
IMMUTABLE = "immutable_revision"
MUTABLE = "mutable_identifier"
NOT_APPLICABLE = "not_applicable"

SOURCE_METADATA = {
    "crewai": {
        "display_name": "public CrewAI projects",
        "external": True,
        "project_url": "https://github.com/topics/crewai",
        "license_status": "mixed; unresolved records remain",
        "identifier_kind": IMMUTABLE,
    },
    "injecagent": {
        "display_name": "InjecAgent",
        "external": True,
        "project_url": "https://github.com/uiuc-kang-lab/InjecAgent",
        "license_status": "established: MIT",
        "identifier_kind": IMMUTABLE,
    },
    "mcp": {
        "display_name": "Model Context Protocol Registry servers",
        "external": True,
        "project_url": "https://registry.modelcontextprotocol.io/",
        "license_status": "mixed; unresolved records remain",
        "identifier_kind": IMMUTABLE,
    },
    "n8n": {
        "display_name": "n8n workflow templates",
        "external": True,
        "project_url": "https://n8n.io/workflows/",
        "license_status": "unresolved",
        "identifier_kind": MUTABLE,
        "identifier_note": (
            "a template ID addressing the live n8n template gallery; the template behind an ID "
            "can be edited or withdrawn, so it records where a workflow came from, not which "
            "bytes were read"
        ),
        "artifact_base_url": "https://api.n8n.io/api/templates/workflows/",
    },
    "sweagent": {
        "display_name": "SWE-agent trajectories",
        "external": True,
        "project_url": "https://www.swebench.com/",
        "license_status": "unresolved",
        "identifier_kind": MUTABLE,
        "identifier_note": (
            "a SWE-bench submission label, not an object version; "
            "tools/pre_harvest_sweagent.py downloads each trajectory from the S3 prefix below, "
            "which is served without a recorded checksum"
        ),
        "artifact_base_url": (
            "https://swe-bench-submissions.s3.amazonaws.com/lite/20240728_sweagent_gpt4o/trajs/"
        ),
    },
    "synthetic": {
        "display_name": "CatchBench-authored synthetic PRE records",
        "external": False,
        "project_url": "https://github.com/yzhao062/catchbench",
        "license_status": "established: MIT",
        "identifier_kind": NOT_APPLICABLE,
    },
    WHO_AND_WHEN: {
        "display_name": "Who&When",
        "external": True,
        "project_url": "https://huggingface.co/datasets/Kevin355/Who_and_When",
        "license_status": "unresolved for dataset; associated code is MIT",
        "identifier_kind": IMMUTABLE,
    },
}


class ManifestError(RuntimeError):
    """The data tree cannot be represented completely and accurately."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read valid JSON from {path}: {exc}") from exc


def _repo_url(repo: str) -> str:
    if repo == "authored":
        return "https://github.com/yzhao062/catchbench"
    if repo == "n8n.io/workflows":
        return "https://n8n.io/workflows/"
    if repo.startswith(("https://", "http://")):
        return repo.removesuffix(".git")
    if "/" in repo:
        return f"https://github.com/{repo.removesuffix('.git')}"
    raise ManifestError(f"cannot form an upstream URL for repository {repo!r}")


def _pre_source_sets(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    source_sets: dict[str, dict[str, Any]] = {}
    instances: dict[str, dict[str, Any]] = {}

    for source in PRE_SOURCES:
        path = root / "data" / "pre" / f"{source}.json"
        rows = _read_json(path)
        if not isinstance(rows, list) or not rows:
            raise ManifestError(f"{path} must be a non-empty JSON list")

        licenses: Counter[str] = Counter()
        kind = SOURCE_METADATA[source]["identifier_kind"]
        identifiers: dict[tuple[str, str, str], dict[str, Any]] = {}
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ManifestError(f"{path} row {index} is not an object")
            instance_id = row.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id:
                raise ManifestError(f"{path} row {index} has no instance_id")
            if instance_id in instances:
                raise ManifestError(f"duplicate PRE instance_id {instance_id!r}")
            if row.get("source") != source and not (
                source == "synthetic" and str(row.get("source", "")).startswith("synth_")
            ):
                raise ManifestError(f"{path} row {index} has unexpected source {row.get('source')!r}")

            provenance = row.get("provenance")
            if not isinstance(provenance, dict):
                raise ManifestError(f"{path} row {index} has no provenance object")
            missing = [key for key in ("repo", "commit", "path", "license")
                       if not isinstance(provenance.get(key), str) or not provenance[key]]
            if missing:
                raise ManifestError(f"{path} row {index} has invalid provenance fields: {missing}")

            instances[instance_id] = {"source": source, "provenance": provenance}
            license_id = provenance["license"]
            licenses[license_id] += 1
            identifier_key = (provenance["repo"], provenance["commit"], license_id)
            identifier = identifiers.setdefault(
                identifier_key,
                {
                    "upstream": provenance["repo"],
                    "upstream_url": _repo_url(provenance["repo"]),
                    "identifier": provenance["commit"],
                    "identifier_kind": kind,
                    "declared_license": license_id,
                    "record_count": 0,
                    "paths": set(),
                },
            )
            identifier["record_count"] += 1
            identifier["paths"].add(provenance["path"])

        rendered_identifiers = []
        for identifier in identifiers.values():
            identifier["paths"] = sorted(identifier["paths"])
            rendered_identifiers.append(identifier)
        rendered_identifiers.sort(
            key=lambda item: (item["upstream_url"].casefold(), item["identifier"],
                              item["declared_license"])
        )

        source_sets[source] = {
            **SOURCE_METADATA[source],
            "record_count": len(rows),
            "declared_license_distribution": dict(sorted(licenses.items())),
            "upstream_identifiers": rendered_identifiers,
        }
    return source_sets, instances


def _whoandwhen_source_set(root: Path) -> dict[str, Any]:
    legacy_path = root / "data" / "llm_judge" / "legacy_run_keys.json"
    legacy = _read_json(legacy_path)
    keys = legacy.get("keys") if isinstance(legacy, dict) else None
    if not isinstance(keys, dict) or not keys:
        raise ManifestError(f"{legacy_path} must contain a non-empty keys object")
    if any(not isinstance(old, str) or not isinstance(new, str) for old, new in keys.items()):
        raise ManifestError(f"{legacy_path} contains a non-string key mapping")

    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from catchbench.corpora import CORPUS_REVISIONS

    matches = [corpus for corpus in CORPUS_REVISIONS if corpus.name == "Who&When"]
    if len(matches) != 1:
        raise ManifestError("src/catchbench/corpora.py must contain exactly one Who&When pin")
    corpus = matches[0]
    count = len(keys)
    return {
        **SOURCE_METADATA[WHO_AND_WHEN],
        "record_count": count,
        "declared_license_distribution": {"UNRESOLVED": count},
        "upstream_identifiers": [
            {
                "upstream": corpus.repo_id,
                "upstream_url": f"https://huggingface.co/datasets/{corpus.repo_id}",
                "identifier": corpus.revision,
                "identifier_kind": SOURCE_METADATA[WHO_AND_WHEN]["identifier_kind"],
                "declared_license": "UNRESOLVED",
                "record_count": count,
                "paths": ["dataset snapshot"],
            }
        ],
    }


def _contains_key(value: Any, wanted: str) -> bool:
    if isinstance(value, dict):
        return wanted in value or any(_contains_key(child, wanted) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, wanted) for child in value)
    return False


def _row_count(path: Path, value: Any) -> int:
    relative = path.as_posix()
    if relative.endswith("data/llm_judge/legacy_run_keys.json"):
        keys = value.get("keys") if isinstance(value, dict) else None
        if not isinstance(keys, dict):
            raise ManifestError(f"{path} has no keys object")
        return len(keys)
    if "/data/llm_judge/whoandwhen__" in relative:
        predictions = value.get("predictions") if isinstance(value, dict) else None
        if not isinstance(predictions, dict):
            raise ManifestError(f"{path} has no predictions object")
        provenance = value.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("n_runs") != len(predictions):
            raise ManifestError(f"{path} provenance.n_runs does not match its predictions")
        if any(not isinstance(row, dict) or "raw" not in row for row in predictions.values()):
            raise ManifestError(f"{path} has a prediction without raw model output")
        return len(predictions)
    if isinstance(value, (list, dict)):
        return len(value)
    raise ManifestError(f"{path} must contain a JSON object or list")


def _artifact_class(relative: str) -> tuple[list[str], bool, str]:
    if relative == "data/llm_judge/legacy_run_keys.json":
        return [WHO_AND_WHEN], False, "content-address mapping for the Who&When judge caches"
    if relative.startswith("data/llm_judge/whoandwhen__") and relative.endswith(".json"):
        return [WHO_AND_WHEN], True, "LLM localization predictions over Who&When traces"

    for source in PRE_SOURCES:
        if relative == f"data/pre/{source}.json":
            return [source], False, "normalized PRE benchmark records"
    for directory in ("claude_judge_votes", "gpt_judge_votes"):
        prefix = f"data/pre/{directory}/"
        if relative.startswith(prefix) and relative.endswith(".json"):
            source = Path(relative).stem
            if source not in PRE_SOURCES:
                raise ManifestError(f"unrecognized PRE source in {relative}")
            retains = directory == "gpt_judge_votes"
            return [source], retains, "cached PRE label-judge votes"
    if relative == "data/pre/llm_judge_method/llama-3.3-70b.json":
        return list(PRE_SOURCES), False, "parsed PRE baseline predictions"
    if relative == "data/pre/llm_judge_method_votes/llama-3.3-70b.json":
        return list(PRE_SOURCES), True, "cached raw PRE baseline model votes"
    if relative == "data/pre/LABEL_QUALITY.md":
        return ["crewai", "mcp", "n8n"], False, "generated PRE judge-agreement report"
    raise ManifestError(f"unclassified data artifact: {relative}")


def build_manifest(root: Path = ROOT) -> dict[str, Any]:
    source_sets, instances = _pre_source_sets(root)
    source_sets[WHO_AND_WHEN] = _whoandwhen_source_set(root)

    data_paths = sorted(
        (path for path in (root / "data").rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not data_paths:
        raise ManifestError(f"no artifacts found under {root / 'data'}")

    artifacts = []
    who_count = source_sets[WHO_AND_WHEN]["record_count"]
    for path in data_paths:
        relative = path.relative_to(root).as_posix()
        sources, prose_survives, description = _artifact_class(relative)
        if path.suffix == ".json":
            value = _read_json(path)
            count = _row_count(path, value)
        elif relative == "data/pre/LABEL_QUALITY.md":
            value = path.read_text(encoding="utf-8")
            count = sum(source_sets[source]["record_count"] for source in sources)
        else:  # guarded by _artifact_class; kept explicit so a future class cannot emit by accident
            raise ManifestError(f"no row-count rule for {relative}")

        if relative.startswith("data/llm_judge/whoandwhen__") and count != who_count:
            raise ManifestError(f"{relative} has {count} predictions; expected {who_count}")
        if prose_survives and not (_contains_key(value, "raw") or
                                   _contains_key(value, "raw_response")):
            raise ManifestError(f"{relative} is classified as retaining output but has no raw field")

        artifact_ids = None
        if relative.startswith("data/pre/llm_judge_method") or relative.startswith(
            "data/pre/gpt_judge_votes/"
        ):
            if not isinstance(value, dict):
                raise ManifestError(f"{relative} must be keyed by PRE instance id")
            artifact_ids = set(value)
        elif relative.startswith("data/pre/claude_judge_votes/"):
            if not isinstance(value, list) or any(
                not isinstance(row, dict) or not isinstance(row.get("instance_id"), str)
                for row in value
            ):
                raise ManifestError(f"{relative} has invalid vote rows")
            artifact_ids = {row["instance_id"] for row in value}
            if len(artifact_ids) != len(value):
                raise ManifestError(f"{relative} contains duplicate PRE instance ids")

        if artifact_ids is None:
            artifact_distribution = {
                source: source_sets[source]["declared_license_distribution"]
                for source in sources
            }
        else:
            invalid = sorted(
                instance_id
                for instance_id in artifact_ids
                if instance_id not in instances
                or instances[instance_id]["source"] not in sources
            )
            if invalid:
                raise ManifestError(
                    f"{relative} contains unprovenanced PRE instance ids: {invalid[:5]}"
                )
            artifact_distribution = {}
            for source in sources:
                licenses = Counter(
                    instances[instance_id]["provenance"]["license"]
                    for instance_id in artifact_ids
                    if instances[instance_id]["source"] == source
                )
                artifact_distribution[source] = dict(sorted(licenses.items()))

        artifacts.append(
            {
                "path": relative,
                "description": description,
                "sources": sources,
                "row_count": count,
                "declared_license_distribution": artifact_distribution,
                "source_prose_survives": prose_survives,
                "upstream_identifier_sets": sources,
            }
        )

    return {
        "schema_version": 1,
        "generated_by": "tools/emit_asset_manifest.py",
        "scope": "every artifact under data/",
        "license_field_note": (
            "Declared distributions reproduce artifact provenance metadata. UNRESOLVED, "
            "unverified, NOASSERTION, and Other are not licence grants. An artifact row counts "
            "the declarations of the records it actually contains, so a subset artifact does "
            "not inherit the whole source-set distribution. Its identifier list is the one "
            "named by upstream_identifier_sets."
        ),
        "identifier_field_note": (
            "identifier_kind separates what an upstream reference proves. immutable_revision "
            "is a git commit or a Hugging Face dataset revision, which addresses fixed bytes. "
            "mutable_identifier is an n8n template ID or a SWE-bench submission label, which "
            "names a location whose content can change; artifact_base_url records where those "
            "artifacts were fetched. not_applicable marks first-party authored records."
        ),
        "source_prose_field_note": (
            "True means a raw model-output field can quote or restate upstream task or trace prose. "
            "False permits normalized identifiers and capabilities but no retained prose field."
        ),
        "source_sets": {source: source_sets[source] for source in sorted(source_sets)},
        "artifacts": artifacts,
    }


def render_manifest(root: Path = ROOT) -> str:
    return json.dumps(build_manifest(root), indent=2, ensure_ascii=False) + "\n"


def check_manifest(path: Path = MANIFEST, root: Path = ROOT) -> int:
    expected = render_manifest(root)
    try:
        actual = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"ASSET MANIFEST MISSING: {path} ({exc})", file=sys.stderr)
        return 1
    if actual == expected:
        count = len(json.loads(expected)["artifacts"])
        print(f"asset manifest is current: {count} artifacts")
        return 0

    print("ASSET MANIFEST STALE", file=sys.stderr)
    diff = difflib.unified_diff(
        actual.splitlines(), expected.splitlines(), fromfile=str(path), tofile="generated",
        lineterm="",
    )
    for line in list(diff)[:80]:
        print(line, file=sys.stderr)
    print("Regenerate with: python tools/emit_asset_manifest.py --update", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="fail if the committed manifest is stale")
    action.add_argument("--update", action="store_true", help="regenerate the committed manifest")
    parser.add_argument("--manifest", type=Path, default=MANIFEST,
                        help="manifest path to write or check")
    args = parser.parse_args()

    try:
        rendered = render_manifest(ROOT)
    except ManifestError as exc:
        print(f"refusing to emit a partial asset manifest: {exc}", file=sys.stderr)
        return 2

    if args.check:
        return check_manifest(args.manifest, ROOT)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(args.manifest)
    count = len(json.loads(rendered)["artifacts"])
    print(f"wrote {args.manifest} ({count} artifacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
