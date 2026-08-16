"""Replace committed PRE task prose with scanner features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pre_spec_features import ROOT, deidentify_rows, write_json_rows

from auditablebench.pre_static_scanner import redact_spec_identifiers


CORPUS_FILES = (
    "crewai.json",
    "injecagent.json",
    "mcp.json",
    "n8n.json",
    "sweagent.json",
    "synthetic.json",
)
VOTE_FILE = ROOT / "data" / "pre" / "llm_judge_method_votes" / "llama-3.3-70b.json"


def convert_corpus(path: Path) -> tuple[int, int]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"expected a list in {path}")
    raw_count = sum("task_or_role_spec" in row for row in rows)
    converted = deidentify_rows(rows)
    write_json_rows(converted, path, sort_keys=path.name == "sweagent.json")
    return len(converted), raw_count


def redact_vote_responses(path: Path) -> int:
    votes = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for entry in votes.values():
        if not isinstance(entry, dict) or not isinstance(entry.get("raw_response"), str):
            continue
        redacted = redact_spec_identifiers(entry["raw_response"])
        if redacted != entry["raw_response"]:
            entry["raw_response"] = redacted
            changed += 1
    path.write_text(json.dumps(votes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-dir", type=Path, default=ROOT / "data" / "pre")
    args = parser.parse_args()

    for name in CORPUS_FILES:
        path = args.pre_dir / name
        rows, raw_count = convert_corpus(path)
        print(f"converted={path} rows={rows} raw_specs={raw_count}")
    if args.pre_dir.resolve() == (ROOT / "data" / "pre").resolve():
        print(f"redacted_vote_entries={redact_vote_responses(VOTE_FILE)}")


if __name__ == "__main__":
    main()
