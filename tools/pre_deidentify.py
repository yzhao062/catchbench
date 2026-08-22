"""Replace committed PRE task prose with scanner features."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pre_spec_features import ROOT, deidentify_rows, write_json_rows

from catchbench.pre_static_scanner import redact_spec_identifiers


CORPUS_FILES = (
    "crewai.json",
    "injecagent.json",
    "mcp.json",
    "n8n.json",
    "sweagent.json",
    "synthetic.json",
)
VOTE_FILE = ROOT / "data" / "pre" / "llm_judge_method_votes" / "llama-3.3-70b.json"

# Every committed model output keyed by a PRE instance id, which is the set the copyleft pass below
# has to cover. The held-out judge is one of them; the label-making judges are the others, and they
# were prompted with the same prose. Naming the directories rather than one file means a re-judge
# that lands a new cache is covered on the next run instead of being missed the way the label-making
# votes were missed the first time this pass was written.
PRE_VOTE_DIRS = (
    ROOT / "data" / "pre" / "llm_judge_method_votes",
    ROOT / "data" / "pre" / "gpt_judge_votes",
)


def pre_vote_files() -> list[Path]:
    return sorted(p for d in PRE_VOTE_DIRS if d.is_dir() for p in d.glob("*.json"))

# The held-out judge was prompted with each config's task and role prose, and its cached reply
# sometimes restates that prose. For most sources that is unremarkable, but a copyleft licence
# attaches to the prose itself, so a reply that reproduces an upstream goal field redistributes
# upstream text no matter how the surrounding record was de-identified. Records declaring one of
# these licences get a second pass that removes the reproduced spans. The set is read against each
# record's declared licence rather than against a list of instance ids, so a source added later is
# covered without a code change.
UPSTREAM_REDACT_LICENSES = frozenset({"GPL-3.0"})

# Word runs at or above this length are treated as reproduced upstream text rather than as
# coincidental overlap. Four is short enough to catch a restated goal clause and long enough that
# ordinary English ("the implementation of the") survives on its own terms; the shortest run this
# actually removed from the committed cache is five words.
UPSTREAM_MIN_RUN_WORDS = 4

UPSTREAM_MARKER = "[upstream text removed: see THIRD_PARTY_LICENSES.md]"

# Not under data/pre/: that directory is glob-loaded by catchbench.pre._load_data_dir, which
# reads every *.json there as a list of records. This manifest belongs with the licence texts
# it explains anyway.
REDACTION_MANIFEST = ROOT / "third_party" / "UPSTREAM_REDACTIONS.json"

_WORD = re.compile(r"[A-Za-z']+")


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


def upstream_slug(repo: str, commit: str, path: str) -> str:
    """The local filename an operator fetches an upstream source into."""
    return "%s@%s__%s" % (repo.replace("/", "_"), commit[:8], path.replace("/", "_"))


def _tokens(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0).lower(), m.start(), m.end()) for m in _WORD.finditer(text)]


def redact_upstream_spans(
    text: str, upstream: str, min_run: int = UPSTREAM_MIN_RUN_WORDS
) -> tuple[str, list[str]]:
    """Blank every maximal word run of `text` that also occurs in `upstream`.

    Matching is on lowercased word sequences, so it ignores punctuation and case and catches a
    clause that was reflowed or re-inflected around its edges. Runs are grown greedily from the
    left and only the maximal ones are removed, so one reproduced sentence yields one marker
    rather than a marker per overlapping window.
    """
    hay = _tokens(text)
    src = [t for t, _, _ in _tokens(upstream)]
    src_runs = {
        " ".join(src[i:i + n])
        for n in range(min_run, len(src) + 1)
        for i in range(len(src) - n + 1)
    } if src else set()
    if not src_runs:
        return text, []

    spans: list[tuple[int, int, str]] = []
    i = 0
    while i < len(hay):
        best = 0
        n = min_run
        while i + n <= len(hay) and " ".join(t for t, _, _ in hay[i:i + n]) in src_runs:
            best = n
            n += 1
        if best:
            start, end = hay[i][1], hay[i + best - 1][2]
            spans.append((start, end, text[start:end]))
            i += best
        else:
            i += 1
    if not spans:
        return text, []

    out, cursor, removed = [], 0, []
    for start, end, quoted in spans:
        out.append(text[cursor:start])
        out.append(UPSTREAM_MARKER)
        removed.append(quoted)
        cursor = end
    out.append(text[cursor:])
    return "".join(out), removed


def _licensed_records(pre_dir: Path) -> list[dict]:
    """Every PRE record whose declared licence calls for the upstream pass."""
    out = []
    for name in CORPUS_FILES:
        path = pre_dir / name
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            continue
        for row in rows:
            prov = row.get("provenance") or {}
            if prov.get("license") in UPSTREAM_REDACT_LICENSES:
                out.append({"instance_id": row.get("instance_id"), "provenance": prov})
    return out


def redact_licensed_upstream(pre_dir: Path, upstream_dir: Path, vote_files: list[Path]) -> dict:
    """Remove reproduced upstream text from the cached judge replies of licensed records.

    `upstream_dir` holds the upstream sources an operator fetched at the pinned commits, named by
    `upstream_slug`. Those files are never committed: carrying them would redistribute in full the
    text this pass exists to remove.

    Every cache in `vote_files` is walked, and a record with no reproduced run is still recorded
    with zero spans, so the manifest says which caches were examined rather than only which ones
    happened to need work.
    """
    records, entries, missing = _licensed_records(pre_dir), [], []
    previous = {}
    if REDACTION_MANIFEST.exists():
        for e in json.loads(REDACTION_MANIFEST.read_text(encoding="utf-8")).get("records", []):
            previous[(e.get("cache"), e.get("instance_id"))] = e
    for vote_file in vote_files:
        votes = json.loads(vote_file.read_text(encoding="utf-8"))
        touched = False
        for rec in records:
            prov = rec["provenance"]
            source = upstream_dir / upstream_slug(prov["repo"], prov["commit"], prov["path"])
            if not source.exists():
                missing.append(source.name)
                continue
            entry = votes.get(rec["instance_id"])
            if not isinstance(entry, dict) or not isinstance(entry.get("raw_response"), str):
                continue
            redacted, removed = redact_upstream_spans(
                entry["raw_response"], source.read_text(encoding="utf-8", errors="replace")
            )
            if removed:
                entry["raw_response"] = redacted
                touched = True
            cache_rel = vote_file.relative_to(ROOT).as_posix()
            prior = previous.get((cache_rel, rec["instance_id"]), {})
            entries.append({
                "cache": cache_rel,
                "instance_id": rec["instance_id"],
                "license": prov["license"],
                "source": "%s@%s/%s" % (prov["repo"], prov["commit"][:8], prov["path"]),
                # The count of markers now in the file, not the count this run removed. A second
                # run finds nothing left to remove, and a delta would then record zero and
                # contradict the file it describes.
                "markers": entry["raw_response"].count(UPSTREAM_MARKER),
                "longest_span_words": max(
                    [len(_WORD.findall(s)) for s in removed] + [prior.get("longest_span_words", 0)]
                ),
            })
        if touched:
            vote_file.write_text(json.dumps(votes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if missing:
        raise SystemExit(
            "fetch these upstream sources into %s first: %s" % (upstream_dir, ", ".join(sorted(set(missing))))
        )
    entries.sort(key=lambda e: (e["cache"], e["instance_id"]))
    return {
        "licenses": sorted(UPSTREAM_REDACT_LICENSES),
        "min_run_words": UPSTREAM_MIN_RUN_WORDS,
        "marker": UPSTREAM_MARKER,
        "caches_examined": [p.relative_to(ROOT).as_posix() for p in vote_files],
        "records": entries,
    }


def check_licensed_upstream(pre_dir: Path, vote_files: list[Path], manifest_path: Path) -> list[str]:
    """Offline invariants: the manifest covers every licensed record and the markers survive.

    Whether some other upstream sentence still remains is an online question, because settling it
    needs the upstream text this repository deliberately does not carry. What is checkable here is
    that the recorded pass covered every cache and every licensed record, and was not later undone.
    """
    problems: list[str] = []
    if not manifest_path.exists():
        return ["%s is missing; run pre_deidentify.py --upstream-dir" % manifest_path.name]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = {(e["cache"], e["instance_id"]): e for e in manifest.get("records", [])}
    examined = set(manifest.get("caches_examined", []))
    licensed = _licensed_records(pre_dir)
    live = {r["instance_id"] for r in licensed}

    for vote_file in vote_files:
        rel = vote_file.relative_to(ROOT).as_posix()
        if rel not in examined:
            problems.append("%s holds PRE model output and the redaction manifest never examined it" % rel)
            continue
        votes = json.loads(vote_file.read_text(encoding="utf-8"))
        for rec in licensed:
            iid = rec["instance_id"]
            raw = (votes.get(iid) or {}).get("raw_response")
            if not isinstance(raw, str):
                continue
            entry = recorded.get((rel, iid))
            if entry is None:
                problems.append("%s in %s declares %s and the redaction manifest does not cover it"
                                % (iid, rel, rec["provenance"].get("license")))
                continue
            if raw.count(UPSTREAM_MARKER) != entry["markers"]:
                problems.append("%s in %s should carry %d redaction marker(s) and carries %d"
                                % (iid, rel, entry["markers"], raw.count(UPSTREAM_MARKER)))
    for cache, iid in recorded:
        if iid not in live:
            problems.append("the redaction manifest covers %s in %s, which no record declares" % (iid, cache))
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-dir", type=Path, default=ROOT / "data" / "pre")
    parser.add_argument("--upstream-dir", type=Path, default=None,
                        help="directory of upstream sources fetched at their pinned commits; "
                             "runs the copyleft pass and rewrites the redaction manifest")
    parser.add_argument("--check-upstream", action="store_true",
                        help="verify the recorded copyleft redaction offline, and change nothing")
    args = parser.parse_args()

    if args.check_upstream:
        problems = check_licensed_upstream(args.pre_dir, pre_vote_files(), REDACTION_MANIFEST)
        for line in problems:
            print(line)
        raise SystemExit(1 if problems else 0)

    if args.upstream_dir is not None:
        manifest = redact_licensed_upstream(args.pre_dir, args.upstream_dir, pre_vote_files())
        REDACTION_MANIFEST.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        total = sum(r["markers"] for r in manifest["records"])
        longest = max((r["longest_span_words"] for r in manifest["records"]), default=0)
        print("upstream_redaction records=%d markers=%d longest_span_words=%d"
              % (len(manifest["records"]), total, longest))
        return

    for name in CORPUS_FILES:
        path = args.pre_dir / name
        rows, raw_count = convert_corpus(path)
        print(f"converted={path} rows={rows} raw_specs={raw_count}")
    if args.pre_dir.resolve() == (ROOT / "data" / "pre").resolve():
        print(f"redacted_vote_entries={redact_vote_responses(VOTE_FILE)}")


if __name__ == "__main__":
    main()
