"""Held-out LLM judge as a PRE over-privilege METHOD (a leaderboard baseline).

This is distinct from ``tools/pre_label_llm_judge.py`` + ``tools/pre_merge_judges.py``, which run
gpt-5.5 and claude-opus-4-8 to CREATE the labels. This tool runs a HELD-OUT model, one that made
no PRE label, as a leaderboard METHOD: it reads each config's needed-vs-excess judgment and writes
a prediction cache the board scores with no API call. Holding the judge out of the label set keeps
the baseline non-circular, so the method never scores against a label its own family produced.

It writes two files per model:
  - ``data/pre/llm_judge_method/<label>.json``       {instance_id: [needed_capability_names]}
    The board reads exactly this (``auditablebench.pre_baselines.LlmJudgeNeededMethod`` builds one
    method per file and treats each value as that judge's NEEDED list; excess = declared - needed).
  - ``data/pre/llm_judge_method_votes/<label>.json``  {instance_id: {raw, needed, status, ...}}
    Raw provenance plus the resume cache; never read by the board.

The backend is AWS Bedrock's Converse API, inlined here so this PRE tool stays independent of the
POST ``llm_judge`` module and its GRADE bridge. Cross-vendor held-out judges: llama-3.3-70b,
deepseek-r1, qwen3-32b, nova-micro. A held-out model made none of the PRE labels, which is what
keeps the baseline non-circular.

Keys come from the environment, so nothing secret is committed: AWS credentials or
``AWS_BEARER_TOKEN_BEDROCK``. Reruns are zero-API for instances already judged by the same model
with the same prompt.

Usage (py312 interpreter, from the repo root):
    python tools/pre_judge_method.py --model llama-3.3-70b            # full board (all data/pre/*.json)
    python tools/pre_judge_method.py --model llama-3.3-70b --limit 5  # pilot (does not shrink the cache)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC_DIR = REPO / "src"
TOOLS_DIR = REPO / "tools"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(TOOLS_DIR))

# Reuse the exact needed-vs-excess prompt and parser the label makers used, so the method judge
# answers the byte-identical question and the comparison is fair. Neither symbol imports the GRADE
# bridge (they are pure string helpers).
from pre_label_llm_judge import build_prompt, parse_needed  # noqa: E402

PRE_DIR = REPO / "data" / "pre"
OUT_DIR = PRE_DIR / "llm_judge_method"
VOTES_DIR = PRE_DIR / "llm_judge_method_votes"

# Short board label -> Bedrock model id. All are cross-vendor held-out judges (they made no PRE
# label). Add a model here to score it as its own board row via pre_baselines.LlmJudgeNeededMethod.
BEDROCK_MODELS = {
    "llama-3.3-70b": "us.meta.llama3-3-70b-instruct-v1:0",
    "deepseek-r1": "us.deepseek.r1-v1:0",
    "qwen3-32b": "qwen.qwen3-32b-v1:0",
    "nova-micro": "amazon.nova-micro-v1:0",
}

_USAGE = {"in": 0, "out": 0}


def _write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def bedrock_complete(model_id: str, max_tokens: int = 4000):
    import boto3

    client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    def complete(prompt: str) -> str:
        resp = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
        )
        usage = resp.get("usage", {})
        _USAGE["in"] += usage.get("inputTokens", 0)
        _USAGE["out"] += usage.get("outputTokens", 0)
        # Reasoning models (DeepSeek-R1, Qwen3) emit a reasoningContent block first; keep the text.
        parts = resp["output"]["message"]["content"]
        return " ".join(p["text"] for p in parts if "text" in p).strip()

    return complete


def make_complete(label: str):
    if label in BEDROCK_MODELS:
        return bedrock_complete(BEDROCK_MODELS[label])
    raise SystemExit(f"unknown model label {label!r}; known: {sorted(BEDROCK_MODELS)}")


def load_board_instances() -> list[dict]:
    """Every committed board config (data/pre/*.json). Instance ids must be globally unique."""
    seen: dict[str, dict] = {}
    for path in sorted(PRE_DIR.glob("*.json")):
        for row in json.load(open(path, encoding="utf-8")):
            iid = row["instance_id"]
            if iid in seen:
                raise SystemExit(f"duplicate instance_id {iid!r} in data/pre/*.json (at {path.name})")
            seen[iid] = {
                "instance_id": iid,
                "source": row["source"],
                "task_or_role_spec": row["task_or_role_spec"],
                "declared_capabilities": row["declared_capabilities"],
            }
    return list(seen.values())


def _judge_one(row: dict, complete, model: str) -> tuple[str, dict]:
    prompt = build_prompt(row)
    declared = {c["name"] for c in row["declared_capabilities"]}
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    raw, needed, error = "", None, "empty response"
    for _ in (1, 2):  # two attempts, matching the label tool
        try:
            raw = complete(prompt)
        except Exception as exc:  # noqa: BLE001  one bad call -> a miss, re-runnable on resume
            error = f"{type(exc).__name__}: {str(exc)[:120]}"
            continue
        needed, error = parse_needed(raw, declared)
        if needed is not None and error is None:
            break
    entry = {
        "model": model,
        "model_id": BEDROCK_MODELS.get(model, model),  # resolved backend id, so a remap re-judges
        "prompt_sha256": prompt_sha,
        "raw_response": raw,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if needed is not None and error is None:
        entry["needed"] = needed
        entry["status"] = "ok"
    else:
        entry["status"] = "malformed"
        entry["error"] = error or "empty response"
    return row["instance_id"], entry


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a held-out LLM judge as a PRE over-privilege method.")
    ap.add_argument("--model", default="llama-3.3-70b",
                    help="board label; known: " + ", ".join(sorted(BEDROCK_MODELS)))
    ap.add_argument("--limit", type=int, default=None, help="cap configs (pilot); omit for the full board")
    ap.add_argument("--workers", type=int, default=8, help="concurrent API calls")
    args = ap.parse_args()

    full_instances = load_board_instances()  # always the full board; --limit only caps what is judged
    to_judge = full_instances[: args.limit] if args.limit is not None else full_instances
    resolved_id = BEDROCK_MODELS.get(args.model, args.model)

    votes_path = VOTES_DIR / f"{args.model}.json"
    out_path = OUT_DIR / f"{args.model}.json"
    votes: dict[str, dict] = {}
    if votes_path.exists():
        votes = json.load(open(votes_path, encoding="utf-8"))

    current_sha = {
        r["instance_id"]: hashlib.sha256(build_prompt(r).encode("utf-8")).hexdigest()
        for r in full_instances
    }

    # A vote is fresh only if it is ok, from THIS model (short label AND resolved backend id, so a
    # remapped BEDROCK_MODELS re-judges), carries a needed list, and was made with the CURRENT prompt
    # for that instance. Older caches without model_id fall back to this label's resolved id. The one
    # predicate governs both resume (skip fresh) and publication (emit only fresh), so a pilot after a
    # prompt change publishes only the re-judged rows instead of stale-prompt predictions, and a
    # steady-state pilot still keeps every entry.
    def _fresh(entry: dict, iid: str) -> bool:
        return (
            entry.get("status") == "ok"
            and entry.get("model") == args.model
            and entry.get("model_id", resolved_id) == resolved_id
            and isinstance(entry.get("needed"), list)
            and entry.get("prompt_sha256") == current_sha.get(iid)
        )

    pending = [r for r in to_judge if not _fresh(votes.get(r["instance_id"], {}), r["instance_id"])]
    print(f"model={args.model} board={len(full_instances)} judging={len(to_judge)} "
          f"fresh={len(to_judge) - len(pending)} pending={len(pending)}")

    if pending:
        complete = make_complete(args.model)
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_judge_one, r, complete, args.model): r for r in pending}
            for fut in concurrent.futures.as_completed(futures):
                iid, entry = fut.result()
                votes[iid] = entry
                done += 1
                if done % 25 == 0 or done == len(pending):
                    ok = sum(1 for v in votes.values() if v.get("status") == "ok")
                    print(f"  {done}/{len(pending)} done, ok={ok}", flush=True)
                    _write_json_atomic(votes_path, votes)  # checkpoint for resume
        _write_json_atomic(votes_path, votes)

    # Publish ONLY votes fresh for the current prompt over the FULL board (see _fresh): a pilot never
    # publishes stale-prompt predictions, and when nothing changed it never shrinks the cache either.
    method_cache = {iid: votes[iid]["needed"] for iid in current_sha if _fresh(votes.get(iid, {}), iid)}
    _write_json_atomic(out_path, method_cache)

    print(f"\nwrote {out_path.relative_to(REPO)}  "
          f"({len(method_cache)}/{len(full_instances)} board configs fresh)")
    print(f"wrote {votes_path.relative_to(REPO)}  (raw provenance + resume)")
    print(f"tokens(in/out)={_USAGE['in']}/{_USAGE['out']}")


if __name__ == "__main__":
    main()
