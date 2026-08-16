"""Run the LLM-judge panel: one cached prediction set per model, the reproducible multi-model
"just ask an LLM" control for POST localization.

Two backends, both standard APIs so a second party can regenerate every row:
  - NAIRR gateway (OpenAI-compatible): frontier judges (gpt-5.5, claude-opus-4.8, gemini, gpt-5.4).
  - AWS Bedrock (Converse API): open-weights + small proprietary judges (Llama-3.3-70B, Qwen3-32B,
    DeepSeek-R1, Mistral, Gemma, OpenAI gpt-oss, and Amazon Nova).

Gateway configuration comes from the environment, so no endpoint or secret is committed:
  - NAIRR_GATEWAY_URL  (required; HTTPS except for localhost or 127.0.0.1 development servers)
  - NAIRR_GATEWAY_KEY  (a least-privilege key limited to the required judge models)
  - AWS_BEARER_TOKEN_BEDROCK  (or standard AWS credentials)

The prompt templates, parsing, scoring, caching, resume, and checkpointing all live in
``auditablebench.llm_judge`` and are shared across every backend. Calls run concurrently
(``--workers``); the committed cache records model and prompt provenance. The board scores from the
cache, so generation is paid once and the benchmark itself stays zero-API and deterministic.

Usage (py312 interpreter, from the repo root, with the two env keys set):
    python tools/run_llm_judge_panel.py all_at_once --models gpt-5.5 llama-3.3-70b --limit 5   # pilot
    python tools/run_llm_judge_panel.py all_at_once --models gpt-5.5 claude-opus-4.8 \
        llama-3.3-70b qwen3-32b deepseek-r1 gemini                                              # full panel
"""
from __future__ import annotations

import argparse
import os
import sys
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from auditablebench import llm_judge as lj  # noqa: E402

# Short board label -> provider model id.
GATEWAY_MODELS = {
    "gpt-5.5": "gpt-5.5",
    "claude-opus-4.8": "claude-opus-4.8",
    "gemini": "gemini",
    "gpt-5.4": "gpt-5.4",
}
BEDROCK_MODELS = {
    "llama-3.3-70b": "us.meta.llama3-3-70b-instruct-v1:0",
    "qwen3-32b": "qwen.qwen3-32b-v1:0",
    "deepseek-r1": "us.deepseek.r1-v1:0",
    "mistral-small": "mistral.mistral-small-2402-v1:0",
    "gemma-3-12b": "google.gemma-3-12b-it",
    "gpt-oss-20b": "openai.gpt-oss-20b-1:0",
    "nova-micro": "amazon.nova-micro-v1:0",
}

# Token accounting, summed across calls, so the run can report measured usage (cost transparency).
_USAGE = {"in": 0, "out": 0}


def _gateway_url() -> str:
    url = os.environ.get("NAIRR_GATEWAY_URL", "").strip()
    if not url:
        raise SystemExit(
            "set NAIRR_GATEWAY_URL to the gateway API base URL "
            "(HTTPS required; HTTP is allowed only for localhost or 127.0.0.1)"
        )
    parsed = urlparse(url)
    if not parsed.hostname:
        raise SystemExit("NAIRR_GATEWAY_URL must be an absolute URL with a host")
    if parsed.scheme == "https":
        return url
    if parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}:
        return url
    raise SystemExit(
        "NAIRR_GATEWAY_URL must use HTTPS; HTTP is allowed only for localhost or "
        "127.0.0.1 development servers"
    )


def gateway_complete(model_id: str, max_tokens: int = 8000, timeout: int = 240):
    gateway_url = _gateway_url()
    key = os.environ.get("NAIRR_GATEWAY_KEY")
    if not key:
        raise SystemExit(
            "set NAIRR_GATEWAY_KEY to a least-privilege key limited to the required judge models"
        )

    from openai import OpenAI

    client = OpenAI(base_url=gateway_url, api_key=key)

    def complete(prompt: str) -> str:
        try:
            # No explicit temperature: some gateway models (Azure GPT-5.5) reject temperature=0 and
            # only accept their default. The committed cache, not a sampling setting, is what makes
            # the board reproducible, so the judge runs at the model's default temperature.
            resp = client.chat.completions.create(
                model=model_id, messages=[{"role": "user", "content": prompt}],
                timeout=timeout, max_tokens=max_tokens,
            )
            usage = getattr(resp, "usage", None)
            if usage:
                _USAGE["in"] += getattr(usage, "prompt_tokens", 0) or 0
                _USAGE["out"] += getattr(usage, "completion_tokens", 0) or 0
            return (resp.choices[0].message.content or "").strip()
        except Exception as err:  # noqa: BLE001  one bad call -> a miss, re-runnable on resume
            sys.stderr.write(f"[gateway {model_id} error] {type(err).__name__}: {str(err)[:160]}\n")
            return ""

    return complete


def bedrock_complete(model_id: str, max_tokens: int = 8000):
    import boto3

    client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    def complete(prompt: str) -> str:
        try:
            resp = client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
            )
            usage = resp.get("usage", {})
            _USAGE["in"] += usage.get("inputTokens", 0)
            _USAGE["out"] += usage.get("outputTokens", 0)
            # Reasoning models (DeepSeek-R1, Qwen3) emit a reasoningContent block first; keep text.
            parts = resp["output"]["message"]["content"]
            texts = [p["text"] for p in parts if "text" in p]
            return " ".join(texts).strip()
        except Exception as err:  # noqa: BLE001
            sys.stderr.write(f"[bedrock {model_id} error] {type(err).__name__}: {str(err)[:160]}\n")
            return ""

    return complete


def make_complete(label: str):
    if label in GATEWAY_MODELS:
        return gateway_complete(GATEWAY_MODELS[label])
    if label in BEDROCK_MODELS:
        return bedrock_complete(BEDROCK_MODELS[label])
    raise SystemExit(f"unknown model label {label!r}; known: "
                     f"{sorted(GATEWAY_MODELS) + sorted(BEDROCK_MODELS)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the LLM-judge panel via NAIRR gateway + Bedrock.")
    ap.add_argument("method", choices=["all_at_once", "step_by_step", "binary_search"])
    ap.add_argument("--models", nargs="+", required=True, help="board labels, e.g. gpt-5.5 llama-3.3-70b")
    ap.add_argument("--limit", type=int, default=None, help="cap runs (pilot); omit for full board")
    ap.add_argument("--workers", type=int, default=8, help="concurrent API calls per model")
    args = ap.parse_args()

    completes = {label: make_complete(label) for label in args.models}
    runs = lj.load_judge_runs()
    if args.limit:
        runs = runs[: args.limit]
    for label in args.models:
        _USAGE["in"] = _USAGE["out"] = 0
        complete = completes[label]
        path = lj.regenerate_cache(args.method, label, complete, limit=args.limit,
                                   max_workers=args.workers)
        preds = lj.load_cache(args.method, label)["predictions"]
        hits = sum(1 for r in runs if preds.get(lj.run_key(r), {}).get("top") == r["mistake"])
        parsed = sum(1 for r in runs if preds.get(lj.run_key(r), {}).get("top") is not None)
        print(f"\n== {label} :: {args.method} == wrote {os.path.basename(path)}")
        print(f"   runs={len(runs)} parsed={parsed} top1_hits={hits} top1={hits / len(runs):.3f} "
              f"tokens(in/out)={_USAGE['in']}/{_USAGE['out']}\n")


if __name__ == "__main__":
    main()
