"""Regenerate the LLM-judge cache using an OpenAI-compatible chat endpoint as the backend.

This is the reproducible-anywhere generation path. Point it at any OpenAI-compatible ``/v1`` server:
a local open-weights model on DGX Spark (vLLM or Ollama), LM Studio, or the OpenAI API itself. Unlike
the Codex CLI backend, the served model and endpoint are fully specified, so a second party can rerun
the exact baseline without a Codex subscription. The prompt templates, parsing, and scoring live in
``auditablebench.llm_judge`` and are shared with every backend.

Serve a local judge on DGX Spark, then point this at it, for example:
    # on DGX Spark (Ollama):     ollama serve  &&  ollama run qwen2.5:72b
    #   endpoint -> http://<dgx-host>:11434/v1 , model "qwen2.5:72b"
    # on DGX Spark (vLLM):       python -m vllm.entrypoints.openai.api_server \
    #                              --model Qwen/Qwen2.5-72B-Instruct --port 8000
    #   endpoint -> http://<dgx-host>:8000/v1 , model "Qwen/Qwen2.5-72B-Instruct"

Usage (py312 interpreter, from the repo root):
    python tools/llm_judge_openai_compatible.py all_at_once \
        --base-url http://<dgx-host>:11434/v1 --model qwen2.5:72b --limit 3   # pilot
    python tools/llm_judge_openai_compatible.py all_at_once \
        --base-url http://<dgx-host>:11434/v1 --model qwen2.5:72b              # full board
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from auditablebench import llm_judge as lj  # noqa: E402


def make_complete(base_url: str, model: str, api_key: str, max_tokens: int = 512, timeout: int = 240):
    """Build a ``complete(prompt) -> str`` that POSTs one chat completion to an OpenAI-compatible
    endpoint at temperature 0 (deterministic given the server). Returns an empty string on any error,
    which the parser treats as a miss for that run (re-runnable later)."""
    url = base_url.rstrip("/") + "/chat/completions"

    def complete(prompt: str) -> str:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key or 'EMPTY'}",
        })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return (payload["choices"][0]["message"]["content"] or "").strip()
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError, TimeoutError) as exc:
            sys.stderr.write(f"[endpoint error] {type(exc).__name__}: {str(exc)[:200]}\n")
            return ""

    return complete


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate LLM-judge cache via an OpenAI-compatible API.")
    ap.add_argument("method", choices=["all_at_once", "step_by_step", "binary_search"])
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible base, e.g. http://host:11434/v1")
    ap.add_argument("--model", required=True, help="served model id, recorded in provenance")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", help="env var holding the key (optional)")
    ap.add_argument("--limit", type=int, default=None, help="cap runs (pilot); omit for full board")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    complete = make_complete(args.base_url, args.model, api_key)
    path = lj.regenerate_cache(args.method, args.model, complete, limit=args.limit)
    cache = lj.load_cache(args.method, args.model)
    preds = cache["predictions"]
    runs = lj.load_judge_runs()
    if args.limit:
        runs = runs[: args.limit]
    hits = sum(1 for r in runs if preds.get(lj.run_key(r), {}).get("top") == r["mistake"])
    parsed = sum(1 for r in runs if preds.get(lj.run_key(r), {}).get("top") is not None)
    print(f"\nwrote {path}")
    print(f"runs={len(runs)} parsed={parsed} top1_hits={hits} top1={hits / len(runs):.3f}")


if __name__ == "__main__":
    main()
