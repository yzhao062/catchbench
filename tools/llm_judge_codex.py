"""Regenerate the LLM-judge prediction cache using GPT-5.5 through the Codex CLI as the backend.

This is the opt-in generation path for the LLM-judge localization baselines (see
``auditablebench.llm_judge``). It is the ONLY place live LLM calls happen; the board scores from the
committed JSON cache this writes, so running the benchmark never needs an API key or a Codex
subscription. We drive GPT-5.5 with ``codex exec`` (no OpenAI API key is configured on this machine,
and the Codex subscription is the available GPT-5.5 access), capturing only the model's final message
via ``-o``. Each prompt is self-contained, so the sandbox is read-only and no tools are needed.

Usage (from the repo root, py312 interpreter):
    python tools/llm_judge_codex.py all_at_once --limit 3      # pilot
    python tools/llm_judge_codex.py all_at_once                # full 126-run board

Reproducibility: anyone with an OpenAI API key can swap ``_codex_complete`` for a one-line API call
and regenerate the same cache; the prompt templates and parsing live in ``llm_judge`` and are backend
agnostic. The committed cache records model and prompt provenance.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from auditablebench import llm_judge as lj  # noqa: E402

MODEL = "gpt-5.5"  # the Codex-configured model; recorded in the cache provenance
_NEUTRAL_CWD = tempfile.gettempdir()
_EFFORT = "high"  # reasoning effort override: the config default (xhigh) times out per call; high is
# ~13s/call and gives the judge a strong, fair shot. Set by --effort.


def _resolve_codex() -> list:
    """Find the codex launcher and return the argv prefix. On Windows npm installs a ``codex.cmd``
    shim, which CreateProcess cannot exec directly, so it is routed through ``cmd /c``."""
    for cand in ("codex.cmd", "codex.exe", "codex"):
        path = shutil.which(cand)
        if path:
            return ["cmd", "/c", path] if path.lower().endswith((".cmd", ".bat")) else [path]
    fallback = os.path.join(os.environ.get("APPDATA", ""), "npm", "codex.cmd")
    return ["cmd", "/c", fallback]


_CODEX = _resolve_codex()


def _codex_complete(prompt: str, timeout: int = 240) -> str:
    """Send one self-contained prompt to GPT-5.5 via ``codex exec`` and return its final message.
    Read-only sandbox, ephemeral session, prompt piped on stdin, answer captured with ``-o``. Returns
    an empty string on failure, which the parser treats as a miss for that run (re-runnable later)."""
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False, encoding="utf-8") as ans:
        ans_path = ans.name
    try:
        proc = subprocess.run(
            _CODEX + ["exec", "--ephemeral", "-s", "read-only", "--skip-git-repo-check",
                      "--color", "never", "-c", f"model_reasoning_effort={_EFFORT}",
                      "-C", _NEUTRAL_CWD, "-o", ans_path, "-"],
            input=prompt, text=True, capture_output=True, timeout=timeout,
        )
        if proc.returncode != 0:
            sys.stderr.write(f"[codex non-zero {proc.returncode}] {proc.stderr[-300:]}\n")
        with open(ans_path, encoding="utf-8") as fh:
            return fh.read().strip()
    except subprocess.TimeoutExpired:
        sys.stderr.write("[codex timeout]\n")
        return ""
    finally:
        try:
            os.remove(ans_path)
        except OSError:
            pass


def main() -> None:
    global _EFFORT
    ap = argparse.ArgumentParser(description="Regenerate LLM-judge cache via Codex GPT-5.5.")
    ap.add_argument("method", choices=["all_at_once", "step_by_step", "binary_search"])
    ap.add_argument("--limit", type=int, default=None, help="cap runs (pilot); omit for full board")
    ap.add_argument("--model", default=MODEL, help="model id recorded in provenance")
    ap.add_argument("--effort", default=_EFFORT, choices=["low", "medium", "high", "xhigh"],
                    help="reasoning effort (xhigh times out per call; high is the fast strong default)")
    args = ap.parse_args()
    _EFFORT = args.effort

    path = lj.regenerate_cache(args.method, args.model, _codex_complete, limit=args.limit)
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
