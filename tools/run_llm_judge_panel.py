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
``catchbench.llm_judge`` and are shared across every backend. Calls run concurrently
(``--workers``); the committed cache records model and prompt provenance. The board scores from the
cache, so generation is paid once and the benchmark itself stays zero-API and deterministic.

Usage (py312 interpreter, from the repo root, with the two env keys set):
    python tools/run_llm_judge_panel.py all_at_once --models gpt-5.5 llama-3.3-70b --limit 5   # pilot
    python tools/run_llm_judge_panel.py all_at_once --models gpt-5.5 claude-opus-4.8 \
        llama-3.3-70b qwen3-32b deepseek-r1 gemini                                              # full panel
"""
from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from catchbench import llm_judge as lj  # noqa: E402

# Short board label -> provider model id.
#
# The first four gateway labels and the seven Bedrock labels are the published panel. Never name one
# of them in --models again: _flush at llm_judge.py:518 runs even when there is nothing to compute,
# and a no-op pass over the gpt-5.5 cache rewrites 127 of its 1869 lines while changing none of its
# 126 predictions. 126 of those lines are run keys, because every committed cache uses the legacy
# order-prefixed form and load_cache normalises to content addresses on read. The rule and the
# measurement are in research/catchbench-m4-declaration-v3.md in the notes repository.
#
# The last five are the 2026-09-06 addendum, added under that declaration and signed off in
# research/catchbench-round16-declaration-signoff.md. They are a separate exploratory family and are
# never merged into the eight-judge band.
GATEWAY_MODELS = {
    "gpt-5.5": "gpt-5.5",
    "claude-opus-4.8": "claude-opus-4.8",
    "gemini": "gemini",
    "gpt-5.4": "gpt-5.4",
    # Addendum, declared and NOT run. Erratum 2 of the declaration records why: no subscription CLI
    # serves it, its only route is this gateway, and _gateway_url refuses plain HTTP to a remote
    # host, so it needs an SSH forward that was deliberately not set up. Its claim id is retired
    # unused. The entry stays so the label resolves rather than looking like an oversight; running it
    # would put the addendum back to five models and eight contrasts and is not a free choice.
    "gpt-5.6-sol": "gpt-5.6-sol",
}
BEDROCK_MODELS = {
    "llama-3.3-70b": "us.meta.llama3-3-70b-instruct-v1:0",
    "qwen3-32b": "qwen.qwen3-32b-v1:0",
    "deepseek-r1": "us.deepseek.r1-v1:0",
    "mistral-small": "mistral.mistral-small-2402-v1:0",
    "gemma-3-12b": "google.gemma-3-12b-it",
    "gpt-oss-20b": "openai.gpt-oss-20b-1:0",
    "nova-micro": "amazon.nova-micro-v1:0",
    # Addendum. Two earlier 70B Llama generations, so the series holds scale and architecture fixed
    # against the panel's existing llama-3.3-70b and moves only the generation. Both ids smoke-tested
    # against us-east-1 on 2026-09-06. Note that llama-3-70b and llama-3.3-70b are different labels
    # and neither sanitises into the other.
    "llama-3-70b": "meta.llama3-70b-instruct-v1:0",
    "llama-3.1-70b": "us.meta.llama3-1-70b-instruct-v1:0",
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


# Output-token ceilings for the models that have one below the 8000 default. Bedrock rejects the
# whole call with ValidationException when maxTokens exceeds the model's limit, so this is a hard
# floor rather than a tuning knob: Llama 3 70B caps at 2048 and returns nothing at all above it.
#
# Only models that actually need a cap appear here, so every model the published panel used keeps the
# 8000 it ran with and its configuration is unchanged. Judge replies run 40 to 560 output tokens in
# practice, so 2048 is not a binding constraint on the answer.
_MAX_OUTPUT_TOKENS = {
    "meta.llama3-70b-instruct-v1:0": 2048,
}


def bedrock_complete(model_id: str, max_tokens: int = 8000):
    import boto3

    max_tokens = min(max_tokens, _MAX_OUTPUT_TOKENS.get(model_id, max_tokens))

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


# Subscription-CLI backends. This is the channel the published gpt-5.5 reference was generated on:
# tools/llm_judge_codex.py drives `codex exec` because no OpenAI API key is configured here. The two
# addendum models below reach their vendor the same way, which needs no key and puts nothing on the
# network in plaintext. What it does not give is a pinned checkpoint, because a CLI serves whatever
# its subscription currently ships. Neither does the gateway for these families, whose Azure
# deployments carry versionUpgradeOption OnceNewDefaultVersionAvailable, so this trades one unpinned
# route for another rather than giving up a pin. Erratum 1 of the declaration records that.
#
# label -> (which CLI, model argument, extra argv)
CLI_MODELS = {
    "claude-opus-5": ("claude", "claude-opus-5"),
    "gpt-6-astra": ("codex", "gpt-6-astra"),
}

# The judge prompt is self-contained and states its own answer format, so the call should be as close
# to a plain API call as the CLI allows. Left at its default, `claude -p` would run with the coding
# assistant's system prompt, this repository's CLAUDE.md, and the user's own settings, none of which
# belong in a judgement about someone else's trace. Every flag below removes one of those.
_CLAUDE_SYSTEM_PROMPT = "You are a careful analyst. Answer exactly in the format the user asks for."
_CLAUDE_FLAGS = ["--system-prompt", _CLAUDE_SYSTEM_PROMPT,
                 "--restricted", "--strict-mcp-config", "--no-session-persistence"]

# codex exec times out per call at the configured xhigh; high is the setting llm_judge_codex.py
# settled on for the published reference, so the addendum matches it rather than picking its own.
_CODEX_EFFORT = "high"
_NEUTRAL_CWD = tempfile.gettempdir()


def _resolve_cli(*candidates) -> list:
    """argv prefix for a CLI. npm installs a .cmd shim on Windows, which CreateProcess cannot exec
    directly, so a shim is routed through `cmd /c` the way llm_judge_codex.py does."""
    for cand in candidates:
        path = shutil.which(cand)
        if path:
            return ["cmd", "/c", path] if path.lower().endswith((".cmd", ".bat")) else [path]
    raise SystemExit(f"none of {candidates} is on PATH; the CLI backend cannot run")


def claude_cli_complete(model_id: str, timeout: int = 300):
    argv = _resolve_cli("claude.cmd", "claude.exe", "claude")

    def complete(prompt: str) -> str:
        try:
            proc = subprocess.run(
                argv + ["-p", "--model", model_id] + _CLAUDE_FLAGS,
                input=prompt, text=True, encoding="utf-8", errors="replace",
                capture_output=True, timeout=timeout, cwd=_NEUTRAL_CWD,
            )
            if proc.returncode != 0:
                sys.stderr.write(f"[claude-cli {model_id} exit {proc.returncode}] "
                                 f"{proc.stderr[-300:]}\n")
                return ""
            return (proc.stdout or "").strip()
        except subprocess.TimeoutExpired:
            sys.stderr.write(f"[claude-cli {model_id} timeout]\n")
            return ""
        except Exception as err:  # noqa: BLE001  one bad call -> a miss, re-runnable on resume
            sys.stderr.write(f"[claude-cli {model_id} error] {type(err).__name__}: {str(err)[:160]}\n")
            return ""

    return complete


def codex_cli_complete(model_id: str, timeout: int = 300):
    argv = _resolve_cli("codex.cmd", "codex.exe", "codex")

    def complete(prompt: str) -> str:
        # -o writes the final message to a file. On Windows it cannot be a process substitution or
        # /dev/stdout, so a temp file is the portable route; llm_judge_codex.py does the same.
        with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False, encoding="utf-8") as ans:
            ans_path = ans.name
        try:
            proc = subprocess.run(
                argv + ["exec", "--ephemeral", "-s", "read-only", "--skip-git-repo-check",
                        "--color", "never", "-m", model_id,
                        "-c", f"model_reasoning_effort={_CODEX_EFFORT}",
                        "-C", _NEUTRAL_CWD, "-o", ans_path, "-"],
                input=prompt, text=True, encoding="utf-8", errors="replace",
                capture_output=True, timeout=timeout,
            )
            if proc.returncode != 0:
                # Reject rather than log and continue. -o can hold a partial or stale message when
                # the process fails, and returning it would put unattributable text into a cache
                # that the declaration treats as a generation record. The Claude backend already
                # returns empty on a nonzero status; this makes the two agree. An empty return is
                # a null prediction, which the pass summary counts and the operator sees.
                sys.stderr.write(f"[codex-cli {model_id} exit {proc.returncode}, answer rejected] "
                                 f"{proc.stderr[-300:]}\n")
                return ""
            with open(ans_path, encoding="utf-8", errors="replace") as fh:
                return fh.read().strip()
        except subprocess.TimeoutExpired:
            sys.stderr.write(f"[codex-cli {model_id} timeout]\n")
            return ""
        except Exception as err:  # noqa: BLE001
            sys.stderr.write(f"[codex-cli {model_id} error] {type(err).__name__}: {str(err)[:160]}\n")
            return ""
        finally:
            try:
                os.remove(ans_path)
            except OSError:
                pass

    return complete


def make_complete(label: str):
    if label in GATEWAY_MODELS:
        return gateway_complete(GATEWAY_MODELS[label])
    if label in BEDROCK_MODELS:
        return bedrock_complete(BEDROCK_MODELS[label])
    if label in CLI_MODELS:
        kind, model_id = CLI_MODELS[label]
        return (claude_cli_complete if kind == "claude" else codex_cli_complete)(model_id)
    raise SystemExit(f"unknown model label {label!r}; known: "
                     f"{sorted(GATEWAY_MODELS) + sorted(BEDROCK_MODELS) + sorted(CLI_MODELS)}")


# The labels whose caches must not land in the globbed directory. Declared here rather than left to
# an operator, because the isolation is a property the shipped command has to have.
ADDENDUM_LABELS = frozenset({"claude-opus-5", "gpt-6-astra", "gpt-5.6-sol",
                             "llama-3-70b", "llama-3.1-70b"})

ADDENDUM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "llm_judge_addendum")


def _is_addendum(label: str, override) -> bool:
    """Whether this label's cache belongs outside the globbed directory.

    ``regenerate_cache`` writes to the directory ``LLMJudgeLocalization`` globs, so a label left
    there becomes a discoverable arena method the moment it is generated and moves the published
    entrant count. The separation used to depend on an operator remembering to move the file
    afterwards, which is not a property of the shipped command. This makes it one. ``--addendum``
    and ``--no-addendum`` override the table for a label it does not yet know about.
    """
    if override is not None:
        return override
    return label in ADDENDUM_LABELS


@contextlib.contextmanager
def _cache_dir(path: str):
    """Point every cache read and write at ``path`` for the duration of the block.

    ``cache_path`` reads the module-level ``_CACHE_DIR``, and ``load_cache``, the resume scan, each
    checkpoint flush and the final write all go through it. Redirecting it here makes the addendum
    directory the destination from the start.

    The earlier version generated into the arena directory and moved the file afterwards. That is not
    isolation: the second invocation of a resumable pass could not see its own previous predictions,
    so it re-issued every call, wrote them into the arena directory, and only then failed on the
    existing destination, leaving a discoverable cache behind. Redirection rather than relocation also
    keeps the declared pilot-then-full and null-retry workflow working.

    ``llm_judge.py`` is deliberately not edited. Its bytes must stay identical to the committed
    version, because the sidecar records' prompt reconstruction is only legitimate while they are.
    """
    previous = lj._CACHE_DIR
    os.makedirs(path, exist_ok=True)
    lj._CACHE_DIR = path
    try:
        yield
    finally:
        lj._CACHE_DIR = previous


def _refuse_if_complete(method: str, label: str, expected_runs: int) -> None:
    """Reject a finished cache before any call is made or any byte is written.

    A complete addendum cache is immutable under the declaration. Discovering that after spending a
    generation pass would be both wasteful and a chance to overwrite a published artifact.
    """
    existing = lj.load_cache(method, label)
    if existing is None:
        return
    preds = existing.get("predictions") or {}
    nulls = sum(1 for row in preds.values() if row.get("top") is None)
    if len(preds) >= expected_runs and nulls == 0:
        raise SystemExit(
            f"{lj.cache_path(method, label)} is already complete at {len(preds)} runs with no "
            f"nulls. It is immutable under the declaration; delete it deliberately to regenerate.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the LLM-judge panel via NAIRR gateway + Bedrock.")
    ap.add_argument("method", choices=["all_at_once", "step_by_step", "binary_search"])
    ap.add_argument("--models", nargs="+", required=True, help="board labels, e.g. gpt-5.5 llama-3.3-70b")
    ap.add_argument("--limit", type=int, default=None, help="cap runs (pilot); omit for full board")
    ap.add_argument("--workers", type=int, default=8, help="concurrent API calls per model")
    ap.add_argument("--addendum", dest="addendum", action="store_true", default=None,
                    help="write the cache to data/llm_judge_addendum/ so the arena glob cannot "
                         "reach it; the default is on for every label in ADDENDUM_LABELS")
    ap.add_argument("--no-addendum", dest="addendum", action="store_false",
                    help="keep the cache in the globbed directory, which makes it an arena entrant")
    args = ap.parse_args()

    completes = {label: make_complete(label) for label in args.models}
    runs = lj.load_judge_runs()
    if args.limit:
        runs = runs[: args.limit]
    for label in args.models:
        _USAGE["in"] = _USAGE["out"] = 0
        complete = completes[label]
        with contextlib.ExitStack() as stack:
            if _is_addendum(label, args.addendum):
                stack.enter_context(_cache_dir(ADDENDUM_DIR))
                _refuse_if_complete(args.method, label, len(runs))
            path = lj.regenerate_cache(args.method, label, complete, limit=args.limit,
                                       max_workers=args.workers)
            preds = lj.load_cache(args.method, label)["predictions"]
        parsed = sum(1 for r in runs if preds.get(lj.run_key(r), {}).get("top") is not None)
        # top == mistake, which is NOT the scored Top-1. The scored outcome is rank == 1 after the
        # stable sort, and the two disagree on the runs whose top is null; a null wins Top-1 when
        # the gold mistake is step 0. Named for what it is so nobody reports it as Top-1.
        top_eq_mistake = sum(1 for r in runs
                             if preds.get(lj.run_key(r), {}).get("top") == r["mistake"])
        print(f"\n== {label} :: {args.method} == wrote {path}")
        print(f"   runs={len(runs)} parsed={parsed} top_eq_mistake={top_eq_mistake} "
              f"tokens(in/out)={_USAGE['in']}/{_USAGE['out']}")
        print("   top_eq_mistake is a plumbing signal, not the scored Top-1; score with "
              "tools/score_judge_addendum.py\n")


if __name__ == "__main__":
    main()
