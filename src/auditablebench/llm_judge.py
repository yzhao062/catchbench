"""LLM-judge localization baselines: the "just ask a frontier LLM" control for POST localization.

These are the field-standard baselines a reviewer asks for first. Given a failed multi-agent run,
prompt an LLM over the trace and ask which step is the decisive mistake, then score that answer on the
same 126-run Who&When board the structural methods use, with the same Top-1 / Top-3 / MRR. Faithful to
Who&When (Zhang et al., 2025, arXiv:2505.00212), which defines three query strategies:

  - all-at-once   : show the whole trace once, ask for the single decisive mistake step.
  - step-by-step  : walk the trace forward, ask at each step whether the mistake has occurred yet.
  - binary-search : halve the interval each query until one suspected step remains.

Reproducibility contract. LLM calls are non-deterministic and cost-bearing, so they never run inside
the scoring path. A generation pass (``regenerate_cache``) writes per-run predictions to a committed
JSON cache with full provenance (model, date, prompt hash); the board scores from that cache
deterministically. ``run.py`` reads the cache when present and skips the method with a clear note when
it is absent, so the core board stays zero-API and reproducible. The backend is a pluggable
``complete(prompt) -> str`` callable, so the same harness runs GPT-5.5 (via the Codex CLI), a local
open-weights model, or any other judge without touching the scoring code.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from functools import lru_cache
from typing import Callable, List, Mapping, Optional

from auditablebench import _reuse  # noqa: F401  side effect: sets sys.path for grade + auditable

import numpy as np  # noqa: E402

from agent_failure_localization import _rank_metrics  # noqa: E402  GRADE's verified ranking metric

import agent_graph_characterization as ww  # noqa: E402  the Who&When loader (raw history + labels)

_METRIC_NAMES = ("top1", "top3", "mrr")
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "llm_judge")


@lru_cache(maxsize=1)
def load_judge_runs() -> List[dict]:
    """The same 126 failed Who&When runs the structural board uses, but carrying the raw step text the
    LLM needs. Mirrors ``agent_failure_localization.load_failed_runs`` filter for filter, in the same
    order, so a prediction here scores against the same ``mistake_step`` the structural methods do.
    Memoized: the corpus is static within a process, and the board calls this once per judge method."""
    ww._ensure_corpus()
    runs = []
    for t in ww.load_tasks():
        if str(t.get("is_correct", "")).lower() != "false":
            continue
        steps = ww.to_steps(t)
        if len(steps) < 3:
            continue
        try:
            ms = int(t.get("mistake_step"))
        except (TypeError, ValueError):
            continue
        if not (0 <= ms < len(steps)):
            continue
        history = t.get("history", [])
        texts = [str(history[s["idx"]].get("content", "")) for s in steps]  # idx == history index
        runs.append({"steps": steps, "texts": texts, "mistake": ms})
    for i, run in enumerate(runs):  # stable load order (sorted glob + fixed filter), verified to match
        run["key"] = f"{i:03d}-{_content_digest(run)}"  # the board's load_failed_runs order run for run
    return runs


def _content_digest(run: dict) -> str:
    parts = [f"{s['agent']}/{s['kind']}/{hashlib.md5(t.encode('utf-8')).hexdigest()[:8]}"
             for s, t in zip(run["steps"], run["texts"])]
    return hashlib.md5(("|".join(parts) + f"#{len(run['steps'])}").encode("utf-8")).hexdigest()[:12]


def run_key(run: dict) -> str:
    """The stable per-run cache key set by ``load_judge_runs``: the load-order index (unique, and
    identical between generation and scoring because both iterate the corpus deterministically)
    prefixed onto a content digest. The index defeats true duplicate traces in the corpus, which
    share a content digest; the digest guards against a corpus reordering going unnoticed."""
    return run["key"]


# --- prompt construction ----------------------------------------------------------

def _trace_block(run: dict, max_chars: int = 1500) -> str:
    """Render the run as a numbered step list (index, agent, content) for the prompt. Long step
    contents are truncated to keep the whole trace within a sane budget; the index and agent, which
    the answer is keyed to, are never truncated."""
    lines = []
    for s, text in zip(run["steps"], run["texts"]):
        body = " ".join(text.split())
        if len(body) > max_chars:
            body = body[:max_chars] + " ...[truncated]"
        lines.append(f"[step {s['idx']}] ({s['agent']}, {s['kind']}): {body}")
    return "\n".join(lines)


def all_at_once_prompt(run: dict) -> str:
    """Who&When all-at-once: the whole trace, one query, name the decisive mistake step. We also ask
    for a short ranked shortlist so the board's Top-3 and MRR columns have a ranking to score, while
    Top-1 uses the single decisive step the method is defined to produce."""
    n = len(run["steps"])
    return (
        "You are auditing a failed multi-agent run. Each step is one agent action, indexed from 0 to "
        f"{n - 1}. Exactly one step is the decisive mistake: the first step whose error caused the "
        "run to fail (not a downstream symptom of an earlier error).\n\n"
        "TRACE:\n" + _trace_block(run) + "\n\n"
        "Identify the decisive mistake step. Then list the steps from most to least likely to be the "
        "decisive mistake (most likely first), as a comma-separated list of step indices.\n"
        "Answer in exactly this format, nothing else:\n"
        "MISTAKE_STEP: <index>\n"
        "RANKING: <index>, <index>, <index>, ...\n"
    )


_PROMPTS: Mapping[str, Callable[[dict], str]] = {"all_at_once": all_at_once_prompt}


# --- response parsing -------------------------------------------------------------

def parse_prediction(text: str, n_steps: int) -> dict:
    """Pull MISTAKE_STEP and RANKING out of an LLM response, clamped to valid step indices. Robust to
    extra prose: takes the first integer after each label. A missing ranking falls back to the single
    mistake step; a missing mistake step falls back to the ranking head; total failure yields an empty
    prediction the scorer treats as a miss (score 0 for that run)."""
    top: Optional[int] = None
    ranking: List[int] = []
    m = re.search(r"MISTAKE_STEP\s*:\s*(\d+)", text, re.IGNORECASE)
    if m:
        v = int(m.group(1))
        if 0 <= v < n_steps:
            top = v
    m = re.search(r"RANKING\s*:\s*([0-9,\s]+)", text, re.IGNORECASE)
    if m:
        seen = set()
        for tok in re.findall(r"\d+", m.group(1)):
            v = int(tok)
            if 0 <= v < n_steps and v not in seen:
                seen.add(v)
                ranking.append(v)
    if top is None and ranking:
        top = ranking[0]
    if top is not None and not ranking:
        ranking = [top]
    return {"top": top, "ranking": ranking}


def _score_vector(pred: dict, n_steps: int) -> np.ndarray:
    """Turn a parsed prediction into a per-step score (higher = more suspected), so the board's
    ranking metric can read it. The ranked shortlist gets descending scores; the decisive step is
    forced to the top; unranked steps get zero."""
    scores = np.zeros(n_steps, dtype=float)
    ranking = pred.get("ranking") or []
    for rank, idx in enumerate(ranking):
        if 0 <= idx < n_steps:
            scores[idx] = max(scores[idx], float(n_steps - rank))
    top = pred.get("top")
    if top is not None and 0 <= top < n_steps:
        scores[top] = float(n_steps + 1)  # the single decisive step outranks the shortlist
    return scores


# --- cache I/O --------------------------------------------------------------------

def cache_path(method: str, model: str) -> str:
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model)
    return os.path.join(_CACHE_DIR, f"whoandwhen__{method}__{safe_model}.json")


def load_cache(method: str, model: str) -> Optional[dict]:
    path = cache_path(method, model)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --- the board method -------------------------------------------------------------

class LLMJudgeLocalization:
    """Score a cached LLM-judge run set on the POST localization board. Deterministic: reads the
    committed prediction cache and scores it with GRADE's ``_rank_metrics``. If the cache is missing,
    ``evaluate`` raises, and ``run.py`` skips the method with a note rather than making live calls."""

    supports = {"post_localization"}

    def __init__(self, method: str, model: str, label: Optional[str] = None) -> None:
        self.method = method
        self.model = model
        self.method_id = label or f"llm-judge {method.replace('_', '-')} ({model})"

    def _cache_errors(self, cache) -> List[str]:
        """Why the cache cannot be scored as a full board: it must be a dict whose ``predictions`` is
        a dict keyed EXACTLY by the current run keys (no missing, no extra), each entry shaped
        {top, ranking, raw} with valid value types and in-range indices. A count- or key-only check
        would still let a malformed payload (a string ``top``, an out-of-range index, a non-dict root)
        crash or silently mis-score at evaluation; validating the values too is what keeps the zero-API
        board trustworthy if the corpus changes or a cache file is edited underneath it."""
        if not isinstance(cache, dict):
            return ["cache root is not a dict"]
        predictions = cache.get("predictions")
        if not isinstance(predictions, dict):
            return ["missing predictions object"]
        runs = {run_key(run): run for run in load_judge_runs()}
        expected, actual = set(runs), set(predictions)
        errors = []
        if expected - actual:
            errors.append(f"missing {len(expected - actual)} expected run predictions")
        if actual - expected:
            errors.append(f"has {len(actual - expected)} unknown run predictions")
        for key in expected & actual:  # by key, so the in-range check knows each run's step count
            shape_error = self._pred_shape_error(predictions[key], len(runs[key]["steps"]))
            if shape_error:
                errors.append(f"{key}: {shape_error}")
                break
        return errors

    @staticmethod
    def _pred_shape_error(pred, n_steps: int) -> Optional[str]:
        """A reason ``pred`` is not a scoreable prediction for a run of ``n_steps``, else None. Guards
        the value types ``_score_vector`` relies on: ``top`` is None or an in-range int, ``ranking`` is
        a list of in-range ints, ``raw`` is a string."""
        if not isinstance(pred, dict) or not {"top", "ranking", "raw"} <= set(pred):
            return "not a {top, ranking, raw} dict"
        top = pred["top"]
        if top is not None and not (isinstance(top, int) and 0 <= top < n_steps):
            return "top is not None or an in-range int"
        ranking = pred["ranking"]
        if not isinstance(ranking, list) or any(
                not (isinstance(i, int) and 0 <= i < n_steps) for i in ranking):
            return "ranking is not a list of in-range step indices"
        if not isinstance(pred["raw"], str):
            return "raw is not a string"
        return None

    def available(self) -> bool:
        """True only when a cache exists AND its keys exactly match the current runs, so a partial,
        stale, or wrong-key cache never silently scores as a full board."""
        cache = load_cache(self.method, self.model)
        return cache is not None and not self._cache_errors(cache)

    def evaluate(self, task) -> Mapping[str, float]:
        task.setup()
        cache = load_cache(self.method, self.model)
        if cache is None:
            raise FileNotFoundError(f"no LLM-judge cache for {self.method}/{self.model}; "
                                    "run regenerate_cache first (opt-in, needs a backend)")
        errors = self._cache_errors(cache)
        if errors:
            raise ValueError(f"unscoreable LLM-judge cache for {self.method}/{self.model}: "
                             + "; ".join(errors))
        preds = cache["predictions"]
        score_chunks = []
        for run in load_judge_runs():  # same order as PostLocalization, so pooled scores align
            score_chunks.append(_score_vector(preds[run_key(run)], len(run["steps"])))
        scores = np.concatenate(score_chunks)
        metrics = _rank_metrics(scores, task.groups, task.mistake_row)
        return dict(zip(_METRIC_NAMES, (float(v) for v in metrics)))


def llm_judge_methods(model: str) -> list:
    """The LLM-judge baselines whose caches exist for ``model``, in display order. Methods without a
    committed cache are omitted, so the board only shows reproducible rows."""
    out = []
    for method in ("all_at_once", "step_by_step", "binary_search"):
        m = LLMJudgeLocalization(method, model)
        if m.available():
            out.append(m)
    return out


def discovered_llm_judge_methods() -> list:
    """Every LLM-judge model with a complete cache, discovered from the cache directory, so the board
    shows the full panel without hardcoding model names. Ordered by method then model for a stable
    leaderboard. A partial cache is skipped by ``available``."""
    out = []
    for path in sorted(glob.glob(os.path.join(_CACHE_DIR, "whoandwhen__*__*.json"))):
        parts = os.path.basename(path)[:-len(".json")].split("__")
        if len(parts) != 3:
            continue
        _, method, model = parts
        if method not in _PROMPTS and method not in ("step_by_step", "binary_search"):
            continue
        candidate = LLMJudgeLocalization(method, model)
        if candidate.available():
            out.append(candidate)
    return out


# --- generation (opt-in, never on the scoring path) -------------------------------

def regenerate_cache(method: str, model: str, complete: Callable[[str], str],
                     limit: Optional[int] = None, verbose: bool = True,
                     checkpoint_every: int = 10, resume: bool = True, max_workers: int = 1) -> str:
    """Build the prediction cache for one method by calling ``complete(prompt) -> str`` per run. This
    is the only place LLM calls happen; it is opt-in and writes a committed artifact.

    Resumable and checkpointed (a generation pass can be killed and restarted): with ``resume`` it
    loads any existing cache and skips runs already predicted, retrying only runs that previously
    failed (``top`` is None, e.g. a timeout). It flushes the cache every ``checkpoint_every`` new
    predictions and once at the end. ``max_workers`` > 1 runs the (I/O-bound) API calls concurrently;
    the predictions dict is mutated and flushed only from the main thread, so checkpointing is safe."""
    if method not in _PROMPTS:
        raise ValueError(f"prompt for method {method!r} not implemented yet")
    runs = load_judge_runs()
    if limit is not None:
        runs = runs[:limit]
    build = _PROMPTS[method]
    prompt_sha = hashlib.md5(build(runs[0]).encode("utf-8")).hexdigest()[:12] if runs else ""
    path = cache_path(method, model)

    predictions = {}
    if resume:
        existing = load_cache(method, model)
        if existing:
            predictions = dict(existing.get("predictions", {}))

    def _flush() -> None:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        out = {
            "provenance": {"benchmark": "AuditableBench", "board": "post_localization",
                           "corpus": "whoandwhen", "method": method, "model": model,
                           "n_runs": len(runs), "prompt_sha": prompt_sha,
                           "paper": "Zhang et al. 2025 (Who&When), arXiv:2505.00212"},
            "predictions": predictions,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)

    todo = [r for r in runs if not (resume and predictions.get(run_key(r), {}).get("top") is not None)]
    if verbose:
        print(f"{model}/{method}: {len(runs) - len(todo)} cached, {len(todo)} to do "
              f"(workers={max_workers})", flush=True)

    def _work(run: dict):
        raw = complete(build(run))
        return run_key(run), parse_prediction(raw, len(run["steps"])), raw, run["mistake"]

    def _record(result, done: int) -> None:
        key, pred, raw, gold = result
        predictions[key] = {**pred, "raw": raw}
        if verbose:
            print(f"[{done}/{len(todo)}] key={key} top={pred['top']} gold={gold} "
                  f"{'HIT' if pred['top'] == gold else ''}", flush=True)
        if checkpoint_every and done % checkpoint_every == 0:
            _flush()

    if max_workers <= 1:
        for i, run in enumerate(todo, 1):
            _record(_work(run), i)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_work, run): run for run in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                _record(fut.result(), i)
    _flush()
    return path
