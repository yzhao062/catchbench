r"""Write the sidecar records the frozen judge declaration requires, and name what is missing.

The declaration at ``research/catchbench-m4-declaration-v3.md`` in the notes repository requires each
addendum cache to carry a sidecar with six categories of record. The caches were generated without
them. This reconstructs every category that is deterministically reconstructible from the committed
inputs, and marks the rest ``null`` with the reason, in a ``not_recorded`` block that lists each
missing requirement by its number in the declaration.

The distinction that decides what may be reconstructed
------------------------------------------------------
A rendered prompt is a pure function of the run and the prompt code. If the prompt code has not moved
since generation, re-rendering it now produces the same bytes it produced then, so hashing it now is
recomputation rather than backfill. A wall-clock generation time is an observation. It was either
recorded at the time or it is gone, and a file modification time is not a substitute for it.

So this tool reconstructs categories 2, 3, 4 and 5 and the model-identifier half of 6, and it refuses
categories 1 and the pass-index half of 6. ``--verify-code-unchanged`` is the guard on the first
claim: it fails unless ``src/catchbench/llm_judge.py`` in the working tree is byte-identical to the
committed version. That is present equality with HEAD and reaches no further back. It does not show
that HEAD is the code that generated these caches, so re-rendering is legitimate exactly to the
extent that the committed prompt code is the generating code. Run it with that flag, or the sidecar
records the prompt hashes as unverified.

``tools/run_llm_judge_panel.py`` is a separate matter. It was edited after generation. The sidecar
records its worktree hash and its HEAD hash and says which is which. Neither is its generation-time
hash, and that hash is unknown.

Usage:
    python tools/emit_addendum_sidecars.py --verify-code-unchanged
    python tools/emit_addendum_sidecars.py --check
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import inspect
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from catchbench import llm_judge as lj  # noqa: E402
from run_llm_judge_panel import _MAX_OUTPUT_TOKENS, bedrock_complete  # noqa: E402

ADDENDUM_DIR = ROOT / "data" / "llm_judge_addendum"
SIDECAR_SUFFIX = ".sidecar.json"

# The commit the declaration froze the protocol against. Recorded in each sidecar so a reader can
# tell which reference the reconstruction was performed under; "matches HEAD" stops meaning anything
# the moment HEAD moves.
FROZEN_COMMIT = "874433dbd79243e5756d80ed2782617b34bc8ec1"

# The five functions the declaration names. Hashing the source of exactly these, rather than the
# module, is what the declaration asks for; the module hash is recorded separately.
HASHED_FUNCTIONS = ("_trace_block", "all_at_once_prompt", "_all_at_once",
                    "parse_prediction", "_score_vector")

# Provider model id strings exactly as sent, from the tables in tools/run_llm_judge_panel.py. The
# Bedrock pair carries the full invocation id including the cross-region prefix where present.
PROVIDER_MODEL_IDS = {
    "llama-3-70b": {"channel": "aws-bedrock", "model_id": "meta.llama3-70b-instruct-v1:0"},
    "llama-3.1-70b": {"channel": "aws-bedrock", "model_id": "us.meta.llama3-1-70b-instruct-v1:0"},
    "claude-opus-5": {"channel": "claude-code-cli", "model_id": "claude-opus-5"},
    "gpt-6-astra": {"channel": "codex-cli", "model_id": "gpt-6-astra"},
    # Generation-ladder arms, run 2026-09-12 under
    # research/catchbench-generation-ladder-declaration-2026-09-12.md. Same provider models as the
    # two CLI rows above, reached over the NAIRR gateway so that channel is held fixed across these
    # newly generated gateway caches. It is not held fixed against the published claude-opus-4.8 and
    # gpt-5.4, whose own channels are recorded nowhere, which is why each gets its own gateway arm.
    # The provider returned a model identifier on every call:
    # claude-opus-5 and gpt-6-astra respectively, and gpt-5.6-sol on all 126 of its own.
    "claude-opus-5-gw": {"channel": "nairr-gateway", "model_id": "claude-opus-5"},
    "gpt-6-astra-gw": {"channel": "nairr-gateway", "model_id": "gpt-6-astra"},
    "gpt-5.6-sol": {"channel": "nairr-gateway", "model_id": "gpt-5.6-sol"},
    "claude-opus-5-gw2": {"channel": "nairr-gateway", "model_id": "claude-opus-5"},
    "gpt-6-astra-gw2": {"channel": "nairr-gateway", "model_id": "gpt-6-astra"},
    "gpt-5.5-gw": {"channel": "nairr-gateway", "model_id": "gpt-5.5"},
    "gpt-5.4-gw": {"channel": "nairr-gateway", "model_id": "gpt-5.4"},
    "claude-opus-4.8-gw": {"channel": "nairr-gateway", "model_id": "claude-opus-4.8"},
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                             timeout=60)
    except Exception:  # noqa: BLE001
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _committed_blob(relative: str) -> bytes | None:
    """The committed bytes of a path at HEAD, for comparing against the working tree."""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{relative}"],
                             capture_output=True, timeout=60)
    except Exception:  # noqa: BLE001
        return None
    return out.stdout if out.returncode == 0 else None


def _matches_head(relative: str) -> bool | None:
    """Whether git considers the working tree copy unchanged from HEAD.

    Asked of git rather than computed from bytes on purpose. ``git show`` hands back the blob with
    LF while the checkout carries CRLF under autocrlf, so a raw byte comparison reports every file
    on this platform as modified. The first version of this guard did exactly that and refused to
    run on a file git called clean. A guard that fires on line endings gets switched off, which is
    worse than not having it.
    """
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", relative],
                             capture_output=True, timeout=60)
    except Exception:  # noqa: BLE001
        return None
    return True if out.returncode == 0 else (False if out.returncode == 1 else None)


def _normalised_sha256(data: bytes) -> str:
    """Hash with CRLF folded to LF, so the value is comparable across checkouts."""
    return _sha256_bytes(data.replace(b"\r\n", b"\n"))


def code_manifest() -> dict:
    """Repository commit, module hashes, and the five named function source hashes."""
    judge_rel = "src/catchbench/llm_judge.py"
    panel_rel = "tools/run_llm_judge_panel.py"
    judge_path, panel_path = ROOT / judge_rel, ROOT / panel_rel
    judge_head, panel_head = _committed_blob(judge_rel), _committed_blob(panel_rel)

    functions = {}
    for name in HASHED_FUNCTIONS:
        fn = getattr(lj, name)
        functions[name] = _sha256_bytes(inspect.getsource(fn).encode("utf-8"))

    return {
        "repository_commit": _git("rev-parse", "HEAD"),
        "worktree_dirty": bool(_git("status", "--porcelain")),
        "llm_judge_py": {
            "sha256_worktree": _sha256_file(judge_path),
            "sha256_worktree_lf": _normalised_sha256(judge_path.read_bytes()),
            "sha256_head_lf": (_normalised_sha256(judge_head) if judge_head is not None else None),
            "identical_to_head": _matches_head(judge_rel),
            "note": ("identical_to_head comes from git diff, which honours autocrlf. The raw "
                     "worktree hash differs from the blob hash on a CRLF checkout even when the "
                     "file is unmodified, so compare the _lf values across machines."),
        },
        "run_llm_judge_panel_py": {
            "sha256_worktree": _sha256_file(panel_path),
            "sha256_worktree_lf": _normalised_sha256(panel_path.read_bytes()),
            "sha256_head_lf": (_normalised_sha256(panel_head) if panel_head is not None else None),
            "identical_to_head": _matches_head(panel_rel),
            "note": ("This file was edited after generation. Neither hash above is asserted to be "
                     "its generation-time hash."),
        },
        "function_source_sha256": functions,
    }


def prompt_manifest(runs) -> dict:
    """Per-run rendered-prompt hashes, their ordered concatenation, and the historical md5 tag."""
    per_run, ordered = {}, hashlib.sha256()
    first_rendered = None
    for run in runs:
        rendered = lj.all_at_once_prompt(run).encode("utf-8")
        if first_rendered is None:
            first_rendered = rendered
        per_run[lj.run_key(run)] = _sha256_bytes(rendered)
        ordered.update(rendered)
    return {
        "per_run_sha256": per_run,
        "concatenated_sha256": ordered.hexdigest(),
        # The quantity the historical caches' prompt_sha field originally named, computed the way
        # commit f957d14 computed it, so the addendum stays comparable with their 12cff8a0af5b.
        "first_run_md5_12": hashlib.md5(first_rendered).hexdigest()[:12],
        "live_protocol_prompt_sha": lj._protocol_prompt_sha("all_at_once"),
    }


def truncation_manifest(runs) -> dict:
    """How much of the corpus the 1500-character step cap actually bites, per the current code.

    A property of the runs and the cap, so it is identical for all four models. The reply-side
    ceiling is a different matter and is not reconstructible; see ``not_recorded``.
    """
    runs_hit, steps_hit = 0, 0
    for run in runs:
        n = sum(1 for text in run["texts"] if len(" ".join(text.split())) > 1500)
        steps_hit += n
        runs_hit += 1 if n else 0
    return {"cap_chars": 1500, "runs_with_a_truncated_step": runs_hit,
            "truncated_steps_total": steps_hit, "runs_total": len(runs)}


def parser_manifest(predictions: dict, runs) -> dict:
    """Where each run's top came from, recomputed from the stored raw reply.

    The declaration asks for the origin of every top and the failure class of every null. Both are
    recoverable because every prediction retains its raw reply, so this category needs no
    observation that was not kept.
    """
    origins = {"mistake_step_label": 0, "ranking_head": 0, "null": 0}
    per_run, raw_lengths = {}, []
    for run in runs:
        key = lj.run_key(run)
        row = predictions[key]
        raw = row.get("raw") or ""
        raw_lengths.append(len(raw))
        n_steps = len(run["steps"])
        labelled = None
        m = re.search(r"MISTAKE_STEP\s*:\s*(\d+)", raw, re.IGNORECASE)
        if m and 0 <= int(m.group(1)) < n_steps:
            labelled = int(m.group(1))
        if row.get("top") is None:
            origin = "null"
        elif labelled is not None:
            origin = "mistake_step_label"
        else:
            origin = "ranking_head"
        origins[origin] += 1
        per_run[key] = {"origin": origin, "top": row.get("top"), "raw_chars": len(raw)}
    raw_lengths.sort()
    return {
        "origin_counts": origins,
        "raw_reply_chars": {
            "min": raw_lengths[0], "max": raw_lengths[-1],
            "median": raw_lengths[len(raw_lengths) // 2],
        },
        "null_failure_classes": ({} if origins["null"] == 0 else None),
        "per_run": per_run,
    }


def not_recorded(label: str) -> list:
    """The declaration requirements this sidecar cannot satisfy, each with the reason.

    Written as data rather than prose so a later reader cannot mistake silence for compliance. Every
    entry names the requirement number from the declaration's "What the Five New Caches Must Record",
    or names the channel erratum where the obligation comes from there.

    Round 18 widened this list. The first version listed five items and read as though that were the
    whole gap; four more obligations were missing from it, and one entry was inaccurate for the
    Bedrock pair, for which a CLI version is not a meaningful field at all.
    """
    provider = PROVIDER_MODEL_IDS.get(label) or {}
    channel = provider.get("channel", "unknown")
    is_cli = channel in {"claude-code-cli", "codex-cli"}
    ceiling = f"an output ceiling (limit not recorded for the {channel} channel)"
    if channel == "aws-bedrock":
        default = inspect.signature(bedrock_complete).parameters["max_tokens"].default
        limit = min(default, _MAX_OUTPUT_TOKENS.get(provider["model_id"], default))
        ceiling = f"the configured {limit}-token ceiling"
    entries = [
        {"requirement": 1, "field": "generation_start_end_utc", "value": None,
         "reason": ("Not recorded during generation. A wall-clock time is an observation, not a "
                    "derivable quantity, and the declaration forbids substituting file "
                    "modification times.")},
        {"requirement": 5, "field": "replies_stopped_at_output_ceiling", "value": None,
         "reason": ("No stop reason was retained. Reply length alone cannot distinguish a reply "
                    f"that ended at {ceiling} from one that ended on its own.")},
        {"requirement": 6, "field": "pass_index_per_prediction", "value": None,
         "reason": ("Not recorded, and the attempt history is unrecoverable. A cache being complete "
                    "shows what the final state is, not how many passes or calls produced it, so "
                    "the number of passes is unknown rather than one.")},
        {"requirement": 3, "field": "run_llm_judge_panel_generation_time_hash", "value": None,
         "reason": ("The panel script was edited after generation. Its committed and worktree "
                    "hashes are both recorded above; neither is its generation-time hash.")},
        {"requirement": 3, "field": "function_hashes_observed_at_generation", "value": None,
         "reason": ("The declaration requires the five function hashes to be taken when the first "
                    "plumbing check begins and re-checked after the last cache write. Neither "
                    "observation was made. The hashes in code_manifest were computed afterwards and "
                    "are evidence about the current code, not about the code that ran.")},
        {"requirement": 6, "field": "served_checkpoint_version", "value": None,
         "reason": ("No served checkpoint version was retained in this cache or sidecar. "
                    "A response model identifier may be only a model label and does not by "
                    "itself establish an immutable checkpoint.")},
    ]
    if is_cli:
        entries.append(
            {"requirement": "channel erratum", "field": "reasoning_effort_at_generation",
             "value": None,
             "reason": ("The erratum requires the reasoning-effort setting used. It was not "
                        "retained per call. The current script sets Codex to high, which is "
                        "evidence of present configuration rather than an invocation log.")})
        entries.append(
            {"requirement": "channel erratum", "field": "cli_version_at_generation", "value": None,
             "reason": ("Not recorded for the %s channel. A version observed today is not evidence "
                        "of the version that ran, so none is asserted." % channel)})
        entries.append(
            {"requirement": "protocol", "field": "effective_isolation_and_tool_use", "value": None,
             "reason": ("The declaration's intent was an invocation close to a plain API call. The "
                        "command does not establish isolation from user configuration, rules or "
                        "tools, and no per-call record of effective settings or tool use was kept. "
                        "This is a protocol deviation, separate from the six numbered categories, "
                        "and no claim is made either way about whether any call used a tool.")})
    return entries


def build(label: str, cache: dict, runs, shared: dict) -> dict:
    return {
        "schema_version": 1,
        "kind": "catchbench-addendum-sidecar",
        "written_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "written_by": "tools/emit_addendum_sidecars.py",
        "label": label,
        "cache_provenance": cache["provenance"],
        "reconstructed": True,
        "reconstruction_note": (
            "Written after generation, not during it. Every field here is either a deterministic "
            "function of committed inputs or is null with a reason in not_recorded. No value was "
            "inferred from a file timestamp."),
        # Declared configuration read from the current code tables, not an invocation record. No
        # log establishes what string was actually sent, so the field says which it is.
        "provider_declared_configuration": PROVIDER_MODEL_IDS.get(label),
        "provider_from_invocation_record": None,
        "code_manifest": shared["code"],
        "prompt_manifest": shared["prompts"],
        "truncation_manifest": shared["truncation"],
        "parser_manifest": parser_manifest(cache["predictions"], runs),
        "not_recorded": not_recorded(label),
    }


def check_one(target: pathlib.Path, cache: dict, expected: dict) -> int:
    """Validate one stored sidecar. Returns the number of problems found.

    The first version of this called ``exists()`` and moved on, so a sidecar containing arbitrary
    bytes passed a check that advertised detecting stale records. It now parses the file and compares
    the fields that are reconstructible, while deliberately leaving alone the fields that record a
    historical observation: re-stamping ``written_at_utc`` or the recorded commit from today's values
    would destroy the record rather than check it.
    """
    if not target.exists():
        print("missing sidecar: %s" % target.name)
        return 1
    try:
        stored = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        print("unreadable sidecar %s: %s" % (target.name, type(err).__name__))
        return 1
    if not isinstance(stored, dict):
        print("sidecar %s is not a JSON object" % target.name)
        return 1

    problems = 0

    def fail(message: str) -> None:
        nonlocal problems
        problems += 1
        print("%s: %s" % (target.name, message))

    if stored.get("kind") != expected["kind"]:
        fail("kind is %r" % stored.get("kind"))
    if stored.get("label") != expected["label"]:
        fail("label is %r, expected %r" % (stored.get("label"), expected["label"]))

    # Identity: the sidecar must describe this cache's runs, not merely the same number of them.
    cache_keys = set(cache["predictions"])
    prompts = (stored.get("prompt_manifest") or {}).get("per_run_sha256")
    parsed = (stored.get("parser_manifest") or {}).get("per_run")
    if not isinstance(prompts, dict) or not isinstance(parsed, dict):
        fail("missing per-run prompt or parser records")
    else:
        if set(prompts) != cache_keys:
            fail("prompt keys do not match the cache's %d run keys" % len(cache_keys))
        if set(parsed) != cache_keys:
            fail("parser keys do not match the cache's %d run keys" % len(cache_keys))
        if prompts != expected["prompt_manifest"]["per_run_sha256"]:
            fail("stored prompt hashes differ from a fresh reconstruction")

    for section, field in (("prompt_manifest", "concatenated_sha256"),
                           ("prompt_manifest", "first_run_md5_12"),
                           ("truncation_manifest", "truncated_steps_total"),
                           ("truncation_manifest", "runs_with_a_truncated_step")):
        if (stored.get(section) or {}).get(field) != expected[section][field]:
            fail("%s.%s is stale" % (section, field))

    if (stored.get("parser_manifest") or {}).get("origin_counts") != \
            expected["parser_manifest"]["origin_counts"]:
        fail("parser origin counts are stale")

    stored_gaps = stored.get("not_recorded")
    if not isinstance(stored_gaps, list) or not stored_gaps:
        fail("not_recorded is missing or empty, which would read as full compliance")
    else:
        stored_fields = {e.get("field") for e in stored_gaps if isinstance(e, dict)}
        missing = {e["field"] for e in expected["not_recorded"]} - stored_fields
        if missing:
            fail("not_recorded omits %s" % ", ".join(sorted(missing)))
        for entry in stored_gaps:
            if isinstance(entry, dict) and entry.get("value") is not None:
                fail("not_recorded entry %r carries a value" % entry.get("field"))

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="Write or check addendum sidecar records.")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if a sidecar is missing or stale rather than writing")
    ap.add_argument("--verify-code-unchanged", action="store_true",
                    help="refuse to write unless llm_judge.py matches its committed bytes")
    args = ap.parse_args()

    caches = sorted(ADDENDUM_DIR.glob("whoandwhen__*__*.json"))
    caches = [p for p in caches if not p.name.endswith(SIDECAR_SUFFIX)]
    if not caches:
        raise SystemExit("no addendum caches in %s" % ADDENDUM_DIR)

    code = code_manifest()
    if args.verify_code_unchanged and code["llm_judge_py"]["identical_to_head"] is not True:
        raise SystemExit(
            "git reports src/catchbench/llm_judge.py as modified relative to HEAD, or could not be "
            "asked. Re-rendering the prompts would not reproduce what was sent, so the prompt "
            "hashes would be fiction. Refusing.")
    # Named for what the guard establishes rather than for what a reader might hope it does. It
    # shows the prompt code matches the frozen reference commit *now*. It does not show which bytes
    # ran during generation, nor that the code held still throughout it, and the reference itself
    # moves whenever HEAD moves. Round 18 rejected the earlier "prompt_reconstruction_verified",
    # which asserted more than this.
    code = dict(code,
                prompt_code_matches_reference_commit_at_write_time=bool(args.verify_code_unchanged),
                reference_commit=FROZEN_COMMIT,
                reference_commit_is_head=(code.get("repository_commit") == FROZEN_COMMIT),
                reconstruction_scope=(
                    "Prompt hashes are recomputed from the committed runs under the code present at "
                    "write time. This is a reconstruction under a stated protocol, not a record of "
                    "what was sent."))

    runs = lj.load_judge_runs()
    shared = {"code": code, "prompts": prompt_manifest(runs), "truncation": truncation_manifest(runs)}

    problems = 0
    for path in caches:
        label = path.stem.split("__")[-1]
        cache = json.loads(path.read_text(encoding="utf-8"))
        sidecar = build(label, cache, runs, shared)
        target = path.with_name(path.stem + SIDECAR_SUFFIX)
        rendered = json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n"
        if args.check:
            problems += check_one(target, cache, sidecar)
            continue
        target.write_text(rendered, encoding="utf-8")
        print("wrote %s  (%d prompt hashes, %d unmet requirement(s))"
              % (target.name, len(shared["prompts"]["per_run_sha256"]), len(sidecar["not_recorded"])))

    if args.check:
        print("%d problem(s)" % problems)
        return 1 if problems else 0
    print("\n  Prompt hashes are a reconstruction under the frozen reference commit, not a record")
    print("  of what was sent. Run with --verify-code-unchanged to assert the guard ran.")
    print("  Unmet declaration and protocol obligations are listed in each sidecar's not_recorded")
    print("  block; none of them is inferred, and none is filled in from a file timestamp.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
