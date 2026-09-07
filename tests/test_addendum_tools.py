r"""Command-level coverage for the addendum tools.

Every defect these tests pin was found by review rather than by the suite, and each was invisible to
the checks that already existed. The stdlib-shadow test only compares tool names. Nothing executed
these scripts, so adding a sidecar file beside the caches broke the public scoring command while the
whole suite stayed green.

The tests run the real commands against the real committed artifacts. That is the point: the failures
they cover all came from how the shipped files interact, not from logic that a fixture would reach.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
ADDENDUM = ROOT / "data" / "llm_judge_addendum"
SIDECAR_SUFFIX = ".sidecar.json"

pytestmark = pytest.mark.skipif(not ADDENDUM.is_dir(),
                                reason="addendum caches are not present in this checkout")


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOLS / script), *args],
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          cwd=str(ROOT), timeout=600)


def caches() -> list[pathlib.Path]:
    return sorted(p for p in ADDENDUM.glob("whoandwhen__*__*.json")
                  if not p.name.endswith(SIDECAR_SUFFIX))


def sidecars() -> list[pathlib.Path]:
    return sorted(ADDENDUM.glob("*" + SIDECAR_SUFFIX))


def test_the_scorer_runs_with_sidecars_sitting_beside_the_caches():
    """The sidecars share the cache name prefix, so a naive glob reads one as a prediction cache.

    That is exactly what happened: the documented command exited 1 with KeyError: 'predictions'
    while the suite reported no failure, because no test ran the script.
    """
    assert sidecars(), "this test is meaningless without a sidecar present"
    result = run("score_judge_addendum.py")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "Top-1" in result.stdout


def test_the_scorer_reports_positions_and_an_endpoint_count_but_never_a_tie_group():
    """The output must not describe overlapping intervals as tied or equal.

    Overlap is not transitive, so a shared endpoint count is not an equality class. The tool said it
    was, until review produced a counterexample from the tool's own output.
    """
    result = run("score_judge_addendum.py")
    assert result.returncode == 0, result.stderr[-2000:]
    lowered = result.stdout.lower()
    assert "above" in lowered
    for forbidden in ("share a rank", "tied", "are equal", "statistically equal"):
        assert forbidden not in lowered, f"output claims {forbidden!r}"


def test_equal_scores_take_equal_positions():
    """Two arms with the same Top-1 count must not be ordered by load order."""
    result = run("score_judge_addendum.py")
    assert result.returncode == 0, result.stderr[-2000:]
    rows = [line.split() for line in result.stdout.splitlines()
            if line.startswith("  ") and len(line.split()) > 6 and line.split()[0].isdigit()]
    header = next(line.split() for line in result.stdout.splitlines()
                  if line.lstrip().startswith("# "))
    score_column = header.index("Top-1")
    by_score: dict[str, set[str]] = {}
    for row in rows:
        by_score.setdefault(row[score_column], set()).add(row[0])
    for score, positions in by_score.items():
        assert len(positions) == 1, f"score {score} occupies positions {sorted(positions)}"


def test_the_archived_family_tool_still_runs():
    """Kept executable on purpose: it is the record of what the declared family concluded."""
    result = run("score_judge_addendum_family.py")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "0 of 7 separate" in result.stdout


def test_the_sidecar_check_rejects_a_sidecar_that_is_not_json(tmp_path):
    """``--check`` used to call exists() and continue, so arbitrary bytes passed."""
    target = sidecars()[0]
    backup = tmp_path / target.name
    shutil.copy2(target, backup)
    try:
        target.write_text("not json at all", encoding="utf-8")
        result = run("emit_addendum_sidecars.py", "--check")
        assert result.returncode == 1, "a corrupt sidecar passed the check"
    finally:
        shutil.copy2(backup, target)
    assert run("emit_addendum_sidecars.py", "--check").returncode == 0


def test_the_sidecar_check_rejects_records_naming_the_wrong_runs(tmp_path):
    """Identity, not arithmetic: matching counts do not make two maps describe the same runs."""
    target = sidecars()[0]
    backup = tmp_path / target.name
    shutil.copy2(target, backup)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        per_run = payload["parser_manifest"]["per_run"]
        victim = next(iter(per_run))
        per_run["renamed-" + victim] = per_run.pop(victim)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        result = run("emit_addendum_sidecars.py", "--check")
        assert result.returncode == 1, "a sidecar naming different runs passed the check"
    finally:
        shutil.copy2(backup, target)
    assert run("emit_addendum_sidecars.py", "--check").returncode == 0


def test_every_sidecar_declares_what_was_not_recorded():
    """An empty gap list would read as full compliance with the declaration."""
    for path in sidecars():
        payload = json.loads(path.read_text(encoding="utf-8"))
        gaps = payload["not_recorded"]
        assert gaps, f"{path.name} claims no unmet obligations"
        for entry in gaps:
            assert entry["value"] is None, f"{path.name} filled in {entry['field']}"
            assert entry["reason"].strip(), f"{path.name} gives no reason for {entry['field']}"


def test_sidecars_cover_exactly_their_cache_runs():
    for cache_path in caches():
        sidecar_path = cache_path.with_name(cache_path.stem + SIDECAR_SUFFIX)
        assert sidecar_path.exists(), f"no sidecar for {cache_path.name}"
        cache_keys = set(json.loads(cache_path.read_text(encoding="utf-8"))["predictions"])
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert set(payload["prompt_manifest"]["per_run_sha256"]) == cache_keys
        assert set(payload["parser_manifest"]["per_run"]) == cache_keys


def test_addendum_labels_are_kept_out_of_the_arena_directory():
    """The isolation has to be a property of the command, not of an operator remembering to move."""
    sys.path.insert(0, str(TOOLS))
    try:
        import run_llm_judge_panel as panel
    finally:
        sys.path.pop(0)
    for label in ("claude-opus-5", "gpt-6-astra", "llama-3-70b", "llama-3.1-70b"):
        assert panel._is_addendum(label, None), f"{label} would generate into the arena directory"
    assert not panel._is_addendum("gpt-5.5", None)
    assert panel._is_addendum("gpt-5.5", True)
    assert not panel._is_addendum("llama-3-70b", False)
