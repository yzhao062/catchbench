r"""Sections C and D of the POST generalization declaration, run before any arm is fitted.

The analysis is ``research/catchbench-post-generalization-declaration-2026-09-11.md``, frozen at
2026-09-11T01:05:49Z before any arm was fitted on either new corpus.
``catchbench.post_generalization_audit`` holds the arms, the fold path and the checks; this script
is the driver, and the split is the one ``tools/emit_live_size_audit.py`` keeps from
``catchbench.live_size_audit``.

    GRADE_DIR=/path/to/grade python tools/post_generalization_preflight.py \
        --declaration /path/to/catchbench-post-generalization-declaration-2026-09-11.md

**This script fits no arm of the declared batch.** Section J makes the batch one pass of 200
classifier fits and forbids a rerun to chase a disappointing effect, so a preflight that scored the
batch's cells would have spent it. What it does fit is the ScienceWorld parity comparison that
section D requires, twice over, once down each fold path; those 200 fits answer whether the new
partition route changed the fitting and produce no declared quantity, and the comparison deliberately
carries the differences between the two paths rather than the scores themselves.

What each section asks for, and where it is answered:

  C  the OpenHands population, exactly          ``post_generalization_audit.check_population``
  C  the ScienceWorld population, exactly       ``check_population``
  C  the ScienceWorld task identities           ``_scienceworld_identities``, through the raw JSON
  C  the SWE-Gym issue overlap                  ``swegym_issue_overlap``, through the raw parquet
  D  the fold protocol and its assertions       ``check_fold_protocol``
  D  parity with ``_cv_estimator_detail``       ``check_saved_split_parity``
  B  only the size columns are splined          ``check_column_boundaries``

Two kinds of failure, handled differently and on purpose. A structural check raises: a moved
population, a fold missing a class, a group split across a fold boundary, a moved column boundary or
a replayed identity that does not line up all mean the cells are not the cells the declaration
describes, and no record of such a run is worth writing. A parity comparison is collected instead,
because whether a disagreement is one arm or the whole route is what decides what it means and
stopping at the first one cannot tell those apart. Nothing here relaxes
``post_generalization_audit.PARITY_ATOL``, which was written into that module before the first fit
for exactly this situation.

Neither corpus is registered anywhere. ``detection._LOADERS`` is untouched, no board roster gains a
row, and the nine boards, the 72 entrants and the 138-record registry are what they were. The two
new Hugging Face repositories are recorded by resolved revision rather than added to
``catchbench.corpora.CORPUS_REVISIONS``: that tuple is what ``revision_header`` prints onto the
board, so an entry there would move a board header for corpora that are not on the board. The
population check is what catches a corpus that moved under the audit.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

CORPORA = ("openhands", "scienceworld")

# The corpus whose partition is grouped, and therefore the one the parity check cannot run on: its
# saved splits are not the splits ``_cv_estimator_detail`` builds, so comparing the two paths there
# would compare two different analyses rather than two implementations of one.
PARITY_CORPUS = "scienceworld"

DEFAULT_OUTPUT = ROOT / "tools" / "post_generalization_preflight.json"

# The three Hugging Face repositories this preflight reads, recorded by resolved head so the record
# says which population it measured. SWE-Gym is in ``catchbench.corpora.CORPUS_REVISIONS`` and its
# fetches are pinned; the two new ones are not, for the reason the module docstring gives.
REPOSITORIES = {
    "OpenHands / SWE-rebench": "nebius/SWE-rebench-openhands-trajectories",
    "ScienceWorld": "lclan/webshop_expert_trajectories",
    "SWE-Gym": "SWE-Gym/OpenHands-Sampled-Trajectories",
}

_HUB_WRAPPED = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpora", nargs="+", choices=CORPORA, default=list(CORPORA))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--declaration", type=Path, default=None,
                        help="the frozen declaration, recorded by path and sha256")
    parser.add_argument("--no-hub-revisions", action="store_true",
                        help="skip resolving the dataset heads; for an offline rerun over a cache a "
                             "previous verified run already populated")
    return parser.parse_args()


def _wrap_hub_listing(attempts: int = 5, pause: float = 5.0) -> None:
    """Cache and retry ``list_repo_files``, because the Hub rate-limits repeated listing calls.

    Three loaders reach for the file list on this run and two of them do it twice, once to build the
    graphs and once to replay the identities. A 429 on the third call is not a corpus that is
    unavailable, it is the same corpus a moment later, so the wrapper waits and asks again and
    remembers the answer. The cache is per process and per repository, which is the scope of one
    preflight.

    It wraps whatever ``list_repo_files`` currently is, so it sits OUTSIDE the revision pin
    ``catchbench.corpora.install_hub_revision_pins`` installed when ``catchbench`` was imported. A
    cached listing is therefore a listing of the pinned revision, not a way around it.
    """
    global _HUB_WRAPPED
    if _HUB_WRAPPED:
        return
    import huggingface_hub

    inner = huggingface_hub.list_repo_files
    cache: dict[tuple, list] = {}

    @functools.wraps(inner)
    def listing(repo_id, *args, **kwargs):
        key = (repo_id, kwargs.get("repo_type"), kwargs.get("revision"))
        if key in cache:
            return cache[key]
        last = None
        for attempt in range(attempts):
            try:
                cache[key] = list(inner(repo_id, *args, **kwargs))
                return cache[key]
            except Exception as error:  # noqa: BLE001 - the Hub raises several types for a 429
                last = error
                if attempt + 1 < attempts:
                    time.sleep(pause * (attempt + 1))
        raise RuntimeError(
            f"listing {repo_id} failed {attempts} times, the last with {type(last).__name__}: "
            f"{last}"
        ) from last

    huggingface_hub.list_repo_files = listing
    _HUB_WRAPPED = True


def _hub_revisions(attempts: int = 5, pause: float = 5.0) -> dict:
    """The resolved ``main`` head of each repository this preflight reads.

    Recorded rather than enforced for the two new corpora, which are deliberately absent from
    ``CORPUS_REVISIONS``. A head that moved between this preflight and the batch would move the
    populations, and section C's counts are what catch that.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    resolved = {}
    for name, repo_id in REPOSITORIES.items():
        for attempt in range(attempts):
            try:
                resolved[name] = {"repo_id": repo_id,
                                  "head": str(api.dataset_info(repo_id, revision="main").sha)}
                break
            except Exception as error:  # noqa: BLE001 - the Hub raises several types for a 429
                if attempt + 1 == attempts:
                    resolved[name] = {"repo_id": repo_id,
                                      "head": f"unresolved ({type(error).__name__}: {error})"}
                else:
                    time.sleep(pause * (attempt + 1))
    return resolved


# --- provenance ----------------------------------------------------------------------------------
# Defined here rather than imported from tools/emit_live_size_audit.py. The two declarations are
# independent documents, and making one audit's driver a dependency of the other's to reach four
# five-line helpers would couple them for no gain.


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(path: Path) -> str:
    """The checked-out commit of a repository, or a reason string; never an exception."""
    try:
        completed = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                                   capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:  # pragma: no cover - environment
        return f"unavailable ({error})"
    return completed.stdout.strip()


def _git_worktree(path: Path) -> dict:
    """Whether the worktree is clean, and what is in it if not.

    The paths matter as much as the flag. This run writes its own record into the repository it
    reads, so ``clean`` is false by the time the record is emitted even when every input was read at
    the declared commit. Listing what is uncommitted lets a reader see that rather than take the
    flag at face value.
    """
    try:
        completed = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                                   capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:  # pragma: no cover - environment
        return {"clean": f"unavailable ({error})", "uncommitted": None}
    entries = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {"clean": not entries, "uncommitted": entries}


def _inputs(declaration: Path | None) -> dict:
    """Every file this preflight read, by path and sha256, and the repositories it read them at."""
    import agent_failure_detection

    grade_root = Path(agent_failure_detection.__file__).resolve().parents[1]
    paper_dir = os.environ.get("CATCHBENCH_PAPER_DIR")
    repositories = {
        "catchbench": {"path": str(ROOT), "commit": _git_commit(ROOT), **_git_worktree(ROOT)},
        "grade": {"path": str(grade_root), "commit": _git_commit(grade_root),
                  **_git_worktree(grade_root)},
    }
    if paper_dir:
        repositories["paper"] = {"path": paper_dir, "commit": _git_commit(Path(paper_dir)),
                                 **_git_worktree(Path(paper_dir)), "written_to": False}
    packages = {}
    for package in ("numpy", "scipy", "scikit-learn"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover - environment
            packages[package] = "not installed"
    grade = Path(agent_failure_detection.__file__).resolve()
    inputs = {
        "code": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in (
                ("src/catchbench/detection.py", ROOT / "src" / "catchbench" / "detection.py"),
                ("src/catchbench/post_generalization_audit.py",
                 ROOT / "src" / "catchbench" / "post_generalization_audit.py"),
                ("tools/post_generalization_preflight.py", Path(__file__).resolve()),
                ("grade/experiment/agent_failure_detection.py", grade),
                ("grade/experiment/agent_graph_openhands.py",
                 grade.parent / "agent_graph_openhands.py"),
                ("grade/experiment/agent_graph_scienceworld.py",
                 grade.parent / "agent_graph_scienceworld.py"),
                ("grade/experiment/agent_graph_swegym.py", grade.parent / "agent_graph_swegym.py"),
            )
        },
        "repositories": repositories,
        "package_versions": packages,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "records_read": [],
        "note": "the preflight reads corpora and fits only the ScienceWorld parity comparison. It "
                "consumes no committed result record, and it writes nothing outside its own output.",
    }
    if declaration is not None:
        inputs["declaration"] = {"path": str(declaration), "sha256": _sha256(declaration),
                                 "sections": "C (populations) and D (folds and fitting)"}
    return inputs


# --- the preflight -------------------------------------------------------------------------------


def preflight(corpora: list[str], hub_revisions: bool, declaration: Path | None) -> dict:
    """Sections C and D on every requested corpus, before any declared quantity is computed."""
    from catchbench import post_generalization_audit as audit
    from catchbench.corpora import verify_pinned_fetches

    _wrap_hub_listing()
    revisions = _hub_revisions() if hub_revisions else {}

    per_corpus: dict[str, dict] = {}
    captured: list[dict] = []
    findings: list[str] = []
    tasks: dict[str, object] = {}

    for corpus in corpora:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            task = audit.PostGeneralizationTask(corpus)
            task.setup()
            tasks[corpus] = task

            population = audit.check_population(task)
            boundaries = audit.check_column_boundaries(task)
            splits = audit.fold_splits(task)
            folds = audit.check_fold_protocol(task, splits)

            parity = None
            if corpus == PARITY_CORPUS:
                parity = audit.check_saved_split_parity(task, splits)
                findings += parity["failures"]
        for warning in caught:
            captured.append({"corpus": corpus, "category": warning.category.__name__,
                             "message": str(warning.message),
                             "where": f"{Path(warning.filename).name}:{warning.lineno}"})
        per_corpus[corpus] = {
            "population": population,
            "column_boundaries": boundaries,
            "fold_protocol": folds,
            "saved_split_parity": parity,
            "fold_membership": audit.fold_membership(splits, len(task.y)),
        }

    overlap = None
    if "openhands" in tasks:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            overlap = audit.swegym_issue_overlap(tasks["openhands"].identities["group_key"])
            verify_pinned_fetches(names={"SWE-Gym"})
        for warning in caught:
            captured.append({"corpus": "swegym", "category": warning.category.__name__,
                             "message": str(warning.message),
                             "where": f"{Path(warning.filename).name}:{warning.lineno}"})

    return {
        "schema_version": "1.0.0",
        "generated_by": "tools/post_generalization_preflight.py",
        "declaration": {"file": audit.DECLARATION, "sections": ["C", "D"],
                        "frozen_utc": "2026-09-11T01:05:49Z"},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cleared": not findings,
        "findings": findings,
        "fits_of_the_declared_batch": 0,
        "fits_note": "section J's batch is 200 fits (2 corpora x 4 arms x 5 seeds x 5 folds) and "
                     "none of them is here. The ScienceWorld parity comparison section D requires "
                     "fits 100 estimators down each of the two fold paths; it produces no declared "
                     "quantity and its arm scores are withheld.",
        "checks": {
            "c_populations": "catchbench.post_generalization_audit.check_population, against the "
                             "counts section C froze",
            "c_scienceworld_identities": "_scienceworld_identities, replayed from the raw JSON "
                                         "because the loader drops task_name, var_num and the "
                                         "source file",
            "c_swegym_overlap": "swegym_issue_overlap, replayed from the raw parquet shard because "
                                "the adapter drops instance_id",
            "d_fold_protocol": "check_fold_protocol: coverage, both classes in every fold, groups "
                               "intact, one partition for all four arms",
            "d_saved_split_parity": "check_saved_split_parity against detection._cv_estimator_detail "
                                    f"on {PARITY_CORPUS}, where the two partitions agree by "
                                    "construction",
            "b_column_boundary": "check_column_boundaries, through "
                                 "detection._MixedSplineAUC.columns",
        },
        "tolerance": {
            "value": audit.PARITY_ATOL,
            "scope": "every ROC-AUC and out-of-fold probability compared between the two fold paths",
            "exact_comparisons": ["fold index arrays", "group membership of every fold",
                                  "the matched arm's column boundary",
                                  "replayed labels and step counts against the scored rows"],
            "declared": "in catchbench.post_generalization_audit.PARITY_ATOL, written into the "
                        "module before the first fit and before any comparison",
        },
        "frozen_before_the_run": {
            "declared_population": dict(audit.DECLARED_POPULATION),
            "fold_protocol": dict(audit.FOLD_PROTOCOL),
            "arms": dict(audit.ARM_ROLES),
            "linear_contrast": list(audit.LINEAR_CONTRAST),
            "matched_contrast": list(audit.MATCHED_CONTRAST),
            "estimands": list(audit.ESTIMANDS),
            "primary_estimand": {"corpus": audit.PRIMARY_ESTIMAND[0],
                                 "estimand": audit.PRIMARY_ESTIMAND[1]},
            "rng_labels": {f"{corpus}.{estimand}": label
                           for (corpus, estimand), label in sorted(audit.RNG_LABELS.items())},
            "rng_base_seed": audit.BOOTSTRAP_BASE_SEED,
            "bootstrap_draws": audit.BOOTSTRAP_DRAWS,
            "resampling_unit": dict(audit.RESAMPLING_UNIT),
            "substantial_threshold": audit.SUBSTANTIAL_EFFECT,
            "branches": list(audit.BRANCHES),
            "parity_tolerance": audit.PARITY_ATOL,
            "seeds": [int(seed) for seed in audit.SEEDS],
            "folds": audit.N_SPLITS,
            "spline": _spline_configuration(),
            "declared_fits": 200,
        },
        "fixed_by_this_implementation": {
            "section_g_row_overlap": "section G's first two rows overlap as written: "
                                     "'an informative replication or boundary' includes the "
                                     "'informative stable-positive boundary' the second row names, "
                                     "so a literal first-match reading leaves mixed-informative "
                                     "unreachable. classify_branch sends the boundary case to the "
                                     "row that names it and leaves the replication case to the "
                                     "first row, which makes the four rows exhaustive and mutually "
                                     "exclusive. Written before any of the six quantities existed.",
            "evidence_of_a_positive_L": "section G's substantial-attenuation condition asks for "
                                        "'evidence of a positive L' without giving it a test. "
                                        "corpus_reading applies the support-then-size order the "
                                        "rest of section G uses: a positive point whose section F "
                                        "interval excludes zero.",
            "parity_scope": "check_saved_split_parity compares all four arms rather than a "
                            "representative one, because the ColumnTransformer in the matched arm "
                            "is where a column-order difference between two paths would surface.",
            "parity_scores_withheld": "the comparison carries the differences between the two "
                                      "paths and not the four ScienceWorld arm scores, which are "
                                      "three of section E's six quantities.",
        },
        "swegym_overlap": overlap,
        "hub_revisions": revisions,
        "hub_revisions_are": "recorded, not enforced, for the two corpora absent from "
                             "catchbench.corpora.CORPUS_REVISIONS; section C's counts are what "
                             "catch a population that moved",
        "board_entrants_added": [],
        "rosters_unchanged": "detection._LOADERS, every board roster and live_streaming_methods() "
                             "are untouched; the nine boards, the 72 entrants and the 138-record "
                             "registry are what they were",
        "inputs": _inputs(declaration),
        "warnings": captured,
        "corpora": per_corpus,
    }


def _spline_configuration() -> dict:
    from catchbench import detection

    return dict(detection._SPLINE)


def _report(result: dict) -> None:
    print(f"\nPOST generalization audit :: preflight ({result['declaration']['file']}, "
          f"sections {', '.join(result['declaration']['sections'])})")
    print(f"  tolerance {result['tolerance']['value']:.1e}, declared "
          f"{result['tolerance']['declared']}")
    for corpus, record in result["corpora"].items():
        population = record["population"]
        observed = population["observed"]
        print(f"\n[{corpus}] {observed['n_runs']} runs "
              f"({observed['n_solved']} solved, {observed['n_failed']} failed), "
              f"declared {population['declared']['n_runs']}/"
              f"{population['declared']['n_solved']}/{population['declared']['n_failed']}")
        identities = population["identities"]
        if "n_distinct_group" in observed:
            print(f"  issue ids        : {observed['n_distinct_group']} distinct, "
                  f"multiplicity {observed['group_multiplicity']}, "
                  f"{observed['n_missing_group']} missing")
        if "n_distinct_task_var_pairs" in identities:
            print(f"  task identities  : {identities['n_distinct_task_var_pairs']} distinct "
                  f"(task_name, var_num) pairs over {identities['n_distinct_task_names']} task "
                  f"names")
            print(f"  source file      : {identities['source_file']!r}")
        boundary = record["column_boundaries"]
        print(f"  column boundary  : splined {len(boundary['splined_columns'])} of "
              f"{boundary['flatdep_shape'][1]}, passthrough "
              f"{len(boundary['passthrough_columns'])}")
        folds = record["fold_protocol"]
        print(f"  fold protocol    : {folds['protocol']}")
        print(f"  folds            : {folds['n_seeds']} seeds x {folds['n_splits']} folds, "
              f"smallest class count in any fold {folds['minimum_class_count_in_any_fold']}, "
              f"{folds['splits_compared']} splits compared across "
              f"{len(folds['layers_compared'])} layers")
        parity = record["saved_split_parity"]
        if parity is not None:
            print(f"  saved-split parity: {len(parity['arms_compared'])} arm(s) against "
                  f"{parity['compared_against'].rsplit('.', 1)[-1]}, "
                  f"{parity['arms_disagreeing']} disagreeing, max |difference| "
                  f"{parity['max_absolute_difference']:.3e}")
            for arm, row in sorted(parity["per_arm"].items()):
                print(f"    {arm:<28} fold {row['max_absolute_fold_auc_difference']:.3e}  "
                      f"oof {row['max_absolute_oof_difference']:.3e}  "
                      f"bit-identical {row['bit_identical_fold_auc'] and row['bit_identical_oof']}")
    overlap = result["swegym_overlap"]
    if overlap is not None:
        print(f"\n[swegym overlap] {overlap['n_distinct_swegym_issues']} scored SWE-Gym issues "
              f"against {overlap['n_distinct_openhands_issues']} OpenHands issues: "
              f"{overlap['n_shared_issues']} shared")
        for issue in overlap["shared_issues"]:
            print(f"    {issue}")
    if result["warnings"]:
        print(f"\n  {len(result['warnings'])} warning(s) captured:")
        for entry in result["warnings"]:
            print(f"    [{entry['corpus']}] {entry['category']} at {entry['where']}: "
                  f"{entry['message']}")
    else:
        print("\n  no warnings captured")
    if result["cleared"]:
        print("\npreflight cleared: every declared population reproduces exactly, every fold "
              "carries both classes,\nand the saved-index fold path reproduces "
              "_cv_estimator_detail within the declared tolerance")
        return
    print(f"\npreflight DID NOT clear: {len(result['findings'])} comparison(s) outside the declared "
          "tolerance")
    for finding in result["findings"]:
        print(f"  {finding}")
    print("\nThe declaration fixed the tolerance before the first fit. Widening it here is not a "
          "resolution;\nthe disagreement is the finding.")


def _write(path: Path, payload: dict) -> None:
    """Write JSON through a temporary sibling, so a run that dies mid-write leaves the old record."""
    import statistical_tests as st

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(st._jsonable(payload), indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = _parse_args()
    started = time.time()
    try:
        result = preflight(list(args.corpora), hub_revisions=not args.no_hub_revisions,
                           declaration=args.declaration)
    except Exception as error:  # noqa: BLE001 - a structural failure is the report
        print(f"\npreflight BLOCKED: {type(error).__name__}: {error}\n")
        traceback.print_exc()
        return 1
    result["elapsed_seconds"] = time.time() - started
    _write(args.output, result)
    _report(result)
    print(f"\nwrote {args.output}")
    return 0 if result["cleared"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
