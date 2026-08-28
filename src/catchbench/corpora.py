"""Immutable Hugging Face revisions used by the POST and LIVE boards."""
from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CorpusRevision:
    name: str
    repo_id: str
    revision: str


CORPUS_REVISIONS = (
    CorpusRevision(
        "Who&When",
        "Kevin355/Who_and_When",
        "59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c",
    ),
    CorpusRevision(
        "SWE-Gym",
        "SWE-Gym/OpenHands-Sampled-Trajectories",
        "baf3a4e4bff514d48ddc08a93a2ade5c126212c7",
    ),
    CorpusRevision(
        "tau-bench",
        "AgentSuite/tau-bench-trajectories",
        "382e57d1784b55c5155f4ef394ef48f1c747a287",
    ),
)

_BY_REPO = {corpus.repo_id: corpus for corpus in CORPUS_REVISIONS}
_PINNED_FETCHES: set[tuple[str, str]] = set()
_PATCHED = False


class CorpusRevisionError(RuntimeError):
    """The runner cannot prove that it will score the recorded corpus revisions."""


def _selected_corpora(names: set[str] | None = None) -> tuple[CorpusRevision, ...]:
    if names is None:
        return CORPUS_REVISIONS
    unknown = names - {corpus.name for corpus in CORPUS_REVISIONS}
    if unknown:
        raise ValueError(f"unknown corpus name(s): {', '.join(sorted(unknown))}")
    return tuple(corpus for corpus in CORPUS_REVISIONS if corpus.name in names)


def verify_corpus_heads(api: Any = None, names: set[str] | None = None) -> dict[str, str]:
    """Resolve selected dataset heads and refuse to score if one differs from the board record."""
    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover - exercised by minimal installations
            raise CorpusRevisionError(
                "Corpus revision preflight needs huggingface_hub; install the dev-seed or full extra."
            ) from exc
        api = HfApi()

    resolved: dict[str, str] = {}
    errors = []
    for corpus in _selected_corpora(names):
        try:
            actual = str(api.dataset_info(corpus.repo_id, revision="main").sha)
        except Exception as exc:
            errors.append(
                f"{corpus.name} ({corpus.repo_id}): could not resolve main ({exc}); "
                f"recorded={corpus.revision}"
            )
            continue
        resolved[corpus.name] = actual
        if actual != corpus.revision:
            errors.append(
                f"{corpus.name} ({corpus.repo_id}): main={actual}; recorded={corpus.revision}"
            )
    if errors:
        raise CorpusRevisionError(
            "Corpus revision preflight failed; refusing to score an unrecorded population:\n  "
            + "\n  ".join(errors)
        )
    return resolved


def revision_header(revisions: dict[str, str] | None = None) -> str:
    """One board-header line recording the exact population revisions being scored."""
    revisions = revisions or {corpus.name: corpus.revision for corpus in CORPUS_REVISIONS}
    cells = [
        f"{corpus.name}={revisions[corpus.name]}"
        for corpus in CORPUS_REVISIONS
        if corpus.name in revisions
    ]
    return "Corpus revisions :: " + " | ".join(cells)


def _pin_hub_call(name: str, function):
    @functools.wraps(function)
    def pinned(*args, **kwargs):
        import inspect

        bound = inspect.signature(function).bind_partial(*args, **kwargs)
        repo_id = bound.arguments.get("repo_id")
        corpus = _BY_REPO.get(repo_id)
        if corpus is None:
            return function(*args, **kwargs)
        requested = bound.arguments.get("revision")
        if requested not in (None, "main", corpus.revision):
            raise CorpusRevisionError(
                f"{name} requested {repo_id}@{requested}, but the board records {corpus.revision}"
            )
        bound.arguments["revision"] = corpus.revision
        result = function(*bound.args, **bound.kwargs)
        _PINNED_FETCHES.add((repo_id, name))
        return result

    return pinned


def install_hub_revision_pins() -> None:
    """Force GRADE's Hub calls through the recorded revisions before its modules are imported."""
    global _PATCHED
    if _PATCHED:
        return
    try:
        import huggingface_hub
    except ImportError:  # the actionable error is emitted by the preflight or a GRADE loader
        return
    for name in ("snapshot_download", "hf_hub_download", "list_repo_files"):
        function = getattr(huggingface_hub, name)
        setattr(huggingface_hub, name, _pin_hub_call(name, function))
    _PATCHED = True


def _whoandwhen_cached_revisions() -> set[str]:
    """Read the commit recorded by snapshot_download's local-dir metadata."""
    try:
        import agent_graph_characterization as whoandwhen
    except ImportError:
        return set()
    metadata_root = Path(whoandwhen.CACHE).parent / ".cache" / "huggingface" / "download"
    revisions = set()
    for path in metadata_root.rglob("*.metadata"):
        lines = path.read_text(encoding="utf-8").splitlines()
        if lines:
            revisions.add(lines[0])
    return revisions


def verify_pinned_fetches(names: set[str] | None = None) -> None:
    """Check that each selected loader used a pinned Hub call or a verified local snapshot."""
    errors = []
    for corpus in _selected_corpora(names):
        calls = {name for repo_id, name in _PINNED_FETCHES if repo_id == corpus.repo_id}
        if corpus.name == "Who&When" and not calls:
            cached = _whoandwhen_cached_revisions()
            if cached == {corpus.revision}:
                continue
            detail = ", ".join(sorted(cached)) if cached else "no snapshot metadata"
            errors.append(f"Who&When local cache is not verified at {corpus.revision} ({detail})")
        elif "hf_hub_download" not in calls and "snapshot_download" not in calls:
            errors.append(
                f"{corpus.name} loader made no observed pinned download for {corpus.repo_id}; "
                "GRADE may have changed its fetch path"
            )
    if errors:
        raise CorpusRevisionError(
            "Corpus fetch verification failed; refusing to print scores:\n  " + "\n  ".join(errors)
        )
