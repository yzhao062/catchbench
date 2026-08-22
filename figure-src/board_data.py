"""Read README-figure data from the scored board instead of copying results.

The benchmark board is the source of truth for every plotted number.  Each accessor below names
the exact section header it expects, so a renamed section is a hard failure rather than an empty
figure.  The PRE figure also displays registered verdict words; those non-numeric annotations come
from ``tools/statistical_tests_results.json`` and are included in the figure provenance payload.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOARD = ROOT / "tests" / "golden" / "board.txt"
DEFAULT_STATS = ROOT / "tools" / "statistical_tests_results.json"

H_PRE_BY_SOURCE = "[PRE] pre_over_privilege :: F1 by source"
H_LIVE_PREFIX = (
    "LIVE streaming early-warning (ROC-AUC by prefix; "
    "t2d = earliest prefix with AUC>=0.70):"
)
H_DETECTION_SWE = "[POST] post_detection :: swegym"
H_DETECTION_TAU = "[POST] post_detection :: tau"
H_LIVE_SWE = "[LIVE] live_streaming :: swegym"
H_LIVE_TAU = "[LIVE] live_streaming :: tau"

FIGURE_IDS = ("board_live_prefix", "board_pre_source")

META_FIGURE = "CatchBench figure"
META_PAYLOAD = "CatchBench data"
META_DIGEST = "CatchBench data SHA256"
META_SOURCE = "CatchBench source"
SOURCE_DESCRIPTION = "tests/golden/board.txt"


class BoardDataError(RuntimeError):
    """The committed board cannot supply an exact figure input."""


def _sections(text: str) -> dict[str, list[str]]:
    """Split the board into ``{non-indented header: [indented lines]}``.

    Repeated headers intentionally share a list.  The LIVE prefix section occurs once per corpus.
    """

    out: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if line.startswith("  "):
            if current is not None:
                out.setdefault(current, []).append(line.strip())
            continue
        current = line.strip()
        out.setdefault(current, [])
    return out


def _load(board_path: Path | str = DEFAULT_BOARD) -> dict[str, list[str]]:
    path = Path(board_path)
    try:
        return _sections(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BoardDataError(f"cannot read board at {path}: {exc}") from exc


def section(header: str, board_path: Path | str = DEFAULT_BOARD) -> list[str]:
    sections = _load(board_path)
    if header not in sections:
        raise BoardDataError(
            f"{Path(board_path)} has no section {header!r}; the figure would plot nothing"
        )
    return sections[header]


def table(
    header: str, ncols: int, board_path: Path | str = DEFAULT_BOARD
) -> tuple[list[str], dict[str, list[str]]]:
    """Read a section with a column header and ``name + ncols cells`` rows.

    Method names may contain spaces, so cells are taken from the right.
    """

    lines = section(header, board_path)
    if not lines:
        raise BoardDataError(f"{Path(board_path)} section {header!r} is empty")
    heading = lines[0].split()
    if not heading or heading[0] != "method":
        raise BoardDataError(f"{Path(board_path)} section {header!r} has no method header")
    columns = heading[1:]
    if len(columns) != ncols:
        raise BoardDataError(
            f"{Path(board_path)} section {header!r} has {len(columns)} columns, "
            f"expected {ncols}: {columns!r}"
        )
    rows: dict[str, list[str]] = {}
    for line in lines[1:]:
        cells = line.split()
        if len(cells) <= ncols:
            continue
        values = cells[-ncols:]
        name = " ".join(cells[:-ncols])
        if name in rows:
            raise BoardDataError(
                f"{Path(board_path)} section {header!r} repeats row {name!r}"
            )
        rows[name] = values
    if not rows:
        raise BoardDataError(f"{Path(board_path)} section {header!r} has no data rows")
    return columns, rows


def floats(
    header: str, ncols: int, board_path: Path | str = DEFAULT_BOARD
) -> tuple[list[str], dict[str, list[float]]]:
    columns, raw_rows = table(header, ncols, board_path)
    rows: dict[str, list[float]] = {}
    for name, values in raw_rows.items():
        try:
            rows[name] = [float(value) for value in values]
        except ValueError:
            continue
    if not rows:
        raise BoardDataError(f"{Path(board_path)} section {header!r} has no numeric rows")
    return columns, rows


def pre_by_source(
    board_path: Path | str = DEFAULT_BOARD,
) -> tuple[list[str], dict[str, list[float]]]:
    """Per-source F1 values for every PRE entrant, keyed by board row name."""

    columns, rows = floats(H_PRE_BY_SOURCE, 7, board_path)
    expected = ["crewai", "n8n", "mcp", "injecagent", "sweagent", "synthetic", "overall"]
    if columns != expected:
        raise BoardDataError(
            f"{Path(board_path)} section {H_PRE_BY_SOURCE!r} columns changed: {columns!r}"
        )
    return columns, rows


def _live_prefix_summary(
    corpus: str, board_path: Path | str = DEFAULT_BOARD
) -> dict[str, float]:
    """The LIVE full-prefix summary, used to tell the two repeated prefix tables apart by content.

    This is a helper for ``live_prefix`` and not a board accessor. It was called ``detection`` and
    renamed: the manuscript's copy of this module has a ``detection`` that reads the POST detection
    block, so one name meant two different sections across the two figure pipelines, and the next
    person to want a detection figure here would have drawn LIVE numbers under a POST caption. The
    real accessor is below.
    """

    headers = {"swegym": H_LIVE_SWE, "tau": H_LIVE_TAU}
    if corpus not in headers:
        raise BoardDataError(f"unknown LIVE corpus {corpus!r}")
    _, rows = floats(headers[corpus], 1, board_path)
    return {name: values[0] for name, values in rows.items()}


def detection(
    corpus: str, board_path: Path | str = DEFAULT_BOARD
) -> dict[str, float]:
    """The POST detection board for one corpus, as ``{method: ROC-AUC}``.

    Same section and same semantics as the manuscript pipeline's ``detection``, so a figure drawn
    here and a figure drawn there read one board the same way.
    """

    headers = {"swegym": H_DETECTION_SWE, "tau": H_DETECTION_TAU}
    if corpus not in headers:
        raise BoardDataError(f"unknown detection corpus {corpus!r}")
    _, rows = floats(headers[corpus], 1, board_path)
    return {name: values[0] for name, values in rows.items()}


def _prefix_table_matches(table_rows: dict[str, list[float]], summary: dict[str, float]) -> bool:
    if set(table_rows) != set(summary):
        return False
    if "random" not in table_rows or "random" not in summary:
        return False
    # The detailed 100% column is scored on prefix features and is not the summary column above.
    # The random floor is corpus-specific and constant by construction, so it identifies the two
    # repeated tables without copying either floor into this source file.
    return all(math.isclose(value, summary["random"], abs_tol=5e-7)
               for value in table_rows["random"])


def live_prefix(
    board_path: Path | str = DEFAULT_BOARD,
) -> tuple[list[int], float, dict[str, dict[str, list[float]]]]:
    """Return prefix percentages, the registered AUC threshold, and both corpus tables.

    The detailed section header repeats without naming its corpus.  Instead of hard-coding a result
    as an order sentinel, this accessor matches each table's constant random floor to the named LIVE
    corpus summary elsewhere in the same board.
    """

    threshold_match = re.search(r"AUC>=([0-9.]+)", H_LIVE_PREFIX)
    if threshold_match is None:
        raise BoardDataError(f"cannot read the LIVE threshold from {H_LIVE_PREFIX!r}")
    threshold = float(threshold_match.group(1))

    tables: list[dict[str, list[float]]] = []
    prefixes: list[int] | None = None
    current: dict[str, list[float]] | None = None
    for line in section(H_LIVE_PREFIX, board_path):
        cells = line.split()
        if cells and cells[0] == "method":
            if len(cells) < 3 or cells[-1] != "t2d":
                raise BoardDataError(f"unexpected LIVE prefix header: {line!r}")
            parsed: list[int] = []
            for label in cells[1:-1]:
                match = re.fullmatch(r"(\d+)%", label)
                if match is None:
                    raise BoardDataError(f"unexpected LIVE prefix column {label!r}")
                parsed.append(int(match.group(1)))
            if prefixes is None:
                prefixes = parsed
            elif prefixes != parsed:
                raise BoardDataError("the repeated LIVE prefix tables use different columns")
            current = {}
            tables.append(current)
            continue
        if current is None or prefixes is None or len(cells) <= len(prefixes):
            continue
        name = " ".join(cells[: -(len(prefixes) + 1)])
        values = cells[-(len(prefixes) + 1):-1]
        try:
            parsed_values = [float(value) for value in values]
        except ValueError:
            continue
        if not name:
            continue
        if name in current:
            raise BoardDataError(f"the LIVE prefix table repeats row {name!r}")
        current[name] = parsed_values

    if prefixes is None or len(tables) != 2 or any(not rows for rows in tables):
        raise BoardDataError(
            f"{Path(board_path)}: expected two populated LIVE prefix tables, found {len(tables)}"
        )

    summaries = {corpus: _live_prefix_summary(corpus, board_path)
                 for corpus in ("swegym", "tau")}
    resolved: dict[str, dict[str, list[float]]] = {}
    used: set[int] = set()
    for corpus, summary in summaries.items():
        matches = [index for index, rows in enumerate(tables)
                   if _prefix_table_matches(rows, summary)]
        if len(matches) != 1:
            raise BoardDataError(
                f"{Path(board_path)}: LIVE prefix table for {corpus!r} matched {len(matches)} "
                "full-prefix summaries"
            )
        index = matches[0]
        if index in used:
            raise BoardDataError("one LIVE prefix table matched both corpora")
        used.add(index)
        resolved[corpus] = tables[index]
    return prefixes, threshold, resolved


def _pre_verdicts(stats_path: Path | str = DEFAULT_STATS) -> dict[str, str]:
    path = Path(stats_path)
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BoardDataError(f"cannot read registered verdicts at {path}: {exc}") from exc
    claims = root.get("claims")
    if not isinstance(claims, list):
        raise BoardDataError(f"{path} has no claims list")
    by_id: dict[str, dict[str, object]] = {}
    for claim in claims:
        if not isinstance(claim, dict) or not isinstance(claim.get("id"), str):
            raise BoardDataError(f"{path} carries a claim without an id")
        claim_id = claim["id"]
        if claim_id in by_id:
            raise BoardDataError(f"{path} repeats claim {claim_id!r}")
        by_id[claim_id] = claim

    verdicts: dict[str, str] = {}
    for source in ("crewai", "n8n", "mcp", "injecagent", "sweagent", "synthetic"):
        claim_id = f"pre.source.{source}.best.vs.flag_all"
        if claim_id not in by_id:
            raise BoardDataError(f"{path} has no registered claim {claim_id!r}")
        raw = by_id[claim_id].get("verdict")
        if raw == "separates_as_stated":
            verdicts[source] = "separates"
        elif raw == "does_not_separate":
            verdicts[source] = "unresolved"
        else:
            raise BoardDataError(f"{path} claim {claim_id!r} has unknown verdict {raw!r}")
    return verdicts


def figure_payload(
    figure_id: str,
    board_path: Path | str = DEFAULT_BOARD,
    stats_path: Path | str = DEFAULT_STATS,
) -> dict[str, object]:
    """The complete semantic input used by one committed README figure."""

    if figure_id == "board_live_prefix":
        prefixes, threshold, corpora = live_prefix(board_path)
        return {
            "figure": figure_id,
            "metric": "ROC-AUC",
            "prefixes": prefixes,
            "threshold": threshold,
            "corpora": corpora,
        }
    if figure_id == "board_pre_source":
        columns, rows = pre_by_source(board_path)
        return {
            "figure": figure_id,
            "metric": "F1",
            "columns": columns,
            "rows": rows,
            "registered_verdicts": _pre_verdicts(stats_path),
        }
    raise BoardDataError(f"unknown README figure {figure_id!r}")


def canonical_payload(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True,
                      separators=(",", ":"))


def payload_digest(payload: dict[str, object]) -> str:
    return hashlib.sha256(canonical_payload(payload).encode("utf-8")).hexdigest()
