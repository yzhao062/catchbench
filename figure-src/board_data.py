"""Read README-figure data from the scored board instead of copying results.

The benchmark board is the source of truth for every plotted number.  Each accessor below names
the exact section header it expects, so a renamed section is a hard failure rather than an empty
figure.  The PRE and LIVE figures also display registered verdicts; those non-numeric annotations
come from ``tools/statistical_tests_results.json`` and are included in the figure provenance payload.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
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

BOARD_HEADER = re.compile(r"^\[(PRE|LIVE|POST)\]\s+(\S+)\s+::\s+(\S+)$")
METHOD_ROW = re.compile(r"^\s{2}(.+?)\s{2,}[-+]?(?:\d+(?:\.\d*)?|\.\d+)")

# Blocks the runner scores and prints that are not part of the competitive arena. Gold v2 is an
# admissibility diagnostic: its rows are a matched floor, six construction controls, and two
# semantic oracles, and the registry declares no contrast over it. Counting it would put
# construction controls into the entrant total and place a board that establishes nothing beside
# boards that do. It joins the arena when it carries registered contrasts and competitive entrants,
# at which point this set, ``_BOARDS`` in tools/emit_boards_table.py, and the paper's board and
# entrant counts all move together. Removing it here alone would produce a board with no row.
#
# The identity is the whole header, not the board id. An id-only set excludes anything that
# reuses the id, so a block printed as ``[POST] gold_v2_namedvalue :: some_other_corpus`` would
# be dropped from every count without anyone deciding that. Two blocks already share the id
# ``pre_over_privilege``, so id collision is a shape this board actually has.
#
# What this set does not reach: ``check_partition`` attributes a contrast to a board through its
# family, so a Gold v2 contrast filed into a family an arena board already covers is attributed to
# that board while the block stays excluded here. Closing that needs per-claim board ownership in
# the registry, which the schema does not carry. ``test_every_printed_block_is_accounted_for``
# below covers the case this set is actually for, a new block appearing with nobody deciding
# whether it competes.
DIAGNOSTIC_HEADERS = frozenset({"[POST] gold_v2_namedvalue :: tau-bench-gold-v2"})

HERO_HEADER = re.compile(r"^\[(PRE|LIVE|POST)\]\s+([^:]+?)\s+::\s+(.+)$", re.MULTILINE)
HERO_PREFIX_DECLARATION = "streaming prefixes [25%, 50%, 75%, 100%]"

FIGURE_IDS = ("board_live_prefix", "board_pre_source", "catchbench_data_at_a_glance",
              "hero-lifecycle")

META_FIGURE = "CatchBench figure"
META_PAYLOAD = "CatchBench data"
META_DIGEST = "CatchBench data SHA256"
META_SOURCE = "CatchBench source"
SOURCE_DESCRIPTION = "tests/golden/board.txt"


class BoardDataError(RuntimeError):
    """The committed board cannot supply an exact figure input."""


@dataclass(frozen=True)
class BoardFacts:
    """The scale and composition counts the README glance figure prints."""

    pre_total: int
    pre_sources: dict[str, int]
    trace_runs: dict[str, int]
    gold_injections: int
    gold_stale: int
    gold_dropped: int
    board_count: int
    entrant_count: int

    @property
    def trace_total(self) -> int:
        return sum(self.trace_runs.values())


def parse_board(path: Path | str = DEFAULT_BOARD) -> BoardFacts:
    """Read the glance figure's counts from the board's own summary lines.

    This lives beside the other accessors, and uses only the standard library, so
    ``tools/check_readme_figures.py`` can gate the glance asset without importing
    matplotlib or Pillow.
    """

    lines = Path(path).read_text(encoding="utf-8").splitlines()

    pre_match = next(
        (re.fullmatch(r"PRE over_privilege: (\d+) configs across (\d+) corpora (\{.*\})", line)
         for line in lines if line.startswith("PRE over_privilege:")),
        None,
    )
    if pre_match is None:
        raise BoardDataError("board.txt has no parseable PRE corpus summary")
    pre_total = int(pre_match.group(1))
    declared_source_count = int(pre_match.group(2))
    pre_sources = ast.literal_eval(pre_match.group(3))
    if not isinstance(pre_sources, dict) or not all(
        isinstance(key, str) and isinstance(value, int) for key, value in pre_sources.items()
    ):
        raise BoardDataError("PRE corpus summary is not a string-to-integer mapping")
    if len(pre_sources) != declared_source_count or sum(pre_sources.values()) != pre_total:
        raise BoardDataError("PRE corpus counts do not reconcile with the declared totals")

    trace_patterns = {
        "Who&When": re.compile(r"^Who&When: (\d+) failed runs,"),
        "SWE-Gym": re.compile(r"^swegym: (\d+) runs \("),
        "tau-bench": re.compile(r"^tau: (\d+) runs \("),
    }
    trace_runs: dict[str, int] = {}
    for label, pattern in trace_patterns.items():
        match = next((pattern.match(line) for line in lines if pattern.match(line)), None)
        if match is None:
            raise BoardDataError(f"board.txt has no parseable trace summary for {label}")
        trace_runs[label] = int(match.group(1))

    gold_pattern = re.compile(
        r"^swegym-gold: (\d+) clean SWE-Gym runs, one injected fault each "
        r"\((\d+) stale-state, (\d+) dropped-grounding\),"
    )
    gold_match = next((gold_pattern.match(line) for line in lines if gold_pattern.match(line)), None)
    if gold_match is None:
        raise BoardDataError("board.txt has no parseable Gold summary")
    gold_injections, gold_stale, gold_dropped = map(int, gold_match.groups())
    if gold_stale + gold_dropped != gold_injections:
        raise BoardDataError("Gold fault counts do not reconcile with the Gold total")

    headers = [(index, match) for index, line in enumerate(lines)
               if (match := BOARD_HEADER.fullmatch(line))
               and match.group(0).strip() not in DIAGNOSTIC_HEADERS]
    entrants: set[str] = set()
    for index, _ in headers:
        for row in lines[index + 2:]:
            if not row.strip():
                break
            method_match = METHOD_ROW.match(row)
            if method_match:
                entrants.add(method_match.group(1).strip())

    facts = BoardFacts(
        pre_total=pre_total,
        pre_sources=pre_sources,
        trace_runs=trace_runs,
        gold_injections=gold_injections,
        gold_stale=gold_stale,
        gold_dropped=gold_dropped,
        board_count=len(headers),
        entrant_count=len(entrants),
    )
    expected = {
        "pre_total": 1187,
        "pre_source_count": 6,
        "trace_total": 1162,
        "trace_source_count": 3,
        "gold_injections": 188,
        "board_count": 9,
        "entrant_count": 72,
    }
    observed = {
        "pre_total": facts.pre_total,
        "pre_source_count": len(facts.pre_sources),
        "trace_total": facts.trace_total,
        "trace_source_count": len(facts.trace_runs),
        "gold_injections": facts.gold_injections,
        "board_count": facts.board_count,
        "entrant_count": facts.entrant_count,
    }
    if observed != expected:
        raise BoardDataError(f"board.txt changed; review the figure claims: {observed!r}")
    return facts


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


def _registered_claims(
    stats_path: Path | str = DEFAULT_STATS,
) -> dict[str, dict[str, object]]:
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
    return by_id


def _pre_verdicts(stats_path: Path | str = DEFAULT_STATS) -> dict[str, str]:
    path = Path(stats_path)
    by_id = _registered_claims(path)

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


def _live_threshold_claims(
    prefixes: list[int],
    corpora: dict[str, dict[str, list[float]]],
    threshold: float,
    stats_path: Path | str = DEFAULT_STATS,
) -> dict[str, dict[str, list[dict[str, str] | None]]]:
    """Registered method-versus-0.70 claims, with absent cells left as point estimates.

    Each cell carries which side of the bar its estimate falls on. The bar families are two-sided,
    so a separation says the cell differs from 0.70 without saying which way, and a caller that
    wants to name a direction has to read it from the estimate. Carrying it here rather than in the
    caller puts it inside the figure payload, so a figure whose subtitle names a side goes stale
    when a side moves. A subtitle that hardcoded the side was how a nine-cell two-sided result was
    rendered as nine cells above the bar, with the asset checker passing because the image matched
    the generator that produced it.
    """

    path = Path(stats_path)
    by_id = _registered_claims(path)
    corpus_ids = {"swegym": "swe", "tau": "tau"}
    threshold_families = {"swegym": "live_swegym_threshold_auc",
                          "tau": "live_tau_threshold_auc"}
    out: dict[str, dict[str, list[dict[str, str] | None]]] = {}
    for corpus, rows in corpora.items():
        corpus_id = corpus_ids[corpus]
        expected_family = threshold_families[corpus]
        out[corpus] = {}
        for method in rows:
            cells: list[dict[str, str] | None] = []
            for prefix in prefixes:
                claim_id = f"live.{corpus_id}.bar.{prefix}.{method}"
                claim = by_id.get(claim_id)
                if claim is None:
                    cells.append(None)
                    continue
                estimate = claim.get("estimate")
                if (claim.get("family") != expected_family
                        or claim.get("metric") != "roc_auc"
                        or not isinstance(estimate, dict)
                        or estimate.get("a_name") != method
                        or estimate.get("b_name") != "fixed bar"
                        or not math.isclose(float(estimate.get("b", math.nan)), threshold,
                                            abs_tol=1e-12)):
                    raise BoardDataError(
                        f"{path} claim {claim_id!r} is not the expected LIVE threshold contrast"
                    )
                raw = claim.get("verdict")
                if raw == "separates_as_stated":
                    verdict = "separates"
                elif raw == "does_not_separate":
                    verdict = "unresolved"
                else:
                    raise BoardDataError(
                        f"{path} claim {claim_id!r} has unknown verdict {raw!r}"
                    )
                difference = estimate.get("difference_a_minus_b")
                if not isinstance(difference, (int, float)):
                    raise BoardDataError(
                        f"{path} claim {claim_id!r} carries no numeric difference"
                    )
                cells.append({"claim_id": claim_id, "family": expected_family,
                              "verdict": verdict,
                              "side": "above" if difference > 0 else "below"})
            out[corpus][method] = cells
    return out


def scored_blocks(board_path: Path | str = DEFAULT_BOARD) -> list[str]:
    """Every scored board block the lifecycle hero counts, as ``phase|board|corpus``.

    The PRE by-source breakdown is excluded. It decomposes the one PRE board rather than
    being a board of its own, and the hero prints board counts. Returning the identifiers
    rather than only the counts is deliberate: a renamed corpus then moves the payload, so
    the committed hero goes stale instead of staying plausible at the same three totals.
    """

    board = Path(board_path).read_text(encoding="utf-8")
    return sorted(
        f"{phase}|{board_id.strip()}|{corpus.strip()}"
        for phase, board_id, corpus in HERO_HEADER.findall(board)
        if corpus.strip() != "F1 by source"
        and f"[{phase}] {board_id.strip()} :: {corpus.strip()}" not in DIAGNOSTIC_HEADERS
    )


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
            "registered_threshold_claims": _live_threshold_claims(
                prefixes, corpora, threshold, stats_path
            ),
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
    if figure_id == "catchbench_data_at_a_glance":
        facts = parse_board(board_path)
        return {"figure": figure_id} | asdict(facts) | {"trace_total": facts.trace_total}
    if figure_id == "hero-lifecycle":
        blocks = scored_blocks(board_path)
        return {
            "figure": figure_id,
            "blocks": blocks,
            "counts": {phase: sum(1 for block in blocks if block.startswith(phase + "|"))
                       for phase in ("PRE", "LIVE", "POST")},
            "declares_prefixes":
                HERO_PREFIX_DECLARATION in Path(board_path).read_text(encoding="utf-8"),
        }
    raise BoardDataError(f"unknown README figure {figure_id!r}")


def canonical_payload(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True,
                      separators=(",", ":"))


def payload_digest(payload: dict[str, object]) -> str:
    return hashlib.sha256(canonical_payload(payload).encode("utf-8")).hexdigest()
