"""Fail when a committed README figure's semantic inputs differ from the scored board."""

from __future__ import annotations

import argparse
import importlib.util
import struct
import sys
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_SOURCE = ROOT / "figure-src" / "board_data.py"
DEFAULT_BOARD = ROOT / "tests" / "golden" / "board.txt"
DEFAULT_STATS = ROOT / "tools" / "statistical_tests_results.json"
DEFAULT_ASSETS = ROOT / "assets"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _load_board_data():
    spec = importlib.util.spec_from_file_location("readme_figure_board_data", FIGURE_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {FIGURE_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    # Register before executing: board_data uses postponed annotations, and @dataclass
    # resolves them through sys.modules[cls.__module__]. A by-path load that skips this
    # step raises AttributeError on None inside dataclasses.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bd = _load_board_data()


def png_metadata(path: Path) -> tuple[dict[str, str], int, int, int, bool]:
    """Read checked tEXt metadata, the IHDR, and whether a tRNS chunk is present.

    Opacity cannot be read off the colour type alone. The PNG specification defines tRNS
    for colour types 0, 2, and 3, so a truecolor asset can be transparent while looking
    safe by its type, which is why the chunk is reported rather than skipped.
    """

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not raw.startswith(PNG_SIGNATURE):
        raise ValueError(f"{path} is not a PNG")

    offset = len(PNG_SIGNATURE)
    metadata: dict[str, str] = {}
    width = height = color_type = -1
    saw_end = False
    has_transparency = False
    while offset < len(raw):
        if offset + 12 > len(raw):
            raise ValueError(f"{path} has a truncated PNG chunk")
        length = struct.unpack(">I", raw[offset:offset + 4])[0]
        chunk_type = raw[offset + 4:offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if crc_end > len(raw):
            raise ValueError(f"{path} has a truncated {chunk_type!r} chunk")
        data = raw[data_start:data_end]
        recorded_crc = struct.unpack(">I", raw[data_end:crc_end])[0]
        actual_crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        if recorded_crc != actual_crc:
            raise ValueError(f"{path} has a corrupt {chunk_type!r} chunk")
        if chunk_type == b"IHDR":
            if length != 13:
                raise ValueError(f"{path} has an invalid IHDR")
            width, height, _, color_type, _, _, _ = struct.unpack(">IIBBBBB", data)
        elif chunk_type == b"tEXt":
            keyword, separator, value = data.partition(b"\0")
            if not separator:
                raise ValueError(f"{path} has a malformed tEXt chunk")
            key = keyword.decode("latin-1")
            if key in metadata:
                raise ValueError(f"{path} repeats PNG metadata key {key!r}")
            metadata[key] = value.decode("latin-1")
        elif chunk_type == b"tRNS":
            has_transparency = True
        elif chunk_type == b"IEND":
            saw_end = True
            if crc_end != len(raw):
                raise ValueError(f"{path} has bytes after IEND")
            break
        offset = crc_end
    if not saw_end or width < 1 or height < 1:
        raise ValueError(f"{path} is an incomplete PNG")
    return metadata, width, height, color_type, has_transparency


def check_assets(board: Path = DEFAULT_BOARD, stats: Path = DEFAULT_STATS,
                 assets: Path = DEFAULT_ASSETS) -> list[str]:
    problems: list[str] = []
    for figure_id in bd.FIGURE_IDS:
        path = assets / f"{figure_id}.png"
        try:
            payload = bd.figure_payload(figure_id, board, stats)
            expected_payload = bd.canonical_payload(payload)
            expected_digest = bd.payload_digest(payload)
        except Exception as exc:
            problems.append(f"{figure_id}: cannot derive current semantic inputs: {exc}")
            continue
        try:
            metadata, width, height, color_type, has_transparency = png_metadata(path)
        except ValueError as exc:
            problems.append(str(exc))
            continue

        if metadata.get(bd.META_FIGURE) != figure_id:
            problems.append(f"{path}: missing or wrong {bd.META_FIGURE!r} metadata")
        found_payload = metadata.get(bd.META_PAYLOAD)
        if found_payload != expected_payload:
            problems.append(
                f"{path}: stale semantic payload; regenerate from {board}"
            )
        found_digest = metadata.get(bd.META_DIGEST)
        if found_digest != expected_digest:
            problems.append(
                f"{path}: data digest is {found_digest!r}, expected {expected_digest}"
            )
        expected_source = bd.SOURCE_DESCRIPTION
        if figure_id == "board_pre_source":
            expected_source += "; tools/statistical_tests_results.json (verdict words only)"
        elif figure_id == "board_live_prefix":
            expected_source += "; tools/statistical_tests_results.json (threshold verdicts and estimate sides)"
        elif figure_id == "hero-lifecycle":
            expected_source += "; README.md (phrase checks only)"
        # catchbench_data_at_a_glance reads the board alone, so the bare description stands.
        if metadata.get(bd.META_SOURCE) != expected_source:
            problems.append(f"{path}: missing or wrong {bd.META_SOURCE!r} metadata")
        if color_type in (4, 6):
            problems.append(
                f"{path}: PNG color type {color_type} carries an alpha channel; README figures "
                "must be opaque for GitHub dark mode"
            )
        elif has_transparency:
            problems.append(
                f"{path}: PNG carries a tRNS chunk; README figures must be opaque for GitHub "
                "dark mode"
            )
        if width < 700 or height < 400:
            problems.append(f"{path}: {width}x{height} is below the web-resolution floor")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", type=Path, default=DEFAULT_BOARD)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    args = parser.parse_args(argv)
    problems = check_assets(args.board, args.stats, args.assets)
    for problem in problems:
        print(f"ERROR: {problem}", file=sys.stderr)
    if problems:
        print(f"README figure check failed: {len(problems)} problem(s)", file=sys.stderr)
        return 1
    for figure_id in bd.FIGURE_IDS:
        payload = bd.figure_payload(figure_id, args.board, args.stats)
        metadata, width, height, _, _ = png_metadata(args.assets / f"{figure_id}.png")
        print(f"OK {figure_id}: {width}x{height} RGB, data {metadata[bd.META_DIGEST]}")
        assert metadata[bd.META_DIGEST] == bd.payload_digest(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
