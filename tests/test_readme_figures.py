"""The README-image gate must detect stale data and stale verdict annotations."""

from __future__ import annotations

import json
import re
import struct
import sys
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import check_readme_figures as checker  # noqa: E402


def test_committed_figures_match_their_semantic_inputs():
    assert checker.check_assets() == []


def _committed_png_text(name: str) -> dict[str, str]:
    metadata, *_ = checker.png_metadata(checker.DEFAULT_ASSETS / name)
    return metadata


def test_no_committed_asset_leaks_an_absolute_path():
    """A figure rendered in a scratch clone must not carry that path into the repo.

    The glance figure shipped one once: it recorded ``board_path.resolve()``, so the
    committed PNG named a throwaway working tree. The check is over every asset rather
    than that one figure, because the defect is a property of writing any local path
    into metadata, not of that generator.
    """
    offenders = []
    for path in sorted(checker.DEFAULT_ASSETS.glob("*.png")):
        for key, value in _committed_png_text(path.name).items():
            text = str(value)
            if re.search(r"[A-Za-z]:[\\/]|(?<![\w.])/(?:home|Users|tmp)/", text):
                offenders.append(f"{path.name}: {key} = {text[:120]}")
    assert offenders == [], offenders


def test_the_glance_figure_is_not_stale_against_the_board():
    """The scale figure is board-derived, so it can go stale even though it self-checks.

    Its generator refuses to render when the board stops matching the numbers the
    figure claims, which prevents rendering a wrong image. It cannot prevent a
    committed image from outliving a board change, which is what this covers.
    """
    sys.path.insert(0, str(ROOT / "figure-src"))
    import board_data as bd  # noqa: E402

    payload = bd.figure_payload("catchbench_data_at_a_glance", checker.DEFAULT_BOARD)
    metadata = _committed_png_text("catchbench_data_at_a_glance.png")

    assert metadata.get(bd.META_FIGURE) == "catchbench_data_at_a_glance"
    assert metadata.get(bd.META_SOURCE) == bd.SOURCE_DESCRIPTION
    assert metadata.get(bd.META_PAYLOAD) == bd.canonical_payload(payload)
    assert metadata.get(bd.META_DIGEST) == bd.payload_digest(payload)


def test_a_moved_board_number_makes_the_live_figure_stale(tmp_path):
    text = checker.DEFAULT_BOARD.read_text(encoding="utf-8")
    old = "  full                       0.813   0.816   0.826   0.819     25%"
    new = "  full                       0.812   0.816   0.826   0.819     25%"
    assert text.count(old) == 1
    board = tmp_path / "board.txt"
    board.write_text(text.replace(old, new), encoding="utf-8")

    problems = checker.check_assets(board=board)

    assert any("board_live_prefix.png: stale semantic payload" in problem
               for problem in problems)


def _stats_with_one_verdict_flipped(tmp_path, claim_id: str) -> Path:
    root = json.loads(checker.DEFAULT_STATS.read_text(encoding="utf-8"))
    claim = next(item for item in root["claims"] if item["id"] == claim_id)
    assert claim["verdict"] == "does_not_separate"
    claim["verdict"] = "separates_as_stated"
    stats = tmp_path / "statistical_tests_results.json"
    stats.write_text(json.dumps(root), encoding="utf-8")
    return stats


def test_a_changed_pre_verdict_makes_only_the_pre_figure_stale(tmp_path):
    """One flipped PRE verdict must reach the PRE figure and leave the LIVE one alone.

    The two figures annotate different claim families out of the same file. A test that
    flipped one of each at once would still pass if those dependencies became crossed,
    so each figure gets its own mutation and its own sibling assertion.
    """
    stats = _stats_with_one_verdict_flipped(tmp_path, "pre.source.mcp.best.vs.flag_all")

    problems = checker.check_assets(stats=stats)

    assert any("board_pre_source.png: stale semantic payload" in problem
               for problem in problems)
    assert not any("board_live_prefix.png: stale semantic payload" in problem
                   for problem in problems)


def test_a_changed_live_verdict_makes_only_the_live_figure_stale(tmp_path):
    stats = _stats_with_one_verdict_flipped(tmp_path, "live.tau.bar.100.full")

    problems = checker.check_assets(stats=stats)

    assert any("board_live_prefix.png: stale semantic payload" in problem
               for problem in problems)
    assert not any("board_pre_source.png: stale semantic payload" in problem
                   for problem in problems)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def _write_png(path: Path, color_type: int, *, palette=None, trns=None) -> None:
    """Write a 2x2 PNG by hand, so these tests need no image library.

    The module under test parses PNG with the standard library precisely so the asset gate
    does not depend on Pillow. Building the fixtures the same way keeps that true, and it
    tests the parser against the format rather than against one encoder's habits.
    """

    width = height = 2
    pixel = b"\x0c\x22\x38" if color_type == 2 else b"\x00"
    raw = (b"\x00" + pixel * width) * height
    chunks = [
        checker.PNG_SIGNATURE,
        _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)),
    ]
    if palette is not None:
        chunks.append(_png_chunk(b"PLTE", palette))
    if trns is not None:
        chunks.append(_png_chunk(b"tRNS", trns))
    chunks.append(_png_chunk(b"IDAT", zlib.compress(raw)))
    chunks.append(_png_chunk(b"IEND", b""))
    path.write_bytes(b"".join(chunks))


def test_a_truecolor_png_carrying_trns_is_not_read_as_opaque(tmp_path):
    """Opacity does not follow from the colour type: 0, 2, and 3 may all carry tRNS.

    The gate used to admit colour types 0 and 2 outright and reject 3 for being able to
    carry transparency. A truecolor asset with a tRNS chunk would therefore have passed as
    opaque, so the parser now reports the chunk and the asset rule keys on it.
    """
    path = tmp_path / "truecolor-with-trns.png"
    _write_png(path, 2, trns=struct.pack(">HHH", 12, 34, 56))

    _, _, _, color_type, has_transparency = checker.png_metadata(path)

    assert color_type == 2
    assert has_transparency is True


def test_an_opaque_palette_png_is_read_as_opaque(tmp_path):
    """A palette image without tRNS is opaque, which is why the type alone cannot decide."""
    path = tmp_path / "opaque-palette.png"
    _write_png(path, 3, palette=b"\x0c\x22\x38")

    _, _, _, color_type, has_transparency = checker.png_metadata(path)

    assert color_type == 3
    assert has_transparency is False


def test_a_renamed_scored_block_makes_only_the_hero_stale(tmp_path):
    """The hero prints the scored-block counts, so a changed board can date it.

    Its payload carries the block identifiers and not only the three totals. A corpus
    renamed without changing any count would otherwise leave the committed hero looking
    current, which is the failure the other figures are already gated against.
    """
    text = checker.DEFAULT_BOARD.read_text(encoding="utf-8")
    old = "[POST] gold_attribution :: swegym-gold"
    assert text.count(old) == 1
    board = tmp_path / "board.txt"
    board.write_text(text.replace(old, old + "-renamed"), encoding="utf-8")

    problems = checker.check_assets(board=board)

    assert any("hero-lifecycle.png: stale semantic payload" in problem
               for problem in problems)
    assert not any("catchbench_data_at_a_glance.png: stale semantic payload" in problem
                   for problem in problems)


def test_a_renamed_board_section_is_a_hard_failure(tmp_path):
    text = checker.DEFAULT_BOARD.read_text(encoding="utf-8")
    assert text.count(checker.bd.H_PRE_BY_SOURCE) == 1
    board = tmp_path / "board.txt"
    board.write_text(text.replace(checker.bd.H_PRE_BY_SOURCE,
                                  "[PRE] renamed :: F1 by source"), encoding="utf-8")

    problems = checker.check_assets(board=board)

    assert any("has no section '[PRE] pre_over_privilege :: F1 by source'" in problem
               for problem in problems)
