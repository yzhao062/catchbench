"""The README-image gate must detect stale data and stale verdict annotations."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import check_readme_figures as checker  # noqa: E402


def test_committed_figures_match_their_semantic_inputs():
    assert checker.check_assets() == []


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


def test_a_changed_registered_verdict_makes_the_pre_figure_stale(tmp_path):
    root = json.loads(checker.DEFAULT_STATS.read_text(encoding="utf-8"))
    claim = next(item for item in root["claims"]
                 if item["id"] == "pre.source.mcp.best.vs.flag_all")
    assert claim["verdict"] == "does_not_separate"
    claim["verdict"] = "separates_as_stated"
    stats = tmp_path / "statistical_tests_results.json"
    stats.write_text(json.dumps(root), encoding="utf-8")

    problems = checker.check_assets(stats=stats)

    assert any("board_pre_source.png: stale semantic payload" in problem
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
