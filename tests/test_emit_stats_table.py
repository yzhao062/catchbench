"""The staleness checker must fail on every corruption it exists to catch.

Its first version passed six of seven mutations, which is worse than having no checker: a green
result on a corrupted table is what a reader trusts. Its second version still passed eight of
twenty-two, because it collected anything in the section that looked like a table row instead of
reading the rows inside the tabular.

Each test here is one of those corruptions, and each pins one behavior on its own. A review showed
that the earlier suite stayed green when four separate count checks were deleted from the checker
one at a time, because two tests removed several statements at once and any single surviving check
still failed them. Bundled assertions measure the bundle, not the parts.

The fixture writes a minimal paper rather than copying the real one, so a legitimate edit to the
manuscript cannot turn these red. The real paper is checked separately, behind an environment
variable, so the unit suite carries no hidden dependency on a sibling checkout.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import emit_stats_table as est  # noqa: E402


DATA = {
    "claims": (
        [{"verdict": "separates_as_stated"}] * 3
        + [{"verdict": "does_not_separate"}] * 2
    ),
    "comparison_families": [
        {"id": "alpha_family", "size": 3, "description": "Three alpha contrasts."},
        {"id": "beta_family", "size": 2, "description": "Two beta contrasts."},
    ],
}

BENCHMARK = """\
Two comparison families are declared in the module and adjusted within family by Holm.
That file is the definition of multiplicity: it regenerates all 5 reported contrasts.
"""

APPENDIX_TEMPLATE = """\
\\section{{Comparison Families}}
\\label{{app:stats}}

The {families} families below are the paper's definition of multiplicity. Of the {total}
contrasts, {separating} separate after correction and {not_separating} do not.

\\begin{{tabular}}{{lrl}}
\\toprule
Family & Tests & Scope \\\\
\\midrule
{rows}
\\midrule
Total & {total} & \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{The comparison families.}}
\\label{{tab:stat-families}}

\\section{{Something Else}}
Total & 1187 & configurations in the PRE corpus. \\\\
"""

FIRST_ROW = "\\texttt{alpha\\_family} & 3 & Three alpha contrasts. \\\\"
SECOND_ROW = "\\texttt{beta\\_family} & 2 & Two beta contrasts. \\\\"


def _appendix(rows_text=None, **overrides):
    n = est.counts(DATA)
    fields = {
        "families": n["families"],
        "total": n["total"],
        "separating": n["separating"],
        "not_separating": n["not_separating"],
        "rows": est.rows(DATA) if rows_text is None else rows_text,
    }
    fields.update(overrides)
    return APPENDIX_TEMPLATE.format(**fields)


@pytest.fixture
def paper(tmp_path):
    """A minimal current paper. Each test mutates one thing and expects check() to fail."""
    (tmp_path / "03_benchmark.tex").write_text(BENCHMARK, encoding="utf-8")
    (tmp_path / "09_appendix.tex").write_text(_appendix(), encoding="utf-8")
    return tmp_path


def _edit(paper, name, old, new, count=-1):
    path = paper / name
    text = path.read_text(encoding="utf-8")
    assert old in text, f"fixture does not contain {old!r}; the test, not the tool, is wrong"
    path.write_text(text.replace(old, new) if count < 0 else text.replace(old, new, count),
                    encoding="utf-8")


def test_current_paper_passes(paper, capsys):
    assert est.check(DATA, paper) == 0
    assert "paper is current" in capsys.readouterr().out


# --- one case per count field, so deleting any single check in the tool turns one test red ---

@pytest.mark.parametrize("name,old,new,field", [
    ("03_benchmark.tex", "Two comparison families", "Three comparison families", "benchmark families"),
    ("03_benchmark.tex", "all 5 reported contrasts", "all 4 reported contrasts", "benchmark total"),
    ("09_appendix.tex", "The 2 families below", "The 3 families below", "appendix families"),
    ("09_appendix.tex", "Of the 5", "Of the 4", "appendix total"),
    ("09_appendix.tex", "5\ncontrasts, 3 separate", "5\ncontrasts, 2 separate", "separating"),
    ("09_appendix.tex", "and 2 do not", "and 1 do not", "not separating"),
    # The Total row is deliberately absent here: it lives inside the tabular, so the table-body
    # comparison pins it, and a prose pattern for it would be a second check on the same line.
    # test_duplicated_total_row_fails and the count parametrization above cover it from both sides.
])
def test_each_count_field_is_checked(paper, name, old, new, field):
    _edit(paper, name, old, new)
    assert est.check(DATA, paper) == 1, f"the {field} count is not actually checked"


# --- table structure ---

def test_wrong_family_size_fails(paper):
    _edit(paper, "09_appendix.tex", "& 3 & Three alpha", "& 4 & Three alpha")
    assert est.check(DATA, paper) == 1


def test_rewritten_scope_description_fails(paper):
    _edit(paper, "09_appendix.tex", "Three alpha contrasts.", "WRONG SCOPE.")
    assert est.check(DATA, paper) == 1


def test_unicode_lookalike_fails(paper):
    _edit(paper, "09_appendix.tex", "Three alpha", "Thre\u0435 alpha")
    assert est.check(DATA, paper) == 1


def test_extra_obsolete_row_fails(paper):
    _edit(paper, "09_appendix.tex", SECOND_ROW,
          SECOND_ROW + "\n\\texttt{gamma\\_family} & 9 & Gone last wave. \\\\")
    assert est.check(DATA, paper) == 1


def test_duplicated_row_fails(paper):
    _edit(paper, "09_appendix.tex", SECOND_ROW, SECOND_ROW + "\n" + SECOND_ROW)
    assert est.check(DATA, paper) == 1


def test_missing_row_fails(paper):
    (paper / "09_appendix.tex").write_text(_appendix(rows_text=FIRST_ROW), encoding="utf-8")
    assert est.check(DATA, paper) == 1


def test_row_order_swapped_fails(paper):
    """Order carries meaning: the table is read alongside the module's declaration order."""
    (paper / "09_appendix.tex").write_text(
        _appendix(rows_text=SECOND_ROW + "\n" + FIRST_ROW), encoding="utf-8")
    assert est.check(DATA, paper) == 1


def test_row_moved_outside_the_tabular_fails(paper):
    """The corruption the second version passed: the row is present but no longer in the table."""
    _edit(paper, "09_appendix.tex", SECOND_ROW + "\n", "")
    _edit(paper, "09_appendix.tex", "\\end{tabular}\n", "\\end{tabular}\n" + SECOND_ROW + "\n")
    assert est.check(DATA, paper) == 1


def test_duplicated_total_row_fails(paper):
    _edit(paper, "09_appendix.tex", "Total & 5 & \\\\\n", "Total & 5 & \\\\\nTotal & 5 & \\\\\n")
    assert est.check(DATA, paper) == 1


def test_row_prefixed_with_empty_group_fails(paper):
    _edit(paper, "09_appendix.tex", SECOND_ROW, "{}" + SECOND_ROW)
    assert est.check(DATA, paper) == 1


def test_missing_tabular_start_fails(paper):
    _edit(paper, "09_appendix.tex", "\\begin{tabular}{lrl}\n", "")
    assert est.check(DATA, paper) == 1


def test_missing_tabular_end_fails(paper):
    """A table that never closes must be reported.

    This pins the outcome, not the mechanism. Replacing the tabular-end guard with a fall-back to
    end of text leaves this test green, because the body comparison then picks up the caption line
    and fails anyway. The guard earns its place on diagnostic quality rather than on detection: it
    turns a wholesale table mismatch into one line naming the real problem.
    """
    _edit(paper, "09_appendix.tex", "\\end{tabular}\n", "")
    assert est.check(DATA, paper) == 1


def test_unsafe_family_id_refuses_to_render():
    """An id needing more than underscore escaping must stop the generator, not be mangled silently."""
    broken = json.loads(json.dumps(DATA))
    broken["comparison_families"][0]["id"] = "alpha&family"
    with pytest.raises(SystemExit, match="needs escaping"):
        est.rows(broken)


def test_indentation_change_passes(paper):
    """TeX whitespace is immaterial, so reindenting a row must not be reported."""
    _edit(paper, "09_appendix.tex", SECOND_ROW, "    " + SECOND_ROW)
    assert est.check(DATA, paper) == 0


def test_restyled_scaffolding_passes(paper):
    """Renaming the column headings is a layout edit and carries no inventory."""
    _edit(paper, "09_appendix.tex", "Family & Tests & Scope \\\\", "Group & Count & Meaning \\\\")
    assert est.check(DATA, paper) == 0


# --- section markers ---

def test_missing_end_label_fails(paper):
    _edit(paper, "09_appendix.tex", "\\label{tab:stat-families}", "")
    assert est.check(DATA, paper) == 1


def test_missing_start_label_fails(paper):
    _edit(paper, "09_appendix.tex", "\\label{app:stats}", "")
    assert est.check(DATA, paper) == 1


def test_commented_out_start_label_fails(paper):
    """A commented marker is not a marker; a substring search accepted it."""
    _edit(paper, "09_appendix.tex", "\\label{app:stats}", "% \\label{app:stats}")
    assert est.check(DATA, paper) == 1


def test_conditional_inside_the_table_fails(paper):
    _edit(paper, "09_appendix.tex", SECOND_ROW, "\\iffalse\n" + SECOND_ROW + "\n\\fi")
    assert est.check(DATA, paper) == 1


def test_conditional_around_the_count_prose_fails(paper):
    """The case only the conditional guard catches.

    A conditional inside the tabular already breaks the table-body comparison. Around the count
    sentence it does not: the regex still finds the text, because TeX conditionals are not comments.
    Without the guard this corruption reads as current, so this test is what pins the guard.
    """
    _edit(paper, "09_appendix.tex", "The 2 families below",
          "\\iffalse\nThe 2 families below")
    _edit(paper, "09_appendix.tex", "and 2 do not.", "and 2 do not.\n\\fi")
    assert est.check(DATA, paper) == 1


# --- prose statements ---

def test_deleted_count_sentence_fails(paper):
    path = paper / "09_appendix.tex"
    text = path.read_text(encoding="utf-8")
    start = text.index("The 2 families below")
    end = text.index("\\begin{tabular}")
    path.write_text(text[:start] + text[end:], encoding="utf-8")
    assert est.check(DATA, paper) == 1


def test_conflicting_second_count_sentence_fails(paper):
    """Two statements of the same count means one of them is stale, whichever reads first."""
    _edit(paper, "09_appendix.tex", "and 2 do not.",
          "and 2 do not.\nOf the 99\ncontrasts, 1 separate after correction and 98 do not.")
    assert est.check(DATA, paper) == 1


def test_commented_out_count_sentence_is_not_counted(paper):
    """A commented claim is not a claim, so the check must report the statement as absent."""
    _edit(paper, "09_appendix.tex", "The 2 families below", "% The 2 families below")
    assert est.check(DATA, paper) == 1


def test_missing_file_fails(paper):
    (paper / "03_benchmark.tex").unlink()
    assert est.check(DATA, paper) == 1


def test_unrelated_total_row_is_not_read(paper):
    """The PRE corpus table's ``Total & 1187`` sits outside the stats section and must be ignored."""
    assert "Total & 1187" in (paper / "09_appendix.tex").read_text(encoding="utf-8")
    assert est.check(DATA, paper) == 0


# --- counting and number words ---

def test_opposite_direction_verdict_counts_as_separating():
    """A claim that separated the other way is still a separation, and the interesting one."""
    flipped = json.loads(json.dumps(DATA))
    flipped["claims"][-1]["verdict"] = "separates_opposite_to_statement"
    assert est.counts(flipped)["separating"] == est.counts(DATA)["separating"] + 1
    assert est.counts(flipped)["not_separating"] == est.counts(DATA)["not_separating"] - 1


def test_unknown_verdict_refuses_to_count():
    broken = json.loads(json.dumps(DATA))
    broken["claims"][0]["verdict"] = "invented_verdict"
    with pytest.raises(SystemExit, match="invented_verdict"):
        est.counts(broken)


@pytest.mark.parametrize("n,word", [(2, "Two"), (16, "Sixteen"), (17, "Seventeen"),
                                    (20, "Twenty"), (21, "Twenty-one"), (47, "Forty-seven")])
def test_number_words_cover_the_hyphenated_range(n, word):
    """A two-entry map reported a correct paper as stale at 17, and read 'Twenty-one' as 'one'."""
    assert est._NUMBER_WORDS[n] == word


@pytest.mark.parametrize("families,word", [(17, "Seventeen"), (21, "Twenty-one")])
def test_spelled_family_count_is_accepted(tmp_path, families, word):
    """The prose spells the count, so the spelled form must satisfy the check at any size.

    17 is where the earlier hardcoded map ran out, and 21 is where the count pattern stopped at the
    hyphen and read ``Twenty-one`` as ``one``. Both reported a correct paper as stale.
    """
    data = {
        "claims": [{"verdict": "does_not_separate"}],
        "comparison_families": [{"id": f"family_{i}", "size": 1, "description": f"Family {i}."}
                                for i in range(families)],
    }
    count = est.counts(data)
    assert count["families"] == families
    (tmp_path / "03_benchmark.tex").write_text(
        f"{word} comparison families are declared in the module.\n"
        f"It regenerates all {count['total']} reported contrasts.\n", encoding="utf-8")
    (tmp_path / "09_appendix.tex").write_text(
        APPENDIX_TEMPLATE.format(families=word, total=count["total"],
                                 separating=count["separating"],
                                 not_separating=count["not_separating"], rows=est.rows(data)),
        encoding="utf-8")
    assert est.check(data, tmp_path) == 0


def test_shipped_results_match_configured_paper():
    """Cross-repository closure, opt-in.

    Discovering a sibling checkout by relative path made the unit suite depend on this workstation's
    layout and on the paper repository's branch state. The environment variable is the same one the
    CLI already honors.
    """
    configured = os.environ.get("CATCHBENCH_PAPER_DIR")
    if not configured:
        pytest.skip("set CATCHBENCH_PAPER_DIR to run the cross-repository integration check")
    assert est.check(est.load(), Path(configured)) == 0


# --- what the counts count, and where the section starts and stops --------------------------------


def test_families_counts_records_not_distinct_ids():
    """Two families that share an id are still two families.

    Counting distinct ids would silently absorb a duplicated family, and the paper's multiplicity
    claim is about how many families the correction ran over, not how many names they had.
    """
    data = {
        "claims": [{"verdict": "separates_as_stated"}] * 4,
        "comparison_families": [
            {"id": "same_id", "size": 2, "description": "First."},
            {"id": "same_id", "size": 2, "description": "Second."},
        ],
    }
    assert est.counts(data)["families"] == 2


def test_a_marker_line_carrying_trailing_content_is_not_a_marker(paper):
    """The marker must be alone on its line, so an unanchored search cannot accept a lookalike."""
    _edit(paper, "09_appendix.tex", "\\label{app:stats}", "\\label{app:stats} % and a note")
    assert est.check(DATA, paper) == 1


def test_a_decoy_end_marker_before_the_start_is_not_used(paper):
    """The end marker is searched from the start marker, never from the top of the file.

    A search that ignored the offset would find this decoy, produce an empty section, and then fail
    to see the count statements that are really there.
    """
    path = paper / "09_appendix.tex"
    text = path.read_text(encoding="utf-8")
    path.write_text("\\label{tab:stat-families}\n" + text, encoding="utf-8")
    assert est.check(DATA, paper) == 0


def test_an_inline_comment_hides_a_count_sentence(paper):
    """A ``%`` anywhere on the line comments out the rest of it, not only at column zero."""
    _edit(paper, "09_appendix.tex", "The 2 families below",
          "Some prose. % The 2 families below")
    assert est.check(DATA, paper) == 1
