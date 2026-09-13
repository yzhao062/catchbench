"""The README checker must fail on every corruption it exists to catch.

``tools/check_board.py`` compares two things now: ``run.py`` against ``tests/golden/board.txt``, and
``README.md`` against that same golden. The second half is what these tests cover, because it is the
half that can be false-green. A board comparison is one string equality and either matches or does
not. A README comparison has to find tables inside prose, decide which golden block each one copies,
and line up rows whose names differ on the two sides. Every one of those steps can silently match
nothing and print a pass.

That is not hypothetical here. The sibling checker in ``tools/emit_stats_table.py`` shipped twice in
a state where most corruptions of its input still produced a green result, once because it collected
anything that looked like a table row and once because its counts were never asserted. So each test
below is one corruption, asserted on the specific problem the checker is supposed to report, not on
"some problem was reported". A test that only asserts a non-empty list passes for the wrong reason
as soon as an unrelated check fires.

The table fixtures are a miniature board and a miniature README written here, so an ordinary README
edit does not invalidate all of the table-corruption tests. The prose extension also has targeted
checks against the shipped README: its registry must account for every current prose numeral, a
one-digit mutation must fail, and every comparative paragraph must bind to the measurements it
states. Those assertions are deliberately specific because matching no prose at all is the false
green they prevent.

The checker validates a stated ordering against golden cells or the sign of a recorded difference,
and reads no verdict. The README sweep is complete: the shipped prose must report no problems,
every numeral and comparative paragraph is accounted for, and their exact census is pinned below.
There is no pending-sweep exemption for printed p-values or any other defect.

``CHECK_BOARD_TOOLS`` points the import at a different copy of the checker. That is the seam used to
mutation-test the checker itself: copy ``tools/check_board.py``, break one check in the copy, and
run this file against it. Every mutation must turn this suite red.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_TOOLS = os.environ.get("CHECK_BOARD_TOOLS",
                        str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, _TOOLS)

import check_board as cb  # noqa: E402

REAL_GOLDEN = Path(__file__).resolve().parents[1] / "tests" / "golden" / "board.txt"
REAL_README = Path(__file__).resolve().parents[1] / "README.md"
REAL_STATS = Path(__file__).resolve().parents[1] / "tools" / "statistical_tests_results.json"


# --------------------------------------------------------------------------------------------
# Fixtures: a miniature board and the README that copies from it
# --------------------------------------------------------------------------------------------

# Two shapes here are deliberate regressions from the real board. The breakdown block's
# "auditable (dep-anomaly)" row leaves a single space before its first value, and the online
# detection block packs two column names into one gap. The first version of the golden parser split
# on runs of two spaces only; it silently truncated the breakdown block at the row above and lost
# the last two Gold rows, and it read the packed header as one column.
GOLDEN = """\
MiniBench :: demo board
Corpus revisions :: corpusA=abc123 | corpusB=def456

demo: 126 failed runs, 1099 steps.

[POST] demo_localization :: corpusA
  method                                 top1      top3       mrr
  random                                0.100     0.200     0.300
  position                              0.159     0.516     0.407
  pygod (graph AD)                      0.048     0.302     0.258
  llm-judge all-at-once (gpt-5.5)       0.452     0.667     0.618

[POST] demo_detection :: corpusA
  method                  roc_auc
  random                    0.483
  auditable (size+deps)     0.804

[POST] demo_detection :: corpusB
  method                  roc_auc
  random                    0.498
  auditable (size+deps)     0.665

Demo per-fault breakdown (Top-1/Top-3/MRR, tie-aware), 3 stale + 4 dropped:
  method                               overall         stale-state     dropped-grounding
  position                0.000/0.000/0.078   0.000/0.000/0.064     0.000/0.000/0.090
  auditable (dep-anomaly) 0.309/0.414/0.402   0.703/0.904/0.813     0.005/0.036/0.085

Demo online detection (n=82 paired runs):
  method                    tpr@5fpr tpr@10fpr
  random                       0.024     0.024
  auditable (span z-score)     0.061     0.110

Reading:
- Prose that is not a table and must not be indexed as one.
"""

README = """\
# MiniBench

Prose above everything.

## The Ecosystem

| Cell | Role |
|---|---|
| Tool | the SDK people build on |
| Evidence | the benchmark methods compete on |

## Fault Localization

Rank the steps of a failed run by how likely each is the fault. 126 failed runs, 1099 steps.

| Method | Top-1 | Top-3 | MRR |
|---|---|---|---|
| **judge panel** | | | |
| GPT-5.5 | **0.452** | 0.667 | **0.618** |
| position prior | 0.159 | 0.516 | 0.407 |
| PyGOD (graph AD, DOMINANT) | 0.048 | 0.302 | 0.258 |
| random | 0.100 | 0.200 | 0.300 |

How to read it. Some prose that mentions 0.452 without being a table.

## Failure Detection

corpusA, 376 runs (188 failed, 188 resolved):

| Method | ROC-AUC |
|---|---|
| random | 0.483 |
| `auditable` (size+deps) | **0.804** |

corpusB, 660 runs (363 failed, 297 resolved):

| Method | ROC-AUC |
|---|---|
| random | 0.498 |
| `auditable` (size+deps) | 0.665 |

## Injected Faults

The numbers below are one representative seed.

| Method | overall Top-1 | stale-state Top-1 | dropped-grounding Top-1 |
|---|---|---|---|
| random (seed-averaged) | 0.100 | -- | -- |
| position (leak check) | 0.000 | 0.000 | 0.000 |
| `auditable` (dep-anomaly) | 0.309 | **0.703** | 0.005 |

## Run It

```text
| Method | ROC-AUC |
|---|---|
| a table inside a fence | 9.999 |
```

Done.
"""

LOC = "[post] demo-localization :: corpusa"
DET_A = "[post] demo-detection :: corpusa"
DET_B = "[post] demo-detection :: corpusb"
BREAKDOWN = "demo per-fault breakdown"


def _detection(name, block, lead_in):
    return cb.TableSpec(
        name=name,
        heading_contains="failure detection",
        lead_in_contains=lead_in,
        header=("method", "roc-auc"),
        columns={"roc-auc": cb.Source(block, "roc-auc")},
        rows={"random": cb.Row("random"),
              "auditable (size+deps)": cb.Row("auditable (size+deps)")},
    )


SPECS = (
    cb.TableSpec(
        name="localization",
        heading_contains="fault localization",
        header=("method", "top-1", "top-3", "mrr"),
        columns={"top-1": cb.Source(LOC, "top1"),
                 "top-3": cb.Source(LOC, "top3"),
                 "mrr": cb.Source(LOC, "mrr")},
        separators=frozenset({"judge panel"}),
        rows={"gpt-5.5": cb.Row("llm-judge all-at-once (gpt-5.5)"),
              "position prior": cb.Row("position"),
              "pygod (graph ad, dominant)": cb.Row("pygod (graph AD)"),
              "random": cb.Row("random")},
    ),
    _detection("detection corpusA", DET_A, "corpusa"),
    _detection("detection corpusB", DET_B, "corpusb"),
    cb.TableSpec(
        name="gold",
        heading_contains="injected faults",
        header=("method", "overall top-1", "stale-state top-1", "dropped-grounding top-1"),
        columns={"overall top-1": cb.Source(BREAKDOWN, "overall", part=0),
                 "stale-state top-1": cb.Source(BREAKDOWN, "stale-state", part=0),
                 "dropped-grounding top-1": cb.Source(BREAKDOWN, "dropped-grounding", part=0)},
        rows={"random (seed-averaged)": cb.Row(
                  "random",
                  sources={"overall top-1": cb.Source(LOC, "top1")},
                  absent=frozenset({"stale-state top-1", "dropped-grounding top-1"})),
              "position (leak check)": cb.Row("position"),
              "auditable (dep-anomaly)": cb.Row("auditable (dep-anomaly)")},
    ),
)

# 4 localization rows x 3 columns, 2 detection tables x 2 rows x 1 column, 3 gold rows x 3 columns
# less the 2 cells the board does not report for the random row.
CLEAN_CELLS = 12 + 2 + 2 + 7


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def edit(text, old, new):
    """Replace one anchor, insisting it occurs exactly once.

    A corruption test that silently replaces nothing passes without testing anything, which is the
    same false green the checker itself is being defended against.
    """
    assert text.count(old) == 1, f"anchor {old!r} occurs {text.count(old)} time(s), expected 1"
    return text.replace(old, new)


def run(readme=README, golden=GOLDEN, specs=SPECS, exemptions=()):
    return cb.check_readme_detailed(
        readme, golden, specs, exemptions, MINI_NUMBER_ALLOWLIST, (), {"claims": []})


def board_claims(readme, data, entries, golden=GOLDEN):
    """Run the claim gate over a fixture, resolving cell orderings against a golden board."""
    blocks, counts = cb.parse_golden(golden)
    return cb.check_board_claims(readme, blocks, counts, data, entries)


def kinds(result):
    return sorted(p.kind for p in result.problems)


def only(result, kind):
    """The problems of one kind, insisting there is at least one."""
    hits = [p for p in result.problems if p.kind == kind]
    assert hits, f"expected a {kind!r} problem, got {kinds(result)}"
    return hits


def readme_line(text, needle):
    """1-based line number of the single line containing ``needle``."""
    hits = [i for i, line in enumerate(text.split("\n"), start=1) if needle in line]
    assert len(hits) == 1, f"{needle!r} occurs on lines {hits}"
    return hits[0]


MINI_NUMBER_ALLOWLIST = (
    cb.ProseNumberAllowance("fixture-localization-counts", "126 failed",
                            ("126", "1099"),
                            "Fixture run and step counts."),
    cb.ProseNumberAllowance("fixture-detection-a-counts", "376 runs",
                            ("376", "188", "188"),
                            "Fixture corpus-A run counts."),
    cb.ProseNumberAllowance("fixture-detection-b-counts", "660 runs",
                            ("660", "363", "297"),
                            "Fixture corpus-B run counts."),
)

CLAIM_README = """\
# Demo

## The Boards

### Result

Alpha beats Beta.
"""

# The verdict here is the negative one on purpose. Under the retired gate this record could not
# license an ordering; under the current one the verdict is not read at all, and what settles the
# sentence is that the recorded difference is positive in the direction the paragraph states.
CLAIM_DATA = {
    "claims": [{"id": "demo.alpha.vs.beta",
                "verdict": "does_not_separate",
                "estimate": {"a": 0.8, "a_name": "Alpha",
                             "b": 0.6, "b_name": "Beta",
                             "difference_a_minus_b": 0.2}}],
}
CLAIM_ENTRY = cb.BoardClaim(
    "demo ordering", "Alpha beats Beta",
    differences=(cb.RecordedDifference("demo.alpha.vs.beta", cb.A_ABOVE_B),))

# The same gate over a golden cell rather than a recorded contrast. corpusA prints 0.804 for the
# dependency block and 0.483 for random.
CELL_CLAIM_README = """\
# Demo

## The Boards

### Result

The dependency block scores higher than random on corpusA.
"""
CELL_CLAIM_ENTRY = cb.BoardClaim(
    "demo cell ordering", "scores higher than random on corpusA",
    orderings=(cb.BoardOrdering(cb.Cell(DET_A, "auditable (size+deps)", "roc-auc"),
                                cb.Cell(DET_A, "random", "roc-auc")),))


# --------------------------------------------------------------------------------------------
# The clean fixture, and what "clean" means
# --------------------------------------------------------------------------------------------


def test_clean_fixture_passes():
    assert run().problems == []
    assert cb.check_adjudicating_numbers(README) == []

    real = REAL_README.read_text(encoding="utf-8")
    golden = REAL_GOLDEN.read_text(encoding="utf-8")

    # The shipped prose-number registry is populated against the current README. Comparative
    # wording defects are checked separately below, so they cannot mask a number-registry hole.
    blocks, _ = cb.parse_golden(golden)
    number_problems, seen, board_backed, allowed = cb.check_prose_numbers(
        real, blocks, cb.PROSE_NUMBER_ALLOWLIST)
    assert number_problems == []
    assert (seen, board_backed, allowed) == (243, 110, 133)
    names = [allowance.name for allowance in cb.PROSE_NUMBER_ALLOWLIST]
    assert len(names) == len(set(names))
    assert all(allowance.values and allowance.reason.strip() and allowance.context_contains.strip()
               for allowance in cb.PROSE_NUMBER_ALLOWLIST)

    # A comparative paragraph whose stated ordering agrees with the recorded difference is the
    # green claim-gate case, and it stays green on a record whose verdict is the negative one.
    claim_result = cb.check_readme_detailed(
        CLAIM_README, "", specs=(), exemptions=(), number_allowances=(),
        board_claims=(CLAIM_ENTRY,), statistical_data=CLAIM_DATA)
    assert claim_result.problems == []
    assert claim_result.comparative_paragraphs == claim_result.comparative_claimed == 1

    # The same for an ordering settled by two golden cells rather than by a recorded contrast.
    cell_problems, cell_seen, cell_claimed = board_claims(
        CELL_CLAIM_README, {"claims": []}, (CELL_CLAIM_ENTRY,))
    assert cell_problems == []
    assert (cell_seen, cell_claimed) == (1, 1)

    # Content anchors are unaffected by unrelated insertions above them, so the same problems come
    # back with the same messages once the shift in line numbers is set aside.
    baseline = cb.check_readme_detailed(real, golden)
    shifted = cb.check_readme_detailed("\n" * 20 + real, golden)

    assert baseline.problems == []
    assert sorted(p.message for p in baseline.problems) == sorted(
        p.message for p in shifted.problems)
    shifted_lines = {q.line for q in shifted.problems}
    assert all(p.line + 20 in shifted_lines for p in baseline.problems if p.line is not None)
    assert baseline.comparative_claimed == shifted.comparative_claimed == 15

    # Removing one paragraph stales the allowances and the board claim anchored inside it, and
    # nothing else. Which entries those are is computed from the removed text rather than listed,
    # so the assertion survives a rewording of the paragraph that remains.
    start = real.index("How to read it. The direct LLM control")
    end = real.index("\n\n### Failure Detection", start)
    # Both registries match against whitespace-joined paragraph text, so the removed span is
    # joined the same way before asking which anchors lived inside it.
    removed = " ".join(line.strip() for line in real[start:end].split("\n"))
    changed = real[:start] + real[end + 2:]
    result = cb.check_readme_detailed(changed, golden)

    anchored = {allowance.name for allowance in cb.PROSE_NUMBER_ALLOWLIST
                if allowance.context_contains in removed}
    anchored |= {entry.name for entry in cb.BOARD_CLAIMS
                 if entry.paragraph_contains in removed}
    assert len(anchored) >= 3
    stale = [problem for problem in result.problems
             if problem.kind in {"allowance-unmatched", "board-claim-unmatched"}]
    assert len(stale) == len(anchored)
    assert all(any(name in problem.message for name in anchored) for problem in stale)
    assert [problem for problem in result.problems if problem not in stale] == []

    # Duplicating an anchor cannot silently absorb a second copy of the same numbers.
    text = "# Demo\n\n126 failed and 1099 steps.\n\n126 failed and 1099 steps.\n"
    allowance = cb.ProseNumberAllowance(
        "counts", "126 failed", ("126", "1099"), "Fixture counts.")
    problems, _, _, _ = cb.check_prose_numbers(text, {}, (allowance,))

    ambiguous = [problem for problem in problems if problem.kind == "allowance-unmatched"]
    assert len(ambiguous) == 1
    assert "occurs 2 times" in ambiguous[0].message


def test_clean_fixture_actually_compares_cells():
    # The load-bearing assertion of the whole file. Without it a checker that claims every table and
    # compares nothing passes every test above.
    result = run()
    assert result.cells_compared == CLEAN_CELLS
    assert result.tables_claimed == 4
    assert result.tables_numeric == 4


def test_report_states_the_coverage():
    assert f"{CLEAN_CELLS} cell(s)" in cb.readme_report(run())


def test_non_numeric_table_needs_no_spec():
    # The ecosystem table has no numbers, so it is not a board table and is not required to map to
    # one. This is the rule that keeps prose tables out of the check.
    result = run()
    assert result.tables_seen == 5
    assert result.tables_numeric == 4


def test_table_inside_a_code_fence_is_not_read():
    # The fenced table's 9.999 is unclaimed and unmapped; if the fence were ignored it would be
    # reported as an unclaimed numeric table.
    assert "9.999" in README
    assert run().problems == []


# --------------------------------------------------------------------------------------------
# Corruption 1-4: the numbers themselves
# --------------------------------------------------------------------------------------------


def test_changed_digit_fails():
    text = edit(README, "| GPT-5.5 | **0.452** | 0.667 | **0.618** |",
                "| GPT-5.5 | **0.451** | 0.667 | **0.618** |")
    problems = only(run(readme=text), "mismatch")
    assert len(problems) == 1
    assert problems[0].line == readme_line(text, "GPT-5.5 |")
    assert "'0.451'" in problems[0].message and "'0.452'" in problems[0].message

    # The same corruption in prose must go red. This uses the current README and shipped registry,
    # not a table fixture, so it proves the hole this extension closes. The numeral to mutate is
    # found rather than named, because the sentence carrying any particular one can be reworded.
    real = REAL_README.read_text(encoding="utf-8")
    blocks, _ = cb.parse_golden(REAL_GOLDEN.read_text(encoding="utf-8"))
    baseline, _, _, _ = cb.check_prose_numbers(real, blocks, cb.PROSE_NUMBER_ALLOWLIST)
    golden_values = cb._golden_cell_values(blocks)
    numbers, _ = cb.scan_prose_numbers(real)

    target, mutated = None, None
    for number in numbers:
        if not cb._matches_golden_precision(number.value, golden_values):
            continue
        for suffix in ("1", "7"):
            candidate = number.value + suffix
            if not cb._matches_golden_precision(candidate, golden_values):
                target, mutated = number, candidate
                break
        if target is not None:
            break
    assert target is not None, "the README prints no board-backed prose numeral to mutate"

    line = real.split("\n")[target.line - 1]
    changed = "\n".join(real.split("\n")[:target.line - 1]
                        + [line[:target.column - 1] + mutated
                           + line[target.column - 1 + len(target.value):]]
                        + real.split("\n")[target.line:])
    prose_problems, _, _, _ = cb.check_prose_numbers(changed, blocks, cb.PROSE_NUMBER_ALLOWLIST)
    hits = [problem for problem in prose_problems
            if problem.kind == "unclaimed-number" and problem not in baseline]
    assert len(hits) == 1
    assert hits[0].line == target.line
    assert repr(mutated) in hits[0].message


def test_truncated_precision_fails():
    # 0.45 is 0.452 rounded. A tolerance would accept it; that is exactly the movement this check
    # exists to catch, so the comparison is on the printed text.
    text = edit(README, "**0.452**", "**0.45**")
    assert len(only(run(readme=text), "mismatch")) == 1


def test_extra_precision_fails():
    text = edit(README, "| random | 0.100 | 0.200 | 0.300 |",
                "| random | 0.1000 | 0.200 | 0.300 |")
    assert len(only(run(readme=text), "mismatch")) == 1


def test_dropped_trailing_zero_fails():
    text = edit(README, "| random | 0.100 | 0.200 | 0.300 |", "| random | 0.1 | 0.200 | 0.300 |")
    assert len(only(run(readme=text), "mismatch")) == 1


def test_value_swapped_between_two_columns_fails():
    text = edit(README, "| position prior | 0.159 | 0.516 | 0.407 |",
                "| position prior | 0.516 | 0.159 | 0.407 |")
    assert len(only(run(readme=text), "mismatch")) == 2


def test_value_copied_from_the_wrong_corpus_fails():
    # 0.498 is the corpusB random score. Pasted into the corpusA table it is a real number from a
    # real board, which is the drift a human reviewer is least likely to spot.
    text = edit(README, "| random | 0.483 |", "| random | 0.498 |")
    problems = only(run(readme=text), "mismatch")
    assert len(problems) == 1
    assert "'0.498'" in problems[0].message and "'0.483'" in problems[0].message


def test_wrong_part_of_a_slash_triple_fails():
    # The gold table copies Top-1 out of "0.309/0.414/0.402"; taking Top-3 by mistake must fail.
    text = edit(README, "| `auditable` (dep-anomaly) | 0.309 |",
                "| `auditable` (dep-anomaly) | 0.414 |")
    assert len(only(run(readme=text), "mismatch")) == 1


# --------------------------------------------------------------------------------------------
# Corruption 5-8: the shape of the table
# --------------------------------------------------------------------------------------------


def test_dropped_row_fails():
    text = edit(README, "| position prior | 0.159 | 0.516 | 0.407 |\n", "")
    problems = only(run(readme=text), "row-missing")
    assert len(problems) == 1
    assert "'position prior'" in problems[0].message


def test_added_row_fails():
    text = edit(README, "| random | 0.100 | 0.200 | 0.300 |",
                "| random | 0.100 | 0.200 | 0.300 |\n| newcomer | 0.900 | 0.900 | 0.900 |")
    problems = only(run(readme=text), "row-undeclared")
    assert len(problems) == 1
    assert problems[0].line == readme_line(text, "newcomer")


def test_renamed_method_fails():
    text = edit(README, "| position prior |", "| positional prior |")
    result = run(readme=text)
    assert "row-undeclared" in kinds(result) and "row-missing" in kinds(result)


def test_truncated_table_fails():
    text = edit(README,
                "| PyGOD (graph AD, DOMINANT) | 0.048 | 0.302 | 0.258 |\n"
                "| random | 0.100 | 0.200 | 0.300 |\n", "")
    assert len(only(run(readme=text), "row-missing")) == 2


def test_reordered_columns_fail():
    text = edit(README, "| Method | Top-1 | Top-3 | MRR |", "| Method | Top-3 | Top-1 | MRR |")
    result = run(readme=text)
    # The header signature is part of the table's identity, so a reordered header stops matching its
    # spec, and the table it left behind is then an unclaimed numeric table.
    assert "spec-unmatched" in kinds(result)
    assert "unclaimed-table" in kinds(result)


def test_duplicated_row_label_fails():
    text = edit(README, "| random | 0.100 | 0.200 | 0.300 |",
                "| random | 0.100 | 0.200 | 0.300 |\n| random | 0.100 | 0.200 | 0.300 |")
    assert len(only(run(readme=text), "duplicate-row")) == 1


# --------------------------------------------------------------------------------------------
# Corruption 9-11: tables that map to nothing
# --------------------------------------------------------------------------------------------


def test_new_numeric_table_is_unclaimed():
    text = README + """
## Something New

| Widget | Score |
|---|---|
| alpha | 0.777 |
"""
    problems = only(run(readme=text), "unclaimed-table")
    assert len(problems) == 1
    assert "NON_BOARD_TABLES" in problems[0].message
    assert problems[0].line == readme_line(text, "| Widget | Score |")


def test_new_prose_table_is_not_unclaimed():
    text = README + """
## Something New

| Widget | Note |
|---|---|
| alpha | a sentence with no number |
"""
    assert run(readme=text).problems == []


def test_a_recorded_exemption_claims_a_non_board_table():
    text = README + """
## Corpus Sizes

| Corpus | Runs |
|---|---|
| corpusA | 376 |
"""
    exemption = cb.Exemption(name="corpus sizes", header=("corpus", "runs"),
                             heading_contains="corpus sizes",
                             reason="counts of the input corpora, not scored cells")
    assert run(readme=text, exemptions=(exemption,)).problems == []


def test_a_stale_exemption_fails():
    exemption = cb.Exemption(name="corpus sizes", header=("corpus", "runs"),
                             reason="counts of the input corpora, not scored cells")
    assert len(only(run(exemptions=(exemption,)), "exemption-unmatched")) == 1

    # Removing a paragraph's registry entry leaves its comparative word unaccounted and goes red.
    claim_problems, seen, claimed = board_claims(CLAIM_README, CLAIM_DATA, ())
    assert (seen, claimed) == (1, 0)
    assert len(claim_problems) == 1
    assert claim_problems[0].kind == "unaccounted-comparison"

    # An entry that claims a paragraph and declares no measurement is the same silent skip in
    # another shape, so it fails unless it records on one line why the paragraph orders nothing.
    empty = cb.BoardClaim("demo empty", "Alpha beats Beta")
    problems, _, _ = board_claims(CLAIM_README, CLAIM_DATA, (empty,))
    assert [problem.kind for problem in problems] == ["board-claim-empty", "unaccounted-comparison"]

    excused = cb.BoardClaim("demo descriptive", "Alpha beats Beta",
                            reports_no_ordering="Fixture paragraph, kept to exercise the reason.")
    problems, _, claimed = board_claims(CLAIM_README, CLAIM_DATA, (excused,))
    assert (problems, claimed) == ([], 1)

    # Two entries cannot both claim one paragraph; that is how a second, unchecked reading of the
    # same sentence would get in.
    problems, _, _ = board_claims(CLAIM_README, CLAIM_DATA, (CLAIM_ENTRY, excused))
    assert [problem.kind for problem in problems] == ["comparison-double-claimed"]

    # A duplicated entry name is a registry defect, not a second claim.
    problems, _, _ = board_claims(CLAIM_README, CLAIM_DATA, (CLAIM_ENTRY, CLAIM_ENTRY))
    assert [problem.kind for problem in problems] == ["board-claim-duplicate"]


def test_spec_that_matches_no_table_fails():
    text = edit(README, "| Method | ROC-AUC |\n|---|---|\n| random | 0.483 |\n"
                        "| `auditable` (size+deps) | **0.804** |\n", "")
    problems = only(run(readme=text), "spec-unmatched")
    assert len(problems) == 1
    assert "detection corpusA" in problems[0].message


def test_spec_that_matches_two_tables_fails():
    # Two tables with the same header under the same heading and the same lead-in word.
    text = edit(README,
                "corpusB, 660 runs (363 failed, 297 resolved):",
                "corpusA again, 660 runs (363 failed, 297 resolved):")
    result = run(readme=text)
    assert len(only(result, "spec-ambiguous")) == 1
    assert "detection corpusA" in only(result, "spec-ambiguous")[0].message


def test_lead_in_is_what_separates_two_identically_shaped_tables():
    # Swapping the two lead-in paragraphs swaps which golden block each table is compared against,
    # so every cell in both tables must now disagree.
    text = README.replace("corpusA, 376 runs", "corpusTMP, 376 runs")
    text = edit(text, "corpusB, 660 runs", "corpusA, 660 runs")
    text = edit(text, "corpusTMP, 376 runs", "corpusB, 376 runs")
    assert len(only(run(readme=text), "mismatch")) == 4


# --------------------------------------------------------------------------------------------
# Corruption 12-15: tables that cannot be parsed
# --------------------------------------------------------------------------------------------


def test_ragged_row_fails():
    text = edit(README, "| position prior | 0.159 | 0.516 | 0.407 |",
                "| position prior | 0.159 | 0.407 |")
    result = run(readme=text)
    assert only(result, "unparsed-table")[0].line == readme_line(text, "| position prior |")
    # A table whose structure is damaged is not trusted cell by cell: it is dropped whole, so its
    # spec then reports that it matched nothing, and none of its cells are counted as checked.
    assert "spec-unmatched" in kinds(result)
    assert result.cells_compared == CLEAN_CELLS - 12


def test_missing_delimiter_row_fails():
    text = edit(README, "| Method | Top-1 | Top-3 | MRR |\n|---|---|---|---|\n",
                "| Method | Top-1 | Top-3 | MRR |\n")
    result = run(readme=text)
    assert "unparsed-table" in kinds(result)
    assert "spec-unmatched" in kinds(result)


def test_delimiter_with_the_wrong_column_count_fails():
    text = edit(README, "| Method | Top-1 | Top-3 | MRR |\n|---|---|---|---|",
                "| Method | Top-1 | Top-3 | MRR |\n|---|---|---|")
    assert len(only(run(readme=text), "unparsed-table")) == 1


def test_header_only_table_fails():
    text = README + "\n## Stub\n\n| Method | Score |\n|---|---|\n"
    assert len(only(run(readme=text), "unparsed-table")) == 1


# --------------------------------------------------------------------------------------------
# Corruption 16-19: placeholders, separators, and cells the board does not have
# --------------------------------------------------------------------------------------------


def test_placeholder_where_the_board_has_a_value_fails():
    text = edit(README, "| position prior | 0.159 |", "| position prior | -- |")
    problems = only(run(readme=text), "mismatch")
    assert "'0.159'" in problems[0].message


def test_number_where_the_board_has_no_cell_fails():
    # The board reports no per-fault split for the random row. Inventing one must fail.
    text = edit(README, "| random (seed-averaged) | 0.100 | -- | -- |",
                "| random (seed-averaged) | 0.100 | 0.050 | -- |")
    problems = only(run(readme=text), "unsupported-value")
    assert "'0.050'" in problems[0].message


def test_absent_declaration_that_the_board_contradicts_fails():
    # If the board grows the cell a spec calls absent, the spec is stale and must say so rather
    # than keep waving the README's placeholder through.
    spec = SPECS[3]
    grown = cb.TableSpec(
        name=spec.name, heading_contains=spec.heading_contains, header=spec.header,
        columns=spec.columns, separators=spec.separators,
        rows=dict(spec.rows, **{"random (seed-averaged)": cb.Row(
            "position",
            sources={"overall top-1": cb.Source(LOC, "top1")},
            absent=frozenset({"stale-state top-1", "dropped-grounding top-1"}))}),
    )
    result = run(specs=SPECS[:3] + (grown,))
    assert len(only(result, "stale-spec")) == 2


def _localization_spec(**overrides):
    spec = SPECS[0]
    fields = dict(name=spec.name, heading_contains=spec.heading_contains,
                  lead_in_contains=spec.lead_in_contains, header=spec.header,
                  columns=spec.columns, rows=spec.rows, separators=spec.separators)
    fields.update(overrides)
    return cb.TableSpec(**fields)


def test_column_with_no_golden_source_fails():
    broken = _localization_spec(columns={"top-1": cb.Source(LOC, "top1")})
    result = run(specs=(broken,) + SPECS[1:])
    assert len(only(result, "spec-gap")) == 2
    # A spec that cannot describe its own table compares none of it.
    assert result.cells_compared == CLEAN_CELLS - 12


def test_row_declared_as_both_separator_and_row_fails():
    broken = _localization_spec(separators=frozenset({"judge panel", "random"}))
    assert len(only(run(specs=(broken,) + SPECS[1:]), "spec-gap")) == 1


def test_separator_row_carrying_a_value_fails():
    text = edit(README, "| **judge panel** | | | |", "| **judge panel** | 0.900 | | |")
    assert len(only(run(readme=text), "separator-has-values")) == 1


def test_undeclared_blank_row_fails():
    text = edit(README, "| **judge panel** | | | |",
                "| **judge panel** | | | |\n| **structural** | | | |")
    problems = only(run(readme=text), "undeclared-separator")
    assert "separators" in problems[0].message


def test_declared_separator_that_disappeared_fails():
    text = edit(README, "| **judge panel** | | | |\n", "")
    problems = only(run(readme=text), "row-missing")
    assert "'judge panel'" in problems[0].message


# --------------------------------------------------------------------------------------------
# Corruption 20-23: the golden side moving under the spec
# --------------------------------------------------------------------------------------------


def test_renamed_golden_block_fails():
    text = edit(GOLDEN, "[POST] demo_localization :: corpusA",
                "[POST] demo_localization :: corpusZ")
    problems = only(run(golden=text), "unresolved")
    assert "no golden block" in problems[0].message


def test_removed_golden_row_fails():
    text = edit(GOLDEN, "  position                              0.159     0.516     0.407\n", "")
    problems = only(run(golden=text), "unresolved")
    assert "has no row 'position'" in problems[0].message

    # A measurement moving under a still-registered paragraph is the statistical analogue. The
    # sign of the recorded difference is what settles the stated ordering.
    flipped = {"claims": [{"id": "demo.alpha.vs.beta",
                           "verdict": "separates_as_stated",
                           "estimate": {"a": 0.6, "a_name": "Alpha",
                                        "b": 0.8, "b_name": "Beta",
                                        "difference_a_minus_b": -0.2}}]}
    claim_problems, seen, claimed = board_claims(CLAIM_README, flipped, (CLAIM_ENTRY,))
    assert (seen, claimed) == (1, 1)
    assert len(claim_problems) == 1
    assert claim_problems[0].kind == "ordering-contradicted"
    assert "'Alpha'" in claim_problems[0].message and "-0.2" in claim_problems[0].message

    # The verdict is the field this gate used to read and now does not. Moving it either way over
    # the same estimate changes nothing, and that is the whole point of the migration.
    for verdict in ("separates_as_stated", "does_not_separate", "something_else", None):
        data = json.loads(json.dumps(CLAIM_DATA))
        data["claims"][0]["verdict"] = verdict
        assert board_claims(CLAIM_README, data, (CLAIM_ENTRY,))[0] == []

    # An exact tie orders nothing, so a stated ordering over one fails and NO_ORDERING passes.
    tied = json.loads(json.dumps(CLAIM_DATA))
    tied["claims"][0]["estimate"].update(b=0.8, difference_a_minus_b=0.0)
    problems, _, _ = board_claims(CLAIM_README, tied, (CLAIM_ENTRY,))
    assert [problem.kind for problem in problems] == ["ordering-contradicted"]
    cited = cb.BoardClaim("demo tie", "Alpha beats Beta",
                          differences=(cb.RecordedDifference("demo.alpha.vs.beta"),))
    assert board_claims(CLAIM_README, tied, (cited,))[0] == []

    # A record whose own arithmetic disagrees with itself cannot settle anything.
    broken = json.loads(json.dumps(CLAIM_DATA))
    broken["claims"][0]["estimate"]["difference_a_minus_b"] = 0.9
    problems, _, _ = board_claims(CLAIM_README, broken, (CLAIM_ENTRY,))
    assert [problem.kind for problem in problems] == ["measurement-inconsistent"]

    # A record with no estimate, a missing ID, a duplicated ID, and an unknown direction each fail
    # by name rather than passing for want of something to compare.
    for mutation, kind in (
            (lambda d: d["claims"][0].pop("estimate"), "measurement-missing"),
            (lambda d: d["claims"][0].__setitem__("id", "demo.other"), "claim-missing"),
            (lambda d: d["claims"].append(dict(d["claims"][0])), "claim-duplicate"),
    ):
        data = json.loads(json.dumps(CLAIM_DATA))
        mutation(data)
        problems, _, _ = board_claims(CLAIM_README, data, (CLAIM_ENTRY,))
        assert [problem.kind for problem in problems] == [kind], kind

    misdirected = cb.BoardClaim(
        "demo direction", "Alpha beats Beta",
        differences=(cb.RecordedDifference("demo.alpha.vs.beta", "higher"),))
    problems, _, _ = board_claims(CLAIM_README, CLAIM_DATA, (misdirected,))
    assert [problem.kind for problem in problems] == ["board-claim-states"]

    # A stated cell ordering the board contradicts fails the same way, and a cell the board no
    # longer prints is reported rather than skipped.
    reversed_cells = cb.BoardClaim(
        "demo reversed", "scores higher than random on corpusA",
        orderings=(cb.BoardOrdering(cb.Cell(DET_A, "random", "roc-auc"),
                                    cb.Cell(DET_A, "auditable (size+deps)", "roc-auc")),))
    problems, _, _ = board_claims(CELL_CLAIM_README, {"claims": []}, (reversed_cells,))
    assert [problem.kind for problem in problems] == ["ordering-contradicted"]
    assert "0.483" in problems[0].message and "0.804" in problems[0].message

    equal_cells = cb.BoardClaim(
        "demo equal", "scores higher than random on corpusA",
        orderings=(cb.BoardOrdering(cb.Cell(DET_A, "random", "roc-auc"),
                                    cb.Cell(DET_A, "random", "roc-auc")),))
    problems, _, _ = board_claims(CELL_CLAIM_README, {"claims": []}, (equal_cells,))
    assert [problem.kind for problem in problems] == ["ordering-contradicted"]
    assert "equals" in problems[0].message

    gone = cb.BoardClaim(
        "demo missing cell", "scores higher than random on corpusA",
        orderings=(cb.BoardOrdering(cb.Cell(DET_A, "no-such-method", "roc-auc"),
                                    cb.Cell(DET_A, "random", "roc-auc")),))
    problems, _, _ = board_claims(CELL_CLAIM_README, {"claims": []}, (gone,))
    assert [problem.kind for problem in problems] == ["board-claim-unresolved"]
    assert "no row" in problems[0].message


def test_renamed_golden_column_fails():
    text = edit(GOLDEN, "  method                                 top1      top3       mrr",
                "  method                                 top_1     top3       mrr")
    problems = only(run(golden=text), "unresolved")
    assert "has no column 'top1'" in problems[0].message


def test_duplicated_golden_title_must_be_disambiguated():
    text = edit(GOLDEN, "[POST] demo_detection :: corpusB\n"
                        "  method                  roc_auc\n"
                        "  random                    0.498\n"
                        "  auditable (size+deps)     0.665\n",
                "[POST] demo_detection :: corpusA\n"
                "  method                  roc_auc\n"
                "  random                    0.498\n"
                "  auditable (size+deps)     0.665\n")
    problems = only(run(golden=text), "unresolved")
    assert "#1" in problems[0].message and "#2" in problems[0].message


def test_unparsable_golden_block_is_reported_not_skipped():
    # A column added to the header and not to the rows. The block is recognizable but unusable, and
    # the spec that names it gets the reason rather than an empty match.
    text = edit(GOLDEN, "  method                  roc_auc\n  random                    0.483",
                "  method                  roc_auc  coverage\n  random                    0.483")
    problems = only(run(golden=text), "unresolved")
    assert "did not parse" in problems[0].message
    assert "no data rows parsed" in problems[0].message


def test_golden_row_with_one_space_before_its_first_value_is_kept():
    # The regression the two-space split caused: this row and everything below it vanished, and the
    # spec that named them failed as if the board had changed.
    blocks, _ = cb.parse_golden(GOLDEN)
    block = blocks[BREAKDOWN]
    assert block.error is None
    assert block.rows["auditable (dep-anomaly)"] == ("0.309/0.414/0.402", "0.703/0.904/0.813",
                                                     "0.005/0.036/0.085")


def test_golden_header_packed_into_one_gap_is_split():
    blocks, _ = cb.parse_golden(GOLDEN)
    block = blocks["demo online detection"]
    assert block.error is None
    assert block.columns == ("method", "tpr@5fpr", "tpr@10fpr")
    assert block.rows["auditable (span z-score)"] == ("0.061", "0.110")


def test_prose_lines_are_not_indexed_as_golden_blocks():
    blocks, _ = cb.parse_golden(GOLDEN)
    assert "reading" not in blocks
    assert "corpus revisions :: corpusa=abc123 | corpusb=def456" not in blocks


# --------------------------------------------------------------------------------------------
# The shipped specs against the shipped golden
# --------------------------------------------------------------------------------------------


def test_live_stale_ordering_has_no_unrelated_gold_license():
    """A new ordering inside ``The Boards`` must be accounted for, and no p-value may return.

    The historical defect this name records: the LIVE stale-state paragraph ordered raw span
    against the z-score while the registry declared no LIVE stale-state contrast at all, and a
    licence anchored there named ``gold.maxspan.stale.floor``. That claim separated, so the gate
    reported the paragraph as licensed, a false negative rather than a miss. The verdict half of
    that gate is gone; the half that caught it, the coverage accounting, is what this holds. An
    ordering added anywhere under ``The Boards`` with no registry entry behind it goes red.
    """
    readme = REAL_README.read_text(encoding="utf-8")
    golden = REAL_GOLDEN.read_text(encoding="utf-8")
    blocks, counts = cb.parse_golden(golden)
    stats = json.loads(REAL_STATS.read_text(encoding="utf-8"))

    anchor = "\n## How a Method Plugs In"
    assert readme.count(anchor) == 1
    bad = readme.replace(
        anchor,
        "\nRaw span scores higher than the z-score, so per-run normalization does not help.\n"
        + anchor)

    clean, _, _ = cb.check_board_claims(readme, blocks, counts, stats, cb.BOARD_CLAIMS)
    problems, _, _ = cb.check_board_claims(bad, blocks, counts, stats, cb.BOARD_CLAIMS)

    added = [problem for problem in problems if problem not in clean]
    assert [problem.kind for problem in added] == ["unaccounted-comparison"]
    assert "'higher'" in added[0].message

    # The other half of the reporting policy is lexical and deliberately two spellings wide.
    printed = cb.check_adjudicating_numbers(
        "# Demo\n\nThe gap holds under correction (Holm p=0.004).\n")
    assert [problem.kind for problem in printed] == ["adjudicating-number"] * 2
    assert all(problem.line == 3 for problem in printed)

    # Ordinary English that shares those stems is not a comparison and must not fire, and neither
    # may a pinned dependency or anything inside a fenced block.
    quiet = (
        "# Demo\n\n"
        "The pillars are separate tracks. SWE-Gym, 376 runs (188 failed, 188 resolved).\n"
        "One unresolved licensing term remains, and pip resolves scipy>=1.10 on its own.\n"
        "```text\nHolm p=0.004\n```\n"
    )
    assert cb.check_adjudicating_numbers(quiet) == []


def test_shipped_specs_resolve_against_the_committed_golden():
    """Every block, column, and row named in TABLE_SPECS must exist in tests/golden/board.txt.

    This reads the committed board and no part of README.md, so an edit to the README cannot turn
    it red and a wrong number in the README cannot turn it green. What it catches is a spec that
    points at a golden cell the board no longer prints.
    """
    blocks, counts = cb.parse_golden(REAL_GOLDEN.read_text(encoding="utf-8"))
    missing = []
    for spec in cb.TABLE_SPECS:
        for label, row in spec.rows.items():
            for column in spec.header[1:]:
                if column in row.absent:
                    continue
                source = row.sources.get(column, spec.columns[column])
                value, why = cb._golden_value(blocks, counts, source, row.golden)
                if value is None:
                    missing.append(f"{spec.name}: {label} / {column}: {why}")
    assert missing == []

    # Every measurement the shipped registry states must hold against the committed board and the
    # committed record. This half reads no allowance and no numeral, so those checks cannot mask a
    # missing or incorrect measurement binding.
    stats = json.loads(REAL_STATS.read_text(encoding="utf-8"))
    claim_problems, seen, claimed = cb.check_board_claims(
        REAL_README.read_text(encoding="utf-8"), blocks, counts, stats, cb.BOARD_CLAIMS)
    assert claim_problems == []
    assert seen == claimed == len(cb.BOARD_CLAIMS) == 15

    # The 8-model Who&When band is bound to the record and ordered by nothing inside it.
    localization = next(entry for entry in cb.BOARD_CLAIMS
                        if entry.name == "Who&When localization board")
    band = [difference for difference in localization.differences
            if difference.claim_id.startswith("loc.band.")]
    assert len(band) == 28
    assert {difference.states for difference in band} == {cb.NO_ORDERING}

    # The registry has teeth on the shipped board: reversing any stated ordering must fail. Doing
    # it for all of them is what rules out a registry of orderings that resolve to nothing.
    reversible = [(entry, ordering) for entry in cb.BOARD_CLAIMS
                  for ordering in entry.orderings]
    assert len(reversible) >= 25
    for entry, ordering in reversible:
        flipped = cb.BoardClaim(entry.name, entry.paragraph_contains,
                                orderings=(cb.BoardOrdering(ordering.below, ordering.above),),
                                reports_no_ordering=entry.reports_no_ordering or "fixture")
        problems, _, _ = cb.check_board_claims(
            REAL_README.read_text(encoding="utf-8"), blocks, counts, stats,
            tuple(other for other in cb.BOARD_CLAIMS if other is not entry) + (flipped,))
        assert [problem.kind for problem in problems] == ["ordering-contradicted"], (
            f"{entry.name}: {ordering.above.render()} over {ordering.below.render()}")

    # The same for every stated direction over a recorded contrast.
    directed = [(entry, difference) for entry in cb.BOARD_CLAIMS
                for difference in entry.differences if difference.states != cb.NO_ORDERING]
    assert len(directed) >= 30
    for entry, difference in directed:
        opposite = cb.A_ABOVE_B if difference.states == cb.B_ABOVE_A else cb.B_ABOVE_A
        flipped = cb.BoardClaim(
            entry.name, entry.paragraph_contains,
            differences=(cb.RecordedDifference(difference.claim_id, opposite),))
        problems, _, _ = cb.check_board_claims(
            REAL_README.read_text(encoding="utf-8"), blocks, counts, stats,
            tuple(other for other in cb.BOARD_CLAIMS if other is not entry) + (flipped,))
        assert [problem.kind for problem in problems] == ["ordering-contradicted"], (
            f"{entry.name}: {difference.claim_id}")


def test_shipped_specs_cover_every_column_they_claim():
    for spec in cb.TABLE_SPECS:
        for column in spec.header[1:]:
            assert column in spec.columns, f"{spec.name}: {column!r} is not mapped"


def test_shipped_specs_have_distinct_identities():
    seen = set()
    for spec in cb.TABLE_SPECS:
        identity = (spec.header, spec.heading_contains, spec.lead_in_contains)
        assert identity not in seen, f"{spec.name} cannot be told apart from another spec"
        seen.add(identity)


# --------------------------------------------------------------------------------------------
# The command line, which is what CI runs
# --------------------------------------------------------------------------------------------


SCRIPT = str(Path(cb.__file__).resolve())


def _cli(tmp_path, readme_text, golden_text):
    readme = tmp_path / "README.md"
    golden = tmp_path / "board.txt"
    readme.write_text(readme_text, encoding="utf-8")
    golden.write_text(golden_text, encoding="utf-8")
    return subprocess.run(
        [sys.executable, SCRIPT, "--readme-only", "--readme", str(readme),
         "--golden", str(golden)],
        capture_output=True, text=True)


@pytest.fixture()
def fixture_registry(monkeypatch):
    """Point the shipped entry points at the miniature registry.

    ``main`` reads ``TABLE_SPECS`` through the module, so this exercises the path CI runs (argument
    parsing, the file reads, the report, the exit code) against fixtures instead of the real
    README, which another edit could change at any time.
    """
    monkeypatch.setattr(cb, "TABLE_SPECS", SPECS)
    monkeypatch.setattr(cb, "NON_BOARD_TABLES", ())
    monkeypatch.setattr(cb, "PROSE_NUMBER_ALLOWLIST", MINI_NUMBER_ALLOWLIST)
    monkeypatch.setattr(cb, "BOARD_CLAIMS", ())


def test_main_exits_zero_on_a_clean_readme(tmp_path, fixture_registry, monkeypatch):
    readme = tmp_path / "README.md"
    golden = tmp_path / "board.txt"
    readme.write_text(README, encoding="utf-8")
    golden.write_text(GOLDEN, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_board.py", "--readme-only",
                                      "--readme", str(readme), "--golden", str(golden)])
    assert cb.main() == 0


def test_main_exits_one_and_names_the_line_and_both_values(tmp_path, fixture_registry, monkeypatch,
                                                           capsys):
    text = edit(README, "| GPT-5.5 | **0.452** | 0.667 | **0.618** |",
                "| GPT-5.5 | **0.999** | 0.667 | **0.618** |")
    readme = tmp_path / "README.md"
    golden = tmp_path / "board.txt"
    readme.write_text(text, encoding="utf-8")
    golden.write_text(GOLDEN, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_board.py", "--readme-only",
                                      "--readme", str(readme), "--golden", str(golden)])
    assert cb.main() == 1
    out = capsys.readouterr().out
    assert f"README.md:{readme_line(text, 'GPT-5.5 |')}" in out
    assert "'0.999'" in out and "'0.452'" in out


def test_cli_reports_drift_and_exits_one(tmp_path):
    # A real subprocess, so the shebang path, the argument parsing, and the exit code are the ones
    # CI sees. The shipped specs do not know this fixture, so the drift reported is the fixture's
    # tables being unclaimed; that is still the fail-closed behavior under test.
    done = _cli(tmp_path, README, GOLDEN)
    assert done.returncode == 1
    assert "README DRIFT" in done.stdout


def test_cli_exits_one_when_the_readme_is_missing(tmp_path):
    golden = tmp_path / "board.txt"
    golden.write_text(GOLDEN, encoding="utf-8")
    done = subprocess.run(
        [sys.executable, SCRIPT, "--readme-only", "--readme", str(tmp_path / "nope.md"),
         "--golden", str(golden)], capture_output=True, text=True)
    assert done.returncode == 1
    assert "no README" in done.stdout


def test_cli_exits_one_when_the_golden_is_missing(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(README, encoding="utf-8")
    done = subprocess.run(
        [sys.executable, SCRIPT, "--readme-only", "--readme", str(readme),
         "--golden", str(tmp_path / "nope.txt")], capture_output=True, text=True)
    assert done.returncode == 1
    assert "no golden" in done.stdout


def test_cli_prints_the_readme_line_number(tmp_path):
    # An end-to-end check that a reported problem carries the location a person needs, through the
    # real argument parsing and the real report renderer.
    text = edit(README, "| GPT-5.5 | **0.452** | 0.667 | **0.618** |",
                "| GPT-5.5 | **0.999** | 0.667 | **0.618** |")
    readme = tmp_path / "README.md"
    golden = tmp_path / "board.txt"
    readme.write_text(text, encoding="utf-8")
    golden.write_text(GOLDEN, encoding="utf-8")
    done = subprocess.run(
        [sys.executable, SCRIPT, "--readme-only", "--readme", str(readme),
         "--golden", str(golden)], capture_output=True, text=True)
    assert done.returncode == 1
    # The shipped specs do not know this fixture, so the drift is reported as unclaimed tables
    # rather than as a mismatch; either way the report must name a line in README.md.
    assert "README.md:" in done.stdout


# --------------------------------------------------------------------------------------------
# What counts as a board table: the loosening that closed a silent skip
# --------------------------------------------------------------------------------------------


def test_a_table_of_spreads_is_seen_rather_than_skipped():
    """A cell like ``0.824 +/- 0.007`` must still make its table a board table.

    Requiring the whole cell to be one bare number excluded any table of seeded spreads. Such a
    table was neither compared nor reported, and the summary still said every numeric table was
    claimed, so a wrong number there would have shipped in silence. The board prints spreads of
    exactly this shape, so a future README table carrying them is a realistic addition.
    """
    text = README + (
        "\n\n### Seed Stability\n\n"
        "| Method | ROC-AUC |\n|---|---|\n"
        "| g-safeguard | 0.824 +/- 0.007 |\n"
    )
    result = run(readme=text)
    assert result.tables_numeric == run().tables_numeric + 1
    assert only(result, "unclaimed-table")


def test_identifier_cells_still_do_not_make_a_board_table():
    """The loosening must not drag reference tables in, which is what the strict form protected.

    These are the shapes the README's own reference tables use. Each is a single token that is not
    a number, so splitting a cell into tokens leaves them excluded exactly as before.
    """
    for cell in ("CWE-272", "OWASP LLM06", "LLM06:2026", "v0.1.0", "--", "n/a",
                 # A standalone integer inside a phrase is an identifier, not a board value. A
                 # rule that read tokens made these numeric and pulled reference tables in.
                 "RFC 6749", "OWASP Top 10", "Passed to the loader"):
        assert not cb.is_numeric(cell), cell
    for cell in ("0.452", "**0.828**", "`0.777`", "11%", "-1", "1187", ".824", "1e-3",
                 # Every shape the board itself prints. The slash triple is the one that stayed
                 # invisible the longest: tests/golden/board.txt prints Gold rows this way.
                 "0.309/0.414/0.402", "0.824 +/- 0.007", "0.824+/-0.007", "0.824±0.007",
                 "[0.817, 0.831]"):
        assert cb.is_numeric(cell), cell


def test_a_bare_zero_is_a_value_and_not_a_placeholder():
    """`0` and `0.000` are real board scores; a placeholder means "no value was printed".

    Folding a bare zero into PLACEHOLDERS would make every zero cell exempt from comparison, and
    zero is exactly the value a broken method prints, so a regression to zero would stop being
    visible. flag_none scores 0.000 on all three PRE metrics today.
    """
    for cell in ("0", "0.0", "0.000", "0%", "-0"):
        assert not cb.is_placeholder(cell), cell
        assert cb.is_numeric(cell), cell
    for cell in ("", "-", "--", "---", "n/a", "NA", "\u2014"):
        assert cb.is_placeholder(cell), cell


def test_a_table_is_numeric_even_when_only_a_late_row_has_a_number():
    """The numeric test must read every row, not a prefix of them.

    A checker that decided from the first row or two would silently drop any table whose leading
    rows are qualitative and whose scores start further down, and would still report every numeric
    table as claimed.
    """
    late = cb.ReadmeTable(
        line=1, heading="future scores", lead_in="", header=("method", "value"),
        rows=tuple((10 + i, label, cells) for i, (label, cells) in enumerate([
            ("random", ("pending",)),
            ("position", ("pending",)),
            ("degree", ("pending",)),
            ("auditable", ("0.804",)),
        ])))
    assert late.numeric, "a number in the last row still makes this a board table"

    qualitative = cb.ReadmeTable(
        line=1, heading="ecosystem", lead_in="", header=("cell", "role"),
        rows=tuple((10 + i, label, cells) for i, (label, cells) in enumerate([
            ("Tool", ("the SDK",)), ("Evidence", ("the benchmark",)),
            ("Knowledge", ("the reading list",)), ("Method", ("graph construction",)),
        ])))
    assert not qualitative.numeric, "a table with no numbers anywhere is not a board table"
