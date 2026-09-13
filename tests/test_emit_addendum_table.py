r"""Hold the judge-addendum block to the scorer's numbers and to the claims it may not make.

Two failures are worth failing loudly for, and both are invisible to a reader of the LaTeX.

The first is a second implementation of Top-1. The declared outcome is ``rank == 1`` after a stable
sort of the per-step score vector, which is not the reading the generation script prints; the two can
disagree on the 20 runs whose gold mistake is step 0. A table that recomputed the number would look
right and be wrong, in the manuscript rather than in a console line. So the emitter owns no score,
and the printed cells are checked against the public report's own rows rather than against the
emitter's private view of them.

It owns exactly one calculation, the Procedure BIN interval, and that is deliberate: two of these six
arms already print in ``tab:protocol`` under BIN, so borrowing the scorer's Wilson bounds would put
one score under two constructions in one appendix. The guard therefore narrows rather than
disappears. numpy may be reached from ``_bin_interval`` and nowhere else, the only float constants
allowed are the two percentile bounds, and the ``above`` column is checked against the intervals the
table actually prints.

The second is a claim the addendum is not entitled to make. Six marginal intervals over shared runs
support no verdict about any pair of arms, and the block's own caption has to say so, because that
sentence is the last unmet clause of the M4 completion test. A forbidden-word scan is a blunt guard,
and blunt is what is wanted: the failure mode is a later edit reaching for a comparative word because
the ordering invites one.
"""
from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import emit_addendum_table as eat  # noqa: E402
import score_judge_addendum as sja  # noqa: E402

SCORER = TOOLS / "score_judge_addendum.py"

# A row of the public report: position, above-count, label, channel, source, then Top-1 and Top-3
# each followed by a bracketed interval, then MRR.
_REPORT_ROW = re.compile(
    r"^\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)"
    r"\s+([\d.]+)\s+\[([\d.]+), ([\d.]+)\]"
    r"\s+([\d.]+)\s+\[([\d.]+), ([\d.]+)\]\s+([\d.]+)\s*$")

# The same arm as the generated block prints it.
_BLOCK_ROW = re.compile(
    r"^\\texttt\{(\S+)\} & \\texttt\{(\S+)\} & (\S+)"
    r" & ([\d.]+) \{\\scriptsize \$\[([\d.]+), ([\d.]+)\]\$\}"
    r" & ([\d.]+) \{\\scriptsize \$\[([\d.]+), ([\d.]+)\]\$\} \\\\$")

# The per-arm position comment. It keeps the cache label, which the printed row no longer shows,
# and it is emitted in the same order as the rows, so zipping the two joins a row to its cache.
_POSITION_COMMENT = re.compile(
    r"^% (\S+): source=(\S+); channel=(\S+); position=(\d+); above=(\d+); n=(\d+)$")


def _ordered_labels(generated: str) -> list[str]:
    """Cache labels in printed order, read from the position comments."""
    return [m.group(1) for m in map(_POSITION_COMMENT.match, generated.splitlines()) if m]


# The full-precision comment lines the block carries beside every printed cell.
_TOP1_COMMENT = re.compile(r"^% (\S+): top1=(\d+)/(\d+)=([\d.]+); interval=\[([\d.]+), ([\d.]+)\]$")
_TOP3_COMMENT = re.compile(
    r"^% (\S+): top3=(\d+)/(\d+)=([\d.]+); interval=\[([\d.]+), ([\d.]+)\]; mrr=([\d.]+)$")

# Every word that would turn a ranking into a verdict. "registered contrast" is handled apart: the
# caption is required to carry it, in the negated form and only there.
FORBIDDEN = ("significant", "separates", "outperforms", "best", "holm", "p =",
             "statistically", "wins", "beats", "superior")


@pytest.fixture(scope="module")
def ranked():
    return sja.rank_arms()


@pytest.fixture(scope="module")
def generated(ranked):
    return eat.build()


@pytest.fixture(scope="module")
def report():
    """The public report's own stdout, so the block is compared against what the tool publishes."""
    result = subprocess.run([sys.executable, str(SCORER)], capture_output=True, text=True,
                            encoding="utf-8", timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout


def _report_rows(report: str) -> list[tuple]:
    return [match.groups() for match in map(_REPORT_ROW.match, report.splitlines()) if match]


def _block_rows(generated: str) -> list[tuple]:
    return [match.groups() for match in map(_BLOCK_ROW.match, generated.splitlines()) if match]


def _comments(generated: str, pattern: re.Pattern) -> dict[str, tuple]:
    found = {}
    for line in generated.splitlines():
        match = pattern.match(line)
        if match:
            found[match.group(1)] = match.groups()[1:]
    return found


def _cli(*args, paper_env=None):
    env = os.environ.copy()
    env.pop("CATCHBENCH_PAPER_DIR", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if paper_env is not None:
        env["CATCHBENCH_PAPER_DIR"] = str(paper_env)
    return subprocess.run([sys.executable, str(Path(eat.__file__)), *map(str, args)],
                          capture_output=True, env=env, timeout=600)


def test_every_arm_matches_the_scorer_report_row_for_row(generated, report):
    """Same arms, same order, same identity fields, same numbers as the published report.

    The comparison runs against the report's printed rows rather than against ``rank_arms``, so a
    block that silently dropped, reordered or relabelled an arm fails here even though it would agree
    with the function it called.
    """
    printed, block = _report_rows(report), _block_rows(generated)
    labels = _ordered_labels(generated)
    expected = len(sja.TABLE_COHORT) + 2  # roster plus the two published references
    assert len(printed) == len(block) == len(labels) == expected, "every roster arm, once"
    top1 = _comments(generated, _TOP1_COMMENT)
    top3 = _comments(generated, _TOP3_COMMENT)
    for console, row, label in zip(printed, block, labels):
        assert console[2] == label, "the row joins the console line for the same cache"
        assert row[0] == sja.TABLE_COHORT.get(label, label), "row prints the provider model name"
        assert console[3:5] == row[1:3], "channel and source"
        label = console[2]
        assert label in top1 and label in top3, f"{label} carries no full-precision comment"
        # The report prints four decimals and the comment lines twelve, so agreement to within half
        # a unit in the report's last place is the strongest statement the two formats support.
        # Order: Top-1 with its interval, Top-3 with its interval, then MRR.
        # Scores must agree with the console. Intervals must not: the console reports Wilson and
        # the block draws Procedure BIN, so that tab:protocol and this table print one cell rather
        # than one score under two constructions. Comparing them here is what used to hide that.
        scores = [(console[5], top1[label][2]), (console[8], top3[label][2]),
                  (console[11], top3[label][5])]
        for value, exact in scores:
            assert float(value) == pytest.approx(float(exact), rel=0, abs=5e-5)
        console_bounds = [float(console[6]), float(console[7]),
                          float(console[9]), float(console[10])]
        block_bounds = [float(top1[label][3]), float(top1[label][4]),
                        float(top3[label][3]), float(top3[label][4])]
        assert console_bounds != block_bounds, (
            f"{label}: the block's interval equals the console's, so it is not Procedure BIN")


def test_the_published_arms_print_the_same_cell_as_the_protocol_table():
    """The reason the interval is drawn here: one score may not carry two intervals in one appendix.

    tab:protocol already prints these two arms' Top-1. If this block used a different construction,
    a reader working through Appendix D meets the same number twice with different bounds about six
    hundred source lines apart.
    """
    published = {"gpt-5.5": (0.365, 0.540), "llama-3.3-70b": (0.246, 0.413)}
    arms, _ = sja.rank_arms()
    by_label = {a["label"]: a for a in eat._with_bin_intervals(arms)}
    for label, (low, high) in published.items():
        arm = by_label[label]
        assert (round(arm["top1_lo"], 3), round(arm["top1_hi"], 3)) == (low, high), (
            f"{label}: BIN gives [{arm['top1_lo']:.3f}, {arm['top1_hi']:.3f}], tab:protocol prints "
            f"[{low}, {high}]")


def test_printed_cells_are_the_full_precision_values_rounded(generated):
    """Each three-decimal cell is its own comment line rounded, not a separately derived number."""
    top1 = _comments(generated, _TOP1_COMMENT)
    top3 = _comments(generated, _TOP3_COMMENT)
    rows = _block_rows(generated)
    labels = _ordered_labels(generated)
    assert rows, "no rows parsed"
    assert len(rows) == len(labels)
    for row, label in zip(rows, labels):
        for cell, exact in zip(row[3:6], top1[label][2:]):
            assert cell == "%.3f" % float(exact)
        for cell, exact in zip(row[6:9], top3[label][2:5]):
            assert cell == "%.3f" % float(exact)


def test_hit_counts_reconstruct_every_printed_proportion(generated):
    """k/n must return the proportion, so a reader can rebuild any interval from the block alone."""
    for pattern in (_TOP1_COMMENT, _TOP3_COMMENT):
        found = _comments(generated, pattern)
        assert len(found) == len(sja.TABLE_COHORT) + 2
        for label, fields in found.items():
            hits, n, point = int(fields[0]), int(fields[1]), float(fields[2])
            assert n == 126, f"{label} is scored on {n} runs"
            assert hits / n == pytest.approx(point, rel=0, abs=1e-12)
            low, high = float(fields[3]), float(fields[4])
            assert low < point < high


def test_the_block_claims_no_test_no_separation_and_no_winner(generated):
    """A ranking with marginal intervals licenses none of these words.

    ``registered contrast`` is the exception that proves the rule. The caption has to carry the
    phrase, because declaring that the block registers none of them is the last unmet clause of the
    M4 completion test, so it is checked for the negated form instead of banned.
    """
    lowered = generated.lower()
    for word in FORBIDDEN:
        assert word not in lowered, f"the block claims {word!r}"
    assert lowered.count("registered contrast") == 1
    assert lowered.count("no registered contrast") == 1
    assert "tests no difference between any two arms" in lowered
    assert "carry no rank" in lowered, "the block must say it is not a ranking"


def test_the_late_run_model_carries_its_dated_history(generated):
    """gpt-5.6-sol is scored now, and the block must date the gap rather than assert it persists.

    Erratum 2 asked for the model, its non-run status, and the reason on the same page. The model
    was run on 2026-09-12, so the status half of that requirement is spent: a block still printing
    it as a non-measurement would be false, and a block silently dropping the history would leave a
    reader unable to tell why the first addendum omitted an OpenAI model it had declared. What has
    to survive is the dated account.
    """
    assert "gpt-5.6-sol" in sja.TABLE_COHORT
    assert r"\texttt{gpt-5.6-sol} & \texttt{nairr-gateway} & addendum" in generated
    assert r"\multicolumn{2}{c}{declared, never run}" not in generated
    assert "declared non-measurement" not in generated
    prose = eat.caption_disclosure()
    assert prose in generated
    assert "declared on 2026-09-06 and not run then" in prose
    assert "run over that forward on 2026-09-12" in prose
    # The dates alone are not the disclosure. Erratum 2 asks for the reason, and asks specifically
    # that it read as a decision: "not setting it up was a choice rather than an obstacle ...
    # Recording it as an unavailability would be false." A revision of this constant once said the
    # forward "was not set up at the time", which is the forbidden reading, and the only assertion
    # then standing was equality with the constant, which cannot see a change of meaning. These
    # three are literals on purpose, so that editing the constant cannot also edit the test.
    assert "configuration decision, not model unavailability" in prose
    assert "Before either OpenAI score existed" in prose
    assert "the authors preferred " + TEXTTT_ASTRA in prose
    # Strip the caption's markup back off and the console's own five lines must come back, word for
    # word and in order. Neither surface can lose a sentence or gain one without failing here.
    assert (prose.replace(r"\texttt{", "").replace("}", "")
            == " ".join(sja.LATE_RUN_HISTORY))


def test_the_caption_disclosure_refuses_latex_specials():
    """A later edit that puts markup in the console constant must fail here, not in a build log."""
    with pytest.raises(ValueError, match="LaTeX special characters"):
        eat.caption_disclosure(("100% of runs were skipped",))
    assert eat.caption_disclosure(("one line", "two line")) == "one line two line"


def test_the_caption_refuses_a_roster_its_prose_does_not_fit(ranked):
    """Both guards carry a formatted message, so both are exercised rather than assumed.

    The counts in the caption are derived from the arms in hand. A roster this prose cannot describe
    has to stop the run: a caption that says four arms over a table of five is worse than no table.
    """
    arms, n_runs = ranked
    with pytest.raises(ValueError, match="caption names two published arms"):
        eat._caption(n_runs, [dict(arm, source="addendum") for arm in arms])
    with pytest.raises(ValueError, match="outgrown the prose"):
        eat._word(len(arms) * 13)
    # An independent literal on the right. Comparing _word(len(arms)) against
    # _word(len(TABLE_COHORT) + 2) is the same number on both sides and can never fail.
    assert len(arms) == 12, "roster of ten plus two published references"
    assert eat._word(len(arms)) == "twelve"
    assert eat._word(0) == "no"


def test_the_emitter_owns_no_arithmetic_of_its_own():
    """The whole reason this tool exists rather than a hand-typed table.

    A second Top-1 in this repository would be wrong on the runs whose gold mistake is step 0 and
    would look correct everywhere a reviewer checked.
    """
    assert eat.rank_arms is sja.rank_arms
    for banned in ("math", "wilson", "per_run_ranks", "score_cache",
                   "_score_vector", "_rank_metrics", "collect"):
        assert not hasattr(eat, banned), f"emit_addendum_table reaches {banned}"
    # numpy is reachable, because the BIN draw happens here on purpose. It may be reached from
    # exactly one function, so a later edit cannot use it to compute a score.
    tree = ast.parse(Path(eat.__file__).read_text("utf-8"))
    users = sorted({node.name for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and any(isinstance(inner, ast.Name) and inner.id == "np"
                            for inner in ast.walk(node))})
    assert users == ["_bin_interval"], f"numpy is reached from {users}, not only the interval draw"
    numbers = {node.value for node in ast.walk(ast.parse(Path(eat.__file__).read_text("utf-8")))
               if isinstance(node, ast.Constant)
               and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)}
    assert 126 not in numbers, "the run count is imported, never typed"
    # The only floats allowed are the two percentile bounds Procedure BIN takes. A score written
    # down here would be a number the records do not govern, which is the whole failure mode.
    floats = {value for value in numbers if isinstance(value, float)}
    assert floats <= {2.5, 97.5}, f"no score is written down here; found {sorted(floats)}"


def test_cli_prints_deterministic_lf_bytes(generated):
    first, second = _cli(), _cli()
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == b""
    assert first.stdout == second.stdout == (generated + "\n").encode("utf-8")
    assert b"\r" not in first.stdout


@pytest.mark.parametrize("use_env", [False, True])
def test_check_accepts_exact_block(tmp_path, generated, use_env):
    (tmp_path / "09_appendix.tex").write_bytes((generated + "\n").encode("utf-8"))
    result = (_cli("--check", paper_env=tmp_path) if use_env
              else _cli("--check", "--paper", tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert b"paper is current: tab:judge-addendum" in result.stdout


@pytest.mark.parametrize("corruption, message", [
    ("no_block", b"appears 0 times"), ("no_appendix", b"cannot read appendix"),
    ("cell", b"table differs"), ("caption", b"table differs"),
    ("missing_begin", b"appears 0 times"), ("missing_end", b"appears 0 times"),
    ("duplicate", b"appears 2 times"), ("reversed", b"precedes its begin marker"),
    ("whitespace", b"table differs"), ("crlf", b"table differs"),
])
def test_check_rejects_a_paper_that_does_not_carry_the_block(tmp_path, generated, corruption,
                                                             message):
    """The red direction. A checker that cannot fail proves nothing about the paper it passes.

    ``no_block`` and ``no_appendix`` are the state of the manuscript before anything is spliced, so
    they pin the exit code this tool must return today.
    """
    text = generated
    if corruption == "no_block":
        text = r"\section{An appendix that has never seen this table}"
    elif corruption == "cell":
        text = text.replace(r"& 0.476 {\scriptsize", r"& 0.999 {\scriptsize", 1)
    elif corruption == "caption":
        text = text.replace("declares no registered contrast", "declares a registered contrast", 1)
    elif corruption == "missing_begin":
        text = text.replace(eat._BEGIN, "")
    elif corruption == "missing_end":
        text = text.replace(eat._END, "")
    elif corruption == "duplicate":
        text += "\n" + eat._BEGIN
    elif corruption == "reversed":
        text = eat._END + "\n" + eat._BEGIN
    elif corruption == "whitespace":
        text = text.replace(r"\centering", r"\centering ", 1)
    elif corruption == "crlf":
        text = text.replace("\n", "\r\n")
    if corruption != "no_appendix":
        (tmp_path / "09_appendix.tex").write_bytes(text.encode("utf-8"))
    result = _cli("--check", "--paper", tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert message in result.stdout
    if corruption == "cell":
        assert b"--- paper/09_appendix.tex" in result.stdout
        assert b"0.999" in result.stdout and b"0.476" in result.stdout


def test_check_requires_paper_path():
    result = _cli("--check")
    assert result.returncode == 2
    assert b"--check needs --paper <dir> or CATCHBENCH_PAPER_DIR" in result.stderr


# The four tests below exist because a review pass broke this suite four ways and it stayed green.
# It appended "claude-opus-5 leads the ranking and is the strongest judge here" to the caption, moved
# 72 entrants to 76, eleven published judges to fourteen, and the 20-run null-top floor to 3. Each
# corruption is a claim about something outside this block, so a guard has to reach outside it too.

COMPARATIVE = ("leads", "leading", "strongest", "weakest", "ahead of", "behind",
               "better than", "worse than", "top performer", "winner", "loses to")


def test_the_block_makes_no_comparative_claim_even_in_unlisted_vocabulary(generated):
    """FORBIDDEN caught only the statistical words; a sorted table invites the ranking words too."""
    prose = " ".join(generated.split()).lower()
    hits = [word for word in COMPARATIVE if word in prose]
    assert hits == [], f"the block ranks arms but may not describe the ranking: {hits}"


def test_the_published_judge_count_matches_the_cache_directory():
    """PUBLISHED_JUDGES is prose in the caption; bind it to the caches it describes."""
    root = Path(sja.__file__).resolve().parents[1]
    caches = sorted((root / "data" / "llm_judge").glob("whoandwhen__all_at_once__*.json"))
    assert len(caches) == eat.PUBLISHED_JUDGES, (
        f"caption says {eat.PUBLISHED_JUDGES} published judges, data/llm_judge/ holds {len(caches)}")


def test_the_arena_counts_match_the_paper():
    """72 entrants and 138 recorded comparisons are claims about the manuscript, not about us."""
    paper_dir = os.environ.get("CATCHBENCH_PAPER_DIR")
    if not paper_dir:
        pytest.skip("no CATCHBENCH_PAPER_DIR")
    sources = " ".join(p.read_text(encoding="utf-8")
                       for p in sorted(Path(paper_dir).glob("*.tex")))
    assert f"{eat.ARENA_ENTRANTS} entrants" in sources, (
        f"caption says {eat.ARENA_ENTRANTS} entrants; the paper does not")
    assert str(eat.RECORDED_COMPARISONS) in sources, (
        f"caption says {eat.RECORDED_COMPARISONS} recorded comparisons; the paper does not")


def test_the_null_top_floor_is_the_one_the_caches_actually_carry():
    """The 20-run figure warns the reader that the console reading is a different quantity."""
    from catchbench.llm_judge import load_judge_runs
    runs = load_judge_runs()
    at_step_zero = sum(1 for r in runs if r["mistake"] == 0)
    assert at_step_zero == eat.NULL_TOP_FLOOR, (
        f"caption says {eat.NULL_TOP_FLOOR} runs with the gold mistake at step 0; the corpus has "
        f"{at_step_zero}")


def test_the_above_column_is_derivable_from_the_printed_intervals():
    """The column and the intervals must tell the same story, and they have separate sources.

    ``above`` is computed by the scorer from its own Wilson bounds; the table prints Procedure BIN.
    The two agree on today's caches, which is luck rather than structure: nothing recomputes the
    column when the interval construction changes. A reader can check this column against the
    printed bounds by eye, so a disagreement would be visible and unexplainable.
    """
    arms, _ = sja.rank_arms()
    binned = eat._with_bin_intervals(arms)
    for arm in binned:
        implied = 1 + sum(1 for other in binned if other["top1_lo"] > arm["top1_hi"])
        assert implied == arm["above"], (
            f"{arm['label']}: the table prints above={arm['above']}, but its printed interval "
            f"[{arm['top1_lo']:.3f}, {arm['top1_hi']:.3f}] implies {implied}")


def test_no_caption_percent_starts_a_latex_comment(generated):
    r"""A bare % in the caption comments out the rest of its line, silently and invisibly.

    This is not hypothetical. A shortened caption kept ``95\%%`` from a string that had used %
    interpolation; the rewrite had no % operator, so both signs reached LaTeX and the second one
    ate ``Procedure BIN intervals, and the references' Top-1 cells repeat Table 6``. The build
    reported zero warnings, --check passed because source and output shared the error, and every
    test here passed because they all read source strings rather than rendered text.

    The generated block is post-interpolation, so any % outside the provenance comment lines must
    be escaped to print. Checking the emitted bytes is what closes the gap.
    """
    offenders = []
    for number, line in enumerate(generated.splitlines(), 1):
        if line.startswith("%"):
            continue  # a provenance comment line, which is meant to be a comment
        for index, char in enumerate(line):
            if char == "%" and (index == 0 or line[index - 1] != "\\"):
                offenders.append(f"line {number} col {index + 1}: {line[max(0, index - 40):index + 20]}")
    assert offenders == [], "unescaped % in the generated block:\n  " + "\n  ".join(offenders)


TEXTTT_ASTRA = r"\texttt{gpt-6-astra}"


def _staged_runs_and_task():
    """The scorer's own inputs, so a held-out replicate is scored by the published code path."""
    task = sja.PostLocalization()
    task.setup()
    return sja.load_judge_runs(), task


def _roster_probe(tmp_path, monkeypatch, mutate):
    """Copy the addendum directory, let `mutate` disturb it, and rank from there."""
    staged = tmp_path / "llm_judge_addendum"
    shutil.copytree(sja.ADDENDUM_DIR, staged)
    mutate(staged)
    monkeypatch.setattr(sja, "ADDENDUM_DIR", staged)
    return sja.rank_arms()


def test_a_roster_member_with_no_cache_stops_the_run(tmp_path, monkeypatch):
    """A shorter self-consistent table is the failure this check exists to prevent."""
    def drop(staged):
        (staged / "whoandwhen__all_at_once__gpt-5.6-sol.json").unlink()

    with pytest.raises(SystemExit, match="gpt-5.6-sol"):
        _roster_probe(tmp_path, monkeypatch, drop)


def test_another_protocol_cannot_stand_in_for_an_all_at_once_arm(tmp_path, monkeypatch):
    """The roster names a protocol, so a same-label cache of a different one is a missing file.

    Keying on the label alone accepted this substitution and then scored it as all-at-once against
    an unchanged manuscript, which is worse than the missing-cache case it was written to catch: the
    table keeps its twelve rows and one of them silently answers a different question.
    """
    def substitute(staged):
        (staged / "whoandwhen__all_at_once__gpt-5.6-sol.json").rename(
            staged / "whoandwhen__step_by_step__gpt-5.6-sol.json")

    with pytest.raises(SystemExit, match="gpt-5.6-sol"):
        _roster_probe(tmp_path, monkeypatch, substitute)


def test_a_second_protocol_beside_a_complete_roster_is_ignored(tmp_path, monkeypatch):
    """Globbing emitted the same model twice; addressing by path leaves the extra file unread."""
    def add(staged):
        shutil.copyfile(staged / "whoandwhen__all_at_once__gpt-5.6-sol.json",
                        staged / "whoandwhen__step_by_step__gpt-5.6-sol.json")

    arms, _ = _roster_probe(tmp_path, monkeypatch, add)
    assert len(arms) == 12
    assert [a["label"] for a in arms].count("gpt-5.6-sol") == 1


def test_figure_fives_outside_the_arena_values_match_the_caches(ranked):
    """Four literals in the manuscript's panel script, bound to the caches they were read from.

    The panel cross-checks its 116 frozen values against the statistics registry, which does not
    reach these four: the addendum caches live here rather than beside the manuscript, so the script
    carries them as literals and the appendix says so. That disclosure is worth having only while
    the numbers still agree, and nothing else compares them. The dots are the two same-channel
    gateway runs per model, which is what the caption claims; the CLI cache of the same model is a
    different channel and is not drawn.
    """
    paper_dir = os.environ.get("CATCHBENCH_PAPER_DIR")
    if not paper_dir:
        pytest.skip("no CATCHBENCH_PAPER_DIR")
    script = Path(paper_dir) / "figure" / "make_localization_panel.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    entries = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "LATER_ARMS" for t in node.targets):
            entries = ast.literal_eval(node.value)
    assert entries is not None, "LATER_ARMS not found in %s" % script
    # The roster is asserted, not merely iterated. Checking only the entries that happen to be
    # present let the whole claude-opus-5 entry be deleted while the test still passed, which
    # leaves the appendix promising four literals over a panel drawing two. The count is taken
    # before the dictionary, so two entries sharing a name cannot collapse unseen.
    assert len(entries) == 2, "the panel declares %d later arms; the appendix says two" % len(entries)
    literals = {entry[0]: sorted(round(float(v), 4) for v in entry[2]) for entry in entries}
    assert set(literals) == {"claude-opus-5", "gpt-6-astra"}, sorted(literals)
    assert all(len(v) == 2 for v in literals.values()), literals
    assert sum(len(v) for v in literals.values()) == 4, literals

    arms, _ = ranked
    scored = {a["label"]: a["top1"] for a in arms}
    runs, task = _staged_runs_and_task()
    for base, drawn in sorted(literals.items()):
        gw2 = sja.ADDENDUM_DIR / ("whoandwhen__all_at_once__%s-gw2.json" % base)
        measured = sorted(round(v, 4) for v in (
            scored[base + "-gw"],
            sja.score_cache(sja.addendum_predictions(gw2), runs, task, base + "-gw2")["top1"],
        ))
        assert drawn == measured, (
            "%s: the panel draws %s and the gateway caches score %s" % (base, drawn, measured))
