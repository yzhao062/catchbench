r"""Hold the POST generalization block to its record, its declaration, and the claims it may not make.

Four failures are worth failing loudly for, and none of them is visible to a reader of the LaTeX.

The first is a number the record does not support. Every printed cell here is a three- or four-decimal
figure that looks equally plausible at 0.694, 0.794 or 0.894, and the two corpora are new, so no
committed board cell stands beside them for a reader to check against. The tests therefore compare
the printed cells against the record's own fields, against the full-precision comment lines the block
carries beside them, and against the second place the record stores every arm score and every effect;
a corrupted record fails and a corrupted block fails.

The second is a reading the batch did not record. Section G's branch is a word, and the four words
are not interchangeable: this record reads ``unresolved-or-modest`` with both corpora unresolved, and
the neighbouring word would turn an audit that resolved nothing into a replication. The branch is
therefore checked against the record's own per-corpus flags in both directions, the three branches
the record did not take are checked to be absent from the block, and the phrase that would state the
strongest of them is banned outright while this record reads as it does.

The third is a claim the audit is not entitled to make. Section I bars four: no unseen-task-family
claim for ScienceWorld, no independent-scaffold claim for OpenHands, no board-evaluation claim for
either corpus, and, since the measured overlap is one shared issue rather than none, no
disjoint-issue-set claim. The caption is required to state each bar, so a blanket ban on the words
would ban the disclosure with the claim; the guard counts occurrences against the record's own
sentences plus the labels they are stated under, and a word that appears once more than those
sentences account for fails. Two corpora printed side by side also invite a sentence about which one
kept its increment, and nothing here tests a difference between them: they are separate refits with
separate populations, separate fold protocols and separate resampling units.

The fourth is an unescaped percent sign, which this repository has already been bitten by once. A
caption elsewhere in this appendix kept ``95\%%`` after its ``%`` interpolation was removed, the
second sign reached LaTeX, and it commented out the rest of the generated line: no build warning, no
--check failure because source and output shared the error, and no test failure because every test
read a source string. The guard here reads the bytes the command line writes.

The record is read and the declaration is imported. ``catchbench.post_generalization_audit`` holds the
frozen arms, contrasts, estimands, primary cell, branch roster and resampling units, so the constants
this block prints are checked against the module that produced them rather than against a copy.
"""
from __future__ import annotations

import ast
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(TOOLS))

import emit_live_size_audit_table as elsat  # noqa: E402
import emit_post_generalization_table as epg  # noqa: E402

from catchbench import post_generalization_audit as audit  # noqa: E402

RECORD = TOOLS / "post_generalization_audit_results.json"
FRAGMENT = TOOLS / "post_generalization_table.tex"

_ROW_END = r" \\"

# The full-precision comment lines the block carries beside every printed cell. Each is anchored at
# both ends, so a line that grew or lost a field stops matching rather than matching loosely.
_ARM_COMMENT = re.compile(
    r"^% (\S+): arm (.+?): role=(.+?); layer=(\S+); n_columns=(\d+); "
    r"seed_averaged_oof=([\d.]+); mean_fold=([\d.]+)$")
_ROLE_COMMENT = re.compile(
    r"^% (\S+): role=(.+?); labelled_as=(.+?); definition=(.+?); rng_label=(\S+); "
    r"base_seed=(\d+)$")
_POINT_COMMENT = re.compile(
    r"^% (\S+): point=(-?[\d.]+); interval=\[(-?[\d.]+), (-?[\d.]+)\]; method=(.+?); axis=(.+?); "
    r"unit=(.+?); n_units=(\d+)$")
_REPRODUCTION_COMMENT = re.compile(
    r"^% (\S+): mean_fold_difference=(-?[\d.]+); interval=(\S+)$")

# Words that would turn an unresolved pair into a result. "survives" is deliberately absent: it is
# the verb of the record's own question, which asks how much of the increment survives, and banning
# it would ban the question. "replicates" is banned only in the two shapes that make a claim, because
# the bare word is a bootstrap field name that every comment line here carries.
FORBIDDEN = ("significant", "significance", "p =", "confirms", "demonstrates", "proves",
             "generalizes", "replicates the", "replicates on", "holds on both", "rules out",
             "robust to", "as predicted")

# Section G's two conditions, in the affirmative shapes a reading sentence would use to assert one
# of them. This record's reading states both conditions in the negative, so the bare phrases are not
# banned: "neither gives an informative stable-positive boundary" is the disclosure, and banning the
# words would ban it, exactly as banning "unseen" would ban section I's bar. What is banned is a
# subject other than "neither" reaching either verb. The guard below turns these on only when the
# record's own two rosters are empty, so a later batch that did show attenuation is described rather
# than refused, and it is what catches a branch reading rewritten in place: asserting that the
# emitter's own constant appears proves nothing when the corruption is in that constant.
AFFIRMATIVE_READINGS = ("shows substantial attenuation", "show substantial attenuation",
                        "showing substantial attenuation", "corpora give informative "
                        "stable-positive", "other gives an informative stable-positive",
                        "corpus gives an informative stable-positive")

# Every occurrence of these has to be accounted for by the record's own sentences plus the labels the
# caption states them under. Each is a word the caption is required to carry in a refusal and would
# be a claim anywhere else: "unseen" and "scaffold" are two of section I's bars by name, and
# "disjoint" is the bar whose label the measured overlap chooses.
ACCOUNTED = ("unseen", "scaffold", "disjoint")

# Two corpora printed side by side order themselves, and the caption may not read that as a verdict:
# these are separate refits over separate populations under separate fold protocols, and nothing here
# tests a difference between them. Two corpora landing on the same reading invite the opposite
# sentence just as strongly, so equivalence is banned with superiority.
COMPARATIVE = ("outperforms", "beats", "better than", "worse than", "larger than", "smaller than",
               "wins", "superior", "strongest", "weakest", "no difference", "equivalent",
               "indistinguishable", "than on")


@pytest.fixture(scope="module")
def record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def generated():
    return epg.build()


@pytest.fixture(scope="module")
def rendered():
    """The bytes the command line writes, so the percent guard reads output rather than source."""
    result = _cli()
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    return result.stdout


@pytest.fixture
def mutable(record):
    return copy.deepcopy(record)


def _cli(*args, paper_env=None):
    env = os.environ.copy()
    env.pop("CATCHBENCH_PAPER_DIR", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if paper_env is not None:
        env["CATCHBENCH_PAPER_DIR"] = str(paper_env)
    return subprocess.run([sys.executable, str(Path(epg.__file__)), *map(str, args)],
                          capture_output=True, env=env, timeout=600)


def _cells(line: str) -> list[str]:
    return line.removesuffix(_ROW_END).split(" & ")


def _table_rows(generated: str) -> list[list[str]]:
    """The rows of the one tabular, located by its own label rather than by position."""
    lines = generated.splitlines()
    marker = r"\label{tab:post-generalization}"
    assert lines.count(marker) == 1, "the block carries one labelled float"
    start = lines.index(r"\begin{tabular}{@{}lrr@{}}")
    stop = lines.index(r"\end{tabular}")
    return [_cells(line) for line in lines[start:stop] if line.endswith(_ROW_END)]


def _labelled_row(generated: str, label: str) -> list[str]:
    """The one body row whose first cell is ``label``, without that cell."""
    rows = [row[1:] for row in _table_rows(generated) if row[0] == label]
    assert len(rows) == 1, f"{label!r} names {len(rows)} rows"
    return rows[0]


def _rendered(generated: str) -> str:
    """Everything a reader of the compiled PDF meets: the block without its comment lines.

    The counting guard below is scoped to this rather than to the whole block, because the provenance
    lines legitimately echo record field names such as ``no_disjoint_issue_set`` and counting those
    against the caption's sentences would measure the wrong thing. The outright bans stay on the whole
    block, where a word planted in a comment still fails.
    """
    return " ".join(line for line in generated.splitlines() if not line.startswith("%"))


def _comments(generated: str, pattern: re.Pattern) -> dict[str, tuple]:
    found = {}
    for line in generated.splitlines():
        match = pattern.match(line)
        if match:
            assert match.group(1) not in found, f"duplicate comment line for {match.group(1)}"
            found[match.group(1)] = match.groups()[1:]
    return found


def _arm_comments(generated: str) -> dict[tuple[str, str], tuple]:
    found = {}
    for line in generated.splitlines():
        match = _ARM_COMMENT.match(line)
        if match:
            key = (match.group(1), match.group(2))
            assert key not in found, f"duplicate arm comment for {key}"
            found[key] = match.groups()[2:]
    return found


def _set_branch(mutable: dict, branch: str, attenuating=(), bounding=()) -> None:
    """Put the record in one of section G's four branches, flags and rosters together.

    The emitter refuses a branch whose rosters contradict its own per-corpus flags, which is the point
    of that refusal, so a test that reaches another branch has to move both.
    """
    reading = mutable["branch"]
    reading["branch"] = branch
    reading["corpora_showing_substantial_attenuation"] = sorted(attenuating)
    reading["corpora_giving_an_informative_stable_positive_boundary"] = sorted(bounding)
    for corpus, flags in reading["per_corpus"].items():
        flags["substantial_attenuation"] = corpus in attenuating
        flags["informative_stable_positive_boundary"] = corpus in bounding
        flags["unresolved"] = corpus not in attenuating and corpus not in bounding


# --- the printed block against the record it came from --------------------------------------------

def test_every_printed_arm_cell_is_the_seed_averaged_score_and_not_the_board_number(generated,
                                                                                    record):
    """Section E defines the estimands on one of two quantities the same arrays both support.

    The mean of the 25 fold AUCs is what a board prints and it is not what ``L``, ``D`` and ``S`` are
    differences of. On ScienceWorld the two differ in the third decimal at three of the four arms, so
    a table that printed one under a caption describing the other would be wrong at every cell and
    would look right at every cell.
    """
    for arm in epg._ARM_ORDER:
        printed_name = elsat._printed_arm_names()[arm]
        cells = _labelled_row(generated, printed_name)
        assert len(cells) == len(epg._CORPUS_ORDER)
        for cell, corpus in zip(cells, epg._CORPUS_ORDER):
            scored = record["corpora"][corpus]["arms"][arm]
            assert cell == f"{scored['seed_averaged_oof_roc_auc']:.3f}"
            if f"{scored['mean_fold_roc_auc']:.3f}" != f"{scored['seed_averaged_oof_roc_auc']:.3f}":
                assert cell != f"{scored['mean_fold_roc_auc']:.3f}"


def test_every_printed_effect_and_interval_is_the_records_own(generated, record):
    """The three estimands and their intervals, cell by cell against the effect rows."""
    rows = {row["cell"]: row for row in record["effects"]}
    names = elsat._printed_arm_names()
    for estimand in epg._ESTIMAND_ORDER:
        points = _labelled_row(generated, epg._estimand_label(record, estimand, names))
        intervals = _labelled_row(generated, epg._interval_label(estimand))
        for point, interval, corpus in zip(points, intervals, epg._CORPUS_ORDER):
            row = rows[corpus + "." + estimand]
            marker = epg._PRIMARY_MARKER if row["role"] == epg._PRIMARY else ""
            assert point == "$" + f"{row['point']:+.4f}" + marker + "$"
            low, high = row["interval_95"]
            assert interval == "$[" + f"{low:+.4f}" + ", " + f"{high:+.4f}" + "]$"


def test_every_printed_number_is_its_comment_line_value_rounded(generated, record):
    """The printed cell, the comment line beside it and the record must be one number.

    The comment lines are the only place a reader can re-derive a row from, so they are checked
    against the record at full precision and the printed cells are checked against them. Corrupting
    either one alone breaks the chain.
    """
    points = _comments(generated, _POINT_COMMENT)
    reproduction = _comments(generated, _REPRODUCTION_COMMENT)
    roles = _comments(generated, _ROLE_COMMENT)
    arms = _arm_comments(generated)
    rows = {row["cell"]: row for row in record["effects"]}
    assert set(points) == set(reproduction) == set(roles) == set(rows)
    for cell, row in rows.items():
        point, low, high, method, axis, unit, units = points[cell]
        assert point == f"{row['point']:.12f}"
        assert [low, high] == [f"{value:.12f}" for value in row["interval_95"]]
        assert [method, axis, unit] == [row["interval_method"], row["interval_axis"],
                                        row["resampling_unit"]]
        assert units == str(row["n_units"])
        assert reproduction[cell][0] == (
            f"{row['reproduction_quantity']['mean_fold_difference']:.12f}")
        assert reproduction[cell][1] == str(row["reproduction_quantity"]["interval"])
        assert roles[cell][0] == row["role"]
        assert roles[cell][1] == row["labelled_as"]
        assert roles[cell][2] == row["definition"]
    for corpus in epg._CORPUS_ORDER:
        for arm in epg._ARM_ORDER:
            scored = record["corpora"][corpus]["arms"][arm]
            fields = arms[(corpus, arm)]
            assert fields[0] == scored["role"]
            assert fields[1] == scored["layer"]
            assert fields[2] == str(scored["n_columns"])
            assert fields[3] == f"{scored['seed_averaged_oof_roc_auc']:.12f}"
            assert fields[4] == f"{scored['mean_fold_roc_auc']:.12f}"
    # and the printed cells are those same comment values rounded, not derived again
    names = elsat._printed_arm_names()
    for arm in epg._ARM_ORDER:
        for cell, corpus in zip(_labelled_row(generated, names[arm]), epg._CORPUS_ORDER):
            assert cell == f"{float(arms[(corpus, arm)][3]):.3f}"
    for estimand in epg._ESTIMAND_ORDER:
        printed = _labelled_row(generated, epg._estimand_label(record, estimand, names))
        for cell, corpus in zip(printed, epg._CORPUS_ORDER):
            exact = float(points[corpus + "." + estimand][0])
            assert cell.startswith("$" + f"{exact:+.4f}")


def test_the_six_cells_are_both_corpora_at_the_three_declared_estimands_in_order(generated, record):
    """Section E declares three estimands at two corpora, and the table prints six cells."""
    assert [row["cell"] for row in record["effects"]] == [
        corpus + "." + estimand
        for corpus in epg._CORPUS_ORDER for estimand in epg._ESTIMAND_ORDER]
    names = elsat._printed_arm_names()
    labels = [row[0] for row in _table_rows(generated)]
    for estimand in epg._ESTIMAND_ORDER:
        assert labels.count(epg._estimand_label(record, estimand, names)) == 1
        assert labels.count(epg._interval_label(estimand)) == 1


def test_the_printed_populations_are_the_declared_and_observed_ones(generated, record):
    """Section C blocks on a population that moved, so the table prints the one it froze."""
    runs = _labelled_row(generated, epg._RUNS_LABEL)
    units = _labelled_row(generated, epg._UNITS_LABEL)
    printed_units = _labelled_row(generated, epg._UNIT_LABEL)
    prose = " ".join(generated.split())
    for cell, unit_count, unit, corpus in zip(runs, units, printed_units, epg._CORPUS_ORDER):
        observed = record["populations"][corpus]["observed"]
        frozen = record["declaration"]["frozen_before_the_run"]["declared_population"][corpus]
        assert observed == frozen == record["populations"][corpus]["declared"]
        assert cell == (f"{observed['n_runs']} ({observed['n_failed']}, "
                        f"{observed['n_solved']})")
        assert unit == record["corpora"][corpus]["resampling_unit"]
        rows = [row for row in record["effects"] if row["corpus"] == corpus]
        assert unit_count == str(rows[0]["n_units"])
        assert epg._population_sentence(record, corpus, epg._CORPUS_PRINTED[corpus]) in prose


def test_the_population_sentence_takes_its_shape_from_the_record(generated, record):
    """One corpus has repeated attempts at an issue and the other has no unit to repeat."""
    prose = " ".join(generated.split())
    grouped, ungrouped = epg._CORPUS_ORDER
    observed = record["populations"][grouped]["observed"]
    assert (f"over {observed['n_distinct_group']} distinct issues, "
            f"{observed['group_multiplicity']['1']} attempted once and "
            f"{observed['group_multiplicity']['2']} attempted twice, and "
            f"{observed['n_missing_group']} rows with no issue identifier.") in prose
    assert "n_distinct_group" not in record["populations"][ungrouped]["observed"]
    assert (epg._CORPUS_PRINTED[ungrouped] + " carries "
            f"{record['populations'][ungrouped]['observed']['n_runs']} runs") in prose
    assert epg._population_sentence(record, ungrouped, "X").endswith("solved.")


def test_the_dagger_marks_exactly_the_primary_cell_and_it_is_the_nominated_one(generated, record):
    """Section E nominated one of the six before any of them existed; the mark says which."""
    frozen = record["declaration"]["frozen_before_the_run"]["primary_estimand"]
    primary = [row for row in record["effects"] if row["role"] == epg._PRIMARY]
    assert len(primary) == 1
    assert primary[0]["cell"] == record["primary"]["cell"]
    assert frozen == {"corpus": primary[0]["corpus"], "estimand": primary[0]["estimand"]}
    marked = [(estimand, corpus)
              for estimand in epg._ESTIMAND_ORDER
              for cell, corpus in zip(
                  _labelled_row(generated,
                                epg._estimand_label(record, estimand,
                                                    elsat._printed_arm_names())),
                  epg._CORPUS_ORDER)
              if epg._PRIMARY_MARKER in cell]
    assert marked == [(frozen["estimand"], frozen["corpus"])]
    low, high = primary[0]["interval_95"]
    assert ("The primary cell is marked $" + epg._PRIMARY_MARKER + "$: "
            + epg._CORPUS_PRINTED[frozen["corpus"]] + " $" + frozen["estimand"] + "$ = "
            + epg._effect(primary[0]["point"]) + ", " + epg._interval(low, high)
            + ", which is " + primary[0]["labelled_as"] + ".") in " ".join(generated.split())


def test_the_printed_reading_is_the_flag_the_record_carries_for_that_corpus(generated, record):
    """Section G's two conditions are flags, and the printed word is read off them."""
    cells = _labelled_row(generated, epg._READING_LABEL)
    for cell, corpus in zip(cells, epg._CORPUS_ORDER):
        flags = record["branch"]["per_corpus"][corpus]
        expected = epg._UNRESOLVED_READING
        for flag, printed in epg._READINGS:
            if flags[flag]:
                expected = printed
        assert cell == expected


def test_the_fold_protocol_printed_is_the_frozen_one_for_each_corpus(generated, record):
    """Section D's protocols differ between the corpora, and the caption prints both verbatim."""
    frozen = record["declaration"]["frozen_before_the_run"]["fold_protocol"]
    prose = " ".join(generated.split())
    for corpus in epg._CORPUS_ORDER:
        protocol = record["fold_protocol"][corpus]["protocol"]
        assert protocol == frozen[corpus]
        assert (epg._CORPUS_PRINTED[corpus] + " folds are " + epg._texttt(protocol) + ".") in prose
        assert f"% Corpus {corpus}: folds {protocol};" in generated
    assert record["fold_protocol"][epg._CORPUS_ORDER[0]]["grouped_by"] is not None
    assert record["fold_protocol"][epg._CORPUS_ORDER[1]]["grouped_by"] is None


def test_the_overlap_the_caption_states_is_the_measured_one_with_its_shared_issue(generated,
                                                                                  record):
    """Section H reports the measured overlap and the appendix discloses what it shared."""
    overlap = record["swegym_overlap"]
    assert overlap["n_shared_issues"] == len(overlap["shared_issues"]) == 1
    prose = " ".join(generated.split())
    assert (f"Measured overlap between the {overlap['n_distinct_openhands_issues']} distinct issues "
            f"of " + epg._CORPUS_PRINTED[epg._CORPUS_ORDER[0]] + " and the "
            f"{overlap['n_distinct_swegym_issues']} distinct issues of the scored "
            + epg._OVERLAP_CORPUS + f" population over {overlap['n_swegym_rows']} rows: one shared "
            "issue, " + epg._texttt(overlap["shared_issues"][0])
            + ", read from the raw shards.") in prose
    assert epg._OVERLAP_LABELS[True] + ": " in prose
    assert epg._OVERLAP_LABELS[False] not in prose


def test_the_arm_names_and_arms_are_the_live_size_audit_tables_own(generated, record):
    """Section B says the arms are Part A's, unchanged; a rename there moves this table."""
    assert epg._ARM_ORDER is elsat._ARM_ORDER
    assert sorted(epg._ARM_ORDER) == sorted(audit.ARM_ROLES)
    names = elsat._printed_arm_names()
    labels = [row[0] for row in _table_rows(generated)]
    for corpus in epg._CORPUS_ORDER:
        assert sorted(record["corpora"][corpus]["arms"]) == sorted(epg._ARM_ORDER)
        for arm in epg._ARM_ORDER:
            assert record["corpora"][corpus]["arm_roles"][arm] == audit.ARM_ROLES[arm]
    for arm in epg._ARM_ORDER:
        assert labels.count(names[arm]) == 1


def test_the_corpus_names_are_the_hub_roster_the_run_recorded(generated, record):
    """Neither corpus prints on another table, so the record's own roster is what vouches."""
    for corpus in epg._CORPUS_ORDER:
        printed = epg._CORPUS_PRINTED[corpus]
        assert printed in record["hub_revisions"]
        assert epg._printed_corpus(record, corpus) == printed
    assert epg._OVERLAP_CORPUS in record["hub_revisions"]
    header = [row for row in generated.splitlines() if row.startswith("Arm or quantity & ")]
    assert header == [" & ".join([epg._HEADER_LABEL,
                                  *[epg._CORPUS_PRINTED[c] for c in epg._CORPUS_ORDER]]) + _ROW_END]


def test_the_frozen_constants_are_the_audit_modules_own():
    """The declaration's frozen quantities live in one module; this table reads them, not a copy."""
    assert sorted(epg._CORPUS_ORDER) == sorted(audit.DECLARED_POPULATION)
    assert epg._ESTIMAND_ORDER == audit.ESTIMANDS
    assert sorted(epg._BRANCH_READING) == sorted(audit.BRANCHES)
    assert epg._DERIVED_ESTIMAND == audit.ESTIMANDS[-1]
    assert epg._ROW_UNIT == audit.RESAMPLING_UNIT[epg._CORPUS_ORDER[1]]
    assert epg._CONTRAST_OF == {"L": "linear_contrast", "D": "matched_contrast"}
    assert audit.LINEAR_CONTRAST[0] in audit.ARM_ROLES
    assert audit.MATCHED_CONTRAST[0] in audit.ARM_ROLES
    assert epg._PRIMARY in epg._ROLES


# --- the red direction: records this block refuses to print ----------------------------------------

def test_an_arm_score_that_disagrees_with_its_own_corpus_block_is_refused(mutable):
    """The record stores every arm score twice; a table that read one would agree with itself."""
    mutable["effects"][0]["arm_scores"]["size (flat)"]["seed_averaged_oof_roc_auc"] = 0.5
    with pytest.raises(ValueError, match="the corpus block carries"):
        epg.table(mutable)


def test_an_effect_that_disagrees_with_its_own_corpus_block_is_refused(mutable):
    mutable["effects"][0]["point"] = 0.5
    with pytest.raises(ValueError, match="the corpus block's own contrast says"):
        epg.table(mutable)


def test_a_contrast_naming_arms_other_than_the_frozen_pair_is_refused(mutable):
    """Section E froze which arm is the high one; a swap would flip a sign nobody could see."""
    mutable["corpora"]["openhands"]["effects"]["L"]["high"] = "size (spline)"
    with pytest.raises(ValueError, match="not the frozen"):
        epg.table(mutable)


def test_a_reproduction_quantity_that_has_been_given_an_interval_is_refused(mutable):
    mutable["effects"][0]["reproduction_quantity"]["interval"] = [0.0, 1.0]
    with pytest.raises(ValueError, match="gives the mean of fold AUCs an interval"):
        epg.table(mutable)


def test_a_cell_set_that_is_not_the_declared_corpora_and_estimands_in_order_is_refused(mutable):
    mutable["effects"][0], mutable["effects"][1] = mutable["effects"][1], mutable["effects"][0]
    with pytest.raises(ValueError, match="not the declared corpora at the declared estimands"):
        epg.table(mutable)


def test_two_primary_cells_are_refused(mutable):
    primary = next(row for row in mutable["effects"] if row["role"] == epg._PRIMARY)
    mutable["effects"][0]["role"] = epg._PRIMARY
    mutable["effects"][0]["labelled_as"] = primary["labelled_as"]
    with pytest.raises(ValueError, match="cells carry the primary role"):
        epg.table(mutable)


def test_a_primary_cell_other_than_the_nominated_one_is_refused(mutable):
    """Section E's nomination happened before the numbers; a moved one happened after them."""
    primary = next(row for row in mutable["effects"] if row["role"] == epg._PRIMARY)
    other = mutable["effects"][0]
    primary["role"], other["role"] = other["role"], primary["role"]
    primary["labelled_as"], other["labelled_as"] = other["labelled_as"], primary["labelled_as"]
    with pytest.raises(ValueError, match="the record's primary is"):
        epg.table(mutable)


def test_an_unrecognised_role_is_refused(mutable):
    mutable["effects"][0]["role"] = "secondary"
    with pytest.raises(ValueError, match="unrecognised role"):
        epg.table(mutable)


def test_two_cells_sharing_a_role_and_labelled_differently_are_refused(mutable):
    mutable["effects"][1]["labelled_as"] = "something else"
    with pytest.raises(ValueError, match="labelled differently"):
        epg.table(mutable)


def test_a_point_outside_its_own_interval_is_refused(mutable):
    mutable["effects"][3]["point"] = 0.9
    mutable["corpora"]["scienceworld"]["effects"]["L"]["seed_averaged_oof_difference"] = 0.9
    with pytest.raises(ValueError, match="is outside its own interval"):
        epg.table(mutable)


def test_a_bootstrap_that_did_not_run_the_frozen_draws_is_refused(mutable):
    mutable["effects"][0]["replicates"] = 5000
    with pytest.raises(ValueError, match="draws against the frozen"):
        epg.table(mutable)


def test_a_population_that_is_not_the_declared_one_is_refused(mutable):
    """Section C blocks on a moved population rather than scoring it."""
    mutable["populations"]["openhands"]["observed"]["n_failed"] = 311
    with pytest.raises(ValueError, match="is not the declared"):
        epg.table(mutable)


def test_a_branch_whose_roster_contradicts_its_own_flags_is_refused(mutable):
    mutable["branch"]["corpora_showing_substantial_attenuation"] = ["openhands"]
    with pytest.raises(ValueError, match="flags say"):
        epg.table(mutable)


def test_an_unresolved_flag_that_contradicts_the_two_conditions_is_refused(mutable):
    mutable["branch"]["per_corpus"]["openhands"]["unresolved"] = False
    with pytest.raises(ValueError, match="contradicts the two conditions"):
        epg.table(mutable)


def test_an_unrecognised_branch_is_refused(mutable):
    mutable["branch"]["branch"] = "whichever"
    with pytest.raises(ValueError, match="unrecognised section G branch"):
        epg.table(mutable)


def test_a_spline_that_reached_the_dependency_columns_is_refused(mutable):
    """Section B splines the flat matrix; a wider block is a different arm under this caption."""
    columns = mutable["verification"]["per_corpus"]["openhands"]["column_boundaries"]
    columns["splined_columns"] = [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="splines the flat matrix and nothing else"):
        epg.table(mutable)


def test_a_fold_with_no_minority_class_row_is_refused(mutable):
    mutable["fold_protocol"]["openhands"]["minimum_class_count_in_any_fold"] = 0
    with pytest.raises(ValueError, match="a fold with no minority-class row"):
        epg.table(mutable)


def test_a_grouped_protocol_at_a_row_resampled_corpus_is_refused(mutable):
    """Section D and section F name the same unit, and the caption states the difference."""
    mutable["fold_protocol"]["scienceworld"]["grouped_by"] = "task_name"
    with pytest.raises(ValueError, match="section D and section F name the same unit"):
        epg.table(mutable)


def test_a_fold_protocol_that_is_not_the_frozen_one_is_refused(mutable):
    mutable["fold_protocol"]["scienceworld"]["protocol"] = "KFold(n_splits=5)"
    with pytest.raises(ValueError, match="is not the frozen"):
        epg.table(mutable)


def test_a_batch_the_preflight_gate_did_not_admit_is_refused(mutable):
    mutable["preflight"]["gate"]["admits_the_batch"] = False
    with pytest.raises(ValueError, match="the preflight gate did not admit this batch"):
        epg.table(mutable)


def test_a_declared_fit_count_the_batch_did_not_perform_is_refused(mutable):
    mutable["settings"]["fits_performed"] = 199
    with pytest.raises(ValueError, match="section J declares"):
        epg.table(mutable)


def test_a_run_that_registered_a_board_entrant_is_refused(mutable):
    """Section H registers none, and a table that printed one would announce a board this is not."""
    mutable["board_entrants_added"] = ["post.openhands.auditable (size+deps)"]
    with pytest.raises(ValueError, match="registers no board entrant"):
        epg.table(mutable)


def test_an_overlap_whose_count_disagrees_with_its_own_list_is_refused(mutable):
    mutable["swegym_overlap"]["n_shared_issues"] = 2
    with pytest.raises(ValueError, match="shared issues and names"):
        epg.table(mutable)


def test_a_corpus_the_hub_roster_does_not_carry_is_refused(mutable):
    mutable["hub_revisions"].pop(epg._CORPUS_PRINTED["scienceworld"])
    with pytest.raises(ValueError, match="hub roster"):
        epg.table(mutable)


def test_rows_whose_shared_sentences_disagree_are_refused(mutable):
    """The caption states one conditionality sentence for six rows, so the six have to agree."""
    mutable["effects"][2]["conditional_on_the_saved_fits"] = "something adjacent to that"
    with pytest.raises(ValueError, match="different conditionality sentences"):
        epg.table(mutable)


# --- the branches section G and section I require the caption to take ------------------------------

@pytest.mark.parametrize("branch, attenuating, bounding", [
    ("substantial-attenuation", ("openhands", "scienceworld"), ()),
    ("mixed-informative", ("openhands",), ("scienceworld",)),
    ("stable-positive-both", (), ("openhands", "scienceworld")),
    ("unresolved-or-modest", (), ()),
])
def test_every_section_g_branch_has_a_caption_that_reads_it(mutable, branch, attenuating, bounding):
    """A caption written for one branch would be wrong rather than incomplete in the others."""
    _set_branch(mutable, branch, attenuating, bounding)
    prose = " ".join(epg.table(mutable).split())
    assert "The recorded branch is " + branch + ". " + epg._BRANCH_READING[branch] in prose
    for other in epg._BRANCH_READING:
        if other != branch:
            assert epg._BRANCH_READING[other] not in prose
    for corpus in epg._CORPUS_ORDER:
        expected = epg._UNRESOLVED_READING
        if corpus in attenuating:
            expected = epg._READINGS[0][1]
        elif corpus in bounding:
            expected = epg._READINGS[1][1]
        assert epg._reading(mutable["branch"], corpus) == expected
    cells = _labelled_row(epg.table(mutable), epg._READING_LABEL)
    assert cells == [epg._reading(mutable["branch"], corpus) for corpus in epg._CORPUS_ORDER]


def test_the_zero_overlap_branch_releases_the_claim_and_says_so(mutable):
    """Section I bars the disjoint claim unless the overlap measured zero; that branch works."""
    mutable["swegym_overlap"]["n_shared_issues"] = 0
    mutable["swegym_overlap"]["shared_issues"] = []
    prose = " ".join(epg.table(mutable).split())
    assert epg._OVERLAP_LABELS[False] + ":" in prose
    assert epg._OVERLAP_LABELS[True] + ":" not in prose
    assert "no shared issues, read from the raw shards." in prose


def test_a_batch_with_a_recorded_failure_says_so_in_the_caption_and_in_the_rows(mutable):
    """Section H asks for every failure, so a lost corpus reaches the caption and the table."""
    mutable["failures"] = [{"kind": "spline_fit", "corpus": "scienceworld", "traceback": "..."}]
    block = epg.table(mutable)
    prose = " ".join(block.split())
    assert "The batch recorded one failure, named in the rows below" in prose
    assert r"\multicolumn{3}{@{}l}{Recorded failure: \texttt{scienceworld}" in block
    assert "% failure: " in block


# --- claims this block may not make ----------------------------------------------------------------

def test_the_block_claims_no_result_the_record_did_not_support(generated):
    """An unresolved pair licenses no verb stronger than the record's own."""
    lowered = " ".join(generated.split()).lower()
    hits = [word for word in FORBIDDEN if word in lowered]
    assert hits == [], f"the block claims {hits}"


def test_the_block_compares_no_two_corpora_in_either_direction(generated):
    """Two separate refits over two populations support no verdict about the pair."""
    lowered = " ".join(generated.split()).lower()
    hits = [word for word in COMPARATIVE if word in lowered]
    assert hits == [], f"the block sets the corpora against each other: {hits}"


def test_the_block_does_not_read_a_branch_the_record_did_not_record(generated, record):
    """The neighbouring branch would turn an audit that resolved nothing into a replication."""
    branch = record["branch"]["branch"]
    assert branch == "unresolved-or-modest"
    for corpus, flags in record["branch"]["per_corpus"].items():
        assert flags["unresolved"] and not flags["substantial_attenuation"]
        assert not flags["informative_stable_positive_boundary"], corpus
    prose = " ".join(generated.split())
    assert epg._BRANCH_READING[branch] in prose
    for other in epg._BRANCH_READING:
        if other != branch:
            assert epg._BRANCH_READING[other] not in prose
    lowered = prose.lower()
    assert lowered.count("the recorded branch is " + branch) == 1
    # The reading sentence is a constant in the emitter, so asserting it is present proves nothing
    # about a sentence rewritten in place. These phrases are not derived from it: the record names no
    # corpus under either roster, so no affirmative statement that one showed either condition can be
    # true of this batch, whichever branch constant it was written into.
    assert not record["branch"]["corpora_showing_substantial_attenuation"]
    assert not record["branch"]["corpora_giving_an_informative_stable_positive_boundary"]
    hits = [phrase for phrase in AFFIRMATIVE_READINGS if phrase in lowered]
    assert hits == [], f"the block reads a condition no corpus is flagged for: {hits}"


def test_the_words_the_caption_must_carry_appear_only_in_the_sentences_that_account_for_them(
        generated, record):
    """A blanket ban would ban the disclosure, so the guard counts instead.

    Section I bars four claims by name and the caption is required to state each bar. One more
    occurrence than the record's sentences and their labels account for is a claim nobody declared.
    """
    printed = _rendered(generated).lower()
    accounted = " ".join([*record["interpretation"].values(), *epg._LABELS.values(),
                          epg._OVERLAP_LABELS[bool(record["swegym_overlap"]["shared_issues"])]
                          ]).lower()
    for word in ACCOUNTED:
        assert printed.count(word) == accounted.count(word), (
            f"{word!r} appears {printed.count(word)} times in the rendered text and "
            f"{accounted.count(word)} times in the sentences that account for it")


def test_the_four_barred_claims_are_stated_under_their_own_labels(generated, record):
    """Section I's bars, each with the record's own sentence behind the label that names it."""
    prose = " ".join(generated.split())
    printed = " ".join(_rendered(generated).split())
    for key in epg._BARRED:
        label = epg._LABELS[key]
        assert prose.count(label + ": ") == 1, f"{label} is stated {prose.count(label)} times"
        sentence = record["interpretation"][key]
        if epg._SCAFFOLD_IDENTIFIER in sentence:
            head, tail = sentence.split(epg._SCAFFOLD_IDENTIFIER)
            assert label + ": " + head + epg._texttt(epg._SCAFFOLD_IDENTIFIER) + tail in prose
        else:
            assert label + ": " + sentence in prose
    assert (epg._OVERLAP_LABELS[True] + ": " + record["interpretation"]["no_disjoint_issue_set"]
            in prose)
    # The identifier reaches the caption escaped and nowhere else: an unescaped underscore in the
    # printed text is a subscript, and the comment lines that carry the raw sentence are not parsed.
    assert epg._SCAFFOLD_IDENTIFIER not in printed
    assert epg._texttt(epg._SCAFFOLD_IDENTIFIER) in printed
    assert r"That board is Table~\ref{" + epg.POST_BOARD_TABLE + "}." in printed


def test_every_required_record_sentence_reaches_the_caption_verbatim(generated, record):
    """The record and the manuscript may not carry different reasons for the same refusal.

    Scoped to what a reader of the compiled PDF meets. The provenance lines carry every
    interpretation sentence as well, so a sweep over the whole block would pass on a caption that had
    dropped one: the comment copy would answer for it, and the manuscript would carry the number
    without the refusal that goes with it.
    """
    prose = " ".join(_rendered(generated).split())
    for key in ("exploratory", "failing_to_reject_is_not_stability", "grouped_protocol_differs",
                "no_disjoint_issue_set", "no_unseen_task_family", "not_a_board", "one_batch"):
        assert record["interpretation"][key] in prose, f"the caption drops {key}"
    assert record["question"] in prose
    assert record["branch"]["rule"] in prose
    assert record["branch"]["no_promotion"] in prose
    assert record["swegym_overlap"]["no_row_is_dropped"] in prose
    assert record["settings"]["bootstrap"]["s_carries"] in prose
    assert record["primary"]["labelled_as"] in prose
    for corpus in epg._CORPUS_ORDER:
        assert record["fold_protocol"][corpus]["establishes"] in prose
        assert record["populations"][corpus]["eligibility"] in prose
    assert record["branch"]["per_corpus"]["openhands"]["threshold_is"] in prose


def test_the_caption_says_the_audit_was_declared_after_its_neighbours_were_read(generated, record):
    """The chronology clause, which is what keeps this table exploratory whichever way it came out.

    It is checked as its own test rather than folded into the verbatim sweep because it is the one
    sentence a reviewer is entitled to look for: the audit was proposed after Part A's result was
    read, so it is a prospectively specified extension of an exploratory analysis rather than an
    independent confirmatory replication.
    """
    sentence = record["interpretation"]["exploratory"]
    assert "proposed after Part A's result" in sentence
    assert "stays labelled exploratory" in sentence
    assert epg._LABELS["exploratory"] + ": " + sentence in " ".join(generated.split())


def test_the_grouped_protocol_difference_is_stated_rather_than_left_to_the_reader(generated,
                                                                                  record):
    """Section D: the grouped folds and the boards' row splits are not like for like."""
    sentence = record["interpretation"]["grouped_protocol_differs"]
    assert "row-split protocol the original POST boards use" in sentence
    assert epg._LABELS["grouped_protocol_differs"] + ": " + sentence in " ".join(generated.split())


def test_the_long_notes_that_carry_markup_are_kept_in_the_comment_lines(generated, record):
    """Record prose carrying characters LaTeX would read as markup stays unparsed."""
    for note in (record["hub_revisions_are"], record["swegym_overlap"]["why_not_the_loader"],
                 record["corpora"]["openhands"]["corpus_line"]):
        with pytest.raises(ValueError, match="LaTeX special characters"):
            epg._prose(note)
    assert "% Hub revisions are " + record["hub_revisions_are"] + "." in generated
    assert "% Inputs note: " + record["inputs"]["note"] in generated
    assert ("% Overlap: why not the loader: " + record["swegym_overlap"]["why_not_the_loader"]
            in generated)
    for corpus in epg._CORPUS_ORDER:
        assert f"% Corpus {corpus}: " + record["corpora"][corpus]["corpus_line"] in generated


# --- escaping and rendering -------------------------------------------------------------------------

def test_no_generated_percent_starts_a_latex_comment(rendered):
    r"""A bare % comments out the rest of its line, silently and invisibly.

    This is not hypothetical. A caption elsewhere in this appendix kept ``95\%%`` from a string that
    had used % interpolation; the rewrite had no % operator, so both signs reached LaTeX and the
    second one ate the rest of the line. The build reported zero warnings, --check passed because
    source and output shared the error, and every test passed because they all read source strings.
    This one reads the bytes the command line writes.
    """
    offenders = []
    for number, line in enumerate(rendered.decode("utf-8").splitlines(), 1):
        if line.startswith("%"):
            continue  # a provenance comment line, which is meant to be a comment
        for index, char in enumerate(line):
            if char == "%" and (index == 0 or line[index - 1] != "\\"):
                offenders.append(f"line {number} col {index + 1}: {line[max(0, index - 40):]}")
    assert offenders == [], "unescaped % in the rendered block:\n  " + "\n  ".join(offenders)


def test_prose_escapes_a_percent_and_refuses_every_other_special():
    # The two sets are written out rather than derived from each other, because the difference
    # operator is banned in the emitter. They must still name the same characters.
    assert epg._PROSE_SPECIALS | {"%"} == epg._LATEX_SPECIALS
    assert "%" not in epg._PROSE_SPECIALS
    assert epg._prose("a 95% interval") == r"a 95\% interval"
    assert epg._prose("plain words") == "plain words"
    for markup in (r"a\b", "a{b}", "a$b", "a&b", "a#b", "a^2", "a_b", "a~b"):
        with pytest.raises(ValueError, match="LaTeX special characters"):
            epg._prose(markup)


def test_texttt_escapes_underscores_and_braces_and_refuses_the_rest():
    assert epg._texttt("agent_graph_openhands.to_steps") == (
        r"\texttt{agent\_graph\_openhands.to\_steps}")
    assert epg._texttt("StratifiedKFold(n_splits=5, shuffle=True)") == (
        r"\texttt{StratifiedKFold(n\_splits=5, shuffle=True)}")
    for bad in (r"a\b", "a$b", "a&b", "a#b", "a^b", "a~b", "a%b"):
        with pytest.raises(ValueError, match="not a printable identifier"):
            epg._texttt(bad)


def test_prose_naming_sets_its_identifiers_and_leaves_the_sentence_recoverable():
    """The record's words reach the caption whole; only the named identifiers are wrapped."""
    sentence = "the adapter imports agent_graph_openhands.to_steps directly"
    rendered = epg._prose_naming(sentence, epg._SCAFFOLD_IDENTIFIER)
    assert rendered == ("the adapter imports " + epg._texttt(epg._SCAFFOLD_IDENTIFIER)
                        + " directly")
    plain = rendered.replace(r"\texttt{", "").replace("}", "").replace("\\_", "_")
    assert plain == sentence
    with pytest.raises(ValueError, match="use the plain guard"):
        epg._prose_naming(sentence)


def test_prose_naming_refuses_a_sentence_that_names_an_identifier_twice_or_not_at_all():
    twice = "detection._LOADERS and detection._LOADERS again"
    with pytest.raises(ValueError, match="appears 2 times"):
        epg._prose_naming(twice, "detection._LOADERS")
    with pytest.raises(ValueError, match="appears 0 times"):
        epg._prose_naming("nothing named here", "detection._LOADERS")


def test_the_caption_refuses_a_roster_its_prose_cannot_describe():
    with pytest.raises(ValueError, match="outgrown the prose"):
        epg._word(len(epg._WORDS))
    assert epg._word(len(epg._ARM_ORDER)) == "four"
    assert epg._word(len(epg._ESTIMAND_ORDER)) == "three"
    assert epg._word(0) == "no"
    assert epg._times(1) == "once" and epg._times(2) == "twice"
    assert epg._times(3) == "three times"


def test_a_corpus_with_no_printed_name_stops_the_run(mutable):
    mutable["hub_revisions"].pop(epg._CORPUS_PRINTED[epg._CORPUS_ORDER[0]])
    with pytest.raises(ValueError, match="stopped recording where the corpus came from"):
        epg._printed_corpus(mutable, epg._CORPUS_ORDER[0])


# --- the emitter owns no arithmetic -----------------------------------------------------------------

def test_the_emitter_owns_no_arithmetic_of_its_own():
    """The whole reason this tool exists rather than a hand-typed table.

    A second definition of ``S`` in this repository would be a plausible number at both corpora and a
    different quantity at both: the printed columns are rounded, and section E defines the estimands
    on the seed-averaged out-of-fold quantity rather than on the mean of the fold AUCs a board prints.
    A recomputed population would be worse, because 600 minus 574 is not the number of repeated
    issues. Banning the operators is what makes both mechanical, and it has a second effect worth
    having: with no ``%`` operator anywhere, the doubled percent sign that once ate a caption line
    cannot be left behind by a rewrite.
    """
    source = Path(epg.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned = (ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.Mod, ast.MatMult)
    offenders = sorted({type(node.op).__name__ for node in ast.walk(tree)
                        if isinstance(node, ast.BinOp) and isinstance(node.op, banned)})
    assert offenders == [], f"the emitter computes: {offenders}"
    for reachable in ("math", "np", "numpy", "statistics", "scipy", "roc_auc_score",
                      "classify_branch", "corpus_reading", "post_generalization_audit"):
        assert not hasattr(epg, reachable), f"emit_post_generalization_table reaches {reachable}"
    numbers = {node.value for node in ast.walk(tree)
               if isinstance(node, ast.Constant)
               and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)}
    floats = sorted(value for value in numbers if isinstance(value, float))
    assert floats == [], f"no measured value is written down here; found {floats}"
    # every count, population and threshold the block states is read, never typed
    for typed in (600, 574, 548, 312, 288, 128, 64, 376, 26, 18, 200, 10000, 20260907, 5, 4, 25):
        assert typed not in numbers, f"{typed} is written down in the emitter"


# --- the command line and the paper check -------------------------------------------------------------

def test_cli_prints_deterministic_lf_bytes(generated, rendered):
    second = _cli()
    assert second.returncode == 0
    assert second.stderr == b""
    assert rendered == second.stdout == (generated + "\n").encode("utf-8")
    assert b"\r" not in rendered


def test_the_shipped_fragment_is_what_the_emitter_prints(rendered):
    """The .tex fragment in tools/ is a checked artifact, not a copy someone took once."""
    assert FRAGMENT.is_file(), f"{FRAGMENT.name} is missing; regenerate it from the emitter"
    assert FRAGMENT.read_bytes() == rendered


def test_a_recorded_windows_path_prints_as_a_file_name_off_windows():
    r"""The provenance line must name a file, on the platform CI runs as well as this one.

    The record stores absolute paths and the block prints only the last segment. Taking it with
    ``Path`` reads correctly here and fails on Linux, where a backslash is an ordinary character and
    ``PosixPath(...).name`` returns the whole string. That shipped: the emitter printed
    ``C:\Users\...\post_generalization_preflight.json`` into the generated block on CI while the
    committed fragment carried the file name, so the two matched on Windows alone.

    The first assertion pins the defect rather than the fix. If POSIX semantics ever changed, the
    second assertion would pass whether or not the helper did anything, and this test would go
    quietly useless.
    """
    recorded = r"C:\Users\somebody\PycharmProjects\auditablebench\tools\post_generalization_preflight.json"

    assert PurePosixPath(recorded).name == recorded, (
        "a Windows path no longer survives POSIX parsing intact, so this test no longer reproduces "
        "the condition it was written for")
    assert epg._recorded_basename(recorded) == "post_generalization_preflight.json"

    assert epg._recorded_basename("/home/runner/work/catchbench/tools/x.json") == "x.json"
    assert epg._recorded_basename("x.json") == "x.json"
    with pytest.raises(ValueError):
        epg._recorded_basename("")


def test_the_provenance_line_names_no_absolute_path(generated):
    """Whatever the record stored, the block a reader sees carries no one's home directory."""
    for line in generated.splitlines():
        assert not re.search(r"[A-Za-z]:[\\/]", line), f"an absolute path reached the block: {line}"
        assert "/home/" not in line, f"an absolute path reached the block: {line}"


@pytest.mark.parametrize("use_env", [False, True])
def test_check_accepts_exact_block(tmp_path, generated, use_env):
    (tmp_path / "09_appendix.tex").write_bytes((generated + "\n").encode("utf-8"))
    result = (_cli("--check", paper_env=tmp_path) if use_env
              else _cli("--check", "--paper", tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert b"paper is current: tab:post-generalization" in result.stdout


@pytest.mark.parametrize("corruption, message", [
    ("no_block", b"appears 0 times"), ("no_appendix", b"cannot read appendix"),
    ("cell", b"block differs"), ("caption", b"block differs"),
    ("missing_begin", b"appears 0 times"), ("missing_end", b"appears 0 times"),
    ("duplicate", b"appears 2 times"), ("reversed", b"precedes its begin marker"),
    ("whitespace", b"block differs"), ("crlf", b"block differs"),
])
def test_check_rejects_a_paper_that_does_not_carry_the_block(tmp_path, generated, corruption,
                                                             message):
    """The red direction. A checker that cannot fail proves nothing about the paper it passes.

    ``no_block`` and ``no_appendix`` are the state of the manuscript today, so they pin the exit code
    this tool returns before anything is spliced.
    """
    text = generated
    if corruption == "no_block":
        text = r"\section{An appendix that has never seen this table}"
    elif corruption == "cell":
        text = text.replace(r"& $+0.0134$ \\", r"& $+0.9999$ \\", 1)
    elif corruption == "caption":
        text = text.replace("reads the pair as unresolved or modest",
                            "reads the pair as a replication", 1)
    elif corruption == "missing_begin":
        text = text.replace(epg._BEGIN, "")
    elif corruption == "missing_end":
        text = text.replace(epg._END, "")
    elif corruption == "duplicate":
        text += "\n" + epg._BEGIN
    elif corruption == "reversed":
        text = epg._END + "\n" + epg._BEGIN
    elif corruption == "whitespace":
        text = text.replace(r"\centering", r"\centering ", 1)
    elif corruption == "crlf":
        text = text.replace("\n", "\r\n")
    assert corruption in ("no_block", "no_appendix") or text != generated, "corruption did nothing"
    if corruption != "no_appendix":
        (tmp_path / "09_appendix.tex").write_bytes(text.encode("utf-8"))
    result = _cli("--check", "--paper", tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert message in result.stdout
    if corruption == "cell":
        assert b"--- paper/09_appendix.tex" in result.stdout
        assert b"+0.9999" in result.stdout and b"+0.0134" in result.stdout


def test_check_requires_paper_path():
    result = _cli("--check")
    assert result.returncode == 2
    assert b"--check needs --paper <dir> or CATCHBENCH_PAPER_DIR" in result.stderr


def test_a_broken_record_exits_one_with_a_readable_reason(tmp_path, monkeypatch, capsys):
    """main() reports rather than tracebacks, so a checker sees a reason instead of a stack."""
    monkeypatch.setattr(epg, "RECORD", tmp_path / "absent.json")
    monkeypatch.setattr(sys, "argv", ["emit_post_generalization_table.py"])
    assert epg.main() == 1
    assert "ERROR: POST generalization table:" in capsys.readouterr().err


# --- claims about the manuscript, which need the manuscript -------------------------------------------

def test_the_post_board_table_this_caption_points_at_exists():
    """The caption sends a reader to the POST board, which is the thing neither corpus is on."""
    paper_dir = os.environ.get("CATCHBENCH_PAPER_DIR")
    if not paper_dir:
        pytest.skip("no CATCHBENCH_PAPER_DIR")
    sources = " ".join(path.read_text(encoding="utf-8")
                       for path in sorted(Path(paper_dir).glob("*.tex")))
    assert r"\label{" + epg.POST_BOARD_TABLE + "}" in sources, (
        f"the caption references {epg.POST_BOARD_TABLE}; the paper does not define it")


def test_this_tables_label_is_defined_at_most_once_in_the_manuscript():
    """A second float under one label makes one of them unreachable by reference.

    The count rather than the absence, because this block is spliced into the appendix outside this
    workflow: before the splice the label is defined nowhere, after it exactly once, and two would
    mean either a collision with another float or a block spliced twice.
    """
    paper_dir = os.environ.get("CATCHBENCH_PAPER_DIR")
    if not paper_dir:
        pytest.skip("no CATCHBENCH_PAPER_DIR")
    sources = " ".join(path.read_text(encoding="utf-8")
                       for path in sorted(Path(paper_dir).glob("*.tex")))
    assert sources.count(r"\label{tab:post-generalization}") <= 1, (
        "tab:post-generalization is defined more than once in the manuscript")
