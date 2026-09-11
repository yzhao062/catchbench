r"""Hold the LIVE size-control block to its record, its declaration, and the claims it may not make.

Four failures are worth failing loudly for, and none of them is visible to a reader of the LaTeX.

The first is the wrong estimand behind a right-looking number. A5 keeps three quantities apart that
the same arrays all support: the ROC-AUC of five-seed-averaged out-of-fold probabilities, which is
what ``D`` is and what the interval belongs to; the mean of the 25 fold AUCs, which is what the LIVE
board prints; and the mean of five per-seed pooled AUCs. On SWE-Gym at 25 percent the first two
differ by more than a printed digit in every arm, so a table that printed one under a caption
describing the other would be wrong at eight cells and would look right at all of them. The first
group below therefore checks the printed arm cells against the record's seed-averaged field, checks
that they are not the mean-fold field, and checks that the printed ``D`` is the difference of the two
printed arm columns to within the last bit a float subtraction can move.

The second is a label that has slid off its cell. A9 asks for the inventory and asks for the 100
percent effects to be labelled reproduction controls, and the strongest effect in this batch is one
of those two: ``D(tau, 1.00)`` is ``+0.0447`` with an interval clear of zero, against a primary cell
that reads unresolved at ``+0.0052``. A8's last bullet exists because of exactly that shape. So the
Role row is checked cell by cell rather than as a set, the primary row is checked to be the one A5
nominated, and the branch is checked to have been applied to the primary cell's own numbers.

The third is a claim the batch did not make. An unresolved primary cell licenses no verb stronger
than the record's own, a table of eight marginal readings licenses no comparison between two of them,
and A4 forbids one particular upgrade by name: the tau-bench evaluation is row-level cross-validation
and the clustered interval does not convert it into an unseen-task evaluation. The caption is
required to say so, so a blanket word ban would ban the disclosure with the claim; the guard counts
occurrences against the record's own sentences instead, and a word that appears once more than those
sentences account for fails.

The fourth is an unescaped percent sign, which this repository has already been bitten by once. A
caption elsewhere in this appendix kept ``95\%%`` after its ``%`` interpolation was removed, the
second sign reached LaTeX, and it commented out the rest of the generated line: no build warning, no
--check failure because source and output shared the error, and no test failure because every test
read a source string. The guard here reads the bytes the command line writes.

The record is read and the declaration is imported. ``catchbench.live_size_audit`` holds the frozen
arms, contrast, primary cell, threshold, seed and labels, so the constants this block prints are
checked against the module that produced them rather than against a copy of them.
"""
from __future__ import annotations

import ast
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(TOOLS))

import emit_live_prefix_table as elpt  # noqa: E402
import emit_live_size_audit as runner  # noqa: E402
import emit_live_size_audit_table as elsat  # noqa: E402

from catchbench import live_size_audit as audit  # noqa: E402

RECORD = TOOLS / "live_size_control_audit_results.json"
FRAGMENT = TOOLS / "live_size_control_table.tex"

_ROW_END = r" \\"

# The full-precision comment lines the block carries beside every printed cell. Each is anchored at
# both ends, so a line that grew or lost a field stops matching rather than matching loosely.
_ARM_COMMENT = re.compile(
    r"^% (\S+): arm (.+?): seed_averaged_oof=([\d.]+); mean_fold=([\d.]+)$")
_EFFECT_COMMENT = re.compile(
    r"^% (\S+): D=(-?[\d.]+); interval=\[(-?[\d.]+), (-?[\d.]+)\]; method=(.+?); axis=(.+)$")
_REPRODUCTION_COMMENT = re.compile(
    r"^% (\S+): mean_fold_difference=(-?[\d.]+); interval=(\S+)$")
_ROLE_COMMENT = re.compile(
    r"^% (\S+): role=(.+?); labelled_as=(.+?); independent_new_evidence=(True|False); "
    r"rng_label=(\S+); base_seed=(\d+)$")

# Words that would turn an unresolved reading into a result. "survives" is deliberately absent: it is
# the verb of the record's own question, which asks how much of the increment survives, and banning
# it would ban the question. "significant" and "p =" are absent from the record entirely and stay
# banned outright.
FORBIDDEN = ("significant", "significance", "p =", "confirms", "demonstrates", "proves",
             "rules in", "holds up", "early warning gain", "detects failures earlier",
             "robust to", "as predicted")

# Every occurrence of these has to be accounted for by the record's own sentences. Each is a word the
# caption is required to carry in a refusal and would be a claim anywhere else: "unseen-task" is the
# upgrade A4 forbids by name, "Holm" is the family A9 refuses to open, "establishes" appears only
# inside A3's "establishes neither", and "substantial" is A8's branch vocabulary.
ACCOUNTED = ("unseen-task", "unseen task", "holm", "establishes", "substantial")

# Eight cells printed side by side invite a sentence about which prefix is best, and nothing here
# tests a difference between two cells: they are marginal readings with one nominated primary
# quantity. Two cells landing on the same reading invite the opposite sentence just as strongly, so
# equivalence is banned with superiority.
COMPARATIVE = ("outperforms", "beats", "leads", "better than", "worse than", "wins", "superior",
               "strongest", "weakest", "best prefix", "no difference", "equivalent",
               "indistinguishable", "larger early")


@pytest.fixture(scope="module")
def record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def generated():
    return elsat.build()


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
    return subprocess.run([sys.executable, str(Path(elsat.__file__)), *map(str, args)],
                          capture_output=True, env=env, timeout=600)


def _cells(line: str) -> list[str]:
    return line.removesuffix(_ROW_END).split(" & ")


def _table_rows(generated: str) -> list[list[str]]:
    """The rows of the one tabular, located by its own label rather than by position."""
    lines = generated.splitlines()
    marker = r"\label{tab:live-size-control}"
    assert lines.count(marker) == 1, "the block carries one labelled float"
    start = lines.index(r"\begin{tabular}{@{}lrrrr@{}}")
    stop = lines.index(r"\end{tabular}")
    return [_cells(line) for line in lines[start:stop] if line.endswith(_ROW_END)]


def _labelled_row(generated: str, label: str) -> list[list[str]]:
    """Every body row whose first cell is ``label``: one per corpus block."""
    return [row[1:] for row in _table_rows(generated) if row[0] == label]


def _rendered(generated: str) -> str:
    """Everything a reader of the compiled PDF meets: the block without its comment lines.

    The counting guard below is scoped to this rather than to the whole block, because the
    provenance lines legitimately echo record field names such as ``upper_endpoint_below_substantial``
    and counting those against the caption's sentences would measure the wrong thing. The outright
    bans stay on the whole block, where a word planted in a comment still fails.
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


# --- the printed block against the record it came from --------------------------------------------

def test_every_printed_arm_cell_is_the_seed_averaged_score_and_not_the_board_number(generated,
                                                                                   record):
    """A5's separation, checked on the cells rather than on a claim that it was respected.

    The mean of the 25 fold AUCs is what Table~\\ref{tab:live-stream} prints for the two linear arms,
    and it is a different number at every cell here. Both are in the record, so a table that read the
    board field would still print four plausible columns under a caption describing the other one.
    """
    for arm in elsat._ARM_ORDER:
        printed = _labelled_row(generated, elsat._printed_arm_names()[arm])
        assert len(printed) == len(elsat._CORPUS_ORDER), f"{arm} prints one row per corpus"
        for corpus, cells in zip(elsat._CORPUS_ORDER, printed):
            block = [row for row in record["effects"] if row["corpus"] == corpus]
            assert len(cells) == len(block)
            for cell, row in zip(cells, block):
                assert cell == f"{row['arm_scores'][arm]['seed_averaged_oof_roc_auc']:.3f}"
            # A cell where the two quantities round the same way cannot say which field was read,
            # so the check is only as good as its discrimination: at least one cell per corpus and
            # arm has to separate them, and every one of them does today.
            separating = [row for row in block
                          if f"{row['arm_scores'][arm]['seed_averaged_oof_roc_auc']:.3f}"
                          != f"{row['arm_scores'][arm]['mean_fold_roc_auc']:.3f}"]
            assert separating, f"{corpus} {arm}: the two quantities never separate at three decimals"


def test_every_printed_effect_and_interval_is_the_records_own(generated, record):
    """Same cells, same order, same numbers, same roles as the shipped record.

    Order is compared rather than looked up, because a table whose roles or intervals had slid one
    column against their prefixes would still carry eight correct numbers and three correct labels.
    """
    effects = _labelled_row(generated, elsat._EFFECT_LABEL)
    intervals = _labelled_row(generated, elsat._INTERVAL_LABEL)
    roles = _labelled_row(generated, elsat._ROLE_LABEL)
    assert len(effects) == len(intervals) == len(roles) == len(elsat._CORPUS_ORDER)
    for index, corpus in enumerate(elsat._CORPUS_ORDER):
        block = [row for row in record["effects"] if row["corpus"] == corpus]
        assert len(block) == 4, "four prefixes per corpus"
        for cell, interval, role, row in zip(effects[index], intervals[index], roles[index], block):
            assert cell == "$" + f"{row['point']:+.4f}" + "$"
            low, high = row["interval_95"]
            assert interval == "$[" + f"{low:+.4f}" + ", " + f"{high:+.4f}" + "]$"
            assert role == row["role"]


def test_the_printed_effect_is_the_difference_of_the_two_printed_arm_columns(generated, record):
    """The column a reader would subtract has to be the column the effect came from.

    The emitter is forbidden arithmetic, so this is the one place the relation is checked, and it is
    checked against the printed cells rather than against the record alone: the high arm's column
    minus the low arm's column is ``D``, at every cell, to the last bit a float subtraction moves.
    """
    high, low = audit.MATCHED_CONTRAST
    for row in record["effects"]:
        difference = (row["arm_scores"][high]["seed_averaged_oof_roc_auc"]
                      - row["arm_scores"][low]["seed_averaged_oof_roc_auc"])
        assert row["point"] == pytest.approx(difference, rel=0, abs=1e-15), row["cell"]
    names = elsat._printed_arm_names()
    for index, corpus in enumerate(elsat._CORPUS_ORDER):
        printed_high = _labelled_row(generated, names[high])[index]
        printed_low = _labelled_row(generated, names[low])[index]
        block = [row for row in record["effects"] if row["corpus"] == corpus]
        for cell_high, cell_low, row in zip(printed_high, printed_low, block):
            assert cell_high == f"{row['arm_scores'][high]['seed_averaged_oof_roc_auc']:.3f}"
            assert cell_low == f"{row['arm_scores'][low]['seed_averaged_oof_roc_auc']:.3f}"


def test_every_printed_number_is_its_comment_line_value_rounded(generated, record):
    """The printed cell, the comment line beside it and the record must be one number.

    The comment lines are the only place a reader can re-derive a row from, so they are checked
    against the record at full precision and the printed cells are checked against them. Corrupting
    either one alone breaks the chain.
    """
    arms = _arm_comments(generated)
    effects = _comments(generated, _EFFECT_COMMENT)
    reproduction = _comments(generated, _REPRODUCTION_COMMENT)
    roles = _comments(generated, _ROLE_COMMENT)
    cells = {row["cell"]: row for row in record["effects"]}
    assert set(effects) == set(reproduction) == set(roles) == set(cells)
    assert set(arms) == {(cell, arm) for cell in cells for arm in elsat._ARM_ORDER}
    for name, row in cells.items():
        point, low, high, method, axis = effects[name]
        assert point == f"{row['point']:.12f}"
        assert [low, high] == [f"{value:.12f}" for value in row["interval_95"]]
        assert method == row["interval_method"] and axis == row["interval_axis"]
        assert reproduction[name][0] == (
            f"{row['reproduction_quantity']['mean_fold_difference']:.12f}")
        # A5 refuses this quantity an interval, and the comment line has to show the refusal rather
        # than omit the field, so a reader can see that it is absent on purpose.
        assert reproduction[name][1] == "None"
        assert roles[name][0] == row["role"]
        assert roles[name][1] == row["labelled_as"]
        assert roles[name][2] == str(row["independent_new_evidence"])
        assert roles[name][3] == row["rng_label"]
        assert roles[name][4] == str(row["rng_base_seed"])
        for arm in elsat._ARM_ORDER:
            scores = row["arm_scores"][arm]
            assert arms[(name, arm)][0] == f"{scores['seed_averaged_oof_roc_auc']:.12f}"
            assert arms[(name, arm)][1] == f"{scores['mean_fold_roc_auc']:.12f}"
    # and the printed cells are those same comment values rounded, not derived again
    for index, corpus in enumerate(elsat._CORPUS_ORDER):
        block = [row for row in record["effects"] if row["corpus"] == corpus]
        for cell, row in zip(_labelled_row(generated, elsat._EFFECT_LABEL)[index], block):
            assert cell == "$" + f"{float(effects[row['cell']][0]):+.4f}" + "$"


def test_the_eight_cells_are_both_corpora_at_the_four_declared_prefixes_in_order(generated, record):
    """A9 asks for the inventory, so the header and the blocks have to carry all of it."""
    frozen = record["declaration"]["frozen_before_the_run"]
    header = [row for row in generated.splitlines() if row.startswith("Arm or quantity")]
    assert len(header) == 1
    percents = [percent for percent in _cells(header[0])[1:]]
    block = [row for row in record["effects"] if row["corpus"] == elsat._CORPUS_ORDER[0]]
    assert percents == [f"{row['prefix_percent']}" + r"\%" for row in block]
    assert [row["prefix"] for row in block] == frozen["prefixes"] == list(audit.AUDIT_PREFIXES)
    assert len(record["effects"]) == 8
    for corpus in elsat._CORPUS_ORDER:
        corpus_record = record["corpora"][corpus]
        assert (r"\multicolumn{5}{@{}l}{\textbf{" + elsat._corpus_printed(corpus)[0] + "}, "
                + f"{corpus_record['n_runs']} runs, {corpus_record['n_failed']} failed"
                + r"} \\") in generated


def test_the_two_endpoint_cells_are_labelled_reproduction_controls_in_the_table(generated, record):
    """A9 asks for that label in the table, not only in the caption.

    The strongest effect in this batch is one of the two, so a reader meeting ``+0.0447`` with an
    interval clear of zero has to meet the label in the same column, without having to reach the
    caption to learn that the cell reproduces a number the paper already prints.
    """
    endpoint = record["declaration"]["frozen_before_the_run"]["prefixes"][-1]
    for index, corpus in enumerate(elsat._CORPUS_ORDER):
        block = [row for row in record["effects"] if row["corpus"] == corpus]
        printed = _labelled_row(generated, elsat._ROLE_LABEL)[index]
        for cell, row in zip(printed, block):
            assert (cell == elsat._REPRODUCTION_CONTROL) == (row["prefix"] == endpoint)
            assert (cell == elsat._REPRODUCTION_CONTROL) != row["independent_new_evidence"]
    assert generated.count(elsat._REPRODUCTION_CONTROL + r" \\") == len(elsat._CORPUS_ORDER)
    strongest = max(record["effects"], key=lambda row: row["point"])
    assert strongest["role"] == elsat._REPRODUCTION_CONTROL, (
        "the strongest cell is a reproduction control in this batch; if that changes, the reason "
        "this label has to be in the table changes with it")


def test_the_primary_cell_is_the_one_a5_nominated_and_the_branch_is_its_own(generated, record):
    """A8's last bullet, which a table can break silently by reading a better cell."""
    primary = record["primary"]
    assert (primary["corpus"], primary["prefix"]) == audit.PRIMARY_CELL
    printed = _labelled_row(generated, elsat._ROLE_LABEL)
    assert [cell for block in printed for cell in block].count(elsat._PRIMARY) == 1
    outcome = record["outcome"]
    assert outcome["point"] == primary["point"]
    assert outcome["interval_95"] == primary["interval_95"]
    prose = " ".join(generated.split())
    assert (f"The cell A5 nominated before any of the eight was computed is "
            f"{elsat._corpus_printed(primary['corpus'])[0]} at {primary['prefix_percent']}" + r"\%"
            + ": $D$ = $" + f"{primary['point']:+.4f}" + "$, $[" + f"{primary['interval_95'][0]:+.4f}"
            + ", " + f"{primary['interval_95'][1]:+.4f}" + "]$.") in prose
    assert f"Its branch, by the rule A8 froze before the run, is {outcome['branch']}." in prose
    assert elsat._BRANCH_READING[outcome["branch"]] in prose
    for branch, reading in elsat._BRANCH_READING.items():
        if branch != outcome["branch"]:
            assert reading not in prose, f"the block also reads the {branch} branch"


def test_the_temporal_contrast_is_identified_separately_and_named_as_secondary(generated, record):
    """A7's contrast is not a ninth cell, and a table that let it read as one would invite the claim.

    ``T`` is the quantity the sentence "the increment is larger early than at the endpoint" actually
    requires, and A7 says a positive ``T`` does not rescue an early-signal claim while the primary
    cell is unresolved. It therefore sits below its own rule, spans the whole width, and says
    "secondary" in the row itself.
    """
    contrast = record["temporal_contrast"]
    early = next(row for row in record["effects"] if row["cell"] == "swegym.25")
    late = next(row for row in record["effects"] if row["cell"] == "swegym.100")
    assert contrast["components"] == {"D(swegym, 0.25)": early["point"],
                                      "D(swegym, 1.00)": late["point"]}
    assert contrast["rng_label"] == audit.TEMPORAL_RNG_LABEL
    low, high = contrast["interval_95"]
    line = (r"\multicolumn{5}{@{}l}{Secondary (A7): $T$ = "
            + elsat._corpus_printed(early["corpus"])[0] + " $D$ at "
            + f"{early['prefix_percent']}" + r"\%" + " minus $D$ at "
            + f"{late['prefix_percent']}" + r"\%" + " = $" + f"{contrast['point']:+.4f}"
            + "$, $[" + f"{low:+.4f}" + ", " + f"{high:+.4f}" + "]$} \\\\")
    lines = generated.splitlines()
    assert lines.count(line) == 1
    position = lines.index(line)
    assert lines[position - 1] == r"\midrule", "the contrast sits below a rule of its own"
    role_rows = [index for index, text in enumerate(lines)
                 if text.startswith(elsat._ROLE_LABEL + " &")]
    assert role_rows and position > max(role_rows), "the contrast sits below every cell block"
    assert not line.startswith(elsat._ROLE_LABEL), "the contrast is not a row of the cell table"
    prose = " ".join(generated.split())
    assert contrast["role"] in prose
    assert f"{contrast['replicates']:,} draws of a " + contrast["interval_method"] in prose


def test_the_caption_counts_and_constants_are_the_declarations(generated, record):
    """Every frozen quantity the caption states is the module's, not a copy of it."""
    frozen = record["declaration"]["frozen_before_the_run"]
    assert frozen["matched_contrast"] == list(audit.MATCHED_CONTRAST)
    assert frozen["substantial_threshold"] == audit.SUBSTANTIAL_EFFECT
    assert frozen["rng_base_seed"] == audit.BOOTSTRAP_BASE_SEED
    assert sorted(frozen["arms"]) == sorted(audit.ARM_ROLES) == sorted(elsat._ARM_ORDER)
    prose = " ".join(generated.split())
    assert f"The {audit.SUBSTANTIAL_EFFECT} threshold is " in prose
    roles = record["corpora"][elsat._CORPUS_ORDER[0]]["arm_roles"]
    names = elsat._printed_arm_names()
    for arm in elsat._ARM_ORDER:
        assert names[arm] + " is the " + roles[arm] in prose
        assert audit.ARM_ROLES[arm] == roles[arm]


def test_the_corpus_names_and_sibling_tables_are_the_live_prefix_tables_own(generated):
    """One corpus, one printed name, in every appendix table. A rename moves both or fails here."""
    assert elsat._CORPUS_TOKEN == runner.CLAIM_CORPUS
    assert elsat._CORPUS_ORDER == runner.CORPORA
    for corpus in elsat._CORPUS_ORDER:
        printed, label = elsat._corpus_printed(corpus)
        assert printed == runner.CORPUS_NAMES[corpus]
        assert (printed, label) in [(name, tab) for _, _, name, tab in elpt._CORPORA]
        assert r"\ref{" + label + "}" in generated
    published = dict(elpt._METHODS)
    for arm in elsat._ARM_ORDER:
        if arm in published:
            assert elsat._printed_arm_names()[arm] == published[arm]
        else:
            assert arm in elsat._NEW_ARM_NAMES


# --- the red direction: records this block refuses to print ----------------------------------------

def test_an_effect_that_disagrees_with_its_own_prefix_cell_is_refused(mutable):
    """The record stores each effect twice; printing one while the other disagrees is the failure."""
    mutable["effects"][0]["point"] = 0.5
    with pytest.raises(ValueError, match="the prefix cell's own contrast says"):
        elsat.table(mutable)


def test_an_arm_score_that_disagrees_with_its_own_prefix_cell_is_refused(mutable):
    high = mutable["effects"][0]["high"]
    mutable["effects"][0]["arm_scores"][high]["seed_averaged_oof_roc_auc"] = 0.999
    with pytest.raises(ValueError, match="the effect row carries"):
        elsat.table(mutable)


def test_a_contrast_naming_arms_other_than_the_frozen_pair_is_refused(mutable):
    mutable["effects"][0]["high"] = "auditable (size+deps)"
    with pytest.raises(ValueError, match="not the frozen"):
        elsat.table(mutable)


def test_a_reproduction_quantity_that_has_been_given_an_interval_is_refused(mutable):
    """A5 refuses the mean of fold AUCs the interval computed for the estimand."""
    mutable["effects"][0]["reproduction_quantity"]["interval"] = [-0.01, 0.01]
    with pytest.raises(ValueError, match="A5 refuses the mean of fold AUCs an interval"):
        elsat.table(mutable)


def test_a_cell_set_that_is_not_the_declared_corpora_and_prefixes_in_order_is_refused(mutable):
    mutable["effects"][0], mutable["effects"][1] = mutable["effects"][1], mutable["effects"][0]
    with pytest.raises(ValueError, match="not the declared corpora at the declared"):
        elsat.table(mutable)


def test_a_reproduction_control_label_away_from_the_endpoint_is_refused(mutable):
    """A9 makes the endpoint cells the controls; the label may not travel to another prefix."""
    mutable["effects"][1]["role"] = "reproduction control"
    with pytest.raises(ValueError, match="makes the 1.0 cells the reproduction controls"):
        elsat.table(mutable)


def test_a_role_that_contradicts_the_independent_evidence_flag_is_refused(mutable):
    endpoint = next(row for row in mutable["effects"] if row["role"] == "reproduction control")
    endpoint["independent_new_evidence"] = True
    with pytest.raises(ValueError, match="contradicts independent_new_evidence"):
        elsat.table(mutable)


def test_an_unrecognised_role_is_refused(mutable):
    mutable["effects"][1]["role"] = "supporting"
    with pytest.raises(ValueError, match="unrecognised role"):
        elsat.table(mutable)


def test_two_cells_sharing_a_role_and_labelled_differently_are_refused(mutable):
    mutable["effects"][2]["labelled_as"] = "one of the eight, more or less"
    with pytest.raises(ValueError, match="labelled differently"):
        elsat.table(mutable)


def test_an_interval_that_is_not_the_construction_it_is_labelled_as_is_refused(mutable):
    """A6 gives each corpus one construction, and the caption describes that one."""
    mutable["effects"][0]["interval_95"] = [-0.5, 0.5]
    with pytest.raises(ValueError, match="is not the delong interval"):
        elsat.table(mutable)


def test_an_unrecognised_interval_construction_is_refused(mutable):
    mutable["effects"][0]["interval_method"] = "a bootstrap of some kind"
    with pytest.raises(ValueError, match="unrecognised interval construction"):
        elsat.table(mutable)


def test_a_corpus_whose_rows_mix_two_constructions_is_refused(mutable):
    """The caption states one construction per corpus, so a mixed corpus has no true description."""
    row = mutable["effects"][1]
    swapped = next(other for other in mutable["effects"] if other["corpus"] == "tau")
    row["interval_method"] = swapped["interval_method"]
    row["interval_axis"] = swapped["interval_axis"]
    row["clustered_interval"] = copy.deepcopy(swapped["clustered_interval"])
    row["clustered_interval"]["interval_95"] = row["interval_95"]
    row["reproduction_diagnostic"] = copy.deepcopy(swapped["reproduction_diagnostic"])
    with pytest.raises(ValueError, match="interval constructions"):
        elsat.table(mutable)


def test_a_point_outside_its_own_interval_is_refused(mutable):
    mutable["effects"][0]["delong"]["interval_95"] = [0.5, 0.6]
    mutable["effects"][0]["interval_95"] = [0.5, 0.6]
    with pytest.raises(ValueError, match="outside its own interval"):
        elsat.table(mutable)


def test_an_outcome_read_off_another_cell_is_refused(mutable):
    """A8: if 25 percent disappoints, a better cell is not promoted to primary."""
    strongest = max(mutable["effects"], key=lambda row: row["point"])
    mutable["outcome"]["point"] = strongest["point"]
    mutable["outcome"]["interval_95"] = strongest["interval_95"]
    with pytest.raises(ValueError, match="point and interval are not"):
        elsat.table(mutable)


def test_an_unrecognised_branch_is_refused(mutable):
    mutable["outcome"]["branch"] = "promising"
    with pytest.raises(ValueError, match="unrecognised A8 branch"):
        elsat.table(mutable)


def test_a_threshold_that_is_not_the_frozen_one_is_refused(mutable):
    mutable["outcome"]["substantial_threshold"] = 0.01
    with pytest.raises(ValueError, match="against the frozen"):
        elsat.table(mutable)


def test_a_temporal_component_that_is_not_a_printed_cell_is_refused(mutable):
    mutable["temporal_contrast"]["components"]["D(swegym, 0.25)"] = 0.4
    with pytest.raises(ValueError, match="is not one of the effects this table prints"):
        elsat.table(mutable)


def test_a_temporal_contrast_that_bought_itself_more_fits_is_refused(mutable):
    mutable["temporal_contrast"]["additional_fits"] = 25
    with pytest.raises(ValueError, match="no additional classifier fit"):
        elsat.table(mutable)


def test_a_batch_the_a10_4_gate_did_not_admit_is_refused(mutable):
    """The gate is upstream of the table: an unadmitted batch prints nothing at all."""
    mutable["preflight"]["gate"]["admits_the_batch"] = False
    with pytest.raises(ValueError, match="did not admit this batch"):
        elsat.table(mutable)


def test_a_carried_forward_defect_outside_the_exempted_field_is_refused(mutable):
    defects = mutable["preflight"]["gate"]["record_defects_carried_forward"]
    defects[0]["quantity"] = "seed_averaged_oof_roc_auc"
    with pytest.raises(ValueError, match="outside the exempted field"):
        elsat.table(mutable)


def test_a_population_that_is_not_the_declared_one_is_refused(mutable):
    mutable["corpora"]["swegym"]["n_runs"] = 375
    with pytest.raises(ValueError, match="against the declared"):
        elsat.table(mutable)


# --- the branches the caption has to be able to take -----------------------------------------------

@pytest.mark.parametrize("branch, point, interval", [
    ("substantial-positive", 0.05, [0.01, 0.09]),
    ("small-positive", 0.02, [0.001, 0.04]),
    ("erased-or-reversed", -0.02, [-0.05, -0.001]),
])
def test_every_a8_branch_has_a_caption_that_reads_it(mutable, branch, point, interval):
    """A caption written for one branch would be wrong rather than incomplete in the others.

    The batch landed on ``unresolved``. The other three are reachable records, so each is exercised
    here: the branch A8 would have assigned those numbers is stated, and no other branch's reading
    appears beside it.
    """
    primary = next(row for row in mutable["effects"] if row["role"] == "primary")
    primary["point"] = point
    primary["interval_95"] = interval
    primary["delong"]["interval_95"] = interval
    cell = mutable["corpora"][primary["corpus"]]["by_prefix"][str(primary["prefix_percent"])]
    cell["effects"][primary["high"] + " - " + primary["low"]]["seed_averaged_oof_difference"] = point
    mutable["primary"]["point"] = point
    mutable["primary"]["interval_95"] = interval
    mutable["outcome"].update({"branch": branch, "point": point, "interval_95": interval,
                               "upper_endpoint_below_substantial": False})
    mutable["temporal_contrast"]["components"]["D(swegym, 0.25)"] = point
    prose = " ".join(elsat.table(mutable).split())
    assert f"Its branch, by the rule A8 froze before the run, is {branch}." in prose
    assert elsat._BRANCH_READING[branch] in prose
    for other, reading in elsat._BRANCH_READING.items():
        if other != branch:
            assert reading not in prose
    assert "The upper endpoint sits below the declared" not in prose


def test_no_branch_reading_claims_more_than_its_own_branch(generated, record):
    """The four readings are the emitter's own prose, so their guard may not be read from them.

    Every other check on the branch compares the block against ``elsat._BRANCH_READING``, which a
    rewrite of that constant would move on both sides at once: a reading of the unresolved branch
    that said the increment was supported would still be the reading the block printed and the
    reading the test expected. The vocabulary below is written here instead. An interval containing
    zero supports no verb of support, whatever the point is, and this batch's primary cell is that
    interval.
    """
    assert record["outcome"]["branch"] == "unresolved"
    assert record["outcome"]["interval_excludes_zero"] is False
    unresolved = elsat._BRANCH_READING["unresolved"].lower()
    assert "unresolved" in unresolved
    for claimed in ("supported", "survives", "survive", "confirms", "substantial", "excludes zero",
                    "holds", "clear of zero", "detects"):
        assert claimed not in unresolved, (
            f"the unresolved reading claims {claimed!r}; an interval containing zero supports no "
            f"such word")
    assert elsat._BRANCH_READING["unresolved"] in _rendered(generated)
    # and the three branches this batch did not land in each say which way their interval fell, so
    # that a reading cannot be swapped between branches and still read plausibly
    for branch in ("small-positive", "substantial-positive"):
        assert "positive side" in elsat._BRANCH_READING[branch] or "at least the declared" in (
            elsat._BRANCH_READING[branch])
        assert "unresolved" not in elsat._BRANCH_READING[branch].lower()
    assert "negative side" in elsat._BRANCH_READING["erased-or-reversed"]
    assert sorted(elsat._BRANCH_READING) == sorted(
        ["erased-or-reversed", "small-positive", "substantial-positive", "unresolved"])


def test_the_fourth_bullet_clause_appears_only_when_the_record_carries_that_flag(generated, record):
    """A8 gives an upper endpoint below the threshold no branch name, so it travels as a field."""
    assert record["outcome"]["upper_endpoint_below_substantial"] is True
    prose = " ".join(generated.split())
    assert ("The upper endpoint sits below the declared "
            f"{record['outcome']['substantial_threshold']}, which rules out that planning magnitude "
            "under this conditional analysis and does not establish zero information.") in prose
    assert record["declaration"]["fixed_by_this_implementation"]["a8_fourth_bullet"] in generated


def test_the_preflight_sentence_is_read_from_the_flag_and_not_written_down(generated, mutable,
                                                                           record):
    """The A10.4 branch this batch is in, and the branch a later clean run would be in.

    The preflight did not clear on its own terms here, and the declaration's dated entry is what
    admitted the batch anyway. A caption that stated that as a literal would go on stating it after
    a run that cleared outright, which is the shape of claim this whole block exists to avoid.
    """
    assert record["preflight"]["cleared_on_its_own_terms"] is False
    assert len(record["preflight"]["gate"]["record_defects_carried_forward"]) == 2
    prose = " ".join(generated.split())
    assert "The A10.4 preflight did not clear on its own terms" in prose
    assert "with two disagreements in the" in prose
    assert elsat._texttt(record["preflight"]["gate"]["exempted_field"]["field"]) in generated
    assert "cleared on its own terms and admitted this batch, so every" not in prose
    mutable["preflight"]["cleared_on_its_own_terms"] = True
    mutable["preflight"]["gate"]["record_defects_carried_forward"] = []
    clean = " ".join(elsat.table(mutable).split())
    assert "The A10.4 preflight cleared on its own terms and admitted this batch" in clean
    assert "did not clear on its own terms" not in clean


def test_a_batch_with_a_recorded_failure_says_so_in_the_caption_and_in_the_rows(mutable):
    """A9 asks for every failure and A11 asks for the partial results beside them.

    This batch recorded none, so the branch that reports one is exercised on a mutated record rather
    than left untested: a run that lost a corpus must not print seven cells in silence.
    """
    assert mutable["failures"] == []
    mutable["failures"] = [{"kind": "corpus", "corpus": "tau", "error": "RuntimeError: no rows"}]
    block = elsat.table(mutable)
    prose = " ".join(block.split())
    assert "The batch recorded one failure, named in the rows below" in prose
    assert r"\multicolumn{5}{@{}l}{Recorded failure: \texttt{corpus} on \texttt{tau}" in block
    assert "% failure: {'kind': 'corpus', 'corpus': 'tau'" in block


# --- claims this block may not make ----------------------------------------------------------------

def test_the_block_claims_no_result_the_primary_cell_did_not_support(generated):
    """An unresolved primary licenses no verb stronger than the record's own."""
    lowered = " ".join(generated.split()).lower()
    hits = [word for word in FORBIDDEN if word in lowered]
    assert hits == [], f"the block claims {hits}"


def test_the_block_compares_no_two_cells_in_either_direction(generated):
    """Eight marginal readings support no verdict about any pair of them."""
    lowered = " ".join(generated.split()).lower()
    hits = [word for word in COMPARATIVE if word in lowered]
    assert hits == [], f"the block sets cells against each other: {hits}"


def test_the_words_the_caption_must_carry_appear_only_in_the_sentences_that_account_for_them(
        generated, record):
    """A blanket ban would ban the disclosure, so the guard counts instead.

    A4 forbids describing the tau-bench row-level cross-validation as an unseen-task evaluation, and
    the caption is required to say that it is not one. The same shape holds for A9's refusal to open
    a Holm family and A3's "establishes neither". One more occurrence than these sentences account
    for is a claim nobody declared.
    """
    printed = _rendered(generated).lower()
    accounted = " ".join([*record["interpretation"].values(),
                          record["outcome"]["rule"],
                          record["outcome"]["threshold_is"]]).lower()
    for word in ACCOUNTED:
        assert printed.count(word) == accounted.count(word), (
            f"{word!r} appears {printed.count(word)} times in the rendered text and "
            f"{accounted.count(word)} times in the sentences that account for it")
    assert "does not convert the evaluation into an unseen-task evaluation" in printed


def test_every_required_record_sentence_reaches_the_caption_verbatim(generated, record):
    """The record and the manuscript may not carry different reasons for the same refusal."""
    prose = " ".join(generated.split())
    for key in ("endpoint_cells_are_controls", "exploratory", "no_deployable_alarm",
                "no_simultaneous_coverage", "one_batch", "tau_is_still_row_level_cv"):
        assert record["interpretation"][key] in prose, f"the caption drops {key}"
    assert record["question"] in prose
    assert record["outcome"]["rule"] in prose
    assert record["outcome"]["threshold_is"] in prose
    assert record["temporal_contrast"]["role"] in prose


def test_the_caption_says_the_extension_was_declared_after_its_neighbours_were_read(generated,
                                                                                   record):
    """The chronology clause, which is what keeps this table exploratory whichever way it came out.

    It is checked as its own test rather than folded into the verbatim sweep because it is the one
    sentence a reviewer is entitled to look for: the analysis was specified after the linear prefix
    scores and the POST results were in the manuscript, so it is a prospectively specified extension
    of an exploratory analysis rather than an independent confirmatory replication.
    """
    sentence = record["interpretation"]["exploratory"]
    assert "proposed after the linear LIVE prefix scores and the POST" in sentence
    assert "stays labelled exploratory" in sentence
    assert sentence in " ".join(generated.split())


def test_the_long_notes_that_carry_markup_are_kept_in_the_comment_lines(generated, record):
    """Record prose carrying characters LaTeX would read as markup stays unparsed."""
    note = record["inputs"]["note"]
    assert "% Inputs note: " + note in generated
    with pytest.raises(ValueError, match="LaTeX special characters"):
        elsat._prose(note)
    for corpus in elsat._CORPUS_ORDER:
        line = record["corpora"][corpus]["corpus_line"]
        assert f"% Corpus {corpus}: " + line in generated
    assert "% A10.4 reading: " + record["preflight"]["gate"]["reading"] in generated


# --- escaping and rendering ------------------------------------------------------------------------

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
    assert elsat._PROSE_SPECIALS | {"%"} == elsat._LATEX_SPECIALS
    assert "%" not in elsat._PROSE_SPECIALS
    assert elsat._prose("a 95% interval") == r"a 95\% interval"
    assert elsat._prose("plain words") == "plain words"
    for markup in (r"a\b", "a{b}", "a$b", "a&b", "a#b", "a^2", "a_b", "a~b"):
        with pytest.raises(ValueError, match="LaTeX special characters"):
            elsat._prose(markup)


def test_texttt_escapes_underscores_and_braces_and_refuses_the_rest():
    assert (elsat._texttt("variance_axes.board_point_estimate.value")
            == r"\texttt{variance\_axes.board\_point\_estimate.value}")
    for bad in (r"a\b", "a$b", "a&b", "a#b", "a^b", "a~b", "a%b"):
        with pytest.raises(ValueError, match="not a printable identifier"):
            elsat._texttt(bad)


def test_the_caption_refuses_a_roster_its_prose_cannot_describe():
    with pytest.raises(ValueError, match="outgrown the prose"):
        elsat._word(len(elsat._WORDS))
    assert elsat._word(len(elsat._ARM_ORDER)) == "four"
    assert elsat._word(0) == "no"


def test_an_arm_with_no_printed_name_anywhere_stops_the_run(monkeypatch):
    """A rename in the LIVE prefix tables must fail here rather than print a raw record id."""
    monkeypatch.setattr(elsat, "_ARM_ORDER", elsat._ARM_ORDER + ("size (cubic)",))
    with pytest.raises(ValueError, match="no printed name for the arm"):
        elsat._printed_arm_names()


def test_a_corpus_the_live_prefix_tables_do_not_carry_stops_the_run(monkeypatch):
    monkeypatch.setattr(elsat, "_CORPUS_TOKEN", dict(elsat._CORPUS_TOKEN, swegym="swegym"))
    with pytest.raises(ValueError, match="carries no corpus"):
        elsat._corpus_printed("swegym")


# --- the emitter owns no arithmetic -----------------------------------------------------------------

def test_the_emitter_owns_no_arithmetic_of_its_own():
    """The whole reason this tool exists rather than a hand-typed table.

    A second definition of ``D`` in this repository would be a plausible number at all eight cells
    and a different estimand at all eight, and A5 exists because the three candidates all come from
    the same arrays. Banning the operators is what makes that mechanical. It has a second effect
    worth having: with no ``%`` operator anywhere, the doubled percent sign that once ate a caption
    line cannot be left behind by a rewrite.
    """
    source = Path(elsat.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned = (ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.Mod, ast.MatMult)
    offenders = sorted({type(node.op).__name__ for node in ast.walk(tree)
                        if isinstance(node, ast.BinOp) and isinstance(node.op, banned)})
    assert offenders == [], f"the emitter computes: {offenders}"
    for reachable in ("math", "np", "numpy", "statistics", "scipy", "roc_auc_score",
                      "classify_outcome", "live_size_audit"):
        assert not hasattr(elsat, reachable), f"emit_live_size_audit_table reaches {reachable}"
    numbers = {node.value for node in ast.walk(tree)
               if isinstance(node, ast.Constant)
               and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)}
    floats = sorted(value for value in numbers if isinstance(value, float))
    assert floats == [], f"no measured value is written down here; found {floats}"
    # every count, population and threshold the block states is read, never typed
    for typed in (8, 25, 50, 75, 100, 165, 376, 660, 188, 363, 800, 10000, 20260907):
        assert typed not in numbers, f"{typed} is written down in the emitter"


# --- the command line and the paper check -----------------------------------------------------------

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


@pytest.mark.parametrize("use_env", [False, True])
def test_check_accepts_exact_block(tmp_path, generated, use_env):
    (tmp_path / "09_appendix.tex").write_bytes((generated + "\n").encode("utf-8"))
    result = (_cli("--check", paper_env=tmp_path) if use_env
              else _cli("--check", "--paper", tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert b"paper is current: tab:live-size-control" in result.stdout


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
        text = text.replace(r"& $+0.0447$ \\", r"& $+0.9999$ \\", 1)
    elif corruption == "caption":
        text = text.replace("is unresolved.", "is substantial-positive.", 1)
    elif corruption == "missing_begin":
        text = text.replace(elsat._BEGIN, "")
    elif corruption == "missing_end":
        text = text.replace(elsat._END, "")
    elif corruption == "duplicate":
        text += "\n" + elsat._BEGIN
    elif corruption == "reversed":
        text = elsat._END + "\n" + elsat._BEGIN
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
        assert b"+0.9999" in result.stdout and b"+0.0447" in result.stdout


def test_check_requires_paper_path():
    result = _cli("--check")
    assert result.returncode == 2
    assert b"--check needs --paper <dir> or CATCHBENCH_PAPER_DIR" in result.stderr


def test_a_broken_record_exits_one_with_a_readable_reason(tmp_path, monkeypatch, capsys):
    """main() reports rather than tracebacks, so a checker sees a reason instead of a stack."""
    monkeypatch.setattr(elsat, "RECORD", tmp_path / "absent.json")
    monkeypatch.setattr(sys, "argv", ["emit_live_size_audit_table.py"])
    assert elsat.main() == 1
    assert "ERROR: LIVE prefix size-control table:" in capsys.readouterr().err


# --- claims about the manuscript, which need the manuscript -----------------------------------------

def test_the_live_prefix_tables_this_caption_points_at_exist():
    """The caption sends a reader to the tables that print these cells on the board's quantity."""
    paper_dir = os.environ.get("CATCHBENCH_PAPER_DIR")
    if not paper_dir:
        pytest.skip("no CATCHBENCH_PAPER_DIR")
    sources = " ".join(path.read_text(encoding="utf-8")
                       for path in sorted(Path(paper_dir).glob("*.tex")))
    for corpus in elsat._CORPUS_ORDER:
        label = elsat._corpus_printed(corpus)[1]
        assert r"\label{" + label + "}" in sources, (
            f"the caption references {label}; the paper does not define it")


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
    assert sources.count(r"\label{tab:live-size-control}") <= 1, (
        "tab:live-size-control is defined more than once in the manuscript")
