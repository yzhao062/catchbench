r"""Emit the POST generalization block: four arms and three estimands on two corpora that are no board.

Section A of ``research/catchbench-post-generalization-declaration-2026-09-11.md`` asks Part A's
question of two further corpora: how much of the linear dependency increment survives once the size
reference is allowed to bend. ``tools/emit_post_generalization_audit.py --run`` fitted the declared
batch and wrote ``tools/post_generalization_audit_results.json``. Section H says what the appendix
then carries, and it says it as an inventory rather than as a summary: both corpora, all four arm
scores, ``L``, ``D`` and ``S`` with intervals, the measured populations, the fold protocol, the
measured SWE-Gym overlap, and every failure. This file formats exactly that inventory and computes
nothing.

**Nothing here is arithmetic.** Every score, effect, endpoint, population count and flag is read out
of the record at the precision the record stores it and printed rounded. The ban is mechanical:
``tests/test_emit_post_generalization_table.py`` walks this module's AST and fails on a subtraction,
a multiplication, a division or a modulo anywhere in it. Three failures are what it buys. The first
is a second definition of ``S``. ``S`` is a difference of differences over four arm columns, and a
table that subtracted two printed columns here would produce a number that looks like ``S`` at both
corpora and is a different quantity at both, because the printed columns are rounded and because the
record's ``S`` is scored on seed-averaged out-of-fold probabilities rather than on the mean of the 25
fold AUCs a board prints. The second is a recomputed population: the difference between the run count
and the distinct-issue count is not the number of repeated issues, and a table that inferred one
would be wrong in the manuscript rather than in a console line. The third is the doubled percent
sign. With no ``%`` operator in the module there is no interpolation for a rewrite to leave behind,
which is the failure this appendix has already met once and which no build warning and no ``--check``
would have caught.

The record is read, not trusted. Each refusal between the file and the block exists because the
corrupted record it rejects would still have printed a plausible table: an effect whose point
disagrees with the corpus block it came from, an arm score that differs between the two places the
record stores it, a branch whose rosters contradict its own per-corpus flags, a population that is
not the one the declaration froze, a batch the preflight gate did not admit, a run that registered a
board entrant, a corpus whose splined block is not its flat matrix.

Two sources outside the record are bound rather than restated. ``emit_live_size_audit_table`` supplies
the four arms and their printed names, because section B says the arms are Part A's "unchanged,
reusing the same classes rather than reimplementing them"; a rename there moves this table or fails
here. The record's own hub roster supplies the printed corpus names, so a corpus this table names is
one the run recorded a repository for.

The caption quotes the record's own sentences wherever the record has one. That is what stops the
public record and the manuscript from carrying different reasons for the same refusal: section D's
statement that the grouped OpenHands protocol is not the row-split protocol the POST boards use,
section G's rule that failing to reject ``S = 0`` does not establish stability, and section I's four
barred claims. Sentences that carry characters LaTeX would read as markup either name their
identifiers, which are set in ``\texttt``, or stay in the comment lines, unparsed.

Usage::

    python tools/emit_post_generalization_table.py
    python tools/emit_post_generalization_table.py --check --paper <paper directory>

The default stdout is one UTF-8, LF-delimited appendix table between sentinels. --check also accepts
CATCHBENCH_PAPER_DIR, compares bytes including whitespace and line endings, and exits 1 with a
readable delta on drift. No file is written, and nothing is spliced into the paper.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from emit_live_size_audit_table import _ARM_ORDER, _printed_arm_names  # noqa: E402

# Paths are joined with joinpath rather than with "/". The suite bans the division operator outright,
# so that no score, difference or ratio can be recomputed here, and a path join is the only place "/"
# would otherwise appear. Spelling it out costs a word and leaves the ban with no exceptions.
RECORD = ROOT.joinpath("post_generalization_audit_results.json")

_APPENDIX = "09_appendix.tex"
_BEGIN = ("% BEGIN GENERATED tab:post-generalization -- regenerate with: "
          "python tools/emit_post_generalization_table.py")
_END = "% END GENERATED tab:post-generalization"

# The POST board this audit is not part of. Section I bars describing either corpus as a board
# evaluation and bars saying the POST board covers four corpora, and the shortest way to let a reader
# check that is to send them to the board table, which prints SWE-Gym and tau-bench and no more. The
# label sits here rather than inside a caption string because a literal inside a caption is invisible
# to a test; a test in tests/test_emit_post_generalization_table.py holds it to the manuscript.
POST_BOARD_TABLE = "tab:det"

# The two corpora, in the order the batch scored them and this table prints them. Neither prints on
# any other table in this appendix, so no other table can supply their names. What binds them instead
# is the record's own hub roster, whose keys are the names the run recorded for the repositories it
# read: a corpus renamed there fails here rather than printing a name nothing in the record vouches
# for. The order is checked against settings.corpora_scored at every run.
_CORPUS_ORDER = ("openhands", "scienceworld")
_CORPUS_PRINTED = {"openhands": "OpenHands / SWE-rebench", "scienceworld": "ScienceWorld"}

# The third corpus in that roster, which this audit does not score. It is the scored POST population
# whose issue collection the OpenHands one is intersected with, and section I gates a claim on the
# result, so the overlap is reported whatever it measured.
_OVERLAP_CORPUS = "SWE-Gym"

# Section F's unit at the corpus with nothing to group. It is written down because the two corpora
# differ in exactly this and the caption states the difference, so a corpus that silently became
# grouped, or ungrouped, fails rather than printing the other one's description.
_ROW_UNIT = "row"

# Section E's three estimands in section E's order, the one that is a difference of the other two,
# and which frozen contrast each of the other two is. The order is written here because the record
# stores the effects as a list and a list cannot say which order was declared; it is checked against
# the declaration block inside the record at every run.
_ESTIMAND_ORDER = ("L", "D", "S")
_DERIVED_ESTIMAND = "S"
_CONTRAST_OF = {"L": "linear_contrast", "D": "matched_contrast"}

# The two roles section E gives a quantity, and the one that may appear once. A record that gave two
# rows the primary role, or gave it to a cell other than the nominated one, is a record whose
# nomination happened after the numbers appeared, which is the thing section E exists to prevent.
_ROLES = ("primary", "co-reported")
_PRIMARY = "primary"
# The marker is written for math mode, because it sits on a number that is already in math mode and
# two adjacent math groups are what a reader sees as a spacing bug. The caption wraps it itself.
_PRIMARY_MARKER = r"^{\dagger}"

# Section G's two named conditions and the state that is neither, keyed by the flag the record
# carries for each. A reading this file has never been told how to print is refused rather than
# guessed at, because the printed word is what a reader takes the corpus to have shown.
_READINGS = (("substantial_attenuation", "substantial attenuation"),
             ("informative_stable_positive_boundary", "informative stable-positive boundary"))
_UNRESOLVED_READING = "unresolved"

# What each of section G's four branches means, in that branch's own terms. All four are reachable,
# and a caption written for one of them would be wrong rather than incomplete in the others, which is
# why each has a sentence here and a test that exercises it.
_BRANCH_READING = {
    "substantial-attenuation": "Both corpora show substantial attenuation, which is the replication "
                               "section G's first row describes.",
    "mixed-informative": "One corpus shows substantial attenuation and the other gives an "
                         "informative stable-positive boundary.",
    "stable-positive-both": "Both corpora give informative stable-positive boundaries, so the "
                            "increment stays positive under the changed size control at each.",
    "unresolved-or-modest": "Neither corpus reaches substantial attenuation and neither gives an "
                            "informative stable-positive boundary, so section G reads the pair as "
                            "unresolved or modest.",
}

# Every caption clause whose record sentence opens with a section reference rather than with a
# capital letter is introduced by a label, so the caption reads as sentences while the record's words
# stay verbatim. The three that bar a claim carry section I's own bullet heading; the rest name the
# clause. The labels are also what makes a blanket word ban unnecessary: the caption is required to
# say "no unseen-task-family claim", so banning the word would ban the disclosure, and the guard in
# the tests counts occurrences against these labels plus the record's own sentences instead.
_LABELS = {
    "not_a_board": "No board-evaluation claim for either corpus",
    "no_unseen_task_family": "No unseen-task-family claim for ScienceWorld",
    "no_independent_scaffold": "No independent-scaffold claim for OpenHands",
    "grouped_protocol_differs": "Protocol difference",
    "failing_to_reject_is_not_stability": "Failing to reject is not stability",
    "exploratory": "Chronology",
    "one_batch": "Stopping rule",
    "conditional_on_the_saved_fits": "Conditionality",
    "branch_rule": "Branch rule",
    "no_promotion": "No promotion",
    "no_row_is_dropped": "No row dropped",
    "rosters_unchanged": "Rosters",
}

# The three of those that bar a claim section I names, in the order the caption states them.
_BARRED = ("not_a_board", "no_unseen_task_family", "no_independent_scaffold")

# Section I bars a disjoint-issue-set claim unless the overlap measured zero, so the label is chosen
# from the measured count rather than written down once. Both branches are exercised by the tests: a
# caption that kept the bar after a zero measurement would refuse a claim the declaration allows, and
# one that dropped it after a nonzero measurement would leave the shared issue undisclosed.
_OVERLAP_LABELS = {True: "No disjoint-issue-set claim",
                   False: "Disjoint issue sets, measured at zero"}

# The identifiers the two caption sentences name inside themselves. They sit here rather than inline
# so that a record sentence which stopped naming one fails at the call instead of printing an
# underscore into LaTeX, and so a reader of this module can see which strings are set in \texttt
# without reading the caption builder.
_SCAFFOLD_IDENTIFIER = "agent_graph_openhands.to_steps"
_ROSTER_IDENTIFIERS = ("detection._LOADERS", "live_streaming_methods()")

# Printed row labels for the rows that are neither an arm nor an estimand.
_HEADER_LABEL = "Arm or quantity"
_RUNS_LABEL = "Runs (failed, solved)"
_UNIT_LABEL = "Resampling unit"
_UNITS_LABEL = "Units resampled"
_READING_LABEL = "Reading (section G)"

# Every character LaTeX would read as markup. A percent sign is the one of them these records
# genuinely carry, and it is escaped on the way through: a bare one would comment out the rest of its
# line without a warning. Any other special means the record grew markup nobody chose for print, and
# refusing says so rather than guessing at an escape. The second set is spelled out rather than
# derived from the first, because the set-difference operator is banned here along with the
# arithmetic; a test holds the two to naming the same characters.
_LATEX_SPECIALS = frozenset(r"\{}$&#^_~%")
_PROSE_SPECIALS = frozenset(r"\{}$&#^_~")
_ESCAPED_PERCENT = r"\%"

# What may reach a \texttt without being rewritten beyond its underscores and braces: record field
# names, module paths, repository ids, issue identifiers and the fold-protocol specification.
# Anything else is refused for the same reason prose is.
_TEXTTT_ALLOWED = re.compile(r"[A-Za-z0-9 ._{}()=,*+:/-]+")

_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
          "eleven", "twelve")

# How many times a run was attempted, in the two shapes English has a word for. A multiplicity this
# prose cannot describe falls back to the spelled count, and a count past the roster above stops the
# run rather than printing a bare integer into a sentence built for words.
_TIMES = {1: "once", 2: "twice"}


def _prose(text: str) -> str:
    r"""A record sentence, passed through to a caption with its percent signs escaped.

    The caption sentences that carry a claim about what this table is and is not are the record's
    own, word for word. Restating them here would put the reason a claim was barred in two places,
    and the two would drift: the record would say inference on ScienceWorld is conditional on one
    source file while the manuscript said something adjacent to that.
    """
    intruders = sorted(set(text) & _PROSE_SPECIALS)
    if intruders:
        raise ValueError(f"a record sentence carries LaTeX special characters {intruders}; it "
                         f"cannot reach a caption unescaped: {text!r}")
    return text.replace("%", _ESCAPED_PERCENT)


def _texttt(text: str) -> str:
    r"""An identifier set in \texttt, with the underscores and braces LaTeX would otherwise eat."""
    if not _TEXTTT_ALLOWED.fullmatch(text):
        raise ValueError(f"not a printable identifier: {text!r}")
    for character in ("_", "{", "}"):
        text = text.replace(character, "\\" + character)
    return r"\texttt{" + text + "}"


def _prose_naming(text: str, *identifiers: str) -> str:
    r"""A record sentence that names code, with the named parts in \texttt and the rest escaped.

    Two sentences this caption is required to carry name an identifier inside themselves. Section I's
    scaffold bar names ``agent_graph_openhands.to_steps``, which is the whole of its evidence: the
    SWE-Gym adapter imports that function, so the two corpora do not have independent scaffolds. The
    plain guard above refuses such a sentence outright, because an underscore reaching LaTeX is a
    subscript. Paraphrasing it would put the reason a claim was barred in two places, and dropping it
    into a comment line would hide the one sentence a reviewer is entitled to see, so the caller names
    the identifiers and everything between them is still held to the plain guard.

    Each identifier must occur exactly once, so a sentence that grew a second mention fails here
    rather than leaving one of them as a subscript.
    """
    if not identifiers:
        raise ValueError("name at least one identifier, or use the plain guard")
    for identifier in identifiers:
        if text.count(identifier) != 1:
            raise ValueError(f"the identifier {identifier!r} appears {text.count(identifier)} times "
                             f"in the sentence; this caption marks exactly one occurrence of each: "
                             f"{text!r}")
    pattern = re.compile("|".join(re.escape(identifier) for identifier in identifiers))
    rendered = []
    position = 0
    for match in pattern.finditer(text):
        rendered.append(_prose(text[position:match.start()]))
        rendered.append(_texttt(match.group(0)))
        position = match.end()
    rendered.append(_prose(text[position:]))
    return "".join(rendered)


def _and_list(parts: list[str]) -> str:
    """Join caption fragments the way a sentence does rather than the way a list does."""
    if not parts:
        raise ValueError("no fragments to list")
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _word(count: int) -> str:
    """Spell a small count for caption prose, and fail rather than print a bare integer.

    Every count this caption states is derived from the record in hand, so a batch that grew or lost
    a corpus moves the prose with the table. A roster this prose cannot describe has to stop the run:
    a caption that says six quantities over a table of nine is worse than no table.
    """
    if not 0 <= count < len(_WORDS):
        raise ValueError(f"no caption word for a count of {count}; the audit has outgrown the prose "
                         f"this caption was written for")
    return _WORDS[count]


def _times(count: int) -> str:
    """How often an issue was attempted, in words."""
    return _TIMES.get(count, _word(count) + " times")


def load() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _shared(values: list, what: str):
    """One fact every corpus or row carries, stated once, and refused when they disagree.

    Section C's eligibility rule, section F's caveat about what these intervals are conditional on,
    and section E's separation of the estimand from the quantity a board prints are identical at both
    corpora. Two copies of a sentence is two places for a later regeneration to drift in, so the
    caption and the comment lines carry one; requiring the sources to agree first is what makes the
    one they carry true of all of them rather than true of whichever was read.
    """
    found = list(dict.fromkeys(values))
    if len(found) != 1:
        raise ValueError(f"the record carries {len(found)} different {what}, and this block states "
                         f"one: {found}")
    return found[0]


def _printed_corpus(record: dict, corpus: str) -> str:
    """The printed name of one scored corpus, checked against the hub roster the run recorded."""
    printed = _CORPUS_PRINTED[corpus]
    if printed not in record["hub_revisions"]:
        raise ValueError(f"the record's hub roster {sorted(record['hub_revisions'])} carries no "
                         f"{printed!r}; this table cannot name {corpus!r} after the run that "
                         f"produced it stopped recording where the corpus came from")
    return printed


def _rows(record: dict) -> list[dict]:
    """The six effect rows, checked against the declaration and against the corpus blocks.

    Order is checked rather than imposed. The record lists the cells corpus-major and, inside a
    corpus, in section E's estimand order; a row set that arrived in another order would still
    produce a plausible table with its intervals and its primary marker against the wrong quantities,
    and that is not a difference a reader could see. Everything else here is a cross-check between
    two places the record stores the same fact, because a table that read only one of them would
    agree with itself.
    """
    frozen = record["declaration"]["frozen_before_the_run"]
    if list(frozen["estimands"]) != list(_ESTIMAND_ORDER):
        raise ValueError(f"the declaration froze the estimands {list(frozen['estimands'])} and this "
                         f"table prints {list(_ESTIMAND_ORDER)}")
    if list(record["settings"]["corpora_scored"]) != list(_CORPUS_ORDER):
        raise ValueError(f"the batch scored {list(record['settings']['corpora_scored'])} and this "
                         f"table prints {list(_CORPUS_ORDER)}")
    rows = record["effects"]
    found = [(row["corpus"], row["estimand"]) for row in rows]
    wanted = [(corpus, estimand) for corpus in _CORPUS_ORDER for estimand in _ESTIMAND_ORDER]
    if found != wanted:
        raise ValueError(f"the record's cells are not the declared corpora at the declared "
                         f"estimands, in order; got {found}, wanted {wanted}")
    labels: dict[str, str] = {}
    for row in rows:
        if row["cell"] != row["corpus"] + "." + row["estimand"]:
            raise ValueError(f"the cell name {row['cell']!r} is not its own corpus and estimand")
        if row["role"] not in _ROLES:
            raise ValueError(f"{row['cell']}: unrecognised role {row['role']!r}")
        if labels.setdefault(row["role"], row["labelled_as"]) != row["labelled_as"]:
            raise ValueError(f"{row['cell']}: two cells share the role {row['role']!r} and are "
                             f"labelled differently")
        if row["rng_label"] != frozen["rng_labels"][row["cell"]]:
            raise ValueError(f"{row['cell']}: RNG label {row['rng_label']!r} is not the frozen "
                             f"{frozen['rng_labels'][row['cell']]!r}")
        if row["rng_base_seed"] != frozen["rng_base_seed"]:
            raise ValueError(f"{row['cell']}: base seed {row['rng_base_seed']} is not the frozen "
                             f"{frozen['rng_base_seed']}")
        if row["resampling_unit"] != frozen["resampling_unit"][row["corpus"]]:
            raise ValueError(f"{row['cell']}: resampled by {row['resampling_unit']!r} against the "
                             f"frozen {frozen['resampling_unit'][row['corpus']]!r}")
        _check_arms(record, row)
        _check_effect(record, row)
    primary = [row for row in rows if row["role"] == _PRIMARY]
    if len(primary) != 1:
        raise ValueError(f"{len(primary)} cells carry the primary role; section E nominates one")
    if primary[0]["cell"] != record["primary"]["cell"]:
        raise ValueError(f"the primary row is {primary[0]['cell']!r} and the record's primary is "
                         f"{record['primary']['cell']!r}")
    nominated = {"corpus": primary[0]["corpus"], "estimand": primary[0]["estimand"]}
    if frozen["primary_estimand"] != nominated:
        raise ValueError(f"the primary row {primary[0]['cell']!r} is not the frozen "
                         f"{frozen['primary_estimand']}")
    return rows


def _check_arms(record: dict, row: dict) -> None:
    """The four arm scores, checked against the corpus block that also stores them.

    The record keeps every arm score twice, inside the effect row and inside the corpus block it came
    from. Printing one while the other says something else is the failure this catches, and it is the
    failure a reader cannot see: the four printed columns would still be plausible ROC-AUCs and the
    three effects beside them would still be numbers the record supports.
    """
    corpus = record["corpora"][row["corpus"]]
    frozen_arms = record["declaration"]["frozen_before_the_run"]["arms"]
    if sorted(corpus["arms"]) != sorted(_ARM_ORDER) or sorted(corpus["arm_roles"]) != sorted(
            _ARM_ORDER):
        raise ValueError(f"{row['cell']}: the arms are {sorted(corpus['arms'])}, not the four "
                         f"declared {sorted(_ARM_ORDER)}")
    for arm in _ARM_ORDER:
        scored, printed = corpus["arms"][arm], row["arm_scores"][arm]
        for quantity in ("seed_averaged_oof_roc_auc", "mean_fold_roc_auc"):
            if scored[quantity] != printed[quantity]:
                raise ValueError(f"{row['cell']} {arm}: the effect row carries "
                                 f"{printed[quantity]!r} for {quantity} and the corpus block "
                                 f"carries {scored[quantity]!r}")
        if scored["role"] != corpus["arm_roles"][arm] or scored["role"] != frozen_arms[arm]:
            raise ValueError(f"{row['cell']} {arm}: role {scored['role']!r} against the corpus's "
                             f"{corpus['arm_roles'][arm]!r} and the frozen {frozen_arms[arm]!r}")
        if scored["nonfinite_fold_scores"] or scored["nonfinite_oof_values"]:
            raise ValueError(f"{row['cell']} {arm}: the record carries non-finite fold scores or "
                             f"out-of-fold values, which section J records as a failure rather than "
                             f"something to print")


def _check_effect(record: dict, row: dict) -> None:
    """One effect, checked against the corpus block's own contrast and against its own interval."""
    effect = record["corpora"][row["corpus"]]["effects"][row["estimand"]]
    frozen = record["declaration"]["frozen_before_the_run"]
    if effect["seed_averaged_oof_difference"] != row["point"]:
        raise ValueError(f"{row['cell']}: the effect row says {row['point']!r} and the corpus "
                         f"block's own contrast says {effect['seed_averaged_oof_difference']!r}")
    reproduction = row["reproduction_quantity"]
    if effect["mean_fold_difference"] != reproduction["mean_fold_difference"]:
        raise ValueError(f"{row['cell']}: the reproduction quantity disagrees with the corpus "
                         f"block's own mean-fold difference")
    if reproduction["interval"] is not None:
        raise ValueError(f"{row['cell']}: section E defines the estimands on the seed-averaged "
                         f"out-of-fold quantity and this record gives the mean of fold AUCs an "
                         f"interval: {reproduction['interval']!r}")
    if row["estimand"] == _DERIVED_ESTIMAND:
        if sorted(row["arms_carried_in_every_draw"]) != sorted(_ARM_ORDER):
            raise ValueError(f"{row['cell']}: section F carries all four score vectors in every "
                             f"draw and this row carries "
                             f"{sorted(row['arms_carried_in_every_draw'])}")
    else:
        contrast = list(frozen[_CONTRAST_OF[row["estimand"]]])
        if [effect["high"], effect["low"]] != contrast:
            raise ValueError(f"{row['cell']}: the contrast is {effect['high']!r} minus "
                             f"{effect['low']!r}, not the frozen {contrast}")
        if sorted(row["arms_carried_in_every_draw"]) != sorted(contrast):
            raise ValueError(f"{row['cell']}: the draw carries "
                             f"{sorted(row['arms_carried_in_every_draw'])} and the contrast is "
                             f"{sorted(contrast)}")
    low, high = row["interval_95"]
    if not low <= row["point"] <= high:
        raise ValueError(f"{row['cell']}: the point {row['point']!r} is outside its own interval "
                         f"[{low!r}, {high!r}]")
    if row["replicates"] != frozen["bootstrap_draws"]:
        raise ValueError(f"{row['cell']}: {row['replicates']} draws against the frozen "
                         f"{frozen['bootstrap_draws']}")
    if not row["point_recomputed_by_the_bootstrap_scorer"]["within_tolerance"]:
        raise ValueError(f"{row['cell']}: the point and the point its own bootstrap scorer "
                         f"recomputed disagree beyond the frozen tolerance")


def _population(record: dict, corpus: str) -> dict:
    """One corpus's measured population, checked against the counts the declaration froze.

    Section C says preflight must reproduce every declared number exactly and blocks if it cannot, so
    a record whose observed counts have drifted from its declared ones is not a weaker result: it is
    a different population, and the audit that ran is not the audit that was declared.
    """
    block = record["populations"][corpus]
    declared = record["declaration"]["frozen_before_the_run"]["declared_population"][corpus]
    if block["observed"] != block["declared"] or block["declared"] != declared:
        raise ValueError(f"{corpus}: the observed population {block['observed']} is not the declared "
                         f"{block['declared']} or not the frozen {declared}")
    corpus_block = record["corpora"][corpus]
    observed = block["observed"]
    if (corpus_block["n_runs"] != observed["n_runs"]
            or corpus_block["n_failed"] != observed["n_failed"]):
        raise ValueError(f"{corpus}: the corpus block carries {corpus_block['n_runs']} runs and "
                         f"{corpus_block['n_failed']} failed against the population's "
                         f"{observed['n_runs']} and {observed['n_failed']}")
    if len(corpus_block["labels"]) != observed["n_runs"]:
        raise ValueError(f"{corpus}: {len(corpus_block['labels'])} labels against "
                         f"{observed['n_runs']} runs")
    return observed


def _fold_protocol(record: dict, corpus: str) -> dict:
    """One corpus's fold protocol, checked against the declaration and against its own fold counts.

    Section D asserts that every fold at every seed contains both classes and blocks if one does not,
    so a minimum class count of zero here would mean the caption describes a partition the batch did
    not have. The grouping is checked against section F's resampling unit for the same reason: the
    two corpora differ in exactly that, and the caption says so.
    """
    block = record["fold_protocol"][corpus]
    frozen = record["declaration"]["frozen_before_the_run"]
    if block["protocol"] != frozen["fold_protocol"][corpus]:
        raise ValueError(f"{corpus}: the fold protocol {block['protocol']!r} is not the frozen "
                         f"{frozen['fold_protocol'][corpus]!r}")
    if block["n_splits"] != frozen["folds"] or block["n_seeds"] != len(frozen["seeds"]):
        raise ValueError(f"{corpus}: {block['n_seeds']} seeds of {block['n_splits']} folds against "
                         f"the frozen {len(frozen['seeds'])} of {frozen['folds']}")
    if block["minimum_class_count_in_any_fold"] <= 0:
        raise ValueError(f"{corpus}: a fold with no minority-class row; section D blocks on that "
                         f"rather than scoring it")
    unit = record["corpora"][corpus]["resampling_unit"]
    if (block["grouped_by"] is not None) != (unit != _ROW_UNIT):
        raise ValueError(f"{corpus}: folds grouped by {block['grouped_by']!r} while the resampling "
                         f"unit is {unit!r}; section D and section F name the same unit")
    return block


def _columns(record: dict, corpus: str) -> dict:
    """The column boundary, which is what makes the matched arm the arm section B declared.

    Section B says only the four flat size and count columns receive the spline. The splined block
    has to be exactly this corpus's flat matrix and the passthrough block exactly the rest of its
    flatdep matrix; anything else means the spline reached the dependency columns, and the caption's
    description of the matched arm would be a description of a different arm.
    """
    block = record["verification"]["per_corpus"][corpus]["column_boundaries"]
    flat_rows, flat_columns = block["flat_shape"]
    dep_rows, dep_columns = block["flatdep_shape"]
    runs = record["populations"][corpus]["observed"]["n_runs"]
    if flat_rows != runs or dep_rows != runs:
        raise ValueError(f"{corpus}: the feature matrices carry {flat_rows} and {dep_rows} rows "
                         f"against {runs} scored runs")
    splined, passthrough = list(block["splined_columns"]), list(block["passthrough_columns"])
    if splined != list(range(flat_columns)):
        raise ValueError(f"{corpus}: the splined block is {splined} and the flat matrix is "
                         f"{flat_columns} columns wide; section B splines the flat matrix and "
                         f"nothing else")
    if sorted(splined + passthrough) != list(range(dep_columns)):
        raise ValueError(f"{corpus}: the splined and passthrough blocks are not the {dep_columns} "
                         f"flatdep columns")
    return block


def _branch(record: dict) -> dict:
    """Section G's branch, checked to be consistent with the per-corpus readings it was read from.

    The branch is a word and the readings are flags, and the two are stored apart. A record whose
    branch said one thing while its flags said another would print a caption a reader could not check
    against the table above it, so the two rosters are compared against the flags in both directions
    and each corpus's unresolved flag is compared against its own two conditions.
    """
    branch = record["branch"]
    if branch["branch"] not in _BRANCH_READING:
        raise ValueError(f"unrecognised section G branch {branch['branch']!r}")
    if sorted(branch["branches"]) != sorted(_BRANCH_READING):
        raise ValueError(f"the record's branch roster {sorted(branch['branches'])} is not section "
                         f"G's {sorted(_BRANCH_READING)}")
    if sorted(branch["per_corpus"]) != sorted(_CORPUS_ORDER):
        raise ValueError(f"section G's branch requires both corpora and the record reads "
                         f"{sorted(branch['per_corpus'])}")
    rosters = {
        "corpora_showing_substantial_attenuation": _READINGS[0][0],
        "corpora_giving_an_informative_stable_positive_boundary": _READINGS[1][0],
    }
    for roster, flag in rosters.items():
        named = sorted(branch[roster])
        flagged = sorted(corpus for corpus in _CORPUS_ORDER if branch["per_corpus"][corpus][flag])
        if named != flagged:
            raise ValueError(f"the record names {named} under {roster} and the per-corpus {flag} "
                             f"flags say {flagged}")
    threshold = record["declaration"]["frozen_before_the_run"]["substantial_threshold"]
    for corpus in _CORPUS_ORDER:
        reading = branch["per_corpus"][corpus]
        if reading["unresolved"] != (not reading[_READINGS[0][0]] and not reading[_READINGS[1][0]]):
            raise ValueError(f"{corpus}: the unresolved flag contradicts the two conditions section "
                             f"G names: {reading}")
        if reading["substantial_threshold"] != threshold:
            raise ValueError(f"{corpus}: the reading is against {reading['substantial_threshold']} "
                             f"and the declaration froze {threshold}")
    return branch


def _reading(branch: dict, corpus: str) -> str:
    """The word one corpus's reading prints, chosen from its flags rather than from the branch."""
    reading = branch["per_corpus"][corpus]
    for flag, printed in _READINGS:
        if reading[flag]:
            return printed
    return _UNRESOLVED_READING


def _overlap(record: dict) -> dict:
    """The measured SWE-Gym overlap, which section H reports whatever it came to.

    The count and the identifiers are stored apart and the appendix discloses the identifiers, so a
    record whose count disagreed with its own list would either hide a shared issue or claim one that
    is not there. The repository is checked against the hub roster for the reason the scored corpora
    are: the sentence names a corpus, and the record has to say where that corpus came from.
    """
    overlap = record["swegym_overlap"]
    openhands = record["populations"][_CORPUS_ORDER[0]]["observed"]
    if overlap["n_shared_issues"] != len(overlap["shared_issues"]):
        raise ValueError(f"the overlap counts {overlap['n_shared_issues']} shared issues and names "
                         f"{len(overlap['shared_issues'])}: {overlap['shared_issues']}")
    if _OVERLAP_CORPUS not in record["hub_revisions"]:
        raise ValueError(f"the record's hub roster {sorted(record['hub_revisions'])} carries no "
                         f"{_OVERLAP_CORPUS!r}, whose issue collection this overlap is measured "
                         f"against")
    if overlap["repository"] != record["hub_revisions"][_OVERLAP_CORPUS]["repo_id"]:
        raise ValueError(f"the overlap was measured against {overlap['repository']!r} and the hub "
                         f"roster records {record['hub_revisions'][_OVERLAP_CORPUS]['repo_id']!r}")
    if overlap["n_distinct_openhands_issues"] != openhands["n_distinct_group"]:
        raise ValueError(f"the overlap intersected {overlap['n_distinct_openhands_issues']} "
                         f"OpenHands issues and the scored population has "
                         f"{openhands['n_distinct_group']}")
    compared = record["verification"]["openhands_ids_match_the_measured_overlap"]["ids_compared"]
    if compared != openhands["n_runs"]:
        raise ValueError(f"{compared} OpenHands ids were compared against the measured overlap and "
                         f"the scored corpus has {openhands['n_runs']} runs")
    return overlap


def _gate(record: dict) -> dict:
    """The preflight gate, which has to have admitted this batch before any of it is printed."""
    gate = record["preflight"]["gate"]
    settings = record["settings"]
    if not gate["admits_the_batch"] or not gate["cleared"] or gate["findings"]:
        raise ValueError(f"the preflight gate did not admit this batch: {len(gate['findings'])} "
                         f"finding(s)")
    if sorted(gate["corpora_covered"]) != sorted(_CORPUS_ORDER) or gate["corpora_missing"]:
        raise ValueError(f"the gate covered {sorted(gate['corpora_covered'])} and this table prints "
                         f"{sorted(_CORPUS_ORDER)}")
    if gate["fits_of_the_declared_batch"]:
        raise ValueError(f"the preflight spent {gate['fits_of_the_declared_batch']} of section J's "
                         f"declared fits, which the batch then cannot have performed in full")
    if settings["declared_fits"] != settings["fits_performed"]:
        raise ValueError(f"section J declares {settings['declared_fits']} fits and the batch "
                         f"performed {settings['fits_performed']}")
    if record["board_entrants_added"]:
        raise ValueError(f"section H registers no board entrant and this run added "
                         f"{record['board_entrants_added']}")
    return gate


def _score(value: float) -> str:
    """One arm cell: three decimals, the precision the boards print scores at."""
    return f"{value:.3f}"


def _effect(value: float, marker: str = "") -> str:
    """One effect cell: four decimals and always a sign, because these differences run to 1e-3."""
    return "$" + f"{value:+.4f}" + marker + "$"


def _interval(low: float, high: float) -> str:
    return "$[" + f"{low:+.4f}" + ", " + f"{high:+.4f}" + "]$"


def _estimand_label(record: dict, estimand: str, names: dict[str, str]) -> str:
    """One estimand's row label, built from the arms the record says its contrast is between.

    ``S`` is the difference of the other two and has no arms of its own, so it prints the record's
    own definition. The other two print their arms, which is what lets a reader check the sign
    against the four columns above without taking the caption's word for which way round it is.
    """
    effects = [record["corpora"][corpus]["effects"][estimand] for corpus in _CORPUS_ORDER]
    if estimand == _DERIVED_ESTIMAND:
        definition = _shared([effect["definition"] for effect in effects],
                             f"definitions of {estimand}")
        return "$" + estimand + " = " + definition + "$"
    high = _shared([effect["high"] for effect in effects], f"high arms for {estimand}")
    low = _shared([effect["low"] for effect in effects], f"low arms for {estimand}")
    return "$" + estimand + "$: " + names[high] + r" $-$ " + names[low]


def _interval_label(estimand: str) -> str:
    return r"95\% interval for $" + estimand + "$"


def _population_sentence(record: dict, corpus: str, printed: str) -> str:
    """One corpus's measured population, in the shape its own record supports.

    OpenHands carries a group key and ScienceWorld does not, so the repeated-attempt clause is
    present at one corpus and absent at the other. Choosing the shape from the record rather than
    writing two sentences is what keeps a corpus whose identities later become recoverable from being
    described by a sentence written before they were.
    """
    observed = record["populations"][corpus]["observed"]
    sentence = (printed + f" carries {observed['n_runs']} runs, {observed['n_failed']} failed and "
                          f"{observed['n_solved']} solved")
    if "n_distinct_group" not in observed:
        return sentence + "."
    attempts = _and_list([f"{count} attempted " + _times(int(times))
                          for times, count in sorted(observed["group_multiplicity"].items())])
    return (sentence + f", over {observed['n_distinct_group']} distinct issues, " + attempts
            + f", and {observed['n_missing_group']} rows with no issue identifier.")


def _overlap_sentence(overlap: dict) -> str:
    """The measured overlap, in the branch the count puts it in.

    Section I gates the disjoint-issue-set claim on a zero measurement, so the two branches are not
    each other's negation: a nonzero overlap has issues to disclose and a zero overlap has a claim to
    release. A sentence written for one would be wrong rather than incomplete in the other.
    """
    shared = overlap["shared_issues"]
    against = ("Measured overlap between the "
               + f"{overlap['n_distinct_openhands_issues']} distinct issues of "
               + _CORPUS_PRINTED[_CORPUS_ORDER[0]] + " and the "
               + f"{overlap['n_distinct_swegym_issues']} distinct issues of the scored "
               + _OVERLAP_CORPUS + f" population over {overlap['n_swegym_rows']} rows: ")
    if not shared:
        return against + _word(len(shared)) + " shared issues, read from the raw shards."
    return (against + _word(len(shared)) + " shared "
            + ("issue" if len(shared) == 1 else "issues") + ", "
            + _and_list([_texttt(issue) for issue in shared]) + ", read from the raw shards.")


def _failure_sentence(record: dict) -> str:
    """What the batch recorded, in the branch it landed in.

    Section H asks for every failure, so a run that lost a corpus has to say so in the caption rather
    than print one corpus as though two were declared. The empty branch is the one this batch took,
    and it is stated rather than left to silence.
    """
    failures, warnings = record["failures"], record["warnings"]
    if not failures:
        return (f"The batch recorded {_word(len(failures))} failures and "
                f"{_word(len(warnings))} warnings.")
    return (f"The batch recorded {_word(len(failures))} "
            + ("failure" if len(failures) == 1 else "failures")
            + ", named in the rows below and given in full in this block's comment lines, and "
            + f"{_word(len(warnings))} warnings.")


def _recorded_basename(value: str) -> str:
    r"""The last segment of a path the record stored, read the same way on every platform.

    The records keep absolute paths and every caller here wants only the file name. ``Path`` is the
    obvious way to take it and the wrong one: it binds to the host flavour, so a Windows path read
    on Linux carries no separator ``PosixPath`` recognises and ``.name`` hands back the whole
    string. That is not hypothetical. ``Path(...).name`` here printed the file name on the author's
    Windows box and the full ``C:\Users\...\post_generalization_preflight.json`` on CI, so the
    generated block and the committed fragment stopped matching on Linux alone, and the shipped
    artifact was the one that could not be reproduced. Splitting on both separators reads a
    recorded path identically wherever the emitter runs.
    """
    text = str(value).replace("\\", "/").rstrip("/")
    if not text:
        raise ValueError("the record stored an empty path where a file name was expected")
    return text.rsplit("/", 1)[-1]


def _provenance(record: dict, rows: list[dict], branch: dict, overlap: dict, gate: dict,
                observed: dict) -> list[str]:
    """Comment lines binding each printed cell to the record's own full-precision value.

    The printed table rounds arm scores to three decimals and effects to four, so a reader who wants
    to re-derive a row, or a checker comparing two runs, has nothing to work from in the printed
    cells. These lines carry what the record stores. They also carry the three things the printed
    body deliberately does not: the mean of the 25 fold AUCs, which section E keeps apart from the
    estimand and refuses an interval; the execution-validity evidence, which is what makes the
    partition and the column boundary checkable rather than asserted; and the record sentences that
    name identifiers LaTeX would read as markup, which a comment line does not parse.
    """
    inputs = record["inputs"]
    declaration = record["declaration"]
    frozen = declaration["frozen_before_the_run"]
    settings = record["settings"]
    bootstrap = settings["bootstrap"]
    lines = [
        _BEGIN,
        f"% Source: tools/{RECORD.name}, schema {record['schema_version']}, generated by "
        f"{record['generated_by']}. This file computes nothing; every number below is read.",
        f"% Declaration: {declaration['file']}, frozen {declaration['frozen_utc']}; sha256 at "
        f"preflight {inputs['declaration']['sha256_at_preflight']}; sha256 when the batch ran "
        f"{inputs['declaration']['sha256']}.",
        f"% Declaration hash moved: {inputs['declaration']['changed_since_preflight']}; "
        f"{inputs['declaration']['why_it_can_change']}",
        f"% Question: {record['question']}",
        f"% Chronology: {declaration['chronology']}",
        f"% Inputs note: {inputs['note']}",
        f"% Records consumed by the batch: {', '.join(inputs['records_consumed_by_the_batch'])}.",
        f"% Fits: {settings['declared_fits']} declared, {settings['fits_performed']} performed; "
        f"seeds {frozen['seeds']}; {frozen['folds']} folds; confidence "
        f"{settings['confidence_level']}; elapsed {record['elapsed_seconds']:.3f}s.",
        f"% Spline: degree={frozen['spline']['degree']}; n_knots={frozen['spline']['n_knots']}; "
        f"knots={frozen['spline']['knots']}; include_bias={frozen['spline']['include_bias']}; "
        f"extrapolation={frozen['spline']['extrapolation']}.",
        f"% Bootstrap: {bootstrap['draws']} draws; percentiles {bootstrap['percentiles']}; "
        f"{bootstrap['interpolation']}; inner AUC {bootstrap['inner_auc']}; base seed "
        f"{frozen['rng_base_seed']}; unit {bootstrap['unit']}.",
        f"% Bootstrap method: {bootstrap['method']}",
        f"% Bootstrap, S: {bootstrap['s_carries']}",
        f"% Environment: Python {inputs['python']}; numpy {inputs['package_versions']['numpy']}; "
        f"scikit-learn {inputs['package_versions']['scikit-learn']}; "
        f"scipy {inputs['package_versions']['scipy']}; {inputs['platform']}.",
        f"% Repositories: catchbench {inputs['repositories']['catchbench']['commit']}; "
        f"grade {inputs['repositories']['grade']['commit']}; "
        f"paper {inputs['repositories']['paper']['commit']}.",
    ]
    for name, revision in record["hub_revisions"].items():
        lines.append(f"% Hub: {name}: {revision['repo_id']} at {revision['head']}")
    lines += [
        f"% Hub revisions are {record['hub_revisions_are']}.",
        f"% Rosters: {record['rosters_unchanged']}; board entrants added: "
        f"{record['board_entrants_added']}.",
        f"% Preflight: cleared={record['preflight']['cleared']}; "
        f"admits_the_batch={gate['admits_the_batch']}; findings={len(gate['findings'])}; "
        f"declared fits spent={gate['fits_of_the_declared_batch']}; record "
        f"{_recorded_basename(record['preflight']['record'])}, sha256 "
        f"{inputs['preflight_record']['sha256']}.",
        f"% Preflight reading: {gate['reading']}",
    ]
    for corpus, parity in record["preflight"]["saved_split_parity"].items():
        lines += [
            f"% Saved-split parity {corpus}: compared_against={parity['compared_against']}; "
            f"arms={parity['arms_compared']}; arms_disagreeing={parity['arms_disagreeing']}; "
            f"max_absolute_difference={parity['max_absolute_difference']:.3e}; "
            f"tolerance={parity['tolerance']:.3e}; failures={len(parity['failures'])}",
            f"% Saved-split parity {corpus} establishes: {parity['establishes']}",
            f"% Saved-split parity {corpus} withholds: {parity['scores_withheld']}",
        ]
    for path, source in sorted(inputs["code"].items()):
        lines.append(f"% Source hash: {path}: {source['sha256']}")
    for corpus in _CORPUS_ORDER:
        block = record["corpora"][corpus]
        population = record["populations"][corpus]
        verification = record["verification"]["per_corpus"][corpus]
        columns = verification["column_boundaries"]
        transforms = verification["training_only_transforms"]
        folds = record["fold_protocol"][corpus]
        identities = population["identities"]
        lines += [
            f"% Corpus {corpus}: {block['corpus_line']}",
            f"% Corpus {corpus}: declared={population['declared']}; observed={observed[corpus]}",
            f"% Corpus {corpus}: eligibility: {population['eligibility']} "
            f"Label 1 means {population['label_1_means']}.",
            f"% Corpus {corpus}: identities from {identities['source']}",
            f"% Corpus {corpus}: alignment verified: "
            f"{'; '.join(identities['alignment_verified'])}",
            f"% Corpus {corpus}: identities note: {identities['note']}",
            f"% Corpus {corpus}: label_hash={block['label_hash']}; feature hashes "
            + "; ".join(f"{layer}={digest}"
                        for layer, digest in sorted(block["feature_hashes"].items())),
            f"% Corpus {corpus}: folds {folds['protocol']}; seeds={folds['n_seeds']}; "
            f"splits={folds['n_splits']}; grouped_by={folds['grouped_by']}; "
            f"minimum_class_count_in_any_fold={folds['minimum_class_count_in_any_fold']}; "
            f"splits_compared={folds['splits_compared']}; layers={folds['layers_compared']}",
            f"% Corpus {corpus}: folds establish: {folds['establishes']}",
            f"% Corpus {corpus}: columns flat={columns['flat_shape']}; "
            f"flatdep={columns['flatdep_shape']}; splined={columns['splined_columns']}; "
            f"passthrough={columns['passthrough_columns']}; checked_by={columns['checked_by']}",
            f"% Corpus {corpus}: columns establish: {columns['establishes']}",
            f"% Corpus {corpus}: training-only transforms inspected "
            f"{transforms['fits_inspected']} fits of {transforms['arms_inspected']}; "
            f"checked_by={transforms['checked_by']}",
            f"% Corpus {corpus}: training-only transforms establish: {transforms['establishes']}",
            f"% Corpus {corpus}: matches the cleared preflight: "
            f"{verification['matches_the_cleared_preflight']['establishes']}",
        ]
        if "group_multiplicity" in identities:
            lines.append(f"% Corpus {corpus}: group key {identities['group_key_field']}; "
                         f"{identities['n_distinct_group']} distinct; multiplicity "
                         f"{identities['group_multiplicity']}; "
                         f"{identities['n_missing_group']} missing")
        if "runs_per_task_name" in identities:
            lines += [
                f"% Corpus {corpus}: source file {identities['source_file']}; "
                f"{identities['n_distinct_task_var_pairs']} distinct task-variation pairs over "
                f"{identities['n_distinct_task_names']} task names; repository "
                f"{identities['repository']}",
                f"% Corpus {corpus}: runs per task name: "
                + "; ".join(f"{name}={count}" for name, count
                            in sorted(identities["runs_per_task_name"].items())),
                f"% Corpus {corpus}: repository name: {identities['repository_name_is_wrong']}",
            ]
        for arm in _ARM_ORDER:
            scored = block["arms"][arm]
            lines.append(f"% {corpus}: arm {arm}: role={scored['role']}; layer={scored['layer']}; "
                         f"n_columns={scored['n_columns']}; "
                         f"seed_averaged_oof={scored['seed_averaged_oof_roc_auc']:.12f}; "
                         f"mean_fold={scored['mean_fold_roc_auc']:.12f}")
    for row in rows:
        low, high = row["interval_95"]
        recomputed = row["point_recomputed_by_the_bootstrap_scorer"]
        lines += [
            f"% {row['cell']}: role={row['role']}; labelled_as={row['labelled_as']}; "
            f"definition={row['definition']}; rng_label={row['rng_label']}; "
            f"base_seed={row['rng_base_seed']}",
            f"% {row['cell']}: point={row['point']:.12f}; interval=[{low:.12f}, {high:.12f}]; "
            f"method={row['interval_method']}; axis={row['interval_axis']}; "
            f"unit={row['resampling_unit']}; n_units={row['n_units']}",
            f"% {row['cell']}: replicates={row['replicates']}; usable={row['usable_replicates']}; "
            f"discarded_single_class_draws={row['discarded_single_class_draws']}; "
            f"bootstrap_mean={row['bootstrap_mean']:.12f}; "
            f"bootstrap_sd={row['bootstrap_sd']:.12f}; "
            f"two_sided_tail_p={row['two_sided_tail_p']:.12f}",
            f"% {row['cell']}: mean_fold_difference="
            f"{row['reproduction_quantity']['mean_fold_difference']:.12f}; "
            f"interval={row['reproduction_quantity']['interval']}",
            f"% {row['cell']}: point recomputed by the bootstrap scorer "
            f"{recomputed['value']:.12f}; abs_difference={recomputed['abs_difference']:.3e}; "
            f"tolerance={recomputed['tolerance']:.3e}; "
            f"within_tolerance={recomputed['within_tolerance']}",
        ]
    lines += [
        f"% Overlap: measured_through={overlap['measured_through']}",
        f"% Overlap: repository={overlap['repository']}; run_id={overlap['run_id_selected']}; "
        f"shards={overlap['shards_read']}",
        f"% Overlap: openhands_issues={overlap['n_distinct_openhands_issues']}; "
        f"swegym_issues={overlap['n_distinct_swegym_issues']}; "
        f"swegym_rows={overlap['n_swegym_rows']}; shared={overlap['n_shared_issues']} "
        f"{overlap['shared_issues']}",
        f"% Overlap: why not the loader: {overlap['why_not_the_loader']}",
        f"% Overlap: claim gate: {overlap['claim_gate']}",
        f"% Overlap: {overlap['no_row_is_dropped']}",
        f"% Overlap: alignment verified: {'; '.join(overlap['alignment_verified'])}",
        f"% Overlap: ids compared against the scored corpus: "
        f"{record['verification']['openhands_ids_match_the_measured_overlap']['ids_compared']}; "
        f"{record['verification']['openhands_ids_match_the_measured_overlap']['establishes']}",
        f"% Branch: {branch['branch']}; roster {branch['branches']}",
        f"% Branch rule: {branch['rule']}",
        f"% Branch: {branch['no_promotion']}",
    ]
    for corpus in _CORPUS_ORDER:
        reading = branch["per_corpus"][corpus]
        lines += [
            f"% Branch {corpus}: " + "; ".join(f"{flag}={value}" for flag, value
                                               in sorted(reading.items())
                                               if not isinstance(value, str)),
            f"% Branch {corpus}: threshold is {reading['threshold_is']}",
        ]
    for key, sentence in sorted(declaration["fixed_by_this_implementation"].items()):
        lines.append(f"% Fixed by the implementation, {key}: {sentence}")
    for key, sentence in sorted(record["interpretation"].items()):
        lines.append(f"% Interpretation, {key}: {sentence}")
    lines.append(f"% Failures: {len(record['failures'])}. Warnings: {len(record['warnings'])}.")
    for failure in record["failures"]:
        lines.append(f"% failure: {failure}")
    for warning in record["warnings"]:
        lines.append(f"% warning: {warning}")
    return lines


def _caption(record: dict, rows: list[dict], branch: dict, overlap: dict, names: dict[str, str],
             printed: dict[str, str]) -> list[str]:
    """One clause per line, so a caption edit shows as a one-line delta in --check."""
    interpretation = record["interpretation"]
    frozen = record["declaration"]["frozen_before_the_run"]
    roles = record["corpora"][_CORPUS_ORDER[0]]["arm_roles"]
    primary = next(row for row in rows if row["role"] == _PRIMARY)
    low, high = primary["interval_95"]
    reading = branch["per_corpus"][primary["corpus"]]
    lines = [r"\caption{" + _prose(record["question"])]
    lines += [_population_sentence(record, corpus, printed[corpus]) for corpus in _CORPUS_ORDER]
    lines += [
        _prose(_shared([record["populations"][corpus]["eligibility"] for corpus in _CORPUS_ORDER],
                       "eligibility rules")),
        "The four arms are the declaration's: "
        + _and_list([names[arm] + " is the " + _prose(roles[arm]) for arm in _ARM_ORDER]) + ".",
        "Arm cells are the ROC-AUC of " + _word(len(frozen["seeds"]))
        + "-seed-averaged out-of-fold failure probabilities, which is the quantity the "
        + _word(len(_ESTIMAND_ORDER)) + " estimands are defined on: "
        + _prose(_shared([row["reproduction_quantity"]["why_no_interval"] for row in rows],
                         "estimand sentences")),
        "Every column is rounded on its own, so the difference of two printed arm scores can differ "
        "from the printed effect in the last place.",
    ]
    lines += [printed[corpus] + " folds are "
              + _texttt(record["fold_protocol"][corpus]["protocol"]) + "."
              for corpus in _CORPUS_ORDER]
    lines += [
        "Both partitions establish " + _prose(_shared(
            [record["fold_protocol"][corpus]["establishes"] for corpus in _CORPUS_ORDER],
            "fold-protocol statements")) + ".",
        _LABELS["grouped_protocol_differs"] + ": "
        + _prose(interpretation["grouped_protocol_differs"]),
    ]
    lines += [printed[corpus] + " intervals are a " + _prose(_shared(
        [row["interval_method"] for row in rows
         if row["corpus"] == corpus and row["estimand"] == _DERIVED_ESTIMAND],
        f"{corpus} interval methods")) + "." for corpus in _CORPUS_ORDER]
    lines += [
        "$" + _DERIVED_ESTIMAND + "$ carries "
        + _prose(record["settings"]["bootstrap"]["s_carries"]) + ".",
        _LABELS["conditional_on_the_saved_fits"] + ": "
        + _prose(_shared([row["conditional_on_the_saved_fits"] for row in rows],
                         "conditionality sentences")),
        "The primary cell is marked $" + _PRIMARY_MARKER + "$: " + printed[primary["corpus"]]
        + " $" + primary["estimand"] + "$ = " + _effect(primary["point"]) + ", "
        + _interval(low, high) + ", which is " + _prose(primary["labelled_as"]) + ".",
        _LABELS["branch_rule"] + ": " + _prose(branch["rule"]) + ".",
        "The recorded branch is " + _prose(branch["branch"]) + ". "
        + _BRANCH_READING[branch["branch"]],
        _LABELS["no_promotion"] + ": " + _prose(branch["no_promotion"]) + ".",
        _LABELS["failing_to_reject_is_not_stability"] + ": "
        + _prose(interpretation["failing_to_reject_is_not_stability"]),
        f"The {reading['substantial_threshold']} magnitude is "
        + _prose(reading["threshold_is"]) + ".",
        _overlap_sentence(overlap),
        _LABELS["no_row_is_dropped"] + ": " + _prose(overlap["no_row_is_dropped"]),
        _OVERLAP_LABELS[bool(overlap["shared_issues"])] + ": "
        + _prose(interpretation["no_disjoint_issue_set"]),
        _LABELS[_BARRED[0]] + ": " + _prose(interpretation[_BARRED[0]])
        + r" That board is Table~\ref{" + POST_BOARD_TABLE + "}.",
        _LABELS[_BARRED[1]] + ": " + _prose(interpretation[_BARRED[1]]),
        _LABELS[_BARRED[2]] + ": "
        + _prose_naming(interpretation[_BARRED[2]], _SCAFFOLD_IDENTIFIER),
        _LABELS["exploratory"] + ": " + _prose(interpretation["exploratory"]),
        _LABELS["one_batch"] + ": " + _prose(interpretation["one_batch"]),
        _LABELS["rosters_unchanged"] + ": "
        + _prose_naming(record["rosters_unchanged"], *_ROSTER_IDENTIFIERS) + ".",
        _failure_sentence(record),
        r"Generated by " + _texttt("tools/emit_post_generalization_table.py") + ".}",
    ]
    return lines


def _body(record: dict, rows: list[dict], branch: dict, names: dict[str, str],
          observed: dict, columns: int) -> list[str]:
    """Population, then the four arms, then the three estimands, then section G's reading."""
    by_cell = {row["cell"]: row for row in rows}
    units = {corpus: _shared([row["n_units"] for row in rows if row["corpus"] == corpus],
                             f"{corpus} resampling-unit counts") for corpus in _CORPUS_ORDER}
    lines = [
        " & ".join([_RUNS_LABEL, *[f"{observed[corpus]['n_runs']} "
                                   f"({observed[corpus]['n_failed']}, "
                                   f"{observed[corpus]['n_solved']})"
                                   for corpus in _CORPUS_ORDER]]) + r" \\",
        " & ".join([_UNIT_LABEL, *[record["corpora"][corpus]["resampling_unit"]
                                   for corpus in _CORPUS_ORDER]]) + r" \\",
        " & ".join([_UNITS_LABEL, *[str(units[corpus]) for corpus in _CORPUS_ORDER]]) + r" \\",
        r"\midrule",
    ]
    for arm in _ARM_ORDER:
        lines.append(" & ".join([names[arm], *[
            _score(record["corpora"][corpus]["arms"][arm]["seed_averaged_oof_roc_auc"])
            for corpus in _CORPUS_ORDER]]) + r" \\")
    lines.append(r"\midrule")
    for estimand in _ESTIMAND_ORDER:
        points, intervals = [], []
        for corpus in _CORPUS_ORDER:
            row = by_cell[corpus + "." + estimand]
            points.append(_effect(row["point"],
                                  _PRIMARY_MARKER if row["role"] == _PRIMARY else ""))
            intervals.append(_interval(*row["interval_95"]))
        lines.append(" & ".join([_estimand_label(record, estimand, names), *points]) + r" \\")
        lines.append(" & ".join([_interval_label(estimand), *intervals]) + r" \\")
    lines += [
        r"\midrule",
        " & ".join([_READING_LABEL, *[_reading(branch, corpus) for corpus in _CORPUS_ORDER]])
        + r" \\",
    ]
    if record["failures"]:
        lines.append(r"\midrule")
        for failure in record["failures"]:
            lines.append(r"\multicolumn{" + str(columns) + r"}{@{}l}{Recorded failure: "
                         + _texttt(str(failure.get("corpus", "-")))
                         + r", in the comment lines} \\")
    return lines


def _split_caption(caption: list[str]) -> tuple[list[str], list[str]]:
    """Split the caption into the float's own line and a standfirst paragraph above the float.

    Section H asks this block to report a great deal, and the result is a caption of roughly a
    thousand words. LaTeX cannot break a ``table`` float, so a caption that long made the float
    taller than the text block and pdflatex reported ``Float too large for page``, which pushes the
    float and leaves an underfull page behind it. Every sentence is kept and the order is kept; all
    that changes is that the sentences after the first are typeset as an ordinary paragraph
    immediately above the float rather than inside it. The first sentence stays as the caption so
    the float still names itself in the list of tables.
    """
    opener = r"\caption{"
    if not caption or not caption[0].startswith(opener):
        raise ValueError("the caption does not begin with a caption command")
    if not caption[-1].endswith("}"):
        raise ValueError("the caption does not end by closing its brace")
    first = caption[0][len(opener):]
    rest = list(caption[1:])
    if not rest:
        return [opener + first], []
    rest[-1] = rest[-1][:-1]
    return [opener + first + "}"], [*rest, ""]


def table(record: dict) -> str:
    """One appendix float: the inventory section H asks for, between one pair of sentinels."""
    rows = _rows(record)
    branch = _branch(record)
    overlap = _overlap(record)
    gate = _gate(record)
    names = _printed_arm_names()
    printed = {corpus: _printed_corpus(record, corpus) for corpus in _CORPUS_ORDER}
    observed = {corpus: _population(record, corpus) for corpus in _CORPUS_ORDER}
    for corpus in _CORPUS_ORDER:
        # Called for the refusal rather than the answer: the caption describes one partition and one
        # column boundary per corpus, so a record whose own evidence contradicts either has no true
        # description here.
        _fold_protocol(record, corpus)
        _columns(record, corpus)
    header = [_HEADER_LABEL, *[printed[corpus] for corpus in _CORPUS_ORDER]]
    lines = _provenance(record, rows, branch, overlap, gate, observed)
    short, standfirst = _split_caption(
        _caption(record, rows, branch, overlap, names, printed))
    lines += standfirst
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        *short,
        r"\label{tab:post-generalization}",
        r"\begin{tabular}{@{}l" + "".join("r" for _ in _CORPUS_ORDER) + r"@{}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
        *_body(record, rows, branch, names, observed, len(header)),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        _END,
    ]
    return "\n".join(lines)


def build() -> str:
    """The one path from record to block, so a caller cannot assemble a different table than --check.

    ``emit_addendum_table`` learned this the hard way: its test suite once called the formatter
    directly, skipped a step ``build`` performs, and confirmed a block against itself while the
    command line published something else.
    """
    return table(load())


def check(paper: Path, generated: str) -> int:
    try:
        raw = paper.joinpath(_APPENDIX).read_bytes()
    except OSError as exc:
        print(f"STALE {_APPENDIX}: cannot read appendix: {exc}")
        return 1
    markers = []
    for marker in (_BEGIN, _END):
        found = list(re.finditer(rb"(?m)^" + re.escape(marker.encode("utf-8")) + rb"\r?$", raw))
        if len(found) != 1:
            print(f"STALE {_APPENDIX}: post-generalization marker {marker!r} appears "
                  f"{len(found)} times, expected once")
            return 1
        markers.append(found[0])
    begin, end = markers
    if end.start() < begin.start():
        print(f"STALE {_APPENDIX}: post-generalization end marker precedes its begin marker")
        return 1
    actual = raw[begin.start():end.start() + len(_END)]
    wanted = generated.encode("utf-8")
    if actual != wanted:
        print(f"STALE {_APPENDIX}: post-generalization block differs from the generated block")
        for index, (got, want) in enumerate(zip(actual.splitlines(keepends=True),
                                                wanted.splitlines(keepends=True)), 1):
            if got != want:
                print(f"first difference at block line {index}: paper={got!r}; generated={want!r}")
                break
        print("\n".join(difflib.unified_diff(
            actual.decode("utf-8", errors="replace").splitlines(), generated.splitlines(),
            fromfile=f"paper/{_APPENDIX}", tofile="generated/tab:post-generalization",
            lineterm="")))
        print("Regenerate with: python tools/emit_post_generalization_table.py")
        return 1
    print("paper is current: tab:post-generalization (two corpora, four arms, three estimands)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="check the marked appendix block instead of printing it")
    parser.add_argument("--paper", default=os.environ.get("CATCHBENCH_PAPER_DIR"),
                        help="paper source directory (or set CATCHBENCH_PAPER_DIR)")
    args = parser.parse_args()
    if args.check and not args.paper:
        parser.error("--check needs --paper <dir> or CATCHBENCH_PAPER_DIR")
    try:
        generated = build()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"ERROR: POST generalization table: {exc}", file=sys.stderr)
        return 1
    if args.check:
        return check(Path(args.paper), generated)
    sys.stdout.buffer.write((generated + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
