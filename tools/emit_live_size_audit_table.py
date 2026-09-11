r"""Emit the LIVE prefix size-control block: four arms at eight cells, and the contrast A7 declared.

Part A of ``research/catchbench-live-size-control-declaration-2026-09-10.md`` asks one question at
each LIVE prefix: how much of the dependency increment survives once the size reference is allowed
to bend. ``tools/emit_live_size_audit.py --run`` fitted the declared batch and wrote
``tools/live_size_control_audit_results.json``. A9 says what the appendix then carries, and it says
it as an inventory rather than as a summary: both corpora, all four prefixes, all four arm scores,
the eight matched effects, ``T``, every interval, and every failure, with the 100 percent effects
labelled reproduction controls. This file formats exactly that inventory and computes nothing.

**Nothing here is arithmetic.** Every score, effect, endpoint, count and flag is read out of the
record at the precision the record stores it and printed rounded. The ban is mechanical rather than
a habit: ``tests/test_emit_live_size_audit_table.py`` walks this module's AST and fails on a
subtraction, a multiplication, a division or a modulo anywhere in it. Two failures are what the ban
buys. The first is a second definition of ``D``. A5 separates three quantities that the same arrays
all support, and the one the interval belongs to is the ROC-AUC of five-seed-averaged out-of-fold
probabilities; the mean of the 25 fold AUCs is what the board prints, and taking one for the other
is the specific mistake A5 was written to prevent. A table that subtracted two arm columns here
would produce a number that looks like ``D`` at every cell and is a different estimand at all eight.
The second is the doubled percent sign. With no ``%`` operator in the module there is no
interpolation for a rewrite to leave behind, which is the failure this appendix has already met once
and which no build warning and no ``--check`` would have caught.

The record is read, not trusted. Twelve refusals sit between the file and the block, and each exists
because the corrupted record it rejects would still have printed a plausible table: an effect whose
point disagrees with the per-cell effects entry it came from, a contrast naming arms other than the
frozen pair, a corpus whose rows mix two interval constructions, a cell whose role contradicts its
``independent_new_evidence`` flag, a reproduction quantity that has been given the interval A5
refuses it, a batch the A10.4 gate did not admit. The arm scores in particular are checked twice,
because the record stores them twice: once inside the effect row and once in the corpus block, and a
table that read one while printing the other would agree with itself.

Two sources outside the record are bound rather than restated.
``emit_live_prefix_table._CORPORA`` supplies the printed corpus names and the labels of the two
LIVE prefix tables that print these same cells on the board's own quantity; a rename there moves
this caption or fails here. ``emit_live_prefix_table._METHODS`` supplies the printed names of the
two linear arms, which are entrants on those tables, so one arm reads the same way in all three.
The two spline-bearing arms are new to this table and are named here.

The caption quotes the record's own sentences wherever the record has one. That is what stops the
public record and the manuscript from carrying different reasons for the same refusal: A5's refusal
to give the mean of fold AUCs an interval, A9's labelling of the endpoint cells, A3's refusal of a
deployable alarm, A4's statement that tau-bench row-level cross-validation is not an unseen-task
evaluation, and the chronology sentence that keeps the extension exploratory whichever way it came
out. Sentences that carry characters LaTeX would read as markup stay in the comment lines, unparsed.

Usage::

    python tools/emit_live_size_audit_table.py
    python tools/emit_live_size_audit_table.py --check --paper <paper directory>

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

from emit_live_prefix_table import _CORPORA, _METHODS  # noqa: E402

# Paths are joined with joinpath rather than with "/". The suite bans the division operator outright,
# so that no score, difference or ratio can be recomputed here, and a path join is the only place "/"
# would otherwise appear. Spelling it out costs a word and leaves the ban with no exceptions.
RECORD = ROOT.joinpath("live_size_control_audit_results.json")

_APPENDIX = "09_appendix.tex"
_BEGIN = ("% BEGIN GENERATED tab:live-size-control -- regenerate with: "
          "python tools/emit_live_size_audit_table.py")
_END = "% END GENERATED tab:live-size-control"

# The record's corpus keys, in the order the batch scored them and this table prints them, mapped to
# the token emit_live_prefix_table indexes its own corpora by. The two vocabularies are not the same
# word: the loader says "swegym" and the LIVE claim ids say "swe", and reading one for the other
# would print SWE-Gym rows under a tau-bench heading. tools/emit_live_size_audit.py carries the same
# map as CLAIM_CORPUS, and a test holds the two together.
_CORPUS_ORDER = ("swegym", "tau")
_CORPUS_TOKEN = {"swegym": "swe", "tau": "tau"}

# A2's four arms in A2's order: the two linear arms that reproduce the published prefix numbers, then
# the controlled pair. Order is written here because the record stores its arms in a sorted mapping,
# so the file cannot supply it; the membership is still checked against the record at every cell.
_ARM_ORDER = ("size (flat)", "auditable (size+deps)", "size (spline)", "size-spline + linear-deps")

# The two arms that print on no other table, so no other table can supply their names. The two linear
# arms are deliberately absent: they are entrants on the LIVE prefix tables and their printed names
# come from there.
_NEW_ARM_NAMES = {
    "size (spline)": "size (spline)",
    "size-spline + linear-deps": "size-spline + linear-deps",
}

# The roles A9 distinguishes. "reproduction control" is the one that has to reach the printed table
# rather than the caption alone: the 100 percent effects reproduce a cell whose value is already
# committed, and a reader scanning the rows has to meet that label beside the number.
_ROLES = ("primary", "declared cell", "reproduction control")
_REPRODUCTION_CONTROL = "reproduction control"
_PRIMARY = "primary"

# The two interval constructions a row may carry, keyed by the (method, axis) pair the record stores,
# and the sub-record each one's endpoints have to come from. Nothing else is accepted: an unrecognised
# pair is a record this file has never been told how to read, and printing it would print an interval
# whose construction the caption then describes wrongly. A6 gives SWE-Gym paired DeLong and tau-bench
# the task-clustered bootstrap, and the two are not interchangeable on this corpus pair, because
# tau-bench's 660 runs are 165 task instances attempted four times each.
_DELONG = ("paired DeLong, conditional on the saved fits", "run-level sampling")
_CLUSTERED = ("task-clustered stratified percentile bootstrap", "task-cluster resampling")
_CONSTRUCTIONS = {_DELONG: "delong", _CLUSTERED: "clustered"}
_INTERVAL_SOURCE = {"delong": "delong", "clustered": "clustered_interval"}
# The second interval each row carries, which the caption names and the comment lines print. Neither
# is a reported uncertainty: the SWE-Gym one is the stratified paired bootstrap the core contrasts
# already carry beside DeLong, and the tau-bench one is the run-level DeLong that treats four
# attempts at one task as four observations.
_SECOND_INTERVAL = {"delong": "crosscheck_stratified_bootstrap",
                    "clustered": "reproduction_diagnostic"}

# Printed row labels for the three rows that are not an arm.
_EFFECT_LABEL = r"$D$ (matched $-$ flexible size)"
_INTERVAL_LABEL = r"95\% interval for $D$"
_ROLE_LABEL = "Role"

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
# names, file names and tool paths. Anything else is refused for the same reason prose is.
_TEXTTT_ALLOWED = re.compile(r"[A-Za-z0-9 ._{}()*+:/-]+")

_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
          "eleven", "twelve")


def _prose(text: str) -> str:
    r"""A record sentence, passed through to a caption with its percent signs escaped.

    The caption sentences that carry a claim about what this table is and is not are the record's
    own, word for word. Restating them here would put the reason a quantity was refused in two
    places, and the two would drift: the record would say the endpoint cells are reproduction
    controls while the manuscript said something adjacent to that.
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
    a cell moves the prose with the table. A roster this prose cannot describe has to stop the run: a
    caption that says eight cells over a table of nine is worse than no table.
    """
    if not 0 <= count < len(_WORDS):
        raise ValueError(f"no caption word for a count of {count}; the audit has outgrown the prose "
                         f"this caption was written for")
    return _WORDS[count]


def load() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _printed_arm_names() -> dict[str, str]:
    """One printed name per declared arm, taken from the LIVE prefix tables wherever they have one.

    ``size (flat)`` and ``auditable (size+deps)`` are entrants on Table~\\ref{tab:live-stream}, which
    prints the very cells this table controls. Naming them here a second time would let a rename
    there leave two spellings of one arm in one appendix, so they are read rather than written, and
    an arm that is in neither source stops the run instead of printing its raw record id.
    """
    published = dict(_METHODS)
    names = {}
    for arm in _ARM_ORDER:
        if arm in published:
            names[arm] = published[arm]
        elif arm in _NEW_ARM_NAMES:
            names[arm] = _NEW_ARM_NAMES[arm]
        else:
            raise ValueError(f"no printed name for the arm {arm!r}; the LIVE prefix tables do not "
                             f"carry it and this table has not been told how to print it")
    return names


def _corpus_printed(corpus: str) -> tuple[str, str]:
    """The printed name and sibling table label of one record corpus, read off the LIVE tables."""
    token = _CORPUS_TOKEN[corpus]
    for name, _family, printed, label in _CORPORA:
        if name == token:
            return printed, label
    raise ValueError(f"emit_live_prefix_table carries no corpus {token!r}; this table cannot name "
                     f"{corpus!r} the way the LIVE prefix tables do")


def _rows(record: dict) -> list[dict]:
    """The eight effect rows, checked against the declaration and against the corpus blocks.

    Order is checked rather than imposed. The record lists the cells corpus-major and, inside a
    corpus, in the declared prefix order; a row set that arrived in another order would still produce
    a plausible table with its roles and intervals against the wrong prefixes, and that is not a
    difference a reader could see. Everything else here is a cross-check between two places the
    record stores the same fact, because a table that read only one of them would agree with itself.
    """
    frozen = record["declaration"]["frozen_before_the_run"]
    rows = record["effects"]
    found = [(row["corpus"], row["prefix"]) for row in rows]
    wanted = [(corpus, prefix) for corpus in _CORPUS_ORDER for prefix in frozen["prefixes"]]
    if found != wanted:
        raise ValueError(f"the record's cells are not the declared corpora at the declared "
                         f"prefixes, in order; got {found}, wanted {wanted}")
    endpoint = frozen["prefixes"][-1]
    labels: dict[str, str] = {}
    for row in rows:
        corpus = record["corpora"][row["corpus"]]
        if row["cell"] != row["corpus"] + "." + str(row["prefix_percent"]):
            raise ValueError(f"the cell name {row['cell']!r} is not its own corpus and prefix")
        if [row["high"], row["low"]] != list(frozen["matched_contrast"]):
            raise ValueError(f"{row['cell']}: the contrast is {row['high']!r} minus {row['low']!r}, "
                             f"not the frozen {frozen['matched_contrast']}")
        if row["role"] not in _ROLES:
            raise ValueError(f"{row['cell']}: unrecognised role {row['role']!r}")
        if (row["role"] == _REPRODUCTION_CONTROL) != (row["prefix"] == endpoint):
            raise ValueError(f"{row['cell']}: role {row['role']!r} at prefix {row['prefix']}; A9 "
                             f"makes the {endpoint} cells the reproduction controls and no others")
        if row["independent_new_evidence"] != (row["role"] != _REPRODUCTION_CONTROL):
            raise ValueError(f"{row['cell']}: role {row['role']!r} contradicts "
                             f"independent_new_evidence={row['independent_new_evidence']}")
        if labels.setdefault(row["role"], row["labelled_as"]) != row["labelled_as"]:
            raise ValueError(f"{row['cell']}: two cells share the role {row['role']!r} and are "
                             f"labelled differently")
        if row["rng_label"] != frozen["rng_labels"][row["cell"]]:
            raise ValueError(f"{row['cell']}: RNG label {row['rng_label']!r} is not the frozen "
                             f"{frozen['rng_labels'][row['cell']]!r}")
        if row["rng_base_seed"] != frozen["rng_base_seed"]:
            raise ValueError(f"{row['cell']}: base seed {row['rng_base_seed']} is not the frozen "
                             f"{frozen['rng_base_seed']}")
        _check_arms(record, row)
        _check_interval(row)
        if corpus["prefixes"] != frozen["prefixes"]:
            raise ValueError(f"{row['corpus']}: prefixes {corpus['prefixes']} against the frozen "
                             f"{frozen['prefixes']}")
        if corpus["n_runs"] != frozen["declared_population"][row["corpus"]]:
            raise ValueError(f"{row['corpus']}: {corpus['n_runs']} runs against the declared "
                             f"{frozen['declared_population'][row['corpus']]}")
    primary = [row for row in rows if row["role"] == _PRIMARY]
    if len(primary) != 1:
        raise ValueError(f"{len(primary)} cells carry the primary role; A5 nominates one")
    if primary[0]["cell"] != record["primary"]["cell"]:
        raise ValueError(f"the primary row is {primary[0]['cell']!r} and the record's primary is "
                         f"{record['primary']['cell']!r}")
    if frozen["primary_cell"] != {"corpus": primary[0]["corpus"], "prefix": primary[0]["prefix"]}:
        raise ValueError(f"the primary row {primary[0]['cell']!r} is not the frozen "
                         f"{frozen['primary_cell']}")
    return rows


def _check_arms(record: dict, row: dict) -> None:
    """The four arm scores, checked against the corpus block that also stores them.

    The record keeps every arm score twice, inside the effect row and inside the prefix cell it came
    from. Printing one while the other says something else is the failure this catches, and it is the
    failure a reader cannot see: the four columns would still sum to nothing in particular and the
    ``D`` column beside them would still be a number the record supports.
    """
    cell = record["corpora"][row["corpus"]]["by_prefix"][str(row["prefix_percent"])]
    roles = record["corpora"][row["corpus"]]["arm_roles"]
    if sorted(cell["arms"]) != sorted(_ARM_ORDER) or sorted(roles) != sorted(_ARM_ORDER):
        raise ValueError(f"{row['cell']}: the arms are {sorted(cell['arms'])}, not the four "
                         f"declared {sorted(_ARM_ORDER)}")
    for arm in _ARM_ORDER:
        scored, printed = cell["arms"][arm], row["arm_scores"][arm]
        for quantity in ("seed_averaged_oof_roc_auc", "mean_fold_roc_auc"):
            if scored[quantity] != printed[quantity]:
                raise ValueError(f"{row['cell']} {arm}: the effect row carries "
                                 f"{printed[quantity]!r} for {quantity} and the prefix cell carries "
                                 f"{scored[quantity]!r}")
        if scored["role"] != roles[arm]:
            raise ValueError(f"{row['cell']} {arm}: role {scored['role']!r} against the corpus's "
                             f"{roles[arm]!r}")
    matched = cell["effects"][row["high"] + " - " + row["low"]]
    if matched["kind"] != "matched":
        raise ValueError(f"{row['cell']}: the contrast this table prints is recorded as "
                         f"{matched['kind']!r} rather than matched")
    if matched["seed_averaged_oof_difference"] != row["point"]:
        raise ValueError(f"{row['cell']}: the effect row says {row['point']!r} and the prefix "
                         f"cell's own contrast says {matched['seed_averaged_oof_difference']!r}")
    reproduction = row["reproduction_quantity"]
    if matched["mean_fold_difference"] != reproduction["mean_fold_difference"]:
        raise ValueError(f"{row['cell']}: the reproduction quantity disagrees with the prefix "
                         f"cell's own mean-fold difference")
    if reproduction["interval"] is not None:
        raise ValueError(f"{row['cell']}: A5 refuses the mean of fold AUCs an interval and this "
                         f"record carries one: {reproduction['interval']!r}")


def _check_interval(row: dict) -> None:
    """One row's interval, checked against the construction the record says produced it."""
    kind = _CONSTRUCTIONS.get((row["interval_method"], row["interval_axis"]))
    if kind is None:
        raise ValueError(f"{row['cell']}: unrecognised interval construction "
                         f"({row['interval_method']!r}, {row['interval_axis']!r})")
    source = row[_INTERVAL_SOURCE[kind]]
    if row["interval_95"] != source["interval_95"]:
        raise ValueError(f"{row['cell']}: the printed interval {row['interval_95']!r} is not the "
                         f"{kind} interval {source['interval_95']!r} it is labelled as")
    low, high = row["interval_95"]
    if not low <= row["point"] <= high:
        raise ValueError(f"{row['cell']}: the point {row['point']!r} is outside its own interval "
                         f"[{low!r}, {high!r}]")
    if not row["delong_point_agrees_with_the_estimand"]["within_tolerance"]:
        raise ValueError(f"{row['cell']}: the interval's own AUC difference and the declared "
                         f"estimand disagree beyond the frozen tolerance")
    if _SECOND_INTERVAL[kind] not in row:
        raise ValueError(f"{row['cell']}: a {kind} row carries no {_SECOND_INTERVAL[kind]}")


def _construction(rows: list[dict], corpus: str) -> str:
    """The one interval construction a corpus's four rows share, refusing a corpus that mixes two."""
    kinds = {_CONSTRUCTIONS[(row["interval_method"], row["interval_axis"])]
             for row in rows if row["corpus"] == corpus}
    if len(kinds) != 1:
        raise ValueError(f"{corpus}: its rows carry {sorted(kinds)} interval constructions; the "
                         f"caption describes one per corpus")
    return kinds.pop()


def _clustered_sample(rows: list[dict]) -> dict:
    """The task-cluster sample every clustered row was resampled at, checked for agreement.

    The caption states the cluster count and the stratum split once for the whole corpus, so rows
    that disagreed with each other would be described by a caption true of some of them.
    """
    clustered = [row["clustered_interval"] for row in rows if "clustered_interval" in row]
    if not clustered:
        raise ValueError("no row carries a clustered interval, so the caption cannot describe one")
    counts = {record["n_clusters"] for record in clustered}
    draws = {record["draws"] for record in clustered}
    noise = {record["monte_carlo_noise_on_one_endpoint"] for record in clustered}
    strata = {tuple(sorted(record["clusters_per_stratum"].items())) for record in clustered}
    if len(counts) != 1 or len(draws) != 1 or len(strata) != 1 or len(noise) != 1:
        raise ValueError(f"the clustered rows disagree on their resampling: clusters "
                         f"{sorted(counts)}, draws {sorted(draws)}, noise {sorted(noise)}")
    return {"clusters": counts.pop(), "draws": draws.pop(), "strata": dict(strata.pop()),
            "noise": noise.pop()}


def _temporal(record: dict, rows: list[dict]) -> dict:
    """A7's secondary contrast, tied to the two shipped cells it is the difference of.

    ``T`` is the quantity the claim "the increment is larger early than at the endpoint" actually
    requires, and it is the one quantity here that is not a cell. Its components are checked against
    the printed rows by name and by value, so it cannot quietly become the difference of two numbers
    this table does not print, and so that a reader who wants to check it has both ends on the page.
    """
    temporal = record["temporal_contrast"]
    frozen = record["declaration"]["frozen_before_the_run"]
    if temporal["rng_label"] != frozen["temporal_rng_label"]:
        raise ValueError(f"the temporal contrast's RNG label {temporal['rng_label']!r} is not the "
                         f"frozen {frozen['temporal_rng_label']!r}")
    if temporal["additional_fits"] != 0:
        raise ValueError(f"A7 allows the temporal contrast no additional classifier fit and the "
                         f"record reports {temporal['additional_fits']}")
    if not temporal["point_recomputed_by_the_bootstrap_scorer"]["within_tolerance"]:
        raise ValueError("the temporal point and the point its own bootstrap scorer recomputed "
                         "disagree beyond the frozen tolerance")
    low, high = temporal["interval_95"]
    if not low <= temporal["point"] <= high:
        raise ValueError(f"the temporal point {temporal['point']!r} is outside its own interval")
    by_key = {f"D({row['corpus']}, {row['prefix']:.2f})": row for row in rows}
    components = []
    for key, value in temporal["components"].items():
        row = by_key.get(key)
        if row is None or row["point"] != value:
            raise ValueError(f"the temporal component {key!r} is not one of the effects this table "
                             f"prints")
        if key not in temporal["quantity"]:
            raise ValueError(f"the temporal quantity {temporal['quantity']!r} does not name its own "
                             f"component {key!r}")
        components.append(row)
    if len(components) != 2:
        raise ValueError(f"the temporal contrast carries {len(components)} components; A7 declares "
                         f"the difference of two cells")
    early, late = components
    if early["corpus"] != late["corpus"]:
        raise ValueError(f"the temporal contrast spans {early['corpus']} and {late['corpus']}; A7 "
                         f"declares it within one corpus")
    return {"record": temporal, "corpus": early["corpus"], "early": early, "late": late}


def _outcome(record: dict, rows: list[dict]) -> dict:
    """A8's branch, checked to be the primary cell's own reading and no other cell's.

    A8's last bullet is the one a table can break silently: if 25 percent disappoints, a better
    result at 50 or 75 percent is not promoted to primary. The strongest cell in this batch is a
    reproduction control at the endpoint, so the temptation is real rather than hypothetical, and the
    check is that the three numbers the branch was applied to are the primary row's own, to the bit.
    """
    outcome = record["outcome"]
    primary = next(row for row in rows if row["role"] == _PRIMARY)
    if outcome["primary_cell"] != primary["cell"]:
        raise ValueError(f"the outcome reads {outcome['primary_cell']!r} and the primary row is "
                         f"{primary['cell']!r}")
    if outcome["point"] != primary["point"] or outcome["interval_95"] != primary["interval_95"]:
        raise ValueError(f"the outcome's point and interval are not {primary['cell']}'s own")
    if outcome["branch"] not in _BRANCH_READING:
        raise ValueError(f"unrecognised A8 branch {outcome['branch']!r}")
    threshold = record["declaration"]["frozen_before_the_run"]["substantial_threshold"]
    if outcome["substantial_threshold"] != threshold:
        raise ValueError(f"the outcome reads a threshold of {outcome['substantial_threshold']!r} "
                         f"against the frozen {threshold!r}")
    return {"record": outcome, "primary": primary}


def _gate(record: dict) -> dict:
    """The A10.4 gate, which has to have admitted this batch before any of it is printed."""
    gate = record["preflight"]["gate"]
    if not gate["admits_the_batch"] or gate["blocking_disagreements"]:
        raise ValueError(f"the A10.4 gate did not admit this batch: "
                         f"{len(gate['blocking_disagreements'])} blocking disagreement(s)")
    exempt = gate["exempted_field"]
    for defect in gate["record_defects_carried_forward"]:
        if defect["source"] != exempt["source"] or defect["quantity"] != exempt["quantity"]:
            raise ValueError(f"a carried-forward defect is outside the exempted field: {defect}")
    return gate


# What each A8 branch means, in that branch's own terms. The four are the record's own vocabulary and
# all four are reachable: a caption written for one of them would be wrong rather than incomplete in
# the others, which is why each has a sentence here and a test that exercises it.
_BRANCH_READING = {
    "unresolved": "An interval containing zero is unresolved whatever the point is, and is "
                  "reported as unresolved rather than as a small effect.",
    "small-positive": "The interval excludes zero on the positive side at a point below the "
                      "declared threshold, so the supported positive is smaller than the size this "
                      "paper declared worthwhile.",
    "substantial-positive": "The point is at least the declared threshold and its interval excludes "
                            "zero, so A8 reads the surviving increment as substantial.",
    "erased-or-reversed": "The interval excludes zero on the negative side, so the increment is "
                          "erased or reversed at this cell once the size reference bends.",
}


def _score(value: float) -> str:
    """One arm cell: three decimals, the precision the LIVE prefix tables print scores at."""
    return f"{value:.3f}"


def _effect(value: float) -> str:
    """One effect cell: four decimals and always a sign, because these differences run to 1e-4."""
    return "$" + f"{value:+.4f}" + "$"


def _interval(low: float, high: float) -> str:
    return "$[" + f"{low:+.4f}" + ", " + f"{high:+.4f}" + "]$"


def _shared(rows: list[dict], *path: str) -> str:
    """One sentence every row carries, stated once, and refused when the rows disagree.

    A5's refusal to give the mean of fold AUCs an interval, A6's caveat about what these intervals
    are conditional on, and the estimand itself are identical at all eight cells. Eight copies of a
    sentence is eight places for a later regeneration to drift in, so the caption and the comment
    lines carry one; requiring the rows to agree first is what makes the one they carry true of all
    of them rather than true of whichever row was read.
    """
    found = set()
    for row in rows:
        value = row
        for key in path:
            value = value[key]
        found.add(value)
    if len(found) != 1:
        raise ValueError(f"the rows carry {len(found)} different {'.'.join(path)} sentences, and "
                         f"this block states one")
    return found.pop()


def _provenance(record: dict, rows: list[dict], temporal: dict, outcome: dict,
                gate: dict) -> list[str]:
    """Comment lines binding each printed cell to the record's own full-precision value.

    The printed table rounds arm scores to three decimals and effects to four, so a reader who wants
    to re-derive a row, or a checker comparing two runs, has nothing to work from in the printed
    cells. These lines carry what the record stores. They also carry the three things the printed
    body deliberately does not: the mean of the 25 fold AUCs, which A5 keeps apart from the estimand
    and refuses an interval; the second interval each row carries, which is a crosscheck on SWE-Gym
    and a reproduction diagnostic on tau-bench and a reported uncertainty on neither; and the record
    sentences that carry characters LaTeX would read as markup, which a comment line does not parse.
    """
    inputs = record["inputs"]
    frozen = record["declaration"]["frozen_before_the_run"]
    fixed = record["declaration"]["fixed_by_this_implementation"]
    settings = record["settings"]
    exempt = gate["exempted_field"]
    lines = [
        _BEGIN,
        f"% Source: tools/{RECORD.name}, schema {record['schema_version']}, generated by "
        f"{record['generated_by']}. This file computes nothing; every number below is read.",
        f"% Declaration: {record['declaration']['file']} Part {record['declaration']['part']}, "
        f"sha256 {inputs['declaration']['sha256']}.",
        f"% Question: {record['question']}",
        f"% Estimand: {_shared(rows, 'estimand')}",
        f"% Reproduction quantity: "
        f"{_shared(rows, 'reproduction_quantity', 'why_no_interval')}",
    ]
    for corpus in _CORPUS_ORDER:
        lines.append(f"% Corpus {corpus}: {record['corpora'][corpus]['corpus_line']}")
    lines += [
        f"% Fits: {settings['declared_fits']} declared; seeds {frozen['seeds']}; "
        f"{frozen['folds']} folds; confidence {settings['confidence_level']}; "
        f"z_975={settings['z_975']}.",
        f"% Spline: degree={frozen['spline']['degree']}; n_knots={frozen['spline']['n_knots']}; "
        f"knots={frozen['spline']['knots']}; include_bias={frozen['spline']['include_bias']}; "
        f"extrapolation={frozen['spline']['extrapolation']}.",
        f"% Environment: Python {inputs['python']}; numpy {inputs['package_versions']['numpy']}; "
        f"scikit-learn {inputs['package_versions']['scikit-learn']}; "
        f"scipy {inputs['package_versions']['scipy']}; {inputs['platform']}.",
        f"% Repositories: catchbench {inputs['repositories']['catchbench']['commit']}; "
        f"grade {inputs['repositories']['grade']['commit']}; "
        f"paper {inputs['repositories']['paper']['commit']}.",
        f"% A10.4 gate: admits_the_batch={gate['admits_the_batch']}; "
        f"preflight_cleared_on_its_own_terms="
        f"{record['preflight']['cleared_on_its_own_terms']}; exempted field {exempt['field']} "
        f"({exempt['quantity']}) in {exempt['source']}; "
        f"{len(gate['record_defects_carried_forward'])} defect(s) carried forward.",
        f"% A10.4 reading: {gate['reading']}",
    ]
    for defect in gate["record_defects_carried_forward"]:
        lines.append(f"% carried forward: {defect['corpus']} {defect['prefix']}% {defect['arm']} "
                     f"{defect['quantity']}: observed={defect['observed']!r}, "
                     f"reference={defect['reference']!r}, difference={defect['difference']:+.3e}")
    lines.append(f"% Inputs note: {inputs['note']}")
    lines.append(f"% Every interval below: {_shared(rows, 'conditional_on_the_saved_fits')}")
    for corpus in _CORPUS_ORDER:
        block = [row for row in rows if row["corpus"] == corpus]
        kind = _CONSTRUCTIONS[(block[0]["interval_method"], block[0]["interval_axis"])]
        roles = {row[_SECOND_INTERVAL[kind]]["role"] for row in block}
        if len(roles) != 1:
            raise ValueError(f"{corpus}: its rows give the second interval {len(roles)} different "
                             f"roles, and this block states one per corpus")
        lines.append(f"% {corpus} second interval ({_SECOND_INTERVAL[kind]}): {roles.pop()}")
    for row in rows:
        low, high = row["interval_95"]
        second = row[_SECOND_INTERVAL[_CONSTRUCTIONS[(row["interval_method"],
                                                      row["interval_axis"])]]]
        second_low, second_high = second["interval_95"]
        lines.append(f"% {row['cell']}: role={row['role']}; labelled_as={row['labelled_as']}; "
                     f"independent_new_evidence={row['independent_new_evidence']}; "
                     f"rng_label={row['rng_label']}; base_seed={row['rng_base_seed']}")
        for arm in _ARM_ORDER:
            scored = row["arm_scores"][arm]
            lines.append(f"% {row['cell']}: arm {arm}: "
                         f"seed_averaged_oof={scored['seed_averaged_oof_roc_auc']:.12f}; "
                         f"mean_fold={scored['mean_fold_roc_auc']:.12f}")
        lines.append(f"% {row['cell']}: D={row['point']:.12f}; "
                     f"interval=[{low:.12f}, {high:.12f}]; method={row['interval_method']}; "
                     f"axis={row['interval_axis']}")
        lines.append(f"% {row['cell']}: mean_fold_difference="
                     f"{row['reproduction_quantity']['mean_fold_difference']:.12f}; "
                     f"interval={row['reproduction_quantity']['interval']}")
        lines.append(f"% {row['cell']}: second interval [{second_low:.12f}, {second_high:.12f}]")
    contrast = temporal["record"]
    low, high = contrast["interval_95"]
    lines += [
        f"% temporal: {contrast['quantity']}; point={contrast['point']:.12f}; "
        f"interval=[{low:.12f}, {high:.12f}]; method={contrast['interval_method']}",
        f"% temporal: components={contrast['components']}; replicates={contrast['replicates']}; "
        f"rng_label={contrast['rng_label']}; base_seed={contrast['rng_base_seed']}; "
        f"additional_fits={contrast['additional_fits']}",
        f"% temporal: bootstrap_mean={contrast['bootstrap_mean']:.12f}; "
        f"bootstrap_sd={contrast['bootstrap_sd']:.12f}; "
        f"two_sided_tail_p={contrast['two_sided_tail_p']:.12f}; "
        f"discarded_single_class_draws={contrast['discarded_single_class_draws']}",
        f"% temporal: {contrast['role']}",
    ]
    reading = outcome["record"]
    low, high = reading["interval_95"]
    lines += [
        f"% outcome: branch={reading['branch']}; primary_cell={reading['primary_cell']}; "
        f"point={reading['point']:.12f}; interval=[{low:.12f}, {high:.12f}]",
        f"% outcome: interval_excludes_zero={reading['interval_excludes_zero']}; "
        f"point_at_least_substantial={reading['point_at_least_substantial']}; "
        f"upper_endpoint_below_substantial={reading['upper_endpoint_below_substantial']}; "
        f"threshold={reading['substantial_threshold']}",
        f"% outcome: {reading['rule']}",
        f"% outcome: branch order: {fixed['a8_branch_order']}",
        f"% outcome: fourth bullet: {fixed['a8_fourth_bullet']}",
        f"% Failures: {len(record['failures'])}. Warnings: {len(record['warnings'])}.",
    ]
    for failure in record["failures"]:
        lines.append(f"% failure: {failure}")
    for warning in record["warnings"]:
        lines.append(f"% warning: {warning}")
    return lines


def _primary_sentences(record: dict, outcome: dict, printed_corpus: dict) -> list[str]:
    """The nominated cell, the branch its numbers fall in, and what that branch does not license."""
    reading = outcome["record"]
    primary = outcome["primary"]
    low, high = reading["interval_95"]
    sentences = [
        f"The cell A5 nominated before any of the {_word(len(record['effects']))} was computed is "
        f"{printed_corpus[primary['corpus']]} at {primary['prefix_percent']}" + r"\%" + ": $D$ = "
        + _effect(reading["point"]) + ", " + _interval(low, high) + ".",
        f"Its branch, by the rule A8 froze before the run, is {_prose(reading['branch'])}.",
        _prose(reading["rule"]) + ".",
        _BRANCH_READING[reading["branch"]],
    ]
    if reading["upper_endpoint_below_substantial"]:
        sentences.append(
            "The upper endpoint sits below the declared "
            + f"{reading['substantial_threshold']}, which rules out that planning magnitude under "
            "this conditional analysis and does not establish zero information.")
    sentences.append(f"The {reading['substantial_threshold']} threshold is "
                     + _prose(reading["threshold_is"]) + ".")
    return sentences


def _failure_sentence(record: dict) -> str:
    """What the batch recorded, in the branch it landed in.

    A9 asks for every failure and A11 asks for the partial results beside them, so a run that lost a
    corpus has to say so in the caption rather than print seven cells as though eight were declared.
    The empty branch is the one this batch took, and it is stated rather than left to silence.
    """
    failures, warnings = record["failures"], record["warnings"]
    if not failures:
        return (f"The batch recorded {_word(len(failures))} failures and "
                f"{_word(len(warnings))} warnings.")
    return (f"The batch recorded {_word(len(failures))} "
            + ("failure" if len(failures) == 1 else "failures")
            + f", named in the rows below and given in full in this block's comment lines, and "
            f"{_word(len(warnings))} warnings.")


def _gate_sentence(record: dict, gate: dict) -> str:
    """What the A10.4 preflight did, in the branch the record is in.

    The preflight did not clear on its own terms on this run: two diagnostic fields inside
    ``statistical_tests_results.json`` do not reproduce, and the declaration's dated entry records
    them as findings about that artifact rather than as a failure of this route. A caption that
    stated that as a literal would keep stating it after a later run cleared outright, so the
    sentence is chosen from the flag and both branches are exercised by the tests.
    """
    exempt = gate["exempted_field"]
    defects = gate["record_defects_carried_forward"]
    if record["preflight"]["cleared_on_its_own_terms"] and not defects:
        return ("The A10.4 preflight cleared on its own terms and admitted this batch, so every "
                "published prefix number this route reproduces did so within the frozen tolerance.")
    return ("The A10.4 preflight did not clear on its own terms and admitted this batch under the "
            "declaration's dated entry, with " + _word(len(defects)) + " "
            + ("disagreement" if len(defects) == 1 else "disagreements") + " in the "
            + _texttt(exempt["field"]) + " field of " + _texttt(exempt["source"])
            + " carried forward as findings about that record; the batch reads feature matrices "
            "and fits arms and consumes no field of it.")


def _caption(record: dict, rows: list[dict], temporal: dict, outcome: dict, gate: dict,
             names: dict[str, str], printed_corpus: dict, sample: dict) -> list[str]:
    """One clause per line, so a caption edit shows as a one-line delta in --check."""
    interpretation = record["interpretation"]
    contrast = temporal["record"]
    roles = record["corpora"][_CORPUS_ORDER[0]]["arm_roles"]
    labels = {corpus: _corpus_printed(corpus)[1] for corpus in _CORPUS_ORDER}
    first = {corpus: next(row for row in rows if row["corpus"] == corpus)
             for corpus in _CORPUS_ORDER}
    seconds = {}
    for corpus, row in first.items():
        kind = _CONSTRUCTIONS[(row["interval_method"], row["interval_axis"])]
        seconds[corpus] = row[_SECOND_INTERVAL[kind]]["role"]
    return [
        r"\caption{" + _prose(record["question"]),
        "The four arms are the declaration's: "
        + _and_list([names[arm] + " is the " + _prose(roles[arm]) for arm in _ARM_ORDER]) + ".",
        "Arm cells are the ROC-AUC of five-seed-averaged out-of-fold failure probabilities, the "
        "quantity $D$ is defined on: " + _prose(_shared(rows, "estimand")) + ".",
        r"Table~\ref{" + labels[_CORPUS_ORDER[0]] + r"} and Table~\ref{" + labels[_CORPUS_ORDER[1]]
        + "} print the board's quantity instead, the mean of the fold AUCs, so their cells for the "
        "two linear arms are not these; "
        + _prose(_shared(rows, "reproduction_quantity", "why_no_interval"))
        + ", and both quantities are carried at full precision in this block's comment lines.",
        "Every column is rounded on its own, so the difference of two printed arm scores can differ "
        "from the printed $D$ in the last place.",
        _prose(_shared(rows, "conditional_on_the_saved_fits")),
        printed_corpus[_CORPUS_ORDER[0]] + " intervals are "
        + _prose(first[_CORPUS_ORDER[0]]["interval_method"]) + "; "
        + printed_corpus[_CORPUS_ORDER[1]] + " intervals are the "
        + _prose(first[_CORPUS_ORDER[1]]["interval_method"]) + " over "
        + f"{sample['clusters']} task clusters ("
        + _and_list([f"{count} {name}" for name, count in sample["strata"].items()])
        + f") at {sample['draws']:,} draws.",
        f"Effects and endpoints are printed to four decimals because several are smaller than a "
        f"thousandth, and the record puts the Monte Carlo noise on one clustered endpoint at "
        f"{sample['noise']} at that "
        + printed_corpus[_CORPUS_ORDER[1]] + " draw count, so a "
        + printed_corpus[_CORPUS_ORDER[1]] + " endpoint is printed to more digits than one draw of "
        "the bootstrap carries.",
        "Each row also carries a second interval in this block's comment lines, "
        + _and_list([printed_corpus[corpus] + " a " + _prose(seconds[corpus]).split(";")[0]
                     for corpus in _CORPUS_ORDER])
        + "; neither is a reported uncertainty.",
        _prose(interpretation["endpoint_cells_are_controls"]),
        f"The {_ROLE_LABEL} row carries that label beside the cells it applies to.",
        *_primary_sentences(record, outcome, printed_corpus),
        "$T$ is the secondary temporal contrast A7 declared before the batch: "
        + f"{contrast['replicates']:,} draws of a " + _prose(contrast["interval_method"]) + ", "
        + _word(contrast["additional_fits"]) + " additional classifier fit.",
        "It is " + _prose(contrast["role"]),
        _prose(interpretation["tau_is_still_row_level_cv"]),
        _prose(interpretation["no_deployable_alarm"]),
        _prose(interpretation["no_simultaneous_coverage"]),
        _prose(interpretation["exploratory"]),
        _prose(interpretation["one_batch"]),
        _failure_sentence(record),
        _gate_sentence(record, gate),
        r"Generated by " + _texttt("tools/emit_live_size_audit_table.py") + ".}",
    ]


def _body(record: dict, rows: list[dict], temporal: dict, names: dict[str, str],
          printed_corpus: dict) -> list[str]:
    """Two corpus blocks of seven rows, then the contrast that is not a cell, then any failure."""
    lines = []
    for corpus in _CORPUS_ORDER:
        block = [row for row in rows if row["corpus"] == corpus]
        corpus_record = record["corpora"][corpus]
        if lines:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{5}{@{}l}{\textbf{" + printed_corpus[corpus] + "}, "
                     + f"{corpus_record['n_runs']} runs, {corpus_record['n_failed']} failed"
                     + r"} \\")
        for arm in _ARM_ORDER:
            cells = [_score(row["arm_scores"][arm]["seed_averaged_oof_roc_auc"]) for row in block]
            lines.append(" & ".join([names[arm], *cells]) + r" \\")
        lines.append(" & ".join([_EFFECT_LABEL, *[_effect(row["point"]) for row in block]])
                     + r" \\")
        lines.append(" & ".join([_INTERVAL_LABEL,
                                 *[_interval(*row["interval_95"]) for row in block]]) + r" \\")
        lines.append(" & ".join([_ROLE_LABEL, *[row["role"] for row in block]]) + r" \\")
    contrast = temporal["record"]
    low, high = contrast["interval_95"]
    lines += [
        r"\midrule",
        r"\multicolumn{5}{@{}l}{Secondary (A7): $T$ = " + printed_corpus[temporal["corpus"]]
        + " $D$ at " + f"{temporal['early']['prefix_percent']}" + r"\%" + " minus $D$ at "
        + f"{temporal['late']['prefix_percent']}" + r"\%" + " = " + _effect(contrast["point"])
        + ", " + _interval(low, high) + r"} \\",
    ]
    if record["failures"]:
        lines.append(r"\midrule")
        for failure in record["failures"]:
            lines.append(r"\multicolumn{5}{@{}l}{Recorded failure: " + _texttt(failure["kind"])
                         + " on " + _texttt(str(failure.get("corpus", "-")))
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
    """One appendix float: the declared inventory A9 asks for, between one pair of sentinels."""
    rows = _rows(record)
    temporal = _temporal(record, rows)
    outcome = _outcome(record, rows)
    gate = _gate(record)
    names = _printed_arm_names()
    printed_corpus = {corpus: _corpus_printed(corpus)[0] for corpus in _CORPUS_ORDER}
    for corpus in _CORPUS_ORDER:
        # Called for the refusal rather than the answer: the caption names one interval
        # construction per corpus, so a corpus whose four rows mix two has no true description.
        _construction(rows, corpus)
    sample = _clustered_sample(rows)
    prefixes = [f"{row['prefix_percent']}" + r"\%"
                for row in rows if row["corpus"] == _CORPUS_ORDER[0]]
    lines = _provenance(record, rows, temporal, outcome, gate)
    short, standfirst = _split_caption(
        _caption(record, rows, temporal, outcome, gate, names, printed_corpus, sample))
    lines += standfirst
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        *short,
        r"\label{tab:live-size-control}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        " & ".join(["Arm or quantity", *prefixes]) + r" \\",
        r"\midrule",
        *_body(record, rows, temporal, names, printed_corpus),
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
            print(f"STALE {_APPENDIX}: live-size-control marker {marker!r} appears "
                  f"{len(found)} times, expected once")
            return 1
        markers.append(found[0])
    begin, end = markers
    if end.start() < begin.start():
        print(f"STALE {_APPENDIX}: live-size-control end marker precedes its begin marker")
        return 1
    actual = raw[begin.start():end.start() + len(_END)]
    wanted = generated.encode("utf-8")
    if actual != wanted:
        print(f"STALE {_APPENDIX}: live-size-control block differs from the generated block")
        for index, (got, want) in enumerate(zip(actual.splitlines(keepends=True),
                                                wanted.splitlines(keepends=True)), 1):
            if got != want:
                print(f"first difference at block line {index}: paper={got!r}; generated={want!r}")
                break
        print("\n".join(difflib.unified_diff(
            actual.decode("utf-8", errors="replace").splitlines(), generated.splitlines(),
            fromfile=f"paper/{_APPENDIX}", tofile="generated/tab:live-size-control", lineterm="")))
        print("Regenerate with: python tools/emit_live_size_audit_table.py")
        return 1
    print("paper is current: tab:live-size-control (eight matched effects, one temporal contrast)")
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
        print(f"ERROR: LIVE prefix size-control table: {exc}", file=sys.stderr)
        return 1
    if args.check:
        return check(Path(args.paper), generated)
    sys.stdout.buffer.write((generated + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
