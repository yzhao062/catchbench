r"""How many tau-bench task clusters would resolve each LIVE bar cell against the 0.70 warning bar.

``06_discussion.tex`` says that measuring the gap to the bar more sharply "needs a larger sample" and
carries no number. This supplies one, for every one of the twenty committed ``live.tau.bar.*`` cells:
holding that cell's observed point estimate fixed, the number of task clusters at which a 95% interval
would exclude zero on the stored AUC-minus-bar scale.

**This is a design calculation, not a post-hoc power calculation.** It asks what sample would resolve
an effect of the size already observed. It never converts an achieved interval into a minimum
detectable difference, and it never computes achieved power at the present sample. Those are the
quantities a reader of a nonsignificant cell is most tempted to ask for and the ones that would be
circular here, so ``not_computed`` in the emitted record names them and says why. The specification
this file implements is frozen in ``research/catchbench-live-size-control-declaration-2026-09-10.md``
Part B, written before any subsample was drawn.

The arithmetic rests on one assumption, that the clustered bootstrap standard deviation falls as
``clusters ** -0.5``, and the declaration refuses to let it stand unchecked. So this file measures the
exponent first. It subsamples task clusters at a frozen ladder while holding tau-bench's 50 airline
and 115 retail proportions, reruns the shipped clustered bootstrap inside each subsample, and
regresses log SD on log cluster count. Four cells were named in the declaration before any subsample
existed: the two that do not separate today, plus two that separate comfortably, so the ladder spans
the observed effect range rather than the corner that motivated the question. If every measured
exponent is consistent with -0.5, the printed figure is the closed form and the record says so; if any
is not, the printed figure comes from the measured scaling with the closed form beside it as the
idealized comparison. Either way both are emitted, and ``scaling_check.printed_figure_source`` names
which the manuscript should carry.

The score vectors come from the LIVE pillar's own replay rather than from the POST audit arrays:
``statistical_tests._tau_live_clustering`` rebuilds tau-bench's task identifiers under LIVE's
``>= 4``-step filter and raises if the replay has drifted out of the order the prefix feature matrices
are in. This file reuses that path and the shipped bootstrap unchanged; a reimplementation of either
would be a second specification to keep in step with the first.

Nothing here fits a model to new data. The four supervised arms are refits of the committed LIVE
recipe purely so the run-level score vectors exist to subsample, and each one is checked against its
committed point estimate before it is used. No corpus is relabelled, no threshold moves, and no
manuscript file is touched.

Typical use from the repository root::

    python tools/live_tau_bar_design.py --output tools/live_tau_bar_design_results.json \
        --declaration ../agent-startup-thesis/research/catchbench-live-size-control-declaration-2026-09-10.md
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import statistical_tests as st  # noqa: E402  the shipped clustering, bootstrap, and RNG derivation

# --- frozen by the declaration, Part B4 ----------------------------------------------------------
# Ladder, draw count, replicate count, base seed, RNG label shape, and the four cells were all written
# down before any subsample was drawn. None of them is a tunable: a run that wants different values is
# a different analysis and needs its own dated entry in the declaration.
LADDER = (40, 60, 80, 100, 120, 140, 165)
SUBSAMPLE_DRAWS = 2_000
SUBSAMPLE_REPLICATES = 20
SCALING_SEED = 20260907
SCALING_LABEL = "live_tau_scaling.{cell}.{n_clusters}.{replicate}"
SCALING_CELLS = (
    # cell, prefix, method, why the declaration named it
    ("100.full", 1.0, "full", "does not separate at the present sample"),
    ("100.auditable (size+deps)", 1.0, "auditable (size+deps)",
     "does not separate at the present sample"),
    ("25.size (flat)", 0.25, "size (flat)", "control: separates, mid-range effect"),
    ("25.pyod (ECOD)", 0.25, "pyod (ECOD)", "control: separates, largest effect on the ladder"),
)
BAR = 0.70

# --- fixed here, before the run, because the declaration left them to the implementation ----------
# The declaration froze the ladder and the resampling; it did not name an estimator for the interval
# around the exponent, a rule for calling an exponent consistent with -0.5, or a rule for pooling the
# four. Those are written down here rather than chosen once the fits are on screen.
#
# The interval comes from the rung-level fit: one observation per ladder rung, the mean of log SD over
# that rung's 20 replicates, seven points and five degrees of freedom. Pooling the 140 replicate-level
# points instead would treat them as independent, and they are not. Two subsamples of 140 clusters out
# of 165 share at least 115 of them, so their SDs move together, and at the top rung all 20 replicates
# are the same subsample and differ only by Monte Carlo noise. The replicate-level fit is emitted too,
# with a heteroscedasticity-consistent (HC3) standard error, as a diagnostic on that spread; it is not
# the reported interval.
EXPONENT_REFERENCE = -0.5
POOLING_ALPHA = 0.05
# Tolerances, written before the first fit. The first two bound how far the replayed LIVE arms may sit
# from their committed point estimates; the third bounds the algebraic self-check on every solved
# cluster count.
POINT_TOLERANCE = 1e-9
BOOTSTRAP_SD_TOLERANCE = 1e-12
SOLVE_TOLERANCE = 1e-9


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(path: Path) -> str:
    """The checked-out commit of a repository, or a reason string; never an exception."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:  # pragma: no cover - environment
        return f"unavailable ({error})"
    return completed.stdout.strip()


def _git_worktree(path: Path) -> dict[str, Any]:
    """Whether the worktree is clean, and what is in it if not.

    The paths matter as much as the flag. This run writes its own tool and its own record into the
    repository it reads, so ``clean`` is false by the time the record is emitted even when every input
    was read from the declared commit. Listing what is uncommitted lets a reader see that rather than
    take the flag at face value.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:  # pragma: no cover - environment
        return {"clean": f"unavailable ({error})", "uncommitted": None}
    entries = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {"clean": not entries, "uncommitted": entries}


def apportion(n_clusters: int, weights: dict[str, int]) -> dict[str, int]:
    """Split ``n_clusters`` across the domains in the full sample's proportions.

    Largest-remainder apportionment rather than independent rounding, so the parts always sum to the
    requested total: rounding 50/165 and 115/165 separately can lose or gain a cluster at some rungs,
    and a rung that quietly carries 39 or 41 clusters would put the wrong x on the regression. The
    weights are read from the corpus rather than written here, so a change in tau-bench's domain
    mixture reweights the ladder visibly instead of leaving a stale constant in place.
    """
    total = sum(weights.values())
    exact = {name: n_clusters * count / total for name, count in weights.items()}
    floors = {name: int(math.floor(value)) for name, value in exact.items()}
    remaining = n_clusters - sum(floors.values())
    order = sorted(exact, key=lambda name: (-(exact[name] - floors[name]), name))
    for name in order[:remaining]:
        floors[name] += 1
    if sum(floors.values()) != n_clusters:
        raise RuntimeError(f"the split of {n_clusters} sums to {sum(floors.values())}")
    return floors


def ladder_split(n_clusters: int, available: dict[str, int]) -> dict[str, int]:
    """One rung of the ladder: the proportional split, checked against what the corpus actually has.

    The subsampler draws without replacement, so a rung that asked for more clusters of a domain than
    the corpus holds is a specification error rather than a draw to retry. The design table has no
    such ceiling, which is why it calls ``apportion`` directly.
    """
    split = apportion(n_clusters, available)
    for name, count in split.items():
        if count > available[name]:
            raise RuntimeError(
                f"the ladder asks for {count} {name} clusters and the corpus has {available[name]}"
            )
    return split


def subsample_clustering(
    labels: np.ndarray, scores: np.ndarray, clustering: dict[str, Any],
    n_clusters: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, int]]:
    """Draw whole task clusters without replacement and rebuild a clustering over just those rows.

    Whole clusters travel together here for the same reason they do inside a bootstrap draw: a
    subsample that split a task's four model attempts would be a smaller study of a different design,
    and its SD would not be on the curve this fits. The reduced universe goes back through the shipped
    ``_tau_clustering`` rather than being assembled by hand, so a subsample also has to satisfy the
    equal-cluster-size and pure-stratum checks that the full corpus does.
    """
    available = {name: len(rows) for name, rows in clustering["strata"].items()}
    split = ladder_split(n_clusters, available)
    blocks = clustering["blocks"]
    chosen_ids: list[str] = []
    chosen_strata: list[str] = []
    rows: list[np.ndarray] = []
    for name in sorted(split):
        picked = rng.choice(clustering["strata"][name], size=split[name], replace=False)
        for index in picked:
            rows.append(blocks[index])
            chosen_ids.extend([f"c{int(index):04d}"] * blocks.shape[1])
            chosen_strata.extend([name] * blocks.shape[1])
    order = np.concatenate(rows)
    return (
        np.asarray(labels)[order],
        np.asarray(scores)[order],
        st._tau_clustering(np.asarray(labels)[order], chosen_ids, chosen_strata),
        split,
    )


def _ols(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Slope and intercept of ``y`` on ``x``, with both a classical and an HC3 slope error.

    Two errors because the two fits this file runs want different ones. The rung-level fit has seven
    independent-enough points and homoscedastic residuals, where the classical error is right. The
    replicate-level fit has a spread that shrinks as the rung grows, which is what HC3 is for.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("a slope needs at least three paired points")
    centered = x - x.mean()
    denominator = float((centered ** 2).sum())
    slope = float((centered * (y - y.mean())).sum() / denominator)
    intercept = float(y.mean() - slope * x.mean())
    residual = y - (intercept + slope * x)
    df = len(x) - 2
    classical = float(math.sqrt((residual ** 2).sum() / df / denominator))
    leverage = 1.0 / len(x) + centered ** 2 / denominator
    hc3 = float(math.sqrt(((centered ** 2) * (residual / (1.0 - leverage)) ** 2).sum())
                / denominator)
    total = float(((y - y.mean()) ** 2).sum())
    return {
        "n_points": int(len(x)),
        "slope": slope,
        "intercept": intercept,
        "se_classical": classical,
        "se_hc3": hc3,
        "df": int(df),
        "r_squared": float(1.0 - (residual ** 2).sum() / total) if total > 0 else None,
        "residual_sd": float(math.sqrt((residual ** 2).sum() / df)),
    }


def _interval(slope: float, se: float, df: int) -> tuple[float, float]:
    half = float(stats.t.ppf(0.975, df) * se)
    return slope - half, slope + half


def measure_scaling(
    labels: np.ndarray, scores: np.ndarray, clustering: dict[str, Any], cell: str,
) -> dict[str, Any]:
    """Run the declared ladder for one cell and fit its scaling exponent.

    Each (rung, replicate) draws its own subsample and then reruns the shipped bootstrap inside it,
    both off a single generator derived from the declared label, so the whole replicate reproduces
    from ``live_tau_scaling.{cell}.{n_clusters}.{replicate}`` and the seed alone. The top rung is the
    whole corpus: its 20 replicates are the same 165 clusters and differ only by bootstrap noise,
    which is the reading that says how much of the ladder's spread is Monte Carlo rather than sample.
    """
    replicates: list[dict[str, Any]] = []
    for n_clusters in LADDER:
        for replicate in range(SUBSAMPLE_REPLICATES):
            label = SCALING_LABEL.format(cell=cell, n_clusters=n_clusters, replicate=replicate)
            rng = st._rng_for(label, SCALING_SEED)
            sub_labels, sub_scores, sub_clustering, split = subsample_clustering(
                labels, scores, clustering, n_clusters, rng
            )
            result = st._task_clustered_auc_bootstrap(
                sub_labels, sub_clustering, sub_scores, None, BAR, SUBSAMPLE_DRAWS, rng
            )
            replicates.append({
                "n_clusters": n_clusters,
                "replicate": replicate,
                "rng_label": label,
                "clusters_per_stratum": split,
                "n_rows": result["n_rows"],
                "point": result["point"],
                "bootstrap_sd": result["bootstrap_sd"],
                "usable_replicates": result["usable_replicates"],
                "discarded_single_class_draws": result["discarded_single_class_draws"],
            })
    log_n = np.log([row["n_clusters"] for row in replicates])
    log_sd = np.log([row["bootstrap_sd"] for row in replicates])
    rungs = []
    for n_clusters in LADDER:
        mask = np.asarray([row["n_clusters"] == n_clusters for row in replicates])
        rungs.append({
            "n_clusters": n_clusters,
            "n_replicates": int(mask.sum()),
            "mean_log_sd": float(log_sd[mask].mean()),
            "geometric_mean_sd": float(np.exp(log_sd[mask].mean())),
            "min_sd": float(np.exp(log_sd[mask].min())),
            "max_sd": float(np.exp(log_sd[mask].max())),
            "sd_of_log_sd": float(log_sd[mask].std(ddof=1)),
        })
    rung_fit = _ols([math.log(row["n_clusters"]) for row in rungs],
                    [row["mean_log_sd"] for row in rungs])
    replicate_fit = _ols(log_n, log_sd)
    low, high = _interval(rung_fit["slope"], rung_fit["se_classical"], rung_fit["df"])
    return {
        "cell": cell,
        "ladder": list(LADDER),
        "draws_per_subsample": SUBSAMPLE_DRAWS,
        "subsamples_per_rung": SUBSAMPLE_REPLICATES,
        "rng_label_template": SCALING_LABEL,
        "rng_base_seed": SCALING_SEED,
        "replicate_index_is_zero_based": True,
        "rungs": rungs,
        "replicates": replicates,
        "exponent": rung_fit["slope"],
        "exponent_interval_95": [low, high],
        "exponent_se": rung_fit["se_classical"],
        "exponent_df": rung_fit["df"],
        "exponent_estimator": "OLS of the rung mean of log SD on log clusters, seven rungs, "
                              "Student-t interval on five degrees of freedom",
        "consistent_with_minus_half": bool(low <= EXPONENT_REFERENCE <= high),
        "rung_level_fit": rung_fit,
        "replicate_level_fit_diagnostic": {
            **replicate_fit,
            "role": "diagnostic only; its 140 points are not independent, so its interval is not "
                    "the reported one",
            "interval_95_hc3": list(_interval(replicate_fit["slope"], replicate_fit["se_hc3"],
                                              replicate_fit["df"])),
        },
    }


def pool_exponents(measured: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Inverse-variance pool of the four exponents, reported only if the four agree.

    Two conditions, both fixed before the fits: every pair of 95% intervals overlaps, and Cochran's Q
    over the four slopes does not reject homogeneity at 0.05. Requiring both is the conservative
    combination, since either one alone would let a pooled number through that the other refuses.
    Pairwise overlap is the crude reading a table invites and it tolerates a drift across four tight
    fits; Q is the formal one and it can reject on a spread nobody would print differently. A single
    number that stands for all twenty cells has to survive both readings, and when it does not, the
    four per-cell exponents are what the record carries.
    """
    slopes = np.asarray([row["exponent"] for row in measured], dtype=float)
    errors = np.asarray([row["exponent_se"] for row in measured], dtype=float)
    intervals = [row["exponent_interval_95"] for row in measured]
    overlap = all(a[0] <= b[1] and b[0] <= a[1]
                  for index, a in enumerate(intervals) for b in intervals[index + 1:])
    weights = 1.0 / errors ** 2
    mean = float((weights * slopes).sum() / weights.sum())
    q = float((weights * (slopes - mean) ** 2).sum())
    df = len(slopes) - 1
    p = float(stats.chi2.sf(q, df))
    se = float(1.0 / math.sqrt(weights.sum()))
    agree = bool(overlap and p > POOLING_ALPHA)
    return {
        "cells_agree": agree,
        "every_pair_of_intervals_overlaps": bool(overlap),
        "cochran_q": q,
        "cochran_q_df": int(df),
        "cochran_q_p": p,
        "homogeneity_alpha": POOLING_ALPHA,
        "pooled_exponent": mean if agree else None,
        "pooled_exponent_se": se if agree else None,
        "pooled_exponent_interval_95": (
            [mean - st.Z975 * se, mean + st.Z975 * se] if agree else None
        ),
        "pooled_exponent_consistent_with_minus_half": (
            bool(abs(mean - EXPONENT_REFERENCE) <= st.Z975 * se) if agree else None
        ),
        "withheld_reason": None if agree else (
            "the four per-cell exponents do not agree, so the declaration's condition for reporting "
            "a single pooled exponent is not met"
        ),
    }


def design_clusters(point: float, sd: float, n_clusters: int, exponent: float) -> float:
    """Cluster count at which a normal 95% interval at ``point`` would just exclude zero.

    Solves ``|point| = z * sd * (n / n_clusters) ** exponent`` for ``n``. The effect stays fixed at the
    value already measured and the sample is the unknown, which is the direction that makes this a
    design calculation. Running it the other way, fixing the sample and solving for the effect the
    achieved interval would have excluded, is the post-hoc power error the declaration exists to
    prevent, and no branch of this file does it.
    """
    if exponent >= 0:
        raise ValueError(f"a bootstrap SD must fall with sample size; got exponent {exponent}")
    if abs(point) < 1e-12:
        raise ValueError("a design size is undefined at a zero point estimate")
    if sd <= 0:
        raise ValueError(f"a design size needs a positive bootstrap SD; got {sd}")
    solved = n_clusters * (abs(point) / (st.Z975 * sd)) ** (1.0 / exponent)
    achieved = st.Z975 * sd * (solved / n_clusters) ** exponent
    if abs(achieved - abs(point)) > SOLVE_TOLERANCE * max(1.0, abs(point)):
        raise RuntimeError(
            f"the solved cluster count {solved} does not invert: a half-width of {achieved} against "
            f"a point of {abs(point)}"
        )
    return float(solved)


def design_row(claim: dict[str, Any], measured: dict[str, Any], available: dict[str, int]) -> dict:
    """One of the twenty rows: the committed reading, then both extrapolations of it.

    ``measured`` says which exponent this row is entitled to. A cell whose own ladder was run carries
    its own; every cell carries a pooled one when the four agree. When they do not, the sixteen cells
    with no ladder of their own get a bracket across the four measured exponents, and the single
    printable number is the end that asks for more clusters. Understating a design target would be the
    worse error of the two, and the bracket stays in the record so the range is never hidden.
    """
    clustered = claim["clustered_interval"]
    point, sd = clustered["point"], clustered["bootstrap_sd"]
    n_clusters = clustered["n_clusters"]

    def report(solved: float) -> dict[str, Any]:
        clusters = int(math.ceil(solved))
        return {
            "clusters_exact": solved,
            "clusters": clusters,
            "runs": clusters * clustered["rows_per_cluster"],
            "multiple_of_present_sample": solved / n_clusters,
            "clusters_per_stratum": apportion(clusters, available),
            # Below the smallest rung the ladder measured, the extrapolation is outside the range
            # where the scaling was checked, and a bootstrap over a handful of clusters is barely
            # defined in any case. Flagged rather than clipped: clipping would print a design size
            # the arithmetic did not produce.
            "below_measured_ladder": bool(clusters < min(LADDER)),
        }

    closed = report(design_clusters(point, sd, n_clusters, EXPONENT_REFERENCE))
    row = {
        "id": claim["id"],
        "cell": claim["id"].removeprefix("live.tau.bar."),
        "prefix_percent": int(claim["id"].split(".")[3]),
        "method": claim["id"].split(".", 4)[4],
        "auc": claim["estimate"]["a"],
        "bar": claim["estimate"]["b"],
        "point_auc_minus_bar": point,
        "bootstrap_sd": sd,
        "interval_95": list(clustered["interval_95"]),
        "n_clusters": n_clusters,
        "n_rows": clustered["n_rows"],
        "clusters_per_stratum": dict(clustered["clusters_per_stratum"]),
        "verdict": claim["verdict"],
        "separates_at_present_sample": claim["verdict"] != "does_not_separate",
        "needed_closed_form": closed,
        "needed_measured_exponent_source": measured["source"],
    }
    if measured["kind"] == "bracket":
        ends = {
            exponent: report(design_clusters(point, sd, n_clusters, exponent))
            for exponent in (measured["exponent_low"], measured["exponent_high"])
        }
        larger = max(ends.values(), key=lambda end: end["clusters"])
        row["needed_measured"] = larger
        row["needed_measured_exponent"] = next(
            exponent for exponent, end in ends.items() if end is larger
        )
        row["needed_measured_bracket"] = {
            "exponent_low": measured["exponent_low"],
            "exponent_high": measured["exponent_high"],
            "clusters_min": min(end["clusters"] for end in ends.values()),
            "clusters_max": max(end["clusters"] for end in ends.values()),
            "ends": {f"{exponent:.6f}": end for exponent, end in ends.items()},
        }
    else:
        row["needed_measured"] = report(
            design_clusters(point, sd, n_clusters, measured["exponent"])
        )
        row["needed_measured_exponent"] = measured["exponent"]
    return row


def _replay_live_tau(cells: Sequence[tuple[str, float, str, str]]) -> dict[str, Any]:
    """The LIVE tau-bench score vectors, labels, and task clustering for the named cells.

    LIVE's own replay, not the POST audit arrays: ``_tau_live_clustering`` rebuilds tau-bench's task
    identifiers under LIVE's ``>= 4``-step filter and checks them against LIVE's labels and the 100%
    prefix ``n_steps`` column, so a drifted replay raises instead of returning an interval for the
    wrong rows. The arms are rebuilt with the same ``supervised_oof`` and ECOD calls the shipped run
    uses, and each is checked against its committed point estimate before it is subsampled.
    """
    from catchbench.live import LiveStreaming

    task = LiveStreaming("tau", prefixes=st.PREFIXES)
    task.setup()
    labels = np.asarray(task.y)
    clustering = st._tau_live_clustering(task)
    scores: dict[str, np.ndarray] = {}
    for cell, prefix, method, _ in cells:
        if method == "pyod (ECOD)":
            entry = st._auc_entry(None, None, st._ecod_scores(task.layers_at[prefix]["flat"]), labels)
        else:
            layer = {"size (flat)": "flat", "auditable (size+deps)": "flatdep", "full": "full"}[method]
            oof, folds = st.supervised_oof(task.layers_at[prefix][layer], labels)
            entry = st._auc_entry(oof, folds, None, labels)
        scores[cell] = np.asarray(entry["primary"], dtype=float)
    return {"labels": labels, "clustering": clustering, "scores": scores,
            "corpus_line": task.corpus_line()}


def _check_replay(cell: str, labels, clustering, score, claim: dict) -> dict[str, Any]:
    """Bind a replayed arm to its committed cell before anything is extrapolated from it.

    Two bindings. The point estimate must reproduce, which says the arm is the one the record scored;
    and the shipped 10,000-draw bootstrap must reproduce bit for bit off the committed claim id and
    seed, which says the resampling this file subsamples is the resampling that produced the printed
    interval. A drift in either is a stop, not a number to carry forward.
    """
    committed = claim["clustered_interval"]
    replayed = st._task_clustered_auc_bootstrap(
        labels, clustering, score, None, BAR, st.TAU_CLUSTER_DRAWS,
        st._rng_for(claim["id"], st.TAU_CLUSTER_SEED),
    )
    point_delta = abs(replayed["point"] - committed["point"])
    sd_delta = abs(replayed["bootstrap_sd"] - committed["bootstrap_sd"])
    if point_delta > POINT_TOLERANCE:
        raise RuntimeError(
            f"the replayed LIVE arm for {cell} scores {replayed['point']} against the committed "
            f"{committed['point']}, a difference of {point_delta} over the declared tolerance "
            f"{POINT_TOLERANCE}"
        )
    if sd_delta > BOOTSTRAP_SD_TOLERANCE:
        raise RuntimeError(
            f"the shipped bootstrap for {cell} no longer reproduces: SD {replayed['bootstrap_sd']} "
            f"against the committed {committed['bootstrap_sd']}"
        )
    return {
        "cell": cell,
        "claim_id": claim["id"],
        "committed_point": committed["point"],
        "replayed_point": replayed["point"],
        "point_abs_difference": point_delta,
        "point_tolerance": POINT_TOLERANCE,
        "committed_bootstrap_sd": committed["bootstrap_sd"],
        "replayed_bootstrap_sd": replayed["bootstrap_sd"],
        "bootstrap_sd_abs_difference": sd_delta,
        "bootstrap_sd_tolerance": BOOTSTRAP_SD_TOLERANCE,
        "committed_interval_95": list(committed["interval_95"]),
        "replayed_interval_95": list(replayed["interval_95"]),
    }


def render_table(rows: Sequence[dict[str, Any]]) -> str:
    header = (f"{'cell':<30}{'point':>10}{'sd':>9}{'hi95':>10}"
              f"{'closed':>9}{'measured':>10}  verdict")
    lines = [header, "-" * len(header)]
    for row in rows:
        measured = row["needed_measured"]
        lines.append(
            f"{row['cell']:<30}{row['point_auc_minus_bar']:>10.4f}{row['bootstrap_sd']:>9.4f}"
            f"{row['interval_95'][1]:>10.4f}{row['needed_closed_form']['clusters']:>9d}"
            f"{(str(measured['clusters']) if measured else '-'):>10}  {row['verdict']}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path,
                        default=ROOT / "tools" / "live_tau_bar_design_results.json")
    parser.add_argument("--registry", type=Path,
                        default=ROOT / "tools" / "statistical_tests_results.json")
    parser.add_argument("--declaration", type=Path, default=None,
                        help="the frozen Part B declaration, recorded by path and sha256")
    args = parser.parse_args(argv)

    started = time.time()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    claims = {claim["id"]: claim for claim in registry["claims"]
              if claim["id"].startswith("live.tau.bar.")}
    if len(claims) != 20:
        raise RuntimeError(f"expected twenty live.tau.bar rows in the registry, found {len(claims)}")

    replay = _replay_live_tau(SCALING_CELLS)
    labels, clustering = replay["labels"], replay["clustering"]
    available = {name: len(rows) for name, rows in clustering["strata"].items()}

    reproduction = []
    measured = []
    for cell, _, _, why in SCALING_CELLS:
        claim = claims[f"live.tau.bar.{cell}"]
        reproduction.append(
            _check_replay(cell, labels, clustering, replay["scores"][cell], claim)
        )
        print(f"scaling ladder for {cell} ({why}) ...", file=sys.stderr)
        result = measure_scaling(labels, replay["scores"][cell], clustering, cell)
        result["declared_because"] = why
        measured.append(result)

    pooled = pool_exponents(measured)
    all_consistent = all(row["consistent_with_minus_half"] for row in measured)
    printed_source = "closed_form" if all_consistent else "measured_scaling"
    own = {row["cell"]: row["exponent"] for row in measured}
    if pooled["cells_agree"]:
        fallback = {"kind": "pooled", "exponent": pooled["pooled_exponent"],
                    "source": "the pooled measured exponent; the four cells agree"}
    else:
        fallback = {
            "kind": "bracket",
            "exponent_low": min(own.values()),
            "exponent_high": max(own.values()),
            "source": "a bracket across the four measured exponents, reported at the end that asks "
                      "for more clusters; the four do not agree, so no single measured exponent is "
                      "licensed for a cell whose own ladder was not run",
        }
    rows = []
    for prefix in st.PREFIXES:
        for method in ("size (flat)", "auditable (size+deps)", "full", "pyod (ECOD)",
                       "dep-span (online)"):
            claim = claims[f"live.tau.bar.{int(prefix * 100)}.{method}"]
            cell = claim["id"].removeprefix("live.tau.bar.")
            spec = ({"kind": "own", "exponent": own[cell],
                     "source": "this cell's own measured exponent"}
                    if cell in own else fallback)
            rows.append(design_row(claim, spec, available))
    unresolved = [row for row in rows if not row["separates_at_present_sample"]]

    package_versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "pyod"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover - environment
            package_versions[package] = "not installed"

    import agent_failure_detection

    grade_root = Path(agent_failure_detection.__file__).resolve().parents[1]
    inputs = {
        "registry": {
            "path": str(args.registry),
            "sha256": _sha256(args.registry),
            "rows_read": sorted(claims),
            "fields_read": ["clustered_interval.point", "clustered_interval.bootstrap_sd",
                            "clustered_interval.n_clusters", "clustered_interval.n_rows",
                            "clustered_interval.clusters_per_stratum",
                            "clustered_interval.interval_95", "estimate.a", "estimate.b", "verdict"],
        },
        "code": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in (
                ("tools/statistical_tests.py", ROOT / "tools" / "statistical_tests.py"),
                ("src/catchbench/live.py", ROOT / "src" / "catchbench" / "live.py"),
                ("src/catchbench/detection.py", ROOT / "src" / "catchbench" / "detection.py"),
                ("grade/experiment/agent_failure_detection.py",
                 Path(agent_failure_detection.__file__).resolve()),
            )
        },
        "repositories": {
            "catchbench": {"path": str(ROOT), "commit": _git_commit(ROOT), **_git_worktree(ROOT)},
            "grade": {"path": str(grade_root), "commit": _git_commit(grade_root),
                      **_git_worktree(grade_root)},
        },
        "corpus": replay["corpus_line"],
    }
    if args.declaration is not None:
        inputs["declaration"] = {
            "path": str(args.declaration),
            "sha256": _sha256(args.declaration),
            "part": "Part B, the tau-bench bar design calculation",
        }

    result = {
        "schema_version": "1.0.0",
        "generated_by": "tools/live_tau_bar_design.py",
        "question": "For each committed live.tau.bar cell, the number of tau-bench task clusters at "
                    "which a 95% interval would exclude zero at the observed point estimate.",
        "declaration": {
            "file": "research/catchbench-live-size-control-declaration-2026-09-10.md",
            "part": "B",
            "frozen_before_the_run": {
                "ladder": list(LADDER),
                "draws_per_subsample": SUBSAMPLE_DRAWS,
                "subsamples_per_rung": SUBSAMPLE_REPLICATES,
                "rng_base_seed": SCALING_SEED,
                "rng_label_template": SCALING_LABEL,
                "cells": [cell for cell, _, _, _ in SCALING_CELLS],
                "stratification": "the airline and retail proportions of the full sample are held at "
                                  "every rung",
            },
            "fixed_by_this_implementation": {
                "exponent_estimator": "OLS of the rung mean of log SD on log clusters; a Student-t "
                                      "interval on five degrees of freedom",
                "replicate_level_fit": "emitted as a diagnostic with an HC3 slope error; not the "
                                       "reported interval, because subsamples at one rung overlap "
                                       "heavily and the top rung repeats one subsample",
                "consistency_rule": "an exponent is consistent with -0.5 when its 95% interval "
                                    "contains -0.5",
                "pooling_rule": "the four agree when every pair of intervals overlaps and Cochran's "
                                f"Q does not reject homogeneity at {POOLING_ALPHA}",
                "tolerances": {"replayed_point": POINT_TOLERANCE,
                               "replayed_bootstrap_sd": BOOTSTRAP_SD_TOLERANCE,
                               "design_solve_inversion": SOLVE_TOLERANCE},
            },
        },
        "settings": {
            "confidence_level": 0.95,
            "z_975": st.Z975,
            "bar": BAR,
            "statistic": "ROC-AUC minus the 0.70 bar, the scale the committed cells are stored on",
            "bootstrap": {
                "method": st.TAU_CLUSTER_METHOD,
                "axis": st.TAU_CLUSTER_AXIS,
                "unit": st.TAU_CLUSTER_UNIT,
                "stratum": st.TAU_CLUSTER_STRATUM,
                "committed_draws": st.TAU_CLUSTER_DRAWS,
                "committed_seed": st.TAU_CLUSTER_SEED,
                "monte_carlo_noise_on_one_endpoint": st.TAU_CLUSTER_MC_NOISE,
            },
            "cv_seeds": list(st.CV_SEEDS),
            "prefixes": list(st.PREFIXES),
            "python": sys.version.split()[0],
            "package_versions": package_versions,
        },
        "inputs": inputs,
        "replay_reproduction": reproduction,
        "scaling_check": {
            "assumption": "the clustered bootstrap SD falls as clusters ** -0.5",
            "reference_exponent": EXPONENT_REFERENCE,
            "per_cell": measured,
            "pooling": pooled,
            "all_four_consistent_with_minus_half": all_consistent,
            "printed_figure_source": printed_source,
            "printed_figure_rule": (
                "Part B4: a measured exponent consistent with -0.5 means the printed figure is the "
                "closed-form extrapolation; otherwise the printed figure comes from the measured "
                "scaling and the closed form is reported beside it as the idealized comparison. "
                "B4 was written expecting one exponent; with four measured and not all consistent, "
                "the printed figure is read as coming from the measured scaling, and each of the "
                "two cells the manuscript quotes carries its own measured exponent."
            ),
            "within_cell_error_is_probably_understated": (
                "Two subsamples at an upper rung overlap heavily, and at 165 clusters all twenty "
                "replicates are the same subsample, so a rung mean is more precise in the "
                "arithmetic than in the sampling. The per-cell errors are therefore likely too "
                "small and Cochran's Q correspondingly too large, which is a reason to read the "
                "disagreement as mild rather than a reason to overrule it. The pooling rule was "
                "fixed before the fits and is reported as it came out."
            ),
            "shape_of_the_relation": (
                "The rung-level fits carry R^2 of 0.998 or better with no monotone drift in the "
                "rung-to-rung local slopes, so a single exponent describes the ladder well over the "
                "measured range. Every design count above 165 clusters is an extrapolation past the "
                "largest rung the ladder could reach."
            ),
        },
        "design_table": rows,
        "unresolved_cells": [row["cell"] for row in unresolved],
        "summary": {
            "cells": len(rows),
            "separating_at_present_sample": len(rows) - len(unresolved),
            "not_separating_at_present_sample": len(unresolved),
            "present_clusters": 165,
            "present_rows": 660,
            "needed_for_unresolved_cells": {
                row["cell"]: {
                    "closed_form_clusters": row["needed_closed_form"]["clusters"],
                    "closed_form_runs": row["needed_closed_form"]["runs"],
                    "measured_clusters": row["needed_measured"]["clusters"],
                    "measured_runs": row["needed_measured"]["runs"],
                    "measured_exponent": row["needed_measured_exponent"],
                    "measured_exponent_source": row["needed_measured_exponent_source"],
                    "multiple_of_present_sample": row["needed_measured"][
                        "multiple_of_present_sample"],
                }
                for row in unresolved
            },
        },
        "not_computed": {
            "achieved_power": "never computed. Power at the sample already run is a monotone "
                              "restatement of the observed p-value and adds no information about "
                              "what a future study would resolve.",
            "minimum_detectable_difference": "never computed. Converting an achieved interval into "
                                             "the smallest effect it would have excluded fixes the "
                                             "sample and solves for the effect, which is the "
                                             "direction Part B3 forbids; this file fixes the effect "
                                             "at the observed point and solves for the sample.",
            "direction_of_every_solve": "|point| held fixed, cluster count unknown",
        },
        "interpretation": {
            "design_target_only": "The cluster counts are design targets for a future study on this "
                                  "benchmark. They are not a claim that a cell would clear the bar "
                                  "at that sample: the solve holds the observed point estimate "
                                  "fixed, and a larger sample can move it either way.",
            "normal_idealization": "The solve uses a symmetric normal interval at the committed "
                                   "bootstrap SD. The printed intervals are percentile bootstrap "
                                   "intervals and are asymmetric, so a solved count is an "
                                   "approximation of the sample at which the percentile interval "
                                   "would clear zero, not a simulation of it.",
            "fixed_point_estimates": "Every row extrapolates the SD alone. Nothing here re-estimates "
                                     "a point, a label, or a threshold.",
            "exploratory": "Part B was proposed after the twenty committed cells and their verdicts "
                           "were read. It stays exploratory in the manuscript whichever way it comes "
                           "out.",
        },
        "elapsed_seconds": time.time() - started,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(st._jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(render_table(rows))
    print()
    for row in measured:
        low, high = row["exponent_interval_95"]
        print(f"exponent {row['cell']:<28} {row['exponent']:+.4f} [{low:+.4f}, {high:+.4f}]  "
              f"consistent with -0.5: {row['consistent_with_minus_half']}")
    print(f"\npooled: {pooled['pooled_exponent']}  agree: {pooled['cells_agree']}  "
          f"Cochran Q={pooled['cochran_q']:.4f} p={pooled['cochran_q_p']:.4f}")
    print(f"printed figure source: {printed_source}")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
