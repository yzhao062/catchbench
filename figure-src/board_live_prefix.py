"""Render the two LIVE early-warning prefix panels for the README."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import board_data as bd
import _figure_base as fb


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "assets" / "board_live_prefix.png"

SERIES = {
    "full": (fb.AB_CORAL, "full features (supervised)", 3.2, "o", "-"),
    "auditable (size+deps)": (fb.AB_NEAR_BLACK, "size + dependencies", 3.0, "s", "-"),
    "pyod (ECOD)": (fb.AB_SUBTITLE, "ECOD (unsupervised)", 2.8, "^", "-."),
    "size (flat)": (fb.AB_GRAY, "size + counts", 2.6, "D", "--"),
    "dep-span (online)": (fb.AB_NEAR_BLACK, "dependency span (online)", 2.4, "v", ":"),
    "random": (fb.AB_GRAY, "random", 2.0, "x", ":"),
}


def render(board_path: Path, stats_path: Path, output: Path) -> None:
    payload = bd.figure_payload("board_live_prefix", board_path, stats_path)
    prefixes = payload["prefixes"]
    threshold = payload["threshold"]
    corpora = payload["corpora"]
    threshold_claims = payload["registered_threshold_claims"]

    for corpus, rows in corpora.items():
        if set(rows) != set(SERIES):
            missing = sorted(set(SERIES) - set(rows))
            extra = sorted(set(rows) - set(SERIES))
            raise bd.BoardDataError(
                f"LIVE {corpus} rows changed; missing={missing!r}, extra={extra!r}"
            )

    fb.style(matplotlib)
    matplotlib.rcParams.update({
        "text.color": fb.AB_NEAR_BLACK,
        "axes.labelcolor": fb.AB_NEAR_BLACK,
        "axes.edgecolor": fb.AB_SUBTITLE,
        "xtick.color": fb.AB_NEAR_BLACK,
        "ytick.color": fb.AB_NEAR_BLACK,
    })
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 8.1), sharex=True, sharey=True)
    all_values = [value for rows in corpora.values() for values in rows.values() for value in values]
    lower = max(0.0, min(all_values) - 0.07)
    upper = min(1.0, max(all_values) + 0.07)

    titles = {"swegym": "SWE-Gym", "tau": "tau-bench"}
    subtitles = {}
    for corpus in ("swegym", "tau"):
        registered = [claim for cells in threshold_claims[corpus].values()
                      for claim in cells if claim is not None]
        separating = [claim for claim in registered if claim["verdict"] == "separates"]
        # Read the side from the claims. Both families are two-sided, so SWE-Gym separates in
        # both directions and a hardcoded word would print the wrong one for three of its cells.
        above = sum(claim["side"] == "above" for claim in separating)
        below = len(separating) - above
        parts = ([f"{above} above 0.70"] if above else []) + ([f"{below} below"] if below else [])
        summary = ", ".join(parts) if parts else "none separate"
        # The SWE-Gym family was added after its scores were examined, so the panel says so where
        # the counts are read. The legend's "registered, separates" is about the marker fill.
        status = "exploratory; " if corpus == "swegym" else ""
        subtitles[corpus] = f"{status}{len(registered)} registered cells; {summary}"
    for ax, corpus in zip(axes, ("swegym", "tau")):
        rows = corpora[corpus]
        for method, (color, label, width, marker, linestyle) in SERIES.items():
            values = rows[method]
            claims = threshold_claims[corpus][method]
            faces = [
                "white" if claim is None else
                color if claim["verdict"] == "separates" else fb.AB_MINT
                for claim in claims
            ]
            ax.plot(prefixes, values, color=color, linewidth=width, linestyle=linestyle,
                    label=label, zorder=4)
            if marker == "x":
                ax.scatter(prefixes, values, color=color, marker=marker, s=70,
                           linewidths=1.8, zorder=5)
            else:
                edge = fb.AB_SUBTITLE if color in (fb.AB_GRAY, fb.AB_MINT) else color
                ax.scatter(prefixes, values, facecolors=faces, edgecolors=edge, marker=marker,
                           s=72, linewidths=1.7, zorder=5)
        ax.axhline(threshold, color=fb.AB_NEAR_BLACK, linewidth=1.6,
                   linestyle=(0, (4, 3)), zorder=2)
        ax.text(prefixes[-1], threshold + 0.012,
                f"early-warning threshold: {threshold:.2f}", fontsize=12.5,
                color=fb.AB_NEAR_BLACK,
                ha="right", va="bottom", bbox={"facecolor": fb.WHITE, "edgecolor": "none",
                                                "pad": 1.5})

        ax.set_title(titles[corpus], fontsize=18, fontweight="bold", loc="left", pad=24)
        ax.text(0.0, 1.015, subtitles[corpus], transform=ax.transAxes, ha="left", va="bottom",
                fontsize=11.5, color=fb.AB_SUBTITLE)
        ax.set_ylabel("Failure ROC-AUC", fontsize=15)
        ax.set_ylim(lower, upper)
        ax.yaxis.grid(True, color=fb.AB_GRAY, linewidth=1.0)
        ax.tick_params(labelsize=13)
        fb.clean_axes(ax)
        ax.spines["left"].set_color(fb.AB_SUBTITLE)
        ax.spines["bottom"].set_color(fb.AB_SUBTITLE)

    axes[-1].set_xlabel("Observed trace prefix (%)", fontsize=15, labelpad=8)
    axes[-1].set_xticks(prefixes)
    axes[-1].set_xticklabels([str(prefix) for prefix in prefixes])
    series_handles = [
        Line2D([], [], color=color, linewidth=width, linestyle=linestyle, marker=marker,
               markersize=7.0, markerfacecolor="white",
               markeredgecolor=fb.AB_SUBTITLE if color in (fb.AB_GRAY, fb.AB_MINT) else color,
               markeredgewidth=1.4, label=label)
        for color, label, width, marker, linestyle in SERIES.values()
    ]
    fig.legend(handles=series_handles, loc="upper center", bbox_to_anchor=(0.5, 0.94),
               ncol=3, frameon=False, fontsize=11.0, columnspacing=1.2, handlelength=2.0)
    evidence_handles = [
        Line2D([], [], color="none", marker="o", markersize=7.0,
               markerfacecolor=fb.AB_NEAR_BLACK, markeredgecolor=fb.AB_NEAR_BLACK,
               label="method-color fill: registered, separates"),
        Line2D([], [], color="none", marker="o", markersize=7.0,
               markerfacecolor=fb.AB_MINT, markeredgecolor=fb.AB_NEAR_BLACK,
               label="mint fill: registered, unresolved"),
        Line2D([], [], color="none", marker="o", markersize=7.0,
               markerfacecolor="white", markeredgecolor=fb.AB_NEAR_BLACK,
               label="hollow: point estimate, no bar test"),
    ]
    fig.legend(handles=evidence_handles, title="Marker fill against the 0.70 bar",
               loc="upper center", bbox_to_anchor=(0.5, 0.85), ncol=3, frameon=False,
               fontsize=9.6, title_fontsize=10.5, columnspacing=1.2, handletextpad=0.4)
    fig.suptitle("LIVE: how early is the failure signal visible?", fontsize=20,
                 fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.67, bottom=0.09, left=0.14, right=0.97, hspace=0.46)
    fb.save_web_png(fig, output, "board_live_prefix", payload)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", type=Path, default=bd.DEFAULT_BOARD)
    parser.add_argument("--stats", type=Path, default=bd.DEFAULT_STATS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    render(args.board, args.stats, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
