"""Render the two LIVE early-warning prefix curves for the README."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import board_data as bd
import _figure_base as fb


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "assets" / "board_live_prefix.png"

SERIES = {
    "full": (fb.ORANGE, "full features (supervised)", 3.2, "o"),
    "auditable (size+deps)": (fb.BLUE, "size + dependencies", 3.0, "s"),
    "pyod (ECOD)": (fb.GREEN, "ECOD (unsupervised)", 2.8, "^"),
    "size (flat)": (fb.PURPLE, "run size", 2.4, "D"),
    "dep-span (online)": ("#7D5A50", "dependency span (online)", 2.2, "v"),
    "random": ("#9AA3AA", "random", 2.0, "x"),
}


def render(board_path: Path, output: Path) -> None:
    payload = bd.figure_payload("board_live_prefix", board_path)
    prefixes = payload["prefixes"]
    threshold = payload["threshold"]
    corpora = payload["corpora"]

    for corpus, rows in corpora.items():
        if set(rows) != set(SERIES):
            missing = sorted(set(SERIES) - set(rows))
            extra = sorted(set(rows) - set(SERIES))
            raise bd.BoardDataError(
                f"LIVE {corpus} rows changed; missing={missing!r}, extra={extra!r}"
            )

    fb.style(matplotlib)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.5), sharex=True, sharey=True)
    all_values = [value for rows in corpora.values() for values in rows.values() for value in values]
    lower = max(0.0, min(all_values) - 0.07)
    upper = min(1.0, max(all_values) + 0.07)

    titles = {"swegym": "SWE-Gym", "tau": "tau-bench"}
    for ax, corpus in zip(axes, ("swegym", "tau")):
        rows = corpora[corpus]
        for method, (color, label, width, marker) in SERIES.items():
            ax.plot(prefixes, rows[method], color=color, linewidth=width, marker=marker,
                    markersize=7.5, markerfacecolor="white" if marker != "x" else color,
                    markeredgecolor=color, markeredgewidth=1.6, label=label, zorder=4)
        ax.axhline(threshold, color=fb.INK, linewidth=1.6, linestyle=(0, (4, 3)), zorder=2)
        ax.text(prefixes[-1], threshold + 0.012,
                f"early-warning threshold: {threshold:.2f}", fontsize=12.5, color=fb.INK,
                ha="right", va="bottom", bbox={"facecolor": fb.WHITE, "edgecolor": "none",
                                                "pad": 1.5})

        first_crossings = []
        for values in rows.values():
            crossing = next((prefix for prefix, value in zip(prefixes, values)
                             if value >= threshold), None)
            if crossing is not None:
                first_crossings.append(crossing)
        message = (f"First crossing at {min(first_crossings)}% of the trace"
                   if first_crossings else "No method reaches the threshold")
        ax.text(0.99, 0.07, message, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=13.0, fontweight="bold",
                color=fb.GREEN if first_crossings else fb.ORANGE)

        ax.set_title(titles[corpus], fontsize=18, fontweight="bold", loc="left", pad=9)
        ax.set_ylabel("Failure ROC-AUC", fontsize=15)
        ax.set_ylim(lower, upper)
        ax.yaxis.grid(True, color=fb.GRID, linewidth=1.0)
        ax.tick_params(labelsize=13)
        fb.clean_axes(ax)

    axes[-1].set_xlabel("Observed trace prefix (%)", fontsize=15, labelpad=8)
    axes[-1].set_xticks(prefixes)
    axes[-1].set_xticklabels([str(prefix) for prefix in prefixes])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3,
               frameon=False, fontsize=11.5, columnspacing=1.25, handlelength=1.8)
    fig.suptitle("LIVE: how early is the failure signal visible?", fontsize=20,
                 fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.78, bottom=0.09, left=0.14, right=0.97, hspace=0.35)
    fb.save_web_png(fig, output, "board_live_prefix", payload)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", type=Path, default=bd.DEFAULT_BOARD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    render(args.board, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
