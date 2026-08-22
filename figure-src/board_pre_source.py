"""Render per-source PRE F1 against each source's flag-everything floor."""

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
DEFAULT_OUTPUT = ROOT / "assets" / "board_pre_source.png"

REFERENCES = {"flag_all", "flag_none", "oracle_privilege_diff"}
SHORT = {
    "flag_risky_perms": "risky-permission scan",
    "owasp_excess_permissions": "excess-permissions rule",
    "owasp_excess_functionality": "excess-functionality rule",
    "owasp_privilege_escalation": "privilege-escalation rule",
    "unrequested_high_impact": "high-impact rule",
    "sensitive_access": "sensitive-access rule",
    "owasp_asi_combined": "combined scanner",
    "llm_judge_needed(llama-3.3-70b)": "held-out LLM judge",
}


def render(board_path: Path, stats_path: Path, output: Path) -> None:
    payload = bd.figure_payload("board_pre_source", board_path, stats_path)
    columns = payload["columns"]
    rows = payload["rows"]
    verdicts = payload["registered_verdicts"]
    sources = [column for column in columns if column != "overall"]

    if "flag_all" not in rows:
        raise bd.BoardDataError("the PRE source table has no flag_all floor")
    methods = [name for name in rows if name not in REFERENCES]
    if not methods:
        raise bd.BoardDataError("the PRE source table has no non-reference methods")
    unknown = sorted(set(methods) - set(SHORT))
    if unknown:
        raise bd.BoardDataError(f"PRE methods need display labels: {unknown!r}")

    comparisons = []
    for source in sources:
        column = columns.index(source)
        best = max(methods, key=lambda name: (rows[name][column], name))
        comparisons.append((source, rows["flag_all"][column], best, rows[best][column],
                            verdicts[source]))

    fb.style(matplotlib)
    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    y_positions = list(reversed(range(len(comparisons))))
    for y, (source, floor, best, score, verdict) in zip(y_positions, comparisons):
        color = fb.GREEN if verdict == "separates" else fb.MUTED
        ax.plot([floor, score], [y, y], color=color, linewidth=3.2, alpha=0.75,
                solid_capstyle="round", zorder=2)
        ax.plot(floor, y, marker="s", markersize=10, markerfacecolor="white",
                markeredgecolor=fb.INK, markeredgewidth=1.8, zorder=4)
        ax.plot(score, y, marker="o", markersize=11,
                markerfacecolor=color if verdict == "separates" else "white",
                markeredgecolor=color, markeredgewidth=2.2, zorder=5)
        ax.text(floor, y + 0.22, f"{floor:.3f}", fontsize=11.5, ha="center", va="bottom",
                color=fb.INK)
        if score > 0.82:
            label_x, align = score - 0.018, "right"
        else:
            label_x, align = score + 0.018, "left"
        ax.text(label_x, y - 0.22, f"{SHORT[best]}  {score:.3f}", fontsize=11.5,
                ha=align, va="top", color=color, fontweight="bold")
        ax.text(1.04, y, verdict, fontsize=12.5, ha="left", va="center", color=color,
                fontweight="bold" if verdict == "separates" else "normal")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(sources, fontsize=14)
    ax.set_xlim(0.0, 1.27)
    ax.set_ylim(-0.65, len(comparisons) - 0.35)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("F1", fontsize=15, labelpad=8)
    ax.tick_params(axis="x", labelsize=13)
    ax.xaxis.grid(True, color=fb.GRID, linewidth=1.0)
    fb.clean_axes(ax)

    legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white",
               markeredgecolor=fb.INK, markeredgewidth=1.8, markersize=9,
               label="flag-everything floor"),
        Line2D([0], [0], marker="o", color=fb.GREEN, markerfacecolor=fb.GREEN,
               markeredgecolor=fb.GREEN, markersize=9, linewidth=2.5,
               label="best method; registered separation"),
        Line2D([0], [0], marker="o", color=fb.MUTED, markerfacecolor="white",
               markeredgecolor=fb.MUTED, markersize=9, linewidth=2.5,
               label="best method; unresolved"),
    ]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.14, 0.90), frameon=False,
               fontsize=11.5, handlelength=2.0)
    fig.suptitle("PRE: does the best method beat flagging everything?", fontsize=19,
                 fontweight="bold", y=0.99)
    fig.text(0.14, 0.93, "Per-source F1; verdicts come from registered source-level tests",
             fontsize=12.5, color=fb.MUTED)
    fig.text(0.14, 0.018,
             "injecagent uses roster-derived labels; its separation is specific to that construction.",
             fontsize=11.5, color=fb.MUTED)
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.15, right=0.98)
    fb.save_web_png(fig, output, "board_pre_source", payload)
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
