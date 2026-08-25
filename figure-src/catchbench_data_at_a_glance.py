"""Render the CatchBench scale-and-composition figure from the committed board.

The figure keeps seven numeric claims in the glanceable layer. Corpus-specific
counts still determine the exact widths of the composition bars, while their
printed labels stay compact enough for README display.

Run: python figure-src/catchbench_data_at_a_glance.py
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, PngImagePlugin

import board_data as bd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOARD = ROOT / "tests" / "golden" / "board.txt"
DEFAULT_OUTPUT = ROOT / "assets" / "catchbench_data_at_a_glance.png"

# Paper palette from figures-preamble.tex. White is the requested card background.
GRAY = "#C9C9C9"
MINT = "#BFDFD2"
CORAL = "#ED8D5A"
NEAR_BLACK = "#1A1A1A"
SUBTITLE = "#666666"
WHITE = "#FFFFFF"



def add_round_rect(ax, x: float, y: float, width: float, height: float,
                   edge: str = GRAY, face: str = WHITE, linewidth: float = 1.4,
                   radius: float = 0.016, zorder: int = 1) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=linewidth, edgecolor=edge, facecolor=face, zorder=zorder,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    return patch


def add_composition_bar(ax, x: float, y: float, width: float, height: float,
                        counts: list[int], labels: list[str],
                        label_sizes: list[float] | None = None) -> list[tuple[float, float]]:
    total = sum(counts)
    cursor = x
    spans = []
    for index, (count, label) in enumerate(zip(counts, labels)):
        segment_width = width * count / total
        rect = Rectangle(
            (cursor, y), segment_width, height, transform=ax.transAxes,
            facecolor=MINT, edgecolor=WHITE, linewidth=2.1, zorder=4,
        )
        ax.add_patch(rect)
        size = label_sizes[index] if label_sizes else 8.0
        if label:
            ax.text(cursor + segment_width / 2, y + height / 2, label,
                    transform=ax.transAxes, ha="center", va="center", fontsize=size,
                    fontweight="bold", color=NEAR_BLACK, zorder=5)
        spans.append((cursor, segment_width))
        cursor += segment_width
    ax.add_patch(Rectangle(
        (x, y), width, height, transform=ax.transAxes,
        facecolor="none", edgecolor=NEAR_BLACK, linewidth=0.75, zorder=6,
    ))
    return spans


def render(facts: bd.BoardFacts, output: Path, board_path: Path) -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "text.color": NEAR_BLACK,
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
    })

    fig = plt.figure(figsize=(10, 5.5), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.05, 0.925, "CatchBench, at a glance", fontsize=28, fontweight="bold",
            ha="left", va="center", color=NEAR_BLACK)
    ax.text(0.05, 0.865,
            "Declared configurations and recorded traces become scored audits across the agent lifecycle",
            fontsize=12.4, ha="left", va="center", color=SUBTITLE)
    ax.plot([0.05, 0.95], [0.825, 0.825], color=GRAY, linewidth=1.1,
            transform=ax.transAxes, zorder=1)

    left = (0.05, 0.49, 0.425, 0.295)
    right = (0.525, 0.49, 0.425, 0.295)
    add_round_rect(ax, *left)
    add_round_rect(ax, *right)

    lx, ly, lw, lh = left
    ax.text(lx + 0.025, ly + lh - 0.045, "DECLARATION DATA  •  PRE",
            fontsize=9.2, fontweight="bold", color=SUBTITLE, ha="left", va="center")
    ax.text(lx + 0.025, ly + 0.165, f"{facts.pre_total:,}", fontsize=31,
            fontweight="bold", color=CORAL, ha="left", va="center")
    ax.text(lx + 0.172, ly + 0.165, "declared agent\nconfigurations", fontsize=13.2,
            fontweight="bold", color=NEAR_BLACK, ha="left", va="center", linespacing=1.05)
    ax.text(lx + lw - 0.025, ly + 0.182, f"{len(facts.pre_sources)}",
            fontsize=19, fontweight="bold", color=NEAR_BLACK, ha="right", va="center")
    ax.text(lx + lw - 0.025, ly + 0.142, "SOURCE\nCORPORA",
            fontsize=7.2, fontweight="bold", color=SUBTITLE, ha="right", va="center",
            linespacing=0.95)

    source_order = sorted(facts.pre_sources, key=facts.pre_sources.get, reverse=True)
    ax.text(lx + lw / 2, ly + 0.102, "INJECAGENT  •  CREWAI  •  N8N",
            fontsize=7.3, fontweight="bold", color=SUBTITLE, ha="center", va="center")
    ax.text(lx + lw / 2, ly + 0.079, "MCP  •  SWEAGENT  •  SYNTHETIC",
            fontsize=7.3, fontweight="bold", color=SUBTITLE, ha="center", va="center")
    add_composition_bar(
        ax, lx + 0.025, ly + 0.025, lw - 0.05, 0.037,
        [facts.pre_sources[name] for name in source_order],
        ["" for _ in source_order],
    )

    rx, ry, rw, rh = right
    ax.text(rx + 0.025, ry + rh - 0.045, "EXECUTION DATA  •  LIVE / POST",
            fontsize=9.2, fontweight="bold", color=SUBTITLE, ha="left", va="center")
    ax.text(rx + 0.025, ry + 0.165, f"{facts.trace_total:,}", fontsize=31,
            fontweight="bold", color=CORAL, ha="left", va="center")
    ax.text(rx + 0.172, ry + 0.165, "recorded agent\nruns", fontsize=13.2,
            fontweight="bold", color=NEAR_BLACK, ha="left", va="center", linespacing=1.05)
    ax.text(rx + rw - 0.025, ry + 0.207, f"{len(facts.trace_runs)}",
            fontsize=19, fontweight="bold", color=NEAR_BLACK, ha="right", va="center")
    ax.text(rx + rw - 0.025, ry + 0.167, "TRACE\nCORPORA",
            fontsize=7.2, fontweight="bold", color=SUBTITLE, ha="right", va="center",
            linespacing=0.95)

    trace_names = list(facts.trace_runs)
    trace_spans = add_composition_bar(
        ax, rx + 0.025, ry + 0.068, rw - 0.05, 0.060,
        [facts.trace_runs[name] for name in trace_names], trace_names,
        [6.8, 7.8, 7.8],
    )
    swe_index = trace_names.index("SWE-Gym")
    swe_x, swe_width = trace_spans[swe_index]
    gold_width = swe_width * facts.gold_injections / facts.trace_runs["SWE-Gym"]
    bracket_y = ry + 0.052
    ax.plot([swe_x, swe_x + gold_width], [bracket_y, bracket_y], color=CORAL,
            linewidth=3.0, solid_capstyle="butt", transform=ax.transAxes, zorder=7)
    ax.plot([swe_x, swe_x], [bracket_y, bracket_y + 0.011], color=CORAL,
            linewidth=1.2, transform=ax.transAxes, zorder=7)
    ax.plot([swe_x + gold_width, swe_x + gold_width], [bracket_y, bracket_y + 0.011],
            color=CORAL, linewidth=1.2, transform=ax.transAxes, zorder=7)
    ax.text(swe_x + gold_width / 2, ry + 0.020,
            f"{facts.gold_injections} GOLD INJECTIONS  •  PAIRED CLEAN CONTROLS",
            fontsize=6.8, fontweight="bold", color=CORAL, ha="center", va="center",
            transform=ax.transAxes)

    board_box = (0.235, 0.145, 0.245, 0.205)
    entrant_box = (0.555, 0.145, 0.245, 0.205)
    add_round_rect(ax, *board_box, edge=CORAL, linewidth=2.0)
    add_round_rect(ax, *entrant_box, edge=MINT, linewidth=2.0)

    connectors = (
        (lx + lw / 2, board_box[0] + 0.08, -0.14),
        (rx + rw / 2, board_box[0] + board_box[2] - 0.08, 0.14),
    )
    for start_x, end_x, bend in connectors:
        arrow = FancyArrowPatch(
            (start_x, ly - 0.006), (end_x, board_box[1] + board_box[3] + 0.006),
            arrowstyle="-|>", mutation_scale=10, linewidth=1.1, color=GRAY,
            connectionstyle=f"arc3,rad={bend}", transform=ax.transAxes, zorder=0,
        )
        ax.add_patch(arrow)

    bx, by, bw, bh = board_box
    ax.text(bx + 0.032, by + 0.125, f"{facts.board_count}", fontsize=29,
            fontweight="bold", color=CORAL, ha="left", va="center")
    ax.text(bx + 0.104, by + 0.126, "scored\nboards", fontsize=13.4,
            fontweight="bold", color=NEAR_BLACK, ha="left", va="center", linespacing=1.0)
    ax.text(bx + bw / 2, by + 0.045, "PRE   •   LIVE   •   POST",
            fontsize=9.5, fontweight="bold", color=SUBTITLE, ha="center", va="center")

    arrow = FancyArrowPatch(
        (bx + bw + 0.012, by + bh / 2), (entrant_box[0] - 0.012, by + bh / 2),
        arrowstyle="-|>", mutation_scale=12, linewidth=1.8, color=CORAL,
        transform=ax.transAxes, zorder=2,
    )
    ax.add_patch(arrow)

    ex, ey, ew, eh = entrant_box
    ax.text(ex + 0.031, ey + 0.125, f"{facts.entrant_count}", fontsize=29,
            fontweight="bold", color=NEAR_BLACK, ha="left", va="center")
    ax.text(ex + 0.106, ey + 0.126, "evaluated\nentrants", fontsize=13.4,
            fontweight="bold", color=NEAR_BLACK, ha="left", va="center", linespacing=1.0)
    ax.text(ex + ew / 2, ey + 0.047,
            "LLM judges  •  rule scanners\nstructure + graph  •  anomaly detectors",
            fontsize=7.3, color=SUBTITLE, ha="center", va="center", linespacing=1.15)

    ax.text(0.95, 0.067, "SOURCE: COMMITTED CATCHBENCH BOARD",
            fontsize=7.2, fontweight="bold", color=SUBTITLE, ha="right", va="center")

    payload = bd.figure_payload("catchbench_data_at_a_glance", board_path)
    # One definition of canonical form, shared with the checker, so a digest cannot
    # disagree because the two sides serialized the same payload differently.
    canonical = bd.canonical_payload(payload)
    digest = bd.payload_digest(payload)
    rendered = io.BytesIO()
    fig.savefig(rendered, format="png", dpi=200, facecolor=WHITE, transparent=False)
    plt.close(fig)
    rendered.seek(0)

    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(rendered) as source:
        image = source.convert("RGB")
        if image.size != (2000, 1100):
            raise ValueError(f"unexpected render dimensions: {image.size!r}")
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text(bd.META_FIGURE, "catchbench_data_at_a_glance")
        # Repo-relative, never an absolute path: an absolute one leaks whatever
        # working tree the figure happened to be rendered in into a committed asset.
        metadata.add_text(bd.META_SOURCE, bd.SOURCE_DESCRIPTION)
        metadata.add_text(bd.META_PAYLOAD, canonical)
        metadata.add_text(bd.META_DIGEST, digest)
        image.save(output, format="PNG", pnginfo=metadata, optimize=True, dpi=(200, 200))
    print(f"wrote {output} (2000x1100, data {digest})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", type=Path, default=DEFAULT_BOARD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    facts = bd.parse_board(args.board)
    render(facts, args.output, args.board)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
