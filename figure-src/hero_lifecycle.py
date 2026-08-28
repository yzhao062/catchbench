"""Render the README lifecycle hero in the paper palette.

The geometry and information-mask framing adapt the paper's ``fig_lifecycle.tex``
panel (a). Labels and counts are checked against the committed README and board.

Run from the repository root:

    python figure-src/hero_lifecycle.py
"""

from __future__ import annotations

import argparse
import io
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image, PngImagePlugin

import board_data as bd


ROOT = Path(__file__).resolve().parents[1]
HERO_SOURCE_DESCRIPTION = bd.SOURCE_DESCRIPTION + "; README.md (phrase checks only)"
DEFAULT_BOARD = ROOT / "tests" / "golden" / "board.txt"
DEFAULT_README = ROOT / "README.md"
DEFAULT_OUTPUT = ROOT / "assets" / "hero-lifecycle.png"

WIDTH = 1800
HEIGHT = 720
DPI = 200

GRAY = "#C9C9C9"
MINT = "#BFDFD2"
CORAL = "#ED8D5A"
NEAR_BLACK = "#1A1A1A"
SUBTITLE = "#666666"
PALETTE = (GRAY, MINT, CORAL, NEAR_BLACK, SUBTITLE)

HEADER_RE = re.compile(r"^\[(PRE|LIVE|POST)\]\s+([^:]+?)\s+::\s+(.+)$", re.MULTILINE)


def _hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.removeprefix("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def fact_check(board_path: Path, readme_path: Path) -> dict[str, int]:
    board = board_path.read_text(encoding="utf-8")
    readme = " ".join(readme_path.read_text(encoding="utf-8").split())

    headers = HEADER_RE.findall(board)
    scored = [
        (phase, board_id.strip(), corpus.strip())
        for phase, board_id, corpus in headers
        if corpus.strip() != "F1 by source"
        and f"[{phase}] {board_id.strip()} :: {corpus.strip()}" not in bd.DIAGNOSTIC_HEADERS
    ]
    counts = Counter(phase for phase, _, _ in scored)
    expected = {"PRE": 1, "LIVE": 3, "POST": 5}
    if dict(counts) != expected:
        raise ValueError(f"scored block counts changed: expected {expected}, found {dict(counts)}")

    if "streaming prefixes [25%, 50%, 75%, 100%]" not in board:
        raise ValueError("tests/golden/board.txt no longer declares the four displayed prefixes")

    readme_facts = (
        "CatchBench holds the run fixed and varies what the auditor may read",
        "only the plan and harness",
        "a growing prefix is visible, the outcome is not",
        "the complete trace and outcome are in hand",
    )
    missing = [fact for fact in readme_facts if fact not in readme]
    if missing:
        raise ValueError(f"README lifecycle wording changed; missing: {missing}")
    return expected


def _cell(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    masked: bool = False,
    linewidth: float = 1.7,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0,rounding_size=0.012",
        facecolor=fill,
        edgecolor=NEAR_BLACK,
        linewidth=linewidth,
        hatch="////" if masked else None,
    )
    ax.add_patch(patch)


def _label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    size: float,
    color: str = NEAR_BLACK,
    weight: str = "normal",
    ha: str = "left",
    va: str = "center",
) -> None:
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        family="DejaVu Serif",
    )


def draw(counts: dict[str, int]) -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor=GRAY)
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, 1), ylim=(0, 1))
    ax.set_facecolor(GRAY)
    ax.axis("off")

    _label(ax, 0.047, 0.905, "CATCHBENCH", size=17, color=CORAL, weight="bold")
    _label(ax, 0.047, 0.830, "ONE RUN. THREE INFORMATION STATES.", size=22, weight="bold")
    _label(
        ax,
        0.047,
        0.765,
        "The run stays fixed. PRE, LIVE, and POST change what the auditor may read.",
        size=11,
        color=SUBTITLE,
    )

    label_x = 0.047
    declaration_x = 0.205
    declaration_w = 0.086
    trace_x = 0.315
    trace_w = 0.049
    trace_gap = 0.010
    outcome_x = 0.812
    outcome_w = 0.083
    cell_h = 0.071

    _label(ax, declaration_x + declaration_w / 2, 0.704, "DECLARATION", size=10, color=SUBTITLE, ha="center")
    _label(ax, trace_x + (8 * trace_w + 7 * trace_gap) / 2, 0.704, "TRACE", size=10, color=SUBTITLE, ha="center")
    _label(ax, outcome_x + outcome_w / 2, 0.704, "OUTCOME", size=10, color=SUBTITLE, ha="center")

    run_y = 0.594
    _label(ax, label_x, run_y + cell_h / 2, "ONE RUN", size=16, weight="bold")
    arrow_y = run_y + cell_h + 0.015
    ax.annotate(
        "",
        xy=(outcome_x - 0.010, arrow_y),
        xytext=(declaration_x + declaration_w + 0.009, arrow_y),
        arrowprops={"arrowstyle": "->", "color": NEAR_BLACK, "linewidth": 1.2},
    )
    _cell(ax, declaration_x, run_y, declaration_w, cell_h, fill=GRAY, linewidth=1.5)
    for i in range(8):
        _cell(ax, trace_x + i * (trace_w + trace_gap), run_y, trace_w, cell_h, fill=GRAY, linewidth=1.5)
    _cell(ax, outcome_x, run_y, outcome_w, cell_h, fill=CORAL, linewidth=2.2)
    _label(ax, outcome_x + outcome_w / 2, run_y + cell_h / 2, "FAIL", size=12, weight="bold", ha="center")
    ax.plot([0.047, 0.953], [0.557, 0.557], color=SUBTITLE, linewidth=0.7)

    rows = (("PRE", 0.444), ("LIVE", 0.310), ("POST", 0.142))
    readable_traces = {"PRE": 0, "LIVE": 4, "POST": 8}
    for phase, y in rows:
        phase_color = CORAL if phase == "LIVE" else NEAR_BLACK
        _label(ax, label_x, y + cell_h * 0.73, phase, size=14, color=phase_color, weight="bold")
        suffix = "block" if counts[phase] == 1 else "blocks"
        _label(
            ax,
            label_x,
            y + cell_h * 0.17,
            f"{counts[phase]} scored {suffix}",
            size=7,
            color=SUBTITLE,
        )

        _cell(ax, declaration_x, y, declaration_w, cell_h, fill=MINT)
        for i in range(8):
            open_now = i < readable_traces[phase]
            _cell(
                ax,
                trace_x + i * (trace_w + trace_gap),
                y,
                trace_w,
                cell_h,
                fill=MINT if open_now else GRAY,
                masked=not open_now,
            )
        outcome_open = phase == "POST"
        _cell(
            ax,
            outcome_x,
            y,
            outcome_w,
            cell_h,
            fill=MINT if outcome_open else GRAY,
            masked=not outcome_open,
        )

    prefix_centers = [
        trace_x + 2 * (trace_w + trace_gap) - trace_gap / 2,
        trace_x + 4 * (trace_w + trace_gap) - trace_gap / 2,
        trace_x + 6 * (trace_w + trace_gap) - trace_gap / 2,
        trace_x + 8 * (trace_w + trace_gap) - trace_gap / 2,
    ]
    for x, pct in zip(prefix_centers, (25, 50, 75, 100), strict=True):
        ax.plot([x, x], [0.292, 0.281], color=SUBTITLE, linewidth=0.8)
        _label(ax, x, 0.263, f"{pct}%", size=10, color=SUBTITLE, ha="center")
    boundary_x = prefix_centers[1]
    ax.plot([boundary_x, boundary_x], [0.302, 0.389], color=CORAL, linewidth=3.0)

    key_y = 0.047
    ax.add_patch(Rectangle((0.205, key_y), 0.025, 0.026, facecolor=MINT, edgecolor=NEAR_BLACK, linewidth=1.3))
    _label(ax, 0.238, key_y + 0.013, "READABLE", size=9, color=SUBTITLE)
    ax.add_patch(
        Rectangle(
            (0.363, key_y),
            0.025,
            0.026,
            facecolor=GRAY,
            edgecolor=NEAR_BLACK,
            linewidth=1.3,
            hatch="////",
        )
    )
    _label(ax, 0.396, key_y + 0.013, "MASKED", size=9, color=SUBTITLE)
    ax.plot([0.535, 0.535], [key_y - 0.002, key_y + 0.028], color=CORAL, linewidth=3.0)
    _label(ax, 0.547, key_y + 0.013, "LIVE PREFIX SHOWN AT 50%", size=9, color=SUBTITLE)
    return fig


def save_exact_palette(fig: plt.Figure, output: Path, board_path: Path) -> None:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI, facecolor=GRAY, edgecolor=GRAY)
    plt.close(fig)
    buffer.seek(0)
    rendered = Image.open(buffer).convert("RGB")
    if rendered.size != (WIDTH, HEIGHT):
        raise ValueError(f"unexpected raster size: {rendered.size}")

    colors = [_hex_rgb(color) for color in PALETTE]
    palette_image = Image.new("P", (1, 1))
    palette_image.putpalette([channel for rgb in colors + [colors[0]] * 251 for channel in rgb])
    exact = rendered.quantize(palette=palette_image, dither=Image.Dither.NONE)

    used = {rgb for _, rgb in exact.convert("RGB").getcolors(maxcolors=WIDTH * HEIGHT) or []}
    allowed = set(colors)
    if not used <= allowed:
        raise ValueError(f"render introduced colors outside the paper palette: {used - allowed}")

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = bd.figure_payload("hero-lifecycle", board_path)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("generator", "figure-src/hero_lifecycle.py")
    metadata.add_text("palette", ", ".join(PALETTE))
    metadata.add_text(bd.META_FIGURE, "hero-lifecycle")
    metadata.add_text(bd.META_SOURCE, HERO_SOURCE_DESCRIPTION)
    metadata.add_text(bd.META_PAYLOAD, bd.canonical_payload(payload))
    metadata.add_text(bd.META_DIGEST, bd.payload_digest(payload))
    # The quantize above is the palette guarantee. Written back as RGB so every committed
    # README asset shares one colour type; the asset checker accepts an opaque palette
    # image too, since it keys on the tRNS chunk rather than on the colour type.
    exact.convert("RGB").save(output, optimize=True, pnginfo=metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", type=Path, default=DEFAULT_BOARD)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    counts = fact_check(args.board, args.readme)
    save_exact_palette(draw(counts), args.output, args.board)
    print(f"wrote {args.output} ({WIDTH}x{HEIGHT}; palette={','.join(PALETTE)})")


if __name__ == "__main__":
    main()
