"""Shared web rendering and semantic provenance for board-derived README figures."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, PngImagePlugin

import board_data as bd


WHITE = "#FFFFFF"
INK = "#17212B"
MUTED = "#65727E"
GRID = "#D9E0E5"
BLUE = "#276FBF"
ORANGE = "#D45135"
GREEN = "#16836B"
PURPLE = "#7452A5"

# The paper palette. README figures opt into these names as they are rebuilt.
AB_GRAY = "#C9C9C9"
AB_MINT = "#BFDFD2"
AB_CORAL = "#ED8D5A"
AB_NEAR_BLACK = "#1A1A1A"
AB_SUBTITLE = "#666666"


def style(matplotlib) -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "axes.facecolor": WHITE,
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "xtick.color": INK,
        "ytick.color": INK,
    })


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="both", length=0)
    ax.set_axisbelow(True)


def save_web_png(fig, output: Path, figure_id: str, payload: dict[str, object], dpi: int = 160) -> None:
    """Save an opaque RGB PNG and bind its semantic inputs as PNG text chunks."""

    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = io.BytesIO()
    fig.savefig(rendered, format="png", dpi=dpi, facecolor=WHITE, transparent=False,
                bbox_inches="tight", pad_inches=0.12)
    rendered.seek(0)
    with Image.open(rendered) as source:
        image = source.convert("RGB")
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text(bd.META_FIGURE, figure_id)
        metadata.add_text(bd.META_PAYLOAD, bd.canonical_payload(payload))
        metadata.add_text(bd.META_DIGEST, bd.payload_digest(payload))
        provenance = bd.SOURCE_DESCRIPTION
        if figure_id == "board_pre_source":
            provenance += "; tools/statistical_tests_results.json (verdict words only)"
        elif figure_id == "board_live_prefix":
            provenance += "; tools/statistical_tests_results.json (threshold verdicts and estimate sides)"
        metadata.add_text(bd.META_SOURCE, provenance)
        image.save(output, format="PNG", pnginfo=metadata, optimize=True, dpi=(dpi, dpi))
        width, height = image.size
    print(f"wrote {output} ({width}x{height}, data {bd.payload_digest(payload)})")
