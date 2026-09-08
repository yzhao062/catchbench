"""Shared matplotlib styling for the board-derived README figures.

The PNG writer that used to live here is gone. It stamped each export with a source string naming
the statistics record, which the two board figures no longer read, and it had no callers left once
they moved to ``board_data.save_board_png``. Keeping a second copy of the metadata contract, with
the wrong source on it, was the failure waiting to happen. This module now carries palette and axis
styling only, so it needs neither Pillow nor the board module.
"""

from __future__ import annotations

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

