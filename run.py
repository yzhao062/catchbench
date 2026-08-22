"""Source-checkout wrapper for the CatchBench command-line interface."""
from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from catchbench.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
