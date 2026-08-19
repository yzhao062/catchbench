"""Bind the test session to the checkout it lives in.

Without this, ``from catchbench import ...`` resolves through whatever the
active environment happens to have installed. A stale editable install can then
point the tests at a different tree entirely, and they pass while testing code
that is not the code under review. Putting this checkout's ``src`` at the front
of ``sys.path`` makes a local run agree with CI, where pip installs from the
checkout and no other copy exists.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
