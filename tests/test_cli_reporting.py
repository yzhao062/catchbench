"""Keep the CLI's reading guidance descriptive, with measurements instead of test verdicts."""

import ast
from pathlib import Path
import re

import pytest

from catchbench import cli


_ADJUDICATION = re.compile(
    r"\bHolm\b"
    r"|\b(?:fail(?:s|ed|ing)? to|do(?:es)? not|did not|cannot) separate\b"
    r"|\bseparates? (?:on|from)\b"
    r"|\b(?:ordering|comparisons?|pairs?|cells?|band|rest|full and auditable)\b"
    r"[^.!?\n]{0,100}\bunresolved\b"
    r"|\b(?:registered|paired|two-sided) (?:tests?|contrasts?|family)\b"
    r"|\b(?:tests?|contrasts?|famil(?:y|ies))\b[^.!?\n]{0,100}"
    r"\b(?:separates?|establish(?:es)?|leaves?|places?|resolves?)\b"
    r"|\bp(?:[- ]values?)?\s*(?:=|<|>|<=|>=|\u2264|\u2265|of\b|is\b)\s*(?:\d|\.\d)"
    r"|\bp[- ]values?\s+(?:\d|\.\d)",
    re.IGNORECASE,
)


def _adjudications(text):
    return _ADJUDICATION.findall(" ".join(text.split()))


def test_cli_reporting_has_no_adjudication(capsys):
    """A blanket word ban would reject SWE-Gym's unresolved outcome labels and
    ordinary descriptions of a baseline that separates two fault kinds. Check
    verdict phrases and quoted numeric p-values, including adjacent source
    literals, and exercise the actual reading printer used by main().
    """
    tree = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
    literals = "\n".join(
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    assert not _adjudications(literals)

    cli._print_reading()
    output = capsys.readouterr().out
    assert output.startswith("\nReading:\n- Localization (Who&When):")
    assert "\n- LIVE stale-state (online detection):" in output
    assert output.endswith("effect estimate.\n")
    assert not _adjudications(output)


@pytest.mark.parametrize("text", [
    "Holm-adjusted p-values",
    "Both fail to separate.",
    "Failing to separate is not evidence that they are equal.",
    "126 runs do not separate the models.",
    "The ordering within that band is unresolved.",
    "The tests leave both comparisons unresolved.",
    "The paired test leaves that pair unresolved.",
    "The Top-1 comparison remains unresolved.",
    "GRADE's method separates from position on Top-3.",
    "The registered contrast separates on SWE-Gym.",
    "The registered test supports the ranking.",
    "The registered tests support the ranking.",
    "The family establishes all entrants below the bar.",
    "The full and auditable cells are unresolved.",
    "Auditable and the rest are unresolved.",
    "The method wins because p=0.01.",
    "The result holds since p < 0.05.",
    "The method wins because p=.01.",
    "The result follows from p-value 0.01.",
    "The p-value of 0.01 supports this comparison.",
])
def test_cli_reporting_guard_rejects_verdicts(text):
    assert _adjudications(text)


@pytest.mark.parametrize("text", [
    'SWE-Gym outcomes are labeled "resolved" and "unresolved".',
    "SWE-Gym: 188 unresolved runs and 188 resolved runs.",
    "The baseline separates two fault kinds.",
    "The structure separates the two causes, each feature keyed to one mechanism.",
    "Can a method separate failing from resolved runs?",
    "The released record retains every p-value.",
    "The difference is +0.046 (95% CI [-0.005, 0.098]).",
])
def test_cli_reporting_guard_allows_measurements_and_outcome_labels(text):
    assert not _adjudications(text)
