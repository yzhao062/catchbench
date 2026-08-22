"""One test that must fail, so that CI can check the suite is still able to report a failure.

Every version of this repository's test gate has read a report produced by the same process the
repository configures. Two adversarial rounds established that this cannot be made sound: reading
pytest's terminal summary was broken eight ways, and reading `--junit-xml` instead moves the attack
from the report's text to the `TestReport` object the XML is built from. Five lines in a conftest
turn every failure into a pass, in the terminal, in the XML, and in the exit status together.

What no such report can supply is an independent view. This file is that view. It is not collected
by `pytest tests`, which passes `--ignore=tests/canary`. CI runs it on its own and requires the run
to come back non-zero with this node id reported as failed. A suite that has been made to report
failures as passes cannot tell this failure apart from a real one, so the same edit that hides a
regression also turns this canary green, and CI goes red for the canary instead.

It is deliberately not a source mutation. Mutating a real function costs a second full suite run per
job, and the mutation goes stale as the code around it moves, which produces a confusing red that
teaches people to ignore the step.

Nothing here imports the package, so this run stays fast and cannot fail for an unrelated reason.
"""


def test_this_test_must_fail():
    """If this ever reports a pass, the suite is no longer reporting outcomes honestly."""
    assert False, "the canary is supposed to fail; CI checks that it did"
