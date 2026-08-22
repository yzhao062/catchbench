"""CI's report check must fail when the suite lies to it about what it did.

The check this replaces read pytest's terminal summary, and its test fed the check synthetic
strings. An adversarial review then reproduced eight working bypasses, and that test passed under
all eight, because it never saw a real report. A test that cannot fail when the thing it guards is
defeated is worse than no test, so this one builds a miniature repository with the shapes this one
has, runs pytest on it for real, and hands the resulting junit-xml report to the shipped
``tools/check_test_report.py``.

Every attack below runs with a truncation regression live in the miniature detector, mirroring the
PyGOD bug this repository actually shipped. So a green result under any attack would mean a real
contract stopped asserting while CI stayed quiet. The control at the top pins both directions: the
sound repository passes, and the regression alone fails.

The three attacks at the end are the ones that still work, or that work with one more line of
care. They are written as passing tests asserting what the check actually returns, so the limits
are executable and cannot quietly stop being true.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tools" / "check_test_report.py"
WORKFLOW = ROOT / ".github" / "workflows" / "verify.yml"

# The miniature repository. Its detector carries the same shape of bug as the PyGOD regression:
# scores land on the head of the chunk and every other node is left at zero.
DETECTOR = {
    "sound": "def scores():\n    return [1.0, 2.0, 3.0]\n",
    "truncated": "def scores():\n    return [1.0, 0.0, 0.0]\n",
}
CONTRACT_BODY = '''from detector import scores


def test_scores_every_node():
    assert all(value > 0 for value in scores())


def test_scores_every_node_it_was_given():
    assert len(scores()) == 3
'''
# The optional-stack file mirrors tests/test_graph_ad_scores_every_node.py: a module marker rather
# than a module-level importorskip, so the node id exists whether or not the stack does.
OPTIONAL = '''import pytest

pytestmark = pytest.mark.skipif(True, reason="needs the graph-ad extra: No module named 'torch'")


def test_needs_the_stack():
    assert False
'''
MANIFEST = """always       tests/test_contract.py::test_scores_every_node
always       tests/test_contract.py::test_scores_every_node_it_was_given
needs-torch  tests/test_optional.py::test_needs_the_stack
"""
TRIVIA = "def test_first():\n    assert True\n\n\ndef test_second():\n    assert True\n"

# Payloads, quoted from the adversarial report. Five of its eight bypasses needed only a conftest.
REWRITE_EXIT_STATUS = '''def pytest_sessionfinish(session, exitstatus):
    session.exitstatus = 0
'''
RELABEL_SKIPS = '''def pytest_report_teststatus(report, config):
    if report.skipped:
        return "passed", ".", "PASSED"
'''
FORGE_SUMMARY = '''def pytest_unconfigure(config):
    print("3 passed in 0.05s")
'''
DROP_THE_CONTRACT = 'collect_ignore = ["test_contract.py"]\n'
SWAP_FOR_A_LAMBDA = '''def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.name == "test_scores_every_node":
            item.obj = lambda *args, **kwargs: None
'''
# The same swap, staying inside the test module. junit-xml records where the callable pytest ran
# was defined, so the lambda above moves the reported file to conftest.py while this one does not.
SWAP_INSIDE_THE_MODULE = '''def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.name == "test_scores_every_node":
            item.obj = item.module.test_scores_every_node_it_was_given
'''
XFAILED_CONTRACT = '''import pytest

from detector import scores


@pytest.mark.xfail(reason="known")
def test_scores_every_node():
    assert all(value > 0 for value in scores())


def test_scores_every_node_it_was_given():
    assert len(scores()) == 3
'''
XPASSING_CONTRACT = '''import pytest

from detector import scores


@pytest.mark.xfail(reason="stale marker")
def test_scores_every_node():
    assert len(scores()) == 3


def test_scores_every_node_it_was_given():
    assert len(scores()) == 3
'''
GUTTED_CONTRACT = '''from detector import scores


def test_scores_every_node():
    pass


def test_scores_every_node_it_was_given():
    assert len(scores()) == 3
'''


def _contract(skip_reason: str | None = None) -> str:
    if skip_reason is None:
        return CONTRACT_BODY
    return f"import pytest\n\npytestmark = pytest.mark.skip(reason={skip_reason!r})\n\n" \
           + CONTRACT_BODY


def _repo(tmp_path: Path, *, detector: str = "sound", contract: str | None = None,
          conftest: str = "", extra_tests: dict[str, str] | None = None) -> Path:
    """A miniature repository: two contracts, one optional-stack file, a badge, and a manifest."""
    tests = tmp_path / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    (tests / "detector.py").write_text(DETECTOR[detector], encoding="utf-8")
    (tests / "test_contract.py").write_text(contract or CONTRACT_BODY, encoding="utf-8")
    (tests / "test_optional.py").write_text(OPTIONAL, encoding="utf-8")
    for name, body in (extra_tests or {}).items():
        (tests / name).write_text(body, encoding="utf-8")
    if conftest:
        (tests / "conftest.py").write_text(conftest, encoding="utf-8")
    # An ini file here stops pytest walking up into this repository's own configuration.
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "![Tests](https://img.shields.io/badge/tests-3-brightgreen.svg)\n", encoding="utf-8")
    (tmp_path / "expected_tests.txt").write_text(MANIFEST, encoding="utf-8")
    return tmp_path


def _pytest(repo: Path, environment: dict[str, str] | None = None) -> tuple[int, str]:
    """Run pytest over the miniature repository with the flags the workflow uses."""
    child = {key: value for key, value in os.environ.items() if key != "PYTEST_ADDOPTS"}
    child.update(environment or {})
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "-ra", "--junit-xml=report.xml",
         "-o", "junit_family=xunit1", "-o", "addopts=", "-p", "no:cacheprovider"],
        cwd=repo, env=child, capture_output=True, text=True)
    return done.returncode, done.stdout + done.stderr


def _check(repo: Path, *extra: str) -> tuple[int, str]:
    """Run the shipped checker over that report, the way the workflow runs it."""
    done = subprocess.run(
        [sys.executable, "-B", str(CHECKER), "--report", str(repo / "report.xml"),
         "--manifest", str(repo / "expected_tests.txt"), "--tests-dir", str(repo / "tests"),
         "--readme", str(repo / "README.md"), *extra],
        cwd=repo, capture_output=True, text=True)
    return done.returncode, done.stdout + done.stderr


def _checker_module():
    spec = importlib.util.spec_from_file_location("_check_test_report", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_sound_repository_passes(tmp_path):
    """The control in the green direction: without it, every red result below proves nothing."""
    repo = _repo(tmp_path)
    code, out = _pytest(repo)

    assert code == 0, out
    assert "2 passed, 1 skipped" in out
    assert _check(repo) == (0, "tests: 3 test(s) reported, matching expected_tests.txt exactly "
                               "(2 passed, 1 skipped); 1 of 3 declared as allowed to skip\n")


def test_the_regression_on_its_own_is_caught(tmp_path):
    """The control in the red direction: the sandbox can fail, so a green attack is a real one."""
    repo = _repo(tmp_path, detector="truncated")
    code, out = _pytest(repo)
    assert code == 1, out

    code, out = _check(repo)

    assert code == 1
    assert "a test reported failed: tests/test_contract.py::test_scores_every_node" in out


def test_a_rewritten_exit_status_no_longer_clears_a_failing_contract(tmp_path):
    """Bypass 1: three lines of conftest make pytest report success while a contract fails."""
    repo = _repo(tmp_path, detector="truncated", conftest=REWRITE_EXIT_STATUS)
    code, out = _pytest(repo)
    assert code == 0, "the attack is meant to leave pytest exiting 0; it did not"
    assert "1 failed" in out

    code, out = _check(repo)

    assert code == 1
    assert "a test reported failed: tests/test_contract.py::test_scores_every_node" in out


def test_relabelling_skips_as_passes_no_longer_hides_a_skipped_contract(tmp_path):
    """Bypass 3: pytest_report_teststatus rewrites the terminal, which junit-xml does not read."""
    repo = _repo(tmp_path, detector="truncated", contract=_contract("quarantined while we debug"),
                 conftest=RELABEL_SKIPS)
    code, out = _pytest(repo)
    assert code == 0, out
    assert "3 passed" in out and "SKIPPED" not in out

    code, out = _check(repo)

    assert code == 1
    assert out.count("skipped, and the manifest says it needs nothing this run lacks") == 2


def test_hiding_the_summary_no_longer_hides_a_skipped_contract(tmp_path):
    """Bypass 4: --no-summary erases every SKIPPED line while the counts still add up.

    PYTEST_ADDOPTS carries it here because that is the half ``-o addopts=`` does not neutralize,
    so the terminal really is blind and the junit-xml report is doing the work.
    """
    repo = _repo(tmp_path, detector="truncated", contract=_contract("needs the graph-ad extra"))
    code, out = _pytest(repo, environment={"PYTEST_ADDOPTS": "--no-summary"})
    assert code == 0, out
    assert "SKIPPED" not in out and "3 skipped" in out

    code, out = _check(repo)

    assert code == 1
    assert out.count("skipped, and the manifest says it needs nothing this run lacks") == 2


def test_a_forged_path_in_a_skip_reason_no_longer_clears_the_skip(tmp_path):
    """Bypass 6: the reason text sat where the old guard looked for a path, and cleared itself.

    Nothing here parses a reason. The node id is the key, and its marker says this test needs
    nothing that a plain run lacks.
    """
    crafted = "/../test_optional.py:1: needs the graph-ad extra"
    repo = _repo(tmp_path, detector="truncated", contract=_contract(crafted))
    code, out = _pytest(repo)
    assert code == 0, out
    assert crafted in out, "the crafted reason should reach the report unaltered"

    code, out = _check(repo)

    assert code == 1
    assert out.count("skipped, and the manifest says it needs nothing this run lacks") == 2


def test_collecting_without_running_no_longer_passes_on_a_forged_summary(tmp_path):
    """Bypass 5: --co runs nothing, and a print supplies the summary line the old guard read."""
    repo = _repo(tmp_path, detector="truncated", conftest=FORGE_SUMMARY)
    code, out = _pytest(repo, environment={"PYTEST_ADDOPTS": "--co -q"})
    assert code == 0, out
    assert "3 passed in 0.05s" in out, "the forged summary line should be in the log"

    code, out = _check(repo)

    assert code == 1
    assert out.count("absent from the run, so a contract did not run") == 3


def test_trading_a_contract_for_trivial_tests_no_longer_balances(tmp_path):
    """Bypass 7: collect_ignore drops a contract and two one-line tests restore the count.

    The terminal line is the one test_a_sound_repository_passes asserts for the sound repository,
    and the badge needs no edit. Counting tests cannot see this; naming them can.
    """
    repo = _repo(tmp_path, detector="truncated", conftest=DROP_THE_CONTRACT,
                 extra_tests={"test_extra.py": TRIVIA})
    code, out = _pytest(repo)
    assert code == 0, out
    assert "2 passed, 1 skipped" in out, "the attack is meant to reproduce the baseline line"

    code, out = _check(repo)

    assert code == 1
    assert out.count("absent from the run, so a contract did not run") == 2
    assert out.count("ran and is not declared in the manifest") == 2
    # The source view, which no pytest hook reaches: the contract still has its def, and the
    # trivia does not appear in the manifest. This is what catches the same trade laundered by
    # regenerating the manifest to match the degraded run.
    assert out.count("defined in tests/ and not declared in the manifest") == 2


def test_an_xfail_marker_does_not_pass_for_a_contract(tmp_path):
    """An xfail records that a contract is broken and lets the run stay green.

    pytest prints XFAIL rather than SKIPPED, so a skip-only check never saw it. junit-xml files it
    as a skip of type pytest.xfail, and this check refuses every one of them.
    """
    repo = _repo(tmp_path, detector="truncated", contract=XFAILED_CONTRACT)
    code, out = _pytest(repo)
    assert code == 0, out
    assert "1 xfailed" in out
    assert "XFAIL tests/test_contract.py::test_scores_every_node" in out

    code, out = _check(repo)

    assert code == 1
    assert "a test reported xfailed: tests/test_contract.py::test_scores_every_node" in out


def test_a_stale_xfail_marker_over_a_passing_test_reads_as_a_pass(tmp_path):
    """The one thing the terminal carried that junit-xml does not, recorded rather than implied.

    pytest writes a non-strict XPASS as a plain pass, so this check cannot flag the stale marker
    the way a grep for XPASS could. What that costs is small: the assertion ran and passed, so no
    contract stopped asserting, and the run where it does break reports xfailed, which the test
    above shows is refused. A strict xpass lands under <skipped> and is refused too.
    """
    repo = _repo(tmp_path, contract=XPASSING_CONTRACT)
    code, out = _pytest(repo)
    assert code == 0, out
    assert "1 xpassed" in out

    code, out = _check(repo)

    assert code == 0, out


def test_quarantining_a_whole_file_is_caught(tmp_path):
    """A module-level skip removes every test in the file and leaves pytest exiting 0."""
    quarantined = ('import pytest\n\npytest.skip("quarantined", allow_module_level=True)\n\n'
                   + CONTRACT_BODY)
    repo = _repo(tmp_path, detector="truncated", contract=quarantined)
    code, out = _pytest(repo)
    assert code == 0, out

    code, out = _check(repo)

    assert code == 1
    assert "a test file did not collect, so none of its tests ran" in out
    assert out.count("absent from the run, so a contract did not run") == 2


def test_a_gutted_test_body_is_still_not_caught(tmp_path):
    """Bypass 2's floor, stated as an executable limit rather than as a caveat in a comment.

    The contract's body is ``pass``. The node id runs, reports ``passed``, and no check that reads
    a report can tell it from a test that asserted. Nothing here is a defect in the check; it is
    where a report stops carrying the answer, and it stays a review responsibility.
    """
    repo = _repo(tmp_path, detector="truncated", contract=GUTTED_CONTRACT)
    code, out = _pytest(repo)
    assert code == 0, out

    code, out = _check(repo)

    assert code == 0, out


def test_a_swapped_callable_is_still_not_caught_when_it_stays_in_the_module(tmp_path):
    """Bypass 2 proper: the conftest points the contract at another function in the same file."""
    repo = _repo(tmp_path, detector="truncated", conftest=SWAP_INSIDE_THE_MODULE)
    code, out = _pytest(repo)
    assert code == 0, out

    code, out = _check(repo)

    assert code == 0, out


def test_a_swapped_callable_is_caught_when_it_moves_out_of_the_module(tmp_path):
    """The easiest form of bypass 2, which is noisy rather than closed.

    ``item.obj = lambda ...`` puts the callable in conftest.py, and junit-xml records where the
    callable it ran was defined. The node id then no longer sits under the module it names, and the
    declared one is missing from the run. An attacker who reads this test moves the replacement
    into the test module, which the test above shows is not caught, so this catches carelessness
    rather than intent.
    """
    repo = _repo(tmp_path, detector="truncated", conftest=SWAP_FOR_A_LAMBDA)
    code, out = _pytest(repo)
    assert code == 0, out

    code, out = _check(repo)

    assert code == 1
    assert "the callable is not the one that file declares" in out
    assert "absent from the run, so a contract did not run" in out


def test_the_workflow_checks_every_pytest_run_it_makes():
    """An orphaned checker would leave every test above green and CI enforcing nothing."""
    # Continuation lines first, so one command is one line whether or not it was wrapped.
    lines = [line.strip() for line in WORKFLOW.read_text(encoding="utf-8")
             .replace("\\\n", " ").splitlines()]
    runs = [line for line in lines if line.startswith("pytest ")]
    checks = [line for line in lines if line.startswith("python tools/check_test_report.py")]

    assert len(runs) == 3, "seed 0, seed 1, and the graph-AD contract"
    assert len(checks) == 3, "every run is checked, or the check is decoration"
    for run in runs:
        assert "--junit-xml=" in run and "-o junit_family=xunit1" in run and "-o addopts=" in run
    assert sum("--only tests/test_graph_ad_scores_every_node.py --require-pass" in check
               for check in checks) == 1
    # The shapes this rewrite removed. A reader who greps for them should find nothing.
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "ALLOWED = {" not in text and "grep -qE" not in text


def test_the_manifest_declares_exactly_the_tests_this_repository_defines():
    """Catches a test added or removed without regenerating tests/expected_tests.txt.

    This half needs no pytest run at all: it compares the committed manifest against the ``def``
    statements in the test sources.
    """
    module = _checker_module()
    declared, problems = module.read_manifest(module.MANIFEST)

    assert problems == []
    assert module.check_sources(declared) == []
    assert len(declared) == len(set(declared))
