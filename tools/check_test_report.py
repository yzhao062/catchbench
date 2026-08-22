"""Check a pytest run against the set of tests this repository declares it runs.

Every earlier version of this check read pytest's terminal summary and looked for bad lines in it.
An adversarial review broke that shape eight ways, six of which were the suite telling the check
what it wanted to hear: a ``pytest_sessionfinish`` hook that rewrites the exit status, a
``pytest_report_teststatus`` hook that relabels skips as passes, ``--no-summary``, a ``print`` that
forges the summary line, and ``collect_ignore`` trading a contract for two trivial tests at par.
None of those changed what the tests did. They changed what the report said, and the check had no
other view.

This check reads three things instead, and the first two are what a terminal summary does not
carry:

1. ``--junit-xml``, which names every test that ran, one element per node id, and carries the
   outcome as a child element rather than as a count. Relabelling a skip for the terminal does not
   reach it, ``--no-summary`` does not remove it, and a forged ``print`` is not in it.
2. ``tests/expected_tests.txt``, the committed list of node ids this repository claims to run. The
   comparison runs in both directions, so a contract that vanishes from the run is as loud as one
   that fails, and trading a contract for a trivial test no longer balances, because names are
   compared rather than a count.
3. The test sources, parsed with ``ast``. This is the one view no pytest hook can reach: a conftest
   can stop a test from running, but it cannot remove its ``def`` from the file. It catches a
   removal that was laundered by regenerating the manifest to match.

What no report-reading check can see, including this one: a test body rewritten to ``pass``, and a
conftest that swaps the test callable for another function in the same module. Both report an
honest ``passed`` for a real node id, so they stay a review responsibility, and this file says so
rather than implying coverage it does not have. The careless form of that swap,
``item.obj = lambda *a, **k: None``, is caught, because junit-xml records where the callable it ran
was defined and the lambda's home is conftest.py. That catches carelessness rather than intent.

Usage:

    python tools/check_test_report.py --report seed0.xml
    python tools/check_test_report.py --report graph_ad.xml \
        --only tests/test_graph_ad_scores_every_node.py --require-pass
    python tools/check_test_report.py --report local.xml --update
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "expected_tests.txt"
TESTS_DIR = ROOT / "tests"
README = ROOT / "README.md"

# What a test needs before it can pass, which is the only reason this repository tolerates a skip.
# Anything else that skips is a contract that stopped asserting.
MARKERS = {
    "always": "must pass wherever the suite runs",
    "needs-torch": "needs the graph-AD stack; may skip where torch is absent",
    "needs-paper": "needs CATCHBENCH_PAPER_DIR; may skip where the paper repository is absent",
}
# Outcomes that are never acceptable, whatever the marker says. An xfail is here because it does
# not assert its contract: it records that the contract is known to be broken and moves on.
NEVER_OK = ("failed", "error", "xfailed", "xpassed")
MAX_LISTED = 20


def node_id(file: str, classname: str, name: str) -> tuple[str, bool]:
    """Rebuild pytest's node id from a junit-xml testcase. Returns ``(node id, located)``.

    ``--junit-xml`` reports a dotted ``classname`` (``tests.test_x.TestGroup``) rather than a node
    id, and the dots are ambiguous on their own. The ``file`` attribute resolves them: it gives the
    module part exactly, so whatever follows it in ``classname`` is the class path. ``file`` is
    written only by the xunit1 family, which is why the workflow passes ``-o junit_family=xunit1``.

    ``classname`` comes from the node id and ``file`` comes from the code location of the callable
    pytest ran, so the two disagree when something replaced that callable with one defined in
    another file. ``located`` is False in that case, and the caller reports it.
    """
    file = file.replace("\\", "/")
    module = re.sub(r"\.py$", "", file).replace("/", ".")
    if classname == module:
        return f"{file}::{name}", True
    if classname.startswith(module + "."):
        return f"{file}::" + classname[len(module) + 1:].replace(".", "::") + f"::{name}", True
    # Neither shape. Report it dotted, so it lands as an undeclared node id rather than quietly
    # matching a manifest entry the run may not have honoured.
    return f"{classname}::{name}", False


def function_of(node: str) -> str:
    """The node id with any parametrization stripped, which is what a ``def`` in the source is."""
    return re.sub(r"\[.*\]$", "", node)


def read_report(path: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Map every reported node id to one outcome. Returns ``(outcomes, problems)``."""
    problems: list[tuple[str, str]] = []
    if not path.exists():
        return {}, [("no junit-xml report at this path, so nothing was checked", str(path))]
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return {}, [("the junit-xml report does not parse", f"{path}: {exc}")]

    cases = list(root.iter("testcase"))
    if any(case.get("classname") and not case.get("file") for case in cases):
        # The xunit2 family, which is pytest's default, drops the per-testcase file attribute, and
        # without it a dotted classname does not resolve to a node id. Say so once rather than
        # reporting every test as unreadable.
        problems.append(("the report has no per-testcase file attribute, so node ids cannot be "
                         "read out of it; run pytest with -o junit_family=xunit1", str(path)))
        return {}, problems

    outcomes: dict[str, str] = {}
    for case in cases:
        classname, name = case.get("classname", ""), case.get("name", "")
        file = case.get("file", "")
        if not classname:
            # pytest writes a bare module entry when a file never collected: an import error, or a
            # module-level skip. Every test in that file is then missing from the run, which the
            # two-way comparison below also catches, but naming the cause here saves a reader the
            # inference.
            problems.append(("a test file did not collect, so none of its tests ran",
                             file or name))
            continue
        kinds = {child.tag for child in case}
        if "error" in kinds:
            outcome = "error"
        elif "failure" in kinds:
            outcome = "failed"
        elif "skipped" in kinds:
            skipped = case.find("skipped")
            if skipped.get("type") == "pytest.xfail":
                outcome = "xfailed"
            elif skipped.get("message") == "xfail-marked test passes unexpectedly":
                # A strict xpass. pytest files it under <skipped> with no type, and the marker is
                # stale whichever way the assertion went. A non-strict xpass is written as a plain
                # pass and cannot be told apart here; the assertion did run and did pass, and the
                # run it breaks in reports xfailed, which the line above refuses.
                outcome = "xpassed"
            else:
                outcome = "skipped"
        else:
            outcome = "passed"
        node, located = node_id(file, classname, name)
        if not located:
            problems.append(("a test ran from a file other than the module its node id names, so "
                             "the callable is not the one that file declares",
                             f"{node} ran from {file}"))
        if "\n" in node or "\r" in node:
            problems.append(("a node id spans lines, so no line-based manifest can hold it",
                             node.replace("\n", "\\n").replace("\r", "\\r")))
            continue
        if node in outcomes:
            problems.append(("two testcases report the same node id", node))
        outcomes[node] = outcome
    return outcomes, problems


def read_manifest(path: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Map every declared node id to its marker."""
    problems: list[tuple[str, str]] = []
    if not path.exists():
        return {}, [("the expected-test manifest is missing", str(path))]
    declared: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        marker, _, node = line.partition(" ")
        node = node.strip()
        if marker not in MARKERS or not node:
            problems.append((f"{path.name}:{number} is not a marker in {sorted(MARKERS)} followed "
                             f"by a node id", line))
            continue
        if node in declared:
            problems.append((f"{path.name}:{number} declares a node id twice", node))
        declared[node] = marker
    return declared, problems


def write_manifest(path: Path, markers: dict[str, str]) -> None:
    header = [
        "# Every test this repository claims to run, one node id per line, checked by",
        "# tools/check_test_report.py against the junit-xml report of an actual pytest run.",
        "# CI fails when the run and this file disagree in either direction, so a contract that",
        "# quietly stops running is as loud as one that fails. The README test badge must equal",
        "# the number of entries here.",
        "#",
        "# The first column says what the test needs before it can pass, which is the only reason",
        "# a skip is tolerated here:",
    ]
    header += [f"#   {name:<12} {why}" for name, why in MARKERS.items()]
    header += [
        "#",
        "# Regenerate after adding or removing a test:",
        "#   pytest tests -q --junit-xml=local.xml -o junit_family=xunit1",
        "#   python tools/check_test_report.py --report local.xml --update",
        "# The update keeps the marker of every test already listed and gives a new one",
        "# 'always'; change that by hand when the test needs torch or the paper repository.",
    ]
    body = [f"{markers[node]:<12} {node}" for node in sorted(markers)]
    path.write_text("\n".join(header + body) + "\n", encoding="utf-8")


def _is_test_class(node: ast.ClassDef) -> bool:
    if node.name.startswith("Test"):
        return True
    for base in node.bases:
        name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
        if name.endswith("TestCase"):
            return True
    return False


def source_functions(tests_dir: Path, root: Path) -> tuple[set[str], list[tuple[str, str]]]:
    """Every test function defined under ``tests/``, read from the source rather than from a run.

    Module-level ``def test_*``, plus methods of a class pytest collects: named ``Test*``, or a
    ``unittest.TestCase`` subclass, which pytest collects whatever its name. Nothing here executes
    the file, so no conftest can change the answer.
    """
    problems: list[tuple[str, str]] = []
    found: set[str] = set()
    for path in sorted(tests_dir.glob("test_*.py")):
        rel = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            problems.append(("a test file does not parse", f"{rel}: {exc}"))
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test"):
                    found.add(f"{rel}::{node.name}")
            elif isinstance(node, ast.ClassDef) and _is_test_class(node):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if sub.name.startswith("test"):
                            found.add(f"{rel}::{node.name}::{sub.name}")
    return found, problems


def check(outcomes: dict[str, str], declared: dict[str, str], only: str | None,
          require_pass: bool) -> list[tuple[str, str]]:
    """Compare one run against the manifest: outcomes first, then membership both ways."""
    problems: list[tuple[str, str]] = []
    expected = declared
    if only:
        expected = {node: marker for node, marker in declared.items()
                    if node.split("::")[0] == only}
        if not expected:
            problems.append(("--only names a file the manifest does not declare", only))

    for node, outcome in sorted(outcomes.items()):
        if outcome in NEVER_OK:
            problems.append((f"a test reported {outcome}", node))

    missing = sorted(set(expected) - set(outcomes))
    extra = sorted(set(outcomes) - set(expected))
    for node in missing[:MAX_LISTED]:
        problems.append(("declared in the manifest and absent from the run, so a contract did not "
                         "run", node))
    if len(missing) > MAX_LISTED:
        problems.append((f"{len(missing) - MAX_LISTED} further declared test(s) did not run",
                         "tests/expected_tests.txt"))
    for node in extra[:MAX_LISTED]:
        problems.append(("ran and is not declared in the manifest; regenerate it with --update so "
                         "the change is reviewable", node))
    if len(extra) > MAX_LISTED:
        problems.append((f"{len(extra) - MAX_LISTED} further undeclared test(s) ran",
                         "tests/expected_tests.txt"))

    for node, marker in sorted(expected.items()):
        if outcomes.get(node) != "skipped":
            continue
        if marker == "always":
            problems.append(("skipped, and the manifest says it needs nothing this run lacks",
                             node))
        elif require_pass:
            problems.append((f"skipped in a run where every test must pass, and it is marked "
                             f"{marker}", node))
    return problems


def check_sources(declared: dict[str, str], tests_dir: Path = TESTS_DIR,
                  root: Path = ROOT) -> list[tuple[str, str]]:
    """The manifest must declare exactly the test functions the source files define."""
    defined, problems = source_functions(tests_dir, root)
    listed = {function_of(node) for node in declared}
    for name in sorted(defined - listed)[:MAX_LISTED]:
        problems.append(("defined in tests/ and not declared in the manifest", name))
    for name in sorted(listed - defined)[:MAX_LISTED]:
        problems.append(("declared in the manifest and defined nowhere in tests/", name))
    return problems


def _check_canary(node: str, outcomes: dict[str, str], problems: list[tuple[str, str]],
                  report: Path) -> int:
    """The canary run had to fail, and this node id had to be what failed.

    Everything else this file does reads a report the suite's own process wrote, and two adversarial
    rounds established that such a report can be made to say anything. This is the one check that
    does not depend on the report being honest to be useful. A conftest that turns failures into
    passes turns the canary into a pass too, and a pass here is the error.

    The manifest is deliberately not consulted: the canary is not one of the tests this repository
    claims to run, and it must stay out of the declared set and out of the badge.
    """
    reported = outcomes.get(node)
    if reported is None:
        problems.append((f"the canary run did not report {node}, so it proved nothing about "
                         "whether this suite can still report a failure", str(report)))
    elif reported != "failed":
        problems.append((f"the canary reported {reported!r} and must report 'failed'; a suite that "
                         "cannot report this failure cannot be trusted to report a real one", node))
    extra = sorted(set(outcomes) - {node})
    for other in extra[:MAX_LISTED]:
        problems.append(("the canary run collected a test that is not the canary", other))

    for why, where in problems:
        print(f"::error::{why}: {where}")
    if problems:
        print(f"{len(problems)} problem(s) against the canary")
        return 1
    print(f"canary: {node} reported failed, so this suite still reports failures")
    return 0


def check_badge(declared: dict[str, str], readme: Path = README) -> list[tuple[str, str]]:
    """The README test badge is hand-typed and nothing else recomputes it.

    It is compared against the manifest rather than against a count parsed out of pytest's summary.
    The manifest is what the badge claims to count, a reader can check it with ``wc -l``, and a
    forged summary line cannot move it. Counting the summary was also how failed and skipped tests
    used to satisfy the badge arithmetic.
    """
    badge = re.search(r"badge/tests-(\d+)-", readme.read_text(encoding="utf-8"))
    if not badge:
        return [("the README test badge is missing or no longer parseable", readme.name)]
    if int(badge.group(1)) != len(declared):
        return [(f"the README test badge says {badge.group(1)} but the manifest declares "
                 f"{len(declared)}", readme.name)]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", required=True, type=Path,
                        help="the junit-xml file pytest wrote for this run")
    parser.add_argument("--only", metavar="TESTS/FILE.PY",
                        help="the run covered one file; check that file's declared tests only")
    parser.add_argument("--require-pass", action="store_true",
                        help="every test in scope must pass; no skip is tolerated")
    parser.add_argument("--update", action="store_true",
                        help="rewrite the manifest from this report, then check it")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--tests-dir", type=Path, default=TESTS_DIR)
    parser.add_argument("--readme", type=Path, default=README)
    parser.add_argument("--no-badge", action="store_true",
                        help="skip the README badge comparison")
    parser.add_argument("--expect-failed", metavar="NODEID",
                        help="canary mode: this run had to fail, and this node id had to be the "
                             "test that failed; the manifest and badge are not consulted")
    args = parser.parse_args(argv)

    outcomes, problems = read_report(args.report)

    if args.expect_failed:
        return _check_canary(args.expect_failed, outcomes, problems, args.report)

    if args.update and not outcomes:
        # Without this, a run that collected nothing rewrites the manifest to empty and takes the
        # record of what this repository runs with it.
        problems.append(("refusing to rewrite the manifest from a report that names no test",
                         str(args.report)))
    elif args.update:
        previous, _ = read_manifest(args.manifest)
        markers = {node: previous.get(node, "always") for node in outcomes}
        write_manifest(args.manifest, markers)
        added = sorted(set(markers) - set(previous))
        dropped = sorted(set(previous) - set(markers))
        print(f"wrote {args.manifest}: {len(markers)} test(s), {len(added)} added, "
              f"{len(dropped)} removed")
        for node in added[:MAX_LISTED]:
            print(f"  added   {markers[node]:<12} {node}")
        for node in dropped[:MAX_LISTED]:
            print(f"  removed {previous[node]:<12} {node}")

    declared, manifest_problems = read_manifest(args.manifest)
    problems += manifest_problems
    if declared:
        problems += check(outcomes, declared, args.only, args.require_pass)
        problems += check_sources(declared, args.tests_dir, args.tests_dir.parent)
        if not args.only and not args.no_badge:
            problems += check_badge(declared, args.readme)
    if not outcomes and not problems:
        problems.append(("the report names no test, so this check proved nothing",
                         str(args.report)))

    for why, where in problems:
        print(f"::error::{why}: {where}")
    if problems:
        print(f"{len(problems)} problem(s) against {args.manifest}")
        return 1

    counted = {outcome: sum(1 for value in outcomes.values() if value == outcome)
               for outcome in sorted(set(outcomes.values()))}
    # The allowance is printed every run. A test moves out of `always` by a one-line edit to the
    # manifest, and this line is where that shows up in a log rather than only in a diff.
    in_scope = {node: marker for node, marker in declared.items()
                if not args.only or node.split("::")[0] == args.only}
    allowed = sum(1 for marker in in_scope.values() if marker != "always")
    allowance = ("no test in scope may skip" if args.require_pass
                 else f"{allowed} of {len(in_scope)} declared as allowed to skip")
    scope = args.only or "tests"
    print(f"{scope}: {len(outcomes)} test(s) reported, matching {args.manifest.name} exactly ("
          + ", ".join(f"{count} {name}" for name, count in counted.items()) + f"); {allowance}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
