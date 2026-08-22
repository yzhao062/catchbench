"""The licence terms CatchBench redistributes under must travel as files, not as links.

Apache-2.0 section 4(a) and GPL-3.0 section 4 both require a copy of the licence to accompany
redistribution, and the MIT permission notice has to be included with the material it covers. A URL
pointing at someone else's server satisfies none of these, so these tests check the local files
rather than the pointers.

The last group tests the coverage of the check itself. An audit found the earlier version inspecting
a hardcoded {MIT, Apache-2.0} set while the documentation claimed it covered every declared licence,
which let nine GPL-3.0 and one CC-BY-4.0 declarations pass with no text in the tree. The check now
enumerates the licence values the data actually declares, and a value it has no rule for has to fail
by name rather than pass silently.
"""
import json
import re
import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import emit_third_party_notices as etpn  # noqa: E402


# Apache-2.0 section 4(a), the clause that a pointer cannot satisfy.
APACHE_REDISTRIBUTION_CLAUSE = (
    "(a) You must give any other recipients of the Work or\n"
    "          Derivative Works a copy of this License; and"
)

# GPL-3.0 section 4, the same obligation for the copyleft records the release keeps.
GPL_REDISTRIBUTION_CLAUSE = (
    "keep intact all notices of the absence of any warranty; and give all\n"
    "recipients a copy of this License along with the Program."
)

# CC-BY-4.0 section 3(a)(1)(A), the attribution obligation for the one CC-BY record.
CC_BY_ATTRIBUTION_CLAUSE = (
    "            a. retain the following if it is supplied by the Licensor\n"
    "               with the Licensed Material:"
)

PRE_CORPUS_FILES = ("crewai", "injecagent", "mcp", "n8n", "sweagent", "synthetic")


@pytest.fixture
def licenses_copy(tmp_path, monkeypatch):
    """A writable copy of third_party/licenses, wired into the module under test."""
    copy = tmp_path / "licenses"
    shutil.copytree(etpn.LICENSE_DIR, copy)
    monkeypatch.setattr(etpn, "LICENSE_DIR", copy)
    return copy


@pytest.fixture
def corpus_copy(tmp_path):
    """A writable root holding the committed PRE records and the files that carry their terms.

    The check reads its licence enumeration out of these records, so injecting a declaration here is
    how a test asks what the check does with a licence nobody anticipated.
    """
    root = tmp_path / "repo"
    (root / "data" / "pre").mkdir(parents=True)
    for name in PRE_CORPUS_FILES:
        shutil.copy(ROOT / "data" / "pre" / f"{name}.json", root / "data" / "pre" / f"{name}.json")
    for name in ("LICENSE", "THIRD_PARTY_LICENSES.md"):
        shutil.copy(ROOT / name, root / name)
    return root


def _rewrite_declaration(root: Path, source: str, replaced: str, license_id: str) -> None:
    path = root / "data" / "pre" / f"{source}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    row = next(row for row in rows if row["provenance"]["license"] == replaced)
    row["provenance"]["license"] = license_id
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def test_every_declared_licence_has_local_text():
    assert etpn.check(ROOT) == 0


def test_the_declared_set_is_exactly_what_the_repository_carries_or_records():
    """The declared values and the rules that classify them must not drift apart."""
    declared = set(etpn.declared_license_records(etpn.declared_projects(ROOT)))
    known = set(etpn.CANONICAL_LICENSES) | etpn.PER_PROJECT_LICENSES | etpn.NON_DECLARATIONS

    assert declared <= known
    assert {"GPL-3.0", "CC-BY-4.0"} <= declared


@pytest.mark.parametrize("license_id", sorted(etpn.CANONICAL_LICENSES))
def test_each_canonical_copy_carries_its_own_terms(license_id):
    spec = etpn.CANONICAL_LICENSES[license_id]
    text = (ROOT / "third_party" / "licenses" / spec.filename).read_text(encoding="utf-8")

    assert spec.marker in text


def test_the_apache_copy_carries_the_redistribution_clause():
    text = (ROOT / "third_party" / "licenses" / "Apache-2.0.txt").read_text(encoding="utf-8")

    assert "Version 2.0, January 2004" in text
    assert APACHE_REDISTRIBUTION_CLAUSE in text


def test_the_gpl_copy_carries_the_redistribution_clause():
    """The nine GPL-3.0 records stay in the release, so its section 4 obligation is carried."""
    text = (ROOT / "third_party" / "licenses" / "GPL-3.0.txt").read_text(encoding="utf-8")

    assert "Version 3, 29 June 2007" in text
    assert GPL_REDISTRIBUTION_CLAUSE in text


def test_the_cc_by_copy_carries_the_attribution_clause():
    text = (ROOT / "third_party" / "licenses" / "CC-BY-4.0.txt").read_text(encoding="utf-8")

    assert "Attribution 4.0 International" in text
    assert CC_BY_ATTRIBUTION_CLAUSE in text


@pytest.mark.parametrize("license_id", sorted(etpn.CANONICAL_LICENSES))
def test_every_record_of_a_canonical_licence_is_covered_by_that_one_copy(license_id):
    declared = etpn.declared_license_records(etpn.declared_projects(ROOT))

    assert declared.get(license_id, 0) > 0
    assert (ROOT / "third_party" / "licenses" /
            etpn.CANONICAL_LICENSES[license_id].filename).is_file()


@pytest.mark.parametrize("license_id", sorted(etpn.CANONICAL_LICENSES))
def test_the_index_attributes_every_project_declaring_a_canonical_licence(license_id):
    index = (ROOT / "third_party" / "licenses" / etpn.CANONICAL_INDEX_NAME).read_text(
        encoding="utf-8"
    )
    blocks = etpn._ENTRY.split(index)[1:]
    block = dict(zip(blocks[::2], blocks[1::2]))[license_id]
    listed = {row.group("project") for row in etpn._INDEX_ROW.finditer(block)}
    declared = {
        etpn._project_name(str(project["upstream_url"]))
        for project in etpn._projects_declaring(etpn.declared_projects(ROOT), license_id)
    }

    assert listed == declared


def test_the_index_records_the_digest_of_each_local_copy():
    """The recorded SHA-256 must be the digest of the file it names, not a stale copy of one."""
    index = (ROOT / "third_party" / "licenses" / etpn.CANONICAL_INDEX_NAME).read_text(
        encoding="utf-8"
    )
    blocks = etpn._ENTRY.split(index)[1:]
    sections = dict(zip(blocks[::2], blocks[1::2]))

    for license_id, spec in etpn.CANONICAL_LICENSES.items():
        recorded = etpn._LOCAL_SHA.search(sections[license_id])
        text = (ROOT / "third_party" / "licenses" / spec.filename).read_text(encoding="utf-8")

        assert recorded is not None, license_id
        assert recorded.group("sha") == etpn._digest(text), license_id


@pytest.mark.parametrize("source", sorted(etpn.MIT_SOURCES))
def test_each_mit_notice_carries_a_copyright_line_and_the_permission_notice(source):
    text = (ROOT / "third_party" / "licenses" / f"MIT-{source}.md").read_text(encoding="utf-8")
    blocks = etpn._ENTRY.split(text)[1:]
    found = dict(zip(blocks[::2], blocks[1::2]))
    declared = etpn.declared_projects(ROOT)[source]["MIT"]

    assert len(found) == len(declared)
    for name, block in found.items():
        body = etpn._FENCE.search(block)
        if body is None:
            assert etpn.UNRESOLVED in block, name
            continue
        assert etpn.MIT_MARKER in body.group("body"), name
        assert re.search(r"(?i)^copyright ", body.group("body"), re.MULTILINE), name


def test_every_source_declaring_mit_has_somewhere_to_carry_its_notice():
    """MIT names the holder, so a source declaring it needs its own carrier, not a shared copy."""
    grouped = etpn.declared_projects(ROOT)
    declaring = {source for source in grouped if grouped[source].get("MIT")}
    classified = set(etpn.MIT_SOURCES) | set(etpn.INLINE_MIT_SOURCES) | etpn.FIRST_PARTY_SOURCES

    assert declaring <= classified
    assert declaring == {"crewai", "mcp", "injecagent", "synthetic"}


def test_an_unmodified_copy_still_passes(licenses_copy):
    """Guards the fixture itself: without this, a later 'it fails' proves nothing."""
    assert etpn.check(ROOT) == 0


@pytest.mark.parametrize("license_id", sorted(etpn.CANONICAL_LICENSES))
def test_a_missing_canonical_copy_fails_the_check(licenses_copy, license_id):
    (licenses_copy / etpn.CANONICAL_LICENSES[license_id].filename).unlink()

    assert etpn.check(ROOT) == 1


@pytest.mark.parametrize("license_id", sorted(etpn.CANONICAL_LICENSES))
def test_a_canonical_copy_that_is_not_the_licence_fails_the_check(licenses_copy, license_id):
    path = licenses_copy / etpn.CANONICAL_LICENSES[license_id].filename
    path.write_text("see the upstream URL\n", encoding="utf-8")

    assert etpn.check(ROOT) == 1


def test_a_missing_canonical_index_fails_the_check(licenses_copy):
    (licenses_copy / etpn.CANONICAL_INDEX_NAME).unlink()

    assert etpn.check(ROOT) == 1


def test_a_dropped_attribution_row_fails_the_check(licenses_copy):
    """Removing a GPL-3.0 project from the index leaves its records unattributed."""
    path = licenses_copy / etpn.CANONICAL_INDEX_NAME
    text = path.read_text(encoding="utf-8")
    row = next(
        row for row in etpn._INDEX_ROW.finditer(text)
        if row.group("project") == "opahopa/crewai-factory-crew"
    )
    line_end = text.index("\n", row.end()) + 1
    path.write_text(text[: row.start()] + text[line_end:], encoding="utf-8")

    assert etpn.check(ROOT) == 1


def test_a_stale_recorded_digest_fails_the_check(licenses_copy):
    path = licenses_copy / etpn.CANONICAL_INDEX_NAME
    text = path.read_text(encoding="utf-8")
    hit = etpn._LOCAL_SHA.search(text)
    stale = f"- SHA-256 of the local copy: `{'0' * 64}`"
    path.write_text(text[: hit.start()] + stale + text[hit.end():], encoding="utf-8")

    assert etpn.check(ROOT) == 1


def test_the_check_survives_a_crlf_checkout(licenses_copy):
    """A Windows clone under core.autocrlf=true writes these files with CRLF.

    `.gitattributes` pins them to LF so the redistributed copy stays byte-identical to what the
    licence steward serves. The recorded digests must not depend on that pin having been applied,
    or the check would fail on a clone nobody edited.
    """
    for path in licenses_copy.iterdir():
        raw = path.read_bytes()
        path.write_bytes(raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))

    assert etpn.check(ROOT) == 0


def test_a_missing_mit_appendix_fails_the_check(licenses_copy):
    (licenses_copy / "MIT-crewai.md").unlink()

    assert etpn.check(ROOT) == 1


def test_a_dropped_project_notice_fails_the_check(licenses_copy):
    path = licenses_copy / "MIT-mcp.md"
    text = path.read_text(encoding="utf-8")
    first = etpn._ENTRY.search(text)
    following = etpn._ENTRY.search(text, first.end())
    path.write_text(text[: first.start()] + text[following.start() :], encoding="utf-8")

    assert etpn.check(ROOT) == 1


def test_a_notice_stripped_to_a_pointer_fails_the_check(licenses_copy):
    path = licenses_copy / "MIT-crewai.md"
    text = path.read_text(encoding="utf-8")
    body = etpn._FENCE.search(text)
    fence = body.group("fence")
    path.write_text(
        text[: body.start()] + f"{fence}text\nsee the pinned LICENSE link above\n{fence}"
        + text[body.end() :],
        encoding="utf-8",
    )

    assert etpn.check(ROOT) == 1


def test_a_drifted_record_count_fails_the_check(licenses_copy):
    path = licenses_copy / "MIT-mcp.md"
    text = path.read_text(encoding="utf-8")
    hit = etpn._COUNT.search(text)
    bumped = f"- CatchBench records: {int(hit.group('count')) + 1}"
    path.write_text(text[: hit.start()] + bumped + text[hit.end() :], encoding="utf-8")

    assert etpn.check(ROOT) == 1


def test_the_copied_corpus_still_passes(corpus_copy):
    """Guards the corpus fixture, so the injection tests below prove something."""
    assert etpn.check(corpus_copy) == 0


def test_a_record_declaring_an_unanticipated_licence_fails_the_check(corpus_copy, capsys):
    """The class fix. A licence with no rule must fail by name, not slip through a hardcoded set.

    EUPL-1.2 is chosen because nothing in this repository anticipates it: no canonical text, no
    per-project carrier, and no entry in the recorded non-declarations. A record that starts
    declaring it is exactly the case the old check could not see.
    """
    _rewrite_declaration(corpus_copy, "crewai", "NOASSERTION", "EUPL-1.2")

    assert etpn.check(corpus_copy) == 1
    assert "EUPL-1.2" in capsys.readouterr().err


def test_the_unanticipated_licence_is_reported_even_when_nothing_else_is_wrong(corpus_copy, capsys):
    """The failure has to be the unclassified value itself, not a count that drifted with it."""
    _rewrite_declaration(corpus_copy, "n8n", "unverified", "SSPL-1.0")

    assert etpn.check(corpus_copy) == 1
    problems = [line for line in capsys.readouterr().err.splitlines() if line.startswith("  ")]

    assert len(problems) == 1
    assert "SSPL-1.0" in problems[0]


def test_a_new_project_declaring_a_carried_licence_fails_until_it_is_attributed(corpus_copy, capsys):
    """A carried licence is not a blanket pass: the new project still has to reach the index."""
    _rewrite_declaration(corpus_copy, "sweagent", "unverified", "GPL-3.0")

    assert etpn.check(corpus_copy) == 1
    problems = capsys.readouterr().err

    assert "GPL-3.0 table omits" in problems
