"""Carry the third-party licence terms that CatchBench redistributes under.

A provenance link is not a delivered licence. Apache-2.0 section 4(a) requires a copy of the licence
to accompany redistribution, GPL-3.0 section 4 requires the same, CC-BY-4.0 section 3(a)(1) requires
the licence or its URI to accompany the material, and the MIT permission notice has to be included
with the material it covers. The text has to exist as files in this repository rather than as URLs
pointing at someone else's server.

Every value that appears as ``provenance.license`` in the committed PRE records falls into exactly
one of three classes, and ``--check`` fails on any value that falls into none of them:

* ``CANONICAL_LICENSES`` are licences whose text is identical for every project that declares them,
  so one copy fetched from the licence steward's own server covers every record. ``Apache-2.0.txt``,
  ``GPL-3.0.txt``, and ``CC-BY-4.0.txt`` are those copies. ``CANONICAL-TEXTS.md`` records, for each
  one, where it was retrieved, the SHA-256 of the local copy, and every upstream project that
  declares it.
* ``PER_PROJECT_LICENSES`` are licences whose text names the copyright holder, so one project's copy
  cannot stand in for another's. MIT is the only one. ``MIT-<source>.md`` reproduces the notice of
  each pinned upstream project in that source set; the two single-project sources are handled by
  ``INLINE_MIT_SOURCES`` and ``FIRST_PARTY_SOURCES``.
* ``NON_DECLARATIONS`` are the values that record the *absence* of a declaration. They grant
  nothing, so there is no text to carry, and THIRD_PARTY_LICENSES.md marks the records that carry
  them unfinished.

The check enumerates rather than hardcodes because of a defect found by audit. An earlier version
inspected a fixed {MIT, Apache-2.0} set while THIRD_PARTY_LICENSES.md advertised that the check
covered every declared licence, so the nine GPL-3.0 and one CC-BY-4.0 declarations passed a green
check with no licence text anywhere in the tree. A gate that overstates its own coverage is worse
than no gate, because it converts an unexamined risk into a documented all-clear. A licence value
nobody anticipated now fails the check by name instead of passing silently.

``--fetch`` needs the network and rewrites every generated file from the pinned upstream revisions.
``--check`` needs neither the network nor an upstream, and is what the test suite runs.

Usage:

    python tools/emit_third_party_notices.py --check
    python tools/emit_third_party_notices.py --fetch
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emit_asset_manifest import ManifestError, _pre_source_sets  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
LICENSE_DIR = ROOT / "third_party" / "licenses"
CANONICAL_INDEX_NAME = "CANONICAL-TEXTS.md"


class CanonicalText(NamedTuple):
    """One licence whose text is the same for every project that declares it."""

    filename: str
    url: str
    marker: str


# Retrieved from the body that publishes the licence, not from a project that happens to use it, so
# the copy carried here is the steward's own text. Each marker is a phrase that pins the version:
# "GNU GENERAL PUBLIC LICENSE" alone would accept the version 2 text for a version 3 declaration.
CANONICAL_LICENSES: dict[str, CanonicalText] = {
    "Apache-2.0": CanonicalText(
        "Apache-2.0.txt",
        "https://www.apache.org/licenses/LICENSE-2.0.txt",
        "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION",
    ),
    "GPL-3.0": CanonicalText(
        "GPL-3.0.txt",
        "https://www.gnu.org/licenses/gpl-3.0.txt",
        "Version 3, 29 June 2007",
    ),
    "CC-BY-4.0": CanonicalText(
        "CC-BY-4.0.txt",
        "https://creativecommons.org/licenses/by/4.0/legalcode.txt",
        "Creative Commons Attribution 4.0 International Public License",
    ),
}

# MIT's text carries the copyright holder's name, so it cannot be collapsed into one shared copy.
PER_PROJECT_LICENSES = {"MIT"}

# Sources whose records carry per-project MIT declarations, each generated into its own appendix.
MIT_SOURCES = {
    "crewai": "public CrewAI projects",
    "mcp": "Model Context Protocol Registry servers",
}

# Sources that declare MIT for a single upstream project whose notice is reproduced inline in
# THIRD_PARTY_LICENSES.md rather than in a generated appendix. The value is the copyright line that
# has to be present there, so deleting the inline block fails the check instead of going unnoticed.
INLINE_MIT_SOURCES = {"injecagent": "Copyright (c) 2023 Qiusi Zhan"}

# Sources whose MIT declaration is CatchBench's own licence on CatchBench-authored records.
FIRST_PARTY_SOURCES = {"synthetic"}

# Values that record the absence of a declaration rather than a grant. `unverified` marks a source
# whose terms were never established; `NOASSERTION` and `Other` are what the upstream registry
# returned. None of the three is a licence, so none of them has text to carry, and the records that
# carry them are marked unfinished in THIRD_PARTY_LICENSES.md rather than cleared here.
NON_DECLARATIONS = {"NOASSERTION", "Other", "unverified", "UNRESOLVED"}

MIT_MARKER = "Permission is hereby granted"
UNRESOLVED = "UNRESOLVED, because the pinned licence file could not be retrieved."

CANDIDATE_FILENAMES = (
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "LICENCE",
    "LICENCE.md",
    "LICENCE.txt",
    "license",
    "license.md",
    "COPYING",
)
USER_AGENT = "catchbench-third-party-notice-generator"
TIMEOUT = 30

_ENTRY = re.compile(r"^### (?P<project>\S+)$", re.MULTILINE)
_COUNT = re.compile(r"^- CatchBench records: (?P<count>\d+)$", re.MULTILINE)
_LOCAL_SHA = re.compile(r"^- SHA-256 of the local copy: `(?P<sha>[0-9a-f]{64})`$", re.MULTILINE)
_FENCE = re.compile(r"^(?P<fence>`{3,})text\n(?P<body>.*?)\n(?P=fence)$", re.MULTILINE | re.DOTALL)
_INDEX_ROW = re.compile(
    r"^\| \[(?P<project>[^\]]+)\]\([^)]*\) \| `(?P<identifier>[^`]+)` \| (?P<count>\d+) \|",
    re.MULTILINE,
)


class NoticeError(RuntimeError):
    """The redistributed terms are not carried completely and accurately."""


def _appendix_path(source: str) -> Path:
    return LICENSE_DIR / f"MIT-{source}.md"


def _canonical_path(license_id: str) -> Path:
    return LICENSE_DIR / CANONICAL_LICENSES[license_id].filename


def _canonical_index_path() -> Path:
    return LICENSE_DIR / CANONICAL_INDEX_NAME


def _project_name(upstream_url: str) -> str:
    return "/".join(upstream_url.rstrip("/").split("/")[-2:])


def _digest(text: str) -> str:
    """SHA-256 of the text with line endings normalised.

    A Windows clone under ``core.autocrlf=true`` checks these files out with CRLF, so hashing the
    bytes on disk would make the recorded digest depend on the reader's git configuration rather
    than on the licence. `.gitattributes` pins the files to LF as well, which keeps the redistributed
    copy byte-identical to what the steward serves; this normalisation keeps the check correct even
    where that pin has not been applied.
    """
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def declared_projects(root: Path = ROOT) -> dict[str, dict[str, list[dict[str, object]]]]:
    """Group each source's upstream projects by declared licence.

    Reads the same committed PRE records the asset manifest reads, so the notice set and the
    manifest cannot disagree about which upstream projects a licence was declared for. Projects are
    keyed on URL and identifier together, because one URL can carry many identifiers: every n8n
    template shares a single gallery URL.
    """
    source_sets, _ = _pre_source_sets(root)
    grouped: dict[str, dict[str, list[dict[str, object]]]] = {}
    for source, source_set in source_sets.items():
        by_license: dict[str, dict[tuple[str, str], dict[str, object]]] = {}
        for entry in source_set["upstream_identifiers"]:
            projects = by_license.setdefault(entry["declared_license"], {})
            key = (entry["upstream_url"], entry["identifier"])
            project = projects.setdefault(
                key,
                {
                    "upstream": entry["upstream"],
                    "upstream_url": entry["upstream_url"],
                    "identifier": entry["identifier"],
                    "record_count": 0,
                },
            )
            project["record_count"] += entry["record_count"]
        grouped[source] = {
            license_id: [projects[key] for key in sorted(projects, key=lambda k: (k[0].casefold(), k[1]))]
            for license_id, projects in by_license.items()
        }
    return grouped


def declared_license_records(
    grouped: dict[str, dict[str, list[dict[str, object]]]],
) -> dict[str, int]:
    """Every distinct declared licence value, with the number of records that declare it.

    This is the enumeration the check is built on. Nothing here is hardcoded, so a source that
    starts declaring a value no one anticipated appears the first time the check runs.
    """
    counts: Counter[str] = Counter()
    for source in grouped.values():
        for license_id, projects in source.items():
            counts[license_id] += sum(int(project["record_count"]) for project in projects)
    return dict(counts)


def _projects_declaring(
    grouped: dict[str, dict[str, list[dict[str, object]]]], license_id: str
) -> list[dict[str, object]]:
    """Every upstream project declaring one licence, merged across the sources that hold it."""
    merged: dict[tuple[str, str], dict[str, object]] = {}
    for source in sorted(grouped):
        for project in grouped[source].get(license_id, []):
            key = (str(project["upstream_url"]), str(project["identifier"]))
            entry = merged.setdefault(key, {**project, "record_count": 0})
            entry["record_count"] = int(entry["record_count"]) + int(project["record_count"])
    return [merged[key] for key in sorted(merged, key=lambda k: (k[0].casefold(), k[1]))]


def _sources_declaring(
    grouped: dict[str, dict[str, list[dict[str, object]]]], license_id: str
) -> set[str]:
    return {source for source, by_license in grouped.items() if by_license.get(license_id)}


def _raw_urls(upstream_url: str, identifier: str) -> list[tuple[str, str]]:
    name = _project_name(upstream_url)
    return [
        (filename, f"https://raw.githubusercontent.com/{name}/{identifier}/{filename}")
        for filename in CANDIDATE_FILENAMES
    ]


def _get(url: str) -> str | None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None


def _retrieve_license(project: dict[str, object], marker: str) -> tuple[str | None, str | None]:
    """The project's own licence file at its pinned revision, or (None, None) if none matched."""
    for candidate, raw_url in _raw_urls(str(project["upstream_url"]), str(project["identifier"])):
        text = _get(raw_url)
        if text is not None and marker in text:
            return candidate, text.replace("\r\n", "\n")
    return None, None


def _fence_for(text: str) -> str:
    """A fence at least one backtick longer than any run inside the reproduced text."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def _render_entry(project: dict[str, object], filename: str | None, text: str | None) -> str:
    lines = [f"### {_project_name(str(project['upstream_url']))}", ""]
    if text is None:
        lines += [
            f"- Pinned revision: `{project['identifier']}`",
            f"- CatchBench records: {project['record_count']}",
            f"- Status: {UNRESOLVED} The records derived from this project stay unfinished.",
            "",
        ]
        return "\n".join(lines)
    blob = f"{project['upstream_url']}/blob/{project['identifier']}/{filename}"
    fence = _fence_for(text)
    lines += [
        f"- Pinned licence file: [{filename}]({blob})",
        f"- CatchBench records: {project['record_count']}",
        f"- SHA-256 of the retrieved file: `{_digest(text)}`",
        "",
        f"{fence}text",
        text.rstrip("\n"),
        fence,
        "",
    ]
    return "\n".join(lines)


def _render_appendix(entries: list[str], display_name: str) -> str:
    header = [
        f"# MIT notices for {display_name}",
        "",
        "Generated by `tools/emit_third_party_notices.py --fetch`. Do not edit by hand.",
        "",
        "Each block below is the MIT notice of one upstream project, reproduced verbatim from that",
        "project's licence file at the revision CatchBench derived its records from. MIT requires the",
        "copyright notice and the permission notice to travel with the material, so these are carried",
        "here as files rather than as links. A project marked UNRESOLVED could not be retrieved at its",
        "pinned revision and its records remain unfinished for redistribution; see",
        "[`THIRD_PARTY_LICENSES.md`](../../THIRD_PARTY_LICENSES.md).",
        "",
        f"Projects: {len(entries)}.",
        "",
    ]
    return "\n".join(header) + "\n" + "\n".join(entries).rstrip("\n") + "\n"


def _render_canonical_section(
    license_id: str,
    spec: CanonicalText,
    text: str,
    rows: list[tuple[dict[str, object], str | None, bool]],
) -> str:
    records = sum(int(project["record_count"]) for project, _, _ in rows)
    lines = [
        f"### {license_id}",
        "",
        f"- Local copy: [`{spec.filename}`]({spec.filename})",
        f"- Retrieved from: <{spec.url}>",
        f"- SHA-256 of the local copy: `{_digest(text)}`",
        f"- CatchBench records: {records}",
        "",
        "| Upstream project | Pinned revision | Records | SHA-256 of that project's licence file |"
        " Byte-identical to the local copy |",
        "|---|---|---|---|---|",
    ]
    for project, project_sha, identical in rows:
        name = _project_name(str(project["upstream_url"]))
        digest = f"`{project_sha}`" if project_sha else UNRESOLVED.rstrip(".")
        same = "-" if project_sha is None else ("yes" if identical else "no")
        lines.append(
            f"| [{name}]({project['upstream_url']}) | `{project['identifier']}` |"
            f" {project['record_count']} | {digest} | {same} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_canonical_index(sections: list[str]) -> str:
    header = [
        "# Canonical licence texts carried locally",
        "",
        "Generated by `tools/emit_third_party_notices.py --fetch`. Do not edit by hand.",
        "",
        "Each licence below has one text that is the same for every project declaring it, so a single",
        "copy in this directory covers every record. The copy is retrieved from the body that",
        "publishes the licence rather than from a project that uses it. The table under each licence",
        "is the attribution: it names every pinned upstream project whose records declare that",
        "licence, so a reader can go from a licence to the projects it covers without opening",
        "[`ASSET_MANIFEST.json`](../../ASSET_MANIFEST.json).",
        "",
        "The last two columns record what the project's own licence file contained at its pinned",
        "revision. `Byte-identical to the local copy` says whether that file matched the steward's",
        "text exactly after line endings were normalised; `no` means the project shipped the same",
        "licence with cosmetic differences such as tabs or a typo, which is common and changes",
        "nothing about the terms. A row marked UNRESOLVED could not be retrieved at its pinned",
        "revision, and its records remain unfinished for redistribution; see",
        "[`THIRD_PARTY_LICENSES.md`](../../THIRD_PARTY_LICENSES.md).",
        "",
        f"Licences carried: {len(sections)}.",
        "",
    ]
    return "\n".join(header) + "\n" + "\n".join(sections).rstrip("\n") + "\n"


def fetch(root: Path = ROOT) -> int:
    grouped = declared_projects(root)
    LICENSE_DIR.mkdir(parents=True, exist_ok=True)

    sections = []
    for license_id, spec in CANONICAL_LICENSES.items():
        projects = _projects_declaring(grouped, license_id)
        if not projects:
            continue
        text = _get(spec.url)
        if text is None or spec.marker not in text:
            raise NoticeError(f"could not retrieve the canonical {license_id} text from {spec.url}")
        text = text.replace("\r\n", "\n")
        path = _canonical_path(license_id)
        path.write_text(text, encoding="utf-8", newline="\n")

        rows, unresolved = [], []
        for project in projects:
            _, body = _retrieve_license(project, spec.marker)
            if body is None:
                unresolved.append(_project_name(str(project["upstream_url"])))
            rows.append((project, _digest(body) if body else None, body == text))
        sections.append(_render_canonical_section(license_id, spec, text, rows))
        records = sum(int(project["record_count"]) for project in projects)
        print(f"wrote {path} (covers {records} {license_id} records, {len(projects)} projects)")
        for name in unresolved:
            print(f"  unresolved: {name}", file=sys.stderr)

    index = _canonical_index_path()
    index.write_text(_render_canonical_index(sections), encoding="utf-8", newline="\n")
    print(f"wrote {index} ({len(sections)} canonical licences)")

    for source, display_name in MIT_SOURCES.items():
        entries, unresolved = [], []
        for project in grouped[source].get("MIT", []):
            filename, body = _retrieve_license(project, MIT_MARKER)
            if body is None:
                unresolved.append(_project_name(str(project["upstream_url"])))
            entries.append(_render_entry(project, filename, body))
        path = _appendix_path(source)
        path.write_text(_render_appendix(entries, display_name), encoding="utf-8", newline="\n")
        print(f"wrote {path} ({len(entries)} projects, {len(unresolved)} unresolved)")
        for name in unresolved:
            print(f"  unresolved: {name}", file=sys.stderr)
    return 0


def _records(count: int) -> str:
    return f"{count} record{'' if count == 1 else 's'}"


def _check_unclassified(declared: dict[str, int]) -> list[str]:
    """The class fix: a declared value this repository has no rule for must fail by name."""
    known = set(CANONICAL_LICENSES) | PER_PROJECT_LICENSES | NON_DECLARATIONS
    return [
        f"{license_id!r} is declared by {_records(declared[license_id])}, and this repository "
        f"neither carries licence text for it nor records it as a non-declaration; classify it in "
        f"CANONICAL_LICENSES, PER_PROJECT_LICENSES, or NON_DECLARATIONS in "
        f"tools/emit_third_party_notices.py"
        for license_id in sorted(set(declared) - known)
    ]


def _check_canonical(
    grouped: dict[str, dict[str, list[dict[str, object]]]], declared: dict[str, int]
) -> tuple[list[str], list[str]]:
    """The shared texts, their recorded digests, and the attribution table for each."""
    problems: list[str] = []
    carried: list[str] = []
    index_path = _canonical_index_path()
    index_text = index_path.read_text(encoding="utf-8") if index_path.is_file() else None
    if index_text is None and any(declared.get(l) for l in CANONICAL_LICENSES):
        problems.append(f"{index_path.name} is missing, so no canonical text is attributed")
    blocks = _ENTRY.split(index_text)[1:] if index_text else []
    sections = dict(zip(blocks[::2], blocks[1::2]))
    if len(sections) * 2 != len(blocks):
        problems.append(f"{index_path.name} repeats a licence heading")

    for license_id, spec in CANONICAL_LICENSES.items():
        records = declared.get(license_id, 0)
        path = _canonical_path(license_id)
        if not records:
            if license_id in sections:
                problems.append(
                    f"{index_path.name} carries a {license_id} section, which no record declares"
                )
            continue

        text = None
        if not path.is_file():
            problems.append(
                f"{license_id} is declared by {_records(records)} but {spec.filename} is missing"
            )
        else:
            text = path.read_text(encoding="utf-8")
            if spec.marker not in text:
                problems.append(f"{spec.filename} does not contain the {license_id} terms")
                text = None

        block = sections.get(license_id)
        if block is None:
            problems.append(f"{index_path.name} carries no {license_id} section")
            continue

        recorded = _LOCAL_SHA.search(block)
        if recorded is None:
            problems.append(f"{index_path.name}: the {license_id} section records no SHA-256")
        elif text is not None and recorded.group("sha") != _digest(text):
            problems.append(
                f"{index_path.name}: the recorded {license_id} SHA-256 is not the digest of "
                f"{spec.filename}"
            )

        count = _COUNT.search(block)
        if count is None or int(count.group("count")) != records:
            reported = count.group("count") if count else "no"
            problems.append(
                f"{index_path.name}: {license_id} reports {reported} records; the data has {records}"
            )

        projects = _projects_declaring(grouped, license_id)
        expected = {_project_name(str(p["upstream_url"])): p for p in projects}
        if len(expected) != len(projects):
            problems.append(f"two {license_id} upstream URLs share one project name")
        listed = {row.group("project"): row for row in _INDEX_ROW.finditer(block)}
        for name in sorted(set(expected) - set(listed)):
            problems.append(f"{index_path.name}: the {license_id} table omits {name}")
        for name in sorted(set(listed) - set(expected)):
            problems.append(
                f"{index_path.name}: the {license_id} table lists {name}, which declares no "
                f"{license_id} record"
            )
        for name in sorted(set(expected) & set(listed)):
            row, project = listed[name], expected[name]
            if row.group("identifier") != str(project["identifier"]):
                problems.append(
                    f"{index_path.name}: {name} is attributed to revision "
                    f"{row.group('identifier')}; the data records {project['identifier']}"
                )
            if int(row.group("count")) != int(project["record_count"]):
                problems.append(
                    f"{index_path.name}: {name} reports {row.group('count')} {license_id} records; "
                    f"the data has {project['record_count']}"
                )
        if text is not None:
            carried.append(f"{license_id} covering {_records(records)}")
    return problems, carried


def _check_per_project(
    grouped: dict[str, dict[str, list[dict[str, object]]]], root: Path
) -> tuple[list[str], int]:
    """MIT, whose text names the holder, so every declaring source needs its own carrier."""
    problems: list[str] = []
    carried = 0

    classified = set(MIT_SOURCES) | set(INLINE_MIT_SOURCES) | FIRST_PARTY_SOURCES
    for source in sorted(_sources_declaring(grouped, "MIT") - classified):
        count = sum(int(p["record_count"]) for p in grouped[source]["MIT"])
        problems.append(
            f"{source} declares MIT on {count} records but carries no notice; add it to "
            f"MIT_SOURCES, INLINE_MIT_SOURCES, or FIRST_PARTY_SOURCES"
        )

    inline_path = root / "THIRD_PARTY_LICENSES.md"
    inline_text = inline_path.read_text(encoding="utf-8") if inline_path.is_file() else ""
    for source, copyright_line in INLINE_MIT_SOURCES.items():
        if not grouped.get(source, {}).get("MIT"):
            continue
        if copyright_line not in inline_text or MIT_MARKER not in inline_text:
            problems.append(
                f"{inline_path.name} carries no inline MIT notice for {source} "
                f"({copyright_line!r} and the permission notice must both appear there)"
            )
        else:
            carried += 1

    for source in sorted(FIRST_PARTY_SOURCES):
        if not grouped.get(source, {}).get("MIT"):
            continue
        own = root / "LICENSE"
        if not own.is_file() or MIT_MARKER not in own.read_text(encoding="utf-8"):
            problems.append(f"{source} records are first-party MIT but LICENSE carries no MIT notice")
        else:
            carried += 1

    for source in MIT_SOURCES:
        projects = grouped[source].get("MIT", [])
        expected = {_project_name(str(p["upstream_url"])): p for p in projects}
        path = _appendix_path(source)
        if len(expected) != len(projects):
            problems.append(f"two {source} upstream URLs share one project name")
        if not path.is_file():
            problems.append(f"{len(projects)} MIT {source} projects have no {path.name}")
            continue
        blocks = _ENTRY.split(path.read_text(encoding="utf-8"))[1:]
        found = dict(zip(blocks[::2], blocks[1::2]))
        if len(found) * 2 != len(blocks):
            problems.append(f"{path.name} repeats a project heading")
        for name in sorted(set(expected) - set(found)):
            problems.append(f"{path.name} carries no notice for MIT project {name}")
        for name in sorted(set(found) - set(expected)):
            problems.append(f"{path.name} carries a notice for {name}, which declares no MIT record")

        for name in sorted(set(expected) & set(found)):
            block = found[name]
            count = _COUNT.search(block)
            declared = expected[name]["record_count"]
            if count is None or int(count.group("count")) != declared:
                reported = count.group("count") if count else "no"
                problems.append(
                    f"{path.name}: {name} reports {reported} records; the data has {declared}"
                )
            body = _FENCE.search(block)
            if body is None:
                if UNRESOLVED not in block:
                    problems.append(f"{path.name}: {name} has neither a notice nor an UNRESOLVED mark")
            elif MIT_MARKER not in body.group("body"):
                problems.append(f"{path.name}: the {name} block has no MIT permission notice")
            else:
                carried += 1
    return problems, carried


def check(root: Path = ROOT) -> int:
    """Offline: the local files must carry terms for every record that declares them.

    The enumeration comes from the data, so the coverage of this check cannot silently fall behind
    what the corpus declares. Every distinct licence value is either carried as text or recorded as a
    non-declaration, and anything else is a failure naming the value.
    """
    grouped = declared_projects(root)
    declared = declared_license_records(grouped)

    problems = _check_unclassified(declared)
    canonical_problems, canonical = _check_canonical(grouped, declared)
    per_project_problems, mit_projects = _check_per_project(grouped, root)
    problems += canonical_problems + per_project_problems

    if problems:
        print("THIRD-PARTY NOTICES INCOMPLETE", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print("Regenerate with: python tools/emit_third_party_notices.py --fetch", file=sys.stderr)
        return 1

    recorded = sorted(license_id for license_id in NON_DECLARATIONS if declared.get(license_id))
    print(
        f"third-party notices carried locally: {mit_projects} MIT projects"
        + (f", canonical text for {', '.join(canonical)}" if canonical else "")
        + (f"; {', '.join(recorded)} record no declaration and stay unfinished" if recorded else "")
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true",
                        help="fail if any licence declared in the data has no local text")
    action.add_argument("--fetch", action="store_true",
                        help="re-retrieve every pinned licence file")
    args = parser.parse_args()

    try:
        return fetch(ROOT) if args.fetch else check(ROOT)
    except (NoticeError, ManifestError) as exc:
        print(f"refusing to emit incomplete third-party notices: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
