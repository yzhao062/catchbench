"""The generated asset inventory must cover the data tree and reject stale copies."""
import json
import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import emit_asset_manifest as eam  # noqa: E402


def _read(root, relative):
    return json.loads((root / relative).read_text(encoding="utf-8"))


def _write(root, relative, value):
    (root / relative).write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@pytest.fixture
def data_copy(tmp_path):
    """A writable copy of the data tree, so a test can inject one defect into it."""
    shutil.copytree(ROOT / "data", tmp_path / "data")
    return tmp_path


def test_committed_manifest_matches_the_data_tree():
    assert eam.check_manifest(ROOT / "ASSET_MANIFEST.json", ROOT) == 0


def test_every_data_artifact_has_exactly_one_manifest_row():
    generated = eam.build_manifest(ROOT)
    paths = [row["path"] for row in generated["artifacts"]]
    expected = sorted(path.relative_to(ROOT).as_posix()
                      for path in (ROOT / "data").rglob("*") if path.is_file())
    assert paths == expected
    assert len(paths) == len(set(paths))


def test_a_perturbed_manifest_copy_fails_the_staleness_check(tmp_path):
    stale = json.loads(eam.render_manifest(ROOT))
    stale["artifacts"][0]["row_count"] += 1
    copy = tmp_path / "ASSET_MANIFEST.json"
    copy.write_text(json.dumps(stale, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    assert eam.check_manifest(copy, ROOT) == 1


def test_an_unmodified_data_copy_still_generates(data_copy):
    """Guards the fixture itself: without this, a later 'it raises' proves nothing."""
    assert eam.build_manifest(data_copy)["artifacts"]


def test_a_cached_vote_with_no_source_record_fails_generation(data_copy):
    """A row whose instance ID no released record provenances cannot borrow a distribution.

    The manifest states an artifact's licence distribution as fact. An ID with no source record
    supplies no repository, revision, path, or licence, so the only honest outcome is to refuse to
    emit rather than to attribute the row to the rest of its source set.
    """
    relative = "data/pre/gpt_judge_votes/mcp.json"
    votes = _read(data_copy, relative)
    votes["mcp-9999-not-a-released-server"] = {
        "model": "gpt-5.5",
        "needed": ["search"],
        "raw_response": "NEEDED: search",
        "status": "ok",
    }
    _write(data_copy, relative, votes)

    with pytest.raises(eam.ManifestError, match="mcp-9999-not-a-released-server"):
        eam.build_manifest(data_copy)


def test_a_claude_vote_row_with_no_source_record_fails_generation(data_copy):
    relative = "data/pre/claude_judge_votes/n8n.json"
    votes = _read(data_copy, relative)
    votes.append({"instance_id": "n8n-000000", "needed": []})
    _write(data_copy, relative, votes)

    with pytest.raises(eam.ManifestError, match="n8n-000000"):
        eam.build_manifest(data_copy)


def test_a_repeated_claude_vote_row_fails_generation(data_copy):
    relative = "data/pre/claude_judge_votes/n8n.json"
    votes = _read(data_copy, relative)
    votes.append(dict(votes[0]))
    _write(data_copy, relative, votes)

    with pytest.raises(eam.ManifestError, match="duplicate PRE instance ids"):
        eam.build_manifest(data_copy)


def test_a_subset_artifact_does_not_inherit_the_whole_source_distribution(data_copy):
    """Drop one MIT-declared record from a cache; only that cache's MIT count may move."""
    dropped = next(
        row["instance_id"]
        for row in _read(data_copy, "data/pre/mcp.json")
        if row["provenance"]["license"] == "MIT"
    )
    relative = "data/pre/gpt_judge_votes/mcp.json"
    votes = _read(data_copy, relative)
    del votes[dropped]
    _write(data_copy, relative, votes)

    manifest = eam.build_manifest(data_copy)
    source_mit = manifest["source_sets"]["mcp"]["declared_license_distribution"]["MIT"]
    artifact = next(row for row in manifest["artifacts"] if row["path"] == relative)

    assert artifact["declared_license_distribution"]["mcp"]["MIT"] == source_mit - 1
    assert sum(artifact["declared_license_distribution"]["mcp"].values()) == artifact["row_count"]


def test_every_artifact_accounts_for_exactly_the_rows_it_holds():
    """The arithmetic that made the earlier inherited distributions wrong."""
    mismatched = [
        (row["path"], row["row_count"], total)
        for row in eam.build_manifest(ROOT)["artifacts"]
        for total in [sum(sum(d.values()) for d in row["declared_license_distribution"].values())]
        if total != row["row_count"]
    ]
    assert mismatched == []


def test_a_mutable_identifier_is_never_presented_as_a_revision():
    manifest = eam.build_manifest(ROOT)
    kinds = {name: source["identifier_kind"] for name, source in manifest["source_sets"].items()}

    assert kinds["n8n"] == eam.MUTABLE
    assert kinds["sweagent"] == eam.MUTABLE
    assert kinds["crewai"] == kinds["mcp"] == kinds["injecagent"] == eam.IMMUTABLE
    assert kinds["whoandwhen"] == eam.IMMUTABLE

    for name, source in manifest["source_sets"].items():
        assert "upstream_revisions" not in source, name
        for entry in source["upstream_identifiers"]:
            assert entry["identifier_kind"] == kinds[name]
            assert "revision" not in entry, name
    for row in manifest["artifacts"]:
        assert "upstream_revision_sets" not in row, row["path"]


def test_a_mutable_source_records_where_its_artifacts_were_fetched():
    """A mutable identifier proves nothing on its own, so the fetch location has to be recorded."""
    sys.path.insert(0, str(ROOT / "tools"))
    import pre_harvest_sweagent as harvester

    manifest = eam.build_manifest(ROOT)
    sweagent = manifest["source_sets"]["sweagent"]

    assert sweagent["artifact_base_url"] == (
        f"{harvester.BUCKET_BASE_URL}/{harvester.S3_PREFIX}/"
    )
    assert harvester.SUBMISSION == sweagent["upstream_identifiers"][0]["identifier"]
    assert manifest["source_sets"]["n8n"]["artifact_base_url"].startswith("https://api.n8n.io/")
