"""PRE point/support regression, independent resampling controls, and two-block checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import emit_pre_tables as emit


@pytest.fixture(scope="module")
def data():
    return emit.load()


@pytest.fixture(scope="module")
def values(data):
    return emit.quantities(data)


@pytest.fixture(scope="module")
def blocks(data, values):
    return emit.tables(data, values)


def _paper(blocks):
    return "\n".join((emit.BOARD_START, blocks[emit.LABELS[0]], emit.MAIN_START,
                      blocks[emit.LABELS[1]], emit.BOARD_END, ""))


def test_all_110_original_printed_points(values):
    source = """
0.388 0.154 0.654 0.750 0.574 0.763 0.601
0.000 0.000 0.000 0.000 0.000 0.000 0.000
0.326 0.095 0.575 0.827 0.025 0.803 0.480
0.327 0.052 0.566 0.801 0.007 0.825 0.505
0.451 0.514 0.632 0.957 0.574 0.539 0.642
0.000 0.000 0.018 0.065 0.000 0.000 0.020
0.066 0.041 0.211 0.605 0.000 0.248 0.240
0.014 0.000 0.058 0.000 0.000 0.000 0.030
0.448 0.411 0.644 0.961 0.570 0.842 0.654
1.000 1.000 1.000 1.000 1.000 1.000 1.000
0.518 0.362 0.744 0.990 0.467 0.972 0.695
"""
    main = """
0.430 1.000 0.601
0.000 0.000 0.000
0.418 0.564 0.480
0.504 0.506 0.505
0.538 0.796 0.642
0.811 0.010 0.020
0.633 0.148 0.240
0.763 0.016 0.030
0.511 0.910 0.654
0.594 0.839 0.695
1.000 1.000 1.000
"""
    for method, expected in zip(emit.SOURCE_METHODS, source.strip().splitlines()):
        assert [f"{values[method, s, 'f1']['point']:.3f}" for s in (*emit.SOURCES, "overall")] == expected.split()
    for method, expected in zip(emit.MAIN_METHODS, main.strip().splitlines()):
        assert [f"{values[method, 'overall', m]['point']:.3f}" for m in emit.METRICS] == expected.split()


def test_record_order_and_parsed_support(data):
    for source, count, parsed in zip(emit.SOURCES, emit.SOURCE_COUNTS, emit.PARSED_COUNTS):
        raw = json.loads((emit.ROOT / f"data/pre/{source}.json").read_bytes())
        rows, strata = emit.support(data, "flag_all", source)
        assert [row.instance_id for row in rows] == [row["instance_id"] for row in raw]
        assert len(rows) == count
        assert strata[0].tolist() == list(range(count))
        judged, _ = emit.support(data, emit.JUDGE, source)
        assert len(judged) == parsed
    rows, strata = emit.support(data, emit.JUDGE, "overall")
    assert len(rows) == 1182
    assert tuple(map(len, strata)) == emit.PARSED_COUNTS
    for source, indices in zip(emit.SOURCES, strata):
        assert all(emit._pre_logical_source(rows[i].source) == source for i in indices)


@pytest.mark.parametrize("method,source,metric", [
    ("owasp_excess_functionality", "n8n", "f1"),
    ("owasp_excess_permissions", "overall", "precision"),
    (emit.JUDGE, "overall", "recall"),
])
def test_independent_cluster_ratio_bootstrap(data, values, method, source, metric):
    """A separate implementation catches macro averaging and source/draw order changes."""
    rows, strata = emit.support(data, method, source)
    counts = []
    for row in rows:
        prediction = data["predictions"][method][row.instance_id]
        truth = set(row.labels["excess_set"])
        counts.append((len(prediction & truth), len(prediction - truth), len(truth - prediction)))
    counts = np.array(counts)
    kind = "parsed" if method == emit.JUDGE else "all"
    label = f"per-arm/pre/{source}/{kind}"
    seed = int(hashlib.sha256(label.encode()).hexdigest()[:8], 16) ^ 20260907
    rng = np.random.default_rng(seed)
    totals = np.zeros((10_000, 3), dtype=np.int64)
    for start in range(0, 10_000, 500):
        for indices in strata:
            selected = rng.integers(len(indices), size=(500, len(indices)))
            totals[start:start + 500] += counts[indices[selected]].sum(axis=1)
    tp, fp, fn = totals.T
    numerator, denominator = {
        "precision": (tp, tp + fp), "recall": (tp, tp + fn),
        "f1": (2 * tp, 2 * tp + fp + fn),
    }[metric]
    draws = np.divide(numerator, denominator, out=np.zeros(10_000), where=denominator != 0)
    expected = np.quantile(draws, [0.025, 0.975], method="linear")
    interval = values[method, source, metric]["interval"]
    assert [interval["low"], interval["high"]] == expected.tolist()
    assert interval["rng_seed"] == seed


def test_stream_resets_for_every_method_and_metric(data, monkeypatch):
    original_rng = emit.stats._rng_for
    original_bootstrap = emit.stats._pre_configuration_bootstrap
    calls = []

    def rng_for(label, seed):
        rng = original_rng(label, seed)
        calls.append((label, seed, rng.bit_generator.state))
        return rng

    def bootstrap(a, b, strata, metric, draws, rng):
        assert rng.bit_generator.state == calls[-1][2]
        return original_bootstrap(a, b, strata, metric, draws, rng)

    monkeypatch.setattr(emit, "DRAWS", 7)
    monkeypatch.setattr(emit.stats, "_rng_for", rng_for)
    monkeypatch.setattr(emit.stats, "_pre_configuration_bootstrap", bootstrap)
    emit.quantities(data)
    overall = [state for label, seed, state in calls if label == "per-arm/pre/overall/all"]
    assert len(overall) == 24  # 9 methods * 3 metrics, less 3 recorded precision intervals.
    assert all(state == overall[0] for state in overall)
    parsed = [state for label, seed, state in calls if label == "per-arm/pre/overall/parsed"]
    assert len(parsed) == 3 and all(state == parsed[0] for state in parsed)


def test_recorded_precision_intervals_keep_endpoints_and_seed(data, values):
    for method in emit.stats.PRE_NARROW_RULES:
        label = f"pre.precision.{method}.vs.base_rate"
        claim = next(c for c in data["record"]["claims"] if c["id"] == label)
        interval = values[method, "overall", "precision"]["interval"]
        for key, expected in claim["precision_interval"].items():
            assert interval[key] == expected
        assert interval["seed"] == 20260817
        assert interval["rng_label"] == label
        assert interval["draws"] == 10_000


def test_zero_cells_exemptions_and_metadata(values, blocks):
    assert len(values) == 99
    for (method, source, metric), q in values.items():
        interval = q["interval"]
        if method == emit.ORACLE:
            assert interval is None and q["exemption"] == "construction identity"
            assert "identity" in emit._cell(q)
            continue
        assert interval["level"] == 0.95
        assert interval["draws"] == interval["usable_draws"] == 10_000
        assert interval["discarded_draws"] == 0
        assert interval["rng_engine"] == "PCG64"
        assert interval["quantile_method"] == "linear"
        if q["point"] == 0:
            assert interval["low"] == interval["high"] == 0
            assert interval["empirical_degenerate"]
            assert "[0.000,0.000]" in emit._cell(q)
    for label, expected in zip(emit.LABELS, (77, 33)):
        block = blocks[label]
        emitted = [json.loads(line.removeprefix("% PRE quantity ")) for line in block.splitlines()
                   if line.startswith("% PRE quantity ")]
        assert len(emitted) == expected
        assert sum(q["interval"] is None for q in emitted) == (7 if expected == 77 else 3)
        assert "base seed 20260907" in block
        assert "cannot recover their pairing" in block
        assert "source-stratified over configuration clusters" in block
        assert "empirical degenerate intervals" in block
        assert "no performance interval" in block
        assert "PRE provenance" in block


def test_check_accepts_both_blocks_and_environment(tmp_path, blocks, monkeypatch, capsys):
    (tmp_path / emit.APPENDIX).write_bytes(_paper(blocks).encode())
    monkeypatch.setenv("CATCHBENCH_PAPER_DIR", str(tmp_path))
    monkeypatch.setattr(emit, "load", lambda: None)
    monkeypatch.setattr(emit, "tables", lambda data: blocks)
    assert emit.main(["--check"]) == 0
    assert "tab:pre-source and tab:pre-main" in capsys.readouterr().out


@pytest.mark.parametrize("label", emit.LABELS)
@pytest.mark.parametrize("damage", ("cell", "missing_begin", "missing_end", "duplicate", "reversed"))
def test_check_rejects_damage_in_each_block(tmp_path, blocks, monkeypatch, capsys, label, damage):
    changed = dict(blocks)
    begin, end = emit.markers(label)
    if damage == "cell":
        # Modify only the visible point, leaving the machine-readable quantity intact.
        changed[label] = changed[label].replace(r"\shortstack{0.", r"\shortstack{9.", 1)
    elif damage == "missing_begin":
        changed[label] = changed[label].replace(begin, "")
    elif damage == "missing_end":
        changed[label] = changed[label].replace(end, "")
    elif damage == "duplicate":
        changed[label] += "\n" + end
    else:
        changed[label] = changed[label].replace(begin, "PLACEHOLDER").replace(end, begin).replace("PLACEHOLDER", end)
    (tmp_path / emit.APPENDIX).write_bytes(_paper(changed).encode())
    monkeypatch.setattr(emit, "load", lambda: None)
    monkeypatch.setattr(emit, "tables", lambda data: blocks)
    assert emit.main(["--check", "--paper", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "STALE" in output and label in output
    if damage == "cell":
        assert f"--- paper/{label}" in output and f"+++ generated/{label}" in output


@pytest.mark.parametrize("content", (None, "", "No PRE tables here.",
                                     "\n".join((emit.BOARD_START, emit.MAIN_START, emit.BOARD_END))))
def test_check_rejects_directory_without_tables(tmp_path, blocks, content, capsys):
    if content is not None:
        (tmp_path / emit.APPENDIX).write_bytes(content.encode())
    assert emit.check(tmp_path, blocks) == 1
    assert "STALE" in capsys.readouterr().out


@pytest.mark.parametrize("marker", (emit.BOARD_START, emit.MAIN_START, emit.BOARD_END))
def test_section_guards_are_required(tmp_path, blocks, marker, capsys):
    (tmp_path / emit.APPENDIX).write_bytes(_paper(blocks).replace(marker, "% " + marker).encode())
    assert emit.check(tmp_path, blocks) == 1
    assert marker in capsys.readouterr().out


def test_check_rejects_hidden_misplaced_and_crlf_blocks(tmp_path, blocks):
    original = _paper(blocks)
    variants = (
        original.replace(emit.BOARD_START, emit.BOARD_START + "\n\\iffalse"),
        "\n".join((emit.BOARD_START, *blocks.values(), emit.MAIN_START, emit.BOARD_END)),
        original.replace("\n", "\r\n"),
    )
    for changed in variants:
        (tmp_path / emit.APPENDIX).write_bytes(changed.encode())
        assert emit.check(tmp_path, blocks) == 1


def test_cli_missing_paper_argument_exits_nonzero(monkeypatch):
    monkeypatch.delenv("CATCHBENCH_PAPER_DIR", raising=False)
    result = subprocess.run([sys.executable, "-B", str(Path(emit.__file__)), "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "--check needs --paper" in result.stderr
