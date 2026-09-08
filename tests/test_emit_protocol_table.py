"""Guard the frozen BIN construction and both directions of the paper checker.

The shipped scalars must recover the original 30 printed rates. Corrupted
records must fail before rendering, and CLI checks use temporary paper trees
so no sibling manuscript checkout is needed or changed.
"""
import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import emit_protocol_table as ept  # noqa: E402
from statistical_tests import _rng_for  # noqa: E402


@pytest.fixture
def record():
    return ept.load()


@pytest.fixture(scope="module")
def generated():
    return ept.table(ept.load())


def _claim(record, protocol="step_by_step"):
    return next(c for c in record["claims"]
                if c["id"] == f"loc.protocol.gpt-5.5.all.vs.{protocol}")


def _cli(*args, paper_env=None):
    env = os.environ.copy()
    env.pop("CATCHBENCH_PAPER_DIR", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if paper_env is not None:
        env["CATCHBENCH_PAPER_DIR"] = str(paper_env)
    return subprocess.run([sys.executable, str(Path(ept.__file__)), *map(str, args)],
                          capture_output=True, env=env, timeout=60)


def test_all_30_counts_reproduce_the_original_printed_rates(record):
    printed = (
        "0.452 0.397 0.421 0.421 0.389 0.357 0.413 0.381 0.365 "
        "0.405 0.317 0.405 0.357 0.341 0.357 0.349 0.254 0.127 "
        "0.333 0.222 0.222 0.206 0.230 0.159 0.135 0.190 0.214 "
        "0.127 0.167 0.167"
    ).split()
    cells = ept.cells(record)
    assert len(cells) == len(printed) == 30
    for cell, expected in zip(cells, printed):
        assert cell["n"] == 126
        assert cell["point"] * 126 == pytest.approx(cell["k"], abs=1e-10, rel=0)
        assert f"{cell['k'] / 126:.3f}" == expected


def test_bin_uses_the_frozen_stream_for_every_cell(record):
    assert ept._rng_for is _rng_for
    for cell in ept.cells(record):
        label = (f"per-arm/bin/whoandwhen/{cell['protocol']}/"
                 f"{cell['model']}/top1")
        assert cell["rng_label"] == label
        rng = _rng_for(label, 20260907)
        assert cell["rng_seed"] == rng.bit_generator.seed_seq.entropy
        expected = np.percentile(rng.binomial(126, cell["k"] / 126, 10000) / 126,
                                 [2.5, 97.5], method="linear")
        np.testing.assert_array_equal([cell["low"], cell["high"]], expected)


def test_all_at_once_labels_are_shared_with_localization(record):
    cells = [c for c in ept.cells(record) if c["protocol"] == "all_at_once"]
    assert len(cells) == 10
    for cell in cells:
        assert cell["rng_label"] == (
            f"per-arm/bin/whoandwhen/all_at_once/{cell['model']}/top1")


def test_claim_order_and_pairwise_intervals_cannot_change_marginal_results(record):
    original = ept.cells(record)
    changed = copy.deepcopy(record)
    changed["claims"].reverse()
    for claim in changed["claims"]:
        claim.pop("interval", None)
        claim.pop("test", None)
    ignore = {"selector", "n_selector"}
    assert [{k: v for k, v in c.items() if k not in ignore} for c in original] == [
        {k: v for k, v in c.items() if k not in ignore} for c in ept.cells(changed)]


@pytest.mark.parametrize("point", [0.452, 0.5 / 126, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_scalar_is_a_hard_failure(record, point):
    _claim(record)["estimate"]["b"] = point
    with pytest.raises(ValueError, match="Top-1"):
        ept.table(record)


def test_roundoff_within_tolerance_is_accepted():
    assert ept._count((57 + 1e-12) / 126, "fixture") == 57


@pytest.mark.parametrize("corruption, message", [
    ("support", "expected n=126"),
    ("duplicate_arm", "duplicate all-at-once scalars disagree"),
    ("metric", "incompatible metric or arm identity"),
    ("identity", "incompatible metric or arm identity"),
    ("missing_claim", "found 0"),
    ("duplicate_claim", "found 2"),
])
def test_incompatible_record_is_a_hard_failure(record, corruption, message):
    claim = _claim(record)
    if corruption == "support":
        claim["variance_axes"]["run_sampling"]["n"] = 125
    elif corruption == "duplicate_arm":
        _claim(record, "binary_search")["estimate"]["a"] += 1 / 126
    elif corruption == "metric":
        claim["metric"] = "mrr"
    elif corruption == "identity":
        claim["estimate"]["b_name"] = "another-model:step_by_step"
    elif corruption == "missing_claim":
        record["claims"].remove(claim)
    else:
        record["claims"].append(copy.deepcopy(claim))
    with pytest.raises(ValueError, match=message):
        ept.table(record)


def test_cli_prints_deterministic_lf_bytes(generated):
    first, second = _cli(), _cli()
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == b""
    assert first.stdout == second.stdout == (generated + "\n").encode("utf-8")
    assert b"\r" not in first.stdout
    assert generated.count(r"\shortstack{") == 30
    assert r"\ref{app:stats}" in generated
    assert "base seed 20260907" in generated
    assert "support no comparison between two cells" in generated


@pytest.mark.parametrize("use_env", [False, True])
def test_check_cli_accepts_matching_block(tmp_path, generated, use_env):
    (tmp_path / "09_appendix.tex").write_bytes(
        ("unrelated before\n" + generated + "\nunrelated after\n").encode("utf-8"))
    result = (_cli("--check", paper_env=tmp_path) if use_env
              else _cli("--check", "--paper", tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert b"paper is current: tab:protocol" in result.stdout


@pytest.mark.parametrize("corruption, message", [
    ("changed_cell", b"protocol table differs"),
    ("missing_begin", b"appears 0 times"),
    ("missing_end", b"appears 0 times"),
    ("no_table", b"appears 0 times"),
    ("no_appendix", b"cannot read appendix"),
    ("duplicate_marker", b"appears 2 times"),
    ("reversed_markers", b"precedes its begin marker"),
    ("whitespace", b"protocol table differs"),
    ("crlf", b"protocol table differs"),
])
def test_check_cli_rejects_stale_paper(tmp_path, generated, corruption, message):
    text = generated
    if corruption == "changed_cell":
        text = text.replace(r"\shortstack{0.452", r"\shortstack{0.999", 1)
    elif corruption == "missing_begin":
        text = text.replace(ept._BEGIN, "")
    elif corruption == "missing_end":
        text = text.replace(ept._END, "")
    elif corruption == "no_table":
        text = r"\section{An appendix without this table}"
    elif corruption == "duplicate_marker":
        text += "\n" + ept._BEGIN
    elif corruption == "reversed_markers":
        text = ept._END + "\n" + ept._BEGIN
    elif corruption == "whitespace":
        text = text.replace(r"\centering", r"\centering ", 1)
    elif corruption == "crlf":
        text = text.replace("\n", "\r\n")
    if corruption != "no_appendix":
        (tmp_path / "09_appendix.tex").write_bytes(text.encode("utf-8"))
    result = _cli("--check", "--paper", tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert message in result.stdout
    if corruption == "changed_cell":
        assert b"--- paper/09_appendix.tex" in result.stdout
        assert b"0.999" in result.stdout and b"0.452" in result.stdout


def test_check_cli_requires_paper_path():
    result = _cli("--check")
    assert result.returncode == 2
    assert b"--check needs --paper <dir> or CATCHBENCH_PAPER_DIR" in result.stderr
