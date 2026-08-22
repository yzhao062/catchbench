"""The wheel must contain and run the offline PRE board it advertises.

Source-tree tests cannot catch an incomplete wheel: ``tests/conftest.py`` deliberately puts this
checkout's ``src`` directory first, and PRE historically found its records by walking back to the
checkout root. These tests cross that boundary. They build one wheel per test session, inspect the
archive, then install it in a new virtual environment and run its console command from outside the
repository.

Set ``CATCHBENCH_TEST_WHEEL`` to an already-built wheel in a release job. Ordinary test runs build
from a temporary copy with build isolation and dependency resolution disabled, so the gate makes no
network request and does not dirty the checkout.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_METHODS = {
    "flag_all",
    "flag_none",
    "flag_risky_perms",
    "owasp_excess_permissions",
    "owasp_excess_functionality",
    "owasp_privilege_escalation",
    "unrequested_high_impact",
    "sensitive_access",
    "owasp_asi_combined",
    "oracle_privilege_diff",
    "llm_judge_needed(llama-3.3-70b)",
}


def _run(command: list[str], *, cwd: Path,
         env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    done = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    assert done.returncode == 0, (
        f"command failed ({done.returncode}): {subprocess.list2cmdline(command)}\n"
        f"stdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    )
    return done


@pytest.fixture(scope="session")
def catchbench_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    supplied = os.environ.get("CATCHBENCH_TEST_WHEEL")
    if supplied:
        wheel = Path(supplied).resolve()
        assert wheel.is_file() and wheel.suffix == ".whl", (
            "CATCHBENCH_TEST_WHEEL must name an existing .whl file"
        )
        return wheel

    workspace = tmp_path_factory.mktemp("catchbench-wheel")
    source = workspace / "source"
    shutil.copytree(
        ROOT,
        source,
        ignore=shutil.ignore_patterns(
            ".git", ".pytest_cache", "__pycache__", "*.pyc", "*.egg-info", "build", "dist"
        ),
    )
    wheelhouse = workspace / "wheelhouse"
    wheelhouse.mkdir()
    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            # Build isolation stays on. The backend is hatchling, which the test environment is not
            # required to carry, and turning isolation off made this test depend on whatever backend
            # happened to be installed rather than on the one pyproject.toml declares.
            "--wheel-dir",
            str(wheelhouse),
            str(source),
        ],
        cwd=workspace,
        env=env,
    )
    wheels = list(wheelhouse.glob("catchbench-*.whl"))
    assert len(wheels) == 1, f"expected one CatchBench wheel, found {wheels}"
    return wheels[0]


def test_wheel_contains_every_committed_pre_file(catchbench_wheel: Path):
    required = [path.relative_to(ROOT).as_posix() for path in (ROOT / "data" / "pre").rglob("*")
                if path.is_file()]
    assert required, "the source tree has no committed PRE records to package"

    with zipfile.ZipFile(catchbench_wheel) as archive:
        names = [name.rstrip("/") for name in archive.namelist() if not name.endswith("/")]
        missing = [relative for relative in required
                   if not any(name == relative or name.endswith(f"/{relative}") for name in names)]
        assert not missing, "the wheel omitted committed PRE files:\n" + "\n".join(missing)


def test_wheel_excludes_the_downloaded_corpora(catchbench_wheel: Path):
    with zipfile.ZipFile(catchbench_wheel) as archive:
        uncompressed_bytes = sum(info.file_size for info in archive.infolist())
        assert uncompressed_bytes < 50 * 1024 * 1024, (
            f"wheel expands to {uncompressed_bytes:,} bytes; the package must not carry the "
            "roughly 320 MB downloaded corpora"
        )


def test_a_fresh_wheel_install_scores_all_pre_rows(catchbench_wheel: Path, tmp_path: Path):
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"PYTHONHOME", "PYTHONPATH"}}
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

    virtualenv = tmp_path / "venv"
    _run([sys.executable, "-m", "venv", str(virtualenv)], cwd=tmp_path, env=environment)
    scripts = virtualenv / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    _run(
        [str(python), "-m", "pip", "install", "--no-deps", str(catchbench_wheel)],
        cwd=tmp_path,
        env=environment,
    )

    probe = _run(
        [
            str(python),
            "-I",
            "-c",
            (
                "from pathlib import Path; import catchbench, sys; "
                "package = Path(catchbench.__file__).resolve(); "
                "checkout = Path(sys.argv[1]).resolve(); "
                "assert checkout not in package.parents, package; "
                "paths = [Path(p).resolve() for p in sys.path if p]; "
                "assert all(checkout != path and checkout not in path.parents for path in paths), paths; "
                "print(package)"
            ),
            str(ROOT),
        ],
        cwd=tmp_path,
        env=environment,
    )
    assert str(virtualenv.resolve()).lower() in probe.stdout.strip().lower()

    command = scripts / ("catchbench.exe" if os.name == "nt" else "catchbench")
    assert command.is_file(), "the wheel did not install the documented 'catchbench' console command"
    scored = _run([str(command), "--task", "pre"], cwd=tmp_path, env=environment).stdout

    assert "PRE over_privilege: 1187 configs across 6 corpora" in scored
    marker = "[PRE] pre_over_privilege :: multi"
    breakdown = "[PRE] pre_over_privilege :: F1 by source"
    assert marker in scored and breakdown in scored
    board_lines = [line.strip()
                   for line in scored.split(marker, 1)[1].split(breakdown, 1)[0].splitlines()
                   if line.strip()]
    assert board_lines[0].startswith("method")
    assert len(board_lines[1:]) == len(EXPECTED_METHODS)
    methods = {line.split()[0] for line in board_lines[1:]}
    assert methods == EXPECTED_METHODS
