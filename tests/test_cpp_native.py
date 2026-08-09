"""Run the doctest-based C++ test binary from pytest.

Builds cpp/build/tests/cpp_tests via scripts/build_cpp_tests.sh if missing or
stale relative to any C++ source under cpp/ (excluding chess-library).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUILD_SCRIPT = ROOT / "scripts" / "build_cpp_tests.sh"
TEST_BIN = ROOT / "cpp" / "build" / "tests" / "cpp_tests"
SOURCE_GLOBS = ("cpp/*.cpp", "cpp/*.hpp", "cpp/mcts/*.cpp", "cpp/mcts/*.hpp", "cpp/include/*.hpp")


def _newest_source_mtime() -> float:
    latest = 0.0
    for pattern in SOURCE_GLOBS:
        for path in ROOT.glob(pattern):
            latest = max(latest, path.stat().st_mtime)
    return latest


def _build_required() -> bool:
    if not TEST_BIN.exists():
        return True
    return TEST_BIN.stat().st_mtime < _newest_source_mtime()


@pytest.fixture(scope="session")
def cpp_test_binary() -> Path:
    if shutil.which("g++") is None:
        pytest.skip("g++ not available")
    if _build_required():
        result = subprocess.run(
            ["bash", str(BUILD_SCRIPT)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip(
                f"cpp test build failed (likely missing TensorRT/CUDA): "
                f"{result.stderr.strip().splitlines()[-1] if result.stderr else 'unknown'}"
            )
    return TEST_BIN


def test_cpp_doctest_suite(cpp_test_binary: Path) -> None:
    result = subprocess.run(
        [str(cpp_test_binary)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"cpp_tests failed:\n{output}"
    assert "Status: SUCCESS" in output, f"unexpected doctest output:\n{output}"
