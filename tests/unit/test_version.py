# SPDX-License-Identifier: Apache-2.0
"""The version is written once, in the package, and read from there.

hatchling copies `__version__` into the core metadata at build time, so what
`dr --version` prints and what pip reports come from the same line. They can still
disagree in an editable install that predates a bump, and the metadata test says so
when that happens.

Dynamic version from a source file: https://hatch.pypa.io/latest/version/
"""

from __future__ import annotations

import importlib.metadata
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import deadreckoning
from deadreckoning.cli import app

REPO_ROOT = Path(__file__).resolve().parents[2]


def _pyproject() -> dict[str, Any]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_version_flag_prints_the_package_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It answers before any command runs, so an empty directory with no dr.toml will do."""
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout == f"{deadreckoning.__version__}\n"


def test_the_installed_metadata_matches_the_package() -> None:
    """A failure here means the editable install is older than the last bump: make install."""
    assert importlib.metadata.version("deadreckoning") == deadreckoning.__version__


def test_pyproject_takes_the_version_from_the_package() -> None:
    pyproject = _pyproject()
    assert "version" not in pyproject["project"]
    assert pyproject["project"]["dynamic"] == ["version"]
    source = REPO_ROOT / pyproject["tool"]["hatch"]["version"]["path"]
    assert f'__version__ = "{deadreckoning.__version__}"' in source.read_text(encoding="utf-8")


def test_the_python_floor_agrees_everywhere() -> None:
    """requires-python, the type checker's target and the README state one version."""
    pyproject = _pyproject()
    floor: str = pyproject["project"]["requires-python"].removeprefix(">=")
    assert re.fullmatch(r"3\.\d+", floor)
    assert pyproject["tool"]["pyright"]["pythonVersion"] == floor
    assert f"Python {floor} or newer" in (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert sys.version_info >= tuple(int(part) for part in floor.split("."))
