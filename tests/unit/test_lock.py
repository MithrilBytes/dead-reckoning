# SPDX-License-Identifier: Apache-2.0
"""The pins in constraints.txt, checked without asking the package index.

`make lock-check` is the complete test and it needs the network. These catch the
usual mistakes offline: a requirement added to pyproject.toml without a new lock, a
pin that is not exact, and a file resolved for a different Python.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

from scripts.lock import CONSTRAINTS, canonical_name, render

REPO_ROOT = Path(__file__).resolve().parents[2]

PIN = re.compile(r"(?P<name>[a-z0-9]+(?:-[a-z0-9]+)*)==(?P<version>\S+)")

TOOLCHAIN = ("ruff", "pyright", "pytest", "hypothesis")


def _pins() -> dict[str, str]:
    lines = [
        line
        for line in CONSTRAINTS.read_text(encoding="utf-8").splitlines()
        if not line.startswith("#")
    ]
    matches = [PIN.fullmatch(line) for line in lines]
    assert all(matches), [line for line, m in zip(lines, matches, strict=True) if m is None]
    pins = {m["name"]: m["version"] for m in matches if m is not None}
    assert len(pins) == len(lines), "a package is pinned more than once"
    return pins


def _requirement_names() -> set[str]:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements: list[str] = [*project["dependencies"], *project["optional-dependencies"]["dev"]]
    names: set[str] = set()
    for requirement in requirements:
        match = re.match(r"[A-Za-z0-9._-]+", requirement)
        assert match, requirement
        names.add(canonical_name(match[0]))
    return names


def test_every_line_is_one_exact_pin() -> None:
    assert _pins()


def test_every_requirement_in_pyproject_is_pinned() -> None:
    assert _requirement_names() - _pins().keys() == set()


def test_the_toolchain_is_pinned() -> None:
    pins = _pins()
    assert all(tool in pins for tool in TOOLCHAIN), sorted(pins)


def test_the_pins_were_resolved_for_the_python_floor() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    floor = pyproject["project"]["requires-python"].removeprefix(">=")
    header = CONSTRAINTS.read_text(encoding="utf-8").splitlines()[0]
    assert f" on Python {floor}. " in header


def test_render_leaves_out_the_project_and_normalises_names() -> None:
    report: dict[str, Any] = {
        "install": [
            {"is_direct": True, "metadata": {"name": "deadreckoning", "version": "0.1.0"}},
            {"is_direct": False, "metadata": {"name": "typing_extensions", "version": "4.16.0"}},
            {"is_direct": False, "metadata": {"name": "Pygments", "version": "2.21.0"}},
            {"metadata": {"name": "annotated.doc", "version": "0.0.5"}},
        ]
    }
    assert render(report, (3, 13)).splitlines() == [
        "# Exact versions for .[dev] on Python 3.13. Rewrite with make lock.",
        "annotated-doc==0.0.5",
        "pygments==2.21.0",
        "typing-extensions==4.16.0",
    ]
