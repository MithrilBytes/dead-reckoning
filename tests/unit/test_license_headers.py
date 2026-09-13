# SPDX-License-Identifier: Apache-2.0
"""Every source file names its license in a form a scanner can read.

LICENSE covers the repository. A file copied out of it on its own carries nothing
unless it says so itself, and the SPDX short identifier is the form that license
scanners and people both recognise. The identifier also has to agree with the
expression in pyproject.toml, because that is what the wheel's metadata reports.

SPDX license identifiers in source files: https://spdx.dev/learn/handling-license-info/
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIRS = ("src", "tests", "demo", "hub")

SUFFIXES = (".py", ".sh")

HEADER = "# SPDX-License-Identifier: Apache-2.0"


def missing_headers(root: Path) -> list[str]:
    """Files whose first line, or the line after a shebang, is not the header."""
    missing: list[str] = []
    for directory in SOURCE_DIRS:
        for path in sorted((root / directory).rglob("*")):
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            lines = path.read_text(encoding="utf-8").splitlines()[:2]
            if lines and lines[0].startswith("#!"):
                lines = lines[1:]
            if not lines or lines[0] != HEADER:
                missing.append(path.relative_to(root).as_posix())
    return sorted(missing)


def test_every_source_file_carries_the_header() -> None:
    assert missing_headers(REPO_ROOT) == []


def test_a_file_without_the_header_is_found(tmp_path: Path) -> None:
    """The check is worthless unless it fails when it should."""
    package = tmp_path / "src" / "pkg"
    package.mkdir(parents=True)
    (package / "headed.py").write_text(f"{HEADER}\nVALUE = 1\n")
    (package / "bare.py").write_text("VALUE = 1\n")
    (package / "empty.py").write_text("")
    scripts = tmp_path / "demo"
    scripts.mkdir()
    (scripts / "run.sh").write_text(f"#!/usr/bin/env bash\n{HEADER}\necho ok\n")
    (scripts / "late.sh").write_text(f"#!/usr/bin/env bash\n\n{HEADER}\n")

    assert missing_headers(tmp_path) == ["demo/late.sh", "src/pkg/bare.py", "src/pkg/empty.py"]


def test_the_header_agrees_with_the_package_metadata() -> None:
    """PEP 639 puts the expression and the license files in the core metadata.

    https://peps.python.org/pep-0639/
    """
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["license"] == HEADER.removeprefix("# SPDX-License-Identifier: ")
    assert project["license-files"] == ["LICENSE", "NOTICE"]
    assert all((REPO_ROOT / name).is_file() for name in project["license-files"])
