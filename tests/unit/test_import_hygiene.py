# SPDX-License-Identifier: Apache-2.0
"""Importing this package must do nothing observable.

A library that opens a socket, reads a file, or looks at an environment variable
while being imported cannot be reasoned about offline, and a node that phones home
on import would break the central claim of this runtime on the machine where it
matters most.

Every check below runs in a fresh interpreter with its guard installed BEFORE the
package is imported. A pytest fixture cannot do this job: by the time a fixture
runs, collection has already imported the module under test, so the guard would
watch nothing and the test would pass for the wrong reason.
"""

from __future__ import annotations

import ast
import pkgutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import deadreckoning

REPO_ROOT = Path(__file__).resolve().parents[2]

PACKAGE_MODULES = sorted(
    module.name for module in pkgutil.iter_modules(deadreckoning.__path__, "deadreckoning.")
)

IMPORT_ALL = "; ".join(f"import {name}" for name in PACKAGE_MODULES)


def _run_probe(body: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, f"probe failed:\n{result.stdout}\n{result.stderr}"
    return result.stdout.strip()


def test_there_are_modules_to_check() -> None:
    assert len(PACKAGE_MODULES) >= 6


def test_importing_opens_no_socket() -> None:
    output = _run_probe(f"""
        import socket
        opened = []
        socket.socket.connect = lambda self, address: opened.append(address)
        socket.create_connection = lambda *a, **k: opened.append(a)
        {IMPORT_ALL}
        print(repr(opened))
    """)
    assert output == "[]"


ENV_WATCHER = """
    import os
    read = []

    class Watching(dict):
        def __getitem__(self, key):
            read.append(key)
            return dict.__getitem__(self, key)

        def get(self, key, default=None):
            read.append(key)
            return dict.get(self, key, default)

        def __contains__(self, key):
            read.append(key)
            return dict.__contains__(self, key)

    os.environ = Watching(os.environ)
    os.getenv = os.environ.get
"""


DEPENDENCY_FREE = ["deadreckoning.canonical", "deadreckoning.clock", "deadreckoning.runtime"]
"""Modules that import nothing outside the standard library, so their import cost
is entirely this package's own and can be pinned at exactly zero."""


def test_the_dependency_free_modules_read_nothing() -> None:
    core = "; ".join(f"import {name}" for name in DEPENDENCY_FREE)
    output = _run_probe(f"{ENV_WATCHER}\n    {core}\n    print(repr(sorted(set(read))))")
    assert output == "[]", f"these modules read {output} while being imported"


def test_no_module_reads_a_setting_or_secret_at_import() -> None:
    """The real invariant, across the whole package.

    Pydantic and the console libraries read HOME, sysconfig paths and their own
    feature switches while being imported. That is outside this package's control
    and changes nothing about its behaviour. What must never happen is an
    endpoint, a token, or a runtime setting being picked up at import, because the
    caller could then neither override it nor see that it had happened.
    """
    output = _run_probe(f"{ENV_WATCHER}\n    {IMPORT_ALL}\n    print(repr(sorted(set(read))))")
    names: list[str] = ast.literal_eval(output)
    offending = [
        name
        for name in names
        if name.startswith("DR_")
        or any(
            part in name.upper()
            for part in ("KEY", "TOKEN", "SECRET", "PASSWORD", "URL", "ENDPOINT")
        )
    ]
    assert offending == [], f"configuration read at import time: {offending}"


def test_importing_opens_no_configuration_or_data_file() -> None:
    """Source and packaging metadata are read by the interpreter; data is not.

    The line that matters is between what Python needs to load a module and what
    this runtime would read to decide how to behave. A config file, a database, or
    a fixture opened at import would mean behaviour fixed before the caller had a
    say in it.
    """
    output = _run_probe(f"""
        import builtins, io
        real_open = builtins.open
        opened = []

        def watching(file, *args, **kwargs):
            opened.append(str(file))
            return real_open(file, *args, **kwargs)

        builtins.open = watching
        io.open = watching
        {IMPORT_ALL}
        print(repr(sorted(set(opened))))
    """)
    paths: list[str] = eval(output)
    data_like = [p for p in paths if p.endswith((".toml", ".db", ".json", ".sqlite", ".sqlite3"))]
    assert data_like == [], f"data read at import time: {data_like}"


@pytest.mark.parametrize("violation", ["socket", "environ"])
def test_the_probes_actually_catch_a_violation(violation: str) -> None:
    """The guards above are worthless unless they fail when they should.

    This plants the violation in a throwaway module rather than in the package,
    so the check proves the probe works without ever making the package dirty.
    """
    offence = (
        "import socket; socket.create_connection(('10.0.0.1', 80))"
        if violation == "socket"
        else "import os; os.environ['DR_HUB_TOKEN']"
    )
    probe = f"""
        import os, socket
        caught = []
        socket.create_connection = lambda *a, **k: caught.append(a)

        class Watching(dict):
            def __getitem__(self, key):
                caught.append(key)
                return dict.__getitem__(self, key)

        os.environ = Watching(os.environ)
        os.environ['DR_HUB_TOKEN'] = 'x'
        caught.clear()
        {offence}
        print(repr(bool(caught)))
        """
    assert _run_probe(probe) == "True"
