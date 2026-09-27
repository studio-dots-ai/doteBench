"""The installed package imports without repository-only helper modules."""

import os
import subprocess
import sys


def test_standalone_imports():
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import importlib.util; import dotebench; import dotebench.cli; "
                'assert importlib.util.find_spec("pipelines") is None; '
                'assert importlib.util.find_spec("tools.common") is None '
                'if importlib.util.find_spec("tools") else True'
            ),
        ],
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
