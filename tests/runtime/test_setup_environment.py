from pathlib import Path

import pytest

from dotebench.candidates.setup_environment import validate_interpreter_location


def make_environment(tmp_path: Path, interpreter: Path, home: Path) -> Path:
    environment = tmp_path / "envs" / "candidate"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin" / "python").symlink_to(interpreter)
    (environment / "pyvenv.cfg").write_text(
        f"home = {home}\nimplementation = CPython\n", encoding="utf-8"
    )
    return environment


def test_shared_interpreter_and_home_are_accepted(tmp_path):
    install_root = tmp_path / "shared" / "python"
    interpreter_home = install_root / "cpython" / "bin"
    interpreter_home.mkdir(parents=True)
    interpreter = interpreter_home / "python3.12"
    interpreter.touch()
    environment = make_environment(tmp_path, interpreter, interpreter_home)

    validate_interpreter_location(environment, install_root)


@pytest.mark.parametrize("field", ["link", "home"])
def test_node_local_interpreter_paths_are_rejected(tmp_path, field):
    install_root = tmp_path / "shared" / "python"
    shared_home = install_root / "cpython" / "bin"
    local_home = tmp_path / "root" / ".local" / "uv" / "python" / "bin"
    shared_home.mkdir(parents=True)
    local_home.mkdir(parents=True)
    shared_interpreter = shared_home / "python3.12"
    local_interpreter = local_home / "python3.12"
    shared_interpreter.touch()
    local_interpreter.touch()
    environment = make_environment(
        tmp_path,
        local_interpreter if field == "link" else shared_interpreter,
        local_home if field == "home" else shared_home,
    )

    with pytest.raises(ValueError, match="shared Python install directory"):
        validate_interpreter_location(environment, install_root)
