"""Install a locked candidate environment into a shared uv environment root."""

import argparse
import os
import subprocess
from pathlib import Path


def validate_interpreter_location(environment: Path, python_install_dir: Path) -> None:
    """Reject environments whose base interpreter is not on shared storage."""
    shared_python_root = python_install_dir.resolve()
    interpreter = (environment / "bin/python").resolve(strict=True)
    if not interpreter.is_relative_to(shared_python_root):
        raise ValueError(
            "Environment interpreter must reside in the shared Python install "
            f"directory: {interpreter} is outside {shared_python_root}"
        )

    pyvenv_config = environment / "pyvenv.cfg"
    fields = {}
    for line in pyvenv_config.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            fields[key.strip()] = value.strip()
    if "home" not in fields:
        raise ValueError(f"Environment is missing home in {pyvenv_config}")
    interpreter_home = Path(fields["home"]).resolve(strict=True)
    if not interpreter_home.is_relative_to(shared_python_root):
        raise ValueError(
            "Environment base interpreter home must reside in the shared Python "
            f"install directory: {interpreter_home} is outside {shared_python_root}"
        )


def main():
    # Direct script invocation must work before doteBench is installed.
    if __package__:
        from .registry import native_candidates
    else:
        from registry import native_candidates

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        required=True,
        choices=native_candidates(),
    )
    parser.add_argument("--environment-root", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--python-install-dir", required=True, type=Path)
    args = parser.parse_args()
    project = (
        Path(__file__).resolve().parent
        / args.candidate.replace("-", "_")
        / "environment"
    )
    environment = args.environment_root.resolve() / ("dotebench-" + args.candidate)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.environment_root.mkdir(parents=True, exist_ok=True)
    if args.cache_dir.stat().st_dev != args.environment_root.stat().st_dev:
        raise ValueError(
            "uv cache and environments must share a filesystem for hard links"
        )
    args.python_install_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "UV_CACHE_DIR": str(args.cache_dir.resolve()),
        "UV_LINK_MODE": "hardlink",
        "UV_PYTHON_INSTALL_DIR": str(args.python_install_dir.resolve()),
        "UV_MANAGED_PYTHON": "1",
        "UV_PROJECT_ENVIRONMENT": str(environment),
    }
    subprocess.run(
        ["uv", "sync", "--frozen", "--project", str(project)], env=env, check=True
    )
    subprocess.run(
        ["uv", "pip", "check", "--python", str(environment / "bin/python")],
        env=env,
        check=True,
    )
    validate_interpreter_location(environment, args.python_install_dir)
    print(environment)


if __name__ == "__main__":
    main()
