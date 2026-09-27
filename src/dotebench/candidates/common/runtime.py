"""Prepare verified upstream inference trees without modifying submodules."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from dotebench.candidates.registry import native_candidates
from dotebench.compilation import source_path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_directory(name):
    return source_path(f"dotebench.candidates.{name}.compile").parent


def runtime_identity(name):
    directory = candidate_directory(name)
    manifest = json.loads((directory / "runtime.json").read_text())
    dependencies = [
        directory / "runtime.json",
        source_path(f"dotebench.candidates.{name}.backend"),
        directory / "environment/pyproject.toml",
        directory / "environment/uv.lock",
        Path(__file__),
        Path(__file__).with_name("audio.py"),
        Path(__file__).with_name("launch.py"),
    ]
    for patch in manifest["patches"]:
        path = directory / patch["path"]
        if sha256(path) != patch["sha256"]:
            raise ValueError(f"Patch checksum mismatch: {path}")
        dependencies.append(path)
    files = {
        str(p.relative_to(directory))
        if p.is_relative_to(directory)
        else str(p.relative_to(source_path("dotebench").parent)): sha256(p)
        for p in dependencies
    }
    digest = hashlib.sha256(
        json.dumps(
            {"sources": files},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "upstream": manifest["upstream"],
        "files": files,
        "sha256": digest,
    }


def checked(command, **kwargs):
    return subprocess.check_output(command, text=True, **kwargs).strip()


def verify_tree(root, expected):
    actual = {
        str(p.relative_to(root)): sha256(p)
        for p in root.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.name != ".dotebench-runtime.json"
    }
    if actual != expected:
        raise ValueError(f"Prepared runtime changed: {root}")


def check_source_registration(name, source_root):
    """Match the packaged runtime lock to the repository's pinned submodule."""
    source_root = Path(source_root)
    directory = candidate_directory(name)
    manifest = json.loads((directory / "runtime.json").read_text())
    upstream = manifest["upstream"]
    path = "third_party/" + upstream["directory"]
    tree = checked(
        ["git", "-C", str(source_root), "ls-tree", "HEAD", "--", path]
    ).split()
    if (
        len(tree) != 4
        or tree[:2] != ["160000", "commit"]
        or tree[2] != upstream["commit"]
    ):
        raise ValueError(f"Runtime lock differs from Git submodule: {name}")
    url = checked(
        [
            "git",
            "config",
            "--file",
            str(source_root / ".gitmodules"),
            "--get",
            f"submodule.{path}.url",
        ]
    )
    if url != upstream["url"]:
        raise ValueError(f"Runtime upstream URL differs from Git submodule: {name}")
    for patch in manifest["patches"]:
        if sha256(directory / patch["path"]) != patch["sha256"]:
            raise ValueError(f"Patch checksum mismatch: {patch['path']}")


def _check_upstream(upstream, manifest):
    if (
        checked(["git", "-C", str(upstream), "rev-parse", "HEAD"])
        != manifest["upstream"]["commit"]
    ):
        raise ValueError("Upstream commit mismatch")
    if checked(
        ["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=all"]
    ):
        raise ValueError("Upstream checkout must be clean")


def _prepare_runtime(name, *, upstream, destination):
    identity = runtime_identity(name)
    directory = candidate_directory(name)
    manifest = json.loads((directory / "runtime.json").read_text())
    upstream = Path(upstream).resolve()
    _check_upstream(upstream, manifest)
    destination = Path(destination).resolve() / (
        name.replace("_", "-") + "-" + identity["sha256"]
    )
    if destination.exists():
        stamp = json.loads((destination / ".dotebench-runtime.json").read_text())
        if stamp["identity"] != identity:
            raise ValueError("Prepared runtime identity mismatch")
        verify_tree(destination, stamp["tree"])
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix="prepare-"
    ) as temporary:
        stage = Path(temporary) / "tree"
        stage.mkdir()
        archive = Path(temporary) / "upstream.tar"
        subprocess.run(
            [
                "git",
                "-C",
                str(upstream),
                "archive",
                "--format=tar",
                "-o",
                str(archive),
                manifest["upstream"]["commit"],
            ],
            check=True,
        )
        with tarfile.open(archive) as handle:
            # The archive comes from the verified upstream commit.
            for member in handle.getmembers():
                if member.name.startswith("/") or ".." in Path(member.name).parts:
                    raise ValueError("Invalid upstream archive path")
            handle.extractall(stage)
        for relative, record in manifest["files"].items():
            path = stage / relative
            if (sha256(path) if path.exists() else None) != record["before"]:
                raise ValueError(f"Patch input mismatch: {relative}")
        for patch in manifest["patches"]:
            path = directory / patch["path"]
            subprocess.run(
                ["git", "apply", "--check", str(path)], cwd=stage, check=True
            )
            subprocess.run(["git", "apply", str(path)], cwd=stage, check=True)
        for relative, record in manifest["files"].items():
            if sha256(stage / relative) != record["after"]:
                raise ValueError(f"Patch output mismatch: {relative}")
        tree = {
            str(p.relative_to(stage)): sha256(p)
            for p in stage.rglob("*")
            if p.is_file()
        }
        (stage / ".dotebench-runtime.json").write_text(
            json.dumps({"identity": identity, "tree": tree}, indent=2) + "\n"
        )
        stage.rename(destination)
    return destination


def prepare_runtime(name, *, upstream, destination):
    """Serialize initialization and validation of a shared candidate cache."""
    manifest = json.loads((candidate_directory(name) / "runtime.json").read_text())
    _check_upstream(Path(upstream).resolve(), manifest)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / (name.replace("_", "-") + ".lock")).open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return _prepare_runtime(name, upstream=upstream, destination=destination)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def activate_runtime(name):
    root = Path(os.environ["DOTEBENCH_NATIVE_RUNTIME"]).resolve()
    stamp = json.loads((root / ".dotebench-runtime.json").read_text())
    if stamp["identity"] != runtime_identity(name):
        raise ValueError(
            "Service runtime fingerprint mismatch; prepare the runtime again"
        )
    verify_tree(root, stamp["tree"])
    sys.path.insert(0, str(root))
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        required=True,
        choices=native_candidates(),
    )
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args(argv)
    print(
        prepare_runtime(
            args.candidate.replace("-", "_"),
            upstream=args.upstream,
            destination=args.destination,
        )
    )


if __name__ == "__main__":
    main()
