"""Install the pinned Praat server executable into the invoking Python environment."""

import argparse
import hashlib
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

VERSION = "6.1.38"
URL = (
    "https://github.com/praat/praat.github.io/releases/download/"
    "v6.1.38/praat6138_linux64barren.tar.gz"
)
ARCHIVE_SHA256 = "a5c117b13b7434672740aebb745bc5c71a0368d2c363fd8d68a4b515211c98b4"
BINARY_SHA256 = "b7b4bce6017170082627546197dd34d8b7be60fd63f0e0b17876e7e28dfdc072"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install(environment):
    if platform.system() != "Linux" or platform.machine().lower() not in {
        "x86_64",
        "amd64",
    }:
        raise RuntimeError("The pinned Praat installer supports Linux x86_64")
    environment = Path(environment)
    if not (environment / "pyvenv.cfg").is_file():
        raise ValueError("Run this script with the target virtual environment's Python")
    destination = environment / "bin" / "praat"
    if destination.exists() or destination.is_symlink():
        if destination.is_file() and sha256(destination) == BINARY_SHA256:
            if not destination.stat().st_mode & 0o111:
                raise ValueError(f"Praat is not executable: {destination}")
            return destination
        raise FileExistsError(
            f"A different Praat executable already exists: {destination}"
        )
    with tempfile.TemporaryDirectory(
        prefix=".praat-install-", dir=destination.parent
    ) as tmp:
        archive = Path(tmp) / "praat.tar.gz"
        with urlopen(URL, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        if sha256(archive) != ARCHIVE_SHA256:
            raise ValueError("Praat download checksum mismatch")
        binary = Path(tmp) / "praat"
        with tarfile.open(archive, "r:gz") as package:
            member = package.getmember("praat_barren")
            if not member.isfile():
                raise ValueError("Praat archive must contain a regular executable")
            with package.extractfile(member) as source, binary.open("wb") as output:
                shutil.copyfileobj(source, output)
        if sha256(binary) != BINARY_SHA256:
            raise ValueError("Praat executable checksum mismatch")
        binary.chmod(0o755)
        version = subprocess.run(
            [str(binary), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        if version.split()[:2] != ["Praat", VERSION]:
            raise ValueError(f"Unexpected Praat version: {version}")
        # Link publishes a complete file atomically and refuses concurrent replacement.
        try:
            destination.hardlink_to(binary)
        except FileExistsError:
            if not destination.is_file() or sha256(destination) != BINARY_SHA256:
                raise
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(install(Path(sys.prefix)))


if __name__ == "__main__":
    main()
