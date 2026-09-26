import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = (
    "/" + "newcpfs/",
    "wang" + "hankun",
    "code." + "devops.xiaohongshu.com",
)
REQUIRED = (
    "README.md",
    "LICENSE",
    "NOTICE",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "release/AUDIO-NOTICE.md",
    "release/DATA-NOTICE.md",
)
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def tracked_files():
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    ).decode("utf-8")
    return [ROOT / name for name in output.split("\0") if name]


def text(path):
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def broken_links(path, content):
    failures = []
    for raw in LINK.findall(content):
        target = raw.strip().split(maxsplit=1)[0].strip("<>")
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        target = unquote(target.split("#", 1)[0])
        resolved = (path.parent / target).resolve()
        if not resolved.is_relative_to(ROOT) or not resolved.exists():
            failures.append(f"{path.relative_to(ROOT)}: broken link {raw}")
    return failures


def test_public_release_hygiene():
    failures = []
    for name in REQUIRED:
        if not (ROOT / name).is_file():
            failures.append(f"missing public file: {name}")
    if (ROOT / "AGENTS.md").exists():
        failures.append("AGENTS.md is local agent configuration, not a public file")
    for path in tracked_files():
        if not path.is_file():
            continue
        content = text(path)
        if content is None:
            continue
        for marker in FORBIDDEN:
            if marker in content:
                failures.append(f"{path.relative_to(ROOT)}: contains {marker!r}")
        if path.suffix.lower() == ".md":
            failures.extend(broken_links(path, content))
    assert not failures, "\n".join(failures)
