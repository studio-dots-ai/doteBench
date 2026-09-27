"""Generic dependency-ordered service lifecycle with bounded readiness and cleanup."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from uuid import uuid4

from .resolver import ResolvedServiceGraph


def _tail_file(path: Path, max_bytes: int = 8192) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - max_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return "<log unavailable>"


def wait_for_urls(
    urls: Sequence[str],
    *,
    timeout_sec: float,
    diagnostics: Mapping[str, Mapping[str, Any]] | None = None,
    on_ready: Callable[[str], None] | None = None,
) -> None:
    deadline = time.monotonic() + timeout_sec
    pending = set(urls)
    while pending:
        for url in tuple(pending):
            diagnostic = (diagnostics or {}).get(url, {})
            process = diagnostic.get("process")
            detail = f"health_url={url}, log_path={diagnostic.get('log_path')}"
            if process is not None and process.poll() is not None:
                log = _tail_file(Path(diagnostic["log_path"]))
                raise RuntimeError(
                    f"service exited before readiness: {detail}, returncode={process.returncode}\n{log}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                tails = "\n".join(
                    _tail_file(Path(d["log_path"]))
                    for u, d in (diagnostics or {}).items()
                    if u in pending and d.get("log_path")
                )
                raise TimeoutError(
                    f"timed out after {timeout_sec}s waiting for {sorted(pending)}\n{detail}\n{tails}"
                )
            try:
                with urlopen(
                    Request(url, method="GET"), timeout=min(2.0, remaining)
                ) as response:
                    ready = 200 <= response.status < 300
            except (OSError, ValueError):
                ready = False
            if ready:
                pending.remove(url)
                if on_ready is not None:
                    on_ready(url)
        if pending:
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))


def terminate_processes(
    processes: Sequence[subprocess.Popen[Any]], *, timeout_sec: float = 5.0
) -> None:
    # Signal groups even when a shell leader has exited: descendants may survive it.
    for process in reversed(processes):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout_sec
    for process in reversed(processes):
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process in reversed(processes):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


@dataclass
class ManagedServiceGroup:
    processes: list[subprocess.Popen[Any]]
    log_paths: dict[str, Path]
    _stopped: bool = False

    def stop(self) -> None:
        if not self._stopped:
            terminate_processes(self.processes)
            self._stopped = True

    def __enter__(self) -> ManagedServiceGroup:
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()


def start_services(
    *,
    graph: ResolvedServiceGraph,
    project_root: Path,
    log_dir: Path,
    base_env: Mapping[str, str] | None = None,
) -> ManagedServiceGroup:
    """Execute argument arrays after each dependency becomes ready."""
    log_dir.mkdir(parents=True, exist_ok=True)
    group = ManagedServiceGroup([], {})
    visited: set[str] = set()
    events: list[dict[str, Any]] = []
    snapshot = log_dir / f"service_startup_{uuid4().hex[:12]}.json"

    def record(event: str, **fields: Any) -> None:
        events.append({"event": event, "time": time.time(), **fields})
        snapshot.write_text(json.dumps(events, indent=2) + "\n", encoding="utf-8")

    def start(node: Any) -> None:
        if node.instance_key in visited:
            return
        for dependency in node.dependencies.values():
            start(dependency)
        diagnostic: dict[str, Any] = {"name": node.instance_key}
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in (base_env or {}).items()})
        env.update(node.process_env)
        path = log_dir / f"{node.process_name}_{uuid4().hex[:12]}.log"
        with path.open("ab") as handle:
            process = subprocess.Popen(
                node.command,
                cwd=project_root,
                env=env,
                shell=False,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        group.processes.append(process)
        group.log_paths[node.process_name] = path
        diagnostic.update(process=process, log_path=path)
        record(
            "spawned",
            instance_key=node.instance_key,
            pid=process.pid,
            log_path=str(path),
        )
        wait_for_urls(
            [node.health_url],
            timeout_sec=node.startup_timeout_sec,
            diagnostics={node.health_url: diagnostic},
        )
        if node.startup_sleep_sec:
            time.sleep(node.startup_sleep_sec)
        process = diagnostic.get("process")
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"service {node.instance_key} exited after readiness")
        visited.add(node.instance_key)
        record("ready", instance_key=node.instance_key)

    try:
        for node in graph.roots.values():
            start(node)
        for node in graph.instances.values():
            start(node)
        record("group_ready")
        return group
    except BaseException as exc:
        try:
            record("failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            group.stop()
        raise
