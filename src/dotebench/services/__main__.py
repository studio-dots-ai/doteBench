"""Run a configured service group until interrupted."""

import argparse
import json
from pathlib import Path
import signal
import threading

from . import load_service_bundle, start_services


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)
    graph = load_service_bundle(args.config, overrides=args.override)
    if args.print_only:
        print(json.dumps(graph.direct_services.as_urls(), indent=2))
        return 0
    stopped = threading.Event()
    old_handlers = {}

    def stop(signum, frame):
        stopped.set()
        # Interrupt startup so its exception handler tears down partial groups.
        if not started:
            raise KeyboardInterrupt

    started = False
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, stop)
        with start_services(
            graph=graph, project_root=args.project_root, log_dir=args.log_dir
        ):
            started = True
            print(json.dumps(graph.direct_services.as_urls(), indent=2), flush=True)
            stopped.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
