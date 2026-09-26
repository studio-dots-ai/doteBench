"""Command-line dependency assembly."""

import argparse
import importlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path

from dotebench import dataset
from dotebench.candidates.registry import CANDIDATES
from dotebench.execution import EvaluationRunner, GenerationRunner
from dotebench.models.base import CompiledCandidate
from dotebench.storage import RunStore


def main(argv=None):
    parser = argparse.ArgumentParser(prog="dotebench")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-data", "generate", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--category")
        command.add_argument("--shard")
        command.add_argument("--data-root", type=Path)
        command.add_argument("--audio-root", required=True, type=Path)
        if name != "validate-data":
            command.add_argument("--run", required=True, type=Path)
            if name == "generate":
                command.add_argument(
                    "--candidate",
                    choices=tuple(n.replace("_", "-") for n in CANDIDATES),
                    default="identity",
                )
            command.add_argument("--config", type=Path)
            command.add_argument("--override", action="append", default=[])
        if name == "generate":
            command.add_argument(
                "--compiler", choices=("one_take", "sequential"), default="one_take"
            )
    preparation = commands.add_parser("prepare-evaluation-models")
    preparation.add_argument("--model-root", required=True, type=Path)
    from dotebench.evaluation.models import model_locks

    preparation.add_argument("--model", choices=tuple(model_locks()))
    preparation.add_argument("--check", action="store_true")
    materialization_preparation = commands.add_parser("prepare-materialization-models")
    materialization_preparation.add_argument("--model-root", required=True, type=Path)
    from dotebench.materialization.assets import model_locks as materialization_locks

    materialization_preparation.add_argument(
        "--model", choices=tuple(materialization_locks())
    )
    materialization_preparation.add_argument(
        "--accept-license", action="append", default=[]
    )
    materialization_preparation.add_argument("--check", action="store_true")
    materialize = commands.add_parser("materialize-data")
    materialize.add_argument("--config", type=Path)
    materialize.add_argument("--override", action="append", default=[])
    materialize.add_argument("--data-root", type=Path)
    materialize.add_argument("--output-root", required=True, type=Path)
    materialize.add_argument("--bundle-root", required=True, type=Path)
    materialize.add_argument("--common-voice-root", required=True, type=Path)
    materialize.add_argument("--timeout", type=float, default=1800)
    verify = commands.add_parser("verify-materialization")
    verify.add_argument("--data-root", required=True, type=Path)
    verify.add_argument("--reference", type=Path)
    report = commands.add_parser("report")
    report.add_argument("--run", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare-materialization-models":
        from dotebench.materialization.assets import (
            model_directory,
            prepare_model,
            prepared_identity,
        )

        identities = {}
        accepted = set(args.accept_license)
        selected_models = (
            (args.model,)
            if args.model
            else tuple(
                name
                for name, lock in materialization_locks().items()
                if lock["role"] == "generate"
            )
        )
        for model in selected_models:
            path = model_directory(args.model_root, model)
            identities[model] = (
                prepared_identity(model, path, full_check=True)
                if args.check
                else prepare_model(model, path, accepted=accepted)
            )
        print(json.dumps(identities, indent=2))
        return 0
    if args.command == "verify-materialization":
        from dotebench.materialization.verification import verify_materialization

        print(
            json.dumps(
                verify_materialization(args.data_root, reference=args.reference),
                indent=2,
            )
        )
        return 0
    if args.command == "materialize-data":
        from dotebench.materialization.manifest import PROVIDERS
        from dotebench.materialization.orchestrator import Materializer
        from dotebench.services.utils.config import load_config
        from dotebench.services.utils.processes import start_services
        from dotebench.services.utils.resolver import resolve_service_graph

        config_path = args.config or (
            dataset.resources() / "configs" / "materialization.yaml"
        )
        output_key = "DOTEBENCH_SERVICE_OUTPUTS"
        default_output = output_key not in os.environ
        if default_output:
            os.environ[output_key] = str((args.output_root / "services").resolve())
        try:
            payload = load_config(config_path, overrides=args.override)
        finally:
            if default_output:
                os.environ.pop(output_key, None)
        graph = resolve_service_graph(payload.get("services", {}))
        worker = Materializer(
            data_root=args.data_root or dataset.resources(),
            output_root=args.output_root,
            bundle_root=args.bundle_root,
            upstream_root=args.common_voice_root,
            timeout=args.timeout,
        )
        worker.materialize_static()
        for provider in PROVIDERS:
            if not any(
                a.provider == provider for a in worker.inventory.assets.values()
            ):
                continue
            selected = graph.select_roots(provider)
            with start_services(
                graph=selected,
                project_root=Path.cwd(),
                log_dir=args.output_root / "services" / provider,
            ):
                worker.materialize_provider(
                    provider, selected.direct_services.url(provider)
                )
        changed = any(
            record["sha256"] != record["reference_sha256"]
            for record in worker.records.values()
        )
        if changed:
            selected = graph.select_roots("qwen3_aligner")
            with start_services(
                graph=selected,
                project_root=Path.cwd(),
                log_dir=args.output_root / "services" / "qwen3_aligner",
            ):
                worker.register_aligner(selected.direct_services.url("qwen3_aligner"))
                receipt = worker.finalize(
                    aligner_url=selected.direct_services.url("qwen3_aligner")
                )
        else:
            receipt = worker.finalize(aligner_url=None)
        print(json.dumps(receipt, indent=2))
        return 0
    if args.command == "prepare-evaluation-models":
        from dotebench.evaluation.models import (
            model_directory,
            prepare_model,
            prepared_identity,
        )

        identities = {}
        for model in (args.model,) if args.model else model_locks():
            path = model_directory(args.model_root, model)
            identities[model] = (
                prepared_identity(model, path, full_check=True)
                if args.check
                else prepare_model(model, path)
            )
        print(json.dumps(identities, indent=2))
        return 0
    if args.command == "report":
        store = RunStore(args.run, Path("."))
        config = store.read_json("run.json")
        if config.get("evaluation_status") != "complete":
            parser.error("Formal evaluation is incomplete")
        summary = store.read_json("summary.json")
        if summary.get("status") != "complete":
            parser.error("Formal summary is incomplete")
        from dotebench.evaluation.evaluator import protocol_version

        if summary.get("metrics", {}).get("protocol_version") != protocol_version():
            parser.error("Unsupported evaluation protocol version")
        print(json.dumps(summary, indent=2))
        return 0
    entries = dataset.select_shards(args.category, args.shard, data_root=args.data_root)
    selection = [
        {
            **e,
            "manifest_sha256": dataset.sha256(dataset.manifest_path(e, args.data_root)),
        }
        for e in entries
    ]
    loaded = {}

    def load():
        if not loaded:
            for entry in entries:
                loaded[(entry["category"], entry["shard"])] = dataset.load_cases(
                    entry,
                    data_root=args.data_root,
                    audio_root=args.audio_root,
                    verify_audio=True,
                )
        return tuple(case for cases in loaded.values() for case in cases)

    if args.command == "validate-data":
        cases = load()
        if len({c.id for c in cases}) != len(cases):
            raise ValueError("Duplicate IDs across shards")
        print(json.dumps({"shards": len(entries), "cases": len(cases)}))
        return 0
    from dotebench.evaluation.evaluator import DoteBenchEvaluator
    from dotebench.services.utils.config import load_config
    from dotebench.services.utils.processes import start_services
    from dotebench.services.utils.resolver import resolve_service_graph

    config_path = args.config
    if config_path is None and not (
        args.command == "generate" and args.candidate == "identity"
    ):
        filename = (
            args.candidate.replace("-", "_")
            if args.command == "generate"
            else "evaluation_bundle"
        )
        config_path = dataset.resources() / "configs" / (filename + ".yaml")
    overrides = list(args.override)
    if config_path:
        # Supply only the absent environment default; explicit YAML and overrides win.
        output_key = "DOTEBENCH_SERVICE_OUTPUTS"
        default_output = output_key not in os.environ
        if default_output:
            os.environ[output_key] = str(
                (args.run.parent / (args.run.name + "-services")).resolve()
            )
        try:
            payload = load_config(config_path, overrides=overrides)
        finally:
            if default_output:
                os.environ.pop(output_key, None)
    else:
        if overrides:
            parser.error("Overrides require a configuration")
        payload = {"services": {}}
    graph = resolve_service_graph(payload.get("services", {}))
    config = {"dataset_version": dataset.data_version(), "dataset": selection}
    load()
    if args.command == "generate":
        declaration = payload.get("candidate", {})
        if declaration.get("name", args.candidate) != args.candidate:
            parser.error("Candidate selection does not match configuration")
        options = dict(declaration.get("options", {}))
        if {"url", "aligner_url", "compiler"} & options.keys():
            parser.error(
                "URLs come from the service graph; select compiler with --compiler"
            )
        name = args.candidate.replace("-", "_")
        spec = CANDIDATES[name]
        factory = getattr(
            importlib.import_module(f"dotebench.candidates.{name}.adapter"),
            spec.adapter,
        )
        if issubclass(factory, CompiledCandidate):
            options["url"] = graph.direct_services.url("candidate")
            if "aligner" in graph.direct_services:
                options["aligner_url"] = graph.direct_services.url("aligner")
            candidate = factory(compiler=args.compiler, **options)
        else:
            if args.compiler != "one_take":
                parser.error("Identity supports only one_take")
            candidate = factory(**options)
        candidate.compilation_identity()
        config["generation"] = {
            "candidate": args.candidate,
            "compiler": args.compiler,
            "options": options,
        }
        runner = GenerationRunner
        component = {"candidate": candidate}
    else:
        options = dict(payload.get("evaluation", {}))
        if {"urls", "case_groups"} & options.keys():
            parser.error("Evaluation URLs and case groups are supplied by the run")
        options.update(
            urls=graph.direct_services.as_urls(),
            case_groups={
                case.id: {"category": key[0], "shard": key[1]}
                for key, cases in loaded.items()
                for case in cases
            },
        )
        evaluator = DoteBenchEvaluator(**options)
        config["evaluation"] = {"options": options}
        runner = EvaluationRunner
        component = {"evaluator": evaluator}
    config["services"] = {
        key: {
            "config_key": node.config_key,
            "url": node.url,
            "command": node.command,
            "env": node.env,
            "gpu_ids": node.gpu_ids,
        }
        for key, node in graph.instances.items()
    }
    config["software"] = {
        "dotebench": version("dotebench"),
        "python": platform.python_version(),
    }
    with start_services(
        graph=graph,
        project_root=Path.cwd(),
        log_dir=args.run.parent / (args.run.name + "-services"),
    ):
        result = runner(
            load=load,
            store=RunStore(args.run, args.audio_root),
            config=config,
            **component,
        ).run()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0
