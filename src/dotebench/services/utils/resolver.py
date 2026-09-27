"""Resolve service declarations into a scoped, deduplicated dependency graph."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def _command(raw: Mapping[str, Any]) -> list[str]:
    command = raw.get("command", [])
    if not isinstance(command, list) or not all(
        type(part) in (str, int, float) for part in command
    ):
        raise ValueError("Service command must be an argument array")
    return [str(part) for part in command]


def _url_for(raw: Mapping[str, Any]) -> str:
    explicit = raw.get("url")
    if explicit:
        return str(explicit).rstrip("/")
    scheme = str(raw.get("scheme") or "http")
    host = str(raw.get("host") or "127.0.0.1")
    port = raw.get("port")
    base_path = str(raw.get("base_path") or "").strip("/")
    netloc = f"{host}:{port}" if port not in (None, "") else host
    url = f"{scheme}://{netloc}"
    if base_path:
        url = f"{url}/{base_path}"
    return url.rstrip("/")


def _health_url_for(raw: Mapping[str, Any], url: str) -> str:
    explicit = raw.get("health_url")
    if explicit:
        return str(explicit)
    health_path = str(raw.get("health_path") or "/health")
    if not health_path.startswith("/"):
        health_path = "/" + health_path
    return url.rstrip("/") + health_path


@dataclass(frozen=True)
class ServiceCaller:
    """One scoped dependency edge that references a service instance."""

    caller_instance_key: str | None
    dep_ref_key: str
    config_path: str


@dataclass
class ResolvedServiceNode:
    """One deduplicated service instance in a resolved graph."""

    instance_key: str
    config_key: str
    url: str
    health_url: str
    command: list[str]
    env: dict[str, str]
    gpu_ids: str
    port: str
    startup_timeout_sec: float
    startup_sleep_sec: float
    graph_path: str
    dependencies: dict[str, ResolvedServiceNode] = field(default_factory=dict)
    callers: list[ServiceCaller] = field(default_factory=list)

    @property
    def process_env(self) -> dict[str, str]:
        env = dict(self.env)
        env["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        return env

    @property
    def process_name(self) -> str:
        return self.graph_path.replace(".", "_").replace(":", "_").replace("/", "_")

    def command_spec(self) -> dict[str, Any] | None:
        if not self.command:
            return None
        return {
            "name": self.process_name,
            "instance_key": self.instance_key,
            "graph_path": self.graph_path,
            "command": self.command,
            "env": self.process_env,
            "startup_timeout_sec": self.startup_timeout_sec,
            "startup_sleep_sec": self.startup_sleep_sec,
            "health_url": self.health_url,
            "port": self.port,
        }


@dataclass(frozen=True)
class ResolvedServiceEndpoint:
    """One direct service endpoint exposed to a graph consumer."""

    dep_ref_key: str
    instance_key: str
    url: str


@dataclass(frozen=True)
class ResolvedServiceRoots:
    """Opaque direct-dependency view derived from resolved graph roots.

    Consumers can address only dependencies declared at their own root scope.
    Nested service dependencies remain private to the service that declared them.
    """

    endpoints: Mapping[str, ResolvedServiceEndpoint]

    @classmethod
    def from_nodes(
        cls,
        roots: Mapping[str, ResolvedServiceNode],
    ) -> ResolvedServiceRoots:
        return cls(
            endpoints=MappingProxyType(
                {
                    dep_ref_key: ResolvedServiceEndpoint(
                        dep_ref_key=dep_ref_key,
                        instance_key=node.instance_key,
                        url=node.url,
                    )
                    for dep_ref_key, node in roots.items()
                }
            )
        )

    def __contains__(self, dep_ref_key: object) -> bool:
        return dep_ref_key in self.endpoints

    def url(self, dep_ref_key: str) -> str:
        try:
            return self.endpoints[dep_ref_key].url
        except KeyError as exc:
            raise KeyError(
                f"direct service dependency {dep_ref_key!r} is not declared"
            ) from exc

    def optional_url(self, dep_ref_key: str) -> str | None:
        endpoint = self.endpoints.get(dep_ref_key)
        return endpoint.url if endpoint is not None else None

    def keys(self) -> tuple[str, ...]:
        return tuple(self.endpoints)

    def as_urls(self) -> dict[str, str]:
        return {key: endpoint.url for key, endpoint in self.endpoints.items()}


@dataclass(frozen=True)
class ResolvedServiceGraph:
    """A fully resolved runtime graph and its caller-facing root projection."""

    roots: Mapping[str, ResolvedServiceNode]
    instances: Mapping[str, ResolvedServiceNode]

    @property
    def direct_services(self) -> ResolvedServiceRoots:
        return ResolvedServiceRoots.from_nodes(self.roots)

    def command_specs(self) -> list[dict[str, Any]]:
        ordered: list[dict[str, Any]] = []
        visited: set[str] = set()

        def visit(node: ResolvedServiceNode) -> None:
            if node.instance_key in visited:
                return
            for dependency in node.dependencies.values():
                visit(dependency)
            visited.add(node.instance_key)
            spec = node.command_spec()
            if spec is not None:
                ordered.append(spec)

        for root in self.roots.values():
            visit(root)
        for node in self.instances.values():
            visit(node)
        return ordered

    def select_roots(self, *dep_ref_keys: str) -> ResolvedServiceGraph:
        """Return only selected direct roots and their private dependencies."""
        if not dep_ref_keys:
            raise ValueError("Select at least one service root")
        roots: dict[str, ResolvedServiceNode] = {}
        instances: dict[str, ResolvedServiceNode] = {}

        def collect(node: ResolvedServiceNode) -> None:
            prior = instances.get(node.instance_key)
            if prior is not None:
                if prior is not node:
                    raise ValueError(f"Conflicting service instance: {node.instance_key}")
                return
            instances[node.instance_key] = node
            for dependency in node.dependencies.values():
                collect(dependency)

        for key in dep_ref_keys:
            try:
                node = self.roots[key]
            except KeyError as exc:
                raise KeyError(f"Unknown service root: {key}") from exc
            roots[key] = node
            collect(node)
        return ResolvedServiceGraph(
            roots=MappingProxyType(roots),
            instances=MappingProxyType(dict(sorted(instances.items()))),
        )

    @property
    def health_urls(self) -> list[str]:
        return [
            str(spec["health_url"])
            for spec in self.command_specs()
            if spec.get("health_url")
        ]

    @property
    def startup_timeout_sec(self) -> float:
        return max(
            (float(spec["startup_timeout_sec"]) for spec in self.command_specs()),
            default=0.0,
        )


def _iter_raw_edges(
    services: Mapping[str, Any],
) -> Iterable[tuple[str | None, str, str, Mapping[str, Any]]]:
    def visit(
        caller_instance_key: str | None,
        dep_ref_key: str,
        config_path: str,
        raw: Mapping[str, Any],
    ) -> Iterable[tuple[str | None, str, str, Mapping[str, Any]]]:
        yield caller_instance_key, dep_ref_key, config_path, raw
        instance_key = str(raw.get("instance_key") or "")
        dependencies = raw.get("dependencies") or {}
        if not isinstance(dependencies, Mapping):
            raise ValueError(f"service {config_path!r} dependencies must be a mapping")
        for child_ref, child_raw in dependencies.items():
            if not isinstance(child_raw, Mapping):
                raise ValueError(
                    f"service dependency {config_path}.dependencies.{child_ref} must be a mapping"
                )
            child_path = f"{config_path}.dependencies.{child_ref}"
            yield from visit(instance_key, str(child_ref), child_path, child_raw)

    for dep_ref_key, raw in services.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"service {dep_ref_key!r} must be a mapping")
        yield from visit(None, str(dep_ref_key), str(dep_ref_key), raw)


def _new_node(raw: Mapping[str, Any], *, config_path: str) -> ResolvedServiceNode:
    if "managed" in raw:
        raise ValueError("All declared services must be launched by doteBench")
    instance_key = str(raw.get("instance_key") or "")
    config_key = str(raw.get("config_key") or "")
    if not instance_key:
        raise ValueError(f"service {config_path!r} is missing instance_key")
    if not config_key:
        raise ValueError(f"service {config_path!r} is missing config_key")
    url = _url_for(raw)
    gpu_ids = raw.get("gpu_ids", "")
    return ResolvedServiceNode(
        instance_key=instance_key,
        config_key=config_key,
        url=url,
        health_url=_health_url_for(raw, url),
        command=_command(raw),
        env={str(key): str(value) for key, value in dict(raw.get("env") or {}).items()},
        gpu_ids="" if gpu_ids is None else str(gpu_ids),
        port=str(raw.get("port")) if raw.get("port") not in (None, "") else "",
        startup_timeout_sec=float(raw.get("startup_timeout_sec", 300.0)),
        startup_sleep_sec=float(raw.get("startup_sleep_sec", 0.0)),
        graph_path=config_path,
    )


def _validate_same_instance(
    existing: ResolvedServiceNode,
    raw: Mapping[str, Any],
    *,
    config_path: str,
) -> None:
    config_key = str(raw.get("config_key") or "")
    command = _command(raw)
    if existing.config_key != config_key:
        raise ValueError(
            "conflicting config_key values resolve to the same service instance "
            f"{existing.instance_key}: {existing.config_key} at {existing.graph_path} "
            f"vs {config_key} at {config_path}"
        )
    if existing.command != command:
        raise ValueError(
            "conflicting startup commands resolve to the same service instance "
            f"{existing.instance_key}: {existing.graph_path} vs {config_path}"
        )
    candidate = _new_node(raw, config_path=config_path)
    comparable = (
        "url",
        "health_url",
        "env",
        "gpu_ids",
        "port",
        "startup_timeout_sec",
        "startup_sleep_sec",
    )
    for field_name in comparable:
        if getattr(existing, field_name) != getattr(candidate, field_name):
            raise ValueError(
                f"conflicting {field_name} values resolve to the same service instance "
                f"{existing.instance_key}: {existing.graph_path} vs {config_path}"
            )


def resolve_service_graph(raw: Mapping[str, Any] | None) -> ResolvedServiceGraph:
    """Resolve one worker's raw declarations into a scoped dependency graph."""

    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("service graph must be a mapping")

    instances: dict[str, ResolvedServiceNode] = {}
    edge_targets: dict[tuple[str | None, str], str] = {}
    raw_edges = list(_iter_raw_edges(raw))

    for caller_instance_key, dep_ref_key, config_path, service_raw in raw_edges:
        instance_key = str(service_raw.get("instance_key") or "")
        edge_key = (caller_instance_key, dep_ref_key)
        previous_target = edge_targets.get(edge_key)
        if previous_target is not None and previous_target != instance_key:
            caller_label = caller_instance_key or "<root>"
            raise ValueError(
                f"caller {caller_label!r} dependency {dep_ref_key!r} resolves to "
                f"multiple instances: {previous_target} vs {instance_key} at {config_path}"
            )
        edge_targets[edge_key] = instance_key

        node = instances.get(instance_key)
        if node is None:
            node = _new_node(service_raw, config_path=config_path)
            instances[instance_key] = node
        else:
            _validate_same_instance(node, service_raw, config_path=config_path)
        caller = ServiceCaller(caller_instance_key, dep_ref_key, config_path)
        if caller not in node.callers:
            node.callers.append(caller)

    roots: dict[str, ResolvedServiceNode] = {}
    for caller_instance_key, dep_ref_key, _config_path, service_raw in raw_edges:
        node = instances[str(service_raw["instance_key"])]
        if caller_instance_key is None:
            roots[dep_ref_key] = node
        else:
            parent = instances[caller_instance_key]
            parent.dependencies[dep_ref_key] = node

    _validate_topology(instances)
    return ResolvedServiceGraph(
        roots=MappingProxyType(dict(roots)),
        instances=MappingProxyType(dict(sorted(instances.items()))),
    )


def combine_resolved_service_graphs(
    graphs: Sequence[ResolvedServiceGraph],
) -> ResolvedServiceGraph:
    """Combine worker graphs for one process group without reparsing declarations."""

    roots: dict[str, ResolvedServiceNode] = {}
    instances: dict[str, ResolvedServiceNode] = {}

    def clone(node: ResolvedServiceNode) -> ResolvedServiceNode:
        return ResolvedServiceNode(
            instance_key=node.instance_key,
            config_key=node.config_key,
            url=node.url,
            health_url=node.health_url,
            command=node.command,
            env=dict(node.env),
            gpu_ids=node.gpu_ids,
            port=node.port,
            startup_timeout_sec=node.startup_timeout_sec,
            startup_sleep_sec=node.startup_sleep_sec,
            graph_path=node.graph_path,
            callers=list(node.callers),
        )

    comparable = (
        "config_key",
        "url",
        "health_url",
        "command",
        "env",
        "gpu_ids",
        "port",
        "startup_timeout_sec",
        "startup_sleep_sec",
    )
    for graph_index, graph in enumerate(graphs):
        for instance_key, node in graph.instances.items():
            existing = instances.get(instance_key)
            if existing is None:
                instances[instance_key] = clone(node)
                continue
            for field_name in comparable:
                if getattr(existing, field_name) != getattr(node, field_name):
                    raise ValueError(
                        f"resolved graphs contain conflicting {field_name} values for "
                        f"service instance {instance_key}"
                    )
            for caller in node.callers:
                if caller not in existing.callers:
                    existing.callers.append(caller)

        for key, node in graph.roots.items():
            roots[f"worker_{graph_index}_{key}"] = instances[node.instance_key]

    for graph in graphs:
        for instance_key, source in graph.instances.items():
            target = instances[instance_key]
            for dep_ref_key, dependency in source.dependencies.items():
                existing_dependency = target.dependencies.get(dep_ref_key)
                if (
                    existing_dependency is not None
                    and existing_dependency.instance_key != dependency.instance_key
                ):
                    raise ValueError(
                        f"resolved graphs bind service instance {instance_key} dependency "
                        f"{dep_ref_key!r} to multiple instances"
                    )
                target.dependencies[dep_ref_key] = instances[dependency.instance_key]

    _validate_topology(instances)
    return ResolvedServiceGraph(
        roots=MappingProxyType(roots),
        instances=MappingProxyType(dict(sorted(instances.items()))),
    )


def _validate_topology(instances: Mapping[str, ResolvedServiceNode]) -> None:
    active: set[str] = set()
    visited: set[str] = set()
    bound: dict[tuple[str, int], str] = {}
    from urllib.parse import urlsplit

    for node in instances.values():
        if node.startup_timeout_sec <= 0 or node.startup_sleep_sec < 0:
            raise ValueError(f"invalid startup timing for {node.instance_key}")
        if not node.command:
            raise ValueError(f"service {node.instance_key!r} requires a command")
        parsed = urlsplit(node.url)
        if parsed.port:
            endpoint = (parsed.hostname or "", parsed.port)
            previous = bound.setdefault(endpoint, node.instance_key)
            if previous != node.instance_key:
                raise ValueError(
                    f"multiple service instances bind {endpoint}: {previous}, {node.instance_key}"
                )

    def visit(node: ResolvedServiceNode) -> None:
        if node.instance_key in active:
            raise ValueError(f"service dependency cycle at {node.instance_key!r}")
        if node.instance_key in visited:
            return
        active.add(node.instance_key)
        for dependency in node.dependencies.values():
            visit(dependency)
        active.remove(node.instance_key)
        visited.add(node.instance_key)

    for node in instances.values():
        visit(node)
