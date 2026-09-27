"""Configuration-driven service graph and process management."""

from .utils.config import load_service_bundle
from .utils.processes import ManagedServiceGroup, start_services, wait_for_urls
from .utils.resolver import (
    ResolvedServiceGraph,
    ResolvedServiceRoots,
    combine_resolved_service_graphs,
    resolve_service_graph,
)

__all__ = [
    "load_service_bundle",
    "start_services",
    "wait_for_urls",
    "ManagedServiceGroup",
    "ResolvedServiceGraph",
    "ResolvedServiceRoots",
    "resolve_service_graph",
    "combine_resolved_service_graphs",
]
