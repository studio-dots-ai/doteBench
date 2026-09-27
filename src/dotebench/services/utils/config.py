"""Compose application settings and generic service declarations with Hydra."""

from pathlib import Path
from typing import Sequence

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .resolver import ResolvedServiceGraph, resolve_service_graph


def load_config(
    path: str | Path,
    *,
    overrides: Sequence[str] = (),
    config_root: str | Path | None = None,
) -> dict:
    path = Path(path).resolve()
    root = Path(config_root).resolve() if config_root else path.parent
    name = str(path.relative_to(root).with_suffix(""))
    with initialize_config_dir(config_dir=str(root), version_base="1.3"):
        config = compose(config_name=name, overrides=list(overrides))
    payload = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(payload, dict):
        raise ValueError("Configuration must compose to a mapping")
    return payload


def load_service_bundle(
    path: str | Path,
    *,
    overrides: Sequence[str] = (),
    config_root: str | Path | None = None,
) -> ResolvedServiceGraph:
    payload = load_config(path, overrides=overrides, config_root=config_root)
    services = payload.get("services", payload)
    if "instance_key" in services:
        services = {services.get("config_key", path.stem): services}
    return resolve_service_graph(services)
