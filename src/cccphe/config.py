from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration is not a mapping: {path}")
    parent = cfg.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        cfg = _deep_merge(load_config(parent_path), cfg)
    cfg["_config_path"] = str(path)
    storage_root = str(Path(cfg["project"]["storage_root"]).expanduser())
    code_root = str(Path(cfg["project"]["code_root"]).expanduser())
    for key, value in cfg.get("paths", {}).items():
        if value is not None:
            cfg["paths"][key] = str(value).format(
                storage_root=storage_root, code_root=code_root
            )
    return cfg


def ensure_storage(cfg: dict[str, Any]) -> None:
    for value in cfg.get("paths", {}).values():
        if value:
            Path(value).mkdir(parents=True, exist_ok=True)


def path_for(cfg: dict[str, Any], section: str, *parts: str) -> Path:
    return Path(cfg["paths"][section]).joinpath(*parts)


def configured_path(
    cfg: dict[str, Any], key: str, default_section: str, *default_parts: str
) -> Path:
    override = cfg.get("analysis_paths", {}).get(key)
    return Path(override) if override else path_for(cfg, default_section, *default_parts)
