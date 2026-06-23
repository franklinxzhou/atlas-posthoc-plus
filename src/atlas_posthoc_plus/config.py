from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when a project configuration is missing or malformed."""


def _nested_get(data: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = data
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


@dataclass(frozen=True)
class ProjectConfig:
    """Loaded TOML configuration with paths anchored at the analysis project root."""

    path: Path
    data: Mapping[str, Any]
    root: Path

    def get(self, dotted_key: str, default: Any = None) -> Any:
        return _nested_get(self.data, dotted_key, default)

    def require(self, dotted_key: str) -> Any:
        value = self.get(dotted_key)
        if value is None or value == "":
            raise ConfigError(f"configuration value {dotted_key!r} is required")
        return value

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.root / path
        return path.resolve()

    def path_value(self, dotted_key: str, default: str | Path | None = None) -> Path | None:
        value = self.get(dotted_key, default)
        if value is None or value == "":
            return None
        return self.resolve_path(value)

    def path_list(self, dotted_key: str) -> list[Path]:
        values = self.get(dotted_key, [])
        if not isinstance(values, list):
            raise ConfigError(f"configuration value {dotted_key!r} must be a list")
        return [self.resolve_path(value) for value in values]

    @property
    def output_root(self) -> Path:
        value = self.get("outputs.root", "outputs")
        return self.resolve_path(value)

    def output_path(self, relative: str | Path) -> Path:
        return (self.output_root / relative).resolve()


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"configuration file does not exist: {config_path}")

    with config_path.open("rb") as stream:
        data = tomllib.load(stream)

    root_value = _nested_get(data, "project.root", ".")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    root = root.resolve()

    if not root.is_dir():
        raise ConfigError(f"project.root is not a directory: {root}")

    return ProjectConfig(path=config_path, data=data, root=root)
