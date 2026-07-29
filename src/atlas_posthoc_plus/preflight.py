from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from atlas_posthoc_plus.config import ConfigError, ProjectConfig
from atlas_posthoc_plus.registry import Binding, OutputDefault, WorkflowSpec


@dataclass(frozen=True)
class Check:
    ok: bool
    label: str
    detail: str


def _check_import(module: str) -> Check:
    try:
        imported = importlib.import_module(module)
    except Exception as exc:
        return Check(False, f"import {module}", f"{type(exc).__name__}: {exc}")
    version = getattr(imported, "__version__", None)
    return Check(True, f"import {module}", str(version or "available"))


def _configured_value(config: ProjectConfig, binding: Binding):
    value = config.get(binding.key)
    return binding.default if value is None else value


def _input_paths(config: ProjectConfig, spec: WorkflowSpec) -> Iterable[tuple[str, Path]]:
    for binding in spec.bindings:
        if binding.key.startswith("outputs."):
            continue
        value = _configured_value(config, binding)
        if value is None:
            continue
        if isinstance(value, OutputDefault):
            yield binding.key, config.output_path(value.relative)
            continue
        if binding.kind == "path":
            yield binding.key, config.resolve_path(value)
        elif binding.kind == "paths":
            for index, item in enumerate(value):
                yield f"{binding.key}[{index}]", config.resolve_path(item)
        elif binding.kind == "path_mapping":
            for label, item in value.items():
                yield f"{binding.key}.{label}", config.resolve_path(item)


def check_workflow(config: ProjectConfig, spec: WorkflowSpec) -> list[Check]:
    checks: list[Check] = []

    try:
        spec.build_args(config)
    except ConfigError as exc:
        checks.append(Check(False, "configuration", str(exc)))
        return checks
    checks.append(Check(True, "configuration", f"{spec.name} arguments resolved"))

    try:
        module = importlib.import_module(spec.module)
        checks.append(Check(True, "workflow module", str(module.__file__)))
    except Exception as exc:
        checks.append(
            Check(False, "workflow module", f"{type(exc).__name__}: {exc}")
        )
        module = None

    if module is not None:
        preflight_paths = getattr(module, "preflight_paths", None)
        if callable(preflight_paths):
            try:
                for key, path in preflight_paths(config):
                    checks.append(Check(path.exists(), str(key), str(path)))
            except ConfigError as exc:
                checks.append(Check(False, "workflow configuration", str(exc)))

    for key, path in _input_paths(config, spec):
        checks.append(Check(path.exists(), key, str(path)))

    graph_path = config.path_value("inputs.graph_json")
    if graph_path is not None and any(
        binding.key == "inputs.graph_json" for binding in spec.bindings
    ):
        try:
            from atlas_posthoc.io.graph_reader import read_graph_json

            graph = read_graph_json(graph_path)
            checks.append(
                Check(
                    True,
                    "atlas-posthoc graph reader",
                    f"{len(graph.nodes):,} nodes; {len(graph.columns)} columns",
                )
            )
        except Exception as exc:
            checks.append(
                Check(False, "atlas-posthoc graph reader", f"{type(exc).__name__}: {exc}")
            )

    if any(binding.key == "chains.files" for binding in spec.bindings):
        try:
            from atlas_posthoc.io.atlas_reader import read_atlas_header

            for chain_path in config.path_list("chains.files"):
                header = read_atlas_header(chain_path)
                detail = header.description or header.atlas_param_type or "header parsed"
                checks.append(Check(True, f"AtlasIO header {chain_path.name}", str(detail)))
        except Exception as exc:
            checks.append(
                Check(False, "atlas-posthoc AtlasIO reader", f"{type(exc).__name__}: {exc}")
            )

    return checks


def dependency_checks() -> list[Check]:
    return [
        _check_import("atlas_posthoc"),
        _check_import("numpy"),
        _check_import("pandas"),
        _check_import("geopandas"),
        _check_import("matplotlib"),
        _check_import("openpyxl"),
    ]


def print_checks(checks: Iterable[Check]) -> bool:
    all_ok = True
    for check in checks:
        marker = "OK" if check.ok else "FAIL"
        print(f"[{marker:4}] {check.label}: {check.detail}")
        all_ok = all_ok and check.ok
    return all_ok
