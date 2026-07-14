from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Sequence

from atlas_posthoc_plus.config import ProjectConfig
from atlas_posthoc_plus.io.chain_sweep import (
    SWEEP_DIR_ENV,
    SWEEP_LIMIT_ENV,
    materialize_chain_sweep,
)
from atlas_posthoc_plus.registry import WORKFLOWS
from atlas_posthoc_plus.runner import (
    PreparedWorkflow,
    _subprocess_environment,
    prepare_workflow,
)


@dataclass(frozen=True)
class SuiteResult:
    workflow: str
    command: str
    returncode: int
    elapsed_seconds: float


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _last_int_option(
    argv: Sequence[str],
    flag: str,
    default: int | None,
) -> int | None:
    value = default
    for index, argument in enumerate(argv):
        if argument == flag and index + 1 < len(argv):
            value = int(argv[index + 1])
        elif argument.startswith(f"{flag}="):
            value = int(argument.split("=", 1)[1])
    return value


def _last_option(argv: Sequence[str], flags: Sequence[str], default: str) -> str:
    value = default
    for index, argument in enumerate(argv):
        for flag in flags:
            if argument == flag and index + 1 < len(argv):
                value = argv[index + 1]
            elif argument.startswith(f"{flag}="):
                value = argument.split("=", 1)[1]
    return value


def _shared_option(
    workflows: Sequence[PreparedWorkflow],
    flags: Sequence[str],
    default: str,
) -> str:
    values = {_last_option(workflow.argv, flags, default) for workflow in workflows}
    if len(values) != 1:
        raise ValueError(
            f"suite chain workflows must use one shared {flags[0]} value; got {sorted(values)}"
        )
    return values.pop()


def _cache_record_limit(
    workflows: Sequence[PreparedWorkflow],
    configured_max: int | None,
) -> int | None:
    required: list[int] = []
    for workflow in workflows:
        max_maps = _last_int_option(
            workflow.argv,
            "--max-maps-per-chain",
            configured_max,
        )
        if max_maps is None:
            return None
        thin_step = _last_int_option(workflow.argv, "--thin-step", 1)
        if thin_step is None or thin_step <= 0:
            raise ValueError("--thin-step must be a positive integer")
        required.append(0 if max_maps <= 0 else (max_maps - 1) * thin_step + 1)
    return max(required, default=0)


def run_suite(
    name: str,
    workflows: Sequence[str],
    config: ProjectConfig,
    *,
    dry_run: bool = False,
    keep_going: bool = False,
) -> int:
    prepared = [
        prepare_workflow(WORKFLOWS[workflow_name], config)
        for workflow_name in workflows
    ]

    if dry_run:
        for item in prepared:
            print(f"\n==> {item.spec.name}")
            print(item.display_command())
        return 0

    started_at = _timestamp()
    config.output_root.mkdir(parents=True, exist_ok=True)
    environment = _subprocess_environment()
    environment.pop(SWEEP_DIR_ENV, None)
    environment.pop(SWEEP_LIMIT_ENV, None)
    chain_workflows = [
        item
        for item in prepared
        if any(binding.key == "chains.files" for binding in item.spec.bindings)
    ]
    if chain_workflows:
        cache_dir = config.output_root / ".atlas-chain-cache"
        raw_max = config.get("analysis.max_maps_per_chain")
        configured_max = int(raw_max) if raw_max is not None else None
        max_records = _cache_record_limit(chain_workflows, configured_max)
        node_id_col = _shared_option(chain_workflows, ("--node-id-col",), "id")
        key_col = _shared_option(
            chain_workflows,
            ("--atlas-key-col", "--name-col"),
            "NAME",
        )
        county_col = _shared_option(chain_workflows, ("--county-col",), "COUNTY")
        max_counties = {
            _last_int_option(
                workflow.argv,
                "--max-districts-per-county",
                int(config.get("analysis.max_districts_per_county", 2)),
            )
            for workflow in chain_workflows
        }
        if len(max_counties) != 1:
            raise ValueError(
                "suite chain workflows must use one shared county-compliance threshold"
            )
        max_county = max_counties.pop()
        if max_county is None:
            raise ValueError("max districts per county must be configured")
        graph_path = config.path_value("inputs.graph_json")
        if graph_path is None:
            raise ValueError("chain suites require inputs.graph_json")

        print(f"\nPreparing one-pass chain sweep: {cache_dir}", flush=True)
        for chain_path in config.path_list("chains.files"):
            reused = materialize_chain_sweep(
                chain_path,
                graph_path,
                cache_dir,
                node_id_col=node_id_col,
                key_col=key_col,
                county_col=county_col,
                max_districts_per_county=max_county,
                max_records=max_records,
            )
            action = "Reusing" if reused else "Cached"
            print(f"  {action}: {chain_path}", flush=True)
        environment[SWEEP_DIR_ENV] = str(cache_dir)
        if max_records is not None:
            environment[SWEEP_LIMIT_ENV] = str(max_records)

    results: list[SuiteResult] = []
    exit_code = 0

    for item in prepared:
        print(f"\n==> {item.spec.name}", flush=True)
        started = time.monotonic()
        process = subprocess.run(
            item.command,
            cwd=item.cwd,
            check=False,
            text=True,
            env=environment,
        )
        elapsed = time.monotonic() - started
        results.append(
            SuiteResult(
                workflow=item.spec.name,
                command=item.display_command(),
                returncode=process.returncode,
                elapsed_seconds=round(elapsed, 3),
            )
        )

        if process.returncode:
            exit_code = process.returncode
            if not keep_going:
                break

    manifest = {
        "suite": name,
        "config": str(config.path),
        "project_root": str(config.root),
        "output_root": str(config.output_root),
        "started_at": started_at,
        "finished_at": _timestamp(),
        "success": exit_code == 0,
        "results": [asdict(result) for result in results],
    }
    manifest_path = config.output_root / f"suite-{name}-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nSuite manifest: {manifest_path}")
    return exit_code
