from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Sequence

from atlas_posthoc_plus.config import ProjectConfig
from atlas_posthoc_plus.registry import WORKFLOWS
from atlas_posthoc_plus.runner import (
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

    config.output_root.mkdir(parents=True, exist_ok=True)
    started_at = _timestamp()
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
            env=_subprocess_environment(),
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
