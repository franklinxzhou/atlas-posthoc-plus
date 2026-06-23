from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from atlas_posthoc_plus.config import ProjectConfig
from atlas_posthoc_plus.registry import WorkflowSpec


def _subprocess_environment() -> dict[str, str]:
    """Keep the package and its AtlasIO compatibility module importable."""
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1])
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not existing else os.pathsep.join((source_root, existing))
    )
    return environment


@dataclass(frozen=True)
class PreparedWorkflow:
    spec: WorkflowSpec
    argv: tuple[str, ...]
    cwd: Path

    @property
    def command(self) -> tuple[str, ...]:
        return (sys.executable, "-m", self.spec.module, *self.argv)

    def display_command(self) -> str:
        return shlex.join(self.command)


def prepare_workflow(
    spec: WorkflowSpec,
    config: ProjectConfig,
    *,
    extra_args: Sequence[str] = (),
) -> PreparedWorkflow:
    argv = spec.build_args(config)
    argv.extend(arg for arg in extra_args if arg != "--")
    return PreparedWorkflow(spec=spec, argv=tuple(argv), cwd=config.root)


def run_prepared(
    prepared: PreparedWorkflow,
    *,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    if dry_run:
        print(prepared.display_command())
        return None

    return subprocess.run(
        prepared.command,
        cwd=prepared.cwd,
        check=False,
        text=True,
        env=_subprocess_environment(),
    )


def run_workflow(
    spec: WorkflowSpec,
    config: ProjectConfig,
    *,
    extra_args: Sequence[str] = (),
    dry_run: bool = False,
) -> int:
    if not dry_run:
        config.output_root.mkdir(parents=True, exist_ok=True)
    prepared = prepare_workflow(spec, config, extra_args=extra_args)
    result = run_prepared(prepared, dry_run=dry_run)
    return 0 if result is None else result.returncode
