from __future__ import annotations

import argparse
import sys
from pathlib import Path

from atlas_posthoc_plus.config import ConfigError, load_config
from atlas_posthoc_plus.preflight import check_workflow, dependency_checks, print_checks
from atlas_posthoc_plus.registry import (
    SUITES,
    WORKFLOW_ALIASES,
    WORKFLOWS,
    get_workflow,
    workflow_for_mode,
)
from atlas_posthoc_plus.runner import prepare_workflow, run_workflow
from atlas_posthoc_plus.suite import run_suite
from atlas_posthoc_plus.template import EXAMPLE_CONFIG


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlas-posthoc-plus",
        description="Run project-specific AtlasIO post-processing workflows from one TOML file.",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="atlas-posthoc-plus.toml",
        help="Project TOML configuration (default: atlas-posthoc-plus.toml).",
    )
    figure_mode = parser.add_mutually_exclusive_group()
    figure_mode.add_argument(
        "--paper-mode",
        dest="paper_mode",
        action="store_true",
        default=None,
        help="Write publication-style vector PDFs with consistent typography and panel labels.",
    )
    figure_mode.add_argument(
        "--no-paper-mode",
        dest="paper_mode",
        action="store_false",
        help="Override figures.paper_mode and retain normal screen figures.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available workflows and pipelines.")

    workflow_choices = sorted(WORKFLOWS)
    workflow_aliases = {"type": lambda name: WORKFLOW_ALIASES.get(name, name)}
    show = subparsers.add_parser("show", help="Print the resolved command for a workflow.")
    show.add_argument("workflow", choices=workflow_choices, **workflow_aliases)

    doctor = subparsers.add_parser("doctor", help="Validate dependencies and workflow inputs.")
    doctor.add_argument("workflow", nargs="?", choices=workflow_choices, **workflow_aliases)

    run = subparsers.add_parser("run", help="Run one workflow.")
    run.add_argument("workflow", choices=workflow_choices, **workflow_aliases)
    run.add_argument("--dry-run", action="store_true")

    suite = subparsers.add_parser("suite", help="Run a named analysis suite.")
    suite.add_argument("suite", choices=sorted(SUITES))
    suite.add_argument("--dry-run", action="store_true")
    suite.add_argument("--keep-going", action="store_true")

    pipeline = subparsers.add_parser(
        "pipeline",
        help="Deprecated alias for `suite`.",
    )
    pipeline.add_argument("pipeline", choices=sorted(SUITES))
    pipeline.add_argument("--dry-run", action="store_true")
    pipeline.add_argument("--keep-going", action="store_true")

    init = subparsers.add_parser("init", help="Write a documented example configuration.")
    init.add_argument("path", nargs="?", default="atlas-posthoc-plus.toml")
    init.add_argument("--force", action="store_true")
    return parser


def _list() -> None:
    categories: dict[str, list] = {}
    for spec in WORKFLOWS.values():
        categories.setdefault(spec.category, []).append(spec)

    for category in sorted(categories):
        print(f"{category}:")
        for spec in sorted(categories[category], key=lambda item: item.name):
            suffix = " [optional/external]" if spec.legacy else ""
            print(f"  {spec.name:<38} {spec.summary}{suffix}")
        print()

    print("suites:")
    for name, workflows in SUITES.items():
        print(f"  {name:<12} {' -> '.join(workflows)}")


def _init(path: str, *, force: bool) -> int:
    destination = Path(path).expanduser().resolve()
    if destination.exists() and not force:
        raise ConfigError(f"refusing to overwrite existing file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(EXAMPLE_CONFIG, encoding="utf-8")
    print(destination)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args, extra_args = parser.parse_known_args(argv)
    if extra_args and args.command != "run":
        parser.error(f"unrecognized arguments: {' '.join(extra_args)}")
    try:
        if args.command == "list":
            _list()
            return 0
        if args.command == "init":
            return _init(args.path, force=args.force)

        config = load_config(args.config)
        municipality_mode = str(config.get("municipality.mode", "")).upper()

        if args.command == "show":
            prepared = prepare_workflow(
                get_workflow(args.workflow, municipality_mode=municipality_mode),
                config,
                paper_mode=args.paper_mode,
            )
            print(f"cwd: {prepared.cwd}")
            print(prepared.display_command())
            return 0

        if args.command == "doctor":
            ok = print_checks(dependency_checks())
            if args.workflow:
                specs = [get_workflow(args.workflow, municipality_mode=municipality_mode)]
            else:
                active = {}
                for name in sorted(WORKFLOWS):
                    spec = workflow_for_mode(WORKFLOWS[name], municipality_mode)
                    if spec is not None:
                        active.setdefault(spec.name, spec)
                specs = active.values()
            for spec in specs:
                print(f"\n[{spec.name}]")
                ok = print_checks(check_workflow(config, spec)) and ok
            return 0 if ok else 1

        if args.command == "run":
            return run_workflow(
                get_workflow(args.workflow, municipality_mode=municipality_mode),
                config,
                extra_args=extra_args,
                dry_run=args.dry_run,
                paper_mode=args.paper_mode,
            )

        if args.command in {"suite", "pipeline"}:
            suite_name = args.suite if args.command == "suite" else args.pipeline
            return run_suite(
                suite_name,
                SUITES[suite_name],
                config,
                dry_run=args.dry_run,
                keep_going=args.keep_going,
                paper_mode=args.paper_mode,
            )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
