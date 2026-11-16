from pathlib import Path

from atlas_posthoc_plus.cli import main
from atlas_posthoc_plus.config import load_config
from atlas_posthoc_plus.registry import WORKFLOWS


def test_init_config_resolves_cousub_workflow_arguments(tmp_path: Path) -> None:
    config_path = tmp_path / "analysis.toml"

    assert main(["init", str(config_path)]) == 0

    config = load_config(config_path)
    args = WORKFLOWS["municipal-histograms-compliant"].build_args(config)
    assert args[args.index("--municipality-mode") + 1] == "COUSUB"
    assert args[args.index("--graph-path") + 1] == str(tmp_path / "inputs/graph.json")
    assert args[args.index("--output-namespace") + 1] == "cousub"


def test_run_dry_run_does_not_create_output_directory(
    tmp_path: Path,
    capsys,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "blocks.shp").touch()
    (project / "precincts.shp").touch()

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[project]
root = "project"

[inputs]
blocks = "blocks.shp"
precincts = "precincts.shp"

[outputs]
root = "results"
""",
        encoding="utf-8",
    )

    code = main(
        [
            "--config",
            str(config_path),
            "run",
            "build-crosswalk",
            "--dry-run",
            "--",
            "--min-weight",
            "0.1",
        ]
    )

    assert code == 0
    assert not (project / "results").exists()
    assert "--min-weight 0.1" in capsys.readouterr().out
