from pathlib import Path

from atlas_posthoc_plus.cli import main


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
