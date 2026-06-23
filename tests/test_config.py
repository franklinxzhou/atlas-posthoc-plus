from pathlib import Path

import pytest

from atlas_posthoc_plus.config import ConfigError, load_config


def test_load_config_resolves_paths_from_project_root(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[project]\nroot = "project"\n[outputs]\nroot = "results"\n',
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.root == project.resolve()
    assert config.output_root == (project / "results").resolve()


def test_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.toml")
