import importlib.util
from pathlib import Path

from atlas_posthoc_plus.config import load_config
from atlas_posthoc_plus.registry import WORKFLOWS
from atlas_posthoc_plus.runner import prepare_workflow


def test_all_workflow_modules_are_registered() -> None:
    modules = {spec.module for spec in WORKFLOWS.values()}
    assert len(WORKFLOWS) == 20
    assert len(modules) == 20
    assert all(importlib.util.find_spec(module) is not None for module in modules)


def test_prepare_workflow_builds_absolute_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    for filename in ("blocks.shp", "precincts.shp"):
        (project / filename).touch()

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

    prepared = prepare_workflow(WORKFLOWS["build-crosswalk"], load_config(config_path))

    assert prepared.spec.module == "atlas_posthoc_plus.workflows.setup.crosswalk"
    assert str((project / "blocks.shp").resolve()) in prepared.argv
    expected_output = (
        project / "results/crosswalk/oh_block_to_precinct_crosswalk.csv"
    ).resolve()
    assert str(expected_output) in prepared.argv
