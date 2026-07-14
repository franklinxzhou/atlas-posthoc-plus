import importlib.util
from pathlib import Path

import pytest

from atlas_posthoc_plus.config import ConfigError, load_config
from atlas_posthoc_plus.registry import SUITES, WORKFLOWS
from atlas_posthoc_plus.runner import prepare_workflow


def test_all_workflow_modules_are_registered() -> None:
    modules = {spec.module for spec in WORKFLOWS.values()}
    assert len(WORKFLOWS) == 23
    assert len(modules) == 23
    assert all(importlib.util.find_spec(module) is not None for module in modules)


def test_municipal_marginal_workflows_are_in_inventory() -> None:
    assert WORKFLOWS["municipal-outside-population-marginals"].module == (
        "atlas_posthoc_plus.workflows.municipal.outside_population_marginals"
    )
    assert WORKFLOWS["municipal-cut-mcd-histograms"].module == (
        "atlas_posthoc_plus.workflows.municipal.cut_mcd_histograms"
    )
    assert "municipal-outside-population-marginals" in SUITES["municipal"]
    assert "municipal-cut-mcd-histograms" in SUITES["municipal"]


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


def test_prepare_workflow_builds_trace_collection_args(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "mcd20.json").touch()
    (project / "mcd5.json").touch()

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[project]
root = "project"

[inputs.municipal_trace_collections]
"mcd_w=20" = "mcd20.json"
"mcd_w=5" = "mcd5.json"

[municipality]
mode = "COUSUB"
output_namespace = "cousub"

[outputs]
root = "results"
""",
        encoding="utf-8",
    )

    prepared = prepare_workflow(
        WORKFLOWS["municipal-tradeoff-weights"],
        load_config(config_path),
    )

    trace_args = [
        prepared.argv[i + 1]
        for i, arg in enumerate(prepared.argv)
        if arg == "--trace-collection"
    ]

    assert trace_args == [
        f"mcd_w=20={(project / 'mcd20.json').resolve()}",
        f"mcd_w=5={(project / 'mcd5.json').resolve()}",
    ]
    assert "--municipality-mode" in prepared.argv
    assert "COUSUB" in prepared.argv
    assert str((project / "results/cousub/municipal/official-plan-summaries.json").resolve()) in prepared.argv


def test_place_registry_bindings_use_place_graph_and_namespace(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    for name in ("place.json", "blocks.shp", "places.shp", "crosswalk.csv", "plan.xlsx"):
        (project / name).touch()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[project]
root = "project"

[chains]
files = ["place.json"]

[inputs]
crosswalk_csv = "crosswalk.csv"
block_metadata = "blocks.shp"

[plans]
enacted = "plan.xlsx"
overturned = "plan.xlsx"
remedial = "plan.xlsx"

[municipality]
mode = "PLACE"
graph_json = "place.json"
mapping_column = "PLACE"
place_shapefile = "places.shp"
output_namespace = "place"

[outputs]
root = "results"
""",
        encoding="utf-8",
    )

    prepared = prepare_workflow(
        WORKFLOWS["district-rank-municipal"], load_config(config_path)
    )

    assert prepared.argv[prepared.argv.index("--municipality-mode") + 1] == "PLACE"
    assert prepared.argv[prepared.argv.index("--graph-json") + 1] == str(
        (project / "place.json").resolve()
    )
    assert prepared.argv[prepared.argv.index("--place-shapefile") + 1] == str(
        (project / "places.shp").resolve()
    )
    assert prepared.argv[prepared.argv.index("--output-dir") + 1] == str(
        (project / "results/place/district-rank-municipal").resolve()
    )


def test_place_registry_requires_place_layer(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[project]
root = "project"
[municipality]
mode = "PLACE"
output_namespace = "place"
[outputs]
root = "results"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="place_shapefile"):
        prepare_workflow(
            WORKFLOWS["municipal-tradeoff-weights"], load_config(config_path)
        )


def test_prepare_workflow_builds_polsby_histogram_args(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "polsby_popper_traces.json").touch()

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[project]
root = "project"

[inputs]
polsby_traces_json = "polsby_popper_traces.json"

[outputs]
root = "results"

[workflow.polsby-histogram]
stat = "minimum"
bins = 25
""",
        encoding="utf-8",
    )

    prepared = prepare_workflow(
        WORKFLOWS["polsby-histogram"],
        load_config(config_path),
    )

    assert prepared.spec.module == "atlas_posthoc_plus.workflows.compactness.polsby_histogram"
    assert "--ensemble-pp-json" in prepared.argv
    assert str((project / "polsby_popper_traces.json").resolve()) in prepared.argv
    assert "--stat" in prepared.argv
    assert "minimum" in prepared.argv
