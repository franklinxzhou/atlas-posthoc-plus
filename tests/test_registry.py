import importlib.util
from pathlib import Path

import pytest

from atlas_posthoc_plus.config import ConfigError, load_config
from atlas_posthoc_plus.registry import SUITES, WORKFLOWS, get_workflow
from atlas_posthoc_plus.runner import prepare_workflow


def test_all_workflow_modules_are_registered() -> None:
    modules = {spec.module for spec in WORKFLOWS.values()}
    assert len(WORKFLOWS) == 32
    assert len(modules) == 32
    assert all(importlib.util.find_spec(module) is not None for module in modules)


def test_full_suite_runs_compactness_histograms_after_the_polsby_trace() -> None:
    full = SUITES["full"]
    assert full.index("official-polsby") < full.index("polsby-trace")
    assert full.index("polsby-trace") < full.index("polsby-histogram")
    assert "reock-histogram" in full


def test_mct_diagnostics_is_opt_in() -> None:
    assert WORKFLOWS["mct-diagnostics"].module == (
        "atlas_posthoc_plus.workflows.municipal.mct_diagnostics"
    )
    assert all("mct-diagnostics" not in workflows for workflows in SUITES.values())
    assert get_workflow("municipal-tradeoff", municipality_mode="MCT").name == "mct-diagnostics"
    assert get_workflow("municipal-histograms", municipality_mode="MCT").name == "mct-diagnostics"
    for mode in ("COUSUB", "PLACE"):
        assert get_workflow("municipal-tradeoff", municipality_mode=mode).name == "municipal-tradeoff"
    with pytest.raises(ConfigError, match="unavailable in municipality.mode=MCT"):
        get_workflow("municipal-distributions", municipality_mode="MCT")


def test_municipal_marginal_workflows_are_in_inventory() -> None:
    assert WORKFLOWS["municipal-outside-population-marginals"].module == (
        "atlas_posthoc_plus.workflows.municipal.outside_population_marginals"
    )
    assert WORKFLOWS["municipal-cut-mcd-histograms"].module == (
        "atlas_posthoc_plus.workflows.municipal.cut_mcd_histograms"
    )
    assert "municipal-outside-population-marginals" in SUITES["municipal"]
    assert "municipal-cut-mcd-histograms" in SUITES["municipal"]


def test_place_partisanship_workflow_is_in_municipal_suite() -> None:
    assert WORKFLOWS["municipal-place-partisanship"].module == (
        "atlas_posthoc_plus.workflows.municipal.place_partisanship"
    )
    assert "municipal-place-partisanship" in SUITES["municipal"]


def test_city_partisan_context_workflow_is_in_municipal_suite() -> None:
    workflow = WORKFLOWS["municipal-city-partisan-context"]
    assert workflow.module == (
        "atlas_posthoc_plus.workflows.municipal.city_partisan_context"
    )
    assert "municipal-city-partisan-context" in SUITES["municipal"]
    output_mode = next(
        binding for binding in workflow.bindings if binding.flag == "--output-mode"
    )
    assert output_mode.default == "signed"


def test_place_partisanship_workflow_requires_place_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[project]
root = "."
[municipality]
mode = "COUSUB"
output_namespace = "cousub"
[outputs]
root = "results"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="requires municipality.mode=PLACE"):
        prepare_workflow(
            WORKFLOWS["municipal-place-partisanship"], load_config(config_path)
        )


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


def test_place_municipal_pipeline_defaults_share_namespace(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[project]
root = "project"

[chains]
files = ["chain.jsonl"]

[inputs]
block_metadata = "blocks.gpkg"
public_plan_dir = "public-plans"

[workflow.municipal-tradeoff-public]
pool_chains = true
ensemble_label = "Place-5 ensemble"

[workflow.municipal-city-partisan-context]
output_mode = "both"

[plans]
enacted = "enacted.xlsx"
overturned = "overturned.xlsx"
remedial = "remedial.xlsx"

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
    config = load_config(config_path)

    producer = prepare_workflow(WORKFLOWS["municipal-histograms-compliant"], config)
    reference_json = producer.argv[producer.argv.index("--reference-json") + 1]
    traces_json = producer.argv[producer.argv.index("--output-traces-json") + 1]

    for name in (
        "municipal-outside-population-marginals",
        "municipal-cut-mcd-histograms",
        "municipal-tradeoff-public",
    ):
        consumer = prepare_workflow(WORKFLOWS[name], config)
        assert consumer.argv[consumer.argv.index("--reference-json") + 1] == reference_json

    public = prepare_workflow(WORKFLOWS["municipal-tradeoff-public"], config)
    assert public.argv[public.argv.index("--traces-json") + 1] == traces_json
    assert "--pool-chains" in public.argv
    assert public.argv[public.argv.index("--ensemble-label") + 1] == "Place-5 ensemble"

    partisanship = prepare_workflow(
        WORKFLOWS["municipal-place-partisanship"], config
    )
    assert partisanship.argv[partisanship.argv.index("--output-dir") + 1] == str(
        (project / "results/place/municipal-place-partisanship").resolve()
    )

    context = prepare_workflow(WORKFLOWS["municipal-city-partisan-context"], config)
    assert context.argv[context.argv.index("--output-dir") + 1] == str(
        (project / "results/place/municipal-city-partisan-context").resolve()
    )
    assert context.argv[context.argv.index("--block-metadata") + 1] == str(
        (project / "blocks.gpkg").resolve()
    )
    assert context.argv[context.argv.index("--output-mode") + 1] == "both"

    namespace_root = str((project / "results/place").resolve())
    for name in (
        "municipal-outside-population-marginals",
        "municipal-cut-mcd-histograms",
        "municipal-tradeoff-public",
    ):
        consumer = prepare_workflow(WORKFLOWS[name], config)
        output = consumer.argv[consumer.argv.index("--output") + 1]
        assert output.startswith(namespace_root)


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
