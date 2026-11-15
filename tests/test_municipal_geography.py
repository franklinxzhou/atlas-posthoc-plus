import json
from pathlib import Path

import pandas as pd
import pytest

from atlas_posthoc_plus.workflows.municipal.geography import MunicipalGeography
from atlas_posthoc_plus.workflows.municipal.district_rank import (
    compute_district_municipal_burden as compute_rank_burden,
)
from atlas_posthoc_plus.workflows.municipal.district_rank_compliant import (
    compute_district_municipal_burden as compute_compliant_rank_burden,
)


def test_cousub_graph_rows_retain_full_node_population() -> None:
    geography = MunicipalGeography(mode="COUSUB", scalar_column="MCD")
    nodes = pd.DataFrame(
        [
            {"id": 0, "MCD": "3900108350", "POP20": 125},
            {"id": 1, "MCD": "3900108350", "POP20": 75},
        ]
    )

    rows = geography.graph_population_rows(nodes)

    assert rows.to_dict("records") == [
        {"unit_id": 0, "municipality": "3900108350", "population": 125},
        {"unit_id": 1, "municipality": "3900108350", "population": 75},
    ]


def test_place_graph_rows_explode_mappings_and_exclude_empty_nodes(tmp_path: Path) -> None:
    place_layer = tmp_path / "places.geojson"
    place_layer.touch()
    geography = MunicipalGeography(
        mode="PLACE", place_shapefile=place_layer, output_namespace="place"
    )
    nodes = pd.DataFrame(
        [
            {"id": 0, "PLACE": {"Columbus": 80, "Bexley": 20}, "POP20": 125},
            {"id": 1, "PLACE": {}, "POP20": 75},
        ]
    )

    rows = geography.graph_population_rows(nodes)

    assert rows.to_dict("records") == [
        {"unit_id": 0, "municipality": "Columbus", "population": 80.0},
        {"unit_id": 0, "municipality": "Bexley", "population": 20.0},
    ]
    assert rows["population"].sum() == 100


def test_place_layer_directory_resolves_to_its_shapefile(tmp_path: Path) -> None:
    place_directory = tmp_path / "tl_2024_39_place"
    place_directory.mkdir()
    place_shapefile = place_directory / "tl_2024_39_place.shp"
    place_shapefile.touch()
    for suffix in (".dbf", ".prj", ".shx"):
        (place_directory / f"tl_2024_39_place{suffix}").touch()

    geography = MunicipalGeography(
        mode="PLACE",
        place_shapefile=place_directory,
        output_namespace="place",
    )

    assert geography.place_shapefile == place_shapefile


def test_ambiguous_place_layer_directory_requires_explicit_file(tmp_path: Path) -> None:
    place_directory = tmp_path / "places"
    place_directory.mkdir()
    (place_directory / "north.shp").touch()
    (place_directory / "south.shp").touch()

    with pytest.raises(ValueError, match="multiple .shp files"):
        MunicipalGeography(mode="PLACE", place_shapefile=place_directory)


def test_place_district_burden_explodes_before_grouping(tmp_path: Path) -> None:
    place_layer = tmp_path / "places.geojson"
    place_layer.touch()
    geography = MunicipalGeography(
        mode="PLACE", place_shapefile=place_layer, output_namespace="place"
    )
    nodes = pd.DataFrame(
        [
            {"id": 0, "PLACE": {"A": 60, "B": 10}, "POP20": 100},
            {"id": 1, "PLACE": {"A": 40}, "POP20": 40},
            {"id": 2, "PLACE": {"B": 30}, "POP20": 30},
            {"id": 3, "PLACE": {}, "POP20": 500},
        ]
    )
    assignment = pd.Series({0: 1, 1: 2, 2: 2, 3: 2})

    split_count, ousted = geography.district_burden(assignment, nodes)

    assert split_count.to_dict() == {1: 2.0, 2: 2.0}
    assert ousted.to_dict() == {1: 10.0, 2: 40.0}

    for implementation in (compute_rank_burden, compute_compliant_rank_burden):
        actual_split, actual_ousted = implementation(
            assignment,
            nodes,
            node_id_col="id",
            pop_col="POP20",
            municipality_cols=["PLACE"],
            geography=geography,
        )
        assert actual_split.to_dict() == split_count.to_dict()
        assert actual_ousted.to_dict() == ousted.to_dict()


def test_cousub_adapter_preserves_legacy_district_rank_result() -> None:
    nodes = pd.DataFrame(
        [
            {"id": 0, "MCD": "a", "POP20": 60},
            {"id": 1, "MCD": "a", "POP20": 40},
            {"id": 2, "MCD": "b", "POP20": 10},
        ]
    )
    assignment = pd.Series({0: 1, 1: 2, 2: 2})
    legacy = compute_rank_burden(
        assignment,
        nodes,
        node_id_col="id",
        pop_col="POP20",
        municipality_cols=["MCD"],
    )
    adapted = compute_rank_burden(
        assignment,
        nodes,
        node_id_col="id",
        pop_col="POP20",
        municipality_cols=["MCD"],
        geography=MunicipalGeography(mode="COUSUB"),
    )

    assert adapted[0].to_dict() == legacy[0].to_dict()
    assert adapted[1].to_dict() == legacy[1].to_dict()


def test_cousub_block_rows_keep_county_in_local_cousub_key(tmp_path: Path) -> None:
    blocks = tmp_path / "blocks.csv"
    pd.DataFrame(
        {
            "GEOID20": ["1", "2"],
            "COUNTYFP20": ["1", "3"],
            "COUSUBFP": ["42", "42"],
            "POP20": [10, 20],
        }
    ).to_csv(blocks, index=False)

    records = MunicipalGeography(mode="COUSUB").read_block_nodes(blocks)

    assert records == [
        {"id": "000000000000001", "MCD": "001_00042", "POP20": 10},
        {"id": "000000000000002", "MCD": "003_00042", "POP20": 20},
    ]


def test_place_summary_uses_only_represented_place_population(tmp_path: Path) -> None:
    place_layer = tmp_path / "places.geojson"
    place_layer.touch()
    geography = MunicipalGeography(
        mode="PLACE", place_shapefile=place_layer, output_namespace="place"
    )
    nodes = [
        {"id": "a", "PLACE": {"A": 60}, "POP20": 100},
        {"id": "b", "PLACE": {"A": 40}, "POP20": 40},
        {"id": "outside", "PLACE": {}, "POP20": 900},
    ]

    summary = geography.summarize_assignment(
        {"a": 1, "b": 2, "outside": 2}, nodes
    )

    assert summary.n_cut_municipalities == 1
    assert summary.ousted_population == 40
    assert summary.represented_population == 100
    assert summary.ousted_population_share == pytest.approx(0.4)

    with pytest.raises(ValueError, match="positive-population"):
        geography.summarize_assignment(
            {"a": 1, "outside": 2},
            nodes,
            require_complete_assignment=True,
        )


@pytest.fixture
def place_layers(tmp_path: Path) -> tuple[Path, Path]:
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    blocks_path = tmp_path / "blocks.geojson"
    places_path = tmp_path / "places.shp"
    gpd.GeoDataFrame(
        {
            "GEOID20": ["1", "2", "3", "4"],
            "POP20": [25, 75, 50, 100],
            "geometry": [box(x, 0, x + 1, 1) for x in (0, 5, 10, 15)],
        },
        crs="EPSG:4326",
    ).to_file(blocks_path)
    gpd.GeoDataFrame(
        {
            "NAME": ["First", "Second", "Third"],
            "GEOID": ["0100001", "0100002", "0100003"],
            "geometry": [box(x - 1, -1, x + 2, 2) for x in (0, 5, 10)],
        },
        crs="EPSG:4326",
    ).to_file(places_path)
    return blocks_path, places_path


def write_place_graph(
    tmp_path: Path, mapping: dict[str, int], *, mapping_column: str = "PLACE"
) -> Path:
    graph_path = tmp_path / "place-graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "directed": False,
                "multigraph": False,
                "graph": [],
                "nodes": [{"id": 0, mapping_column: mapping, "POP20": 250}],
                "adjacency": [[]],
            }
        ),
        encoding="utf-8",
    )
    return graph_path


@pytest.mark.parametrize(
    ("graph_mapping", "selected"),
    [
        pytest.param({" 0100001.0 ": 25}, [("0100001", 25)], id="subset-geoid"),
        pytest.param(
            {"Second": 75, "First": 25},
            [("First", 25), ("Second", 75)],
            id="multiple-names",
        ),
        pytest.param(
            {"0100001": 25, "0100002": 75, "0100003": 50},
            [("0100001", 25), ("0100002", 75), ("0100003", 50)],
            id="full-graph",
        ),
        pytest.param({}, [], id="empty-selection"),
        pytest.param(None, [("First", 25), ("Second", 75), ("Third", 50)], id="no-graph"),
    ],
)
def test_place_block_assignment(
    tmp_path: Path,
    place_layers: tuple[Path, Path],
    graph_mapping: dict[str, int] | None,
    selected: list[tuple[str, int]],
) -> None:
    blocks_path, places_path = place_layers
    graph_path = (
        None
        if graph_mapping is None
        else write_place_graph(tmp_path, graph_mapping, mapping_column="municipalities")
    )
    geography = MunicipalGeography(
        mode="PLACE",
        graph_json=graph_path,
        mapping_column="municipalities",
        place_shapefile=places_path,
    )

    assert geography.read_block_nodes(blocks_path) == [
        {"id": f"{block:015d}", "municipalities": {key: pop}, "POP20": pop}
        for block, (key, pop) in enumerate(selected, start=1)
    ]


def test_place_block_assignment_reports_missing_geometry(
    tmp_path: Path, place_layers: tuple[Path, Path]
) -> None:
    blocks_path, places_path = place_layers
    geography = MunicipalGeography(
        mode="PLACE",
        graph_json=write_place_graph(tmp_path, {"0100001": 25, "0100999": 0}),
        place_shapefile=places_path,
    )

    with pytest.raises(ValueError, match=r"missing from column 'GEOID'.*0100999"):
        geography.read_block_nodes(blocks_path)


def test_place_block_assignment_rejects_population_mismatch(
    tmp_path: Path, place_layers: tuple[Path, Path]
) -> None:
    blocks_path, places_path = place_layers
    geography = MunicipalGeography(
        mode="PLACE",
        graph_json=write_place_graph(tmp_path, {"0100001": 26}),
        place_shapefile=places_path,
    )

    with pytest.raises(
        ValueError, match=r"25 versus 26.*block-geometry checkpoint.*0100001.*25.0, 26.0"
    ):
        geography.read_block_nodes(blocks_path)


def test_cache_metadata_rejects_other_municipality_mode(tmp_path: Path) -> None:
    graph = tmp_path / "OHpct20.json"
    graph.write_text("{}", encoding="utf-8")
    cousub = MunicipalGeography(mode="COUSUB", graph_json=graph)
    metadata = cousub.provenance()
    place_layer = tmp_path / "places.geojson"
    place_layer.touch()
    place = MunicipalGeography(
        mode="PLACE",
        graph_json=graph,
        place_shapefile=place_layer,
        output_namespace="place",
    )

    with pytest.raises(ValueError, match="municipality_mode"):
        place.assert_compatible_metadata(metadata)

    with pytest.raises(ValueError, match="trace mode"):
        place.validate_trace_metadata([{"geography_mode": "COUSUB"}])
