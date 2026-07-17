from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atlas_posthoc_plus.workflows.municipal.city_partisan_context import (
    CityContext,
    add_population_adjusted_votes,
    build_city_contexts,
    normalized_context_metrics,
    plot_city_context_duo,
    read_block_context_rows,
    summarize_context_metrics,
)


def test_build_city_contexts_normalizes_population_within_city() -> None:
    population_rows = pd.DataFrame(
        {
            "unit_id": [1, 2, 3],
            "municipality": ["16000", "16000", "18000"],
            "population": [60, 40, 200],
        }
    )
    place_names = pd.DataFrame(
        {
            "municipality": ["16000", "18000"],
            "place_name": ["Cleveland", "Columbus"],
        }
    )

    [city] = build_city_contexts(population_rows, place_names, ["Cleveland"])

    assert city.population == 100
    assert city.unit_ids.tolist() == [1, 2]
    assert city.weights.sum() == pytest.approx(1)
    assert city.weights.tolist() == pytest.approx([0.6, 0.4])


def test_context_metrics_are_population_normalized_vote_share_changes() -> None:
    weights = np.array([0.6, 0.4])
    centroid = np.array([0.6, 0.6])
    ensemble_vectors = np.array([[0.7, 0.5], [0.5, 0.7]])

    distance, direction = normalized_context_metrics(
        ensemble_vectors, centroid, weights
    )
    official_distance, official_direction = normalized_context_metrics(
        np.array([0.8, 0.7]), centroid, weights
    )

    assert distance.tolist() == pytest.approx([0.1, 0.1])
    assert direction.tolist() == pytest.approx([0.02, -0.02])
    assert official_distance.item() == pytest.approx(0.16)
    assert official_direction.item() == pytest.approx(0.16)


def test_context_metrics_reject_unnormalized_population_weights() -> None:
    with pytest.raises(ValueError, match="must sum to one"):
        normalized_context_metrics(
            np.array([0.8, 0.7]),
            np.array([0.6, 0.6]),
            np.array([60, 40]),
        )


def test_split_graph_votes_keep_one_population_allocated_copy() -> None:
    nodes = pd.DataFrame(
        {
            "id": [1, 2],
            "ORIG_NAME": ["A_X", "A_X"],
            "POP20": [60, 40],
            "G20PREDEM": [70, 70],
            "G20PREREP": [30, 30],
        }
    )

    adjusted = add_population_adjusted_votes(nodes)

    assert adjusted["_CITY_CONTEXT_DEM"].tolist() == pytest.approx([42, 28])
    assert adjusted["_CITY_CONTEXT_REP"].tolist() == pytest.approx([18, 12])
    assert adjusted["_CITY_CONTEXT_DEM"].sum() == pytest.approx(70)


def test_block_rows_resolve_components_of_merged_graph_units(tmp_path: Path) -> None:
    graph_nodes = pd.DataFrame(
        {
            "id": [7],
            "ORIG_NAME": ["A_X,B_X"],
            "ENACTED_UPPER": [1],
            "ENACTED_LOWER": [2],
        }
    )
    block_path = tmp_path / "blocks.csv"
    pd.DataFrame(
        {
            "GEOID20": ["1", "2"],
            "NAME": ["A_X", "B_X"],
            "ENACTED_UPPER": [1, 1],
            "ENACTED_LOWER": [2, 2],
            "POP20": [40, 60],
            "PLACE": ['{"16000": 30}', '{"16000": 20}'],
        }
    ).to_csv(block_path, index=False)
    city = CityContext(
        municipality="16000",
        place_name="Cleveland",
        population=50,
        unit_ids=np.array([7]),
        weights=np.array([1.0]),
    )

    block_rows, city_rows = read_block_context_rows(
        block_path, graph_nodes, [city]
    )

    assert block_rows["unit_id"].tolist() == [7, 7]
    assert city_rows["city_population"].sum() == pytest.approx(50)


def test_summary_and_duo_plot_include_official_comparisons(tmp_path: Path) -> None:
    city = CityContext(
        municipality="16000",
        place_name="Cleveland",
        population=100,
        unit_ids=np.array([1, 2]),
        weights=np.array([0.6, 0.4]),
    )
    matrices = {
        "16000": np.array(
            [
                [0.55, 0.65],
                [0.60, 0.60],
                [0.65, 0.55],
                [0.70, 0.50],
            ]
        )
    }
    plans = pd.DataFrame(
        {
            "ensemble_index": range(4),
            "chain_id": ["run1"] * 4,
            "map_index": range(1, 5),
            "record_name": [None] * 4,
        }
    )
    official_vectors = {
        "Enacted ('25)": {"16000": np.array([0.75, 0.65])},
        "Overturned ('21)": {"16000": np.array([0.50, 0.45])},
        "Remedial ('22)": {"16000": np.array([0.65, 0.60])},
    }

    ensemble, official, centroids = summarize_context_metrics(
        [city], matrices, plans, official_vectors
    )
    output = plot_city_context_duo(
        city,
        ensemble,
        official,
        tmp_path / "cleveland-city-partisan-context.png",
        bins=5,
    )

    assert output.is_file()
    assert len(ensemble) == 4
    assert set(official["direction_interpretation"]) == {
        "packing",
        "cracking",
    }
    assert centroids["city_population_weight"].sum() == pytest.approx(1)
