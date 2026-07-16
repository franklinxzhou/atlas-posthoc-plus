from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atlas_posthoc_plus.workflows.municipal.geography import MunicipalGeography
from atlas_posthoc_plus.workflows.municipal.place_partisanship import (
    build_analysis_table,
    compute_place_ousted_shares,
    compute_place_vote_shares,
    plot_place_ousted_share_by_partisanship,
    plot_place_ousted_share_residual_by_partisanship,
    summarize_ensemble,
    summarize_official_plans,
)


def test_place_ousted_share_is_one_minus_largest_piece_share() -> None:
    rows = pd.DataFrame(
        [
            {"unit_id": 1, "municipality": "A", "population": 60},
            {"unit_id": 2, "municipality": "A", "population": 30},
            {"unit_id": 3, "municipality": "A", "population": 10},
            {"unit_id": 3, "municipality": "B", "population": 20},
        ]
    )
    assignment = pd.Series({1: 1, 2: 2, 3: 2})

    result = compute_place_ousted_shares(
        assignment, rows, require_complete_assignment=True
    ).set_index("municipality")

    assert result.loc["A", "place_population"] == 100
    assert result.loc["A", "largest_piece_population"] == 60
    assert result.loc["A", "ousted_population_share"] == pytest.approx(0.4)
    assert result.loc["A", "n_districts"] == 2
    assert result.loc["B", "ousted_population_share"] == 0


def test_place_votes_are_population_allocated_within_graph_nodes(tmp_path: Path) -> None:
    place_layer = tmp_path / "places.shp"
    place_layer.touch()
    geography = MunicipalGeography(mode="PLACE", place_shapefile=place_layer)
    nodes = pd.DataFrame(
        [
            {
                "id": 1,
                "POP20": 100,
                "PLACE": {"A": 75, "B": 25},
                "G20PREDEM": 60,
                "G20PREREP": 40,
            },
            {
                "id": 2,
                "POP20": 100,
                "PLACE": {"A": 50},
                "G20PREDEM": 20,
                "G20PREREP": 80,
            },
        ]
    )

    result = compute_place_vote_shares(nodes, geography).set_index("municipality")

    assert result.loc["A", "dem_votes"] == pytest.approx(55)
    assert result.loc["A", "rep_votes"] == pytest.approx(70)
    assert result.loc["A", "dem_two_party_share"] == pytest.approx(0.44)
    assert result.loc["B", "dem_two_party_share"] == pytest.approx(0.60)


def _synthetic_summary() -> pd.DataFrame:
    plans = ["Enacted ('25)", "Overturned ('21)", "Remedial ('22)"]
    records = []
    for plan_index, plan in enumerate(plans):
        for index in range(8):
            median = 0.04 + 0.02 * index
            official = np.clip(median + (plan_index - 1) * 0.015 - 0.01 * index, 0, 1)
            records.append(
                {
                    "official_plan": plan,
                    "municipality": f"place-{index}",
                    "place_name": f"Place {index}",
                    "place_population": 30_000 + 20_000 * index,
                    "dem_votes": 100 + index,
                    "rep_votes": 100 - index,
                    "dem_two_party_share": 0.25 + 0.07 * index,
                    "n_districts": 1 if official == 0 else 2,
                    "largest_piece_population": 25_000,
                    "ousted_population": official * (30_000 + 20_000 * index),
                    "ousted_population_share": official,
                    "ensemble_q05": max(0, median - 0.04),
                    "ensemble_q25": max(0, median - 0.02),
                    "ensemble_median": median,
                    "ensemble_q75": median + 0.02,
                    "ensemble_q95": median + 0.04,
                    "ensemble_plan_count": 200,
                    "official_ensemble_percentile": 0.5,
                    "official_minus_ensemble_median": official - median,
                }
            )
    return pd.DataFrame(records)


def test_ensemble_summary_and_analysis_residual() -> None:
    matrix = np.asarray([[0.0, 0.2], [0.1, 0.4], [0.2, 0.6]])
    ensemble = summarize_ensemble(["A", "B"], matrix)
    official = pd.DataFrame(
        {
            "official_plan": ["Plan", "Plan"],
            "municipality": ["A", "B"],
            "place_population": [100, 200],
            "n_districts": [2, 2],
            "largest_piece_population": [85, 100],
            "ousted_population": [15, 100],
            "ousted_population_share": [0.15, 0.5],
            "official_ensemble_percentile": [0.75, 0.5],
        }
    )
    votes = pd.DataFrame(
        {
            "municipality": ["A", "B"],
            "place_population": [100, 200],
            "dem_votes": [60, 40],
            "rep_votes": [40, 60],
            "dem_two_party_share": [0.6, 0.4],
        }
    )

    result = build_analysis_table(official, ensemble, votes)

    row_a = result.set_index("municipality").loc["A"]
    assert row_a["ensemble_median"] == pytest.approx(0.1)
    assert row_a["official_minus_ensemble_median"] == pytest.approx(0.05)


def test_official_empirical_percentile_uses_midrank_for_ties(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.xlsx"
    pd.DataFrame(
        {"GEOID20": ["1", "2"], "DistrictID": [1, 2]}
    ).to_excel(plan_path, index=False)
    block_rows = pd.DataFrame(
        [
            {"unit_id": "000000000000001", "municipality": "A", "population": 60},
            {"unit_id": "000000000000002", "municipality": "A", "population": 40},
        ]
    )
    ensemble = np.asarray([[0.0], [0.4], [0.5]])

    result = summarize_official_plans(
        {"Plan": plan_path}, block_rows, ["A"], ensemble
    )

    assert result.loc[0, "ousted_population_share"] == pytest.approx(0.4)
    assert result.loc[0, "official_ensemble_percentile"] == pytest.approx(0.5)


def test_place_partisanship_plotters_write_named_pngs(tmp_path: Path) -> None:
    summary = _synthetic_summary()
    interval = tmp_path / "place-ousted-share-by-partisanship.png"
    residual = tmp_path / "place-ousted-share-residual-by-partisanship.png"

    plot_place_ousted_share_by_partisanship(
        summary, interval, minimum_population=25_000, label_count=2
    )
    plot_place_ousted_share_residual_by_partisanship(
        summary, residual, minimum_population=25_000, label_count=2
    )

    assert interval.is_file() and interval.stat().st_size > 10_000
    assert residual.is_file() and residual.stat().st_size > 10_000
