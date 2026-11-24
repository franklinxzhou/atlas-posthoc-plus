import json
from math import pi
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from pypdf import PdfReader
from shapely.geometry import Polygon, box

from atlas_posthoc_plus.config import load_config
from atlas_posthoc_plus.suite import run_suite
from atlas_posthoc_plus.workflows.compactness.polsby_histogram import load_ensemble_scores
from atlas_posthoc_plus.workflows.compactness.reock import reock_score


def test_reock_uses_district_area_and_exact_circle_area() -> None:
    geometry = Polygon(
        [(0, 0), (3, 0), (3, 3), (0, 3)],
        holes=[[(1, 1), (1, 2), (2, 2), (2, 1)]],
    )

    assert reock_score(geometry) == pytest.approx(8 / (pi * 4.5))
    with pytest.raises(ValueError, match="positive area"):
        reock_score(geometry.boundary)


def test_compactness_histograms_run_with_shared_chain_sweep(tmp_path: Path, capfd) -> None:
    # Graph node IDs, geometry feature order, and Atlas keys are independent.
    graph = {
        "nodes": [
            {"id": 10, "NAME": "a", "COUNTY": "X"},
            {"id": 40, "NAME": "b", "COUNTY": "X"},
            {"id": 20, "NAME": "c", "COUNTY": "Y"},
            {"id": 30, "NAME": "d", "COUNTY": "Y"},
        ],
        "adjacency": [[], [], [], []],
    }
    (tmp_path / "graph.json").write_text(json.dumps(graph))
    chain = [
        {
            "name": f"step{step}",
            "districting": {'["d"]': 2, '["c"]': 2, '["b"]': 1, '["a"]': 1},
            "data": {"compactness_scores": [18, 32]},
        }
        for step in (0, 13, 42)
    ]
    (tmp_path / "chain.jsonl").write_text(
        '"Atlas"\n{}\n{}\n' + "\n".join(map(json.dumps, chain)) + "\n"
    )
    units = gpd.GeoDataFrame(
        {"NAME": ["c", "d", "a", "c", "b"], "GEOID20": ["3", "5", "1", "4", "2"]},
        geometry=[
            box(2, 0, 2.5, 1),
            box(4, 1, 5, 2),
            box(0, 0, 1, 1),
            box(2.5, 0, 3, 1),
            box(1, 0, 2, 1),
        ],
        crs="EPSG:5070",
    )
    units.to_file(tmp_path / "units.gpkg", index=False)
    assignment = "GEOID20,DISTRICT\n1,1\n2,1\n3,2\n4,2\n5,2\n"
    (tmp_path / "plan.csv").write_text(assignment)
    (tmp_path / "public").mkdir()
    (tmp_path / "public/submitted.csv").write_text(assignment)
    config = tmp_path / "config.toml"
    config.write_text(
        """
[chains]
files = ["chain.jsonl"]
ids = ["run-test"]

[analysis]
n_districts = 2
max_maps_per_chain = 2

[inputs]
graph_json = "graph.json"
precincts = "units.gpkg"
block_shapefile = "units.gpkg"
reock_public_plan_dir = "public"

[plans]
enacted = "plan.csv"
overturned = "plan.csv"
remedial = "plan.csv"

[workflow.reock-histogram]
stat = "minimum"
burnin = 14
thin_step = 2
"""
    )

    status = run_suite(
        "compactness",
        ["official-polsby", "polsby-trace", "polsby-histogram", "reock-histogram"],
        load_config(config),
    )

    assert status == 0, capfd.readouterr()
    outputs = tmp_path / "outputs"
    assert (outputs / "polsby/polsby-histogram.png").is_file()
    reock = outputs / "reock"
    assert {path.name for path in reock.iterdir()} == {
        "district-scores.csv",
        "ensemble-scores.csv",
        "official-summary.csv",
        "reock-histogram.pdf",
    }
    page, = PdfReader(reock / "reock-histogram.pdf").pages
    assert "Minimum district Reock" in page.extract_text()
    traces = outputs / "polsby/polsby-popper-traces.json"
    assert load_ensemble_scores(traces, "minimum", burnin=13).to_dict("records") == [
        {"chain_id": "run-test", "proposal": 13, "score": pytest.approx(pi / 8)}
    ]
    # The surviving map and official references have the same multipart district.
    scores = {1: 8 / (5 * pi), 2: 8 / (13 * pi)}
    assert pd.read_csv(reock / "ensemble-scores.csv").to_dict("records") == [
        {
            "chain_id": "run-test",
            "map_index": 2,
            "proposal": 42,
            "stat": "minimum",
            "score": pytest.approx(scores[2]),
        }
    ]
    official = pd.read_csv(reock / "official-summary.csv")
    assert official["label"].tolist() == ["Enacted ('25)", "Overturned ('21)", "Remedial ('22)"]
    assert official["stat"].tolist() == ["minimum"] * 3
    assert official["score"].tolist() == pytest.approx([scores[2]] * 3)

    districts = pd.read_csv(reock / "district-scores.csv", keep_default_na=False)
    assert districts[["source", "plan_id", "chain_id", "map_index", "proposal"]].values.tolist() == (
        [["ensemble", "", "run-test", "2", "42"]] * 2
        + [["official", label, "", "", ""] for label in official["label"] for _ in scores]
        + [["public", "submitted", "", "", ""]] * 2
    )
    assert districts["district"].tolist() == list(scores) * 5
    assert districts["score"].tolist() == pytest.approx(list(scores.values()) * 5)

    ensemble, references = capfd.readouterr().out.rsplit("Loaded 3 official plan scores.", 1)
    assert "mean     0.195883" in ensemble
    assert references.count("0.195883") == 3
