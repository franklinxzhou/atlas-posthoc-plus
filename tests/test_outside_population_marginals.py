import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from atlas_posthoc_plus.workflows.municipal.outside_population_marginals import (
    build_collection_series,
    plot_outside_population_marginals,
)
from atlas_posthoc_plus.workflows.municipal.tradeoff_weights import load_all_traces


def _trace_file(path: Path, *, cuts: list[int], shares: list[float]) -> Path:
    path.write_text(
        json.dumps(
            [
                {
                    "chain_id": "run-a",
                    "n_cut_municipalities": cuts,
                    "ousted_population_share": shares,
                    "ousted_population": [share * 1_000_000 for share in shares],
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


def _load_two_collections(tmp_path: Path) -> list[dict]:
    traces_a = _trace_file(
        tmp_path / "a.json",
        cuts=[7, 8, 9, 9],
        shares=[0.021, 0.024, 0.025, 0.027],
    )
    traces_b = _trace_file(
        tmp_path / "b.json",
        cuts=[12, 13, 13, 14],
        shares=[0.031, 0.033, 0.034, 0.036],
    )
    args = Namespace(
        trace_collection=[
            f"mcd_w=20, runs 37-40={traces_a}",
            f"mcd_w=5, runs 53-56={traces_b}",
        ],
        traces_json_mcd20="",
        traces_json_mcd5="",
        traces_json_mcd1="",
    )
    return load_all_traces(args)


def test_outside_population_series_pool_values_by_collection(tmp_path: Path) -> None:
    traces = _load_two_collections(tmp_path)

    series = build_collection_series(traces, metric_key="ousted_population_share")

    assert [item.label for item in series] == [
        "mcd_w=20, runs 37-40",
        "mcd_w=5, runs 53-56",
    ]
    assert [item.values.tolist() for item in series] == [
        [0.021, 0.024, 0.025, 0.027],
        [0.031, 0.033, 0.034, 0.036],
    ]
    assert all(np.isfinite(item.values).all() for item in series)


def test_outside_population_plotter_writes_png(tmp_path: Path) -> None:
    traces = _load_two_collections(tmp_path)
    refs = {
        "Enacted ('25)": {
            "n_cut_municipalities": 34,
            "ousted_population_share": 0.0364,
            "ousted_population": 430_056.0,
        },
        "Overturned ('21)": {
            "n_cut_municipalities": 28,
            "ousted_population_share": 0.0296,
            "ousted_population": 348_747.0,
        },
        "Remedial ('22)": {
            "n_cut_municipalities": 26,
            "ousted_population_share": 0.0315,
            "ousted_population": 371_555.0,
        },
    }
    output = tmp_path / "outside.png"

    plot_outside_population_marginals(
        traces=traces,
        refs=refs,
        output=output,
        metric_key="ousted_population_share",
        metric_label="Ousted MCD population share",
        use_percent_axis=True,
        title="Outside population marginals",
    )

    assert output.exists()
    assert output.stat().st_size > 0
