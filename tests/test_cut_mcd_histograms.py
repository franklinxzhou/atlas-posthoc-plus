import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from atlas_posthoc_plus.workflows.municipal.cut_mcd_histograms import (
    build_collection_series,
    integer_centered_bins,
    plot_cut_mcd_histograms,
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


def test_cut_histogram_bins_are_centered_on_integer_counts(tmp_path: Path) -> None:
    traces = _load_two_collections(tmp_path)
    series = build_collection_series(traces)

    bins = integer_centered_bins(series, reference_values=[26, 28, 34])

    assert bins[0] == 6.5
    assert bins[-1] == 34.5
    assert np.allclose(np.diff(bins), 1.0)


def test_cut_histogram_series_assigns_unique_colors_to_many_collections(
    tmp_path: Path,
) -> None:
    trace_collection = []
    for idx in range(24):
        traces = _trace_file(
            tmp_path / f"collection_{idx}.json",
            cuts=[idx + 1, idx + 2],
            shares=[0.01 + idx * 0.001, 0.011 + idx * 0.001],
        )
        trace_collection.append(f"Collection {idx}={traces}")

    args = Namespace(
        trace_collection=trace_collection,
        traces_json_mcd20="",
        traces_json_mcd5="",
        traces_json_mcd1="",
    )
    traces = load_all_traces(args)
    series = build_collection_series(traces)
    colors = [item.color for item in series]

    assert len(colors) == len(trace_collection)
    assert len(set(colors)) == len(colors)


def test_cut_histogram_plotter_writes_png(tmp_path: Path) -> None:
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
    output = tmp_path / "cuts.png"

    plot_cut_mcd_histograms(
        traces=traces,
        refs=refs,
        output=output,
        density=True,
        title="Cut MCD histograms",
    )

    assert output.exists()
    assert output.stat().st_size > 0
