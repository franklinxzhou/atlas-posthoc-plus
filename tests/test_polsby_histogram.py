from pathlib import Path

import pandas as pd

from atlas_posthoc_plus.workflows.compactness.polsby_histogram import (
    load_ensemble_scores,
    load_official_scores,
)


def test_load_ensemble_scores_reads_atlas_posthoc_trace_json(tmp_path: Path) -> None:
    traces = tmp_path / "polsby_popper_traces.json"
    traces.write_text(
        """
[
  {
    "chain_id": "run37",
    "proposals": [0, 10, 20],
    "average_polsby_popper": [0.1, 0.2, null],
    "minimum_polsby_popper": [0.01, 0.02, 0.03],
    "maximum_polsby_popper": [0.5, 0.6, 0.7]
  }
]
""".strip(),
        encoding="utf-8",
    )

    scores = load_ensemble_scores(traces, stat="average", burnin=10)

    assert scores.to_dict("records") == [
        {"chain_id": "run37", "proposal": 10, "score": 0.2}
    ]


def test_load_official_scores_accepts_official_polsby_summary_csv(tmp_path: Path) -> None:
    summary = tmp_path / "official-summary.csv"
    pd.DataFrame(
        [
            {"plan": "Enacted ('25)", "average": 0.31, "minimum": 0.11, "maximum": 0.51},
            {"plan": "Remedial ('22)", "average": 0.32, "minimum": 0.12, "maximum": 0.52},
        ]
    ).to_csv(summary, index=False)

    scores = load_official_scores(summary, stat="minimum")

    assert scores.to_dict("records") == [
        {"label": "Enacted ('25)", "score": 0.11},
        {"label": "Remedial ('22)", "score": 0.12},
    ]

