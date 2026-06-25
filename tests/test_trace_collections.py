from argparse import Namespace
from pathlib import Path

from atlas_posthoc_plus.workflows.municipal.tradeoff_weights import (
    load_all_traces,
    parse_trace_collection_spec,
)


def _trace_file(path: Path) -> Path:
    path.write_text(
        """
[
  {
    "chain_id": "run-a",
    "n_cut_municipalities": [1, 2, 3],
    "ousted_population_share": [0.01, 0.02, 0.03]
  }
]
""".strip(),
        encoding="utf-8",
    )
    return path


def test_trace_collection_spec_allows_equals_in_label(tmp_path: Path) -> None:
    traces = _trace_file(tmp_path / "municipal_split_traces.json")

    spec = parse_trace_collection_spec(
        f"mcd_w=20, runs 37-40={traces}",
        style_index=0,
    )

    assert spec.label == "mcd_w=20, runs 37-40"
    assert spec.path == traces


def test_load_all_traces_accepts_arbitrary_labeled_collections(tmp_path: Path) -> None:
    traces_a = _trace_file(tmp_path / "a.json")
    traces_b = _trace_file(tmp_path / "b.json")
    args = Namespace(
        trace_collection=[
            f"Primary chain set={traces_a}",
            f"External comparison={traces_b}",
        ],
        traces_json_mcd20="",
        traces_json_mcd5="",
        traces_json_mcd1="",
    )

    loaded = load_all_traces(args)

    assert [item["collection_label"] for item in loaded] == [
        "Primary chain set",
        "External comparison",
    ]
    assert [item["collection_style_index"] for item in loaded] == [0, 1]

