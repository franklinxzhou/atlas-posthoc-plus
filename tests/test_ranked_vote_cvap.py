import pandas as pd

from atlas_posthoc_plus.workflows.demographics.ranked_vote_cvap import (
    atlasio_districting_to_series,
)


def test_atlasio_districting_to_series_accepts_raw_dict_keys() -> None:
    graph_nodes = pd.DataFrame(
        {
            "id": [10, 20],
            "NAME": ["node-a", "node-b"],
        }
    )
    districting = {
        '["node-a"]': 1,
        '["node-b"]': 2,
    }

    out = atlasio_districting_to_series(districting, graph_nodes)

    assert out.to_dict() == {10: 1, 20: 2}


def test_atlasio_districting_to_series_accepts_canonical_tuple_keys() -> None:
    graph_nodes = pd.DataFrame(
        {
            "id": [10, 20],
            "NAME": ["node-a", "node-b"],
        }
    )
    districting = {
        ("node-a",): 1,
        ("node-b",): 2,
    }

    out = atlasio_districting_to_series(districting, graph_nodes)

    assert out.to_dict() == {10: 1, 20: 2}

