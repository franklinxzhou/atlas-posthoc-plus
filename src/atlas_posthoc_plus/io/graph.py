"""Graph-table adapters built on :mod:`atlas_posthoc.io.graph_reader`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from atlas_posthoc.io.graph_reader import read_graph_json


def read_graph_frame(
    path: str | Path,
    *,
    node_id_col: str = "id",
) -> pd.DataFrame:
    graph = read_graph_json(path, node_id_col=node_id_col)
    return pd.DataFrame(graph.nodes)

