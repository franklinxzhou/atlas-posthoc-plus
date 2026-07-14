import json
from pathlib import Path

import atlas_posthoc_plus.io.chain_sweep as chain_sweep
from atlas_posthoc_plus.io.atlas_records import iter_map_records


def _write_graph(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "directed": False,
                "multigraph": False,
                "graph": [],
                "nodes": [
                    {"id": 10, "NAME": "node-a", "COUNTY": "A"},
                    {"id": 20, "NAME": "node-b", "COUNTY": "A"},
                    {"id": 30, "NAME": "node-c", "COUNTY": "B"},
                ],
                "adjacency": [[], [], []],
            }
        ),
        encoding="utf-8",
    )


def _write_chain(path: Path) -> None:
    path.write_text(
        '"Atlas"\n'
        '{"description": "fixture"}\n'
        '{"parameter": 1}\n'
        '{"name": "map1", "data": {"compactness_scores": [1, 2]}, '
        '"districting": [{"[\\"node-a\\"]": 1}, {"[\\"node-b\\"]": 1}, '
        '{"[\\"node-c\\"]": 2}]}\n'
        '{"name": "map2", "data": {"compactness_scores": [2, 3]}, '
        '"districting": [{"[\\"node-a\\"]": 1}, {"[\\"node-b\\"]": 2}, '
        '{"[\\"node-c\\"]": 2}]}\n',
        encoding="utf-8",
    )


def test_one_sweep_supplies_multiple_consumers_without_reprocessing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    graph = tmp_path / "graph.json"
    chain = tmp_path / "chain.jsonl"
    cache_dir = tmp_path / "cache"
    _write_graph(graph)
    _write_chain(chain)

    reused = chain_sweep.materialize_chain_sweep(
        chain,
        graph,
        cache_dir,
        max_districts_per_county=1,
    )
    assert reused is False

    monkeypatch.setenv(chain_sweep.SWEEP_DIR_ENV, str(cache_dir))
    monkeypatch.setattr(
        chain_sweep,
        "iter_atlas_records",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("chain reparsed")),
    )
    monkeypatch.setattr(
        chain_sweep,
        "_dense_assignment",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("assignment rebuilt")),
    )

    first_consumer = list(iter_map_records(chain))
    second_consumer = list(iter_map_records(chain))

    assert len(first_consumer) == len(second_consumer) == 2
    first = first_consumer[0]["districting"]
    second = first_consumer[1]["districting"]
    assert first.as_series().to_dict() == {10: 1, 20: 1, 30: 2}
    assert first.compliance["compliant"] is True
    assert second.compliance["compliant"] is False


def test_sweep_cache_is_reused_until_chain_or_graph_changes(tmp_path: Path) -> None:
    graph = tmp_path / "graph.json"
    chain = tmp_path / "chain.jsonl"
    cache_dir = tmp_path / "cache"
    _write_graph(graph)
    _write_chain(chain)

    assert chain_sweep.materialize_chain_sweep(chain, graph, cache_dir) is False
    assert chain_sweep.materialize_chain_sweep(chain, graph, cache_dir) is True

    _write_chain(chain)
    chain.touch()
    assert chain_sweep.materialize_chain_sweep(chain, graph, cache_dir) is False
