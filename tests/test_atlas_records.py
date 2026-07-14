from pathlib import Path

import atlas_posthoc_plus.io.atlas_records as atlas_records


def _write_chain(path: Path, names: list[str]) -> None:
    records = [
        (
            f'{{"name": "{name}", "weight": 1, "data": {{}}, '
            f'"districting": [{{"[\\"node-a\\"]": 1}}]}}\n'
        )
        for name in names
    ]
    path.write_text(
        '"Atlas"\n{"description": "fixture"}\n{"parameter": 1}\n'
        + "".join(records),
        encoding="utf-8",
    )


def test_materialized_cache_avoids_reparsing_source(tmp_path: Path, monkeypatch) -> None:
    chain = tmp_path / "chain.jsonl"
    cache_dir = tmp_path / "cache"
    _write_chain(chain, ["map1", "map2"])

    assert atlas_records.materialize_chain_cache(chain, cache_dir) is False
    assert atlas_records.materialize_chain_cache(chain, cache_dir) is True

    monkeypatch.setenv(atlas_records.CACHE_DIR_ENV, str(cache_dir))
    monkeypatch.setattr(
        atlas_records,
        "_iter_atlas_records",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("source reparsed")),
    )

    records = list(atlas_records.iter_map_records(chain))
    assert [record["name"] for record in records] == ["map1", "map2"]


def test_cache_is_invalidated_when_source_changes(tmp_path: Path, monkeypatch) -> None:
    chain = tmp_path / "chain.jsonl"
    cache_dir = tmp_path / "cache"
    _write_chain(chain, ["map1"])
    atlas_records.materialize_chain_cache(chain, cache_dir)

    _write_chain(chain, ["map1", "map2"])
    calls = 0
    original = atlas_records._iter_atlas_records

    def counting_reader(*args, **kwargs):
        nonlocal calls
        calls += 1
        yield from original(*args, **kwargs)

    monkeypatch.setattr(atlas_records, "_iter_atlas_records", counting_reader)
    assert atlas_records.materialize_chain_cache(chain, cache_dir) is False
    assert calls == 1


def test_cache_respects_suite_record_limit(tmp_path: Path, monkeypatch) -> None:
    chain = tmp_path / "chain.jsonl"
    cache_dir = tmp_path / "cache"
    _write_chain(chain, ["map1", "map2", "map3"])
    atlas_records.materialize_chain_cache(chain, cache_dir)

    monkeypatch.setenv(atlas_records.CACHE_DIR_ENV, str(cache_dir))
    monkeypatch.setenv(atlas_records.CACHE_LIMIT_ENV, "2")
    records = list(atlas_records.iter_map_records(chain))

    assert [record["name"] for record in records] == ["map1", "map2"]
