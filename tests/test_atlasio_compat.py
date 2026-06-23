from pathlib import Path

from atlas_posthoc_plus.io import atlas_compat


def test_legacy_atlasio_api_uses_atlas_posthoc_reader(tmp_path: Path) -> None:
    chain = tmp_path / "chain.jsonl"
    chain.write_text(
        '"Atlas"\n'
        '{"description": "fixture"}\n'
        '{"parameter": 1}\n'
        '{"name": "map1", "weight": 1, "data": {}, '
        '"districting": [{"[\\"node-a\\"]": 1}, {"[\\"node-b\\"]": 2}]}\n',
        encoding="utf-8",
    )

    atlas = atlas_compat.openAtlas(chain)
    first = atlas_compat.nextMap(atlas)

    assert first is not None
    assert first.name == "map1"
    assert first.districting['["node-a"]'] == 1
    assert atlas_compat.nextMap(atlas) is None
    atlas_compat.closeAtlas(atlas)
