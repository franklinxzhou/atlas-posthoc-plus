"""Canonical AtlasIO record access with optional suite-level caching."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

from atlas_posthoc.io.atlas_reader import (
    AtlasRecord,
    ParseSummary,
    iter_atlas_records as _iter_atlas_records,
)
from atlas_posthoc_plus.io.chain_sweep import configured_sweep, iter_sweep_records

CACHE_DIR_ENV = "ATLAS_POSTHOC_PLUS_CHAIN_CACHE_DIR"
CACHE_LIMIT_ENV = "ATLAS_POSTHOC_PLUS_CHAIN_CACHE_LIMIT"
_CACHE_SCHEMA = 1


def _cache_paths(cache_dir: Path, source: Path) -> tuple[Path, Path]:
    digest = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:20]
    stem = f"{source.name}.{digest}"
    return cache_dir / f"{stem}.pickle", cache_dir / f"{stem}.json"


def _source_metadata(source: Path, max_records: int | None) -> dict[str, Any]:
    stat = source.stat()
    return {
        "schema": _CACHE_SCHEMA,
        "source": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "max_records": max_records,
    }


def _cache_is_valid(
    metadata_path: Path,
    data_path: Path,
    expected: dict[str, Any],
) -> bool:
    if not metadata_path.is_file() or not data_path.is_file():
        return False
    try:
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False

    for key in ("schema", "source", "size", "mtime_ns"):
        if actual.get(key) != expected[key]:
            return False

    cached_limit = actual.get("max_records")
    requested_limit = expected["max_records"]
    return cached_limit is None or (
        requested_limit is not None and cached_limit >= requested_limit
    )


def materialize_chain_cache(
    path: str | Path,
    cache_dir: str | Path,
    *,
    max_records: int | None = None,
) -> bool:
    """Build a parsed-record cache, returning True when an existing cache was reused."""
    source = Path(path).resolve()
    destination = Path(cache_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    data_path, metadata_path = _cache_paths(destination, source)
    expected = _source_metadata(source, max_records)

    if _cache_is_valid(metadata_path, data_path, expected):
        return True

    data_tmp = data_path.with_suffix(data_path.suffix + ".tmp")
    metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    try:
        count = 0
        with data_tmp.open("wb") as stream:
            for record in _iter_atlas_records(source, skip_bad_lines=True):
                if max_records is not None and count >= max_records:
                    break
                pickle.dump(record, stream, protocol=pickle.HIGHEST_PROTOCOL)
                count += 1

        metadata = {**expected, "records": count}
        metadata_tmp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        os.replace(data_tmp, data_path)
        os.replace(metadata_tmp, metadata_path)
    finally:
        data_tmp.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)

    return False


def _configured_cache() -> tuple[Path, int | None] | None:
    value = os.environ.get(CACHE_DIR_ENV)
    if not value:
        return None
    raw_limit = os.environ.get(CACHE_LIMIT_ENV)
    return Path(value), int(raw_limit) if raw_limit else None


def _iter_cached_records(data_path: Path) -> Iterator[AtlasRecord]:
    with data_path.open("rb") as stream:
        while True:
            try:
                yield pickle.load(stream)
            except EOFError:
                return


def iter_records(
    path: str | Path,
    *,
    chain_id: str | None = None,
    thin_step: int = 1,
    skip_bad_lines: bool = True,
    summary: ParseSummary | None = None,
) -> Iterator[AtlasRecord]:
    """Yield Atlas records, reusing the parsed suite cache when configured."""
    cache = _configured_cache()
    if cache is None or not skip_bad_lines:
        yield from _iter_atlas_records(
            path,
            chain_id=chain_id,
            thin_step=thin_step,
            skip_bad_lines=skip_bad_lines,
            summary=summary,
        )
        return

    source = Path(path).resolve()
    cache_dir, max_records = cache
    materialize_chain_cache(source, cache_dir, max_records=max_records)
    data_path, metadata_path = _cache_paths(cache_dir.resolve(), source)
    expected = _source_metadata(source, max_records)
    if not _cache_is_valid(metadata_path, data_path, expected):
        raise RuntimeError(f"could not create a valid chain cache for {source}")

    yielded = 0
    for index, record in enumerate(_iter_cached_records(data_path)):
        if max_records is not None and index >= max_records:
            break
        if index % thin_step != 0:
            if summary is not None:
                summary.skipped_by_thinning += 1
            continue
        if chain_id is not None and record.chain_id != chain_id:
            record = replace(record, chain_id=chain_id)
        if summary is not None:
            summary.records_seen += 1
            summary.records_yielded += 1
        yielded += 1
        yield replace(record, map_index=yielded - 1)


def iter_map_records(
    path: str | Path,
    *,
    skip_bad_json: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield legacy-shaped records using the maintained Atlas parser."""
    if configured_sweep() is not None and skip_bad_json:
        for record in iter_sweep_records(path):
            yield {
                "name": record.name,
                "weight": record.weight,
                "data": record.data,
                "districting": record.districting,
                "record_index": record.record_index,
                "line_number": record.line_number,
            }
        return

    for record in iter_records(path, skip_bad_lines=skip_bad_json):
        yield {
            "name": record.name,
            "weight": record.weight,
            "data": record.data,
            "districting": (
                record.raw_assignment
                if record.raw_assignment is not None
                else record.assignment
            ),
        }
