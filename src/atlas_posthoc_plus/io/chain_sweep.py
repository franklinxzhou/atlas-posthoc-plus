"""One-pass, graph-indexed chain materialization for analysis suites."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from atlas_posthoc.io.atlas_reader import canonicalize_unit_key, iter_atlas_records

from atlas_posthoc_plus.io.graph import read_graph_frame

SWEEP_DIR_ENV = "ATLAS_POSTHOC_PLUS_SWEEP_DIR"
SWEEP_LIMIT_ENV = "ATLAS_POSTHOC_PLUS_SWEEP_LIMIT"
SWEEP_SCHEMA = 1
COMPLIANCE_ATTR = "atlas_posthoc_plus_compliance"


@dataclass(frozen=True)
class DenseDistricting:
    """A map assignment already aligned to graph node ids."""

    node_ids: np.ndarray
    districts: np.ndarray
    compliance: dict[str, Any]

    def as_series(self, *, node_id_col: str = "id") -> pd.Series:
        assignment = pd.Series(self.districts, index=self.node_ids, dtype=int)
        assignment.index.name = node_id_col
        assignment.attrs[COMPLIANCE_ATTR] = self.compliance
        return assignment


@dataclass(frozen=True)
class SweepRecord:
    chain_id: str
    record_index: int
    line_number: int
    name: str | None
    weight: int | float | None
    data: dict[str, Any]
    districting: DenseDistricting


def cached_assignment_series(
    districting: Any,
    *,
    node_id_col: str = "id",
) -> pd.Series | None:
    if isinstance(districting, DenseDistricting):
        return districting.as_series(node_id_col=node_id_col)
    return None


def cached_compliance(value: Any) -> dict[str, Any] | None:
    if isinstance(value, DenseDistricting):
        return value.compliance
    if isinstance(value, pd.Series):
        result = value.attrs.get(COMPLIANCE_ATTR)
        return dict(result) if result is not None else None
    return None


def _cache_paths(cache_dir: Path, source: Path) -> tuple[Path, Path]:
    digest = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:20]
    stem = f"{source.name}.{digest}.sweep"
    return cache_dir / f"{stem}.pickle", cache_dir / f"{stem}.json"


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _expected_metadata(
    source: Path,
    graph_path: Path,
    *,
    node_id_col: str,
    key_col: str,
    county_col: str,
    max_districts_per_county: int,
    max_records: int | None,
) -> dict[str, Any]:
    return {
        "schema": SWEEP_SCHEMA,
        "source": _file_signature(source),
        "graph": _file_signature(graph_path),
        "node_id_col": node_id_col,
        "key_col": key_col,
        "county_col": county_col,
        "max_districts_per_county": max_districts_per_county,
        "max_records": max_records,
    }


def _cache_is_valid(metadata_path: Path, data_path: Path, expected: dict[str, Any]) -> bool:
    if not metadata_path.is_file() or not data_path.is_file():
        return False
    try:
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False

    for key in (
        "schema",
        "source",
        "graph",
        "node_id_col",
        "key_col",
        "county_col",
        "max_districts_per_county",
    ):
        if actual.get(key) != expected[key]:
            return False

    cached_limit = actual.get("max_records")
    requested_limit = expected["max_records"]
    return cached_limit is None or (
        requested_limit is not None and cached_limit >= requested_limit
    )


def _graph_index(
    graph_path: Path,
    *,
    node_id_col: str,
    key_col: str,
    county_col: str,
) -> tuple[np.ndarray, dict[tuple[str, ...], int], np.ndarray]:
    graph = read_graph_frame(graph_path, node_id_col=node_id_col)
    missing = [col for col in (node_id_col, key_col, county_col) if col not in graph.columns]
    if missing:
        raise ValueError(f"graph is missing sweep columns {missing}")

    node_ids = pd.to_numeric(graph[node_id_col], errors="raise").astype(int).to_numpy()
    key_to_position: dict[tuple[str, ...], int] = {}
    for position, raw_key in enumerate(graph[key_col]):
        key = canonicalize_unit_key(raw_key)
        if key in key_to_position:
            raise ValueError(f"duplicate graph key for {key_col!r}: {key!r}")
        key_to_position[key] = position
    counties = graph[county_col].astype(str).to_numpy()
    return node_ids, key_to_position, counties


def _dense_assignment(
    assignment: dict[tuple[str, ...], int],
    key_to_position: dict[tuple[str, ...], int],
    n_nodes: int,
) -> np.ndarray:
    districts = np.full(n_nodes, -1, dtype=np.int16)
    unmatched: list[tuple[str, ...]] = []
    for key, district in assignment.items():
        position = key_to_position.get(key)
        if position is None:
            unmatched.append(key)
            continue
        districts[position] = int(district)

    if unmatched:
        raise ValueError(
            f"{len(unmatched)} Atlas keys did not match the graph; examples={unmatched[:5]}"
        )
    missing = int(np.count_nonzero(districts < 0))
    if missing:
        raise ValueError(f"assignment is missing {missing} of {n_nodes} graph nodes")
    return districts


def _compliance(
    districts: np.ndarray,
    counties: np.ndarray,
    *,
    max_districts_per_county: int,
) -> dict[str, Any]:
    split_sizes = [
        int(np.unique(districts[counties == county]).size)
        for county in np.unique(counties)
    ]
    worst_split = max(split_sizes, default=0)
    double_cross_count = sum(
        split_size > max_districts_per_county for split_size in split_sizes
    )
    return {
        "compliant": double_cross_count == 0,
        "assigned_nodes": int(districts.size),
        "missing_nodes": 0,
        "assigned_fraction": 1.0,
        "double_cross_count": int(double_cross_count),
        "worst_split": int(worst_split),
    }


def materialize_chain_sweep(
    path: str | Path,
    graph_path: str | Path,
    cache_dir: str | Path,
    *,
    node_id_col: str = "id",
    key_col: str = "NAME",
    county_col: str = "COUNTY",
    max_districts_per_county: int = 2,
    max_records: int | None = None,
) -> bool:
    """Create dense assignments and compliance in one pass over a chain."""
    source = Path(path).resolve()
    graph_source = Path(graph_path).resolve()
    destination = Path(cache_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    data_path, metadata_path = _cache_paths(destination, source)
    expected = _expected_metadata(
        source,
        graph_source,
        node_id_col=node_id_col,
        key_col=key_col,
        county_col=county_col,
        max_districts_per_county=max_districts_per_county,
        max_records=max_records,
    )
    if _cache_is_valid(metadata_path, data_path, expected):
        return True

    node_ids, key_to_position, counties = _graph_index(
        graph_source,
        node_id_col=node_id_col,
        key_col=key_col,
        county_col=county_col,
    )
    data_tmp = data_path.with_suffix(data_path.suffix + ".tmp")
    metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    count = 0
    try:
        with data_tmp.open("wb") as stream:
            pickle.dump({"node_ids": node_ids}, stream, protocol=pickle.HIGHEST_PROTOCOL)
            for record in iter_atlas_records(source, skip_bad_lines=True):
                if max_records is not None and count >= max_records:
                    break
                districts = _dense_assignment(
                    record.assignment,
                    key_to_position,
                    len(node_ids),
                )
                payload = {
                    "chain_id": record.chain_id,
                    "record_index": record.map_index,
                    "line_number": record.line_number,
                    "name": record.name,
                    "weight": record.weight,
                    "data": record.data,
                    "districts": districts,
                    "compliance": _compliance(
                        districts,
                        counties,
                        max_districts_per_county=max_districts_per_county,
                    ),
                }
                pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
                count += 1

        metadata_tmp.write_text(
            json.dumps({**expected, "records": count}, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(data_tmp, data_path)
        os.replace(metadata_tmp, metadata_path)
    finally:
        data_tmp.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)
    return False


def configured_sweep() -> tuple[Path, int | None] | None:
    value = os.environ.get(SWEEP_DIR_ENV)
    if not value:
        return None
    raw_limit = os.environ.get(SWEEP_LIMIT_ENV)
    return Path(value), int(raw_limit) if raw_limit else None


def iter_sweep_records(path: str | Path) -> Iterator[SweepRecord]:
    configured = configured_sweep()
    if configured is None:
        raise RuntimeError("suite sweep cache is not configured")
    cache_dir, max_records = configured
    source = Path(path).resolve()
    data_path, _ = _cache_paths(cache_dir.resolve(), source)
    with data_path.open("rb") as stream:
        header = pickle.load(stream)
        node_ids = header["node_ids"]
        index = 0
        while max_records is None or index < max_records:
            try:
                payload = pickle.load(stream)
            except EOFError:
                return
            yield SweepRecord(
                chain_id=payload["chain_id"],
                record_index=payload["record_index"],
                line_number=payload["line_number"],
                name=payload["name"],
                weight=payload["weight"],
                data=payload["data"],
                districting=DenseDistricting(
                    node_ids=node_ids,
                    districts=payload["districts"],
                    compliance=payload["compliance"],
                ),
            )
            index += 1
