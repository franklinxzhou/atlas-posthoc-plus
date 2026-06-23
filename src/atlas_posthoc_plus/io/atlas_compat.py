"""Compatibility adapter for the older ``openAtlas``/``nextMap`` API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from atlas_posthoc.io.atlas_reader import AtlasRecord, AtlasRecordError, iter_atlas_records


@dataclass(frozen=True)
class Map:
    name: str | None
    weight: int | float | None
    data: dict[str, Any]
    districting: dict[Any, Any]


@dataclass
class AtlasHandle:
    path: Path
    records: Iterator[AtlasRecord]


def openAtlas(path: str | Path) -> AtlasHandle:
    source = Path(path)
    return AtlasHandle(
        path=source,
        records=iter_atlas_records(source, skip_bad_lines=False),
    )


def nextMap(atlas: AtlasHandle) -> Map | None:
    try:
        record = next(atlas.records)
    except StopIteration:
        return None
    except AtlasRecordError as exc:
        raise json.JSONDecodeError(str(exc), "", 0) from exc

    if record.raw_assignment is not None:
        districting = record.raw_assignment
    else:
        districting = {
            json.dumps(list(key), separators=(",", ":")): district
            for key, district in record.assignment.items()
        }

    return Map(
        name=record.name,
        weight=record.weight,
        data=record.data,
        districting=districting,
    )


def closeAtlas(atlas: AtlasHandle) -> None:
    close = getattr(atlas.records, "close", None)
    if close is not None:
        close()

