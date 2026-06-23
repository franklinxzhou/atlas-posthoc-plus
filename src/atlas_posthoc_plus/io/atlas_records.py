"""Canonical AtlasIO record access for project workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from atlas_posthoc.io.atlas_reader import iter_atlas_records


def iter_map_records(
    path: str | Path,
    *,
    skip_bad_json: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield legacy-shaped records using the maintained atlas-posthoc parser."""
    for record in iter_atlas_records(path, skip_bad_lines=skip_bad_json):
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

