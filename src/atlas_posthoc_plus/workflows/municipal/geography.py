"""Shared municipal-geography adapter for COUSUB and PLACE workflows.

The adapter deliberately exposes one configuration object to every municipal
workflow.  COUSUB nodes carry one scalar municipality and their full POP20;
PLACE nodes carry a mapping of Place name to the population represented by
that Place.  Reference and public plans are normalized to the same semantics
at Census-block resolution.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import pandas as pd

from atlas_posthoc.observables.municipal_splits import (
    MunicipalSplitPlanSummary,
    compute_municipal_split_statistics_for_assignment,
)

MunicipalityMode = Literal["COUSUB", "PLACE"]

_MUNICIPALITY_NAMES = {"COUSUB": "county subdivision", "PLACE": "place"}


def municipality_name_for_traces(traces: Iterable[Mapping[str, Any] | Any]) -> str:
    """Name the trace geography, retaining COUSUB for legacy trace files."""
    modes = {
        str(
            trace.get("geography_mode", "COUSUB")
            if isinstance(trace, Mapping)
            else getattr(trace, "geography_mode", "COUSUB")
        ).upper()
        for trace in traces
    }
    if modes - _MUNICIPALITY_NAMES.keys():
        raise ValueError(f"Unknown municipal trace geography modes: {sorted(modes)}")
    if len(modes) > 1:
        raise ValueError("Cannot label COUSUB and PLACE traces on the same municipal axis")
    return _MUNICIPALITY_NAMES[next(iter(modes), "COUSUB")]


_DATASET_EXTENSIONS = (
    ".gpkg",
    ".shp",
    ".geojson",
    ".json",
    ".zip",
    ".parquet",
    ".pq",
    ".csv",
)


def _resolve_dataset_path(path: str | Path) -> Path:
    """Resolve a dataset file, accepting a directory containing one dataset.

    TIGER/Line layers are commonly configured by their extracted directory
    rather than by the ``.shp`` member inside it.  Search extensions in
    spatial-data priority order and reject genuinely ambiguous directories.
    """
    path = Path(path)
    if not path.is_dir():
        return path

    for extension in _DATASET_EXTENSIONS:
        candidates = sorted(
            candidate
            for candidate in path.iterdir()
            if candidate.is_file() and candidate.suffix.lower() == extension
        )
        if not candidates:
            continue
        if len(candidates) == 1:
            return candidates[0]

        same_stem = [candidate for candidate in candidates if candidate.stem == path.name]
        if len(same_stem) == 1:
            return same_stem[0]
        examples = ", ".join(candidate.name for candidate in candidates[:5])
        raise ValueError(
            f"Dataset directory {path} contains multiple {extension} files "
            f"({examples}); configure the intended file explicitly"
        )

    raise FileNotFoundError(
        f"Dataset directory {path} contains none of: {', '.join(_DATASET_EXTENSIONS)}"
    )


def _normalize_token(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _normalize_block_id(value: Any) -> str:
    text = _normalize_token(value)
    return text.zfill(15) if text else ""


def _canonical(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _resolve_column(
    columns: Iterable[str], candidates: Iterable[str | None], *, label: str
) -> str:
    columns = list(columns)
    normalized = {_canonical(column): column for column in columns}
    for candidate in candidates:
        if candidate is not None and _canonical(candidate) in normalized:
            return normalized[_canonical(candidate)]
    raise ValueError(
        f"Could not resolve {label} column from {list(candidates)}. "
        f"Available columns: {columns}"
    )


def _read_spatial_or_table(path: Path) -> pd.DataFrame:
    path = _resolve_dataset_path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".zip":
        import geopandas as gpd

        return gpd.read_file(f"zip://{path.resolve()}")
    if suffix in {".gpkg", ".shp", ".geojson", ".json"}:
        import geopandas as gpd

        return gpd.read_file(path)
    return pd.read_csv(path, dtype=str)


@dataclass(frozen=True)
class MunicipalGeography:
    """Explicit bindings for one municipal-analysis arm."""

    mode: MunicipalityMode = "COUSUB"
    graph_json: Path | None = None
    mapping_column: str = "PLACE"
    scalar_column: str = "MCD"
    place_shapefile: Path | None = None
    output_namespace: str = "cousub"
    place_key_column: str | None = None

    def __post_init__(self) -> None:
        mode = str(self.mode).upper()
        if mode not in {"COUSUB", "PLACE"}:
            raise ValueError(f"municipality mode must be COUSUB or PLACE, got {self.mode!r}")
        object.__setattr__(self, "mode", mode)
        if self.graph_json is not None:
            object.__setattr__(self, "graph_json", Path(self.graph_json))
        if self.place_shapefile is not None:
            object.__setattr__(
                self,
                "place_shapefile",
                _resolve_dataset_path(self.place_shapefile),
            )
        if mode == "PLACE" and self.place_shapefile is None:
            raise ValueError("PLACE mode requires a place_shapefile")
        if not self.output_namespace or Path(self.output_namespace).name != self.output_namespace:
            raise ValueError("output_namespace must be one non-empty path component")

    @property
    def municipality_columns(self) -> tuple[str, ...]:
        return (self.scalar_column,) if self.mode == "COUSUB" else (self.mapping_column,)

    @property
    def display_name(self) -> str:
        return _MUNICIPALITY_NAMES[self.mode]

    def provenance(self) -> dict[str, Any]:
        graph = None if self.graph_json is None else self.graph_json.resolve()
        fingerprint = None
        if graph is not None and graph.is_file():
            digest = hashlib.sha256()
            with graph.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            fingerprint = digest.hexdigest()
        return {
            "municipality_mode": self.mode,
            "graph_json": None if graph is None else str(graph),
            "graph_sha256": fingerprint,
            "mapping_column": self.mapping_column if self.mode == "PLACE" else None,
            "scalar_column": self.scalar_column if self.mode == "COUSUB" else None,
            "place_shapefile": (
                None if self.place_shapefile is None else str(self.place_shapefile.resolve())
            ),
            "place_key_column": self.place_key_column or "auto",
            "output_namespace": self.output_namespace,
        }

    def assert_compatible_metadata(self, metadata: Mapping[str, Any]) -> None:
        expected = self.provenance()
        for key in ("municipality_mode", "graph_json", "graph_sha256", "output_namespace"):
            if metadata.get(key) != expected.get(key):
                raise ValueError(
                    f"municipal cache is incompatible for {key}: "
                    f"expected {expected.get(key)!r}, found {metadata.get(key)!r}"
                )

    def validate_trace_metadata(self, traces: Iterable[Mapping[str, Any] | Any]) -> None:
        """Reject trace collections produced for the other geography arm."""
        for trace in traces:
            if isinstance(trace, Mapping):
                mode = trace.get("geography_mode", "COUSUB")
                graph_path = trace.get("graph_path")
            else:
                mode = getattr(trace, "geography_mode", "COUSUB")
                graph_path = getattr(trace, "graph_path", None)
            if str(mode).upper() != self.mode:
                raise ValueError(
                    f"municipal trace mode {mode!r} is incompatible with {self.mode}"
                )
            if self.graph_json is not None and graph_path is not None:
                if Path(graph_path).name != self.graph_json.name:
                    raise ValueError(
                        f"municipal trace graph {graph_path!r} is incompatible with "
                        f"{self.graph_json}"
                    )

    def write_metadata(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.provenance(), indent=2) + "\n", encoding="utf-8")

    def validate_graph_nodes(
        self,
        graph_nodes: pd.DataFrame | list[Mapping[str, Any]],
        *,
        node_id_col: str = "id",
        pop_col: str = "POP20",
    ) -> None:
        columns = (
            set(graph_nodes.columns)
            if isinstance(graph_nodes, pd.DataFrame)
            else set().union(*(node.keys() for node in graph_nodes))
        )
        required = {node_id_col, pop_col, *self.municipality_columns}
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"Graph nodes are missing municipal columns {missing}")

        records = (
            graph_nodes.to_dict("records")
            if isinstance(graph_nodes, pd.DataFrame)
            else graph_nodes
        )
        if self.mode == "PLACE":
            for node in records:
                mapping = node.get(self.mapping_column)
                if not isinstance(mapping, Mapping):
                    raise ValueError(
                        f"Graph node {node.get(node_id_col)!r} has non-mapping "
                        f"{self.mapping_column}={mapping!r}"
                    )
                place_pop = 0.0
                for place, population in mapping.items():
                    if not str(place).strip():
                        raise ValueError("PLACE mappings may not contain empty names")
                    value = float(population)
                    if value < 0:
                        raise ValueError("PLACE populations must be nonnegative")
                    place_pop += value
                if place_pop > float(node[pop_col]) + 1e-9:
                    raise ValueError(
                        f"Graph node {node.get(node_id_col)!r} has PLACE population "
                        f"{place_pop} above {pop_col}={node[pop_col]}"
                    )

    def graph_population_rows(
        self,
        graph_nodes: pd.DataFrame,
        *,
        node_id_col: str = "id",
        pop_col: str = "POP20",
    ) -> pd.DataFrame:
        """Return ``unit_id, municipality, population`` rows for graph nodes."""
        self.validate_graph_nodes(graph_nodes, node_id_col=node_id_col, pop_col=pop_col)
        if self.mode == "COUSUB":
            out = graph_nodes[[node_id_col, self.scalar_column, pop_col]].copy()
            out = out.rename(
                columns={
                    node_id_col: "unit_id",
                    self.scalar_column: "municipality",
                    pop_col: "population",
                }
            )
            out["municipality"] = out["municipality"].map(_normalize_token)
            out["population"] = pd.to_numeric(out["population"], errors="raise")
            return out[out["municipality"] != ""].reset_index(drop=True)

        rows: list[dict[str, Any]] = []
        for node in graph_nodes[[node_id_col, self.mapping_column]].to_dict("records"):
            for place, population in node[self.mapping_column].items():
                value = float(population)
                if value > 0:
                    rows.append(
                        {
                            "unit_id": node[node_id_col],
                            "municipality": str(place),
                            "population": value,
                        }
                    )
        return pd.DataFrame(rows, columns=["unit_id", "municipality", "population"])

    def summarize_assignment(
        self,
        assignment: Mapping[Any, Any] | Iterable[Any],
        nodes: list[Mapping[str, Any]],
        *,
        node_id_col: str = "id",
        pop_col: str = "POP20",
        require_complete_assignment: bool = False,
    ) -> MunicipalSplitPlanSummary:
        atlas_require_complete = require_complete_assignment
        if require_complete_assignment and isinstance(assignment, Mapping):
            expected: set[Any] = set()
            for node in nodes:
                if self.mode == "COUSUB":
                    represented = float(node.get(pop_col, 0.0)) > 0
                else:
                    mapping = node.get(self.mapping_column, {})
                    represented = isinstance(mapping, Mapping) and any(
                        float(population) > 0 for population in mapping.values()
                    )
                if represented:
                    expected.add(node[node_id_col])
            missing = expected - set(assignment)
            if missing:
                raise ValueError(
                    f"{len(missing)} positive-population municipal units are missing "
                    f"district assignments; examples: {sorted(missing, key=str)[:5]}"
                )
            # The positive-population completeness contract is stronger for the
            # denominator and deliberately ignores omitted zero-population blocks.
            atlas_require_complete = False
        return compute_municipal_split_statistics_for_assignment(
            assignment,
            nodes,
            municipality_cols=(self.scalar_column,),
            geography_mode=self.mode,
            place_col=self.mapping_column,
            graph_path=self.graph_json,
            node_id_col=node_id_col,
            pop_col=pop_col,
            skip_missing_municipality=True,
            require_complete_assignment=atlas_require_complete,
        )

    def district_burden(
        self,
        assignment: pd.Series,
        graph_nodes: pd.DataFrame,
        *,
        node_id_col: str = "id",
        pop_col: str = "POP20",
        require_complete_assignment: bool = False,
    ) -> tuple[pd.Series, pd.Series]:
        """Compute district-level split count and ousted population.

        PLACE mode explodes the mapping before grouping, so a multi-Place node
        contributes only the mapped population of each Place.
        """
        rows = self.graph_population_rows(
            graph_nodes, node_id_col=node_id_col, pop_col=pop_col
        )
        if require_complete_assignment:
            missing = set(rows["unit_id"]) - set(assignment.index)
            if missing:
                raise ValueError(
                    f"{len(missing)} represented municipal units are missing district "
                    f"assignments; examples: {sorted(missing, key=str)[:5]}"
                )
        rows = rows.merge(
            assignment.rename("DISTRICT").rename_axis("unit_id").reset_index(),
            on="unit_id",
            how="inner",
        )
        pieces = (
            rows.groupby(["municipality", "DISTRICT"], as_index=False)["population"]
            .sum()
            .rename(columns={"population": "piece_pop"})
        )
        if pieces.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        anchors = (
            pieces.sort_values(
                ["municipality", "piece_pop", "DISTRICT"],
                ascending=[True, False, True],
            )
            .drop_duplicates("municipality")[["municipality", "DISTRICT"]]
            .rename(columns={"DISTRICT": "ANCHOR_DISTRICT"})
        )
        counts = (
            pieces[pieces["piece_pop"] > 0]
            .groupby("municipality")["DISTRICT"]
            .nunique()
            .rename("n_districts")
            .reset_index()
        )
        split = pieces.merge(anchors, on="municipality").merge(counts, on="municipality")
        split = split[split["n_districts"] > 1].copy()
        split_count = split.groupby("DISTRICT").size().astype(float)
        split["ousted_piece_pop"] = split["piece_pop"].where(
            split["DISTRICT"] != split["ANCHOR_DISTRICT"], 0.0
        )
        ousted = split.groupby("DISTRICT")["ousted_piece_pop"].sum().astype(float)
        return split_count, ousted

    def read_block_nodes(
        self,
        block_metadata: str | Path,
        *,
        block_col: str | None = None,
        county_col: str | None = None,
        pop_col: str | None = None,
        block_mcd_col: str | None = None,
        mcd_shapefile: str | Path | None = None,
        mcd_shape_county_col: str | None = None,
        mcd_shape_mcd_col: str | None = None,
    ) -> list[dict[str, Any]]:
        """Normalize Census blocks for reference, remedial, and public plans."""
        path = Path(block_metadata)
        if not path.exists():
            raise FileNotFoundError(f"Block metadata file not found: {path}")
        blocks = _read_spatial_or_table(path)
        if blocks.empty:
            raise ValueError(f"Block metadata file is empty: {path}")

        block_col = _resolve_column(
            blocks.columns,
            [block_col, "GEOID20", "BLOCK", "GEOID", "GEOID10", "BLOCKID"],
            label="block id",
        )
        pop_col = _resolve_column(
            blocks.columns,
            [pop_col, "POP20", "POPULATION", "POP", "P0010001", "P1_001N"],
            label="population",
        )

        if self.mode == "PLACE":
            return self._read_place_block_nodes(blocks, block_col=block_col, pop_col=pop_col)

        county_col = _resolve_column(
            blocks.columns,
            [county_col, "COUNTY", "COUNTYFP20", "COUNTYFP", "COUNTYFP10"],
            label="county",
        )
        if block_mcd_col is not None:
            if block_mcd_col not in blocks.columns:
                raise ValueError(f"block MCD column {block_mcd_col!r} was not found")
            scalar_col = block_mcd_col
        else:
            try:
                scalar_col = _resolve_column(
                    blocks.columns,
                    ["MCD", "COUSUB", "COUSUBFP20", "COUSUBFP", "COUSUBFP10"],
                    label="county subdivision",
                )
            except ValueError:
                if mcd_shapefile is None:
                    raise ValueError(
                        "COUSUB mode requires a block MCD column or mcd_shapefile"
                    )
                blocks = self._assign_blocks_to_cousubs(
                    blocks,
                    Path(mcd_shapefile),
                    county_col=county_col,
                    shape_county_col=mcd_shape_county_col,
                    shape_mcd_col=mcd_shape_mcd_col,
                )
                scalar_col = "__municipality__"

        out = blocks[[block_col, scalar_col, pop_col]].copy()
        out[block_col] = out[block_col].map(_normalize_block_id)
        county_key = blocks.loc[out.index, county_col].map(_normalize_token).str.zfill(3)
        scalar_key = out[scalar_col].map(_normalize_token)
        # TIGER COUSUBFP is county-local. Preserve already-composite/full GEOIDs,
        # but combine the common five-digit field with its county as before.
        out[scalar_col] = [
            f"{county}_{value.zfill(5)}"
            if value.isdigit() and len(value) <= 5
            else (
                value
                if (value.isdigit() and len(value) >= 8) or "_" in value
                else f"{county}_{value}"
            )
            for county, value in zip(county_key, scalar_key)
        ]
        out[pop_col] = pd.to_numeric(out[pop_col], errors="coerce").fillna(0.0)
        out = out[(out[block_col] != "") & (out[scalar_col] != "")]
        out = out.rename(
            columns={block_col: "id", scalar_col: self.scalar_column, pop_col: "POP20"}
        )
        out = out.groupby(["id", self.scalar_column], as_index=False)["POP20"].sum()
        return out.to_dict("records")

    def _read_place_block_nodes(
        self, blocks: pd.DataFrame, *, block_col: str, pop_col: str
    ) -> list[dict[str, Any]]:
        import geopandas as gpd

        if not isinstance(blocks, gpd.GeoDataFrame) or blocks.crs is None:
            raise ValueError("PLACE mode requires block geometry with a CRS")
        expected = self._graph_place_totals()
        if expected == {}:
            # All graph nodes are outside the selected municipality universe.
            return []

        assert self.place_shapefile is not None
        places = gpd.read_file(self.place_shapefile)
        if places.crs is None or (expected is None and places.empty):
            raise ValueError("Place layer must be non-empty and have a CRS")
        place_key_col = self._resolve_place_key_column(places, graph_keys=expected)
        if expected is not None:
            place_keys = places[place_key_col].map(_normalize_token)
            missing = expected.keys() - set(place_keys)
            if missing:
                raise ValueError(
                    f"PLACE identifiers from graph {self.graph_json} are missing from "
                    f"column {place_key_col!r} in Place layer {self.place_shapefile} "
                    f"({len(missing)} missing, examples: {sorted(missing)[:5]}). "
                    "Check place_key_column and supply a layer covering the selected graph."
                )
            places = places.loc[place_keys.isin(expected.keys())].copy()
        if places.crs != blocks.crs:
            places = places.to_crs(blocks.crs)

        points = blocks[[block_col, pop_col, "geometry"]].copy()
        points["geometry"] = points.geometry.representative_point()
        joined = gpd.sjoin(
            points,
            places[[place_key_col, "geometry"]],
            how="inner",
            predicate="within",
        )
        # The inner join excludes blocks outside the selected Places.
        joined[block_col] = joined[block_col].map(_normalize_block_id)
        joined[place_key_col] = joined[place_key_col].map(_normalize_token)
        joined[pop_col] = pd.to_numeric(joined[pop_col], errors="coerce").fillna(0.0)
        joined = joined[(joined[block_col] != "") & (joined[place_key_col] != "")]
        joined = joined.groupby([block_col, place_key_col], as_index=False)[pop_col].sum()
        records = [
            {
                "id": row[block_col],
                self.mapping_column: {row[place_key_col]: float(row[pop_col])},
                "POP20": float(row[pop_col]),
            }
            for _, row in joined.iterrows()
        ]
        if expected is not None:
            self._validate_place_block_totals(records, expected)
        return records

    def _graph_place_totals(self) -> dict[str, float] | None:
        """Read normalized PLACE totals, retaining keys with zero population."""
        if self.graph_json is None:
            return None
        from atlas_posthoc_plus.io.graph import read_graph_frame

        graph = read_graph_frame(self.graph_json)
        if self.mapping_column not in graph.columns:
            raise ValueError(
                f"Graph {self.graph_json} is missing PLACE mapping column {self.mapping_column!r}"
            )
        expected: dict[str, float] = {}
        for mapping in graph[self.mapping_column]:
            if not isinstance(mapping, Mapping):
                raise TypeError(
                    f"Graph {self.graph_json} requires a mapping in {self.mapping_column!r} "
                    "for each node. Use an empty mapping for nodes outside selected Places."
                )
            for place, population in mapping.items():
                key = _normalize_token(place)
                if not key:
                    raise ValueError("PLACE mappings may not contain empty names")
                expected[key] = expected.get(key, 0.0) + float(population)
        return expected

    def _validate_place_block_totals(
        self, records: list[dict[str, Any]], expected: Mapping[str, float]
    ) -> None:
        """Ensure the reference-plan block geography reproduces the graph arm."""
        actual: dict[str, float] = {}
        for record in records:
            for place, population in record[self.mapping_column].items():
                key = _normalize_token(place)
                actual[key] = actual.get(key, 0.0) + float(population)
        if actual != expected:
            unassigned = sorted(expected.keys() - actual.keys())
            unexpected = sorted(actual.keys() - expected.keys())
            differing = [
                (key, actual[key], expected[key])
                for key in sorted(actual.keys() & expected.keys())
                if actual[key] != expected[key]
            ]
            raise ValueError(
                "Block-to-Place assignment does not reproduce the selected graph's "
                f"PLACE populations ({sum(actual.values()):,.0f} versus "
                f"{sum(expected.values()):,.0f}); use the same block-geometry "
                "checkpoint as graph generation. Selected keys with no assigned blocks "
                f"include {unassigned[:5]}. Unexpected assigned keys include {unexpected[:5]}. "
                f"Population differences (key, block total, graph total) include {differing[:5]}."
            )

    def _resolve_place_key_column(
        self, places: pd.DataFrame, *, graph_keys: Iterable[str] | None = None
    ) -> str:
        if self.place_key_column is not None:
            return _resolve_column(places.columns, [self.place_key_column], label="Place key")

        candidates = [
            column
            for column in ("GEOID", "PLACEFP", "NAME", "NAME20", "NAMELSAD")
            if column in places.columns
        ]
        if graph_keys is None and self.graph_json is not None and self.graph_json.is_file():
            graph_keys = self._graph_place_totals()
        if graph_keys:
            selected = set(graph_keys)
            scores = {
                column: len(selected.intersection(places[column].map(_normalize_token)))
                for column in candidates
            }
            best = max(scores, key=scores.get, default=None)
            if best is not None and scores[best] > 0:
                return best

        # Compatibility with the older name-keyed PLACE graph described in the
        # handoff. Current sampler graphs are detected as GEOID above.
        return _resolve_column(
            places.columns,
            ["NAME", "NAME20", "NAMELSAD", "GEOID", "PLACEFP"],
            label="Place key",
        )

    @staticmethod
    def _assign_blocks_to_cousubs(
        blocks: pd.DataFrame,
        shapefile: Path,
        *,
        county_col: str,
        shape_county_col: str | None,
        shape_mcd_col: str | None,
    ) -> pd.DataFrame:
        import geopandas as gpd

        if not isinstance(blocks, gpd.GeoDataFrame) or blocks.crs is None:
            raise ValueError("COUSUB spatial assignment requires block geometry with a CRS")
        cousubs = gpd.read_file(shapefile)
        shape_county_col = _resolve_column(
            cousubs.columns,
            [shape_county_col, "COUNTYFP", "COUNTYFP20", "COUNTY"],
            label="COUSUB county",
        )
        shape_mcd_col = _resolve_column(
            cousubs.columns,
            [shape_mcd_col, "COUSUBFP", "COUSUBFP20", "MCD", "COUSUB"],
            label="COUSUB id",
        )
        if cousubs.crs != blocks.crs:
            cousubs = cousubs.to_crs(blocks.crs)
        points = blocks.copy()
        points["geometry"] = points.geometry.representative_point()
        joined = gpd.sjoin(
            points,
            cousubs[[shape_county_col, shape_mcd_col, "geometry"]],
            how="left",
            predicate="within",
        )
        missing = joined[shape_mcd_col].isna()
        if missing.any():
            nearest = gpd.sjoin_nearest(
                points.loc[missing, ["geometry"]].to_crs("EPSG:26917"),
                cousubs[[shape_county_col, shape_mcd_col, "geometry"]].to_crs(
                    "EPSG:26917"
                ),
                how="left",
            )
            joined.loc[missing, shape_county_col] = nearest[shape_county_col].to_numpy()
            joined.loc[missing, shape_mcd_col] = nearest[shape_mcd_col].to_numpy()
        county = joined[shape_county_col].map(_normalize_token).str.zfill(3)
        cousub = joined[shape_mcd_col].map(_normalize_token).str.zfill(5)
        out = blocks.copy()
        out["__municipality__"] = county + "_" + cousub
        return out


def geography_from_args(args: Any, *, graph_json: str | Path | None = None) -> MunicipalGeography:
    """Build the shared adapter from the common municipal CLI flags."""
    mode = str(getattr(args, "municipality_mode", "COUSUB")).upper()
    namespace = getattr(args, "output_namespace", None) or mode.lower()
    place_shape = getattr(args, "place_shapefile", None)
    configured_graph = graph_json or getattr(args, "municipality_graph_json", None)
    return MunicipalGeography(
        mode=mode,
        graph_json=(Path(configured_graph) if configured_graph is not None else None),
        mapping_column=getattr(args, "mapping_column", "PLACE"),
        scalar_column=getattr(args, "scalar_column", "MCD"),
        place_shapefile=(Path(place_shape) if place_shape is not None else None),
        output_namespace=namespace,
        place_key_column=getattr(args, "place_key_column", None),
    )


def add_municipality_arguments(parser: Any) -> None:
    parser.add_argument("--municipality-mode", choices=["COUSUB", "PLACE"], default="COUSUB")
    parser.add_argument("--mapping-column", default="PLACE")
    parser.add_argument("--scalar-column", default="MCD")
    parser.add_argument("--place-shapefile", type=Path, default=None)
    parser.add_argument("--place-key-column", default=None)
    parser.add_argument("--municipality-graph-json", type=Path, default=None)
    parser.add_argument("--output-namespace", default=None)
