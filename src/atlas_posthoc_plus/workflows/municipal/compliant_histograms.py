#!/usr/bin/env python3
"""
=== USE THIS! ===
%%% run under oh-gerrymandering root %%%
atlas-posthoc-plus run municipal-histograms-compliant \
  --graph-path data/graph/OHpct20.json \
  --chains \
    outputs/chains/soft_constraint/congressional/run37_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run38_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run39_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run40_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
  --chain-ids run37 run38 run39 run40 \
  --block-metadata data/raw/tl_2024_39_tabblock20/tl_2024_39_tabblock20.shp \
  --block-col GEOID20 \
  --county-col-block COUNTYFP20 \
  --pop-col-block POP20 \
  --mcd-shapefile data/raw/tl_2024_39_cousub/tl_2024_39_cousub.shp \
  --mcd-shape-county-col COUNTYFP \
  --mcd-shape-mcd-col COUSUBFP \
  --enacted "inputs/plans/October 31 2025 CD BAF.xlsx" \
  --overturned "inputs/plans/SubSB258-Block-Assignment-File.xlsx" \
  --remedial "inputs/plans/March 2 2022 CD BAF.xlsx" \
  --output-dir outputs/municipal_split_compliant_37_40 \
  --max-maps-per-chain 1001
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from atlas_posthoc.observables.municipal_splits import (
    MunicipalSplitTrace,
    compute_municipal_split_statistics_for_assignment,
)
from atlas_posthoc.io.atlas_reader import iter_atlas_records
from atlas_posthoc.io.graph_reader import read_graph_json


# =============================================================================
# Basic normalization
# =============================================================================


def normalize_token(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def normalize_geoid(x: Any, width: int = 15) -> str:
    if pd.isna(x):
        return ""

    s = str(x).strip()

    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]

    s = re.sub(r"\s+", "", s)

    if s.isdigit() and len(s) <= width:
        s = s.zfill(width)

    return s


def normalize_district(x: Any) -> int | None:
    if pd.isna(x):
        return None

    s = str(x).strip()
    if s == "":
        return None

    if re.fullmatch(r"\d+(\.0)?", s):
        return int(float(s))

    m = re.search(r"\d+", s)
    if m:
        return int(m.group())

    return None


def normalize_atlas_key(x: Any) -> str:
    """
    Normalize AtlasIO districting keys.

    Common AtlasIO key:
        '["TROY 1-B_MIAMI__SD5__HD80"]'

    Graph NAME:
        TROY 1-B_MIAMI__SD5__HD80
    """
    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            return normalize_token(x[0])
        return "|".join(normalize_token(v) for v in x)

    s = normalize_token(x)

    try:
        obj = json.loads(s)
    except Exception:
        return s

    if isinstance(obj, list):
        if len(obj) == 1:
            return normalize_token(obj[0])
        return "|".join(normalize_token(v) for v in obj)

    if isinstance(obj, str):
        return normalize_token(obj)

    return s


def _pad_code(value: Any, width: int | None = None) -> str:
    s = normalize_token(value)
    if not s:
        return ""
    if width is not None and s.isdigit():
        return s.zfill(width)
    return s


def first_existing_column(columns: list[str], candidates: list[str]) -> str:
    normalized = {
        c.lower().replace(" ", "").replace("_", "").replace(":", ""): c
        for c in columns
    }

    for candidate in candidates:
        key = candidate.lower().replace(" ", "").replace("_", "").replace(":", "")
        if key in normalized:
            return normalized[key]

    # Accept columns like DistrictID:1 when candidate is DistrictID.
    for candidate in candidates:
        key = candidate.lower().replace(" ", "").replace("_", "").replace(":", "")
        for norm_col, original_col in normalized.items():
            if norm_col.startswith(key):
                return original_col

    raise ValueError(
        f"Could not find any of these columns: {candidates}. "
        f"Available columns: {columns}"
    )


def resolve_column(
    df: pd.DataFrame,
    user_col: str | None,
    candidates: list[str],
    *,
    label: str,
) -> str:
    if user_col is not None:
        if user_col not in df.columns:
            raise ValueError(
                f"{label} column {user_col!r} not found. "
                f"Available columns: {list(df.columns)}"
            )
        return user_col

    return first_existing_column(list(df.columns), candidates)


# =============================================================================
# AtlasIO chain reading and graph assignment conversion
# =============================================================================


def open_text_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("rt")


def iter_json_values_from_line(line: str):
    """
    Robustly parse one physical line that may contain one JSON object or
    multiple concatenated JSON objects.
    """
    decoder = json.JSONDecoder()
    idx = 0
    n = len(line)

    while idx < n:
        while idx < n and line[idx].isspace():
            idx += 1

        if idx >= n:
            break

        obj, end = decoder.raw_decode(line, idx)
        yield obj
        idx = end


def maybe_json_loads(x: Any) -> Any:
    """Handle occasional double-encoded JSON strings."""
    while isinstance(x, str):
        s = x.strip()
        if not s:
            return x
        if not (s.startswith("{") or s.startswith("[") or s.startswith('"')):
            return x
        try:
            x = json.loads(s)
        except json.JSONDecodeError:
            return x
    return x


def iter_atlasio_districtings(chain_path: Path):
    """Yield raw districting payloads through the atlas-posthoc reader."""
    for record in iter_atlas_records(chain_path, skip_bad_lines=True):
        districting = (
            record.raw_assignment
            if record.raw_assignment is not None
            else record.assignment
        )
        yield record.line_number, districting


def _node_attributes(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}

    if isinstance(record.get("attributes"), dict):
        return record["attributes"]

    if isinstance(record.get("properties"), dict):
        return record["properties"]

    return record


def load_graph_nodes(
    graph_path: Path,
    *,
    node_id_col: str,
    name_col: str,
    county_col: str,
    municipality_cols: list[str],
    pop_col: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """
    Return:
      graph_nodes: list of node dictionaries for atlas-posthoc municipal splits
      nodes_by_name: name -> node dictionary
      id_to_node: node id -> node dictionary
    """
    graph = read_graph_json(graph_path, node_id_col=node_id_col)
    records = [dict(node) for node in graph.nodes]

    required = [node_id_col, name_col, county_col, pop_col] + municipality_cols
    for attrs in records:
        missing = [col for col in required if col not in attrs]
        if missing:
            raise KeyError(
                f"Graph node is missing required columns {missing}. "
                f"Node example: {attrs}"
            )

    graph_nodes: list[dict[str, Any]] = []
    nodes_by_name: dict[str, dict[str, Any]] = {}
    id_to_node: dict[Any, dict[str, Any]] = {}

    for attrs in records:
        clean = dict(attrs)
        clean[node_id_col] = str(clean[node_id_col])
        clean[name_col] = str(clean[name_col])
        clean[county_col] = str(clean[county_col])

        for col in municipality_cols:
            clean[col] = normalize_token(clean[col])

        clean[pop_col] = float(pd.to_numeric(clean[pop_col], errors="coerce"))

        graph_nodes.append(clean)
        nodes_by_name[clean[name_col]] = clean
        id_to_node[clean[node_id_col]] = clean

    if len(nodes_by_name) != len(graph_nodes):
        raise ValueError(f"Duplicate graph node names found using {name_col!r}")

    if len(id_to_node) != len(graph_nodes):
        raise ValueError(f"Duplicate graph node IDs found using {node_id_col!r}")

    return graph_nodes, nodes_by_name, id_to_node


def atlasio_districting_to_assignment(
    districting: Any,
    graph_nodes: list[dict[str, Any]],
    nodes_by_name: dict[str, dict[str, Any]],
    *,
    node_id_col: str,
    name_col: str,
) -> dict[str, int]:
    """
    Convert an AtlasIO districting payload to:
        {graph_node_id: district}

    Supports the common run37--run40 list-of-singleton-dicts format:
        [{'["PRECINCT_NAME"]': 15}, ...]
    and raw dict districtings.
    """
    name_to_id = {
        normalize_atlas_key(node[name_col]): str(node[node_id_col])
        for node in graph_nodes
    }

    assignment: dict[str, int] = {}
    unmatched: list[str] = []

    if isinstance(districting, list):
        # list of singleton dictionaries
        if all(isinstance(x, dict) and len(x) == 1 for x in districting):
            for item in districting:
                raw_key, raw_dist = next(iter(item.items()))
                key = normalize_atlas_key(raw_key)
                node_id = name_to_id.get(key)
                if node_id is None:
                    unmatched.append(key)
                    continue
                dist = normalize_district(raw_dist)
                if dist is None:
                    raise ValueError(f"Could not normalize district label: {raw_dist!r}")
                assignment[node_id] = dist

        # node-order vector fallback
        elif len(districting) == len(graph_nodes) and not isinstance(
            districting[0], (list, tuple, dict, set)
        ):
            for node, raw_dist in zip(graph_nodes, districting):
                dist = normalize_district(raw_dist)
                if dist is None:
                    raise ValueError(f"Could not normalize district label: {raw_dist!r}")
                assignment[str(node[node_id_col])] = dist

        else:
            raise ValueError(
                "Unsupported list-valued districting payload: "
                f"len={len(districting)}, first type={type(districting[0]) if districting else None}"
            )

    elif isinstance(districting, dict):
        for raw_key, raw_dist in districting.items():
            key = normalize_atlas_key(raw_key)
            node_id = name_to_id.get(key, key)

            dist = normalize_district(raw_dist)
            if dist is None:
                raise ValueError(f"Could not normalize district label: {raw_dist!r}")

            assignment[str(node_id)] = dist

    else:
        raise ValueError(f"Unsupported districting payload type: {type(districting)}")

    if unmatched:
        raise ValueError(
            f"{len(unmatched)} AtlasIO keys did not match graph {name_col!r}. "
            f"Examples: {unmatched[:5]}"
        )

    if len(assignment) != len(graph_nodes):
        raise ValueError(
            f"Matched {len(assignment)} assignments, but graph has {len(graph_nodes)} nodes."
        )

    return assignment


# =============================================================================
# Standardized county-compliance filter
# =============================================================================


def county_compliance_summary_for_assignment(
    assignment: dict[str, int],
    id_to_node: dict[str, dict[str, Any]],
    *,
    county_col: str,
    max_districts_per_county: int,
) -> dict[str, Any]:
    county_to_districts: dict[str, set[int]] = defaultdict(set)

    assigned_nodes = 0
    missing_nodes = 0

    for node_id, attrs in id_to_node.items():
        dist = assignment.get(str(node_id))

        if dist is None:
            missing_nodes += 1
            continue

        assigned_nodes += 1
        county_to_districts[str(attrs[county_col])].add(dist)

    split_sizes = [len(v) for v in county_to_districts.values()]
    worst_split = max(split_sizes) if split_sizes else 0
    double_cross_count = sum(
        1 for s in split_sizes if s > max_districts_per_county
    )

    return {
        "compliant": double_cross_count == 0,
        "assigned_nodes": assigned_nodes,
        "missing_nodes": missing_nodes,
        "assigned_fraction": assigned_nodes / max(assigned_nodes + missing_nodes, 1),
        "double_cross_count": double_cross_count,
        "worst_split": worst_split,
    }


# =============================================================================
# Build compliant municipal traces
# =============================================================================


def _summary_to_trace_values(summary: Any, *, pop_col: str) -> tuple[float, float, float]:
    """
    The atlas-posthoc summary object exposes attributes in current versions and
    to_dict() in the plotting script already used in this project. This helper
    supports both.
    """
    if hasattr(summary, "n_cut_municipalities"):
        return (
            float(summary.n_cut_municipalities),
            float(summary.ousted_population),
            float(summary.ousted_population_share),
        )

    if isinstance(summary, dict):
        return (
            float(summary["n_cut_municipalities"]),
            float(summary["ousted_population"]),
            float(summary["ousted_population_share"]),
        )

    if hasattr(summary, "to_dict"):
        d = summary.to_dict()
        return (
            float(d["n_cut_municipalities"]),
            float(d["ousted_population"]),
            float(d["ousted_population_share"]),
        )

    raise TypeError(f"Unsupported municipal-split summary type: {type(summary)}")


def build_compliant_municipal_split_traces(
    *,
    chains: list[Path],
    chain_ids: list[str],
    graph_nodes: list[dict[str, Any]],
    nodes_by_name: dict[str, dict[str, Any]],
    id_to_node: dict[str, dict[str, Any]],
    node_id_col: str,
    name_col: str,
    county_col: str,
    municipality_cols: list[str],
    pop_col: str,
    max_districts_per_county: int,
    thin_step: int,
    max_maps_per_chain: int | None,
) -> tuple[list[MunicipalSplitTrace], pd.DataFrame]:
    traces: list[MunicipalSplitTrace] = []
    compliance_rows: list[dict[str, Any]] = []

    municipality_cols_tuple = tuple(municipality_cols)

    for chain_id, chain_path in zip(chain_ids, chains):
        print(f"\nSweeping chain {chain_id}: {chain_path}")

        proposals: list[int] = []
        n_cut: list[float] = []
        ousted_pop: list[float] = []
        ousted_share: list[float] = []

        maps_seen = 0
        maps_used_after_thin = 0
        compliant_maps = 0
        invalid_maps = 0

        double_cross_counts: list[int] = []
        worst_splits: list[int] = []

        for _, districting in iter_atlasio_districtings(chain_path):
            maps_seen += 1

            if thin_step > 1 and ((maps_seen - 1) % thin_step != 0):
                continue

            maps_used_after_thin += 1

            if (
                max_maps_per_chain is not None
                and maps_used_after_thin > max_maps_per_chain
            ):
                break

            try:
                assignment = atlasio_districting_to_assignment(
                    districting,
                    graph_nodes,
                    nodes_by_name,
                    node_id_col=node_id_col,
                    name_col=name_col,
                )

                comp = county_compliance_summary_for_assignment(
                    assignment,
                    id_to_node,
                    county_col=county_col,
                    max_districts_per_county=max_districts_per_county,
                )

                double_cross_counts.append(comp["double_cross_count"])
                worst_splits.append(comp["worst_split"])

                if not comp["compliant"]:
                    continue

                summary = compute_municipal_split_statistics_for_assignment(
                    assignment,
                    graph_nodes,
                    municipality_cols=municipality_cols_tuple,
                    node_id_col=node_id_col,
                    pop_col=pop_col,
                    skip_missing_municipality=True,
                    require_complete_assignment=False,
                )

                c, op, os = _summary_to_trace_values(summary, pop_col=pop_col)

                proposals.append((maps_seen - 1) * 10000)
                n_cut.append(c)
                ousted_pop.append(op)
                ousted_share.append(os)

                compliant_maps += 1

            except Exception as exc:
                invalid_maps += 1
                if invalid_maps <= 5:
                    print(f"  FAILED map {maps_seen}: {type(exc).__name__}: {exc}")

        trace = MunicipalSplitTrace(
            chain_id=chain_id,
            proposals=proposals,
            n_cut_municipalities=n_cut,
            ousted_population=ousted_pop,
            ousted_population_share=ousted_share,
            n_maps=len(proposals),
            municipality_cols=municipality_cols,
            pop_col=pop_col,
        )
        traces.append(trace)

        compliance_rows.append(
            {
                "chain_id": chain_id,
                "chain_path": str(chain_path),
                "maps_seen": maps_seen,
                "maps_used_after_thin": maps_used_after_thin,
                "compliant_maps": compliant_maps,
                "invalid_maps": invalid_maps,
                "compliant_fraction_after_thin": compliant_maps
                / max(maps_used_after_thin, 1),
                "avg_double_crosses": float(np.mean(double_cross_counts))
                if double_cross_counts
                else np.nan,
                "max_double_crosses": int(np.max(double_cross_counts))
                if double_cross_counts
                else 0,
                "worst_single_county_split": int(np.max(worst_splits))
                if worst_splits
                else 0,
            }
        )

        print(f"  maps_seen={maps_seen}")
        print(f"  maps_used_after_thin={maps_used_after_thin}")
        print(f"  compliant_maps={compliant_maps}")
        print(f"  invalid_maps={invalid_maps}")
        print(f"  trace_n_maps={len(proposals)}")

    return traces, pd.DataFrame(compliance_rows)


def trace_to_dict(trace: MunicipalSplitTrace) -> dict[str, Any]:
    """
    Avoid assuming dataclass behavior forever; write the stable JSON shape that
    atlas-posthoc already uses for municipal_split_traces.json.
    """
    return {
        "chain_id": trace.chain_id,
        "proposals": list(trace.proposals),
        "n_cut_municipalities": list(trace.n_cut_municipalities),
        "ousted_population": list(trace.ousted_population),
        "ousted_population_share": list(trace.ousted_population_share),
        "n_maps": int(trace.n_maps),
        "municipality_cols": list(trace.municipality_cols),
        "pop_col": trace.pop_col,
    }


def write_traces_json(path: Path, traces: list[MunicipalSplitTrace]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump([trace_to_dict(trace) for trace in traces], f, indent=2)


# =============================================================================
# Reference plans: use the same atlas-posthoc municipal-split function
# =============================================================================


def read_spatial_or_table(path: Path) -> pd.DataFrame:
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

    try:
        return pd.read_csv(path, dtype=str)
    except Exception as exc:
        raise ValueError(
            f"Unsupported block metadata extension: {suffix}. "
            "Use CSV, Parquet, Shapefile, zipped Shapefile, GeoPackage, or GeoJSON."
        ) from exc


def assign_blocks_to_mcds(
    blocks,
    mcd_shapefile: str | Path,
    *,
    block_county_col: str,
    mcd_county_col: str | None = None,
    mcd_col: str | None = None,
    output_col: str = "MCD",
    nearest_crs: str = "EPSG:26917",
):
    import geopandas as gpd

    mcd_shapefile = Path(mcd_shapefile)
    if not mcd_shapefile.exists():
        raise FileNotFoundError(f"MCD shapefile not found: {mcd_shapefile}")

    if not hasattr(blocks, "geometry"):
        raise ValueError("blocks must be a GeoDataFrame with geometry")

    if block_county_col not in blocks.columns:
        raise ValueError(
            f"block_county_col={block_county_col!r} not found in block columns. "
            f"Available columns: {list(blocks.columns)}"
        )

    mcds = gpd.read_file(mcd_shapefile)
    if mcds.empty:
        raise ValueError(f"MCD shapefile is empty: {mcd_shapefile}")

    mcd_county_col = resolve_column(
        mcds,
        mcd_county_col,
        ["COUNTYFP", "COUNTYFP20", "COUNTYFP10", "COUNTY", "COUNTYID"],
        label="MCD county",
    )
    mcd_col = resolve_column(
        mcds,
        mcd_col,
        ["COUSUBFP", "COUSUBFP20", "COUSUBFP10", "MCD", "MCD20", "NAMELSAD", "NAME"],
        label="MCD / county subdivision",
    )

    blocks = blocks.copy()
    mcds = mcds[[mcd_county_col, mcd_col, "geometry"]].copy()

    if blocks.crs is None:
        raise ValueError("Block GeoDataFrame has no CRS")

    if mcds.crs is None:
        raise ValueError("MCD GeoDataFrame has no CRS")

    if blocks.crs != mcds.crs:
        mcds = mcds.to_crs(blocks.crs)

    mcds = mcds.rename(
        columns={mcd_county_col: "__mcd_county__", mcd_col: "__mcd_id__"}
    )

    points = blocks.copy()
    points["geometry"] = points.geometry.representative_point()

    joined = gpd.sjoin(
        points,
        mcds[["__mcd_county__", "__mcd_id__", "geometry"]],
        how="left",
        predicate="within",
    )

    missing_mask = joined["__mcd_id__"].isna()

    if missing_mask.any():
        n_missing = int(missing_mask.sum())
        print(
            f"  WARNING: {n_missing} blocks were not assigned by within-join; "
            "trying nearest fallback."
        )
        points_proj = points.to_crs(nearest_crs)
        mcds_proj = mcds.to_crs(nearest_crs)
        nearest = gpd.sjoin_nearest(
            points_proj.loc[missing_mask, ["geometry"]],
            mcds_proj[["__mcd_county__", "__mcd_id__", "geometry"]],
            how="left",
        )
        joined.loc[missing_mask, "__mcd_county__"] = nearest["__mcd_county__"].to_numpy()
        joined.loc[missing_mask, "__mcd_id__"] = nearest["__mcd_id__"].to_numpy()

    if joined["__mcd_id__"].isna().any():
        n_missing = int(joined["__mcd_id__"].isna().sum())
        raise ValueError(f"{n_missing} blocks could not be assigned to an MCD")

    county_key = joined["__mcd_county__"].map(lambda x: _pad_code(x, 3))
    mcd_key = joined["__mcd_id__"].map(lambda x: _pad_code(x, 5))
    blocks[output_col] = county_key + "_" + mcd_key

    print("  Assigned blocks to MCDs using:")
    print(f"    MCD shapefile: {mcd_shapefile}")
    print(f"    MCD county col: {mcd_county_col}")
    print(f"    MCD id col: {mcd_col}")
    print(f"    output col: {output_col}")

    return blocks


def read_block_metadata(
    path: str | Path,
    *,
    block_col: str | None = None,
    county_col: str | None = None,
    pop_col: str | None = None,
    block_mcd_col: str | None = None,
    mcd_shapefile: str | Path | None = None,
    mcd_shape_county_col: str | None = None,
    mcd_shape_mcd_col: str | None = None,
) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Block metadata file not found: {path}")

    df = read_spatial_or_table(path)
    if df.empty:
        raise ValueError(f"Block metadata file is empty: {path}")

    block_col = resolve_column(
        df,
        block_col,
        ["GEOID20", "BLOCK", "GEOID", "GEOID10", "BLOCKID", "block_geoid", "block_id"],
        label="block id",
    )
    county_col = resolve_column(
        df,
        county_col,
        ["COUNTY", "COUNTYFP20", "COUNTYFP", "COUNTYFP10", "COUNTY_ID", "COUNTYID"],
        label="county",
    )
    pop_col = resolve_column(
        df,
        pop_col,
        ["POP20", "POPULATION", "POP", "P0010001", "P1_001N", "TOTALPOP", "TOTPOP"],
        label="population",
    )

    if block_mcd_col is not None:
        if block_mcd_col not in df.columns:
            raise ValueError(
                f"block_mcd_col={block_mcd_col!r} not found. "
                f"Available columns: {list(df.columns)}"
            )
        mcd_col = block_mcd_col
    else:
        try:
            mcd_col = resolve_column(
                df,
                None,
                ["MCD", "COUSUBFP20", "COUSUBFP", "COUSUBFP10", "MCD_ID", "MCDID", "MCD20"],
                label="MCD / municipality",
            )
        except ValueError:
            if mcd_shapefile is None:
                raise ValueError(
                    "The block file has no MCD / municipality column. "
                    "Provide either --block-mcd-col or --mcd-shapefile."
                )

            df = assign_blocks_to_mcds(
                df,
                mcd_shapefile,
                block_county_col=county_col,
                mcd_county_col=mcd_shape_county_col,
                mcd_col=mcd_shape_mcd_col,
                output_col="__assigned_mcd__",
            )
            mcd_col = "__assigned_mcd__"

    out = df[[block_col, county_col, mcd_col, pop_col]].copy()
    out[block_col] = out[block_col].map(normalize_geoid)
    out[county_col] = out[county_col].map(lambda x: _pad_code(x, 3))
    out[mcd_col] = out[mcd_col].map(normalize_token)
    out[pop_col] = pd.to_numeric(out[pop_col], errors="coerce").fillna(0.0)

    out = out[out[block_col] != ""]
    out = out.rename(
        columns={block_col: "id", county_col: "COUNTY", mcd_col: "MCD", pop_col: "POP20"}
    )
    out = out.groupby(["id", "COUNTY", "MCD"], dropna=False, as_index=False)["POP20"].sum()

    total_pop = float(out["POP20"].sum())
    if total_pop <= 0:
        raise ValueError(
            "Block metadata population sums to zero. "
            "Your block shapefile may not contain population."
        )

    print("  Resolved block metadata columns:")
    print(f"    block id: {block_col}")
    print(f"    county:   {county_col}")
    print(f"    MCD:      {mcd_col}")
    print(f"    pop:      {pop_col}")
    print(f"    total pop in metadata: {total_pop:,.0f}")
    print(f"    rows after aggregation: {len(out):,}")

    return out.to_dict("records")


def read_block_assignment_excel(
    path: str | Path,
    *,
    block_col: str | None = None,
    district_col: str | None = None,
) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Block assignment file not found: {path}")

    df = pd.read_excel(path, dtype=str)

    if df.empty:
        raise ValueError(f"Block assignment spreadsheet is empty: {path}")

    block_col = resolve_column(
        df,
        block_col,
        ["BLOCK", "GEOID20", "GEOID", "Block", "block_geoid", "block_id"],
        label="block id",
    )
    district_col = resolve_column(
        df,
        district_col,
        ["DistrictID", "DISTRICT", "District", "CD", "CONG_DIST", "CONGRESSIONAL", "assignment", "DistrictID:1"],
        label="district",
    )

    out = df[[block_col, district_col]].copy()
    out[block_col] = out[block_col].map(normalize_geoid)
    out[district_col] = out[district_col].astype(str).str.strip()
    out = out[(out[block_col] != "") & (out[district_col] != "")]
    out = out.drop_duplicates(subset=[block_col], keep="last")

    return dict(zip(out[block_col], out[district_col]))


def compute_reference_summaries(
    *,
    block_nodes: list[dict[str, Any]],
    enacted_path: str | Path,
    overturned_path: str | Path,
    remedial_path: str | Path,
    assignment_block_col: str | None = None,
    assignment_district_col: str | None = None,
) -> dict[str, Any]:
    plan_files = {
        "Enacted ('25)": enacted_path,
        "Overturned ('21)": overturned_path,
        "Remedial ('22)": remedial_path,
    }

    summaries: dict[str, Any] = {}

    for label, path in plan_files.items():
        assignment = read_block_assignment_excel(
            path,
            block_col=assignment_block_col,
            district_col=assignment_district_col,
        )
        summary = compute_municipal_split_statistics_for_assignment(
            assignment,
            block_nodes,
            municipality_cols=("COUNTY", "MCD"),
            node_id_col="id",
            pop_col="POP20",
            skip_missing_municipality=True,
            require_complete_assignment=False,
        )
        summaries[label] = summary

    return summaries


# =============================================================================
# Plotting
# =============================================================================


def pooled_values(traces: list[MunicipalSplitTrace], *, metric: str) -> np.ndarray:
    vals = []
    for tr in traces:
        vals.extend(getattr(tr, metric))
    arr = np.asarray(vals, dtype=float)
    return arr[np.isfinite(arr)]


def plot_histograms(
    *,
    traces: list[MunicipalSplitTrace],
    reference_summaries: dict[str, Any],
    ousted_scale: str,
    title: str,
    output_path: str | Path,
) -> None:
    cut_values = pooled_values(traces, metric="n_cut_municipalities")
    ousted_metric = (
        "ousted_population_share" if ousted_scale == "share" else "ousted_population"
    )
    ousted_values = pooled_values(traces, metric=ousted_metric)

    if cut_values.size == 0 or ousted_values.size == 0:
        raise RuntimeError("No compliant municipal split values to plot.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    reference_styles = {
        "Enacted ('25)": {"color": "#9467bd", "linestyle": "-",  "linewidth": 2.2},
        "Overturned ('21)": {"color": "#2ca02c", "linestyle": "--", "linewidth": 2.2},
        "Remedial ('22)": {"color": "#d62728", "linestyle": "-.", "linewidth": 2.2},
    }

    cut_min = int(np.floor(cut_values.min()))
    cut_max = int(np.ceil(cut_values.max()))
    bins_cut = np.arange(cut_min - 0.5, cut_max + 1.5, 1.0)

    axes[0].hist(cut_values, bins=bins_cut, edgecolor="black", alpha=0.75)
    axes[0].set_xlabel("Number of cut municipalities")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Compliant ensemble distribution")

    axes[1].hist(ousted_values, bins=30, edgecolor="black", alpha=0.75)
    axes[1].set_ylabel("Count")
    axes[1].set_title("Compliant ensemble distribution")

    if ousted_scale == "share":
        axes[1].set_xlabel("Ousted municipal population share")
        axes[1].ticklabel_format(axis="x", style="plain")
    else:
        axes[1].set_xlabel("Ousted municipal population")

    handles = []

    for label, summary in reference_summaries.items():
        style = reference_styles[label]
        axes[0].axvline(summary.n_cut_municipalities, zorder=10, alpha=0.95, **style)
        val = summary.ousted_population_share if ousted_scale == "share" else summary.ousted_population
        axes[1].axvline(val, zorder=10, alpha=0.95, **style)
        handles.append(Line2D([0], [0], label=label, **style))

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title)
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_reference_json(path: str | Path, summaries: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump({label: summary.to_dict() for label, summary in summaries.items()}, f, indent=2)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build compliant municipal-split traces using the standardized county "
            "filter, then overlay Enacted / Overturned / Remedial on histograms."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--graph-path", required=True, type=Path)
    parser.add_argument("--chains", required=True, nargs="+", type=Path)
    parser.add_argument("--chain-ids", nargs="*", default=None)

    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-traces-json", default=None, type=Path)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--reference-json", default=None, type=Path)

    parser.add_argument("--n-districts", default=15, type=int)
    parser.add_argument("--node-id-col", default="id")
    parser.add_argument("--name-col", default="NAME")
    parser.add_argument("--county-col", default="COUNTY")
    parser.add_argument("--municipality-cols", nargs="+", default=["COUNTY", "MCD"])
    parser.add_argument("--pop-col", default="POP20")

    parser.add_argument("--max-districts-per-county", default=2, type=int)
    parser.add_argument("--thin-step", default=1, type=int)
    parser.add_argument("--max-maps-per-chain", default=None, type=int)

    parser.add_argument("--block-metadata", required=True, type=Path)
    parser.add_argument("--block-col", default=None)
    parser.add_argument("--county-col-block", default=None)
    parser.add_argument("--pop-col-block", default=None)
    parser.add_argument("--block-mcd-col", default=None)

    parser.add_argument("--mcd-shapefile", type=Path, default=None)
    parser.add_argument("--mcd-shape-county-col", default=None)
    parser.add_argument("--mcd-shape-mcd-col", default=None)

    parser.add_argument("--enacted", required=True, type=Path)
    parser.add_argument("--overturned", required=True, type=Path)
    parser.add_argument("--remedial", required=True, type=Path)
    parser.add_argument("--assignment-block-col", default=None)
    parser.add_argument("--assignment-district-col", default=None)

    parser.add_argument(
        "--ousted-scale",
        choices=["share", "population"],
        default=os.environ.get("ATLAS_POSTHOC_MUNICIPAL_OUSTED_SCALE", "share"),
    )
    parser.add_argument(
        "--title",
        default="Municipal split histograms under compliant ensemble",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    chain_ids = (
        args.chain_ids
        if args.chain_ids
        else [f"chain_{i + 1}" for i in range(len(args.chains))]
    )

    if len(chain_ids) != len(args.chains):
        raise ValueError("--chain-ids must have the same length as --chains")

    output_traces_json = (
        args.output_traces_json
        if args.output_traces_json is not None
        else args.output_dir / "municipal_split_traces_compliant.json"
    )
    output_plot = (
        args.output
        if args.output is not None
        else args.output_dir / "oh_municipal_split_histograms_with_refs_compliant.png"
    )
    reference_json = (
        args.reference_json
        if args.reference_json is not None
        else args.output_dir / "oh_municipal_split_histograms_with_refs_reference_summaries.json"
    )
    compliance_summary_path = args.output_dir / "municipal_split_compliance_summary.csv"

    print("=" * 80)
    print("Compliant municipal-split histograms with reference plans")
    print("=" * 80)

    print("\n[1/5] Loading graph nodes...")
    graph_nodes, nodes_by_name, id_to_node = load_graph_nodes(
        args.graph_path,
        node_id_col=args.node_id_col,
        name_col=args.name_col,
        county_col=args.county_col,
        municipality_cols=args.municipality_cols,
        pop_col=args.pop_col,
    )
    print(f"Loaded {len(graph_nodes):,} graph nodes.")

    print("\n[2/5] Building compliant municipal-split traces...")
    traces, compliance_summary = build_compliant_municipal_split_traces(
        chains=args.chains,
        chain_ids=chain_ids,
        graph_nodes=graph_nodes,
        nodes_by_name=nodes_by_name,
        id_to_node=id_to_node,
        node_id_col=args.node_id_col,
        name_col=args.name_col,
        county_col=args.county_col,
        municipality_cols=args.municipality_cols,
        pop_col=args.pop_col,
        max_districts_per_county=args.max_districts_per_county,
        thin_step=args.thin_step,
        max_maps_per_chain=args.max_maps_per_chain,
    )

    compliance_summary.to_csv(compliance_summary_path, index=False)
    write_traces_json(output_traces_json, traces)
    print(f"Wrote compliance summary: {compliance_summary_path}")
    print(f"Wrote compliant traces: {output_traces_json}")
    print(compliance_summary.to_string(index=False))

    total_compliant = sum(trace.n_maps for trace in traces)
    if total_compliant == 0:
        raise RuntimeError("No compliant maps retained from the chains.")

    print("\n[3/5] Loading block metadata for reference plans...")
    block_nodes = read_block_metadata(
        args.block_metadata,
        block_col=args.block_col,
        county_col=args.county_col_block,
        pop_col=args.pop_col_block,
        block_mcd_col=args.block_mcd_col,
        mcd_shapefile=args.mcd_shapefile,
        mcd_shape_county_col=args.mcd_shape_county_col,
        mcd_shape_mcd_col=args.mcd_shape_mcd_col,
    )
    print(f"Loaded {len(block_nodes):,} block metadata rows.")

    print("\n[4/5] Computing reference-plan municipal split summaries...")
    reference_summaries = compute_reference_summaries(
        block_nodes=block_nodes,
        enacted_path=args.enacted,
        overturned_path=args.overturned,
        remedial_path=args.remedial,
        assignment_block_col=args.assignment_block_col,
        assignment_district_col=args.assignment_district_col,
    )

    for label, summary in reference_summaries.items():
        print(
            f"  {label}: cuts={summary.n_cut_municipalities}, "
            f"ousted_pop={summary.ousted_population:,.0f}, "
            f"ousted_share={summary.ousted_population_share:.4%}"
        )

    write_reference_json(reference_json, reference_summaries)
    print(f"Wrote reference summaries: {reference_json}")

    print("\n[5/5] Plotting...")
    plot_histograms(
        traces=traces,
        reference_summaries=reference_summaries,
        ousted_scale=args.ousted_scale,
        title=args.title,
        output_path=output_plot,
    )
    print(f"Wrote figure: {output_plot}")
    print("\nDone.")


if __name__ == "__main__":
    main()
