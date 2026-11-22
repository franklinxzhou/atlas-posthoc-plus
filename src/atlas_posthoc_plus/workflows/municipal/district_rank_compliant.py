#!/usr/bin/env python3
"""
atlas-posthoc-plus run district-rank-municipal-compliant \
  --graph-json data/graph/OHpct20.json \
  --chain-files \
    outputs/chains/soft_constraint/congressional/run37_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run38_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run39_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run40_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
  --crosswalk-csv outputs/oh_block_to_precinct_crosswalk.csv \
  --block-metadata data/raw/tl_2024_39_tabblock20/tl_2024_39_tabblock20.shp \
  --mcd-shapefile data/raw/tl_2024_39_cousub/tl_2024_39_cousub.shp \
  --enacted-baf "inputs/plans/October 31 2025 CD BAF.xlsx" \
  --overturned-baf "inputs/plans/SubSB258-Block-Assignment-File.xlsx" \
  --remedial-baf "inputs/plans/March 2 2022 CD BAF.xlsx" \
  --output-dir outputs/district_rank_municipal_histograms_compliant_37_40 \
  --max-maps-per-chain 1001
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from atlas_posthoc_plus.paper import (
    figure_legend,
    official_plan_styles,
    paper_mode_enabled,
    save_figure,
    set_panel_title,
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from atlas_posthoc_plus.io.atlas_records import iter_map_records
from atlas_posthoc_plus.io.chain_sweep import (
    cached_assignment_series,
    cached_compliance,
    configured_sweep,
)
from atlas_posthoc_plus.io.graph import read_graph_frame
from atlas_posthoc_plus.workflows.municipal.geography import (
    MunicipalGeography,
    add_municipality_arguments,
    geography_from_args,
)


# ---------------------------------------------------------------------
# Basic normalization helpers
# ---------------------------------------------------------------------
def canonical_colname(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def normalize_token(value: Any) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def normalize_block_id(value: Any) -> str:
    s = normalize_token(value)
    return s.zfill(15) if s else ""


def normalize_atlas_key(value: Any) -> str:
    """
    Normalize AtlasIO districting keys.

    AtlasIO chain keys often look like:
        '["TROY 1-B_MIAMI__SD5__HD80"]'

    while graph NAME is:
        TROY 1-B_MIAMI__SD5__HD80
    """
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return normalize_token(value[0])
        return "|".join(normalize_token(x) for x in value)

    s = normalize_token(value)

    try:
        obj = json.loads(s)
    except Exception:
        return s

    if isinstance(obj, list):
        if len(obj) == 1:
            return normalize_token(obj[0])
        return "|".join(normalize_token(x) for x in obj)

    if isinstance(obj, str):
        return normalize_token(obj)

    return s


def resolve_column(
    columns: Iterable[str],
    candidates: Iterable[str | None],
    *,
    label: str,
) -> str:
    columns = list(columns)
    normalized = {canonical_colname(c): c for c in columns}

    for candidate in candidates:
        if candidate is None:
            continue
        key = canonical_colname(candidate)
        if key in normalized:
            return normalized[key]

    for candidate in candidates:
        if candidate is None:
            continue
        key = canonical_colname(candidate)
        for norm_col, original_col in normalized.items():
            if norm_col.startswith(key):
                return original_col

    raise ValueError(
        f"Could not resolve {label} column from candidates {list(candidates)}. "
        f"Available columns: {columns}"
    )


def coerce_district_label(value: Any) -> int:
    if value is None:
        raise ValueError("district label is None")

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"district label is not finite: {value}")
        if abs(value - round(value)) < 1e-9:
            return int(round(value))
        raise ValueError(f"non-integer district label: {value}")

    s = normalize_token(value)
    if s.lstrip("-").isdigit():
        return int(s)

    raise ValueError(f"cannot convert district label to int: {value!r}")


# ---------------------------------------------------------------------
# Graph / chain readers
# ---------------------------------------------------------------------
def open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("r")


def maybe_json_loads(x: Any) -> Any:
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


def iter_json_values_from_line(line: str):
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


def iter_atlasio_map_records(path: str | Path):
    yield from iter_map_records(path)


def read_graph_nodes(
    graph_json: str | Path,
    geography: MunicipalGeography | None = None,
) -> pd.DataFrame:
    df = read_graph_frame(graph_json).copy()

    geography = geography or MunicipalGeography()
    required = ["id", "NAME", "ORIG_NAME", "COUNTY", "POP20", "G20PREDEM", "G20PREREP"]
    required.extend(geography.municipality_columns)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Graph node table is missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df["id"] = pd.to_numeric(df["id"], errors="raise").astype(int)
    for col in ["POP20", "G20PREDEM", "G20PREREP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["COUNTY"] = df["COUNTY"].map(normalize_token)
    if geography.mode == "COUSUB":
        df[geography.scalar_column] = df[geography.scalar_column].map(normalize_token)
    df["NAME"] = df["NAME"].map(normalize_token)
    df["ORIG_NAME"] = df["ORIG_NAME"].map(normalize_token)

    geography.validate_graph_nodes(df)

    return df


def atlasio_districting_to_series(
    districting: Any,
    graph_nodes: pd.DataFrame,
    *,
    key_col: str = "NAME",
    node_id_col: str = "id",
) -> pd.Series:
    cached = cached_assignment_series(districting, node_id_col=node_id_col)
    if cached is not None:
        return cached.sort_index()

    if key_col not in graph_nodes.columns:
        raise ValueError(
            f"key_col={key_col!r} is not in graph nodes. "
            f"Available columns: {list(graph_nodes.columns)}"
        )

    node_ids = graph_nodes[node_id_col].astype(int).tolist()

    node_key_to_id: dict[str, int] = {}
    duplicate_keys: list[str] = []

    for _, row in graph_nodes.iterrows():
        key = normalize_atlas_key(row[key_col])
        node_id = int(row[node_id_col])
        if key in node_key_to_id:
            duplicate_keys.append(key)
        node_key_to_id[key] = node_id

    if duplicate_keys:
        raise ValueError(f"Duplicate graph keys for {key_col!r}. Examples: {duplicate_keys[:5]}")

    if isinstance(districting, list) and districting:
        # Common Ohio AtlasIO format: list of singleton dicts
        if all(isinstance(x, dict) and len(x) == 1 for x in districting):
            out = {}
            unmatched = []

            for item in districting:
                raw_key, raw_dist = next(iter(item.items()))
                key = normalize_atlas_key(raw_key)
                node_id = node_key_to_id.get(key)

                if node_id is None:
                    unmatched.append(key)
                    continue

                out[node_id] = coerce_district_label(raw_dist)

            if unmatched:
                raise ValueError(
                    f"{len(unmatched)} AtlasIO keys did not match graph {key_col!r}. "
                    f"Examples: {unmatched[:5]}"
                )

            if len(out) != len(graph_nodes):
                raise ValueError(
                    f"Matched {len(out)} assignments but graph has {len(graph_nodes)} nodes."
                )

            s = pd.Series(out, dtype=int)
            s.index.name = node_id_col
            return s.sort_index()

        # Fallback: node-order vector
        if len(districting) == len(graph_nodes) and not isinstance(
            districting[0], (list, tuple, dict, set)
        ):
            s = pd.Series(
                [coerce_district_label(x) for x in districting],
                index=node_ids,
                dtype=int,
            )
            s.index.name = node_id_col
            return s.sort_index()

    if isinstance(districting, dict):
        graph_node_ids = {str(x) for x in node_ids}
        out = {}
        unmatched = []

        for raw_key, raw_dist in districting.items():
            key = normalize_atlas_key(raw_key)

            if str(key) in graph_node_ids:
                node_id = int(key)
            else:
                node_id = node_key_to_id.get(key)

            if node_id is None:
                unmatched.append(key)
                continue

            out[node_id] = coerce_district_label(raw_dist)

        if unmatched:
            raise ValueError(
                f"{len(unmatched)} AtlasIO keys did not match graph nodes. "
                f"Examples: {unmatched[:5]}"
            )

        s = pd.Series(out, dtype=int)
        s.index.name = node_id_col
        return s.sort_index()

    raise ValueError(f"Unsupported AtlasIO districting format: {type(districting)}")


# ---------------------------------------------------------------------
# Standardized compliant filter
# ---------------------------------------------------------------------
def county_compliance_summary(
    assignment: pd.Series,
    graph_nodes: pd.DataFrame,
    *,
    node_id_col: str = "id",
    county_col: str = "COUNTY",
    max_districts_per_county: int = 2,
) -> dict[str, Any]:
    cached = cached_compliance(assignment)
    if cached is not None:
        return cached

    """
    Standardized county-compliance filter.

    Keep a map iff no county appears in more than
    `max_districts_per_county` districts.

    This mirrors the older compliant-ensemble script's logic:
      county -> set of districts touching that county -> reject if any
      county has too many districts.
    """
    assign_df = (
        assignment.rename("DISTRICT")
        .rename_axis(node_id_col)
        .reset_index()
    )
    assign_df[node_id_col] = pd.to_numeric(assign_df[node_id_col], errors="coerce")

    nodes = graph_nodes[[node_id_col, county_col]].copy()
    nodes[node_id_col] = pd.to_numeric(nodes[node_id_col], errors="coerce")

    assigned = nodes.merge(assign_df, on=node_id_col, how="left")

    assigned_nodes = int(assigned["DISTRICT"].notna().sum())
    missing_nodes = int(assigned["DISTRICT"].isna().sum())

    by_county = (
        assigned.dropna(subset=["DISTRICT"])
        .groupby(county_col)["DISTRICT"]
        .nunique()
    )

    split_sizes = by_county.to_numpy(dtype=int)
    worst_split = int(split_sizes.max()) if len(split_sizes) else 0
    double_cross_count = int((by_county > max_districts_per_county).sum())

    return {
        "compliant": double_cross_count == 0,
        "assigned_nodes": assigned_nodes,
        "missing_nodes": missing_nodes,
        "assigned_fraction": assigned_nodes / max(assigned_nodes + missing_nodes, 1),
        "double_cross_count": double_cross_count,
        "worst_split": worst_split,
    }


# ---------------------------------------------------------------------
# District-level municipal metrics
# ---------------------------------------------------------------------
def compute_vote_share_by_district(
    assignment: pd.Series,
    graph_nodes: pd.DataFrame,
    *,
    node_id_col: str = "id",
    dem_col: str = "G20PREDEM",
    rep_col: str = "G20PREREP",
) -> pd.Series:
    df = graph_nodes[[node_id_col, dem_col, rep_col]].merge(
        assignment.rename("DISTRICT").rename_axis(node_id_col).reset_index(),
        on=node_id_col,
        how="inner",
    )

    agg = df.groupby("DISTRICT")[[dem_col, rep_col]].sum()
    denom = agg[dem_col] + agg[rep_col]

    return (agg[dem_col] / denom.replace(0, np.nan)).fillna(0.0)


def compute_district_municipal_burden(
    assignment: pd.Series,
    nodes_df: pd.DataFrame,
    *,
    node_id_col: str,
    pop_col: str,
    municipality_cols: list[str],
    geography: MunicipalGeography | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    For each district:
      split_count[d] = number of split municipalities touching district d
      ousted_pop[d]  = total population in district d that lies outside each
                       split municipality's plurality district
    """
    if geography is not None:
        return geography.district_burden(
            assignment,
            nodes_df,
            node_id_col=node_id_col,
            pop_col=pop_col,
        )
    df = nodes_df[[node_id_col, pop_col] + municipality_cols].merge(
        assignment.rename("DISTRICT").rename_axis(node_id_col).reset_index(),
        on=node_id_col,
        how="inner",
    )

    pieces = (
        df.groupby(municipality_cols + ["DISTRICT"], as_index=False)[pop_col]
        .sum()
        .rename(columns={pop_col: "piece_pop"})
    )

    if pieces.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    anchor = (
        pieces.sort_values(
            municipality_cols + ["piece_pop", "DISTRICT"],
            ascending=[True] * len(municipality_cols) + [False, True],
        )
        .drop_duplicates(subset=municipality_cols, keep="first")
        [municipality_cols + ["DISTRICT"]]
        .rename(columns={"DISTRICT": "ANCHOR_DISTRICT"})
    )

    n_districts_by_muni = (
        pieces.groupby(municipality_cols)["DISTRICT"]
        .nunique()
        .reset_index(name="n_districts")
    )

    pieces = pieces.merge(anchor, on=municipality_cols, how="left")
    pieces = pieces.merge(n_districts_by_muni, on=municipality_cols, how="left")

    split_pieces = pieces[pieces["n_districts"] > 1].copy()

    split_count = split_pieces.groupby("DISTRICT").size().astype(float)

    split_pieces["ousted_piece_pop"] = np.where(
        split_pieces["DISTRICT"] != split_pieces["ANCHOR_DISTRICT"],
        split_pieces["piece_pop"],
        0.0,
    )

    ousted_pop = (
        split_pieces.groupby("DISTRICT")["ousted_piece_pop"]
        .sum()
        .astype(float)
    )

    return split_count, ousted_pop


def order_rank_vectors(
    vote_share: pd.Series,
    split_count: pd.Series,
    ousted_pop: pd.Series,
    *,
    n_districts: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    districts = sorted(set(vote_share.index) | set(split_count.index) | set(ousted_pop.index))

    df = pd.DataFrame(
        {
            "DISTRICT": districts,
            "dem_share": [vote_share.get(d, 0.0) for d in districts],
            "split_count": [split_count.get(d, 0.0) for d in districts],
            "ousted_pop": [ousted_pop.get(d, 0.0) for d in districts],
        }
    )

    df = df.sort_values(["dem_share", "DISTRICT"]).reset_index(drop=True)

    if len(df) != n_districts:
        raise ValueError(f"Expected {n_districts} districts, found {len(df)}.")

    df["rank"] = np.arange(1, n_districts + 1)

    return (
        df["split_count"].to_numpy(float),
        df["ousted_pop"].to_numpy(float),
        df,
    )


# ---------------------------------------------------------------------
# Ensemble construction
# ---------------------------------------------------------------------
def build_compliant_ensemble_rank_matrices(
    chain_files: list[str | Path],
    graph_nodes: pd.DataFrame,
    *,
    atlas_key_col: str,
    n_districts: int,
    node_id_col: str,
    county_col: str,
    dem_col: str,
    rep_col: str,
    pop_col: str,
    municipality_cols: list[str],
    geography: MunicipalGeography | None = None,
    max_districts_per_county: int,
    max_maps_per_chain: int | None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    split_rows = []
    ousted_rows = []
    summary_rows = []

    for chain_file in chain_files:
        chain_file = Path(chain_file)
        chain_id = chain_file.stem
        m = re.search(r"(run\d+)", chain_id)
        if m:
            chain_id = m.group(1)

        action = "Analyzing shared assignments" if configured_sweep() else "Reading chain"
        print(f"\n{action}: {chain_file}")

        used = 0
        seen = 0
        invalid = 0
        double_crosses = []
        worst_splits = []

        for record in iter_atlasio_map_records(chain_file):
            seen += 1

            if max_maps_per_chain is not None and seen > max_maps_per_chain:
                break

            try:
                assignment = atlasio_districting_to_series(
                    maybe_json_loads(record["districting"]),
                    graph_nodes,
                    key_col=atlas_key_col,
                    node_id_col=node_id_col,
                )

                comp = county_compliance_summary(
                    assignment,
                    graph_nodes,
                    node_id_col=node_id_col,
                    county_col=county_col,
                    max_districts_per_county=max_districts_per_county,
                )
                double_crosses.append(comp["double_cross_count"])
                worst_splits.append(comp["worst_split"])

                if not comp["compliant"]:
                    continue

                vote_share = compute_vote_share_by_district(
                    assignment,
                    graph_nodes,
                    node_id_col=node_id_col,
                    dem_col=dem_col,
                    rep_col=rep_col,
                )
                split_count, ousted_pop = compute_district_municipal_burden(
                    assignment,
                    graph_nodes,
                    node_id_col=node_id_col,
                    pop_col=pop_col,
                    municipality_cols=municipality_cols,
                    geography=geography,
                )
                split_vec, ousted_vec, ordered = order_rank_vectors(
                    vote_share,
                    split_count,
                    ousted_pop,
                    n_districts=n_districts,
                )

                split_rows.append(split_vec)
                ousted_rows.append(ousted_vec)
                used += 1

                for _, row in ordered.iterrows():
                    summary_rows.append(
                        {
                            "source": "ensemble",
                            "chain_id": chain_id,
                            "map_index": seen,
                            "rank": int(row["rank"]),
                            "district": int(row["DISTRICT"]),
                            "dem_share": float(row["dem_share"]),
                            "split_count": float(row["split_count"]),
                            "ousted_pop": float(row["ousted_pop"]),
                        }
                    )

            except Exception as exc:
                invalid += 1
                if invalid <= 5:
                    print(f"  FAILED map {seen}: {type(exc).__name__}: {exc}")

        summary = {
            "source": "summary",
            "chain_id": chain_id,
            "maps_seen": seen,
            "maps_retained": used,
            "invalid_maps": invalid,
            "retained_fraction": used / max(seen, 1),
            "avg_double_crosses": float(np.mean(double_crosses)) if double_crosses else np.nan,
            "max_double_crosses": int(np.max(double_crosses)) if double_crosses else 0,
            "worst_single_county_split": int(np.max(worst_splits)) if worst_splits else 0,
        }
        summary_rows.append(summary)

        print(f"  maps_seen={seen}")
        print(f"  maps_retained={used}")
        print(f"  invalid_maps={invalid}")

    if not split_rows:
        raise RuntimeError("No compliant ensemble plans were retained.")

    return np.vstack(split_rows), np.vstack(ousted_rows), pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------
# Official plan helpers
# ---------------------------------------------------------------------
def read_crosswalk(crosswalk_csv: str | Path) -> pd.DataFrame:
    cw = pd.read_csv(crosswalk_csv, dtype=str)

    block_col = resolve_column(cw.columns, ["GEOID20", "BLOCK", "Block", "GEOID"], label="crosswalk block")
    precinct_col = resolve_column(cw.columns, ["NAME", "ORIG_NAME", "PRECINCT"], label="crosswalk precinct")
    weight_col = resolve_column(cw.columns, ["POP20", "ALLOCATED_POP20", "weight", "share"], label="crosswalk weight")

    out = cw[[block_col, precinct_col, weight_col]].copy()
    out = out.rename(
        columns={block_col: "GEOID20", precinct_col: "PRECINCT_NAME", weight_col: "CW_WEIGHT"}
    )
    out["GEOID20"] = out["GEOID20"].map(normalize_block_id)
    out["PRECINCT_NAME"] = out["PRECINCT_NAME"].map(normalize_token)
    out["CW_WEIGHT"] = pd.to_numeric(out["CW_WEIGHT"], errors="coerce").fillna(0.0)

    return out


def read_spatial_or_table(path: Path):
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
    mcds = gpd.read_file(mcd_shapefile)

    def resolve_df_column(df, user_col, candidates, label):
        if user_col is not None:
            if user_col not in df.columns:
                raise ValueError(f"{label} column {user_col!r} not found. Available: {list(df.columns)}")
            return user_col
        return resolve_column(df.columns, candidates, label=label)

    mcd_county_col = resolve_df_column(
        mcds, mcd_county_col,
        ["COUNTYFP", "COUNTYFP20", "COUNTYFP10", "COUNTY", "COUNTYID"],
        "MCD county"
    )
    mcd_col = resolve_df_column(
        mcds, mcd_col,
        ["COUSUBFP", "COUSUBFP20", "COUSUBFP10", "MCD", "MCD20", "NAMELSAD", "NAME"],
        "MCD / county subdivision"
    )

    blocks = blocks.copy()
    mcds = mcds[[mcd_county_col, mcd_col, "geometry"]].copy()

    if blocks.crs is None or mcds.crs is None:
        raise ValueError("Both blocks and MCD shapefile must have CRS.")

    if blocks.crs != mcds.crs:
        mcds = mcds.to_crs(blocks.crs)

    mcds = mcds.rename(columns={mcd_county_col: "__mcd_county__", mcd_col: "__mcd_id__"})

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
        points_proj = points.to_crs(nearest_crs)
        mcds_proj = mcds.to_crs(nearest_crs)
        nearest = gpd.sjoin_nearest(
            points_proj.loc[missing_mask, ["geometry"]],
            mcds_proj[["__mcd_county__", "__mcd_id__", "geometry"]],
            how="left",
        )
        joined.loc[missing_mask, "__mcd_county__"] = nearest["__mcd_county__"].to_numpy()
        joined.loc[missing_mask, "__mcd_id__"] = nearest["__mcd_id__"].to_numpy()

    blocks[output_col] = (
        joined["__mcd_county__"].map(normalize_token) + "_" + joined["__mcd_id__"].map(normalize_token)
    )

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
) -> pd.DataFrame:
    path = Path(path)
    df = read_spatial_or_table(path)

    block_col = resolve_column(
        df.columns,
        [block_col, "GEOID20", "BLOCK", "GEOID", "GEOID10", "BLOCKID", "block_geoid", "block_id"],
        label="block id",
    )
    county_col = resolve_column(
        df.columns,
        [county_col, "COUNTY", "COUNTYFP20", "COUNTYFP", "COUNTYFP10", "COUNTY_ID", "COUNTYID"],
        label="county",
    )
    pop_col = resolve_column(
        df.columns,
        [pop_col, "POP20", "POPULATION", "POP", "P0010001", "P1_001N", "TOTALPOP", "TOTPOP"],
        label="population",
    )

    if block_mcd_col is not None:
        if block_mcd_col not in df.columns:
            raise ValueError(
                f"block_mcd_col={block_mcd_col!r} not found. Available columns: {list(df.columns)}"
            )
        mcd_col = block_mcd_col
    else:
        try:
            mcd_col = resolve_column(
                df.columns,
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
    out[block_col] = out[block_col].map(normalize_block_id)
    out[county_col] = out[county_col].map(normalize_token)
    out[mcd_col] = out[mcd_col].map(normalize_token)
    out[pop_col] = pd.to_numeric(out[pop_col], errors="coerce").fillna(0.0)

    out = out.rename(columns={block_col: "GEOID20", county_col: "COUNTY", mcd_col: "MCD", pop_col: "POP20"})
    out = out[out["GEOID20"] != ""]
    out = out.groupby(["GEOID20", "COUNTY", "MCD"], dropna=False, as_index=False)["POP20"].sum()

    return out


def read_block_assignment_excel(
    path: str | Path,
    *,
    block_col: str | None = None,
    district_col: str | None = None,
) -> pd.DataFrame:
    df = pd.read_excel(path, dtype=str)

    block_col = resolve_column(
        df.columns,
        [block_col, "BLOCK", "Block", "GEOID20", "GEOID"],
        label="block",
    )
    district_col = resolve_column(
        df.columns,
        [district_col, "DistrictID", "DistrictID:1", "DISTRICT", "District", "CD", "CONG_DIST"],
        label="district",
    )

    out = df[[block_col, district_col]].copy()
    out[block_col] = out[block_col].map(normalize_block_id)
    out[district_col] = pd.to_numeric(out[district_col], errors="coerce")
    out = out.dropna(subset=[block_col, district_col])
    out[district_col] = out[district_col].astype(int)

    out = out.rename(columns={block_col: "GEOID20", district_col: "DISTRICT"})
    out = out[out["GEOID20"] != ""]
    out = out.drop_duplicates(subset=["GEOID20"], keep="last")

    return out[["GEOID20", "DISTRICT"]]


def official_precinct_assignment_from_blocks(
    block_plan: pd.DataFrame,
    crosswalk: pd.DataFrame,
    graph_nodes: pd.DataFrame,
    *,
    graph_key_col: str = "ORIG_NAME",
    node_id_col: str = "id",
) -> pd.Series:
    merged = block_plan.merge(crosswalk, on="GEOID20", how="inner")

    if merged.empty:
        raise ValueError("No overlap between official block plan and crosswalk.")

    grouped = (
        merged.groupby(["PRECINCT_NAME", "DISTRICT"], as_index=False)["CW_WEIGHT"]
        .sum()
        .sort_values(["PRECINCT_NAME", "CW_WEIGHT", "DISTRICT"], ascending=[True, False, True])
    )

    precinct_assignment = grouped.drop_duplicates("PRECINCT_NAME", keep="first")[["PRECINCT_NAME", "DISTRICT"]].copy()

    node_lookup = graph_nodes[[node_id_col, graph_key_col]].copy()
    node_lookup[graph_key_col] = node_lookup[graph_key_col].map(normalize_token)

    matched = precinct_assignment.rename(columns={"PRECINCT_NAME": graph_key_col}).merge(
        node_lookup,
        on=graph_key_col,
        how="inner",
    )

    s = pd.Series(
        matched["DISTRICT"].astype(int).to_numpy(),
        index=matched[node_id_col].astype(int),
    )
    s.index.name = node_id_col

    return s[~s.index.duplicated(keep="first")].sort_index().astype(int)


def compute_official_rank_vectors(
    *,
    label: str,
    baf_path: str | Path,
    crosswalk: pd.DataFrame,
    block_nodes: list[dict[str, Any]] | pd.DataFrame,
    graph_nodes: pd.DataFrame,
    official_key_col: str,
    n_districts: int,
    node_id_col: str,
    dem_col: str,
    rep_col: str,
    assignment_block_col: str | None,
    assignment_district_col: str | None,
    geography: MunicipalGeography | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    block_plan = read_block_assignment_excel(
        baf_path,
        block_col=assignment_block_col,
        district_col=assignment_district_col,
    )

    precinct_assignment = official_precinct_assignment_from_blocks(
        block_plan,
        crosswalk,
        graph_nodes,
        graph_key_col=official_key_col,
        node_id_col=node_id_col,
    )

    vote_share = compute_vote_share_by_district(
        precinct_assignment,
        graph_nodes,
        node_id_col=node_id_col,
        dem_col=dem_col,
        rep_col=rep_col,
    )

    # Municipal burden directly from block-level assignment
    block_frame = pd.DataFrame(block_nodes)
    block_assignment = pd.Series(
        block_plan["DISTRICT"].to_numpy(), index=block_plan["GEOID20"].to_numpy()
    )
    if geography is not None:
        split_count, ousted_pop = geography.district_burden(
            block_assignment,
            block_frame,
            node_id_col="id",
            pop_col="POP20",
            require_complete_assignment=True,
        )
    else:
        split_count, ousted_pop = compute_district_municipal_burden(
            block_assignment,
            block_frame,
            node_id_col="id",
            pop_col="POP20",
            municipality_cols=["COUNTY", "MCD"],
        )

    split_vec, ousted_vec, ordered = order_rank_vectors(
        vote_share,
        split_count,
        ousted_pop,
        n_districts=n_districts,
    )
    ordered["source"] = label

    return split_vec, ousted_vec, ordered


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
PLAN_STYLES = {
    "Enacted ('25)": {"color": "#9467bd", "linestyle": "-", "linewidth": 2.0},
    "Overturned ('21)": {"color": "#2ca02c", "linestyle": "--", "linewidth": 2.0},
    "Remedial ('22)": {"color": "#d62728", "linestyle": "-.", "linewidth": 2.0},
}


def add_reference_marks(ax, values: dict[str, float]) -> None:
    ymin, ymax = ax.get_ylim()
    height = ymax - ymin if ymax > ymin else 1.0

    handles = []

    for idx, (label, style) in enumerate(official_plan_styles(PLAN_STYLES)):
        x = values[label]
        ax.axvline(x, zorder=10, alpha=0.95, **style)

        y = ymin + (0.72 - 0.16 * idx) * height
        ax.annotate(
            label,
            xy=(x, ymin + 0.08 * height),
            xytext=(x, y),
            ha="center",
            va="bottom",
            fontsize=8,
            arrowprops={
                "arrowstyle": "-|>",
                "color": style["color"],
                "lw": 1.0,
            },
            color=style["color"],
        )

        handles.append(Line2D([0], [0], label=label, **style))

    figure_legend(ax, handles=handles, loc="upper right", frameon=True, fontsize=8)


def plot_one_rank(
    *,
    rank_idx: int,
    ensemble_split: np.ndarray,
    ensemble_ousted: np.ndarray,
    official_split: dict[str, float],
    official_ousted: dict[str, float],
    output_path: str | Path,
    title_prefix: str,
    municipality_label: str = "county subdivision",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    if rank_idx == 1:
        rank_label = "Rank 1 (least Democratic)"
    elif rank_idx == 15:
        rank_label = "Rank 15 (most Democratic)"
    else:
        rank_label = f"Rank {rank_idx}"

    # Left: ousted population histogram
    ax = axes[0]
    x = ensemble_ousted[np.isfinite(ensemble_ousted)]
    ax.hist(
        x,
        bins=24,
        density=True,
        color="#b9c4d7",
        edgecolor="#4a5568",
        linewidth=0.8,
    )
    ax.set_title(f"Ousted {municipality_label} population")
    ax.set_xlabel("Ousted population")
    ax.set_ylabel("Density of compliant plans")
    add_reference_marks(ax, official_ousted)

    # Right: split-MCD count histogram
    ax = axes[1]
    x = ensemble_split[np.isfinite(ensemble_split)]
    xmin = int(np.floor(x.min()))
    xmax = int(np.ceil(x.max()))
    bins = np.arange(xmin - 0.5, xmax + 1.5, 1.0)
    weights = np.ones_like(x, dtype=float) / len(x)

    ax.hist(
        x,
        bins=bins,
        weights=weights,
        color="#d6b7dc",
        edgecolor="#6b3f74",
        linewidth=0.8,
    )
    ax.set_title(f"Split {municipality_label} count")
    ax.set_xlabel(f"Number of split {municipality_label}s touching district")
    ax.set_ylabel("Fraction of compliant plans")
    add_reference_marks(ax, official_split)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

    if paper_mode_enabled():
        set_panel_title(axes[0], rank_label)
    else:
        fig.suptitle(f"{title_prefix}: {rank_label}")
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build compliant district-rank municipal histograms and overlay "
            "Enacted / Overturned / Remedial official plans."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--graph-json", required=True)
    p.add_argument("--chain-files", nargs="+", required=True)
    p.add_argument("--crosswalk-csv", required=True)
    p.add_argument("--block-metadata", required=True)

    p.add_argument("--mcd-shapefile", default=None)
    p.add_argument("--mcd-shape-county-col", default=None)
    p.add_argument("--mcd-shape-mcd-col", default=None)

    p.add_argument("--enacted-baf", required=True)
    p.add_argument("--overturned-baf", required=True)
    p.add_argument("--remedial-baf", required=True)

    p.add_argument("--output-dir", required=True)

    p.add_argument("--atlas-key-col", default="NAME")
    p.add_argument("--official-key-col", default="ORIG_NAME")
    p.add_argument("--node-id-col", default="id")
    p.add_argument("--county-col", default="COUNTY")
    p.add_argument("--dem-col", default="G20PREDEM")
    p.add_argument("--rep-col", default="G20PREREP")
    p.add_argument("--pop-col", default="POP20")
    p.add_argument("--municipality-cols", nargs="+", default=["COUNTY", "MCD"])
    p.add_argument("--n-districts", type=int, default=15)
    p.add_argument("--max-districts-per-county", type=int, default=2)
    p.add_argument("--max-maps-per-chain", type=int, default=None)

    p.add_argument("--block-col", default=None)
    p.add_argument("--county-col-block", default=None)
    p.add_argument("--pop-col-block", default=None)
    p.add_argument("--block-mcd-col", default=None)

    p.add_argument("--assignment-block-col", default=None)
    p.add_argument("--assignment-district-col", default=None)

    p.add_argument("--title-prefix", default="Ohio Congressional")
    p.add_argument("--write-cache", action="store_true")
    add_municipality_arguments(p)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    geography = geography_from_args(args, graph_json=args.graph_json)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Compliant district-rank municipal histograms")
    print("=" * 80)

    print("\n[1/5] Loading graph nodes...")
    graph_nodes = read_graph_nodes(args.graph_json, geography)
    print(f"Loaded {len(graph_nodes):,} graph nodes.")

    print("\n[2/5] Building compliant ensemble rank matrices...")
    ensemble_split, ensemble_ousted, compliance_summary = build_compliant_ensemble_rank_matrices(
        args.chain_files,
        graph_nodes,
        atlas_key_col=args.atlas_key_col,
        n_districts=args.n_districts,
        node_id_col=args.node_id_col,
        county_col=args.county_col,
        dem_col=args.dem_col,
        rep_col=args.rep_col,
        pop_col=args.pop_col,
        municipality_cols=args.municipality_cols,
        geography=geography,
        max_districts_per_county=args.max_districts_per_county,
        max_maps_per_chain=args.max_maps_per_chain,
    )

    print("Ensemble matrices built:")
    print(f"  split shape : {ensemble_split.shape}")
    print(f"  ousted shape: {ensemble_ousted.shape}")

    compliance_summary_path = output_dir / "district_rank_municipal_compliance_summary.csv"
    compliance_summary.to_csv(compliance_summary_path, index=False)
    print(f"Wrote compliance summary: {compliance_summary_path}")

    print("\n[3/5] Loading crosswalk and block metadata...")
    crosswalk = read_crosswalk(args.crosswalk_csv)
    block_nodes = geography.read_block_nodes(
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

    print("\n[4/5] Computing official plan rank vectors...")
    official_split = {}
    official_ousted = {}
    official_rows = []

    plan_paths = {
        "Enacted ('25)": args.enacted_baf,
        "Overturned ('21)": args.overturned_baf,
        "Remedial ('22)": args.remedial_baf,
    }

    for label, path in plan_paths.items():
        print(f"  {label}")
        split_vec, ousted_vec, rows = compute_official_rank_vectors(
            label=label,
            baf_path=path,
            crosswalk=crosswalk,
            block_nodes=block_nodes,
            graph_nodes=graph_nodes,
            official_key_col=args.official_key_col,
            n_districts=args.n_districts,
            node_id_col=args.node_id_col,
            dem_col=args.dem_col,
            rep_col=args.rep_col,
            assignment_block_col=args.assignment_block_col,
            assignment_district_col=args.assignment_district_col,
            geography=geography,
        )
        official_split[label] = split_vec
        official_ousted[label] = ousted_vec
        official_rows.append(rows)

    if args.write_cache:
        np.savez_compressed(
            output_dir / "district_rank_municipal_matrices.npz",
            ensemble_split=ensemble_split,
            ensemble_ousted=ensemble_ousted,
        )

        with (output_dir / "district_rank_municipal_official_vectors.json").open("w") as f:
            json.dump(
                {
                    label: {
                        "split": official_split[label].tolist(),
                        "ousted": official_ousted[label].tolist(),
                    }
                    for label in official_split
                },
                f,
                indent=2,
            )

        official_df = pd.concat(official_rows, ignore_index=True)
        official_df.to_csv(output_dir / "district_rank_municipal_official_rows.csv", index=False)

    geography.write_metadata(output_dir / "municipal-geography.json")

    print("\n[5/5] Plotting 15 district-rank histograms...")
    for rank0 in range(args.n_districts):
        rank_idx = rank0 + 1

        official_split_marks = {label: float(vec[rank0]) for label, vec in official_split.items()}
        official_ousted_marks = {label: float(vec[rank0]) for label, vec in official_ousted.items()}

        output_path = output_dir / f"district_rank_{rank_idx:02d}_municipal_histograms.png"

        plot_one_rank(
            rank_idx=rank_idx,
            ensemble_split=ensemble_split[:, rank0],
            ensemble_ousted=ensemble_ousted[:, rank0],
            official_split=official_split_marks,
            official_ousted=official_ousted_marks,
            output_path=output_path,
            title_prefix=args.title_prefix,
            municipality_label=geography.display_name,
        )
        print(f"  Wrote {output_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
