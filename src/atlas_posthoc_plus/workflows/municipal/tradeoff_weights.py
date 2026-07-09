#!/usr/bin/env python3
"""

Run from the root of `oh-gerrymandering`:

atlas-posthoc-plus run municipal-tradeoff-weights \
  --output outputs/oh_municipal_split_tradeoff_multi_mcd_weights_finite_run.png
"""

from __future__ import annotations

import argparse
import colorsys
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


from atlas_posthoc.observables.municipal_splits import (
    compute_municipal_split_statistics_for_assignment,
)


# =============================================================================
# Defaults for Franklin's local project layout
# =============================================================================

DEFAULT_TRACES = {
    "20": "outputs/mcd20/municipal_split_traces.json",
    "5": "outputs/mcd5/municipal_split_traces.json",
    "1": "outputs/mcd1/municipal_split_traces.json",
}

DEFAULT_OFFICIAL_PLANS = {
    "Overturned ('21)": "inputs/plans/SubSB258-Block-Assignment-File.xlsx",
    "Remedial ('22)": "inputs/plans/March 2 2022 CD BAF.xlsx",
    "Enacted ('25)": "inputs/plans/October 31 2025 CD BAF.xlsx",
}

WEIGHT_STYLES = {
    "20": {
        "label": "mcd_w=20, runs 37-40",
        "marker": "o",
        "color": "#1f77b4",
        "facecolors": "none",
        "edgecolors": "#1f77b4",
        "linewidths": 1.25,
        "s": 34,
        "alpha": 0.36,
    },
    "5": {
        "label": "mcd_w=5, runs 53-56",
        "marker": "^",
        "color": "#ff7f0e",
        "facecolors": "#ff7f0e",
        "edgecolors": "#ff7f0e",
        "linewidths": 0.65,
        "s": 34,
        "alpha": 0.32,
    },
    "1": {
        "label": "mcd_w=1, runs 57-60; not fully mixed",
        "marker": "x",
        "color": "#d62728",
        "facecolors": "none",
        "edgecolors": "#d62728",
        "linewidths": 1.15,
        "s": 40,
        "alpha": 0.46,
    },
}

COLLECTION_STYLE_CYCLE = [
    {
        "marker": "o",
        "color": "#1f77b4",
        "facecolors": "none",
        "edgecolors": "#1f77b4",
        "linewidths": 1.25,
        "s": 34,
        "alpha": 0.36,
    },
    {
        "marker": "^",
        "color": "#ff7f0e",
        "facecolors": "#ff7f0e",
        "edgecolors": "#ff7f0e",
        "linewidths": 0.65,
        "s": 34,
        "alpha": 0.32,
    },
    {
        "marker": "x",
        "color": "#d62728",
        "facecolors": "none",
        "edgecolors": "#d62728",
        "linewidths": 1.15,
        "s": 40,
        "alpha": 0.46,
    },
    {
        "marker": "s",
        "color": "#2ca02c",
        "facecolors": "none",
        "edgecolors": "#2ca02c",
        "linewidths": 1.0,
        "s": 34,
        "alpha": 0.34,
    },
    {
        "marker": "D",
        "color": "#9467bd",
        "facecolors": "none",
        "edgecolors": "#9467bd",
        "linewidths": 1.0,
        "s": 34,
        "alpha": 0.34,
    },
    {
        "marker": "P",
        "color": "#8c564b",
        "facecolors": "#8c564b",
        "edgecolors": "#8c564b",
        "linewidths": 0.65,
        "s": 34,
        "alpha": 0.32,
    },
    {
        "marker": "v",
        "color": "#e377c2",
        "facecolors": "none",
        "edgecolors": "#e377c2",
        "linewidths": 1.0,
        "s": 34,
        "alpha": 0.34,
    },
    {
        "marker": "<",
        "color": "#17becf",
        "facecolors": "none",
        "edgecolors": "#17becf",
        "linewidths": 1.0,
        "s": 34,
        "alpha": 0.34,
    },
    {
        "marker": ">",
        "color": "#bcbd22",
        "facecolors": "none",
        "edgecolors": "#bcbd22",
        "linewidths": 1.0,
        "s": 34,
        "alpha": 0.34,
    },
    {
        "marker": "h",
        "color": "#393b79",
        "facecolors": "none",
        "edgecolors": "#393b79",
        "linewidths": 1.0,
        "s": 34,
        "alpha": 0.34,
    },
    {
        "marker": "*",
        "color": "#637939",
        "facecolors": "#637939",
        "edgecolors": "#637939",
        "linewidths": 0.65,
        "s": 44,
        "alpha": 0.32,
    },
    {
        "marker": "X",
        "color": "#8c6d31",
        "facecolors": "#8c6d31",
        "edgecolors": "#8c6d31",
        "linewidths": 0.65,
        "s": 34,
        "alpha": 0.32,
    },
    {
        "marker": "o",
        "color": "#843c39",
        "facecolors": "none",
        "edgecolors": "#843c39",
        "linewidths": 1.0,
        "s": 34,
        "alpha": 0.34,
    },
    {
        "marker": "^",
        "color": "#7b4173",
        "facecolors": "none",
        "edgecolors": "#7b4173",
        "linewidths": 1.0,
        "s": 34,
        "alpha": 0.34,
    },
    {
        "marker": "s",
        "color": "#3182bd",
        "facecolors": "none",
        "edgecolors": "#3182bd",
        "linewidths": 1.0,
        "s": 34,
        "alpha": 0.34,
    },
    {
        "marker": "D",
        "color": "#e6550d",
        "facecolors": "none",
        "edgecolors": "#e6550d",
        "linewidths": 1.0,
        "s": 34,
        "alpha": 0.34,
    },
    {
        "marker": "P",
        "color": "#31a354",
        "facecolors": "#31a354",
        "edgecolors": "#31a354",
        "linewidths": 0.65,
        "s": 34,
        "alpha": 0.32,
    },
    {
        "marker": "v",
        "color": "#756bb1",
        "facecolors": "none",
        "edgecolors": "#756bb1",
        "linewidths": 1.0,
        "s": 34,
        "alpha": 0.34,
    },
    {
        "marker": "<",
        "color": "#1b9e77",
        "facecolors": "none",
        "edgecolors": "#1b9e77",
        "linewidths": 1.0,
        "s": 34,
        "alpha": 0.34,
    },
    {
        "marker": ">",
        "color": "#e7298a",
        "facecolors": "none",
        "edgecolors": "#e7298a",
        "linewidths": 1.0,
        "s": 34,
        "alpha": 0.34,
    },
]


def _generated_collection_color(style_index: int) -> str:
    hue = (0.11 + style_index * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.68, 0.78)
    return "#{:02x}{:02x}{:02x}".format(
        round(red * 255),
        round(green * 255),
        round(blue * 255),
    )


def _collection_style_with_color(style: dict[str, Any], color: str) -> dict[str, Any]:
    out = dict(style)
    out["color"] = color
    out["edgecolors"] = color
    if out.get("facecolors") != "none":
        out["facecolors"] = color
    return out


OFFICIAL_STYLES = {
    "Enacted ('25)": {
        "marker": "X",
        "color": "#6a3d9a",
        "label_dx": 0.40,
        "label_dy": 0.00025,
    },
    "Overturned ('21)": {
        "marker": "s",
        "color": "#33a02c",
        "label_dx": 0.40,
        "label_dy": -0.00035,
    },
    "Remedial ('22)": {
        "marker": "D",
        "color": "#e31a1c",
        "label_dx": 0.40,
        "label_dy": 0.00055,
    },
}

PUBLIC_STYLE = {
    # Use plus signs so public plans do not visually conflict with mcd_w=1 crosses.
    "marker": "+",
    "color": "#444444",
    "s": 76,
    "linewidths": 1.55,
    "alpha": 0.70,
}


# =============================================================================
# Basic IO and normalization
# =============================================================================

@dataclass(frozen=True)
class TraceCollection:
    collection_id: str
    label: str
    path: Path
    style_index: int


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def canonical_colname(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


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


def normalize_district(x: Any) -> str | None:
    if pd.isna(x):
        return None

    s = str(x).strip()
    if s == "":
        return None

    if re.fullmatch(r"-?\d+(\.0)?", s):
        return str(int(float(s)))

    m = re.search(r"-?\d+", s)
    if m:
        return str(int(m.group()))

    return None


def _pad_code(value: Any, width: int | None = None) -> str:
    s = normalize_token(value)
    if not s:
        return ""
    if width is not None and s.isdigit():
        return s.zfill(width)
    return s


def resolve_column(
    columns: Iterable[str],
    candidates: Iterable[str | None],
    *,
    label: str,
    required: bool = True,
) -> str | None:
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

    if required:
        raise ValueError(
            f"Could not resolve {label} column from candidates {list(candidates)}. "
            f"Available columns: {columns}"
        )
    return None


def read_spatial_or_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".shp", ".gpkg", ".geojson", ".json"}:
        import geopandas as gpd

        return gpd.read_file(path)

    if suffix == ".zip":
        import geopandas as gpd

        return gpd.read_file(f"zip://{path.resolve()}")

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str)

    return pd.read_csv(path, dtype=str)


# =============================================================================
# Trace loading
# =============================================================================


def _extract_trace_list(raw: Any, *, path: Path) -> list[dict[str, Any]]:
    """Accept both the current atlas-posthoc list format and a dict wrapper."""
    if isinstance(raw, list):
        traces = raw
    elif isinstance(raw, dict):
        for key in ("traces", "chains", "chain_traces"):
            if key in raw and isinstance(raw[key], list):
                traces = raw[key]
                break
        else:
            raise ValueError(
                f"Could not find a trace list in {path}. Available keys: {list(raw.keys())}"
            )
    else:
        raise TypeError(f"Unsupported trace JSON type in {path}: {type(raw)}")

    if not traces:
        raise ValueError(f"No traces found in {path}")
    return traces


def _collection_id(label: str) -> str:
    return canonical_colname(label) or "collection"


def parse_trace_collection_spec(spec: str, *, style_index: int) -> TraceCollection:
    if "=" not in spec:
        raise ValueError(
            f"Trace collection must be LABEL=PATH, got {spec!r}."
        )
    label, path = spec.rsplit("=", 1)
    label = label.strip()
    path = path.strip()
    if not label:
        raise ValueError(f"Trace collection label is empty in {spec!r}.")
    if not path:
        raise ValueError(f"Trace collection path is empty in {spec!r}.")
    return TraceCollection(
        collection_id=f"{style_index}:{_collection_id(label)}",
        label=label,
        path=Path(path),
        style_index=style_index,
    )


def legacy_trace_collections(args: argparse.Namespace) -> list[TraceCollection]:
    specs = [
        TraceCollection(
            collection_id="20",
            label=WEIGHT_STYLES["20"]["label"],
            path=Path(args.traces_json_mcd20),
            style_index=0,
        ),
        TraceCollection(
            collection_id="5",
            label=WEIGHT_STYLES["5"]["label"],
            path=Path(args.traces_json_mcd5),
            style_index=1,
        ),
        TraceCollection(
            collection_id="1",
            label=WEIGHT_STYLES["1"]["label"],
            path=Path(args.traces_json_mcd1),
            style_index=2,
        ),
    ]
    return [spec for spec in specs if str(spec.path)]


def trace_collections_from_args(args: argparse.Namespace) -> list[TraceCollection]:
    if args.trace_collection:
        return [
            parse_trace_collection_spec(spec, style_index=i)
            for i, spec in enumerate(args.trace_collection)
        ]
    return legacy_trace_collections(args)


def collection_style(collection: str, *, style_index: int, label: str) -> dict[str, Any]:
    if collection in WEIGHT_STYLES:
        style = dict(WEIGHT_STYLES[collection])
    elif 0 <= style_index < len(COLLECTION_STYLE_CYCLE):
        style = dict(COLLECTION_STYLE_CYCLE[style_index])
    else:
        base_style = COLLECTION_STYLE_CYCLE[style_index % len(COLLECTION_STYLE_CYCLE)]
        style = _collection_style_with_color(
            base_style,
            _generated_collection_color(style_index),
        )
    style["label"] = label
    return style


def load_traces_for_collection(spec: TraceCollection) -> list[dict[str, Any]]:
    path = spec.path
    raw = load_json(path)
    traces = _extract_trace_list(raw, path=path)

    out: list[dict[str, Any]] = []
    for i, item in enumerate(traces):
        if not isinstance(item, dict):
            raise TypeError(f"Trace item {i} in {path} is not a dict.")
        required = ["n_cut_municipalities", "ousted_population_share"]
        missing = [key for key in required if key not in item]
        if missing:
            raise ValueError(f"Trace item {i} in {path} is missing keys {missing}.")

        row = dict(item)
        row["collection"] = spec.collection_id
        row["collection_label"] = spec.label
        row["collection_style_index"] = spec.style_index
        row["mcd_weight"] = spec.collection_id
        row["trace_source"] = str(path)
        if "chain_id" not in row:
            row["chain_id"] = f"{spec.label}:chain{i + 1}"
        else:
            row["chain_id"] = str(row["chain_id"])
        out.append(row)

    return out


def load_all_traces(args: argparse.Namespace) -> list[dict[str, Any]]:
    all_traces: list[dict[str, Any]] = []
    for spec in trace_collections_from_args(args):
        loaded = load_traces_for_collection(spec)
        print(f"Loaded {spec.label}: {len(loaded)} chains from {spec.path}")
        all_traces.extend(loaded)
    return all_traces


# =============================================================================
# Block metadata and assignment summaries
# =============================================================================


def assign_blocks_to_mcds(
    blocks,
    mcd_shapefile: str | Path,
    *,
    mcd_county_col: str | None = None,
    mcd_col: str | None = None,
    output_col: str = "__assigned_mcd__",
    nearest_crs: str = "EPSG:26917",
):
    import geopandas as gpd

    mcd_shapefile = Path(mcd_shapefile)
    if not mcd_shapefile.exists():
        raise FileNotFoundError(f"MCD shapefile not found: {mcd_shapefile}")

    if not hasattr(blocks, "geometry"):
        raise ValueError(
            "Block metadata has no geometry. Provide --block-mcd-col directly "
            "or pass a block shapefile with geometry."
        )

    mcds = gpd.read_file(mcd_shapefile)
    if mcds.empty:
        raise ValueError(f"MCD shapefile is empty: {mcd_shapefile}")

    mcd_county_col = resolve_column(
        mcds.columns,
        [mcd_county_col, "COUNTYFP", "COUNTYFP20", "COUNTY", "COUNTYID"],
        label="MCD county",
    )
    mcd_col = resolve_column(
        mcds.columns,
        [mcd_col, "COUSUBFP", "COUSUBFP20", "MCD", "MCDID", "COUSUB"],
        label="MCD id",
    )

    mcds = mcds[[mcd_county_col, mcd_col, "geometry"]].copy()

    if blocks.crs is None:
        raise ValueError("Block shapefile has no CRS.")
    if mcds.crs is None:
        raise ValueError("MCD shapefile has no CRS.")

    if blocks.crs != mcds.crs:
        mcds = mcds.to_crs(blocks.crs)

    mcds = mcds.rename(columns={mcd_county_col: "__mcd_county__", mcd_col: "__mcd_id__"})

    joined = gpd.sjoin(
        blocks,
        mcds[["__mcd_county__", "__mcd_id__", "geometry"]],
        how="left",
        predicate="within",
    ).reset_index(drop=True)

    missing_mask = joined["__mcd_id__"].isna()
    if bool(missing_mask.any()):
        blocks_proj = joined.loc[missing_mask].copy()
        stale_left_cols = [
            c
            for c in blocks_proj.columns
            if (
                c in {"index_right", "index_left", "__mcd_county__", "__mcd_id__"}
                or c.startswith("index_right")
                or c.startswith("index_left")
            )
        ]
        if stale_left_cols:
            blocks_proj = blocks_proj.drop(columns=stale_left_cols)

        blocks_proj = blocks_proj.to_crs(nearest_crs)

        mcds_proj = mcds.to_crs(nearest_crs).copy()
        stale_right_cols = [
            c
            for c in mcds_proj.columns
            if c == "index_right"
            or c == "index_left"
            or c.startswith("index_right")
            or c.startswith("index_left")
        ]
        if stale_right_cols:
            mcds_proj = mcds_proj.drop(columns=stale_right_cols)

        nearest_right = mcds_proj[["__mcd_county__", "__mcd_id__", "geometry"]].rename(
            columns={
                "__mcd_county__": "__nearest_mcd_county__",
                "__mcd_id__": "__nearest_mcd_id__",
            }
        )

        blocks_proj["__join_row_id__"] = blocks_proj.index.to_numpy()
        nearest = gpd.sjoin_nearest(blocks_proj, nearest_right, how="left")
        nearest = nearest.drop_duplicates("__join_row_id__", keep="first").set_index(
            "__join_row_id__"
        )
        row_ids = blocks_proj["__join_row_id__"].to_numpy()
        nearest = nearest.reindex(row_ids)

        if nearest["__nearest_mcd_id__"].isna().any():
            n_missing_nearest = int(nearest["__nearest_mcd_id__"].isna().sum())
            raise ValueError(
                f"{n_missing_nearest} blocks could not be assigned by nearest MCD fallback."
            )

        joined.loc[row_ids, "__mcd_county__"] = nearest[
            "__nearest_mcd_county__"
        ].to_numpy()
        joined.loc[row_ids, "__mcd_id__"] = nearest["__nearest_mcd_id__"].to_numpy()

    if joined["__mcd_id__"].isna().any():
        n_missing = int(joined["__mcd_id__"].isna().sum())
        raise ValueError(f"{n_missing} blocks could not be assigned to an MCD.")

    county_key = joined["__mcd_county__"].map(lambda x: _pad_code(x, 3))
    mcd_key = joined["__mcd_id__"].map(lambda x: _pad_code(x, 5))
    joined[output_col] = county_key + "_" + mcd_key

    return joined


def read_block_metadata(
    *,
    block_metadata: str | Path,
    block_col: str | None = None,
    county_col: str | None = None,
    pop_col: str | None = None,
    block_mcd_col: str | None = None,
    mcd_shapefile: str | Path | None = None,
    mcd_shape_county_col: str | None = None,
    mcd_shape_mcd_col: str | None = None,
) -> list[dict[str, Any]]:
    df = read_spatial_or_table(block_metadata)

    block_col = resolve_column(
        df.columns,
        [block_col, "GEOID20", "BLOCK", "Block", "GEOID", "block_geoid", "block_id"],
        label="block id",
    )
    county_col = resolve_column(
        df.columns,
        [county_col, "COUNTYFP20", "COUNTYFP", "COUNTY", "county"],
        label="block county",
    )
    pop_col = resolve_column(
        df.columns,
        [pop_col, "POP20", "P0010001", "population", "POP"],
        label="block population",
    )

    if block_mcd_col is not None:
        if block_mcd_col not in df.columns:
            raise ValueError(
                f"block_mcd_col={block_mcd_col!r} not found. "
                f"Available columns: {list(df.columns)}"
            )
        mcd_col = block_mcd_col
    else:
        maybe_mcd_col = resolve_column(
            df.columns,
            ["MCD", "COUSUBFP", "COUSUBFP20", "MCDID"],
            label="block MCD",
            required=False,
        )

        if maybe_mcd_col is not None:
            mcd_col = maybe_mcd_col
        else:
            if mcd_shapefile is None:
                raise ValueError(
                    "Block metadata does not contain an obvious MCD column. "
                    "Provide --block-mcd-col or --mcd-shapefile."
                )
            df = assign_blocks_to_mcds(
                df,
                mcd_shapefile,
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
        raise ValueError("Block metadata population sums to zero.")

    print("Block metadata:")
    print(f"  rows: {len(out):,}")
    print(f"  total population: {total_pop:,.0f}")

    return out.to_dict("records")


def read_assignment_table(
    path: str | Path,
    *,
    block_col: str | None = None,
    district_col: str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    path = Path(path)
    df = read_spatial_or_table(path)
    if df.empty:
        raise ValueError(f"Assignment file is empty: {path}")

    block_col_resolved = resolve_column(
        df.columns,
        [
            block_col,
            "GEOID20",
            "BLOCKID20",
            "BLOCKID",
            "BLOCK",
            "Block",
            "GEOID",
            "block_geoid",
            "block_id",
        ],
        label="assignment block id",
    )
    district_col_resolved = resolve_column(
        df.columns,
        [
            district_col,
            "District",
            "DistrictID",
            "DISTRICT",
            "CD",
            "CONG_DIST",
            "CONGRESSIONAL",
            "assignment",
            "district_id",
            "DistrictID:1",
        ],
        label="assignment district",
    )

    out = df[[block_col_resolved, district_col_resolved]].copy()
    out[block_col_resolved] = out[block_col_resolved].map(normalize_geoid)
    out[district_col_resolved] = out[district_col_resolved].map(normalize_district)

    out = out[(out[block_col_resolved] != "") & out[district_col_resolved].notna()].copy()
    out = out.drop_duplicates(subset=[block_col_resolved], keep="last")

    assignment = dict(zip(out[block_col_resolved], out[district_col_resolved]))
    meta = {
        "block_col": block_col_resolved,
        "district_col": district_col_resolved,
        "n_rows_raw": int(len(df)),
        "n_rows_clean": int(len(out)),
        "n_unique_blocks": int(out[block_col_resolved].nunique()),
        "n_unique_districts": int(out[district_col_resolved].nunique()),
        "district_labels": sorted(str(x) for x in out[district_col_resolved].dropna().unique()),
    }
    return assignment, meta


def municipal_summary_to_dict(summary: Any) -> dict[str, Any]:
    if isinstance(summary, dict):
        d = dict(summary)
    elif hasattr(summary, "to_dict"):
        d = dict(summary.to_dict())
    elif hasattr(summary, "__dataclass_fields__"):
        from dataclasses import asdict

        d = dict(asdict(summary))
    elif all(
        hasattr(summary, key)
        for key in ["n_cut_municipalities", "ousted_population", "ousted_population_share"]
    ):
        d = {
            "n_cut_municipalities": getattr(summary, "n_cut_municipalities"),
            "ousted_population": getattr(summary, "ousted_population"),
            "ousted_population_share": getattr(summary, "ousted_population_share"),
        }
    else:
        raise TypeError(f"Unsupported municipal-split summary type: {type(summary)}")

    d["n_cut_municipalities"] = int(d["n_cut_municipalities"])
    d["ousted_population"] = float(d["ousted_population"])
    d["ousted_population_share"] = float(d["ousted_population_share"])
    return d


def summarize_assignment(
    *,
    assignment: dict[str, str],
    block_nodes: list[dict[str, Any]],
    label: str,
    filename: str,
    district_labels: list[str],
    block_pop: dict[str, float],
    total_pop: float,
    expected_districts: int,
    min_assigned_pop_share: float,
) -> dict[str, Any]:
    assigned_pop = sum(block_pop.get(block_id, 0.0) for block_id in assignment)
    assigned_pop_share = assigned_pop / total_pop if total_pop > 0 else np.nan

    if len(district_labels) != expected_districts:
        raise ValueError(f"expected {expected_districts} districts, found {len(district_labels)}")

    if not np.isfinite(assigned_pop_share) or assigned_pop_share < min_assigned_pop_share:
        raise ValueError(
            f"assigned population share {assigned_pop_share:.3%} is below threshold "
            f"{min_assigned_pop_share:.3%}"
        )

    summary_obj = compute_municipal_split_statistics_for_assignment(
        assignment,
        block_nodes,
        municipality_cols=("COUNTY", "MCD"),
        node_id_col="id",
        pop_col="POP20",
        skip_missing_municipality=True,
        require_complete_assignment=False,
    )
    summary = municipal_summary_to_dict(summary_obj)
    summary.update(
        {
            "label": label,
            "filename": filename,
            "district_labels": district_labels,
            "assigned_population": float(assigned_pop),
            "assigned_population_share": float(assigned_pop_share),
        }
    )
    return summary


def summarize_official_plans(
    *,
    official_plan_specs: dict[str, str],
    block_nodes: list[dict[str, Any]],
    expected_districts: int,
    min_assigned_pop_share: float,
    official_block_col: str | None = None,
    official_district_col: str | None = None,
) -> dict[str, dict[str, Any]]:
    block_pop = {str(row["id"]): float(row["POP20"]) for row in block_nodes}
    total_pop = sum(block_pop.values())
    summaries: dict[str, dict[str, Any]] = {}

    for label, path_str in official_plan_specs.items():
        path = Path(path_str)
        assignment, meta = read_assignment_table(
            path,
            block_col=official_block_col,
            district_col=official_district_col,
        )
        summary = summarize_assignment(
            assignment=assignment,
            block_nodes=block_nodes,
            label=label,
            filename=path.name,
            district_labels=meta["district_labels"],
            block_pop=block_pop,
            total_pop=total_pop,
            expected_districts=expected_districts,
            min_assigned_pop_share=min_assigned_pop_share,
        )
        summary.update(meta)
        summaries[label] = summary
        print(
            f"Official {label}: cuts={summary['n_cut_municipalities']}, "
            f"ousted_share={summary['ousted_population_share']:.3%}"
        )

    return summaries


def infer_label_from_filename(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 3:
        last = parts[0]
        first = parts[1]
        sub = "_".join(parts[2:])
        return f"{first} {last} #{sub}"
    return stem


def discover_public_plan_files(public_plan_dir: str | Path) -> list[Path]:
    public_plan_dir = Path(public_plan_dir)
    if not public_plan_dir.exists():
        raise FileNotFoundError(f"Public plan directory not found: {public_plan_dir}")
    exts = {".csv", ".xlsx", ".xls"}
    files = sorted(p for p in public_plan_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    return files


def process_public_plans(
    *,
    public_plan_dir: str | Path,
    block_nodes: list[dict[str, Any]],
    expected_districts: int,
    min_assigned_pop_share: float,
    public_block_col: str | None = None,
    public_district_col: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    block_pop = {str(row["id"]): float(row["POP20"]) for row in block_nodes}
    total_pop = sum(block_pop.values())
    files = discover_public_plan_files(public_plan_dir)

    manifest_rows: list[dict[str, Any]] = []
    good_summaries: list[dict[str, Any]] = []

    for path in files:
        label = infer_label_from_filename(path)
        base_row: dict[str, Any] = {
            "filename": str(path),
            "label": label,
            "status": "failed",
            "reason": "",
            "n_rows_raw": np.nan,
            "n_rows_clean": np.nan,
            "n_unique_blocks": np.nan,
            "n_unique_districts": np.nan,
            "assigned_population": np.nan,
            "assigned_population_share": np.nan,
            "n_cut_municipalities": np.nan,
            "ousted_population": np.nan,
            "ousted_population_share": np.nan,
        }

        try:
            assignment, meta = read_assignment_table(
                path,
                block_col=public_block_col,
                district_col=public_district_col,
            )
            base_row.update(meta)

            summary = summarize_assignment(
                assignment=assignment,
                block_nodes=block_nodes,
                label=label,
                filename=str(path),
                district_labels=meta["district_labels"],
                block_pop=block_pop,
                total_pop=total_pop,
                expected_districts=expected_districts,
                min_assigned_pop_share=min_assigned_pop_share,
            )
            summary.update(meta)

            base_row["status"] = "ok"
            base_row["reason"] = ""
            base_row["assigned_population"] = summary["assigned_population"]
            base_row["assigned_population_share"] = summary["assigned_population_share"]
            base_row["n_cut_municipalities"] = summary["n_cut_municipalities"]
            base_row["ousted_population"] = summary["ousted_population"]
            base_row["ousted_population_share"] = summary["ousted_population_share"]

            good_summaries.append(summary)
            manifest_rows.append(base_row)

        except Exception as exc:
            base_row["reason"] = f"{type(exc).__name__}: {exc}"
            manifest_rows.append(base_row)

    print(f"Public plans scanned: {len(files):,}")
    print(f"Public plans accepted: {len(good_summaries):,}")
    print(f"Public plans rejected: {len(files) - len(good_summaries):,}")

    if manifest_rows:
        manifest_df = pd.DataFrame(manifest_rows)
        rejected = manifest_df.loc[manifest_df["status"] != "ok", "reason"].fillna("")
        reason_counts = rejected.str.replace(
            r"^(\w+Error|\w+Exception):.*$", r"\1", regex=True
        ).value_counts().head(12)
        if len(reason_counts):
            print("\nTop rejection reasons:")
            for reason, count in reason_counts.items():
                print(f"  {count:4d}  {reason}")

    return manifest_rows, good_summaries


# =============================================================================
# Plotting
# =============================================================================


def finite_arrays(item: dict[str, Any], y_key: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(item["n_cut_municipalities"], dtype=float)
    y = np.asarray(item[y_key], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def maybe_thin(x: np.ndarray, y: np.ndarray, *, max_points: int | None) -> tuple[np.ndarray, np.ndarray]:
    if max_points is None or max_points <= 0 or len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, num=max_points, dtype=int)
    return x[idx], y[idx]


def compute_axis_limits(
    traces: list[dict[str, Any]],
    refs: dict[str, Any],
    public_summaries: list[dict[str, Any]],
    *,
    y_key: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    xs: list[float] = []
    ys: list[float] = []

    for item in traces:
        x, y = finite_arrays(item, y_key)
        xs.extend(float(v) for v in x)
        ys.extend(float(v) for v in y)

    for summary in refs.values():
        xs.append(float(summary["n_cut_municipalities"]))
        ys.append(float(summary[y_key]))

    for summary in public_summaries:
        xs.append(float(summary["n_cut_municipalities"]))
        ys.append(float(summary[y_key]))

    if not xs or not ys:
        raise ValueError("No finite points are available to set axis limits.")

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_pad = max(1.0, 0.08 * (x_max - x_min))
    y_pad = max(0.0005, 0.12 * (y_max - y_min))

    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def plot_tradeoff(
    *,
    traces: list[dict[str, Any]],
    refs: dict[str, Any],
    public_summaries: list[dict[str, Any]],
    output: str | Path,
    title: str,
    y_key: str,
    y_label: str,
    use_percent_axis: bool,
    label_references: bool,
    label_public: bool,
    max_points_per_chain: int | None,
    warning_text: str | None,
) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 8.0))

    plotted_collections: set[str] = set()
    collection_order: list[str] = []
    collection_styles: dict[str, dict[str, Any]] = {}

    # Ensemble cloud, drawn chain-by-chain but labeled by collection.
    for item in traces:
        collection = str(item.get("collection", item.get("mcd_weight", "collection")))
        collection_label = str(item.get("collection_label", collection))
        style_index = int(item.get("collection_style_index", 0))
        style = collection_style(
            collection,
            style_index=style_index,
            label=collection_label,
        )
        collection_styles.setdefault(collection, style)
        x, y = finite_arrays(item, y_key)
        x, y = maybe_thin(x, y, max_points=max_points_per_chain)

        if collection not in plotted_collections:
            collection_order.append(collection)
            label = style["label"]
        else:
            label = None
        plotted_collections.add(collection)

        scatter_kwargs = {
            "x": x,
            "y": y,
            "s": style["s"],
            "marker": style["marker"],
            "linewidths": style["linewidths"],
            "alpha": style["alpha"],
            "label": label,
            "zorder": 2,
        }
        if style["marker"] in {"x", "+"}:
            scatter_kwargs["color"] = style["color"]
        else:
            scatter_kwargs["facecolors"] = style["facecolors"]
            scatter_kwargs["edgecolors"] = style["edgecolors"]
        ax.scatter(**scatter_kwargs)

    # Public citizen submissions.
    if public_summaries:
        px = np.asarray([float(s["n_cut_municipalities"]) for s in public_summaries], dtype=float)
        py = np.asarray([float(s[y_key]) for s in public_summaries], dtype=float)
        pmask = np.isfinite(px) & np.isfinite(py)
        ax.scatter(
            px[pmask],
            py[pmask],
            s=PUBLIC_STYLE["s"],
            marker=PUBLIC_STYLE["marker"],
            color=PUBLIC_STYLE["color"],
            linewidths=PUBLIC_STYLE["linewidths"],
            alpha=PUBLIC_STYLE["alpha"],
            zorder=8,
            label="Citizen submissions",
        )

    # Official plan points.
    for label, style in OFFICIAL_STYLES.items():
        if label not in refs:
            continue
        x = float(refs[label]["n_cut_municipalities"])
        y = float(refs[label][y_key])
        ax.scatter(
            [x],
            [y],
            s=260,
            marker=style["marker"],
            color=style["color"],
            edgecolors="black",
            linewidths=1.4,
            zorder=10,
        )

    xlim, ylim = compute_axis_limits(traces, refs, public_summaries, y_key=y_key)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    if label_references:
        for label, style in OFFICIAL_STYLES.items():
            if label not in refs:
                continue
            x = float(refs[label]["n_cut_municipalities"])
            y = float(refs[label][y_key])
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(x + style["label_dx"], y + style["label_dy"]),
                fontsize=12,
                fontweight="bold",
                color=style["color"],
                ha="left",
                va="center",
                arrowprops={
                    "arrowstyle": "->",
                    "lw": 1.2,
                    "color": style["color"],
                    "shrinkA": 5,
                    "shrinkB": 5,
                },
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "fc": "white",
                    "ec": style["color"],
                    "alpha": 0.92,
                },
                zorder=20,
            )

    if label_public and public_summaries:
        for s in public_summaries:
            ax.annotate(
                str(s["label"]),
                xy=(float(s["n_cut_municipalities"]), float(s[y_key])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=PUBLIC_STYLE["color"],
                alpha=0.75,
                zorder=9,
            )

    ax.set_title(title, fontsize=21, pad=14)
    ax.set_xlabel("Number of cut MCDs", fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)

    if use_percent_axis:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    if warning_text:
        ax.text(
            0.02,
            0.98,
            warning_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#d62728", "alpha": 0.95},
            zorder=30,
        )

    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    collection_handles = []
    for collection in collection_order:
        style = collection_styles[collection]
        if style["marker"] in {"x", "+"}:
            handle = Line2D(
                [0],
                [0],
                marker=style["marker"],
                linestyle="None",
                color=style["color"],
                markeredgewidth=style["linewidths"],
                markersize=8,
                label=style["label"],
                alpha=0.95,
            )
        else:
            handle = Line2D(
                [0],
                [0],
                marker=style["marker"],
                linestyle="None",
                markerfacecolor=style["facecolors"],
                markeredgecolor=style["edgecolors"],
                markeredgewidth=style["linewidths"],
                markersize=8,
                label=style["label"],
                alpha=0.95,
            )
        collection_handles.append(handle)

    public_handles = []
    if public_summaries:
        public_handles.append(
            Line2D(
                [0],
                [0],
                marker=PUBLIC_STYLE["marker"],
                linestyle="None",
                color=PUBLIC_STYLE["color"],
                markeredgewidth=PUBLIC_STYLE["linewidths"],
                markersize=9,
                label=f"Citizen submissions ({len(public_summaries)})",
            )
        )

    ref_handles = [
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            linestyle="None",
            markerfacecolor=style["color"],
            markeredgecolor="black",
            markeredgewidth=1.2,
            markersize=10,
            label=label,
        )
        for label, style in OFFICIAL_STYLES.items()
        if label in refs
    ]

    legend1 = ax.legend(
        handles=collection_handles,
        title="Plan collections",
        loc="upper right",
        frameon=True,
        fontsize=10.5,
        title_fontsize=11,
    )
    ax.add_artist(legend1)

    if public_handles:
        legend2 = ax.legend(
            handles=public_handles,
            title="Public plans",
            loc="center right",
            frameon=True,
            fontsize=10.5,
            title_fontsize=11,
        )
        ax.add_artist(legend2)

    ax.legend(
        handles=ref_handles,
        title="Official plans",
        loc="lower right",
        frameon=True,
        fontsize=10.5,
        title_fontsize=11,
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


# =============================================================================
# CLI
# =============================================================================


def parse_official_plan_args(args: argparse.Namespace) -> dict[str, str]:
    specs = dict(DEFAULT_OFFICIAL_PLANS)
    if args.official_overturned is not None:
        specs["Overturned ('21)"] = args.official_overturned
    if args.official_remedial is not None:
        specs["Remedial ('22)"] = args.official_remedial
    if args.official_enacted is not None:
        specs["Enacted ('25)"] = args.official_enacted
    return specs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Make a finite-run municipal-split tradeoff plot comparing arbitrary "
            "atlas-posthoc municipal_split_traces.json collections, with official "
            "and public plans overlaid."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--trace-collection",
        dest="trace_collection",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Collection to overlay, where PATH is an atlas-posthoc "
            "diagnostics/municipal_split_traces.json file. May be repeated. "
            "If omitted, the legacy --traces-json-mcd20/5/1 inputs are used."
        ),
    )
    p.add_argument(
        "--traces-json",
        dest="trace_collection",
        action="append",
        metavar="LABEL=PATH",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--traces-json-mcd20", default=DEFAULT_TRACES["20"])
    p.add_argument("--traces-json-mcd5", default=DEFAULT_TRACES["5"])
    p.add_argument("--traces-json-mcd1", default=DEFAULT_TRACES["1"])

    p.add_argument(
        "--output",
        default="outputs/oh_municipal_split_tradeoff_multi_mcd_weights_finite_run.png",
        help="Output PNG path.",
    )
    p.add_argument(
        "--title",
        default="Ohio congressional municipal-split tradeoff, finite-run comparison",
        help="Plot title.",
    )
    p.add_argument(
        "--y-metric",
        choices=["share", "population"],
        default="share",
        help="Use ousted MCD population share or raw ousted MCD population on the y-axis.",
    )
    p.add_argument(
        "--max-points-per-chain",
        type=int,
        default=None,
        help="Optional deterministic thinning for plotting only. Use all saved maps by default.",
    )
    p.add_argument(
        "--no-labels",
        action="store_true",
        help="Use large official-plan markers and legend only; do not annotate official plans.",
    )
    p.add_argument(
        "--warning-text",
        default=None,
        help="Optional text box to place in the upper-left of the plot.",
    )

    # Official plan options.
    p.add_argument("--official-overturned", default=None)
    p.add_argument("--official-remedial", default=None)
    p.add_argument("--official-enacted", default=None)
    p.add_argument(
        "--official-summary-json",
        default="outputs/official_plan_municipal_summaries.json",
        help="Write/read official municipal split summaries JSON.",
    )
    p.add_argument(
        "--reuse-official-summary",
        action="store_true",
        help="Read --official-summary-json instead of recomputing official summaries.",
    )
    p.add_argument("--official-block-col", default=None)
    p.add_argument("--official-district-col", default=None)
    p.add_argument(
        "--skip-official-plans",
        action="store_true",
        help="Do not overlay enacted/overturned/remedial official plans.",
    )

    # Public plan options.
    p.add_argument("--public-plan-dir", default=None)
    p.add_argument(
        "--public-summary-json",
        default="outputs/public_plan_municipal_summaries.json",
        help="Write/read accepted public-plan municipal summaries JSON.",
    )
    p.add_argument(
        "--public-manifest-out",
        default="outputs/public_plan_manifest.csv",
        help="Write manifest CSV with accepted/rejected public plans.",
    )
    p.add_argument(
        "--reuse-public-summary",
        action="store_true",
        help="Read --public-summary-json instead of recomputing public summaries.",
    )
    p.add_argument(
        "--label-public",
        action="store_true",
        help="Annotate citizen-submission points. Usually too cluttered for 105 plans.",
    )
    p.add_argument("--public-block-col", default=None)
    p.add_argument("--public-district-col", default=None)

    # Plan validation.
    p.add_argument("--expected-districts", type=int, default=15)
    p.add_argument("--min-assigned-pop-share", type=float, default=0.98)

    # Block metadata needed for official/public summaries.
    p.add_argument(
        "--block-metadata",
        default="data/raw/tl_2024_39_tabblock20/tl_2024_39_tabblock20.shp",
    )
    p.add_argument("--block-col", default="GEOID20")
    p.add_argument("--county-col-block", default="COUNTYFP20")
    p.add_argument("--pop-col-block", default="POP20")
    p.add_argument("--block-mcd-col", default=None)
    p.add_argument(
        "--mcd-shapefile",
        default="data/raw/tl_2024_39_cousub/tl_2024_39_cousub.shp",
    )
    p.add_argument("--mcd-shape-county-col", default="COUNTYFP")
    p.add_argument("--mcd-shape-mcd-col", default="COUSUBFP")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    traces = load_all_traces(args)

    if args.y_metric == "share":
        y_key = "ousted_population_share"
        y_label = "Ousted MCD population share"
        use_percent_axis = True
    else:
        y_key = "ousted_population"
        y_label = "Ousted MCD population"
        use_percent_axis = False

    official_summaries: dict[str, Any] = {}
    public_summaries: list[dict[str, Any]] = []

    official_summary_path = Path(args.official_summary_json)
    public_summary_path = Path(args.public_summary_json)

    reuse_official = (
        args.reuse_official_summary
        and not args.skip_official_plans
        and official_summary_path.exists()
    )
    compute_official = not args.skip_official_plans and not reuse_official

    if (
        args.reuse_official_summary
        and not args.skip_official_plans
        and not official_summary_path.exists()
    ):
        print(
            f"Official summary cache not found: {official_summary_path}; "
            "recomputing it."
        )

    reuse_public = args.reuse_public_summary and public_summary_path.exists()
    compute_public = bool(args.public_plan_dir) and not reuse_public

    if args.reuse_public_summary and not public_summary_path.exists():
        if args.public_plan_dir:
            print(
                f"Public summary cache not found: {public_summary_path}; "
                "recomputing it."
            )
        else:
            print(
                f"Public summary cache not found: {public_summary_path}; "
                "no --public-plan-dir was provided, so public plans are skipped."
            )

    needs_block_nodes = compute_official or compute_public
    block_nodes: list[dict[str, Any]] | None = None
    if needs_block_nodes:
        block_nodes = read_block_metadata(
            block_metadata=args.block_metadata,
            block_col=args.block_col,
            county_col=args.county_col_block,
            pop_col=args.pop_col_block,
            block_mcd_col=args.block_mcd_col,
            mcd_shapefile=args.mcd_shapefile,
            mcd_shape_county_col=args.mcd_shape_county_col,
            mcd_shape_mcd_col=args.mcd_shape_mcd_col,
        )

    if reuse_official:
        official_summaries = load_json(official_summary_path)
        if not isinstance(official_summaries, dict):
            raise ValueError("--official-summary-json must contain a dict keyed by official plan label.")
        print(f"Loaded official summaries: {len(official_summaries):,}")
    elif compute_official:
        assert block_nodes is not None
        official_summaries = summarize_official_plans(
            official_plan_specs=parse_official_plan_args(args),
            block_nodes=block_nodes,
            expected_districts=args.expected_districts,
            min_assigned_pop_share=args.min_assigned_pop_share,
            official_block_col=args.official_block_col,
            official_district_col=args.official_district_col,
        )
        write_json(official_summary_path, official_summaries)
        print(f"Wrote official summaries: {official_summary_path}")

    if reuse_public:
        public_summaries = load_json(public_summary_path)
        if not isinstance(public_summaries, list):
            raise ValueError("--public-summary-json must contain a list of public-plan summaries.")
        print(f"Loaded public summaries: {len(public_summaries):,}")
    elif compute_public:
        assert block_nodes is not None
        manifest_rows, public_summaries = process_public_plans(
            public_plan_dir=args.public_plan_dir,
            block_nodes=block_nodes,
            expected_districts=args.expected_districts,
            min_assigned_pop_share=args.min_assigned_pop_share,
            public_block_col=args.public_block_col,
            public_district_col=args.public_district_col,
        )

        if args.public_manifest_out:
            manifest_path = Path(args.public_manifest_out)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
            print(f"Wrote public manifest: {manifest_path}")

        if args.public_summary_json:
            write_json(public_summary_path, public_summaries)
            print(f"Wrote public summaries: {public_summary_path}")

    plot_tradeoff(
        traces=traces,
        refs=official_summaries,
        public_summaries=public_summaries,
        output=args.output,
        title=args.title,
        y_key=y_key,
        y_label=y_label,
        use_percent_axis=use_percent_axis,
        label_references=not args.no_labels,
        label_public=args.label_public,
        max_points_per_chain=args.max_points_per_chain,
        warning_text=args.warning_text,
    )


if __name__ == "__main__":
    main()
