#!/usr/bin/env python3
"""
Plot municipal-split tradeoff and overlay official + public citizen-submitted plans.

This extends the earlier tradeoff plot:

  x-axis: number of cut municipalities
  y-axis: ousted municipal population share or raw ousted population

Inputs:
  1. atlas-posthoc municipal_split_traces.json for the ensemble cloud,
  2. reference summary JSON for Enacted ('25), Overturned ('21), Remedial ('22),
  3. a folder of public/citizen block-assignment CSV files.

The public submissions are scanned, validated, summarized, written to a manifest,
and plotted as blue crosses.

Example
-------
atlas-posthoc-plus run municipal-tradeoff-public \
  --traces-json outputs/municipal-compliant/municipal_split_traces_compliant.json \
  --reference-json outputs/oh_municipal_split_distributions_with_refs_reference_summaries.json \
  --public-plan-dir inputs/plans/public/combined_21_25 \
  --block-metadata data/raw/tl_2024_39_tabblock20/tl_2024_39_tabblock20.shp \
  --block-col GEOID20 \
  --county-col-block COUNTYFP20 \
  --pop-col-block POP20 \
  --mcd-shapefile data/raw/tl_2024_39_cousub/tl_2024_39_cousub.shp \
  --mcd-shape-county-col COUNTYFP \
  --mcd-shape-mcd-col COUSUBFP \
  --public-manifest-out outputs/public_plan_manifest.csv \
  --public-summary-json outputs/public_plan_municipal_summaries.json \
  --output outputs/oh_municipal_split_tradeoff_with_refs_and_public.png

Notes
-----
- Public plans are expected to be block-assignment CSVs, usually with columns
  like GEOID20 and District.
- Many public files may be incomplete or malformed. This script records those
  failures in the manifest and only plots plans passing basic validation.
- The default validation requires:
    * resolvable block and district columns,
    * exactly 15 unique district labels,
    * at least 98% of statewide block population assigned.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from atlas_posthoc_plus.paper import save_figure

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
from atlas_posthoc_plus.workflows.municipal.geography import (
    MunicipalGeography,
    add_municipality_arguments,
    geography_from_args,
)


DEFAULT_CHAIN_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
]

OFFICIAL_STYLES = {
    "Enacted ('25)": {
        "marker": "X",
        "color": "#9467bd",
        "label_dx": 0.35,
        "label_dy": 0.00025,
    },
    "Overturned ('21)": {
        "marker": "s",
        "color": "#2ca02c",
        "label_dx": 0.35,
        "label_dy": -0.00035,
    },
    "Remedial ('22)": {
        "marker": "^",
        "color": "#d62728",
        "label_dx": 0.35,
        "label_dy": 0.00055,
    },
}

PUBLIC_STYLE = {
    "marker": "x",
    "color": "#0057B8",
    "s": 90,
    "linewidths": 2.0,
    "alpha": 0.80,
}


# =============================================================================
# Basic IO and normalization
# =============================================================================


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
# Reference label helpers
# =============================================================================


def require_keys(obj: dict[str, Any], keys: list[str], *, context: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise ValueError(
            f"{context} is missing required keys {missing}. "
            f"Available keys: {list(obj.keys())}"
        )


def clean_plan_label(label: str) -> str:
    aliases = {
        "Enacted ('25)": "Enacted ('25)",
        "Enacted ('25')": "Enacted ('25)",
        "Enacted": "Enacted ('25)",
        "Overturned ('21)": "Overturned ('21)",
        "Overturned ('21')": "Overturned ('21)",
        "Overturned": "Overturned ('21)",
        "Remedial ('22)": "Remedial ('22)",
        "Remedial ('22')": "Remedial ('22)",
        "Remedial": "Remedial ('22)",
    }
    return aliases.get(label, label)


def normalize_reference_summaries(raw_refs: dict[str, Any]) -> dict[str, Any]:
    refs: dict[str, Any] = {}
    for label, summary in raw_refs.items():
        refs[clean_plan_label(label)] = summary
    return refs


# =============================================================================
# Block metadata and public plan processing
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
    )

    # Give the joined frame a unique, stable row index. This makes the nearest
    # fallback robust even if GeoPandas carries over duplicate geometry indexes.
    joined = joined.reset_index(drop=True)

    missing_mask = joined["__mcd_id__"].isna()
    if bool(missing_mask.any()):
        # Some blocks touch boundaries or have tiny geometry issues. Nearest fill is
        # better than dropping blocks for a municipal-split statistic.
        #
        # GeoPandas' first sjoin creates an index_right column. If that column is
        # still present when we call sjoin_nearest, GeoPandas raises:
        #   ValueError: 'index_right' cannot be a column name in the frames being joined
        # Drop stale spatial-join index columns before the nearest fallback.
        blocks_proj = joined.loc[missing_mask].copy()

        # Drop stale columns from the first spatial join AND the empty MCD columns
        # on the left. If __mcd_county__ / __mcd_id__ remain on the left,
        # GeoPandas will suffix the right-hand columns in sjoin_nearest, and
        # nearest["__mcd_county__"] will not exist.
        stale_left_cols = [
            c for c in blocks_proj.columns
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
            c for c in mcds_proj.columns
            if c == "index_right" or c == "index_left" or c.startswith("index_right") or c.startswith("index_left")
        ]
        if stale_right_cols:
            mcds_proj = mcds_proj.drop(columns=stale_right_cols)

        # Rename right-hand columns before nearest join to avoid suffix ambiguity.
        nearest_right = mcds_proj[["__mcd_county__", "__mcd_id__", "geometry"]].rename(
            columns={
                "__mcd_county__": "__nearest_mcd_county__",
                "__mcd_id__": "__nearest_mcd_id__",
            }
        )

        # Keep a stable row id because sjoin_nearest can return multiple
        # equidistant nearest MCDs for a single block. We keep the first match.
        blocks_proj["__join_row_id__"] = blocks_proj.index.to_numpy()

        nearest = gpd.sjoin_nearest(
            blocks_proj,
            nearest_right,
            how="left",
        )

        nearest = nearest.drop_duplicates("__join_row_id__", keep="first").set_index("__join_row_id__")
        row_ids = blocks_proj["__join_row_id__"].to_numpy()

        nearest = nearest.reindex(row_ids)

        if nearest["__nearest_mcd_id__"].isna().any():
            n_missing_nearest = int(nearest["__nearest_mcd_id__"].isna().sum())
            raise ValueError(f"{n_missing_nearest} blocks could not be assigned by nearest MCD fallback.")

        joined.loc[row_ids, "__mcd_county__"] = nearest["__nearest_mcd_county__"].to_numpy()
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
    out = out.rename(columns={block_col: "id", county_col: "COUNTY", mcd_col: "MCD", pop_col: "POP20"})
    out = out.groupby(["id", "COUNTY", "MCD"], dropna=False, as_index=False)["POP20"].sum()

    total_pop = float(out["POP20"].sum())
    if total_pop <= 0:
        raise ValueError("Block metadata population sums to zero.")

    print("Block metadata:")
    print(f"  rows: {len(out):,}")
    print(f"  total population: {total_pop:,.0f}")

    return out.to_dict("records")


def infer_label_from_filename(path: Path) -> str:
    stem = path.stem
    # Expected LastName_FirstName_submissionnumber.csv.
    parts = stem.split("_")
    if len(parts) >= 3:
        last = parts[0]
        first = parts[1]
        sub = "_".join(parts[2:])
        return f"{first} {last} #{sub}"
    return stem


def read_public_assignment_csv(
    path: str | Path,
    *,
    block_col: str | None = None,
    district_col: str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    path = Path(path)

    df = pd.read_csv(path, dtype=str)
    if df.empty:
        raise ValueError("CSV is empty.")

    block_col_resolved = resolve_column(
        df.columns,
        [block_col, "GEOID20", "BLOCK", "Block", "GEOID", "block_geoid", "block_id"],
        label="public plan block id",
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
        label="public plan district",
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
    """
    Convert atlas-posthoc municipal-split summary objects across versions.

    In different versions of atlas-posthoc, compute_municipal_split_statistics_for_assignment
    may return a dataclass-like object with attributes, a dict, or an object with
    to_dict(). Do not call dict(summary) blindly.
    """
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

    # Normalize the fields needed downstream.
    d["n_cut_municipalities"] = int(d["n_cut_municipalities"])
    d["ousted_population"] = float(d["ousted_population"])
    d["ousted_population_share"] = float(d["ousted_population_share"])
    return d



def process_public_plans(
    *,
    public_plan_dir: str | Path,
    block_nodes: list[dict[str, Any]],
    expected_districts: int,
    min_assigned_pop_share: float,
    public_block_col: str | None = None,
    public_district_col: str | None = None,
    geography: MunicipalGeography | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    public_plan_dir = Path(public_plan_dir)

    if not public_plan_dir.exists():
        raise FileNotFoundError(f"Public plan directory not found: {public_plan_dir}")

    block_pop = {str(row["id"]): float(row["POP20"]) for row in block_nodes}
    total_pop = sum(block_pop.values())

    files = sorted(public_plan_dir.glob("*.csv"))

    manifest_rows: list[dict[str, Any]] = []
    good_summaries: list[dict[str, Any]] = []

    for path in files:
        label = infer_label_from_filename(path)
        base_row: dict[str, Any] = {
            "filename": path.name,
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
            assignment, meta = read_public_assignment_csv(
                path,
                block_col=public_block_col,
                district_col=public_district_col,
            )
            base_row.update(meta)

            assigned_pop = sum(block_pop.get(block_id, 0.0) for block_id in assignment)
            assigned_pop_share = assigned_pop / total_pop if total_pop > 0 else np.nan

            base_row["assigned_population"] = assigned_pop
            base_row["assigned_population_share"] = assigned_pop_share

            if meta["n_unique_districts"] != expected_districts:
                base_row["reason"] = (
                    f"expected {expected_districts} districts, "
                    f"found {meta['n_unique_districts']}"
                )
                manifest_rows.append(base_row)
                continue

            if not np.isfinite(assigned_pop_share) or assigned_pop_share < min_assigned_pop_share:
                base_row["reason"] = (
                    f"assigned population share {assigned_pop_share:.3%} "
                    f"is below threshold {min_assigned_pop_share:.3%}"
                )
                manifest_rows.append(base_row)
                continue

            summary_obj = (
                geography.summarize_assignment(assignment, block_nodes)
                if geography is not None
                else compute_municipal_split_statistics_for_assignment(
                    assignment,
                    block_nodes,
                    municipality_cols=("COUNTY", "MCD"),
                    node_id_col="id",
                    pop_col="POP20",
                    skip_missing_municipality=True,
                    require_complete_assignment=False,
                )
            )

            summary = municipal_summary_to_dict(summary_obj)
            summary["filename"] = path.name
            summary["label"] = label
            summary["district_labels"] = meta["district_labels"]
            summary["assigned_population"] = assigned_pop
            summary["assigned_population_share"] = assigned_pop_share

            base_row["status"] = "ok"
            base_row["reason"] = ""
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
        reason_counts = (
            pd.DataFrame(manifest_rows)
            .query("status != 'ok'")
            ["reason"]
            .fillna("")
            .str.replace(r"^(\\w+Error|\\w+Exception):.*$", r"\\1", regex=True)
            .value_counts()
            .head(12)
        )
        if len(reason_counts):
            print("\nTop rejection reasons:")
            for reason, count in reason_counts.items():
                print(f"  {count:4d}  {reason}")

    return manifest_rows, good_summaries


# =============================================================================
# Plotting
# =============================================================================


def get_chain_colors(traces: list[dict[str, Any]]) -> dict[str, str]:
    chain_ids = [str(item.get("chain_id", f"chain{i + 1}")) for i, item in enumerate(traces)]
    unique_chain_ids = []
    for chain_id in chain_ids:
        if chain_id not in unique_chain_ids:
            unique_chain_ids.append(chain_id)

    return {
        chain_id: DEFAULT_CHAIN_COLORS[i % len(DEFAULT_CHAIN_COLORS)]
        for i, chain_id in enumerate(unique_chain_ids)
    }


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
        xs.extend(float(x) for x in item["n_cut_municipalities"])
        ys.extend(float(y) for y in item[y_key])

    for summary in refs.values():
        xs.append(float(summary["n_cut_municipalities"]))
        ys.append(float(summary[y_key]))

    for summary in public_summaries:
        xs.append(float(summary["n_cut_municipalities"]))
        ys.append(float(summary[y_key]))

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
    pool_chains: bool = False,
    ensemble_label: str = "Sampled plans",
) -> None:
    if not isinstance(traces, list) or not traces:
        raise ValueError("traces JSON must be a nonempty list.")

    for i, item in enumerate(traces):
        require_keys(
            item,
            ["chain_id", "n_cut_municipalities", y_key],
            context=f"trace item {i}",
        )

    for label, summary in refs.items():
        require_keys(
            summary,
            ["n_cut_municipalities", y_key],
            context=f"reference plan {label}",
        )

    for i, summary in enumerate(public_summaries):
        require_keys(
            summary,
            ["label", "n_cut_municipalities", y_key],
            context=f"public plan {i}",
        )

    chain_colors = get_chain_colors(traces)

    fig, ax = plt.subplots(figsize=(12.0, 7.5))

    # Ensemble cloud.
    if pool_chains:
        x = np.concatenate(
            [np.asarray(item["n_cut_municipalities"], dtype=float) for item in traces]
        )
        y = np.concatenate([np.asarray(item[y_key], dtype=float) for item in traces])
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(
            x[mask],
            y[mask],
            s=34,
            facecolors="none",
            edgecolors=DEFAULT_CHAIN_COLORS[0],
            linewidths=1.4,
            alpha=0.38,
            label=ensemble_label,
            zorder=1,
        )
    else:
        for item in traces:
            chain_id = str(item.get("chain_id", "chain"))
            x = np.asarray(item["n_cut_municipalities"], dtype=float)
            y = np.asarray(item[y_key], dtype=float)

            mask = np.isfinite(x) & np.isfinite(y)

            ax.scatter(
                x[mask],
                y[mask],
                s=34,
                facecolors="none",
                edgecolors=chain_colors.get(chain_id, "gray"),
                linewidths=1.4,
                alpha=0.38,
                label=chain_id,
                zorder=1,
            )

    # Public citizen submissions: blue crosses.
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

    # Official labels after limits are fixed.
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
                    "lw": 1.3,
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
        # Only label public plans if requested; many labels will clutter the plot.
        for s in public_summaries:
            ax.annotate(
                str(s["label"]),
                xy=(float(s["n_cut_municipalities"]), float(s[y_key])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=PUBLIC_STYLE["color"],
                alpha=0.80,
                zorder=9,
            )

    ax.set_title(title, fontsize=22, pad=14)
    ax.set_xlabel("Number of cut municipalities", fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)

    if use_percent_axis:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Separate chain, public, and official legends.
    if pool_chains:
        chain_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor=DEFAULT_CHAIN_COLORS[0],
                markeredgewidth=1.5,
                markersize=8,
                label=ensemble_label,
                alpha=0.8,
            )
        ]
        chain_legend_title = "Ensemble"
    else:
        chain_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=1.5,
                markersize=8,
                label=chain_id,
                alpha=0.8,
            )
            for chain_id, color in chain_colors.items()
        ]
        chain_legend_title = "Chains"

    public_handles = []
    if public_summaries:
        public_handles.append(
            Line2D(
                [0],
                [0],
                marker=PUBLIC_STYLE["marker"],
                linestyle="None",
                color=PUBLIC_STYLE["color"],
                markeredgewidth=2.0,
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
        handles=chain_handles,
        title=chain_legend_title,
        loc="upper right",
        frameon=True,
        fontsize=11,
        title_fontsize=11,
    )
    ax.add_artist(legend1)

    if public_handles:
        legend2 = ax.legend(
            handles=public_handles,
            title="Public plans",
            loc="center right",
            frameon=True,
            fontsize=11,
            title_fontsize=11,
        )
        ax.add_artist(legend2)

    ax.legend(
        handles=ref_handles,
        title="Reference plans",
        loc="lower right",
        frameon=True,
        fontsize=11,
        title_fontsize=11,
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    output = save_figure(fig, output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {output}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Overlay Enacted / Overturned / Remedial plans and public citizen "
            "submissions on an atlas-posthoc municipal split tradeoff plot."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--traces-json", required=True, help="Path to atlas-posthoc municipal_split_traces.json.")
    p.add_argument("--reference-json", required=True, help="Path to reference plan municipal split summaries JSON.")
    p.add_argument("--output", required=True, help="Output PNG path.")
    p.add_argument("--title", default="Municipal split tradeoff", help="Plot title.")
    p.add_argument(
        "--y-metric",
        choices=["share", "population"],
        default="share",
        help="Use ousted population share or raw ousted population on the y-axis.",
    )
    p.add_argument(
        "--no-labels",
        action="store_true",
        help="Use large markers and legend only; do not annotate reference plans.",
    )
    p.add_argument(
        "--pool-chains",
        action="store_true",
        help=(
            "Plot all sampled plans with one style and one legend entry instead "
            "of identifying the chain that produced each point."
        ),
    )
    p.add_argument(
        "--ensemble-label",
        default="Sampled plans",
        help="Legend label for sampled plans when --pool-chains is enabled.",
    )

    # Public plan options.
    p.add_argument("--public-plan-dir", default=None, help="Folder of public/citizen block-assignment CSVs.")
    p.add_argument("--public-summary-json", default=None, help="Write/read accepted public-plan municipal summaries JSON.")
    p.add_argument("--public-manifest-out", default=None, help="Write manifest CSV with accepted/rejected public plans.")
    p.add_argument("--reuse-public-summary", action="store_true", help="Read --public-summary-json instead of recomputing public summaries.")
    p.add_argument("--label-public", action="store_true", help="Annotate citizen-submission crosses. Usually too cluttered.")

    # Public plan validation.
    p.add_argument("--expected-districts", type=int, default=15)
    p.add_argument("--min-assigned-pop-share", type=float, default=0.98)
    p.add_argument("--public-block-col", default=None)
    p.add_argument("--public-district-col", default=None)

    # Required only if computing public plan summaries.
    p.add_argument("--block-metadata", default=None)
    p.add_argument("--block-col", default=None)
    p.add_argument("--county-col-block", default=None)
    p.add_argument("--pop-col-block", default=None)
    p.add_argument("--block-mcd-col", default=None)
    p.add_argument("--mcd-shapefile", default=None)
    p.add_argument("--mcd-shape-county-col", default=None)
    p.add_argument("--mcd-shape-mcd-col", default=None)
    add_municipality_arguments(p)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    geography = geography_from_args(args)

    traces = load_json(args.traces_json)
    geography.validate_trace_metadata(traces)
    geography.assert_compatible_metadata(
        load_json(Path(args.reference_json).with_suffix(".municipality.json"))
    )
    refs = normalize_reference_summaries(load_json(args.reference_json))

    public_summaries: list[dict[str, Any]] = []

    if args.reuse_public_summary:
        if args.public_summary_json is None:
            raise ValueError("--reuse-public-summary requires --public-summary-json")
        metadata_path = Path(args.public_summary_json).with_suffix(".municipality.json")
        if not metadata_path.exists():
            raise ValueError("Public summary cache has no municipality metadata")
        geography.assert_compatible_metadata(load_json(metadata_path))
        public_summaries = load_json(args.public_summary_json)
        if not isinstance(public_summaries, list):
            raise ValueError("--public-summary-json must contain a list of public-plan summaries.")
        print(f"Loaded public summaries: {len(public_summaries):,}")

    elif args.public_plan_dir is not None:
        if args.block_metadata is None:
            raise ValueError("--public-plan-dir requires --block-metadata")
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

        manifest_rows, public_summaries = process_public_plans(
            public_plan_dir=args.public_plan_dir,
            block_nodes=block_nodes,
            expected_districts=args.expected_districts,
            min_assigned_pop_share=args.min_assigned_pop_share,
            public_block_col=args.public_block_col,
            public_district_col=args.public_district_col,
            geography=geography,
        )

        if args.public_manifest_out is not None:
            manifest_path = Path(args.public_manifest_out)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
            print(f"Wrote public manifest: {manifest_path}")

        if args.public_summary_json is not None:
            write_json(args.public_summary_json, public_summaries)
            geography.write_metadata(
                Path(args.public_summary_json).with_suffix(".municipality.json")
            )
            print(f"Wrote public summaries: {args.public_summary_json}")

    if args.y_metric == "share":
        y_key = "ousted_population_share"
        y_label = "Ousted municipal population share"
        use_percent_axis = True
    else:
        y_key = "ousted_population"
        y_label = "Ousted municipal population"
        use_percent_axis = False

    plot_tradeoff(
        traces=traces,
        refs=refs,
        public_summaries=public_summaries,
        output=args.output,
        title=args.title,
        y_key=y_key,
        y_label=y_label,
        use_percent_axis=use_percent_axis,
        label_references=not args.no_labels,
        label_public=args.label_public,
        pool_chains=args.pool_chains,
        ensemble_label=args.ensemble_label,
    )


if __name__ == "__main__":
    main()
