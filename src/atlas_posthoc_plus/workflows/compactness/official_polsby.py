#!/usr/bin/env python3
"""
RUN THIS BEFORE plot_congressional_polsby_popper_traces_from_refcsv.py - this calculates PP in blocks
atlas-posthoc-plus run official-polsby \
  --block-shapefile data/raw/tl_2024_39_tabblock20/tl_2024_39_tabblock20.shp \
  --enacted-baf "inputs/plans/October 31 2025 CD BAF.xlsx" \
  --overturned-baf "inputs/plans/SubSB258-Block-Assignment-File.xlsx" \
  --remedial-baf "inputs/plans/March 2 2022 CD BAF.xlsx" \
  --summary-output outputs/polsby_popper_traces_37_40/official_plan_polsby_popper_from_blocks_summary.csv \
  --district-output outputs/polsby_popper_traces_37_40/official_plan_polsby_popper_from_blocks_by_district.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd

from atlas_posthoc.observables.polsby_popper import (
    polsby_popper_from_area_perimeter,
    summarize_polsby_popper_scores,
)


PLAN_FILES = [
    ("Enacted ('25)", "enacted_baf"),
    ("Overturned ('21)", "overturned_baf"),
    ("Remedial ('22)", "remedial_baf"),
]


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


def normalize_district(value: Any) -> int | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s == "":
        return None
    if re.fullmatch(r"-?\d+(\.0)?", s):
        return int(float(s))
    m = re.search(r"-?\d+", s)
    if m:
        return int(m.group())
    return None


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


def read_assignment_file(
    path: str | Path,
    *,
    block_col: str | None,
    district_col: str | None,
) -> pd.DataFrame:
    path = Path(path)

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str)

    if df.empty:
        raise ValueError(f"Assignment file is empty: {path}")

    block_col = resolve_column(
        df.columns,
        [block_col, "GEOID20", "BLOCK", "Block", "GEOID", "block_geoid", "block_id"],
        label="assignment block id",
    )
    district_col = resolve_column(
        df.columns,
        [
            district_col,
            "DistrictID",
            "DistrictID:1",
            "DISTRICT",
            "District",
            "CD",
            "CONG_DIST",
            "CONGRESSIONAL",
            "assignment",
            "district_id",
        ],
        label="assignment district",
    )

    out = df[[block_col, district_col]].copy()
    out.columns = ["__block_id__", "__district__"]
    out["__block_id__"] = out["__block_id__"].map(normalize_block_id)
    out["__district__"] = out["__district__"].map(normalize_district)
    out = out[(out["__block_id__"] != "") & out["__district__"].notna()].copy()
    out["__district__"] = out["__district__"].astype(int)

    # If duplicated, keep the final record. BAFs should usually not duplicate.
    out = out.drop_duplicates(subset=["__block_id__"], keep="last")
    return out


def prepare_blocks(
    *,
    block_shapefile: str | Path,
    block_col: str | None,
    projected_crs: str,
) -> gpd.GeoDataFrame:
    print(f"Reading block geometry: {block_shapefile}")
    blocks = gpd.read_file(block_shapefile)

    block_col = resolve_column(
        blocks.columns,
        [block_col, "GEOID20", "GEOID", "BLOCKID20"],
        label="block shapefile GEOID",
    )

    blocks = blocks[[block_col, "geometry"]].copy()
    blocks.columns = ["__block_id__", "geometry"]
    blocks["__block_id__"] = blocks["__block_id__"].map(normalize_block_id)
    blocks = blocks[blocks["__block_id__"] != ""].copy()
    blocks = blocks.drop_duplicates(subset=["__block_id__"], keep="first")

    if blocks.crs is None:
        raise ValueError("Block shapefile has no CRS. Set it before running this script.")

    print(f"  blocks: {len(blocks):,}")
    print(f"  source CRS: {blocks.crs}")
    print(f"  projecting to: {projected_crs}")

    return blocks.to_crs(projected_crs)


def compute_plan_from_blocks(
    *,
    blocks: gpd.GeoDataFrame,
    assignment_path: str | Path,
    plan_label: str,
    assignment_block_col: str | None,
    assignment_district_col: str | None,
    expected_districts: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    print(f"\nComputing block-level Polsby-Popper for {plan_label}:")
    print(f"  assignment: {assignment_path}")

    assign = read_assignment_file(
        assignment_path,
        block_col=assignment_block_col,
        district_col=assignment_district_col,
    )

    merged = blocks.merge(assign, on="__block_id__", how="inner", validate="one_to_one")

    assigned_share = len(merged) / len(blocks) if len(blocks) else np.nan
    districts = sorted(merged["__district__"].dropna().unique().astype(int).tolist())

    print(f"  assignment rows: {len(assign):,}")
    print(f"  matched block geometries: {len(merged):,}/{len(blocks):,} ({assigned_share:.2%})")
    print(f"  districts found: {districts}")

    if len(districts) != expected_districts:
        raise ValueError(
            f"{plan_label} expected {expected_districts} districts, found {len(districts)}."
        )

    # Dissolve all blocks assigned to each district. This removes internal block
    # boundaries and leaves district boundary geometry.
    dissolved = merged.dissolve(by="__district__", as_index=False)

    dissolved["area"] = dissolved.geometry.area.astype(float)
    dissolved["perimeter"] = dissolved.geometry.length.astype(float)
    dissolved["polsby_popper"] = polsby_popper_from_area_perimeter(
        dissolved["area"].to_numpy(),
        dissolved["perimeter"].to_numpy(),
        invalid_policy="raise",
        clip=False,
    )

    district_rows = dissolved[["__district__", "area", "perimeter", "polsby_popper"]].copy()
    district_rows = district_rows.rename(columns={"__district__": "district"})
    district_rows.insert(0, "plan", plan_label)

    summary_obj = summarize_polsby_popper_scores(district_rows["polsby_popper"])
    summary = summary_obj.to_dict()
    summary["plan"] = plan_label
    summary["n_districts"] = int(len(district_rows))
    summary["matched_blocks"] = int(len(merged))
    summary["total_blocks_in_shapefile"] = int(len(blocks))
    summary["matched_block_share"] = float(assigned_share)

    for row in district_rows.itertuples(index=False):
        summary[f"district_{int(row.district)}_pp"] = float(row.polsby_popper)

    print(
        f"  average PP={summary['average']:.6f}, "
        f"min PP={summary['minimum']:.6f}, max PP={summary['maximum']:.6f}"
    )

    return summary, district_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute official congressional Polsby-Popper from block geometries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--block-shapefile", required=True)
    p.add_argument("--block-col", default=None)
    p.add_argument(
        "--projected-crs",
        default="EPSG:5070",
        help="Projected CRS used for area/perimeter. EPSG:5070 is CONUS Albers.",
    )

    p.add_argument("--enacted-baf", required=True)
    p.add_argument("--overturned-baf", required=True)
    p.add_argument("--remedial-baf", required=True)
    p.add_argument("--assignment-block-col", default=None)
    p.add_argument("--assignment-district-col", default=None)
    p.add_argument("--expected-districts", type=int, default=15)

    p.add_argument(
        "--summary-output",
        default="outputs/polsby_popper_traces_37_40/official_plan_polsby_popper_from_blocks_summary.csv",
    )
    p.add_argument(
        "--district-output",
        default="outputs/polsby_popper_traces_37_40/official_plan_polsby_popper_from_blocks_by_district.csv",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    blocks = prepare_blocks(
        block_shapefile=args.block_shapefile,
        block_col=args.block_col,
        projected_crs=args.projected_crs,
    )

    summaries: list[dict[str, Any]] = []
    district_tables: list[pd.DataFrame] = []

    for label, attr in PLAN_FILES:
        summary, district_rows = compute_plan_from_blocks(
            blocks=blocks,
            assignment_path=getattr(args, attr),
            plan_label=label,
            assignment_block_col=args.assignment_block_col,
            assignment_district_col=args.assignment_district_col,
            expected_districts=args.expected_districts,
        )
        summaries.append(summary)
        district_tables.append(district_rows)

    summary_df = pd.DataFrame(summaries)
    first_cols = [
        "plan",
        "average",
        "minimum",
        "maximum",
        "n_districts",
        "n_scores",
        "n_finite",
        "matched_blocks",
        "total_blocks_in_shapefile",
        "matched_block_share",
    ]
    summary_df = summary_df[first_cols + [c for c in summary_df.columns if c not in first_cols]]

    district_df = pd.concat(district_tables, ignore_index=True)

    summary_output = Path(args.summary_output)
    district_output = Path(args.district_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    district_output.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(summary_output, index=False)
    district_df.to_csv(district_output, index=False)

    print("\nWrote:")
    print(f"  {summary_output}")
    print(f"  {district_output}")

    print("\nSummary:")
    print(summary_df[["plan", "average", "minimum", "maximum", "matched_block_share"]].to_string(index=False))


if __name__ == "__main__":
    main()
