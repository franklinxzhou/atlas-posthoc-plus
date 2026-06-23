#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


def normalize_geoid(s) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.zfill(15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build block-to-precinct crosswalk for Ohio proposed-plan analysis. "
            "Outputs columns GEOID20, NAME, POP20."
        )
    )

    parser.add_argument(
        "--blocks",
        required=True,
        type=Path,
        help="2020 census block polygon file, e.g. .shp or .gpkg.",
    )
    parser.add_argument(
        "--precincts",
        required=True,
        type=Path,
        help="Precinct / graph-node polygon file whose name column matches OHPct20.json.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV path, e.g. inputs/plan_comparison/oh_block_to_precinct_crosswalk.csv.",
    )

    parser.add_argument("--block-geoid-col", default="GEOID20")
    parser.add_argument("--block-pop-col", default="POP20")
    parser.add_argument("--precinct-name-col", default="NAME")

    parser.add_argument(
        "--area-crs",
        default="EPSG:26917",
        help="Projected CRS used for area allocation.",
    )

    parser.add_argument(
        "--min-weight",
        default=1e-8,
        type=float,
        help="Drop tiny sliver intersections with allocated population weight below this.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("Building Ohio block-to-precinct crosswalk")
    print("=" * 80)

    print("\nReading blocks...")
    blocks = gpd.read_file(args.blocks)

    print("Reading precincts...")
    precincts = gpd.read_file(args.precincts)

    required_block_cols = [args.block_geoid_col, args.block_pop_col]
    for col in required_block_cols:
        if col not in blocks.columns:
            raise KeyError(f"Block file is missing column {col!r}")

    if args.precinct_name_col not in precincts.columns:
        raise KeyError(f"Precinct file is missing column {args.precinct_name_col!r}")

    blocks = blocks[[args.block_geoid_col, args.block_pop_col, "geometry"]].copy()
    precincts = precincts[[args.precinct_name_col, "geometry"]].copy()

    blocks = blocks.rename(
        columns={
            args.block_geoid_col: "GEOID20",
            args.block_pop_col: "block_pop",
        }
    )
    precincts = precincts.rename(columns={args.precinct_name_col: "NAME"})

    blocks["GEOID20"] = blocks["GEOID20"].map(normalize_geoid)
    blocks["block_pop"] = pd.to_numeric(blocks["block_pop"], errors="coerce").fillna(0.0)
    precincts["NAME"] = precincts["NAME"].astype(str)

    blocks = blocks[(blocks["GEOID20"] != "") & blocks.geometry.notna()].copy()
    precincts = precincts[(precincts["NAME"] != "") & precincts.geometry.notna()].copy()

    print(f"  Blocks: {len(blocks):,}")
    print(f"  Precincts: {len(precincts):,}")

    if blocks.crs is None:
        raise ValueError("Block file has no CRS. Set/fix the CRS before running.")
    if precincts.crs is None:
        raise ValueError("Precinct file has no CRS. Set/fix the CRS before running.")

    print(f"  Block CRS: {blocks.crs}")
    print(f"  Precinct CRS: {precincts.crs}")

    print(f"\nProjecting to equal-area CRS: {args.area_crs}")
    blocks = blocks.to_crs(args.area_crs)
    precincts = precincts.to_crs(args.area_crs)

    # Clean invalid geometries lightly.
    print("\nCleaning geometries...")
    blocks["geometry"] = blocks.geometry.make_valid()
    precincts["geometry"] = precincts.geometry.make_valid()

    blocks["block_area"] = blocks.geometry.area
    blocks = blocks[blocks["block_area"] > 0].copy()

    print("\nComputing block/precinct intersections...")
    inter = gpd.overlay(
        blocks,
        precincts,
        how="intersection",
        keep_geom_type=False,
    )

    if inter.empty:
        raise RuntimeError("No block/precinct intersections found. Check CRS/geometries.")

    inter["intersection_area"] = inter.geometry.area
    inter = inter[inter["intersection_area"] > 0].copy()

    inter["area_share"] = inter["intersection_area"] / inter["block_area"]
    inter["POP20"] = inter["block_pop"] * inter["area_share"]

    # Drop numerical slivers.
    inter = inter[inter["POP20"] > args.min_weight].copy()

    out = inter[["GEOID20", "NAME", "POP20"]].copy()

    # Combine accidental duplicate block-precinct rows.
    out = (
        out.groupby(["GEOID20", "NAME"], as_index=False)["POP20"]
        .sum()
        .sort_values(["GEOID20", "NAME"])
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"\nWrote: {args.output}")
    print(f"Rows: {len(out):,}")
    print(f"Unique blocks matched: {out['GEOID20'].nunique():,}")
    print(f"Unique precincts matched: {out['NAME'].nunique():,}")
    print(f"Total allocated POP20: {out['POP20'].sum():,.2f}")

    # QA: block population reconstruction
    original_pop = blocks["block_pop"].sum()
    allocated_pop = out["POP20"].sum()

    print("\nQA:")
    print(f"  Original block POP20 total:  {original_pop:,.2f}")
    print(f"  Allocated POP20 total:       {allocated_pop:,.2f}")
    print(f"  Allocation ratio:            {allocated_pop / original_pop:.6f}")

    block_alloc = out.groupby("GEOID20", as_index=False)["POP20"].sum()
    block_check = blocks[["GEOID20", "block_pop"]].merge(
        block_alloc,
        on="GEOID20",
        how="left",
        suffixes=("_orig", "_allocated"),
    )
    block_check["POP20"] = block_check["POP20"].fillna(0.0)
    block_check["abs_diff"] = (block_check["block_pop"] - block_check["POP20"]).abs()

    print(f"  Blocks with no precinct match: {(block_check['POP20'] == 0).sum():,}")
    print(f"  Max block POP20 allocation error: {block_check['abs_diff'].max():.6f}")


if __name__ == "__main__":
    main()