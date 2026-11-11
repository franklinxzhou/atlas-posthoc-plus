#!/usr/bin/env python3
"""
About This Script
  1. loads atlas-posthoc municipal_split_traces.json,
  2. pools the sampled values across chains,
  3. computes municipal-split summaries for Enacted ('25), Overturned ('21),
     and Remedial ('22) from block-assignment spreadsheets,
  4. overlays those plans as vertical reference lines on two histograms:
       - number of cut municipalities
       - ousted municipal population score (share or raw population)

Call this under oh-gerrymandering root (abs. pathnames used for files not under oh-gerrymandering directory):
### install atlas-posthoc first ###

atlas-posthoc-plus run municipal-histograms \
  --traces-json outputs/municipal-compliant/municipal_split_traces_compliant.json \
  --block-metadata data/raw/tl_2024_39_tabblock20/tl_2024_39_tabblock20.shp \
  --block-col GEOID20 \
  --county-col COUNTYFP20 \
  --pop-col POP20 \
  --mcd-shapefile data/raw/tl_2024_39_cousub/tl_2024_39_cousub.shp \
  --mcd-shape-county-col COUNTYFP \
  --mcd-shape-mcd-col COUSUBFP \
  --enacted "inputs/plans/October 31 2025 CD BAF.xlsx" \
  --overturned "inputs/plans/SubSB258-Block-Assignment-File.xlsx" \
  --remedial "inputs/plans/March 2 2022 CD BAF.xlsx" \
  --output outputs/oh_municipal_split_histograms_with_refs.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from atlas_posthoc_plus.paper import save_figure

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
from atlas_posthoc_plus.workflows.municipal.geography import (
    MunicipalGeography,
    add_municipality_arguments,
    geography_from_args,
    municipality_name_for_traces,
)


DEFAULT_TRACES_JSON = Path("/mnt/data/municipal_split_traces(1).json")
DEFAULT_OUTPUT = Path("outputs/figures/oh_municipal_split_histograms_with_refs.png")
DEFAULT_ENACTED = Path(os.environ.get("ATLAS_POSTHOC_ENACTED_BAF", "/mnt/data/October 31 2025 CD BAF.xlsx"))
DEFAULT_OVERTURNED = Path(os.environ.get("ATLAS_POSTHOC_OVERTURNED_BAF", "/mnt/data/SubSB258-Block-Assignment-File.xlsx"))
DEFAULT_REMEDIAL = Path(os.environ.get("ATLAS_POSTHOC_REMEDIAL_BAF", "/mnt/data/March 2 2022 CD BAF.xlsx"))


def normalize_token(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def normalize_block_id(x: Any) -> str:
    s = normalize_token(x)
    return s.zfill(15) if s else ""


def _pad_code(value: Any, width: int | None = None) -> str:
    s = normalize_token(value)
    if not s:
        return ""
    if width is not None and s.isdigit():
        return s.zfill(width)
    return s


def first_existing_column(columns: list[str], candidates: list[str]) -> str:
    normalized = {
        c.lower().replace(" ", "").replace("_", ""): c
        for c in columns
    }
    for candidate in candidates:
        key = candidate.lower().replace(" ", "").replace("_", "")
        if key in normalized:
            return normalized[key]
    raise ValueError(
        f"Could not find any of these columns: {candidates}. "
        f"Available columns: {columns}"
    )


def resolve_column(df: pd.DataFrame, user_col: str | None, candidates: list[str], *, label: str) -> str:
    if user_col is not None:
        if user_col not in df.columns:
            raise ValueError(
                f"{label} column {user_col!r} not found. "
                f"Available columns: {list(df.columns)}"
            )
        return user_col
    return first_existing_column(list(df.columns), candidates)


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
        n_missing = int(missing_mask.sum())
        print(f"  WARNING: {n_missing} blocks were not assigned by within-join; trying nearest fallback.")
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
                    "Provide either `block_mcd_col` or `mcd_shapefile`."
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
    out[county_col] = out[county_col].map(lambda x: _pad_code(x, 3))
    out[mcd_col] = out[mcd_col].map(normalize_token)
    out[pop_col] = pd.to_numeric(out[pop_col], errors="coerce").fillna(0.0)
    out = out[out[block_col] != ""]

    out = out.rename(columns={block_col: "id", county_col: "COUNTY", mcd_col: "MCD", pop_col: "POP20"})
    out = out.groupby(["id", "COUNTY", "MCD"], dropna=False, as_index=False)["POP20"].sum()

    total_pop = float(out["POP20"].sum())
    if total_pop <= 0:
        raise ValueError(
            "Block metadata population sums to zero. "
            "Your block shapefile may not contain population. "
            "Merge block POP20 into it first."
        )

    print("  Resolved block metadata columns:")
    print(f"    block id: {block_col}")
    print(f"    county:   {county_col}")
    print(f"    MCD:      {mcd_col}")
    print(f"    pop:      {pop_col}")
    print(f"    total pop in metadata: {total_pop:,.0f}")
    print(f"    rows after aggregation: {len(out):,}")

    return out.to_dict("records")


def read_block_assignment_excel(path: str | Path, *, block_col: str | None = None, district_col: str | None = None) -> dict[str, Any]:
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
    out[block_col] = out[block_col].map(normalize_block_id)
    out[district_col] = out[district_col].astype(str).str.strip()
    out = out[(out[block_col] != "") & (out[district_col] != "")]
    out = out.drop_duplicates(subset=[block_col], keep="last")

    return dict(zip(out[block_col], out[district_col]))


def load_municipal_split_traces(path: str | Path) -> list[MunicipalSplitTrace]:
    path = Path(path)
    with path.open("r") as f:
        raw = json.load(f)

    traces = []
    for item in raw:
        traces.append(
            MunicipalSplitTrace(
                chain_id=item["chain_id"],
                proposals=list(item["proposals"]),
                n_cut_municipalities=list(item["n_cut_municipalities"]),
                ousted_population=list(item["ousted_population"]),
                ousted_population_share=list(item["ousted_population_share"]),
                n_maps=int(item["n_maps"]),
                municipality_cols=list(item.get("municipality_cols", ["COUNTY", "MCD"])),
                pop_col=str(item.get("pop_col", "POP20")),
                geography_mode=item.get("geography_mode", "COUSUB"),
                graph_path=item.get("graph_path"),
                population_basis=item.get("population_basis", "POP20"),
                represented_population=item.get("represented_population"),
            )
        )
    return traces


def compute_reference_summaries(
    *,
    block_nodes: list[dict[str, Any]],
    enacted_path: str | Path,
    overturned_path: str | Path,
    remedial_path: str | Path,
    assignment_block_col: str | None = None,
    assignment_district_col: str | None = None,
    geography: MunicipalGeography | None = None,
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
        summary = (
            geography.summarize_assignment(
                assignment, block_nodes, require_complete_assignment=True
            )
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
        summaries[label] = summary
    return summaries


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
    municipality_name = municipality_name_for_traces(traces)
    ousted_metric = "ousted_population_share" if ousted_scale == "share" else "ousted_population"
    ousted_values = pooled_values(traces, metric=ousted_metric)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    reference_styles = {
        "Enacted ('25)": {"color": "#9467bd", "linestyle": "-",  "linewidth": 2.2},
        "Overturned ('21)": {"color": "#2ca02c", "linestyle": "--", "linewidth": 2.2},
        "Remedial ('22)": {"color": "#d62728", "linestyle": "-.", "linewidth": 2.2},
    }

    # Left: cut municipalities
    cut_min = int(np.floor(cut_values.min()))
    cut_max = int(np.ceil(cut_values.max()))
    bins_cut = np.arange(cut_min - 0.5, cut_max + 1.5, 1.0)

    axes[0].hist(cut_values, bins=bins_cut, edgecolor="black", alpha=0.75)
    axes[0].set_xlabel(f"Number of cut {municipality_name}s")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Pooled ensemble distribution")

    # Right: ousted score
    axes[1].hist(ousted_values, bins=30, edgecolor="black", alpha=0.75)
    axes[1].set_ylabel("Count")
    axes[1].set_title("Pooled ensemble distribution")
    if ousted_scale == "share":
        axes[1].set_xlabel(f"Ousted {municipality_name} population share")
        axes[1].ticklabel_format(axis="x", style="plain")
    else:
        axes[1].set_xlabel(f"Ousted {municipality_name} population")

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
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_reference_json(path: str | Path, summaries: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump({label: summary.to_dict() for label, summary in summaries.items()}, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot pooled municipal-split histograms and overlay Enacted / "
            "Overturned / Remedial reference lines."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--traces-json",
        type=Path,
        default=Path(os.environ.get("ATLAS_POSTHOC_MUNICIPAL_TRACES", DEFAULT_TRACES_JSON)),
        help="Path to municipal_split_traces.json generated by atlas-posthoc.",
    )

    parser.add_argument(
        "--block-metadata",
        type=Path,
        default=(Path(os.environ["ATLAS_POSTHOC_BLOCK_METADATA"]) if "ATLAS_POSTHOC_BLOCK_METADATA" in os.environ else None),
        help=(
            "Block-level metadata file. A shapefile is OK. Must contain block ID, "
            "county, population, and either an MCD column or geometry usable with "
            "--mcd-shapefile."
        ),
    )
    parser.add_argument("--block-col", default=None, help="Block ID column in the block metadata file.")
    parser.add_argument("--county-col", default=None, help="County column in the block metadata file.")
    parser.add_argument("--pop-col", default=None, help="Population column in the block metadata file.")
    parser.add_argument(
        "--block-mcd-col",
        default=None,
        help="MCD / municipality column already present in the block metadata file.",
    )

    parser.add_argument(
        "--mcd-shapefile",
        type=Path,
        default=(Path(os.environ["ATLAS_POSTHOC_MCD_SHAPEFILE"]) if "ATLAS_POSTHOC_MCD_SHAPEFILE" in os.environ else None),
        help="County subdivision / MCD shapefile used when the block file lacks an MCD column.",
    )
    parser.add_argument("--mcd-shape-county-col", default=None, help="County column in the MCD shapefile.")
    parser.add_argument("--mcd-shape-mcd-col", default=None, help="MCD / county subdivision column in the MCD shapefile.")

    parser.add_argument("--enacted", type=Path, default=DEFAULT_ENACTED, help="Block assignment spreadsheet for Enacted ('25).")
    parser.add_argument("--overturned", type=Path, default=DEFAULT_OVERTURNED, help="Block assignment spreadsheet for Overturned ('21).")
    parser.add_argument("--remedial", type=Path, default=DEFAULT_REMEDIAL, help="Block assignment spreadsheet for Remedial ('22).")
    parser.add_argument("--assignment-block-col", default=None, help="Block ID column in the BAF spreadsheets.")
    parser.add_argument("--assignment-district-col", default=None, help="District column in the BAF spreadsheets.")

    parser.add_argument(
        "--ousted-scale",
        choices=["share", "population"],
        default=os.environ.get("ATLAS_POSTHOC_MUNICIPAL_OUSTED_SCALE", "share"),
        help="Whether the second histogram uses ousted population share or raw ousted population.",
    )
    parser.add_argument(
        "--title",
        default="Municipal split ensemble histograms",
        help="Figure title.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output figure path.",
    )
    parser.add_argument(
        "--reference-json",
        type=Path,
        default=None,
        help="Optional JSON path for the computed reference plan summaries.",
    )
    add_municipality_arguments(parser)

    args = parser.parse_args()

    if args.block_metadata is None:
        parser.error(
            "--block-metadata is required. A shapefile is OK, but it must contain "
            "block ID, county, population, and either an MCD column or geometry "
            "usable with --mcd-shapefile."
        )

    if (
        args.municipality_mode == "COUSUB"
        and args.block_mcd_col is None
        and args.mcd_shapefile is None
    ):
        parser.error(
            "The block metadata needs an MCD source. Provide either "
            "--block-mcd-col if the block file already has MCDs, or "
            "--mcd-shapefile to spatially assign blocks to county subdivisions."
        )

    return args


def main() -> None:
    args = parse_args()
    geography = geography_from_args(args)

    print("Loading municipal split traces:", args.traces_json)
    traces = load_municipal_split_traces(args.traces_json)
    geography.validate_trace_metadata(traces)

    print("Loading block metadata:", args.block_metadata)
    block_nodes = geography.read_block_nodes(
        args.block_metadata,
        block_col=args.block_col,
        county_col=args.county_col,
        pop_col=args.pop_col,
        block_mcd_col=args.block_mcd_col,
        mcd_shapefile=args.mcd_shapefile,
        mcd_shape_county_col=args.mcd_shape_county_col,
        mcd_shape_mcd_col=args.mcd_shape_mcd_col,
    )
    print(f"  Loaded {len(block_nodes):,} block metadata rows")

    print("Computing reference-plan municipal split summaries...")
    reference_summaries = compute_reference_summaries(
        block_nodes=block_nodes,
        enacted_path=args.enacted,
        overturned_path=args.overturned,
        remedial_path=args.remedial,
        assignment_block_col=args.assignment_block_col,
        assignment_district_col=args.assignment_district_col,
        geography=geography,
    )

    for label, summary in reference_summaries.items():
        print(
            f"  {label}: cuts={summary.n_cut_municipalities}, "
            f"ousted_pop={summary.ousted_population:,.0f}, "
            f"ousted_share={summary.ousted_population_share:.4%}"
        )

    print("Plotting histograms...")
    plot_histograms(
        traces=traces,
        reference_summaries=reference_summaries,
        ousted_scale=args.ousted_scale,
        title=args.title,
        output_path=args.output,
    )
    print("Wrote figure:", args.output)

    reference_json = args.reference_json or args.output.with_name(args.output.stem + "_reference_summaries.json")
    write_reference_json(reference_json, reference_summaries)
    geography.write_metadata(reference_json.with_suffix(".municipality.json"))
    print("Wrote reference summary JSON:", reference_json)


if __name__ == "__main__":
    main()
