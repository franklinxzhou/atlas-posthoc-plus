#!/usr/bin/env python3
"""
NOT PART OF THE STANDARD DUKE QUANTIFYING GERRYMANDERING POST-PROCESSING PIPELINE
This script summarizes municipal-split tradeoff statistics for an external Ohio ensemble.

Expected input structure from data/other_ensembles:
  * OH.geojson: 8,937 features, one feature per geographic unit, with properties
    such as GEOID, county_muni, merge_unit, pop, and cd_2020.
  * OH.csv: 8,937 rows and 5,002 columns. The first column is a row index, the
    cd_2020 column is the reference plan, and the remaining unnamed columns are
    sampled congressional plans.

The script assumes row order in OH.csv matches feature order in OH.geojson. It
validates the row count and, by default, compares the cd_2020 assignments in the
CSV against the cd_2020 property in the GeoJSON.

The statistic computed for each plan is: 
n_cut_municipalities = number of municipality keys assigned to >1 district
ousted_population = sum_m [pop(m) - max_d pop(m assigned to district d)]
ousted_population_share = ousted_population / included_population

Default municipality key is county_muni because that is the MCD/municipality-like
field present in the outside GeoJSON. You can also try --municipality-col merge_unit
or --municipality-col muni, but note that muni has missing values in unincorporated
areas.

Call under the root of oh-gerrymandering:
  atlas-posthoc-plus run summarize-external-ensemble \
    --csv data/other_ensembles/OH.csv \
    --geojson data/other_ensembles/OH.geojson \
    --municipality-col county_muni \
    --output-json outputs/other_ensemble_municipal_summaries.json \
    --output-csv outputs/other_ensemble_municipal_summaries.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def normalize_key(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    return s


def load_geojson_properties(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r") as f:
        obj = json.load(f)
    if obj.get("type") != "FeatureCollection":
        raise ValueError(f"Expected FeatureCollection, found {obj.get('type')!r}")
    features = obj.get("features", [])
    if not features:
        raise ValueError(f"No features found in {path}")
    props = [feature.get("properties", {}) for feature in features]
    return props


def coerce_float(x: Any, *, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        val = float(x)
    except Exception:
        return default
    if not np.isfinite(val):
        return default
    return val


def build_municipality_codes(
    props: list[dict[str, Any]],
    *,
    municipality_col: str,
    fallback_municipality_col: str | None,
    pop_col: str,
    skip_missing_municipality: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    muni_keys: list[str] = []
    pops: list[float] = []
    keep: list[bool] = []

    for p in props:
        key = normalize_key(p.get(municipality_col))
        if not key and fallback_municipality_col:
            key = normalize_key(p.get(fallback_municipality_col))
        pop = coerce_float(p.get(pop_col), default=0.0)

        if not key and skip_missing_municipality:
            keep.append(False)
            muni_keys.append("")
            pops.append(pop)
        else:
            keep.append(True)
            muni_keys.append(key if key else "__MISSING_MUNICIPALITY__")
            pops.append(pop)

    keep_arr = np.asarray(keep, dtype=bool)
    pop_arr = np.asarray(pops, dtype=np.float64)

    kept_keys = [muni_keys[i] for i in np.flatnonzero(keep_arr)]
    unique_keys = sorted(set(kept_keys))
    key_to_code = {key: i for i, key in enumerate(unique_keys)}
    codes = np.asarray([key_to_code[key] for key in kept_keys], dtype=np.int32)
    kept_pop = pop_arr[keep_arr]

    return codes, kept_pop, unique_keys, keep_arr


def compute_one_plan(
    district_assignment: np.ndarray,
    *,
    muni_codes: np.ndarray,
    pop: np.ndarray,
    total_pop: float,
    n_munis: int,
    stride: int,
    eps: float = 1e-9,
) -> dict[str, Any]:
    d = np.asarray(district_assignment, dtype=np.int32)
    if d.shape[0] != muni_codes.shape[0]:
        raise ValueError(
            f"Assignment length {d.shape[0]:,} does not match kept geography units {muni_codes.shape[0]:,}."
        )
    if np.any(d < 0):
        raise ValueError("Negative district labels are not supported.")
    if int(d.max(initial=0)) >= stride:
        raise ValueError(f"District label exceeds stride={stride}; increase --district-stride.")

    flat = muni_codes * stride + d
    sums = np.bincount(flat, weights=pop, minlength=n_munis * stride).reshape(n_munis, stride)
    muni_total = sums.sum(axis=1)
    district_counts = (sums > eps).sum(axis=1)
    cut = district_counts > 1

    n_cut = int(cut.sum())
    ousted = float((muni_total[cut] - sums[cut].max(axis=1)).sum()) if n_cut else 0.0
    share = ousted / total_pop if total_pop > 0 else float("nan")

    labels = sorted(int(x) for x in np.unique(d))

    return {
        "n_cut_municipalities": n_cut,
        "ousted_population": ousted,
        "ousted_population_share": share,
        "n_unique_districts": len(labels),
        "district_labels": labels,
    }


def detect_and_rename_columns(df: pd.DataFrame, *, cd_col: str) -> tuple[pd.DataFrame, str, list[str]]:
    if df.shape[1] < 3:
        raise ValueError("Expected at least row index, cd_2020, and one sampled-plan column.")
    if cd_col not in df.columns:
        raise ValueError(f"Could not find {cd_col!r}. Available first columns: {list(df.columns[:20])}")

    first_col = df.columns[0]
    cd_pos = list(df.columns).index(cd_col)
    if cd_pos != 1:
        print(f"WARNING: {cd_col!r} is at column position {cd_pos}, not 1. Proceeding anyway.")

    sample_cols = list(df.columns[cd_pos + 1 :])
    rename = {col: f"other_{i:04d}" for i, col in enumerate(sample_cols, start=1)}
    df = df.rename(columns=rename)
    plan_cols = [cd_col] + [rename[col] for col in sample_cols]
    return df, first_col, plan_cols


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize municipal-split statistics for data/other_ensembles/OH.csv + OH.geojson.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", default="data/other_ensembles/OH.csv")
    p.add_argument("--geojson", default="data/other_ensembles/OH.geojson")
    p.add_argument("--cd-col", default="cd_2020")
    p.add_argument("--pop-col", default="pop")
    p.add_argument(
        "--municipality-col",
        default="county_muni",
        help="GeoJSON property used as the municipality/MCD-like key.",
    )
    p.add_argument(
        "--fallback-municipality-col",
        default="merge_unit",
        help="Fallback property if --municipality-col is missing/blank.",
    )
    p.add_argument("--skip-missing-municipality", action="store_true")
    p.add_argument("--expected-districts", type=int, default=15)
    p.add_argument(
        "--exclude-cd-2020",
        action="store_true",
        help="Do not include the cd_2020 reference column in the output summaries.",
    )
    p.add_argument(
        "--max-plans",
        type=int,
        default=None,
        help="For testing: summarize only this many sampled plans, plus cd_2020 unless excluded.",
    )
    p.add_argument(
        "--district-stride",
        type=int,
        default=None,
        help="Internal max district label + 1. Auto-detected by default.",
    )
    p.add_argument(
        "--no-validate-cd-2020-against-geojson",
        action="store_true",
        help="Skip checking CSV cd_2020 against GeoJSON property cd_2020.",
    )
    p.add_argument(
        "--output-json",
        default="outputs/other_ensemble_municipal_summaries.json",
    )
    p.add_argument(
        "--output-csv",
        default="outputs/other_ensemble_municipal_summaries.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    geojson_path = Path(args.geojson)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not geojson_path.exists():
        raise FileNotFoundError(geojson_path)

    print(f"Reading GeoJSON properties: {geojson_path}")
    props = load_geojson_properties(geojson_path)
    n_geo = len(props)
    print(f"  features: {n_geo:,}")

    print(f"Reading CSV assignment matrix: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  rows: {len(df):,}")
    print(f"  columns: {len(df.columns):,}")

    if len(df) != n_geo:
        raise ValueError(f"CSV rows ({len(df):,}) != GeoJSON features ({n_geo:,}). Row-order join is unsafe.")

    df, row_index_col, plan_cols = detect_and_rename_columns(df, cd_col=args.cd_col)

    # Validate the first column is a simple row index if possible.
    row_index = pd.to_numeric(df[row_index_col], errors="coerce")
    if row_index.notna().all():
        expected = np.arange(1, len(df) + 1)
        if np.array_equal(row_index.to_numpy(dtype=np.int64), expected):
            print(f"  first CSV column {row_index_col!r} is 1..n row index")
        else:
            print(f"WARNING: first CSV column {row_index_col!r} is numeric but not exactly 1..n.")

    if args.max_plans is not None:
        sampled = [c for c in plan_cols if c != args.cd_col]
        sampled = sampled[: args.max_plans]
        plan_cols = ([] if args.exclude_cd_2020 else [args.cd_col]) + sampled
    elif args.exclude_cd_2020:
        plan_cols = [c for c in plan_cols if c != args.cd_col]

    print(f"  plans to summarize: {len(plan_cols):,}")
    if args.cd_col in plan_cols:
        print("  includes cd_2020 reference column")

    if not args.no_validate_cd_2020_against_geojson and args.cd_col in df.columns:
        if all(args.cd_col in p for p in props):
            csv_cd = pd.to_numeric(df[args.cd_col], errors="coerce").to_numpy()
            geo_cd = np.asarray([coerce_float(p.get(args.cd_col), default=np.nan) for p in props])
            valid = np.isfinite(csv_cd) & np.isfinite(geo_cd)
            mismatches = int(np.sum(csv_cd[valid] != geo_cd[valid]))
            print(f"  cd_2020 CSV-vs-GeoJSON mismatches on valid rows: {mismatches:,}")
            if mismatches:
                print("WARNING: row-order match is questionable because cd_2020 values disagree.")
        else:
            print(f"WARNING: GeoJSON has no {args.cd_col!r} property to validate row order.")

    muni_codes, pop, muni_keys, keep_mask = build_municipality_codes(
        props,
        municipality_col=args.municipality_col,
        fallback_municipality_col=args.fallback_municipality_col,
        pop_col=args.pop_col,
        skip_missing_municipality=args.skip_missing_municipality,
    )
    included_pop = float(pop.sum())
    statewide_pop = float(sum(coerce_float(p.get(args.pop_col), default=0.0) for p in props))
    print(f"Municipality key: {args.municipality_col}")
    if args.fallback_municipality_col:
        print(f"  fallback key: {args.fallback_municipality_col}")
    print(f"  unique municipality keys: {len(muni_keys):,}")
    print(f"  kept geography units: {int(keep_mask.sum()):,} / {len(keep_mask):,}")
    print(f"  included population: {included_pop:,.0f} / statewide {statewide_pop:,.0f}")

    # Convert selected plan columns to a compact numeric matrix.
    plan_df = df.loc[keep_mask, plan_cols]
    assignments = plan_df.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.int16, copy=True)
    max_label = int(assignments.max(initial=0))
    min_label = int(assignments.min(initial=0))
    if min_label < 0:
        raise ValueError(f"Negative district label found: {min_label}")
    stride = args.district_stride if args.district_stride is not None else max_label + 1
    if stride <= max_label:
        raise ValueError(f"--district-stride={stride} is too small for max label {max_label}")
    print(f"  district labels range: {min_label}..{max_label}; stride={stride}")

    n_munis = len(muni_keys)
    summaries: list[dict[str, Any]] = []

    for j, col in enumerate(plan_cols):
        if j == 0 or (j + 1) % 250 == 0 or j + 1 == len(plan_cols):
            print(f"  summarizing plan {j + 1:,}/{len(plan_cols):,}: {col}")
        stats = compute_one_plan(
            assignments[:, j],
            muni_codes=muni_codes,
            pop=pop,
            total_pop=included_pop,
            n_munis=n_munis,
            stride=stride,
        )
        stats["plan_id"] = col
        stats["source"] = "other_ensemble"
        stats["is_reference_cd_2020"] = bool(col == args.cd_col)
        stats["status"] = "ok" if stats["n_unique_districts"] == args.expected_districts else "unexpected_district_count"
        stats["municipality_col"] = args.municipality_col
        stats["included_population"] = included_pop
        stats["statewide_population"] = statewide_pop
        stats["included_population_share"] = included_pop / statewide_pop if statewide_pop > 0 else float("nan")
        summaries.append(stats)

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_json.open("w") as f:
        json.dump(summaries, f, indent=2)
    pd.DataFrame(summaries).to_csv(out_csv, index=False)

    ok_count = sum(1 for s in summaries if s["status"] == "ok")
    print(f"Wrote JSON: {out_json}")
    print(f"Wrote CSV:  {out_csv}")
    print(f"OK plans with {args.expected_districts} districts: {ok_count:,}/{len(summaries):,}")

    if summaries:
        nonref = [s for s in summaries if not s["is_reference_cd_2020"] and s["status"] == "ok"]
        if nonref:
            x = np.asarray([s["n_cut_municipalities"] for s in nonref], dtype=float)
            y = np.asarray([s["ousted_population_share"] for s in nonref], dtype=float)
            print("Other ensemble summary, excluding cd_2020:")
            print(f"  n: {len(nonref):,}")
            print(f"  cut municipalities: min={x.min():.0f}, median={np.median(x):.0f}, max={x.max():.0f}")
            print(f"  ousted share: min={y.min():.4%}, median={np.median(y):.4%}, max={y.max():.4%}")


if __name__ == "__main__":
    main()
