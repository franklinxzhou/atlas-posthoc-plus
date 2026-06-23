#!/usr/bin/env python3
"""
NOT PART OF THE STANDARD DUKE QUANTIFYING GERRYMANDERING POST-PROCESSING PIPELINE - This script processes an external ensemble
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ID_KEYWORDS = (
    "id", "geoid", "geoid20", "block", "blockid", "precinct", "vtd",
    "name", "orig", "county", "countyfp", "cnty", "mcd", "cousub",
    "tract", "index", "gid", "fid", "objectid",
)

PLAN_KEYWORDS = (
    "plan", "assignment", "district", "districtid", "dist", "cd", "cong",
    "map", "sample", "run", "chain",
)


def norm_name(x: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def norm_token(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0") and re.fullmatch(r"-?\d+\.0", s):
        s = s[:-2]
    return s


def safe_int_like(x: Any) -> int | None:
    s = norm_token(x)
    if not s:
        return None
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    return None


def print_section(title: str, lines: list[str], out_lines: list[str]) -> None:
    bar = "=" * len(title)
    out_lines.append("")
    out_lines.append(title)
    out_lines.append(bar)
    out_lines.extend(lines)


def count_csv_rows_and_header(path: Path) -> tuple[int, list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return 0, []
        n = sum(1 for _ in reader)
    return n, header


def summarize_sample_values(series: pd.Series, max_values: int = 10) -> str:
    vals = [norm_token(v) for v in series.dropna().head(max_values).tolist()]
    return ", ".join(repr(v) for v in vals)


def column_role_guess(col: str, series: pd.Series) -> dict[str, Any]:
    ncol = norm_name(col)
    nonnull = series.dropna()
    sample_values = [norm_token(v) for v in nonnull.head(200).tolist()]
    sample_values = [v for v in sample_values if v != ""]
    unique_values = sorted(set(sample_values))

    int_values = [safe_int_like(v) for v in sample_values]
    int_values_clean = [v for v in int_values if v is not None]
    int_like_share = len(int_values_clean) / len(sample_values) if sample_values else 0.0

    min_int = min(int_values_clean) if int_values_clean else None
    max_int = max(int_values_clean) if int_values_clean else None
    n_unique_sample = len(unique_values)

    looks_id = any(k in ncol for k in ID_KEYWORDS)
    looks_named_plan = any(k in ncol for k in PLAN_KEYWORDS)

    # District assignment columns usually have mostly integers, few distinct labels,
    # and values in a plausible congressional district range. Use sample only.
    looks_assignment_by_values = (
        int_like_share >= 0.95
        and n_unique_sample >= 2
        and n_unique_sample <= 30
        and min_int is not None
        and max_int is not None
        and min_int >= -1
        and max_int <= 99
    )

    return {
        "column": col,
        "looks_id_by_name": looks_id,
        "looks_plan_by_name": looks_named_plan,
        "looks_assignment_by_values": looks_assignment_by_values,
        "n_unique_in_sample": n_unique_sample,
        "int_like_share_in_sample": int_like_share,
        "min_int_in_sample": min_int,
        "max_int_in_sample": max_int,
        "sample_values": summarize_sample_values(series),
    }


def inspect_csv(csv_path: Path, *, sample_rows: int, out_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, list[str]]:
    n_rows, header = count_csv_rows_and_header(csv_path)
    n_cols = len(header)

    df = pd.read_csv(csv_path, nrows=sample_rows, dtype=str, low_memory=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(header, name="column").to_csv(out_dir / "OH_csv_columns.txt", index=False, header=False)
    df.head(min(20, len(df))).to_csv(out_dir / "OH_csv_sample_first_rows.csv", index=False)
    df.iloc[:, : min(80, df.shape[1])].head(min(20, len(df))).to_csv(
        out_dir / "OH_csv_sample_first_rows_first_80_cols.csv", index=False
    )

    role_rows = [column_role_guess(c, df[c]) for c in df.columns]
    role_df = pd.DataFrame(role_rows)
    role_df.to_csv(out_dir / "OH_csv_column_role_guesses.csv", index=False)

    likely_id_cols = [r["column"] for r in role_rows if r["looks_id_by_name"]]
    likely_assignment_cols = [
        r["column"]
        for r in role_rows
        if r["looks_assignment_by_values"] and not r["looks_id_by_name"]
    ]
    likely_plan_name_cols = [r["column"] for r in role_rows if r["looks_plan_by_name"]]

    # Orientation heuristic.
    orientation_notes: list[str] = []
    if n_cols >= 50 and len(likely_assignment_cols) >= 10:
        orientation_notes.append(
            "Likely WIDE: one row per geography unit, many district-assignment columns."
        )
    if any(norm_name(c) in {"plan", "planid", "map", "mapid", "sample", "sampleid", "chain"} for c in header) and any(
        "district" in norm_name(c) or norm_name(c) in {"dist", "cd"} for c in header
    ):
        orientation_notes.append(
            "Possibly LONG: columns look like plan/map id plus district assignment."
        )
    if not orientation_notes:
        orientation_notes.append(
            "Orientation unclear from the sample. Check OH_csv_column_role_guesses.csv."
        )

    summary = {
        "path": str(csv_path),
        "file_size_mb": csv_path.stat().st_size / 1_000_000,
        "n_data_rows": n_rows,
        "n_columns": n_cols,
        "first_columns": header[:40],
        "last_columns": header[-20:] if len(header) > 20 else header,
        "likely_id_columns_by_name": likely_id_cols[:80],
        "likely_plan_columns_by_name": likely_plan_name_cols[:80],
        "likely_assignment_columns_by_values_sample": likely_assignment_cols[:80],
        "n_likely_assignment_columns_by_values_sample": len(likely_assignment_cols),
        "orientation_notes": orientation_notes,
    }

    return summary, df, header


def inspect_geojson(geojson_path: Path, *, out_dir: Path) -> dict[str, Any]:
    with geojson_path.open("r", encoding="utf-8", errors="replace") as f:
        gj = json.load(f)

    features = gj.get("features", [])
    n_features = len(features)

    geom_types = Counter()
    prop_keys_counter = Counter()
    sample_props: list[dict[str, Any]] = []

    for i, feat in enumerate(features):
        geom = feat.get("geometry") or {}
        geom_types[geom.get("type", "<missing>")] += 1
        props = feat.get("properties") or {}
        prop_keys_counter.update(props.keys())
        if i < 5:
            sample_props.append(props)

    prop_keys = sorted(prop_keys_counter.keys())

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(prop_keys, name="property").to_csv(out_dir / "OH_geojson_properties.txt", index=False, header=False)
    pd.DataFrame(sample_props).to_csv(out_dir / "OH_geojson_sample_properties.csv", index=False)

    candidate_props = [
        k for k in prop_keys
        if any(word in norm_name(k) for word in ID_KEYWORDS)
    ]

    summary = {
        "path": str(geojson_path),
        "file_size_mb": geojson_path.stat().st_size / 1_000_000,
        "top_level_keys": sorted(gj.keys()),
        "type": gj.get("type"),
        "name": gj.get("name"),
        "crs": gj.get("crs"),
        "bbox": gj.get("bbox"),
        "n_features": n_features,
        "geometry_type_counts": dict(geom_types),
        "n_property_keys": len(prop_keys),
        "first_properties": prop_keys[:80],
        "candidate_join_properties_by_name": candidate_props[:80],
        "sample_properties_first_feature": sample_props[0] if sample_props else {},
    }
    return summary


def compare_join_candidates(csv_header: list[str], geo_summary: dict[str, Any], csv_rows: int) -> dict[str, Any]:
    geo_props = geo_summary.get("first_properties", [])
    # first_properties may be capped; reload property file not needed for initial guess.
    csv_norm = {norm_name(c): c for c in csv_header}
    geo_norm = {norm_name(p): p for p in geo_props}

    exact_norm_matches = []
    for n, c in csv_norm.items():
        if n in geo_norm:
            exact_norm_matches.append((c, geo_norm[n]))

    csv_id_cols = [c for c in csv_header if any(k in norm_name(c) for k in ID_KEYWORDS)]
    geo_id_props = [p for p in geo_props if any(k in norm_name(p) for k in ID_KEYWORDS)]

    return {
        "csv_rows_equal_geojson_features": csv_rows == geo_summary.get("n_features"),
        "csv_rows": csv_rows,
        "geojson_features": geo_summary.get("n_features"),
        "normalized_name_matches_csv_to_geojson": exact_norm_matches[:80],
        "csv_id_candidates": csv_id_cols[:80],
        "geojson_id_candidates": geo_id_props[:80],
    }


def write_report(report_path: Path, report: dict[str, Any]) -> None:
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    lines: list[str] = []

    csv_s = report["csv"]
    print_section(
        "CSV summary",
        [
            f"Path: {csv_s['path']}",
            f"Size: {csv_s['file_size_mb']:.2f} MB",
            f"Rows: {csv_s['n_data_rows']:,}",
            f"Columns: {csv_s['n_columns']:,}",
            "Orientation guess: " + " | ".join(csv_s["orientation_notes"]),
            "First columns: " + ", ".join(map(str, csv_s["first_columns"][:20])),
            "Likely ID columns: " + ", ".join(map(str, csv_s["likely_id_columns_by_name"][:20])),
            "Likely assignment columns from sample: "
            + ", ".join(map(str, csv_s["likely_assignment_columns_by_values_sample"][:20])),
            f"Number of likely assignment columns from sample: {csv_s['n_likely_assignment_columns_by_values_sample']:,}",
        ],
        lines,
    )

    geo_s = report["geojson"]
    print_section(
        "GeoJSON summary",
        [
            f"Path: {geo_s['path']}",
            f"Size: {geo_s['file_size_mb']:.2f} MB",
            f"Type: {geo_s['type']}",
            f"Features: {geo_s['n_features']:,}",
            "Geometry types: " + json.dumps(geo_s["geometry_type_counts"]),
            "Properties: " + ", ".join(map(str, geo_s["first_properties"][:30])),
            "Candidate join properties: " + ", ".join(map(str, geo_s["candidate_join_properties_by_name"][:30])),
        ],
        lines,
    )

    join_s = report["join_candidates"]
    print_section(
        "Join/orientation clues",
        [
            f"CSV rows equal GeoJSON features? {join_s['csv_rows_equal_geojson_features']}",
            f"CSV rows: {join_s['csv_rows']:,}",
            f"GeoJSON features: {join_s['geojson_features']:,}",
            "Normalized column/property matches: "
            + ", ".join(f"{c} <-> {g}" for c, g in join_s["normalized_name_matches_csv_to_geojson"][:30]),
            "If rows match but no clean key appears, the ensemble may rely on row order; verify before computing statistics.",
        ],
        lines,
    )

    print_section(
        "Files written",
        [
            str(report_path),
            str(report_path.parent / "OH_csv_columns.txt"),
            str(report_path.parent / "OH_csv_sample_first_rows.csv"),
            str(report_path.parent / "OH_csv_sample_first_rows_first_80_cols.csv"),
            str(report_path.parent / "OH_csv_column_role_guesses.csv"),
            str(report_path.parent / "OH_geojson_properties.txt"),
            str(report_path.parent / "OH_geojson_sample_properties.csv"),
        ],
        lines,
    )

    text = "\n".join(lines).strip() + "\n"
    (report_path.parent / "OTHER_ENSEMBLE_STRUCTURE_REPORT.txt").write_text(text, encoding="utf-8")
    print(text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect OH.csv and OH.geojson from another group's Ohio ensemble.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", default="data/other_ensembles/OH.csv", help="Path to OH.csv")
    p.add_argument("--geojson", default="data/other_ensembles/OH.geojson", help="Path to OH.geojson")
    p.add_argument("--out-dir", default="outputs/other_ensembles_probe", help="Output folder for reports/samples")
    p.add_argument("--sample-rows", type=int, default=1000, help="Rows to sample from CSV for value/column role guesses")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    geojson_path = Path(args.geojson)
    out_dir = Path(args.out_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")

    csv_summary, _sample_df, csv_header = inspect_csv(csv_path, sample_rows=args.sample_rows, out_dir=out_dir)
    geo_summary = inspect_geojson(geojson_path, out_dir=out_dir)
    join_summary = compare_join_candidates(csv_header, geo_summary, csv_summary["n_data_rows"])

    report = {
        "csv": csv_summary,
        "geojson": geo_summary,
        "join_candidates": join_summary,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_report(out_dir / "other_ensemble_structure_report.json", report)


if __name__ == "__main__":
    main()
