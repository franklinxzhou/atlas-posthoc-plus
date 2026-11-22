#!/usr/bin/env python3
"""


Example
-------
atlas-posthoc-plus run ranked-vote-cvap \
  --chain-files \
    outputs/chains/soft_constraint/congressional/run37_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run38_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run39_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run40_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
  --graph-json data/graph/OHpct20.json \
  --cvap-file data/raw/oh_cvap_2023_b.zip \
  --crosswalk-csv outputs/oh_block_to_precinct_crosswalk.csv \
  --enacted-baf "inputs/plans/October 31 2025 CD BAF.xlsx" \
  --overturned-baf "inputs/plans/SubSB258-Block-Assignment-File.xlsx" \
  --remedial-baf "inputs/plans/March 2 2022 CD BAF.xlsx" \
  --output outputs/oh_ranked_vote_black_cvap_ensemble_vs_plans.png \
  --output-csv outputs/oh_ranked_vote_black_cvap_ensemble_vs_plans.csv

To make it consistent with my earlier code, 
the ensemble is filtered using the same county-compliance rule
used by plot_oh_compliant_ensemble_vs_plans_adjusted.py:
  keep a map iff no county touches more than --max-districts-per-county districts.

It is possible to additionally pass --compliant-records-csv to restrict to a previously
saved list of accepted records, but this is optional and is applied before the
county-compliance check. (don't really recommend, just to save this feature)
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable

from atlas_posthoc_plus.paper import (
    figure_legend,
    official_plan_style,
    paper_mode_enabled,
    save_figure,
    set_panel_title,
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from atlas_posthoc_plus.io.atlas_records import iter_map_records
from atlas_posthoc_plus.io.chain_sweep import (
    cached_assignment_series,
    cached_compliance,
    configured_sweep,
)
from atlas_posthoc_plus.io.graph import read_graph_frame


PLAN_STYLES = {
    "Enacted ('25)": {"color": "#9467bd", "marker": "o", "linewidth": 2.2},
    "Overturned ('21)": {"color": "#2ca02c", "marker": "s", "linewidth": 2.2},
    "Remedial ('22)": {"color": "#d62728", "marker": "^", "linewidth": 2.2},
}


# ---------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------
def canonical_colname(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def resolve_column(columns: Iterable[str], candidates: Iterable[str | None], *, label: str) -> str:
    columns = list(columns)
    normalized = {canonical_colname(c): c for c in columns}

    for candidate in candidates:
        if candidate is None:
            continue
        key = canonical_colname(candidate)
        if key in normalized:
            return normalized[key]

    # Helpful fallback for columns like DistrictID:1.
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

    Example AtlasIO key:
        '["TROY 1-B_MIAMI__SD5__HD80"]'

    Graph NAME:
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
# Input readers
# ---------------------------------------------------------------------
def open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("r")


def read_graph_nodes(graph_json: str | Path) -> pd.DataFrame:
    df = read_graph_frame(graph_json).copy()

    required = ["id", "NAME", "ORIG_NAME", "POP20", "G20PREDEM", "G20PREREP"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Graph node table is missing required columns {missing}.")

    df["id"] = pd.to_numeric(df["id"], errors="raise").astype(int)
    for col in ["POP20", "G20PREDEM", "G20PREREP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def read_cvap_file(cvap_file: str | Path, *, cvap_csv_member: str | None = None) -> pd.DataFrame:
    """
    Read either:
      - the Ohio CVAP zip, containing oh/oh_cvap_2023_2020_b.csv; or
      - the extracted CSV.
    """
    cvap_file = Path(cvap_file)

    if cvap_file.suffix.lower() == ".zip":
        with zipfile.ZipFile(cvap_file) as z:
            if cvap_csv_member is None:
                csv_members = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if len(csv_members) != 1:
                    raise ValueError(
                        f"Could not infer CVAP CSV inside zip. CSV members: {csv_members}"
                    )
                cvap_csv_member = csv_members[0]

            with z.open(cvap_csv_member) as f:
                df = pd.read_csv(f, dtype=str)
    else:
        df = pd.read_csv(cvap_file, dtype=str)

    if "GEOID20" not in df.columns:
        raise ValueError(f"CVAP file must contain GEOID20. Available columns: {list(df.columns)}")

    df["GEOID20"] = df["GEOID20"].map(normalize_block_id)

    numeric_cols = [c for c in df.columns if c != "GEOID20"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def read_crosswalk(crosswalk_csv: str | Path) -> pd.DataFrame:
    cw = pd.read_csv(crosswalk_csv, dtype=str)

    block_col = resolve_column(cw.columns, ["GEOID20", "BLOCK", "Block", "GEOID"], label="crosswalk block")
    precinct_col = resolve_column(cw.columns, ["NAME", "ORIG_NAME", "precinct", "PRECINCT"], label="crosswalk precinct")
    weight_col = resolve_column(cw.columns, ["POP20", "ALLOCATED_POP20", "weight", "share"], label="crosswalk weight")

    out = cw[[block_col, precinct_col, weight_col]].copy()
    out = out.rename(columns={block_col: "GEOID20", precinct_col: "PRECINCT_NAME", weight_col: "CW_WEIGHT"})
    out["GEOID20"] = out["GEOID20"].map(normalize_block_id)
    out["PRECINCT_NAME"] = out["PRECINCT_NAME"].map(normalize_token)
    out["CW_WEIGHT"] = pd.to_numeric(out["CW_WEIGHT"], errors="coerce").fillna(0.0)

    return out


def add_precinct_cvap_to_graph(
    graph_nodes: pd.DataFrame,
    cvap: pd.DataFrame,
    crosswalk: pd.DataFrame,
    *,
    graph_key_col: str,
    cvap_total_col: str,
    black_cvap_col: str,
) -> pd.DataFrame:
    """
    Aggregate block-level CVAP to precincts via block->precinct crosswalk and
    merge onto graph nodes.
    """
    required = ["GEOID20", cvap_total_col, black_cvap_col]
    missing = [c for c in required if c not in cvap.columns]
    if missing:
        raise ValueError(
            f"CVAP file is missing required columns {missing}. "
            f"Available columns include: {list(cvap.columns)[:20]} ..."
        )

    block_to_precinct = crosswalk[["GEOID20", "PRECINCT_NAME"]].drop_duplicates()
    merged = block_to_precinct.merge(cvap[required], on="GEOID20", how="left")

    for col in [cvap_total_col, black_cvap_col]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    precinct_cvap = (
        merged.groupby("PRECINCT_NAME", as_index=False)[[cvap_total_col, black_cvap_col]]
        .sum()
    )

    graph = graph_nodes.copy()

    if graph_key_col not in graph.columns:
        raise ValueError(
            f"graph_key_col={graph_key_col!r} not found in graph nodes. "
            f"Available columns: {list(graph.columns)}"
        )

    graph["_CVAP_JOIN_KEY"] = graph[graph_key_col].map(normalize_token)
    graph = graph.merge(
        precinct_cvap.rename(columns={"PRECINCT_NAME": "_CVAP_JOIN_KEY"}),
        on="_CVAP_JOIN_KEY",
        how="left",
    )

    for col in [cvap_total_col, black_cvap_col]:
        graph[col] = pd.to_numeric(graph[col], errors="coerce").fillna(0.0)

    n_total = len(graph)
    n_with_cvap = int((graph[cvap_total_col] > 0).sum())
    print(f"Graph CVAP merge: {n_with_cvap:,} / {n_total:,} graph nodes have positive {cvap_total_col}")

    if n_with_cvap < 0.90 * n_total:
        print("WARNING: fewer than 90% of graph nodes have positive CVAP after merge.")

    return graph.drop(columns=["_CVAP_JOIN_KEY"])


def iter_atlasio_map_records(path: str | Path) -> Iterable[dict[str, Any]]:
    yield from iter_map_records(path)


def chain_id_from_path(path: str | Path) -> str:
    name = Path(path).name
    m = re.search(r"(run\d+)", name)
    if m:
        return m.group(1)
    return Path(path).stem


def build_node_key_lookup(
    graph_nodes: pd.DataFrame,
    *,
    key_col: str,
    node_id_col: str,
) -> tuple[dict[str, int], tuple[int, ...]]:
    if key_col not in graph_nodes.columns:
        raise ValueError(f"key_col={key_col!r} not found in graph nodes.")

    node_key_to_id: dict[str, int] = {}
    duplicate_keys: list[str] = []
    node_ids: list[int] = []

    for raw_key, raw_node_id in zip(
        graph_nodes[key_col].to_numpy(),
        graph_nodes[node_id_col].to_numpy(),
        strict=True,
    ):
        key = normalize_atlas_key(raw_key)
        node_id = int(raw_node_id)
        node_ids.append(node_id)
        if key in node_key_to_id:
            duplicate_keys.append(key)
        node_key_to_id[key] = node_id

    if duplicate_keys:
        raise ValueError(f"Duplicate graph keys for {key_col!r}: {duplicate_keys[:5]}")

    return node_key_to_id, tuple(node_ids)


def atlasio_districting_to_series(
    districting: Any,
    graph_nodes: pd.DataFrame,
    *,
    key_col: str = "NAME",
    node_id_col: str = "id",
    node_lookup: tuple[dict[str, int], tuple[int, ...]] | None = None,
) -> pd.Series:
    cached = cached_assignment_series(districting, node_id_col=node_id_col)
    if cached is not None:
        return cached.sort_index()

    if node_lookup is None:
        node_key_to_id, node_ids = build_node_key_lookup(
            graph_nodes,
            key_col=key_col,
            node_id_col=node_id_col,
        )
    else:
        node_key_to_id, node_ids = node_lookup

    node_id_set = set(node_ids)
    graph_size = len(node_ids)

    def finish_assignment(out: dict[int, int], unmatched: list[str]) -> pd.Series:
        if unmatched:
            raise ValueError(
                f"{len(unmatched)} AtlasIO districting keys did not match graph {key_col!r}. "
                f"Examples: {unmatched[:5]}"
            )

        if len(out) != graph_size:
            raise ValueError(f"Matched {len(out)} assignments but graph has {graph_size} nodes.")

        s = pd.Series(out, dtype=int)
        s.index.name = node_id_col
        return s.sort_index()

    # Atlas-posthoc exposes flattened AtlasIO assignments as a dict. Keys may be
    # raw AtlasIO strings like '["PRECINCT_KEY"]', canonical tuple keys, or in
    # some fixture/nonstandard cases direct graph node ids.
    if isinstance(districting, dict) and districting:
        out: dict[int, int] = {}
        unmatched: list[str] = []

        for raw_key, raw_dist in districting.items():
            key = normalize_atlas_key(raw_key)
            node_id = node_key_to_id.get(key)

            if node_id is None:
                raw_node_id = None
                try:
                    raw_node_id = coerce_district_label(raw_key)
                except ValueError:
                    pass
                if raw_node_id in node_id_set:
                    node_id = raw_node_id

            if node_id is None:
                unmatched.append(key)
                continue

            out[node_id] = coerce_district_label(raw_dist)

        return finish_assignment(out, unmatched)

    # Your AtlasIO format: [{'["PRECINCT_KEY"]': district}, ...]
    if isinstance(districting, list) and districting:
        if all(isinstance(x, dict) and len(x) == 1 for x in districting):
            out: dict[int, int] = {}
            unmatched: list[str] = []

            for item in districting:
                raw_key, raw_dist = next(iter(item.items()))
                key = normalize_atlas_key(raw_key)
                node_id = node_key_to_id.get(key)
                if node_id is None:
                    unmatched.append(key)
                    continue
                out[node_id] = coerce_district_label(raw_dist)

            return finish_assignment(out, unmatched)

        # Fallback: node-order list.
        if len(districting) == graph_size and not isinstance(districting[0], (list, tuple, dict, set)):
            s = pd.Series(
                [coerce_district_label(x) for x in districting],
                index=node_ids,
                dtype=int,
            )
            s.index.name = node_id_col
            return s.sort_index()

    raise ValueError(f"Unsupported AtlasIO districting format: {type(districting)}")


# ---------------------------------------------------------------------
# Optional post-hoc compliant filter
# ---------------------------------------------------------------------
def load_compliant_filter(path: str | Path | None):
    if path is None:
        return None

    df = pd.read_csv(path)

    colmap = {canonical_colname(c): c for c in df.columns}

    chain_col = colmap.get("chainid")
    if chain_col is None:
        raise ValueError(
            "compliant-records CSV must contain chain_id plus one of "
            "record_index, map_index, proposal, or name."
        )

    for kind, aliases in {
        "record_index": ["recordindex", "mapindex"],
        "proposal": ["proposal", "name"],
    }.items():
        for alias in aliases:
            if alias in colmap:
                value_col = colmap[alias]
                values = set(
                    (str(row[chain_col]), str(row[value_col]))
                    for _, row in df.iterrows()
                )
                return {"kind": kind, "values": values, "chain_col": chain_col, "value_col": value_col}

    raise ValueError(
        "compliant-records CSV must contain chain_id plus one of "
        "record_index, map_index, proposal, or name."
    )


def record_is_kept(
    keep_filter,
    *,
    chain_id: str,
    record_index: int,
    record: dict[str, Any],
) -> bool:
    if keep_filter is None:
        return True

    if keep_filter["kind"] == "record_index":
        return (str(chain_id), str(record_index)) in keep_filter["values"]

    if keep_filter["kind"] == "proposal":
        proposal = record.get("name", record.get("proposal", record_index))
        return (str(chain_id), str(proposal)) in keep_filter["values"]

    raise ValueError(f"Unknown keep filter kind: {keep_filter['kind']}")


def county_compliance_summary_for_assignment(
    assignment: pd.Series,
    graph_nodes: pd.DataFrame,
    *,
    node_id_col: str,
    county_col: str,
    max_districts_per_county: int,
) -> dict[str, Any]:
    cached = cached_compliance(assignment)
    if cached is not None:
        return cached

    """
    Same county-compliance rule used in
    plot_oh_compliant_ensemble_vs_plans_adjusted.py.

    A map is retained iff no county is assigned to more than
    max_districts_per_county districts. For the Ohio congressional post-hoc
    filter, the default is 2.
    """
    if county_col not in graph_nodes.columns:
        raise ValueError(
            f"county_col={county_col!r} not found in graph nodes. "
            f"Available columns: {list(graph_nodes.columns)}"
        )

    assigned = graph_nodes[[node_id_col, county_col]].merge(
        assignment.rename("DISTRICT").rename_axis(node_id_col).reset_index(),
        on=node_id_col,
        how="left",
    )

    missing_nodes = int(assigned["DISTRICT"].isna().sum())
    assigned_nodes = int(len(assigned) - missing_nodes)

    by_county = (
        assigned.dropna(subset=["DISTRICT"])
        .groupby(county_col)["DISTRICT"]
        .nunique()
    )

    worst_split = int(by_county.max()) if len(by_county) else 0
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
# Metric computation
# ---------------------------------------------------------------------
def aggregate_vote_cvap_by_district(
    assignment: pd.Series,
    graph_nodes: pd.DataFrame,
    *,
    node_id_col: str,
    dem_col: str,
    rep_col: str,
    cvap_total_col: str,
    black_cvap_col: str,
) -> pd.DataFrame:
    df = graph_nodes[[node_id_col, dem_col, rep_col, cvap_total_col, black_cvap_col]].merge(
        assignment.rename("DISTRICT").rename_axis(node_id_col).reset_index(),
        on=node_id_col,
        how="inner",
    )

    agg = (
        df.groupby("DISTRICT", as_index=False)[[dem_col, rep_col, cvap_total_col, black_cvap_col]]
        .sum()
    )

    vote_denom = agg[dem_col] + agg[rep_col]
    agg["dem_share"] = agg[dem_col] / vote_denom.replace(0, np.nan)
    agg["black_cvap_share"] = agg[black_cvap_col] / agg[cvap_total_col].replace(0, np.nan)

    agg["dem_share"] = agg["dem_share"].fillna(0.0)
    agg["black_cvap_share"] = agg["black_cvap_share"].fillna(0.0)

    return agg


def ordered_rank_vectors(
    district_df: pd.DataFrame,
    *,
    n_districts: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    ordered = district_df.sort_values(["dem_share", "DISTRICT"]).reset_index(drop=True)

    if len(ordered) != n_districts:
        raise ValueError(f"Expected {n_districts} districts, found {len(ordered)}.")

    ordered["rank"] = np.arange(1, n_districts + 1)

    return (
        ordered["dem_share"].to_numpy(float),
        ordered["black_cvap_share"].to_numpy(float),
        ordered,
    )


def build_ensemble_matrices(
    chain_files: list[str | Path],
    graph_nodes: pd.DataFrame,
    *,
    atlas_key_col: str,
    node_id_col: str,
    county_col: str,
    dem_col: str,
    rep_col: str,
    cvap_total_col: str,
    black_cvap_col: str,
    n_districts: int,
    compliant_filter,
    filter_county_compliant: bool,
    max_districts_per_county: int,
    max_maps_per_chain: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    dem_rows = []
    black_rows = []
    summary_rows = []
    compliance_rows = []
    node_lookup = build_node_key_lookup(
        graph_nodes,
        key_col=atlas_key_col,
        node_id_col=node_id_col,
    )

    for chain_file in chain_files:
        chain_id = chain_id_from_path(chain_file)
        maps_seen = 0
        used = 0
        skipped_by_record_filter = 0
        skipped_noncompliant = 0
        failed = 0
        double_cross_counts: list[int] = []
        worst_splits: list[int] = []

        action = "Analyzing shared assignments" if configured_sweep() else "Reading chain"
        print(f"{action}: {chain_file}")

        for record_index, record in enumerate(iter_atlasio_map_records(chain_file)):
            if max_maps_per_chain is not None and maps_seen >= max_maps_per_chain:
                break

            maps_seen += 1

            if not record_is_kept(
                compliant_filter,
                chain_id=chain_id,
                record_index=record_index,
                record=record,
            ):
                skipped_by_record_filter += 1
                continue

            try:
                assignment = atlasio_districting_to_series(
                    record["districting"],
                    graph_nodes,
                    key_col=atlas_key_col,
                    node_id_col=node_id_col,
                    node_lookup=node_lookup,
                )

                comp = county_compliance_summary_for_assignment(
                    assignment,
                    graph_nodes,
                    node_id_col=node_id_col,
                    county_col=county_col,
                    max_districts_per_county=max_districts_per_county,
                )
                double_cross_counts.append(int(comp["double_cross_count"]))
                worst_splits.append(int(comp["worst_split"]))

                if filter_county_compliant and not bool(comp["compliant"]):
                    skipped_noncompliant += 1
                    continue

                district_df = aggregate_vote_cvap_by_district(
                    assignment,
                    graph_nodes,
                    node_id_col=node_id_col,
                    dem_col=dem_col,
                    rep_col=rep_col,
                    cvap_total_col=cvap_total_col,
                    black_cvap_col=black_cvap_col,
                )

                dem_vec, black_vec, ordered = ordered_rank_vectors(
                    district_df,
                    n_districts=n_districts,
                )

                dem_rows.append(dem_vec)
                black_rows.append(black_vec)

                for _, row in ordered.iterrows():
                    summary_rows.append(
                        {
                            "source": "ensemble",
                            "chain_id": chain_id,
                            "record_index": record_index,
                            "proposal": record.get("name", record_index),
                            "rank": int(row["rank"]),
                            "district": int(row["DISTRICT"]),
                            "dem_share": float(row["dem_share"]),
                            "black_cvap_share": float(row["black_cvap_share"]),
                            "cvap_total": float(row[cvap_total_col]),
                            "black_cvap": float(row[black_cvap_col]),
                            "double_cross_count": int(comp["double_cross_count"]),
                            "worst_split": int(comp["worst_split"]),
                        }
                    )

                used += 1

            except Exception as exc:
                failed += 1
                if failed <= 5:
                    print(f"  FAILED record {record_index}: {type(exc).__name__}: {exc}")

        compliance_rows.append(
            {
                "chain_id": chain_id,
                "chain_path": str(chain_file),
                "maps_seen": maps_seen,
                "used_compliant_maps": used,
                "skipped_by_record_filter": skipped_by_record_filter,
                "skipped_noncompliant": skipped_noncompliant,
                "failed_maps": failed,
                "compliant_fraction_after_record_filter": used
                / max(maps_seen - skipped_by_record_filter, 1),
                "avg_double_crosses": float(np.mean(double_cross_counts))
                if double_cross_counts
                else np.nan,
                "max_double_crosses": int(np.max(double_cross_counts))
                if double_cross_counts
                else 0,
                "worst_single_county_split": int(np.max(worst_splits))
                if worst_splits
                else 0,
                "filter_county_compliant": bool(filter_county_compliant),
                "max_districts_per_county": int(max_districts_per_county),
            }
        )

        print(f"  maps_seen:                 {maps_seen}")
        print(f"  used compliant maps:       {used}")
        print(f"  skipped by record filter:  {skipped_by_record_filter}")
        print(f"  skipped noncompliant maps: {skipped_noncompliant}")
        print(f"  failed maps:               {failed}")

    if not dem_rows:
        raise RuntimeError("No ensemble maps were usable after filtering.")

    return (
        np.vstack(dem_rows),
        np.vstack(black_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(compliance_rows),
    )


def read_block_assignment_excel(
    path: str | Path,
    *,
    block_col: str | None = None,
    district_col: str | None = None,
) -> pd.DataFrame:
    path = Path(path)
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
    graph_key_col: str,
    node_id_col: str,
) -> pd.Series:
    """
    Assign each graph node / precinct to the district receiving the largest
    total crosswalk population from its blocks.
    """
    merged = block_plan.merge(crosswalk, on="GEOID20", how="inner")

    if merged.empty:
        raise ValueError("No overlap between official block plan and crosswalk.")

    grouped = (
        merged.groupby(["PRECINCT_NAME", "DISTRICT"], as_index=False)["CW_WEIGHT"]
        .sum()
        .sort_values(["PRECINCT_NAME", "CW_WEIGHT", "DISTRICT"], ascending=[True, False, True])
    )

    precinct_assignment = grouped.drop_duplicates("PRECINCT_NAME", keep="first")[
        ["PRECINCT_NAME", "DISTRICT"]
    ].copy()

    node_lookup = graph_nodes[[node_id_col, graph_key_col]].copy()
    node_lookup[graph_key_col] = node_lookup[graph_key_col].map(normalize_token)

    matched = precinct_assignment.rename(columns={"PRECINCT_NAME": graph_key_col}).merge(
        node_lookup,
        on=graph_key_col,
        how="inner",
    )

    n_graph = graph_nodes[node_id_col].nunique()
    n_match = matched[node_id_col].nunique()
    print(f"  official precinct assignment matched {n_match:,} / {n_graph:,} graph nodes")

    if n_match == 0:
        raise ValueError(f"No official precinct labels matched graph column {graph_key_col!r}.")

    s = pd.Series(
        matched["DISTRICT"].astype(int).to_numpy(),
        index=matched[node_id_col].astype(int),
    )
    s.index.name = node_id_col

    return s[~s.index.duplicated(keep="first")].sort_index().astype(int)


def official_cvap_by_district_from_blocks(
    block_plan: pd.DataFrame,
    cvap: pd.DataFrame,
    *,
    cvap_total_col: str,
    black_cvap_col: str,
) -> pd.DataFrame:
    merged = block_plan.merge(
        cvap[["GEOID20", cvap_total_col, black_cvap_col]],
        on="GEOID20",
        how="inner",
    )

    if merged.empty:
        raise ValueError("No overlap between official block plan and CVAP blocks.")

    out = (
        merged.groupby("DISTRICT", as_index=False)[[cvap_total_col, black_cvap_col]]
        .sum()
    )

    out["black_cvap_share"] = out[black_cvap_col] / out[cvap_total_col].replace(0, np.nan)
    out["black_cvap_share"] = out["black_cvap_share"].fillna(0.0)

    return out


def build_official_vectors(
    *,
    label: str,
    baf_path: str | Path,
    crosswalk: pd.DataFrame,
    cvap: pd.DataFrame,
    graph_nodes: pd.DataFrame,
    official_key_col: str,
    node_id_col: str,
    dem_col: str,
    rep_col: str,
    cvap_total_col: str,
    black_cvap_col: str,
    n_districts: int,
    assignment_block_col: str | None,
    assignment_district_col: str | None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    block_plan = read_block_assignment_excel(
        baf_path,
        block_col=assignment_block_col,
        district_col=assignment_district_col,
    )

    assignment = official_precinct_assignment_from_blocks(
        block_plan,
        crosswalk,
        graph_nodes,
        graph_key_col=official_key_col,
        node_id_col=node_id_col,
    )

    vote_df = aggregate_vote_cvap_by_district(
        assignment,
        graph_nodes,
        node_id_col=node_id_col,
        dem_col=dem_col,
        rep_col=rep_col,
        cvap_total_col=cvap_total_col,
        black_cvap_col=black_cvap_col,
    )[["DISTRICT", "dem_share"]]

    cvap_df = official_cvap_by_district_from_blocks(
        block_plan,
        cvap,
        cvap_total_col=cvap_total_col,
        black_cvap_col=black_cvap_col,
    )[["DISTRICT", cvap_total_col, black_cvap_col, "black_cvap_share"]]

    district_df = vote_df.merge(cvap_df, on="DISTRICT", how="inner")

    dem_vec, black_vec, ordered = ordered_rank_vectors(
        district_df,
        n_districts=n_districts,
    )

    ordered["source"] = label
    ordered["chain_id"] = ""
    ordered["record_index"] = ""
    ordered["proposal"] = ""

    ordered = ordered.rename(
        columns={
            cvap_total_col: "cvap_total",
            black_cvap_col: "black_cvap",
        }
    )

    ordered = ordered[
        [
            "source",
            "chain_id",
            "record_index",
            "proposal",
            "rank",
            "DISTRICT",
            "dem_share",
            "black_cvap_share",
            "cvap_total",
            "black_cvap",
        ]
    ].rename(columns={"DISTRICT": "district"})

    return dem_vec, black_vec, ordered


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def rank_labels(n_districts: int) -> list[str]:
    labels = []
    mid = (n_districts + 1) // 2

    for i in range(1, n_districts + 1):
        if i == 1:
            labels.append("Most Rep")
        elif i == n_districts:
            labels.append("Most Dem")
        elif i == mid:
            labels.append("Median")
        elif i < mid:
            labels.append(str(i))
        else:
            labels.append(str(n_districts - i + 1))

    return labels


def draw_rank_panel(
    ax,
    ensemble_matrix: np.ndarray,
    official_vectors: dict[str, np.ndarray],
    *,
    ylabel: str,
    title: str,
    n_districts: int,
    percent_axis: bool = True,
    horizontal_line: float | None = None,
) -> None:
    positions = np.arange(1, n_districts + 1)

    data = [ensemble_matrix[:, i] for i in range(n_districts)]

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        boxprops={"facecolor": "#d9d9d9", "edgecolor": "#555555", "linewidth": 1.0},
        whiskerprops={"color": "#555555", "linewidth": 1.0},
        capprops={"color": "#555555", "linewidth": 1.0},
    )

    # Mean dots.
    means = np.nanmean(ensemble_matrix, axis=0)
    ax.scatter(
        positions,
        means,
        s=32,
        color="black",
        zorder=4,
        label="Ensemble mean",
    )

    for label, vec in official_vectors.items():
        style = official_plan_style(label, **PLAN_STYLES[label])
        ax.plot(
            positions,
            vec,
            marker=style["marker"],
            color=style["color"],
            linewidth=style["linewidth"],
            markersize=6,
            label=label,
            zorder=5,
        )

    if horizontal_line is not None:
        ax.axhline(horizontal_line, color="black", linewidth=1.2, linestyle="--", alpha=0.8)

    set_panel_title(ax, title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(rank_labels(n_districts))

    if percent_axis:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_two_panel(
    *,
    ensemble_dem: np.ndarray,
    ensemble_black: np.ndarray,
    official_dem: dict[str, np.ndarray],
    official_black: dict[str, np.ndarray],
    output: str | Path,
    title: str,
    black_label: str,
    n_districts: int,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    paper = paper_mode_enabled()

    draw_rank_panel(
        axes[0],
        ensemble_dem,
        official_dem,
        ylabel="Democratic two-party vote share",
        title=("Democratic vote share by district rank" if paper
               else "Panel A: Democratic vote share by district rank"),
        n_districts=n_districts,
        percent_axis=True,
        horizontal_line=0.5,
    )

    draw_rank_panel(
        axes[1],
        ensemble_black,
        official_black,
        ylabel=black_label,
        title=("Black CVAP share by the same district rank" if paper
               else "Panel B: Black CVAP share by the same district rank"),
        n_districts=n_districts,
        percent_axis=True,
        horizontal_line=0.5,
    )

    axes[1].set_xlabel("Districts ordered from most Republican to most Democratic")

    handles, labels = axes[0].get_legend_handles_labels()
    figure_legend(
        fig,
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(title, fontsize=18)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output = save_figure(fig, output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {output}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare ensemble and official plans using ranked Democratic vote share and Black CVAP share.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--chain-files", nargs="+", required=True)
    p.add_argument("--graph-json", required=True)
    p.add_argument("--cvap-file", required=True, help="Ohio CVAP zip or extracted CSV.")
    p.add_argument("--cvap-csv-member", default=None, help="CSV member inside the CVAP zip; inferred if omitted.")
    p.add_argument("--crosswalk-csv", required=True)

    p.add_argument("--enacted-baf", required=True)
    p.add_argument("--overturned-baf", required=True)
    p.add_argument("--remedial-baf", required=True)

    p.add_argument("--output", required=True)
    p.add_argument("--output-csv", default=None)

    p.add_argument(
        "--compliant-records-csv",
        default=None,
        help=(
            "Optional extra restriction to a previously saved list of records. "
            "The county-compliance filter is still applied by default."
        ),
    )
    p.add_argument(
        "--no-filter-compliant",
        action="store_true",
        help="Disable the built-in county-compliance filter and use all maps that pass any record CSV filter.",
    )
    p.add_argument("--county-col", default="COUNTY")
    p.add_argument("--max-districts-per-county", type=int, default=2)
    p.add_argument(
        "--max-maps-per-chain",
        type=int,
        default=None,
        help="Optional cap on map records read from each chain.",
    )
    p.add_argument("--compliance-summary-csv", default=None)

    p.add_argument("--atlas-key-col", default="NAME")
    p.add_argument("--official-key-col", default="ORIG_NAME")
    p.add_argument("--cvap-graph-key-col", default="ORIG_NAME")
    p.add_argument("--node-id-col", default="id")
    p.add_argument("--dem-col", default="G20PREDEM")
    p.add_argument("--rep-col", default="G20PREREP")

    p.add_argument("--cvap-total-col", default="CVAP_TOT23")
    p.add_argument(
        "--black-cvap-col",
        default="CVAP_BLK23",
        help="Use CVAP_BLK23 for Black alone-or-in-combination, or CVAP_BLA23 for Black alone.",
    )
    p.add_argument(
        "--black-label",
        default="Black CVAP share",
        help="Y-axis label for Panel B.",
    )

    p.add_argument("--n-districts", type=int, default=15)
    p.add_argument("--assignment-block-col", default=None)
    p.add_argument("--assignment-district-col", default=None)

    p.add_argument(
        "--title",
        default="Ohio Congressional Plans: Democratic vote share and Black CVAP",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    graph_nodes = read_graph_nodes(args.graph_json)
    cvap = read_cvap_file(args.cvap_file, cvap_csv_member=args.cvap_csv_member)
    crosswalk = read_crosswalk(args.crosswalk_csv)

    graph_nodes = add_precinct_cvap_to_graph(
        graph_nodes,
        cvap,
        crosswalk,
        graph_key_col=args.cvap_graph_key_col,
        cvap_total_col=args.cvap_total_col,
        black_cvap_col=args.black_cvap_col,
    )

    keep_filter = load_compliant_filter(args.compliant_records_csv)

    ensemble_dem, ensemble_black, ensemble_rows, compliance_summary = build_ensemble_matrices(
        args.chain_files,
        graph_nodes,
        atlas_key_col=args.atlas_key_col,
        node_id_col=args.node_id_col,
        county_col=args.county_col,
        dem_col=args.dem_col,
        rep_col=args.rep_col,
        cvap_total_col=args.cvap_total_col,
        black_cvap_col=args.black_cvap_col,
        n_districts=args.n_districts,
        compliant_filter=keep_filter,
        filter_county_compliant=not args.no_filter_compliant,
        max_districts_per_county=args.max_districts_per_county,
        max_maps_per_chain=args.max_maps_per_chain,
    )

    compliance_summary_path = (
        Path(args.compliance_summary_csv)
        if args.compliance_summary_csv is not None
        else Path(args.output).with_name(Path(args.output).stem + "_compliance_summary.csv")
    )
    compliance_summary_path.parent.mkdir(parents=True, exist_ok=True)
    compliance_summary.to_csv(compliance_summary_path, index=False)
    print(f"Wrote {compliance_summary_path}")

    print("Ensemble matrices:")
    print(f"  Democratic vote share: {ensemble_dem.shape}")
    print(f"  Black CVAP share:      {ensemble_black.shape}")

    official_dem = {}
    official_black = {}
    official_rows = []

    plan_paths = {
        "Enacted ('25)": args.enacted_baf,
        "Overturned ('21)": args.overturned_baf,
        "Remedial ('22)": args.remedial_baf,
    }

    for label, path in plan_paths.items():
        print(f"Reading official plan: {label}")
        dem_vec, black_vec, rows = build_official_vectors(
            label=label,
            baf_path=path,
            crosswalk=crosswalk,
            cvap=cvap,
            graph_nodes=graph_nodes,
            official_key_col=args.official_key_col,
            node_id_col=args.node_id_col,
            dem_col=args.dem_col,
            rep_col=args.rep_col,
            cvap_total_col=args.cvap_total_col,
            black_cvap_col=args.black_cvap_col,
            n_districts=args.n_districts,
            assignment_block_col=args.assignment_block_col,
            assignment_district_col=args.assignment_district_col,
        )
        official_dem[label] = dem_vec
        official_black[label] = black_vec
        official_rows.append(rows)

        print(f"  two most Democratic ranks:")
        for rank in [args.n_districts - 1, args.n_districts]:
            r = rows[rows["rank"] == rank].iloc[0]
            print(
                f"    rank {rank}: district {int(r['district'])}, "
                f"Dem={r['dem_share']:.2%}, Black CVAP={r['black_cvap_share']:.2%}, "
                f"Black CVAP count={r['black_cvap']:.0f}, total CVAP={r['cvap_total']:.0f}"
            )

    plot_two_panel(
        ensemble_dem=ensemble_dem,
        ensemble_black=ensemble_black,
        official_dem=official_dem,
        official_black=official_black,
        output=args.output,
        title=args.title,
        black_label=args.black_label,
        n_districts=args.n_districts,
    )

    if args.output_csv is not None:
        official_df = pd.concat(official_rows, ignore_index=True)
        all_rows = pd.concat([ensemble_rows, official_df], ignore_index=True)
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        all_rows.to_csv(output_csv, index=False)
        print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
