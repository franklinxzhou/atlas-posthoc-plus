#!/usr/bin/env python3
"""
district_rank_municipal_histograms.py

Run from the ct-gerrymandering / oh-gerrymandering project root.

This script makes 15 PNG files, one for each district rank after sorting
districts within each plan by Democratic vote share from least Democratic to
most Democratic.

Each PNG has two panels:
  left:  density histogram of district-level ousted MCD population
  right: fraction histogram of district-level split-MCD count

The ensemble comes from AtlasIO JSONL chain files. The three official plans
come from block-assignment spreadsheets plus a block-to-precinct crosswalk.

Default Ohio assumptions:
  graph key for AtlasIO chains:      NAME
  graph key for official plans:      ORIG_NAME
  crosswalk columns:                 GEOID20, NAME, POP20
  BAF columns are inferred:           BLOCK/Block/GEOID20 and DistrictID/DistrictID:1

Use this bash script
### After installing atlas-posthoc and run the script under the root of oh-gerrymandering ###
-------
atlas-posthoc-plus run district-rank-municipal \
  --chain-files \
    outputs/chains/soft_constraint/congressional/run37_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run38_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run39_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run40_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
  --graph-json data/graph/OHpct20.json \
  --crosswalk-csv outputs/oh_block_to_precinct_crosswalk.csv \
  --enacted-baf "inputs/plans/October 31 2025 CD BAF.xlsx" \
  --overturned-baf "inputs/plans/SubSB258-Block-Assignment-File.xlsx" \
  --remedial-baf "inputs/plans/March 2 2022 CD BAF.xlsx" \
  --output-dir outputs/oh_district_rank_municipal_histograms
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
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
from atlas_posthoc_plus.io.graph import read_graph_frame
from atlas_posthoc_plus.workflows.municipal.geography import (
    MunicipalGeography,
    add_municipality_arguments,
    geography_from_args,
)


# ---------------------------------------------------------------------
# Column / key normalization
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
    Normalize AtlasIO keys.

    AtlasIO districting entries in these chains look like:
        {'["TROY 1-B_MIAMI__SD5__HD80"]': 2}

    The graph NAME is:
        TROY 1-B_MIAMI__SD5__HD80

    This strips the JSON-list wrapper.
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
    """
    Fuzzy column resolver.

    This intentionally handles:
      DistrictID vs DistrictID:1
      BLOCK vs Block
      GEOID20 vs GEOID_20
    """
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
        raise ValueError(f"non-integer float district label: {value}")

    s = normalize_token(value)
    if s.lstrip("-").isdigit():
        return int(s)

    raise ValueError(f"cannot convert district label to int: {value!r}")


# ---------------------------------------------------------------------
# AtlasIO / graph parsing
# ---------------------------------------------------------------------
def open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("r")


def iter_atlasio_map_records(
    path: str | Path,
    *,
    skip_bad_json: bool = True,
) -> Iterable[dict[str, Any]]:
    """
    Yield actual AtlasIO map records only.

    These chain files contain dict metadata/header records and then saved-map
    records. Actual saved-map records have a "districting" field.
    """
    yield from iter_map_records(path, skip_bad_json=skip_bad_json)


def read_graph_nodes(
    graph_json: str | Path,
    geography: MunicipalGeography | None = None,
) -> pd.DataFrame:
    df = read_graph_frame(graph_json).copy()

    geography = geography or MunicipalGeography()
    required = ["id", "COUNTY", "POP20", "G20PREDEM", "G20PREREP"]
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

    geography.validate_graph_nodes(df)

    return df


def atlasio_districting_to_series(
    districting: Any,
    graph_nodes: pd.DataFrame,
    *,
    key_col: str,
    node_id_col: str = "id",
) -> pd.Series:
    """
    Convert AtlasIO districting payload to Series indexed by graph node id.

    Supported AtlasIO formats:
      1. list of singleton dicts:
           [{'["PRECINCT_KEY"]': 15}, ...]
      2. list of district labels in graph-node order:
           [15, 2, 3, ...]
      3. dict keyed by graph node id or graph key
    """
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
        raise ValueError(
            f"Duplicate graph keys found for {key_col!r}. "
            f"Examples: {duplicate_keys[:5]}"
        )

    # Format used by run37--run40:
    # [{'["PRECINCT_KEY"]': district}, ...]
    if isinstance(districting, list) and districting:
        if all(isinstance(x, dict) and len(x) == 1 for x in districting):
            out: dict[int, int] = {}
            unmatched: list[str] = []

            for item in districting:
                key, district = next(iter(item.items()))
                key = normalize_atlas_key(key)
                node_id = node_key_to_id.get(key)

                if node_id is None:
                    unmatched.append(key)
                    continue

                out[node_id] = coerce_district_label(district)

            if unmatched:
                raise ValueError(
                    f"{len(unmatched)} AtlasIO districting keys did not match graph "
                    f"{key_col!r} keys. Examples: {unmatched[:5]}"
                )

            if len(out) != len(graph_nodes):
                raise ValueError(
                    f"Matched {len(out)} assignments but graph has {len(graph_nodes)} nodes."
                )

            s = pd.Series(out, dtype=int)
            s.index.name = node_id_col
            return s.sort_index()

        # Simple node-order vector.
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

    # Dict keyed by node id or graph key.
    if isinstance(districting, dict):
        graph_node_ids = {str(x) for x in node_ids}
        out: dict[int, int] = {}
        unmatched: list[str] = []

        for raw_key, district in districting.items():
            key = normalize_atlas_key(raw_key)

            if str(key) in graph_node_ids:
                node_id = int(key)
            else:
                node_id = node_key_to_id.get(key)

            if node_id is None:
                unmatched.append(key)
                continue

            out[node_id] = coerce_district_label(district)

        if unmatched:
            raise ValueError(
                f"{len(unmatched)} districting keys did not match graph nodes. "
                f"Examples: {unmatched[:5]}"
            )

        s = pd.Series(out, dtype=int)
        s.index.name = node_id_col
        return s.sort_index()

    raise ValueError(
        f"Unsupported AtlasIO districting format: {type(districting)}"
    )


# ---------------------------------------------------------------------
# Official plan conversion
# ---------------------------------------------------------------------
def read_block_assignment_excel(
    path: str | Path,
    *,
    block_col: str | None = None,
    district_col: str | None = None,
) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_excel(path, dtype=str)

    if df.empty:
        raise ValueError(f"Official plan spreadsheet is empty: {path}")

    block_col = resolve_column(
        df.columns,
        [block_col, "BLOCK", "Block", "GEOID20", "GEOID", "block_id"],
        label="block",
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
        ],
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


def block_plan_to_precinct_assignment(
    block_plan: pd.DataFrame,
    crosswalk_csv: str | Path,
    graph_nodes: pd.DataFrame,
    *,
    graph_key_col: str = "ORIG_NAME",
    node_id_col: str = "id",
    crosswalk_block_col: str | None = None,
    crosswalk_precinct_col: str | None = None,
    crosswalk_weight_col: str | None = None,
) -> pd.Series:
    """
    Convert block-level BAF assignment to precinct-level assignment.

    The default crosswalk is the file you inspected:
        GEOID20, NAME, POP20

    For each precinct, the assigned district is the district receiving the
    largest total crosswalk weight from that precinct's blocks.
    """
    cw = pd.read_csv(crosswalk_csv, dtype=str)

    block_col = resolve_column(
        cw.columns,
        [crosswalk_block_col, "GEOID20", "BLOCK", "Block", "GEOID"],
        label="crosswalk block",
    )
    precinct_col = resolve_column(
        cw.columns,
        [crosswalk_precinct_col, "NAME", "ORIG_NAME", "precinct", "PRECINCT"],
        label="crosswalk precinct",
    )
    weight_col = resolve_column(
        cw.columns,
        [crosswalk_weight_col, "POP20", "ALLOCATED_POP20", "weight", "share"],
        label="crosswalk weight",
    )

    if graph_key_col not in graph_nodes.columns:
        raise ValueError(
            f"Official graph key {graph_key_col!r} is not in graph nodes. "
            f"Available columns: {list(graph_nodes.columns)}"
        )

    print("  Resolved crosswalk columns:")
    print(f"    block:    {block_col}")
    print(f"    precinct: {precinct_col}")
    print(f"    weight:   {weight_col}")

    cw = cw[[block_col, precinct_col, weight_col]].copy()
    cw[block_col] = cw[block_col].map(normalize_block_id)
    cw[precinct_col] = cw[precinct_col].map(normalize_token)
    cw[weight_col] = pd.to_numeric(cw[weight_col], errors="coerce").fillna(0.0)

    merged = block_plan.merge(
        cw,
        left_on="GEOID20",
        right_on=block_col,
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "No overlap between official block plan and crosswalk. "
            "Check the block GEOID column."
        )

    grouped = (
        merged.groupby([precinct_col, "DISTRICT"], as_index=False)[weight_col]
        .sum()
        .sort_values(
            [precinct_col, weight_col, "DISTRICT"],
            ascending=[True, False, True],
        )
    )

    precinct_assignment = grouped.drop_duplicates(
        subset=[precinct_col],
        keep="first",
    )[[precinct_col, "DISTRICT"]].copy()

    node_lookup = graph_nodes[[node_id_col, graph_key_col]].copy()
    node_lookup[graph_key_col] = node_lookup[graph_key_col].map(normalize_token)

    precinct_assignment = precinct_assignment.rename(
        columns={precinct_col: graph_key_col}
    )

    matched = precinct_assignment.merge(
        node_lookup,
        on=graph_key_col,
        how="inner",
    )

    n_graph = graph_nodes[node_id_col].nunique()
    n_matched = matched[node_id_col].nunique()
    print(f"  Matched official plan to {n_matched:,} / {n_graph:,} graph nodes")

    if n_matched == 0:
        raise ValueError(
            f"Official plan precinct labels did not match graph column {graph_key_col!r}."
        )

    if n_matched < 0.95 * n_graph:
        print(
            "  WARNING: official plan matched fewer than 95% of graph nodes. "
            "Check --official-key-col / crosswalk precinct labels."
        )

    s = pd.Series(
        matched["DISTRICT"].astype(int).to_numpy(),
        index=matched[node_id_col].astype(int),
    )
    s.index.name = node_id_col

    return s[~s.index.duplicated(keep="first")].sort_index().astype(int)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def compute_vote_share_by_district(
    assignment: pd.Series,
    graph_nodes: pd.DataFrame,
    *,
    node_id_col: str,
    dem_col: str,
    rep_col: str,
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
    graph_nodes: pd.DataFrame,
    *,
    node_id_col: str,
    pop_col: str,
    municipality_cols: list[str],
    geography: MunicipalGeography | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    District-level MCD split metrics.

    split_count[d]:
        number of split MCDs that touch district d.

    ousted_pop[d]:
        for split MCDs, population in district d outside that MCD's plurality
        district. The plurality district is the district containing the largest
        population piece of that MCD.
    """
    if geography is not None:
        return geography.district_burden(
            assignment,
            graph_nodes,
            node_id_col=node_id_col,
            pop_col=pop_col,
        )

    df = graph_nodes[[node_id_col, pop_col] + municipality_cols].merge(
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


def order_by_vote_share(
    vote_share: pd.Series,
    split_count: pd.Series,
    ousted_pop: pd.Series,
    *,
    n_districts: int,
) -> tuple[np.ndarray, np.ndarray]:
    districts = sorted(
        set(vote_share.index) | set(split_count.index) | set(ousted_pop.index)
    )

    df = pd.DataFrame(
        {
            "DISTRICT": districts,
            "vote_share": [vote_share.get(d, 0.0) for d in districts],
            "split_count": [split_count.get(d, 0.0) for d in districts],
            "ousted_pop": [ousted_pop.get(d, 0.0) for d in districts],
        }
    )

    df = df.sort_values(["vote_share", "DISTRICT"]).reset_index(drop=True)

    if len(df) != n_districts:
        raise ValueError(
            f"Expected {n_districts} districts after assignment, found {len(df)}."
        )

    return df["split_count"].to_numpy(float), df["ousted_pop"].to_numpy(float)


def ordered_metrics_for_assignment(
    assignment: pd.Series,
    graph_nodes: pd.DataFrame,
    *,
    n_districts: int,
    node_id_col: str,
    dem_col: str,
    rep_col: str,
    pop_col: str,
    municipality_cols: list[str],
    geography: MunicipalGeography | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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

    return order_by_vote_share(
        vote_share,
        split_count,
        ousted_pop,
        n_districts=n_districts,
    )


# ---------------------------------------------------------------------
# Build matrices
# ---------------------------------------------------------------------
def build_ensemble_rank_matrices(
    chain_files: list[str | Path],
    graph_nodes: pd.DataFrame,
    *,
    atlas_key_col: str,
    n_districts: int,
    node_id_col: str,
    dem_col: str,
    rep_col: str,
    pop_col: str,
    municipality_cols: list[str],
    geography: MunicipalGeography | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    split_rows: list[np.ndarray] = []
    ousted_rows: list[np.ndarray] = []

    for chain_file in chain_files:
        print(f"Reading chain: {chain_file}")
        used = 0
        failed = 0

        for record in iter_atlasio_map_records(chain_file):
            try:
                assignment = atlasio_districting_to_series(
                    record["districting"],
                    graph_nodes,
                    key_col=atlas_key_col,
                    node_id_col=node_id_col,
                )

                split_vec, ousted_vec = ordered_metrics_for_assignment(
                    assignment,
                    graph_nodes,
                    n_districts=n_districts,
                    node_id_col=node_id_col,
                    dem_col=dem_col,
                    rep_col=rep_col,
                    pop_col=pop_col,
                    municipality_cols=municipality_cols,
                    geography=geography,
                )

                split_rows.append(split_vec)
                ousted_rows.append(ousted_vec)
                used += 1

            except Exception as exc:
                failed += 1
                if failed <= 5:
                    print(f"  FAILED record: {type(exc).__name__}: {exc}")

        print(f"  used plans: {used}")
        print(f"  failed records: {failed}")

    if not split_rows:
        raise RuntimeError("No usable ensemble plans were parsed.")

    return np.vstack(split_rows), np.vstack(ousted_rows)


def build_official_rank_vectors(
    *,
    baf_path: str | Path,
    crosswalk_csv: str | Path,
    graph_nodes: pd.DataFrame,
    official_key_col: str,
    n_districts: int,
    node_id_col: str,
    dem_col: str,
    rep_col: str,
    pop_col: str,
    municipality_cols: list[str],
    geography: MunicipalGeography | None = None,
    block_nodes: list[dict[str, Any]] | None = None,
    assignment_block_col: str | None,
    assignment_district_col: str | None,
    crosswalk_block_col: str | None,
    crosswalk_precinct_col: str | None,
    crosswalk_weight_col: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    block_plan = read_block_assignment_excel(
        baf_path,
        block_col=assignment_block_col,
        district_col=assignment_district_col,
    )

    assignment = block_plan_to_precinct_assignment(
        block_plan,
        crosswalk_csv,
        graph_nodes,
        graph_key_col=official_key_col,
        node_id_col=node_id_col,
        crosswalk_block_col=crosswalk_block_col,
        crosswalk_precinct_col=crosswalk_precinct_col,
        crosswalk_weight_col=crosswalk_weight_col,
    )

    vote_share = compute_vote_share_by_district(
        assignment,
        graph_nodes,
        node_id_col=node_id_col,
        dem_col=dem_col,
        rep_col=rep_col,
    )
    if geography is not None and block_nodes is not None:
        block_assignment = pd.Series(
            block_plan["DISTRICT"].to_numpy(),
            index=block_plan["GEOID20"].to_numpy(),
        )
        split_count, ousted_pop = geography.district_burden(
            block_assignment,
            pd.DataFrame(block_nodes),
            node_id_col="id",
            pop_col="POP20",
            require_complete_assignment=True,
        )
        return order_by_vote_share(
            vote_share, split_count, ousted_pop, n_districts=n_districts
        )

    return ordered_metrics_for_assignment(
        assignment,
        graph_nodes,
        n_districts=n_districts,
        node_id_col=node_id_col,
        dem_col=dem_col,
        rep_col=rep_col,
        pop_col=pop_col,
        municipality_cols=municipality_cols,
        geography=geography,
    )


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
    height = ymax - ymin

    handles = []

    for idx, (label, style) in enumerate(official_plan_styles(PLAN_STYLES)):
        x = values[label]
        ax.axvline(x, zorder=10, alpha=0.95, **style)

        # Stagger labels to reduce collisions.
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

    # Left: ousted population density.
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
    ax.set_ylabel("Density of plans")
    add_reference_marks(ax, official_ousted)

    # Right: split-MCD count fraction.
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
    ax.set_ylabel("Fraction of plans")
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
            "Make 15 separate municipal-split histogram PNGs, one per "
            "Democratic vote-share district rank."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--chain-files", nargs="+", required=True)
    p.add_argument("--graph-json", required=True)
    p.add_argument("--crosswalk-csv", required=True)

    p.add_argument("--enacted-baf", required=True)
    p.add_argument("--overturned-baf", required=True)
    p.add_argument("--remedial-baf", required=True)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--block-metadata", required=True)
    p.add_argument("--block-col", default=None)
    p.add_argument("--county-col-block", default=None)
    p.add_argument("--pop-col-block", default=None)
    p.add_argument("--block-mcd-col", default=None)
    p.add_argument("--mcd-shapefile", default=None)
    p.add_argument("--mcd-shape-county-col", default=None)
    p.add_argument("--mcd-shape-mcd-col", default=None)
    add_municipality_arguments(p)

    # Defaults match your Ohio files.
    p.add_argument("--atlas-key-col", default="NAME")
    p.add_argument("--official-key-col", default="ORIG_NAME")
    p.add_argument("--node-id-col", default="id")
    p.add_argument("--dem-col", default="G20PREDEM")
    p.add_argument("--rep-col", default="G20PREREP")
    p.add_argument("--pop-col", default="POP20")
    p.add_argument("--municipality-cols", nargs="+", default=["COUNTY", "MCD"])
    p.add_argument("--n-districts", type=int, default=15)

    # Optional overrides only; usually omit these.
    p.add_argument("--assignment-block-col", default=None)
    p.add_argument("--assignment-district-col", default=None)
    p.add_argument("--crosswalk-block-col", default=None)
    p.add_argument("--crosswalk-precinct-col", default=None)
    p.add_argument("--crosswalk-weight-col", default=None)

    p.add_argument("--title-prefix", default="Ohio Congressional")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    geography = geography_from_args(args, graph_json=args.graph_json)
    graph_nodes = read_graph_nodes(args.graph_json, geography)
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

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    cache_npz = cache_dir / "district_rank_municipal_matrices.npz" if cache_dir else None
    cache_json = cache_dir / "official_plan_rank_vectors.json" if cache_dir else None
    cache_metadata = cache_dir / "municipal-geography.json" if cache_dir else None

    if cache_npz is not None and cache_npz.exists() and cache_json is not None and cache_json.exists():
        if cache_metadata is None or not cache_metadata.exists():
            raise ValueError("Municipal cache has no geography metadata; recompute it")
        geography.assert_compatible_metadata(json.loads(cache_metadata.read_text()))
        print(f"Loading cached matrices from {cache_npz}")
        cached = np.load(cache_npz)
        ensemble_split = cached["ensemble_split"]
        ensemble_ousted = cached["ensemble_ousted"]

        with cache_json.open("r") as f:
            official_raw = json.load(f)

        official_split = {
            label: np.asarray(values["split"], dtype=float)
            for label, values in official_raw.items()
        }
        official_ousted = {
            label: np.asarray(values["ousted"], dtype=float)
            for label, values in official_raw.items()
        }

    else:
        ensemble_split, ensemble_ousted = build_ensemble_rank_matrices(
            args.chain_files,
            graph_nodes,
            atlas_key_col=args.atlas_key_col,
            n_districts=args.n_districts,
            node_id_col=args.node_id_col,
            dem_col=args.dem_col,
            rep_col=args.rep_col,
            pop_col=args.pop_col,
            municipality_cols=args.municipality_cols,
            geography=geography,
        )

        print("Ensemble matrices built:")
        print(f"  split shape:  {ensemble_split.shape}")
        print(f"  ousted shape: {ensemble_ousted.shape}")

        plan_paths = {
            "Enacted ('25)": args.enacted_baf,
            "Overturned ('21)": args.overturned_baf,
            "Remedial ('22)": args.remedial_baf,
        }

        official_split = {}
        official_ousted = {}

        for label, path in plan_paths.items():
            print(f"Reading official plan: {label}")
            split_vec, ousted_vec = build_official_rank_vectors(
                baf_path=path,
                crosswalk_csv=args.crosswalk_csv,
                graph_nodes=graph_nodes,
                official_key_col=args.official_key_col,
                n_districts=args.n_districts,
                node_id_col=args.node_id_col,
                dem_col=args.dem_col,
                rep_col=args.rep_col,
                pop_col=args.pop_col,
                municipality_cols=args.municipality_cols,
                geography=geography,
                block_nodes=block_nodes,
                assignment_block_col=args.assignment_block_col,
                assignment_district_col=args.assignment_district_col,
                crosswalk_block_col=args.crosswalk_block_col,
                crosswalk_precinct_col=args.crosswalk_precinct_col,
                crosswalk_weight_col=args.crosswalk_weight_col,
            )
            official_split[label] = split_vec
            official_ousted[label] = ousted_vec
            print(f"  split:  {split_vec}")
            print(f"  ousted: {ousted_vec}")

        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_npz,
                ensemble_split=ensemble_split,
                ensemble_ousted=ensemble_ousted,
            )
            with cache_json.open("w") as f:
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
            print(f"Wrote cache: {cache_npz}")
            print(f"Wrote cache: {cache_json}")
            geography.write_metadata(cache_metadata)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geography.write_metadata(output_dir / "municipal-geography.json")

    for rank0 in range(args.n_districts):
        rank_idx = rank0 + 1

        official_split_marks = {
            label: float(vec[rank0])
            for label, vec in official_split.items()
        }
        official_ousted_marks = {
            label: float(vec[rank0])
            for label, vec in official_ousted.items()
        }

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

        print(f"Wrote {output_path}")

    print("Done.")


if __name__ == "__main__":
    main()
