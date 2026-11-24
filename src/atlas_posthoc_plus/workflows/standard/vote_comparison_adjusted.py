# Refactored version
# Call: 
"""
atlas-posthoc-plus run vote-comparison-adjusted \
  --graph-path data/graph/OHpct20.json \
  --chains \
    outputs/chains/soft_constraint/congressional/run13_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run14_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run15_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run16_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
  --chain-ids run13 run14 run15 run16 \
  --plan-manifest inputs/plan_manifest.csv \
  --crosswalk outputs/oh_block_to_precinct_crosswalk.csv \
  --output-dir outputs/13_16_adjusted \
  --external-vote-name-col ORIG_NAME \
  --pop-col POP20 \
  --max-maps-per-chain 1001
"""
# Has some weird behavior/tracebacks initially
# Debugged with OpenAI Codex, manually checked, shoud be error free

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import gzip
from collections import defaultdict
from pathlib import Path
from typing import Any

from atlas_posthoc_plus.paper import save_figure
from atlas_posthoc_plus.workflows.elections.public_vote_comparison import (
    plot_public_plan_comparison,
)
from atlas_posthoc_plus.workflows.elections.vote_comparison_plot import (
    POOLED_BOX_WIDTH,
    format_vote_comparison,
    plot_reference_plan,
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from atlas_posthoc.io.graph_reader import read_graph_json

from atlas_posthoc_plus.io.atlas_records import iter_map_records
from atlas_posthoc_plus.io.chain_sweep import (
    cached_assignment_series,
    cached_compliance,
    configured_sweep,
)

# atlas-posthoc
# Use the lower-level boxplot helper because the vote-share convenience wrapper
# does not expose ``box_width``.
from atlas_posthoc.plotting.marginals import (
    plot_ordered_boxplots_by_chain as plot_vote_share_boxplots,
)


_CHAIN_BOX_WIDTH = 0.10


# =============================================================================
# Helpers
# =============================================================================


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


def normalize_district(x: Any) -> int | None:
    if pd.isna(x):
        return None

    s = str(x).strip()
    if s == "":
        return None

    if re.fullmatch(r"\d+(\.0)?", s):
        return int(float(s))

    m = re.search(r"\d+", s)
    if m:
        return int(m.group())

    return None


def atlas_key_candidates(name: str) -> list[str]:
    return [
        json.dumps([name], separators=(",", ":")),
        f'["{name}"]',
        name,
    ]


def get_district_for_node(districting: dict[Any, Any], name: str) -> Any | None:
    for key in atlas_key_candidates(name):
        if key in districting:
            return districting[key]
    return None


def open_text_maybe_gzip(path: Path):
    """Open plain .jsonl or .jsonl.gz as text."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("rt")


def iter_json_values_from_line(line: str):
    """
    Robustly parse one physical line that may contain one JSON object
    or multiple concatenated JSON objects.
    """
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


def maybe_json_loads(x: Any) -> Any:
    """Handle occasional double-encoded JSON strings."""
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


def extract_districtings(obj: Any) -> list[dict[Any, Any]]:
    """
    Extract one or more districting dictionaries from an AtlasIO JSON value.

    Handles common cases:
      - {"districting": {...}}
      - raw districting dict
      - double-encoded JSON strings
      - lists containing map objects
    """
    obj = maybe_json_loads(obj)

    if isinstance(obj, list):
        out: list[dict[Any, Any]] = []
        for item in obj:
            out.extend(extract_districtings(item))
        return out

    if not isinstance(obj, dict):
        return []

    for key in ("districting", "Districting", "assignment", "assignments"):
        if key in obj:
            val = maybe_json_loads(obj[key])
            if isinstance(val, dict):
                return [val]

    # Heuristic for raw Atlas districting dict:
    # keys look like ["PRECINCT_NAME"], values are district labels.
    keys = list(obj.keys())
    vals = list(obj.values())

    if keys and vals:
        atlas_key_like = any(str(k).startswith('["') for k in keys[:20])
        simple_values = all(
            isinstance(v, (int, float, str, type(None))) for v in vals[:100]
        )

        if atlas_key_like and simple_values:
            return [obj]

    return []


def iter_districtings_from_chain(chain_path: Path):
    """
    Yield districting dictionaries from a possibly imperfect AtlasIO JSONL file.

    This bypasses Atlas.nextMap because Atlas.nextMap uses json.loads(line),
    which fails when a line contains multiple concatenated JSON objects.
    """
    with open_text_maybe_gzip(chain_path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                for obj in iter_json_values_from_line(line):
                    for districting in extract_districtings(obj):
                        yield line_no, districting
            except json.JSONDecodeError as e:
                print(
                    f"WARNING: skipping bad JSON at {chain_path}, "
                    f"line {line_no}: {e}"
                )
                continue


def infer_column(columns: list[str], candidates: list[str], fallback_index: int) -> str:
    lower_to_original = {c.lower(): c for c in columns}

    for cand in candidates:
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]

    for c in columns:
        low = c.lower()
        for cand in candidates:
            if cand.lower() in low:
                return c

    return columns[fallback_index]


def add_origname_adjusted_votes(
    nodes_by_name: dict[str, dict[str, Any]],
    graph_votes: pd.DataFrame,
    *,
    name_col: str,
    orig_name_col: str,
    pop_col: str,
    dem_col: str,
    rep_col: str,
    dem_adj_col: str,
    rep_adj_col: str,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    """Create node-level adjusted vote columns for split graph nodes.

    Some graph nodes are split versions of the same original precinct /
    merged-precinct unit. In those cases, the raw graph node rows can carry
    identical full vote totals under the same ORIG_NAME. Summing raw votes over
    NAME-level graph nodes would double-count.

    This function keeps the original vote total once per ORIG_NAME and allocates
    it across split NAME nodes by POP20 share. For non-split ORIG_NAME groups,
    the adjusted columns equal the raw vote columns.
    """
    required = [name_col, orig_name_col, pop_col, dem_col, rep_col]
    missing = [col for col in required if col not in graph_votes.columns]
    if missing:
        raise KeyError(
            "Cannot build adjusted vote columns. Missing graph column(s): "
            f"{missing}. Available columns: {graph_votes.columns.tolist()}"
        )

    g = graph_votes.copy()
    g[orig_name_col] = g[orig_name_col].astype(str)

    for col in [pop_col, dem_col, rep_col]:
        g[col] = pd.to_numeric(g[col], errors="coerce").fillna(0.0)

    group = (
        g.groupby(orig_name_col, as_index=False)
        .agg(
            _orig_pop=(pop_col, "sum"),
            _orig_dem=(dem_col, "first"),
            _orig_rep=(rep_col, "first"),
            _n_splits=(name_col, "size"),
        )
    )

    g = g.merge(group, on=orig_name_col, how="left")

    positive_pop = g["_orig_pop"] > 0
    g["_pop_share"] = np.nan
    g.loc[positive_pop, "_pop_share"] = (
        g.loc[positive_pop, pop_col] / g.loc[positive_pop, "_orig_pop"]
    )

    # Fallback for pathological zero-population groups: split votes equally
    # across the split graph rows. This should be rare for voting units.
    zero_pop = g["_pop_share"].isna()
    if zero_pop.any():
        g.loc[zero_pop, "_pop_share"] = 1.0 / g.loc[zero_pop, "_n_splits"]

    g[dem_adj_col] = g["_orig_dem"] * g["_pop_share"]
    g[rep_adj_col] = g["_orig_rep"] * g["_pop_share"]

    # Push adjusted vote values back into the node-attribute dictionary used
    # by AtlasIO map aggregation.
    for row in g[[name_col, dem_adj_col, rep_adj_col]].itertuples(index=False):
        node_name = str(getattr(row, name_col))
        if node_name in nodes_by_name:
            nodes_by_name[node_name][dem_adj_col] = float(getattr(row, dem_adj_col))
            nodes_by_name[node_name][rep_adj_col] = float(getattr(row, rep_adj_col))

    duplicate_groups = group[group["_n_splits"] > 1]
    summary = {
        "raw_dem_total": float(g[dem_col].sum()),
        "raw_rep_total": float(g[rep_col].sum()),
        "unique_origname_dem_total": float(group["_orig_dem"].sum()),
        "unique_origname_rep_total": float(group["_orig_rep"].sum()),
        "adjusted_dem_total": float(g[dem_adj_col].sum()),
        "adjusted_rep_total": float(g[rep_adj_col].sum()),
        "n_graph_rows": int(len(g)),
        "n_orignames": int(group.shape[0]),
        "n_split_orignames": int(duplicate_groups.shape[0]),
        "n_split_graph_rows": int(duplicate_groups["_n_splits"].sum())
        if not duplicate_groups.empty
        else 0,
    }

    g = g.drop(
        columns=["_orig_pop", "_orig_dem", "_orig_rep", "_n_splits", "_pop_share"]
    )

    return nodes_by_name, g, summary


# =============================================================================
# Graph loading
# =============================================================================


def _node_attributes(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}

    if isinstance(record.get("attributes"), dict):
        return record["attributes"]

    if isinstance(record.get("properties"), dict):
        return record["properties"]

    return record


def load_graph_nodes(
    graph_path: Path,
    *,
    name_col: str,
    county_col: str,
    pop_col: str,
    dem_col: str,
    rep_col: str,
    orig_name_col: str = "ORIG_NAME",
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    graph = read_graph_json(graph_path)
    records = [
        (str(node[name_col]), dict(node))
        for node in graph.nodes
    ]

    nodes_by_name: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for name, attrs in records:
        if county_col not in attrs:
            raise KeyError(f"Node {name!r} is missing {county_col!r}")
        if dem_col not in attrs:
            raise KeyError(f"Node {name!r} is missing {dem_col!r}")
        if rep_col not in attrs:
            raise KeyError(f"Node {name!r} is missing {rep_col!r}")
        if pop_col not in attrs:
            raise KeyError(f"Node {name!r} is missing {pop_col!r}")

        clean = dict(attrs)
        clean[name_col] = name
        clean[orig_name_col] = str(clean.get(orig_name_col, name))
        clean[dem_col] = float(clean[dem_col])
        clean[rep_col] = float(clean[rep_col])
        clean[pop_col] = float(clean[pop_col])

        nodes_by_name[name] = clean
        row = {
            name_col: name,
            county_col: clean[county_col],
            orig_name_col: clean[orig_name_col],
            pop_col: clean[pop_col],
            dem_col: clean[dem_col],
            rep_col: clean[rep_col],
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    if df[name_col].duplicated().any():
        raise ValueError(f"Duplicate graph node names found in {graph_path}")

    dem_adj_col = f"{dem_col}_ADJ"
    rep_adj_col = f"{rep_col}_ADJ"
    nodes_by_name, df, vote_adjustment_summary = add_origname_adjusted_votes(
        nodes_by_name,
        df,
        name_col=name_col,
        orig_name_col=orig_name_col,
        pop_col=pop_col,
        dem_col=dem_col,
        rep_col=rep_col,
        dem_adj_col=dem_adj_col,
        rep_adj_col=rep_adj_col,
    )

    return nodes_by_name, df, vote_adjustment_summary


# =============================================================================
# Compliance filtering + ordered vote vectors from AtlasIO chains
# =============================================================================


def county_compliance_summary_for_map(
    districting: dict[Any, Any],
    nodes_by_name: dict[str, dict[str, Any]],
    *,
    county_col: str,
    max_districts_per_county: int,
) -> dict[str, Any]:
    cached = cached_compliance(districting)
    if cached is not None:
        return cached

    county_to_districts: dict[str, set[Any]] = defaultdict(set)

    assigned_nodes = 0
    missing_nodes = 0

    for name, attrs in nodes_by_name.items():
        dist = get_district_for_node(districting, name)

        if dist is None:
            missing_nodes += 1
            continue

        assigned_nodes += 1
        county_to_districts[str(attrs[county_col])].add(dist)

    split_sizes = [len(v) for v in county_to_districts.values()]
    worst_split = max(split_sizes) if split_sizes else 0
    double_cross_count = sum(1 for s in split_sizes if s > max_districts_per_county)

    return {
        "compliant": double_cross_count == 0,
        "assigned_nodes": assigned_nodes,
        "missing_nodes": missing_nodes,
        "assigned_fraction": assigned_nodes / max(assigned_nodes + missing_nodes, 1),
        "double_cross_count": double_cross_count,
        "worst_split": worst_split,
    }


def ordered_vote_vector_for_atlas_map(
    districting: dict[Any, Any],
    nodes_by_name: dict[str, dict[str, Any]],
    *,
    dem_col: str,
    rep_col: str,
    n_districts: int,
) -> np.ndarray | None:
    dem_by_dist: dict[Any, float] = defaultdict(float)
    rep_by_dist: dict[Any, float] = defaultdict(float)

    cached = cached_assignment_series(districting)
    cached_districts = cached.to_dict() if cached is not None else None

    for name, attrs in nodes_by_name.items():
        dist = (
            cached_districts.get(int(attrs["id"]))
            if cached_districts is not None
            else get_district_for_node(districting, name)
        )
        if dist is None:
            continue

        dem_by_dist[dist] += float(attrs[dem_col])
        rep_by_dist[dist] += float(attrs[rep_col])

    districts = sorted(set(dem_by_dist) | set(rep_by_dist), key=lambda x: str(x))
    if len(districts) != n_districts:
        return None

    shares: list[float] = []
    for dist in districts:
        dem = dem_by_dist[dist]
        rep = rep_by_dist[dist]
        denom = dem + rep
        if denom <= 0:
            return None
        shares.append(dem / denom)

    return np.asarray(sorted(shares), dtype=float)


def build_compliant_vote_runs(
    chain_paths: list[Path],
    chain_ids: list[str],
    nodes_by_name: dict[str, dict[str, Any]],
    *,
    county_col: str,
    dem_col: str,
    rep_col: str,
    n_districts: int,
    max_districts_per_county: int,
    thin_step: int,
    max_maps_per_chain: int | None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    compliant_runs: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, Any]] = []

    for chain_id, chain_path in zip(chain_ids, chain_paths):
        if not chain_path.exists():
            raise FileNotFoundError(chain_path)

        action = "Analyzing shared assignments" if configured_sweep() else "Sweeping chain"
        print(f"\n{action} {chain_id}: {chain_path}")

        map_seen = 0
        map_used_after_thin = 0
        compliant_maps = 0
        invalid_vote_vectors = 0
        bad_json_lines = 0

        double_cross_counts: list[int] = []
        worst_splits: list[int] = []
        vote_vectors: list[np.ndarray] = []

        for current_map in iter_map_records(chain_path):
            map_seen += 1

            if thin_step > 1 and ((map_seen - 1) % thin_step != 0):
                continue

            map_used_after_thin += 1

            if (
                max_maps_per_chain is not None
                and map_used_after_thin > max_maps_per_chain
            ):
                break

            districting = current_map["districting"]

            comp = county_compliance_summary_for_map(
                districting,
                nodes_by_name,
                county_col=county_col,
                max_districts_per_county=max_districts_per_county,
            )

            double_cross_counts.append(comp["double_cross_count"])
            worst_splits.append(comp["worst_split"])

            if not comp["compliant"]:
                continue

            vec = ordered_vote_vector_for_atlas_map(
                districting,
                nodes_by_name,
                dem_col=dem_col,
                rep_col=rep_col,
                n_districts=n_districts,
            )

            if vec is None:
                invalid_vote_vectors += 1
                continue

            vote_vectors.append(vec)
            compliant_maps += 1

        arr = np.vstack(vote_vectors) if vote_vectors else np.empty((0, n_districts))
        compliant_runs[chain_id] = arr

        summary_rows.append(
            {
                "chain_id": chain_id,
                "chain_path": str(chain_path),
                "maps_seen": map_seen,
                "maps_used_after_thin": map_used_after_thin,
                "compliant_maps": compliant_maps,
                "invalid_vote_vectors": invalid_vote_vectors,
                "bad_json_lines": bad_json_lines,
                "compliant_fraction_after_thin": compliant_maps
                / max(map_used_after_thin, 1),
                "avg_double_crosses": float(np.mean(double_cross_counts))
                if double_cross_counts
                else np.nan,
                "max_double_crosses": int(np.max(double_cross_counts))
                if double_cross_counts
                else 0,
                "worst_single_county_split": int(np.max(worst_splits))
                if worst_splits
                else 0,
            }
        )

        print(f"  maps_seen={map_seen}")
        print(f"  maps_used_after_thin={map_used_after_thin}")
        print(f"  compliant_maps={compliant_maps}")
        print(f"  invalid_vote_vectors={invalid_vote_vectors}")
        print(f"  bad_json_lines={bad_json_lines}")
        print(f"  output_shape={arr.shape}")

    return compliant_runs, pd.DataFrame(summary_rows)


# =============================================================================
# External block plans
# =============================================================================


def read_plan_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)

    if "path" not in manifest.columns:
        raise ValueError("Plan manifest must contain a 'path' column")

    if "plan_id" not in manifest.columns:
        manifest["plan_id"] = [Path(p).stem for p in manifest["path"].astype(str)]

    if "plan_group" not in manifest.columns:
        manifest["plan_group"] = "External plan"

    if "plot_role" not in manifest.columns:
        manifest["plot_role"] = "highlight"

    return manifest


def read_block_assignment_plan(
    path: Path,
    *,
    geoid_col: str | None,
    district_col: str | None,
) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str)

    columns = list(df.columns)

    if not geoid_col:
        geoid_col = infer_column(
            columns,
            ["GEOID20", "GEOID", "BLOCKID", "BLOCK_GEOID", "census_block"],
            fallback_index=0,
        )

    if not district_col:
        district_col = infer_column(
            columns,
            ["DISTRICT", "District", "assigned_district", "CD", "assignment"],
            fallback_index=1,
        )

    out = pd.DataFrame(
        {
            "GEOID20": df[geoid_col].map(normalize_geoid),
            "district": df[district_col].map(normalize_district),
        }
    )

    out = out.dropna(subset=["district"])
    out["district"] = out["district"].astype(int)
    out = out[out["GEOID20"] != ""].copy()

    if out["GEOID20"].duplicated().any():
        dupes = out.loc[out["GEOID20"].duplicated(), "GEOID20"].head(10).tolist()
        raise ValueError(f"{path} has duplicate GEOID20 rows, examples: {dupes}")

    return out


def load_block_precinct_crosswalk(
    path: Path,
    *,
    block_col: str,
    precinct_col: str,
    weight_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={block_col: str, precinct_col: str})

    out = pd.DataFrame(
        {
            "GEOID20": df[block_col].map(normalize_geoid),
            "precinct_name": df[precinct_col].astype(str),
        }
    )

    if weight_col in df.columns:
        out["weight"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
    else:
        out["weight"] = 1.0

    out = out[(out["GEOID20"] != "") & (out["weight"] > 0)].copy()
    return out


def ordered_vote_vector_for_block_plan(
    assignment: pd.DataFrame,
    crosswalk: pd.DataFrame,
    graph_votes: pd.DataFrame,
    *,
    name_col: str,
    external_vote_name_col: str,
    dem_col: str,
    rep_col: str,
    n_districts: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    plan_blocks = set(assignment["GEOID20"])
    xwalk_blocks = set(crosswalk["GEOID20"])
    matched_plan_blocks = plan_blocks & xwalk_blocks

    joined = crosswalk.merge(assignment, on="GEOID20", how="left")
    assigned = joined.dropna(subset=["district"]).copy()
    assigned["district"] = assigned["district"].astype(int)

    precinct_district_weight = (
        assigned.groupby(["precinct_name", "district"], as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "precinct_district_weight"})
    )

    precinct_total_weight = (
        assigned.groupby("precinct_name", as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "precinct_assigned_weight"})
    )

    shares = precinct_district_weight.merge(
        precinct_total_weight,
        on="precinct_name",
        how="left",
    )
    shares["share"] = shares["precinct_district_weight"] / shares["precinct_assigned_weight"]

    if external_vote_name_col not in graph_votes.columns:
        raise KeyError(
            f"external_vote_name_col={external_vote_name_col!r} is not in graph_votes. "
            f"Available columns: {graph_votes.columns.tolist()}"
        )

    votes = graph_votes[[external_vote_name_col, dem_col, rep_col]].copy()
    votes = votes.rename(columns={external_vote_name_col: "precinct_name"})
    votes["precinct_name"] = votes["precinct_name"].astype(str)
    votes[dem_col] = pd.to_numeric(votes[dem_col], errors="coerce").fillna(0.0)
    votes[rep_col] = pd.to_numeric(votes[rep_col], errors="coerce").fillna(0.0)

    n_vote_rows_before = len(votes)
    n_vote_names_before = votes["precinct_name"].nunique()

    # Important: ORIG_NAME can be duplicated because graph nodes may be split
    # by enacted SD/HD suffixes. In this graph, duplicate ORIG_NAME rows carry
    # the same full original vote total. Do NOT sum them; take one copy.
    votes = (
        votes.sort_values("precinct_name")
        .groupby("precinct_name", as_index=False)[[dem_col, rep_col]]
        .first()
    )

    n_vote_rows_after = len(votes)

    if n_vote_rows_after < n_vote_rows_before:
        print(
            f"  Aggregated graph vote rows by {external_vote_name_col}: "
            f"{n_vote_rows_before} rows / {n_vote_names_before} names -> "
            f"{n_vote_rows_after} vote rows"
        )

    alloc = shares.merge(votes, on="precinct_name", how="left")
    missing_vote_precincts = int(alloc[dem_col].isna().sum())

    alloc = alloc.dropna(subset=[dem_col, rep_col]).copy()
    alloc["dem_alloc"] = alloc[dem_col].astype(float) * alloc["share"]
    alloc["rep_alloc"] = alloc[rep_col].astype(float) * alloc["share"]

    district_votes = (
        alloc.groupby("district", as_index=False)[["dem_alloc", "rep_alloc"]].sum()
    )

    observed_districts = sorted(district_votes["district"].astype(int).tolist())
    if len(observed_districts) != n_districts:
        raise ValueError(
            f"Plan produced {len(observed_districts)} districts, expected {n_districts}. "
            f"Observed districts: {observed_districts}"
        )

    district_votes["denom"] = district_votes["dem_alloc"] + district_votes["rep_alloc"]
    if (district_votes["denom"] <= 0).any():
        bad = district_votes.loc[district_votes["denom"] <= 0, "district"].tolist()
        raise ValueError(f"Districts with zero two-party vote: {bad}")

    district_votes["dem_share"] = district_votes["dem_alloc"] / district_votes["denom"]
    vec = np.sort(district_votes["dem_share"].to_numpy(dtype=float))

    qa = {
        "n_plan_blocks": len(plan_blocks),
        "n_matched_plan_blocks": len(matched_plan_blocks),
        "plan_block_match_fraction": len(matched_plan_blocks) / max(len(plan_blocks), 1),
        "missing_vote_precinct_rows_after_allocation": missing_vote_precincts,
        "allocated_dem_votes": float(district_votes["dem_alloc"].sum()),
        "allocated_rep_votes": float(district_votes["rep_alloc"].sum()),
        "external_vote_name_col": external_vote_name_col,
        "graph_vote_rows_before_name_aggregation": int(n_vote_rows_before),
        "graph_vote_rows_after_name_aggregation": int(n_vote_rows_after),
        "graph_vote_name_aggregation": "first",
        "min_dem_share": float(district_votes["dem_share"].min()),
        "max_dem_share": float(district_votes["dem_share"].max()),
    }

    return vec, qa


def build_external_plan_vectors(
    manifest: pd.DataFrame,
    *,
    manifest_path: Path,
    crosswalk: pd.DataFrame,
    graph_votes: pd.DataFrame,
    name_col: str,
    external_vote_name_col: str,
    dem_col: str,
    rep_col: str,
    n_districts: int,
    skip_errors: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    vector_rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []

    manifest_dir = manifest_path.parent

    for _, row in manifest.iterrows():
        raw_path = Path(str(row["path"]))
        plan_path = raw_path if raw_path.is_absolute() else manifest_dir / raw_path

        plan_id = str(row["plan_id"])
        plan_group = str(row["plan_group"])
        plot_role = str(row["plot_role"])

        geoid_col = str(row["geoid_col"]) if "geoid_col" in row and pd.notna(row["geoid_col"]) else None
        district_col = str(row["district_col"]) if "district_col" in row and pd.notna(row["district_col"]) else None

        print(f"\nReading external plan {plan_id}: {plan_path}")

        try:
            assignment = read_block_assignment_plan(
                plan_path,
                geoid_col=geoid_col,
                district_col=district_col,
            )

            vec, qa = ordered_vote_vector_for_block_plan(
                assignment,
                crosswalk,
                graph_votes,
                name_col=name_col,
                external_vote_name_col=external_vote_name_col,
                dem_col=dem_col,
                rep_col=rep_col,
                n_districts=n_districts,
            )
        except Exception as exc:
            if not skip_errors:
                raise
            print(f"  Skipping {plan_id}: {type(exc).__name__}: {exc}")
            qa_rows.append(
                {
                    "plan_id": plan_id,
                    "plan_group": plan_group,
                    "plot_role": plot_role,
                    "path": str(plan_path),
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        for rank, share in enumerate(vec, start=1):
            vector_rows.append(
                {
                    "plan_id": plan_id,
                    "plan_group": plan_group,
                    "plot_role": plot_role,
                    "rank": rank,
                    "dem_share": share,
                }
            )

        qa_rows.append(
            {
                "plan_id": plan_id,
                "plan_group": plan_group,
                "plot_role": plot_role,
                "path": str(plan_path),
                "status": "ok",
                "reason": "",
                **qa,
            }
        )

        print(f"  ordered vote vector: {np.round(vec, 4)}")

    return pd.DataFrame(vector_rows), pd.DataFrame(qa_rows)


def public_plan_manifest(public_plan_dir: str | Path) -> pd.DataFrame:
    source = Path(public_plan_dir)
    if not source.exists():
        raise FileNotFoundError(f"Public plan directory not found: {source}")
    paths = sorted(source.glob("*.csv"))
    if not paths:
        raise ValueError(f"No public plan CSV files found in {source}")
    return pd.DataFrame(
        [
            {
                "plan_id": path.stem,
                "plan_group": "Public submitted plans",
                "plot_role": "public",
                "path": str(path.resolve()),
            }
            for path in paths
        ]
    )


# =============================================================================
# Plotting using atlas-posthoc
# =============================================================================


def plot_comparison_using_atlas_posthoc(
    compliant_runs: dict[str, np.ndarray],
    external_long: pd.DataFrame,
    *,
    n_districts: int,
    output_path: Path,
    title: str,
    y_limits: tuple[float, float] | None,
    pool_chains: bool = False,
    ensemble_label: str = "Sampled plans",
    overlay_public_plans: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep the per-chain arrays for compliance and CSV outputs, but optionally
    # pool them into one mapping entry for the visualization. This removes the
    # chain identity from the box colors/legend without changing the analysis.
    plot_runs = compliant_runs
    if pool_chains:
        nonempty_runs = [arr for arr in compliant_runs.values() if arr.size]
        if not nonempty_runs:
            raise RuntimeError("Cannot pool chains: no compliant vote vectors remain.")
        plot_runs = {ensemble_label: np.vstack(nonempty_runs)}

    # atlas-posthoc draws the ensemble boxplot
    fig, ax = plot_vote_share_boxplots(
        plot_runs,
        observable_kind="vote_share",
        box_width=POOLED_BOX_WIDTH if pool_chains else _CHAIN_BOX_WIDTH,
        title=title,
        y_limits=y_limits,
        output_path=None,
    )

    x = np.arange(1, n_districts + 1)

    ax.axhline(0.5, color="black", linewidth=0.8)

    # Public plans: thin gray lines
    if not external_long.empty:
        public = external_long[external_long["plot_role"].str.lower() == "public"]

        if overlay_public_plans and not public.empty:
            ax.scatter(
                public["rank"],
                public["dem_share"],
                s=9,
                marker="o",
                color="#c62828",
                alpha=0.28,
                edgecolors="none",
                label=f"Public submitted plans ({public['plan_id'].nunique()})",
                zorder=1.5,
            )
        else:
            for _, g in public.groupby("plan_id"):
                g = g.sort_values("rank")
                ax.plot(
                    g["rank"],
                    g["dem_share"],
                    color="gray",
                    alpha=0.25,
                    linewidth=0.9,
                    zorder=2,
                )

            if not public.empty:
                ax.plot([], [], color="gray", alpha=0.7, linewidth=1.0, label="Public proposals")

    # Highlighted plans: colored lines
    highlighted = external_long[external_long["plot_role"].str.lower() != "public"]

    for idx, (plan_id, g) in enumerate(highlighted.groupby("plan_id")):
        g = g.sort_values("rank")
        label = str(g["plan_group"].iloc[0]) if "plan_group" in g.columns else plan_id
        plot_reference_plan(ax, g["rank"], g["dem_share"], label, idx)

    format_vote_comparison(ax, x)

    fig.tight_layout()
    output_path = save_figure(fig, output_path, dpi=300)
    plt.close(fig)

    print(f"\nWrote plot: {output_path}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use four AtlasIO chains, keep only county-compliant maps for the "
            "ensemble, then compare against external block-assignment plans."
        )
    )

    parser.add_argument("--graph-path", required=True, type=Path)
    parser.add_argument("--chains", required=True, nargs="+", type=Path)
    parser.add_argument("--chain-ids", nargs="*", default=None)

    parser.add_argument("--plan-manifest", required=True, type=Path)
    parser.add_argument("--crosswalk", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument("--n-districts", default=15, type=int)
    parser.add_argument("--name-col", default="NAME")
    parser.add_argument("--county-col", default="COUNTY")
    parser.add_argument("--pop-col", default="POP20")
    parser.add_argument("--dem-col", default="G20PREDEM")
    parser.add_argument("--rep-col", default="G20PREREP")

    parser.add_argument("--crosswalk-block-col", default="GEOID20")
    parser.add_argument("--crosswalk-precinct-col", default="NAME")
    parser.add_argument("--crosswalk-weight-col", default="POP20")

    parser.add_argument("--max-districts-per-county", default=2, type=int)
    parser.add_argument("--thin-step", default=1, type=int)
    parser.add_argument("--max-maps-per-chain", default=None, type=int)

    parser.add_argument(
        "--title",
        default="Ohio Congressional Plans Under 2020 Presidential Election",
    )
    parser.add_argument(
        "--pool-chains",
        action="store_true",
        help=(
            "Plot all compliant sampled plans as one pooled ensemble instead "
            "of showing a separate box for each chain."
        ),
    )
    parser.add_argument(
        "--ensemble-label",
        default="Sampled plans",
        help="Legend label for the pooled sampled-plan ensemble.",
    )
    parser.add_argument(
        "--overlay-public-plans",
        action="store_true",
        help="Overlay public plans and also write public-plan ordered vote-share boxes.",
    )
    parser.add_argument(
        "--public-output",
        type=Path,
        default=None,
        help="Public boxplot output; defaults to <output-dir>/oh_public_plans_ordered_vote_shares.png.",
    )
    parser.add_argument(
        "--public-plan-dir",
        type=Path,
        default=None,
        help="Directory of public submitted block-assignment CSV files.",
    )

    parser.add_argument(
        "--y-min",
        default=0.20,
        type=float,
        help="y-axis minimum in vote-share units",
    )
    parser.add_argument(
        "--y-max",
        default=0.82,
        type=float,
        help="y-axis maximum in vote-share units",
    )

    parser.add_argument(
        "--external-vote-name-col",
        default="ORIG_NAME",
        help=(
            "Graph vote column used to match external block-plan crosswalk names. "
            "Keep --name-col as NAME for AtlasIO chain lookup; use ORIG_NAME here "
            "when graph NAME has __SD__/__HD__ suffixes."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.chain_ids is None or len(args.chain_ids) == 0:
        chain_ids = [f"chain_{i + 1}" for i in range(len(args.chains))]
    else:
        chain_ids = list(args.chain_ids)

    if len(chain_ids) != len(args.chains):
        raise ValueError("--chain-ids must have the same length as --chains")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Ohio plan comparison with compliant AtlasIO ensemble")
    print("=" * 80)

    print("\n[1/5] Loading graph nodes...")
    nodes_by_name, graph_votes, vote_adjustment_summary = load_graph_nodes(
        args.graph_path,
        name_col=args.name_col,
        county_col=args.county_col,
        pop_col=args.pop_col,
        dem_col=args.dem_col,
        rep_col=args.rep_col,
        orig_name_col=args.external_vote_name_col,
    )
    print(f"Loaded {len(nodes_by_name)} graph nodes.")

    ensemble_dem_col = f"{args.dem_col}_ADJ"
    ensemble_rep_col = f"{args.rep_col}_ADJ"
    print("Vote adjustment summary:")
    for key, value in vote_adjustment_summary.items():
        print(f"  {key}: {value}")
    print(
        "Using adjusted node-level vote columns for the AtlasIO ensemble: "
        f"{ensemble_dem_col}, {ensemble_rep_col}"
    )
    print(
        "Using one-copy-per-ORIG_NAME raw vote totals for external block plans: "
        f"{args.dem_col}, {args.rep_col}"
    )

    print("\n[2/5] Building compliant ensemble vote runs...")
    compliant_runs, compliance_summary = build_compliant_vote_runs(
        args.chains,
        chain_ids,
        nodes_by_name,
        county_col=args.county_col,
        dem_col=ensemble_dem_col,
        rep_col=ensemble_rep_col,
        n_districts=args.n_districts,
        max_districts_per_county=args.max_districts_per_county,
        thin_step=args.thin_step,
        max_maps_per_chain=args.max_maps_per_chain,
    )

    total_compliant = sum(arr.shape[0] for arr in compliant_runs.values())
    if total_compliant == 0:
        raise RuntimeError("No compliant maps were retained from the chains.")

    compliance_summary_path = args.output_dir / "chain_compliance_summary.csv"
    compliance_summary.to_csv(compliance_summary_path, index=False)
    print(f"Wrote compliance summary: {compliance_summary_path}")
    print(compliance_summary.to_string(index=False))

    print("\n[3/5] Loading external plans and crosswalk...")
    crosswalk = load_block_precinct_crosswalk(
        args.crosswalk,
        block_col=args.crosswalk_block_col,
        precinct_col=args.crosswalk_precinct_col,
        weight_col=args.crosswalk_weight_col,
    )
    manifest = read_plan_manifest(args.plan_manifest)

    external_long, external_qa = build_external_plan_vectors(
        manifest,
        manifest_path=args.plan_manifest,
        crosswalk=crosswalk,
        graph_votes=graph_votes,
        name_col=args.name_col,
        external_vote_name_col=args.external_vote_name_col,
        dem_col=args.dem_col,
        rep_col=args.rep_col,
        n_districts=args.n_districts,
    )

    if args.overlay_public_plans:
        if args.public_plan_dir is None:
            raise ValueError("--overlay-public-plans requires --public-plan-dir")
        public_long, public_qa = build_external_plan_vectors(
            public_plan_manifest(args.public_plan_dir),
            manifest_path=args.public_plan_dir / "public-plan-manifest.csv",
            crosswalk=crosswalk,
            graph_votes=graph_votes,
            name_col=args.name_col,
            external_vote_name_col=args.external_vote_name_col,
            dem_col=args.dem_col,
            rep_col=args.rep_col,
            n_districts=args.n_districts,
            skip_errors=True,
        )
        external_long = pd.concat([external_long, public_long], ignore_index=True)
        external_qa = pd.concat([external_qa, public_qa], ignore_index=True)
        print(
            f"Accepted public plans for vote overlay: "
            f"{public_long['plan_id'].nunique() if not public_long.empty else 0}"
        )

    external_vectors_path = args.output_dir / "external_plan_ordered_vote_vectors.csv"
    external_qa_path = args.output_dir / "external_plan_qa.csv"

    external_long.to_csv(external_vectors_path, index=False)
    external_qa.to_csv(external_qa_path, index=False)

    print(f"Wrote external plan vectors: {external_vectors_path}")
    print(f"Wrote external plan QA: {external_qa_path}")

    print("\n[4/5] Saving compliant ensemble vote vectors...")
    ensemble_rows: list[dict[str, Any]] = []
    for chain_id, arr in compliant_runs.items():
        for map_idx in range(arr.shape[0]):
            for rank in range(arr.shape[1]):
                ensemble_rows.append(
                    {
                        "chain_id": chain_id,
                        "map_index_within_compliant_subset": map_idx,
                        "rank": rank + 1,
                        "dem_share": arr[map_idx, rank],
                    }
                )
    ensemble_df = pd.DataFrame(ensemble_rows)
    ensemble_vectors_path = args.output_dir / "compliant_ensemble_ordered_vote_vectors.csv"
    ensemble_df.to_csv(ensemble_vectors_path, index=False)
    print(f"Wrote compliant ensemble vote vectors: {ensemble_vectors_path}")

    print("\n[5/5] Plotting comparison using atlas-posthoc...")
    plot_path = args.output_dir / "oh_compliant_ensemble_vs_plans.png"
    plot_comparison_using_atlas_posthoc(
        compliant_runs,
        external_long,
        n_districts=args.n_districts,
        output_path=plot_path,
        title=args.title,
        y_limits=(args.y_min, args.y_max),
        pool_chains=args.pool_chains,
        ensemble_label=args.ensemble_label,
        overlay_public_plans=args.overlay_public_plans,
    )

    if args.overlay_public_plans:
        public_plot_path = plot_public_plan_comparison(
            external_long,
            n_districts=args.n_districts,
            output_path=(
                args.public_output
                if args.public_output is not None
                else args.output_dir / "oh_public_plans_ordered_vote_shares.png"
            ),
            title=args.title,
            y_limits=(args.y_min, args.y_max),
        )
        print(f"Wrote public-plan ordered vote-share plot: {public_plot_path}")

    print("\nDone.")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
