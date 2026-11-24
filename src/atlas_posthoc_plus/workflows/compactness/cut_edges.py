#!/usr/bin/env python3
"""

For a districting assignment xi on the precinct adjacency graph G=(V,E),

    cut_edges(xi) = #{(u,v) in E : xi(u) != xi(v)}.

This is a plan-level boundary-adjacency statistic. It is NOT the number of
edges cut in one sampled spanning tree/forest proposal.

my way to approach this:
1. Read in AtlasIO chain files.
2. Convert each saved districting to a graph-node assignment.
3. Apply the standard county compliance filter - as used in other scripts.
4. Count graph adjacency edges crossing district boundaries.
5. Convert each official block-assignment plan to a graph-node assignment using
   the block-to-precinct crosswalk (built earlier).
6. Count official-plan cut edges.
7. Plot compliant ensemble histogram with three official-plan reference lines overlayed.

### No dependency on atlas-posthoc, but do call under the root of oh-gerrymandering ###
atlas-posthoc-plus run cut-edges-compliant \
  --graph-json data/graph/OHpct20.json \
  --chain-files \
    outputs/chains/soft_constraint/congressional/run37_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run38_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run39_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run40_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
  --crosswalk-csv outputs/oh_block_to_precinct_crosswalk.csv \
  --enacted-baf "inputs/plans/October 31 2025 CD BAF.xlsx" \
  --overturned-baf "inputs/plans/SubSB258-Block-Assignment-File.xlsx" \
  --remedial-baf "inputs/plans/March 2 2022 CD BAF.xlsx" \
  --output-dir outputs/cut_edges_compliant_37_40 \
  --max-maps-per-chain 1001
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Any, Iterable

from atlas_posthoc_plus.paper import figure_legend, official_plan_style, save_figure

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from atlas_posthoc.io.graph_reader import adjacency_edges, read_graph_json

from atlas_posthoc_plus.io.atlas_records import iter_map_records
from atlas_posthoc_plus.io.chain_sweep import (
    cached_assignment_series,
    cached_compliance,
    configured_sweep,
)


PLAN_STYLES = {
    "Enacted ('25)": {"color": "#9467bd", "linestyle": "-", "linewidth": 2.4},
    "Overturned ('21)": {"color": "#2ca02c", "linestyle": "--", "linewidth": 2.4},
    "Remedial ('22)": {"color": "#d62728", "linestyle": "-.", "linewidth": 2.4},
}


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


def resolve_column(columns: Iterable[str], candidates: Iterable[str | None], *, label: str) -> str:
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

    m = re.search(r"\d+", s)
    if m:
        return int(m.group())

    raise ValueError(f"cannot convert district label to int: {value!r}")


def _node_attributes(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    if isinstance(record.get("attributes"), dict):
        return record["attributes"]
    if isinstance(record.get("properties"), dict):
        return record["properties"]
    return record


def read_graph(graph_json: str | Path) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    graph = read_graph_json(graph_json)
    nodes = pd.DataFrame(graph.nodes).copy()

    required = ["id", "NAME", "ORIG_NAME", "COUNTY"]
    missing = [c for c in required if c not in nodes.columns]
    if missing:
        raise ValueError(f"Graph node table is missing required columns {missing}. Available: {list(nodes.columns)}")

    nodes["id"] = pd.to_numeric(nodes["id"], errors="raise").astype(int)
    nodes["NAME"] = nodes["NAME"].map(normalize_token)
    nodes["ORIG_NAME"] = nodes["ORIG_NAME"].map(normalize_token)
    nodes["COUNTY"] = nodes["COUNTY"].map(normalize_token)

    edges = sorted(
        (int(source), int(target))
        for source, target, _ in adjacency_edges(graph)
    )
    if not edges:
        raise ValueError("No valid graph edges were read.")

    print(f"Loaded graph nodes: {len(nodes):,}")
    print(f"Loaded graph edges: {len(edges):,}")

    return nodes, edges


def open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("r")


def maybe_json_loads(x: Any) -> Any:
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


def iter_json_values_from_line(line: str):
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


def iter_atlasio_map_records(path: str | Path):
    yield from iter_map_records(path)


def chain_id_from_path(path: str | Path) -> str:
    name = Path(path).name
    m = re.search(r"(run\d+)", name)
    if m:
        return m.group(1)
    return Path(path).stem


def atlasio_districting_to_series(
    districting: Any,
    graph_nodes: pd.DataFrame,
    *,
    key_col: str = "NAME",
    node_id_col: str = "id",
) -> pd.Series:
    cached = cached_assignment_series(districting, node_id_col=node_id_col)
    if cached is not None:
        return cached.sort_index()

    if key_col not in graph_nodes.columns:
        raise ValueError(f"key_col={key_col!r} is not in graph nodes.")

    node_ids = graph_nodes[node_id_col].astype(int).tolist()
    node_key_to_id: dict[str, int] = {}

    for _, row in graph_nodes.iterrows():
        node_key_to_id[normalize_atlas_key(row[key_col])] = int(row[node_id_col])

    districting = maybe_json_loads(districting)

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

            if unmatched:
                raise ValueError(f"{len(unmatched)} AtlasIO keys did not match graph {key_col!r}. Examples: {unmatched[:5]}")
            if len(out) != len(graph_nodes):
                raise ValueError(f"Matched {len(out)} assignments but graph has {len(graph_nodes)} nodes.")

            s = pd.Series(out, dtype=int)
            s.index.name = node_id_col
            return s.sort_index()

        if len(districting) == len(graph_nodes) and not isinstance(districting[0], (list, tuple, dict, set)):
            s = pd.Series([coerce_district_label(x) for x in districting], index=node_ids, dtype=int)
            s.index.name = node_id_col
            return s.sort_index()

    if isinstance(districting, dict):
        graph_node_ids = {str(x) for x in node_ids}
        out: dict[int, int] = {}

        for raw_key, raw_dist in districting.items():
            key = normalize_atlas_key(raw_key)
            if str(key) in graph_node_ids:
                node_id = int(key)
            else:
                node_id = node_key_to_id.get(key)
            if node_id is None:
                raise ValueError(f"Districting key {raw_key!r} did not match graph nodes.")
            out[node_id] = coerce_district_label(raw_dist)

        s = pd.Series(out, dtype=int)
        s.index.name = node_id_col
        return s.sort_index()

    raise ValueError(f"Unsupported AtlasIO districting format: {type(districting)}")


def county_compliance_summary(
    assignment: pd.Series,
    graph_nodes: pd.DataFrame,
    *,
    node_id_col: str = "id",
    county_col: str = "COUNTY",
    max_districts_per_county: int = 2,
) -> dict[str, Any]:
    cached = cached_compliance(assignment)
    if cached is not None:
        return cached

    assign_df = assignment.rename("DISTRICT").rename_axis(node_id_col).reset_index()
    assign_df[node_id_col] = pd.to_numeric(assign_df[node_id_col], errors="coerce")

    nodes = graph_nodes[[node_id_col, county_col]].copy()
    nodes[node_id_col] = pd.to_numeric(nodes[node_id_col], errors="coerce")

    assigned = nodes.merge(assign_df, on=node_id_col, how="left")

    by_county = assigned.dropna(subset=["DISTRICT"]).groupby(county_col)["DISTRICT"].nunique()
    double_cross_count = int((by_county > max_districts_per_county).sum())
    worst_split = int(by_county.max()) if len(by_county) else 0

    assigned_nodes = int(assigned["DISTRICT"].notna().sum())
    missing_nodes = int(assigned["DISTRICT"].isna().sum())

    return {
        "compliant": double_cross_count == 0,
        "assigned_nodes": assigned_nodes,
        "missing_nodes": missing_nodes,
        "assigned_fraction": assigned_nodes / max(assigned_nodes + missing_nodes, 1),
        "double_cross_count": double_cross_count,
        "worst_split": worst_split,
    }


def count_cut_edges(assignment: pd.Series, edges: list[tuple[int, int]]) -> int:
    dist = assignment.to_dict()
    cut = 0
    missing = 0

    for u, v in edges:
        du = dist.get(u)
        dv = dist.get(v)
        if du is None or dv is None:
            missing += 1
            continue
        if du != dv:
            cut += 1

    if missing:
        print(f"WARNING: {missing} graph edges had at least one missing endpoint assignment.")

    return int(cut)


def build_compliant_cut_edge_values(
    chain_files: list[str | Path],
    graph_nodes: pd.DataFrame,
    edges: list[tuple[int, int]],
    *,
    atlas_key_col: str,
    node_id_col: str,
    county_col: str,
    max_districts_per_county: int,
    thin_step: int,
    max_maps_per_chain: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for chain_file in chain_files:
        chain_file = Path(chain_file)
        chain_id = chain_id_from_path(chain_file)
        action = "Analyzing shared assignments" if configured_sweep() else "Sweeping chain"
        print(f"\n{action} {chain_id}: {chain_file}")

        maps_seen = 0
        maps_used_after_thin = 0
        compliant_maps = 0
        invalid_maps = 0
        double_cross_counts: list[int] = []
        worst_splits: list[int] = []

        for record in iter_atlasio_map_records(chain_file):
            maps_seen += 1

            if thin_step > 1 and ((maps_seen - 1) % thin_step != 0):
                continue

            maps_used_after_thin += 1

            if max_maps_per_chain is not None and maps_used_after_thin > max_maps_per_chain:
                break

            try:
                assignment = atlasio_districting_to_series(
                    record["districting"],
                    graph_nodes,
                    key_col=atlas_key_col,
                    node_id_col=node_id_col,
                )

                comp = county_compliance_summary(
                    assignment,
                    graph_nodes,
                    node_id_col=node_id_col,
                    county_col=county_col,
                    max_districts_per_county=max_districts_per_county,
                )
                double_cross_counts.append(comp["double_cross_count"])
                worst_splits.append(comp["worst_split"])

                if not comp["compliant"]:
                    continue

                value_rows.append(
                    {
                        "chain_id": chain_id,
                        "map_index": maps_seen,
                        "map_index_after_thin": maps_used_after_thin,
                        "proposal": record.get("name", maps_seen),
                        "n_cut_edges": count_cut_edges(assignment, edges),
                    }
                )
                compliant_maps += 1

            except Exception as exc:
                invalid_maps += 1
                if invalid_maps <= 5:
                    print(f"  FAILED map {maps_seen}: {type(exc).__name__}: {exc}")

        summary_rows.append(
            {
                "chain_id": chain_id,
                "chain_path": str(chain_file),
                "maps_seen": maps_seen,
                "maps_used_after_thin": maps_used_after_thin,
                "compliant_maps": compliant_maps,
                "invalid_maps": invalid_maps,
                "compliant_fraction_after_thin": compliant_maps / max(maps_used_after_thin, 1),
                "avg_double_crosses": float(np.mean(double_cross_counts)) if double_cross_counts else np.nan,
                "max_double_crosses": int(np.max(double_cross_counts)) if double_cross_counts else 0,
                "worst_single_county_split": int(np.max(worst_splits)) if worst_splits else 0,
            }
        )

        print(f"  maps_seen={maps_seen}")
        print(f"  maps_used_after_thin={maps_used_after_thin}")
        print(f"  compliant_maps={compliant_maps}")
        print(f"  invalid_maps={invalid_maps}")

    values = pd.DataFrame(value_rows)
    summary = pd.DataFrame(summary_rows)

    if values.empty:
        raise RuntimeError("No compliant cut-edge values were produced.")

    return values, summary


def read_crosswalk(crosswalk_csv: str | Path) -> pd.DataFrame:
    cw = pd.read_csv(crosswalk_csv, dtype=str)

    block_col = resolve_column(cw.columns, ["GEOID20", "BLOCK", "Block", "GEOID"], label="crosswalk block")
    precinct_col = resolve_column(cw.columns, ["NAME", "ORIG_NAME", "PRECINCT"], label="crosswalk precinct")
    weight_col = resolve_column(cw.columns, ["POP20", "ALLOCATED_POP20", "weight", "share"], label="crosswalk weight")

    out = cw[[block_col, precinct_col, weight_col]].copy()
    out = out.rename(columns={block_col: "GEOID20", precinct_col: "PRECINCT_NAME", weight_col: "CW_WEIGHT"})
    out["GEOID20"] = out["GEOID20"].map(normalize_block_id)
    out["PRECINCT_NAME"] = out["PRECINCT_NAME"].map(normalize_token)
    out["CW_WEIGHT"] = pd.to_numeric(out["CW_WEIGHT"], errors="coerce").fillna(0.0)
    out = out[(out["GEOID20"] != "") & (out["PRECINCT_NAME"] != "")].copy()
    return out


def read_block_assignment_plan(
    path: str | Path,
    *,
    block_col: str | None = None,
    district_col: str | None = None,
) -> pd.DataFrame:
    path = Path(path)

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str)

    block_col = resolve_column(df.columns, [block_col, "BLOCK", "Block", "GEOID20", "GEOID", "block_geoid", "block_id"], label="block")
    district_col = resolve_column(df.columns, [district_col, "DistrictID", "DistrictID:1", "DISTRICT", "District", "CD", "CONG_DIST", "assignment"], label="district")

    out = df[[block_col, district_col]].copy()
    out[block_col] = out[block_col].map(normalize_block_id)
    out[district_col] = out[district_col].map(coerce_district_label)
    out = out[(out[block_col] != "") & out[district_col].notna()]
    out = out.drop_duplicates(subset=[block_col], keep="last")
    out = out.rename(columns={block_col: "GEOID20", district_col: "DISTRICT"})
    return out[["GEOID20", "DISTRICT"]]


def official_assignment_from_block_plan(
    block_plan: pd.DataFrame,
    crosswalk: pd.DataFrame,
    graph_nodes: pd.DataFrame,
    *,
    graph_key_col: str = "ORIG_NAME",
    node_id_col: str = "id",
) -> pd.Series:
    merged = block_plan.merge(crosswalk, on="GEOID20", how="inner")

    if merged.empty:
        raise ValueError("No overlap between official block plan and crosswalk.")

    grouped = (
        merged.groupby(["PRECINCT_NAME", "DISTRICT"], as_index=False)["CW_WEIGHT"]
        .sum()
        .sort_values(["PRECINCT_NAME", "CW_WEIGHT", "DISTRICT"], ascending=[True, False, True])
    )

    precinct_assignment = grouped.drop_duplicates(subset=["PRECINCT_NAME"], keep="first")[["PRECINCT_NAME", "DISTRICT"]].copy()

    if graph_key_col not in graph_nodes.columns:
        raise ValueError(f"graph_key_col={graph_key_col!r} is not in graph nodes.")

    node_lookup = graph_nodes[[node_id_col, graph_key_col]].copy()
    node_lookup[graph_key_col] = node_lookup[graph_key_col].map(normalize_token)

    matched = precinct_assignment.rename(columns={"PRECINCT_NAME": graph_key_col}).merge(
        node_lookup,
        on=graph_key_col,
        how="inner",
    )

    n_graph = graph_nodes[node_id_col].nunique()
    n_matched = matched[node_id_col].nunique()
    print(f"  official precinct assignment matched {n_matched:,} / {n_graph:,} graph nodes")

    if n_matched == 0:
        raise ValueError(f"Official plan precinct labels did not match graph column {graph_key_col!r}.")

    if n_matched < 0.95 * n_graph:
        print("  WARNING: official plan matched fewer than 95% of graph nodes.")

    s = pd.Series(matched["DISTRICT"].astype(int).to_numpy(), index=matched[node_id_col].astype(int))
    s.index.name = node_id_col
    return s[~s.index.duplicated(keep="first")].sort_index().astype(int)


def compute_official_cut_edges(
    *,
    plan_paths: dict[str, str | Path],
    crosswalk: pd.DataFrame,
    graph_nodes: pd.DataFrame,
    edges: list[tuple[int, int]],
    official_key_col: str,
    node_id_col: str,
    assignment_block_col: str | None,
    assignment_district_col: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for label, path in plan_paths.items():
        print(f"\nReading official plan: {label}")
        block_plan = read_block_assignment_plan(
            path,
            block_col=assignment_block_col,
            district_col=assignment_district_col,
        )

        assignment = official_assignment_from_block_plan(
            block_plan,
            crosswalk,
            graph_nodes,
            graph_key_col=official_key_col,
            node_id_col=node_id_col,
        )

        cut_edges = count_cut_edges(assignment, edges)
        rows.append({"plan": label, "path": str(path), "n_cut_edges": int(cut_edges)})
        print(f"  cut_edges={cut_edges:,}")

    return pd.DataFrame(rows)


def plot_cut_edge_histogram(
    *,
    ensemble_values: pd.DataFrame,
    official_values: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> None:
    x = ensemble_values["n_cut_edges"].to_numpy(dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        raise RuntimeError("No ensemble cut-edge values to plot.")

    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    if x.size > 1:
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25
        if iqr > 0:
            bin_width = 2 * iqr / (x.size ** (1 / 3))
            n_bins = int(np.ceil((x.max() - x.min()) / max(bin_width, 1)))
            n_bins = max(12, min(n_bins, 50))
        else:
            n_bins = 20
    else:
        n_bins = 20

    ax.hist(x, bins=n_bins, edgecolor="black", alpha=0.78, color="#b9c4d7")

    handles = []
    for _, row in official_values.iterrows():
        label = row["plan"]
        value = float(row["n_cut_edges"])
        style = official_plan_style(
            label, **PLAN_STYLES.get(label, {"color": "black", "linestyle": "-", "linewidth": 2.0})
        )
        ax.axvline(value, zorder=10, alpha=0.95, **style)
        handles.append(Line2D([0], [0], label=f"{label}: {int(value):,}", **style))

    ax.set_title(title)
    ax.set_xlabel("Number of precinct-adjacency edges crossing district boundaries")
    ax.set_ylabel("Count of compliant ensemble plans")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    figure_legend(
        ax, handles=handles, loc="upper right", frameon=True, fontsize=10, title="Reference plans",
    )

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = save_figure(fig, output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figure: {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build compliant ensemble cut-edge histogram and overlay official plans.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--graph-json", required=True)
    p.add_argument("--chain-files", nargs="+", required=True)
    p.add_argument("--crosswalk-csv", required=True)
    p.add_argument("--enacted-baf", required=True)
    p.add_argument("--overturned-baf", required=True)
    p.add_argument("--remedial-baf", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output", default=None)

    p.add_argument("--atlas-key-col", default="NAME")
    p.add_argument("--official-key-col", default="ORIG_NAME")
    p.add_argument("--node-id-col", default="id")
    p.add_argument("--county-col", default="COUNTY")

    p.add_argument("--max-districts-per-county", type=int, default=2)
    p.add_argument("--thin-step", type=int, default=1)
    p.add_argument("--max-maps-per-chain", type=int, default=None)

    p.add_argument("--assignment-block-col", default=None)
    p.add_argument("--assignment-district-col", default=None)
    p.add_argument("--title", default="Cut-edge distribution under compliant ensemble")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_plot = Path(args.output) if args.output else output_dir / "oh_cut_edge_histogram_with_refs_compliant.png"
    ensemble_csv = output_dir / "compliant_ensemble_cut_edges.csv"
    official_csv = output_dir / "official_plan_cut_edges.csv"
    compliance_csv = output_dir / "cut_edge_compliance_summary.csv"

    print("=" * 80)
    print("Compliant cut-edge histogram with reference plans")
    print("=" * 80)

    print("\n[1/4] Loading graph...")
    graph_nodes, edges = read_graph(args.graph_json)

    print("\n[2/4] Building compliant ensemble cut-edge values...")
    ensemble_values, compliance_summary = build_compliant_cut_edge_values(
        args.chain_files,
        graph_nodes,
        edges,
        atlas_key_col=args.atlas_key_col,
        node_id_col=args.node_id_col,
        county_col=args.county_col,
        max_districts_per_county=args.max_districts_per_county,
        thin_step=args.thin_step,
        max_maps_per_chain=args.max_maps_per_chain,
    )

    ensemble_values.to_csv(ensemble_csv, index=False)
    compliance_summary.to_csv(compliance_csv, index=False)
    print(f"Wrote ensemble values: {ensemble_csv}")
    print(f"Wrote compliance summary: {compliance_csv}")
    print(compliance_summary.to_string(index=False))

    print("\n[3/4] Computing reference-plan cut edges...")
    crosswalk = read_crosswalk(args.crosswalk_csv)

    official_values = compute_official_cut_edges(
        plan_paths={
            "Enacted ('25)": args.enacted_baf,
            "Overturned ('21)": args.overturned_baf,
            "Remedial ('22)": args.remedial_baf,
        },
        crosswalk=crosswalk,
        graph_nodes=graph_nodes,
        edges=edges,
        official_key_col=args.official_key_col,
        node_id_col=args.node_id_col,
        assignment_block_col=args.assignment_block_col,
        assignment_district_col=args.assignment_district_col,
    )

    official_values.to_csv(official_csv, index=False)
    print(f"Wrote official values: {official_csv}")
    print(official_values.to_string(index=False))

    print("\n[4/4] Plotting...")
    plot_cut_edge_histogram(
        ensemble_values=ensemble_values,
        official_values=official_values,
        output_path=output_plot,
        title=args.title,
    )

    print("\nDone.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
