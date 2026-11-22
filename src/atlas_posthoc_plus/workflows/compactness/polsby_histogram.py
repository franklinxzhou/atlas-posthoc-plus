#!/usr/bin/env python3
"""Plot a Polsby-Popper histogram with optional public and official overlays."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any

from atlas_posthoc_plus.paper import (
    figure_legend,
    official_plan_label,
    official_plan_style,
    paper_mode_enabled,
    save_figure,
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PUBLIC_PLAN_EXTENSIONS = {".csv", ".xls", ".xlsx"}

STAT_TO_KEY = {
    "average": "average_polsby_popper",
    "mean": "average_polsby_popper",
    "min": "minimum_polsby_popper",
    "minimum": "minimum_polsby_popper",
    "max": "maximum_polsby_popper",
    "maximum": "maximum_polsby_popper",
}

STAT_TO_SUMMARY_COL = {
    "average": "average",
    "mean": "average",
    "min": "minimum",
    "minimum": "minimum",
    "max": "maximum",
    "maximum": "maximum",
}

def _read_json(path: Path) -> Any:
    with path.open("r") as stream:
        return json.load(stream)


def _trace_records(raw: Any, *, path: Path) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict):
        for key in ("traces", "chains", "chain_traces"):
            value = raw.get(key)
            if isinstance(value, list):
                records = value
                break
        else:
            raise ValueError(
                f"Could not find Polsby-Popper trace records in {path}. "
                f"Available keys: {list(raw.keys())}"
            )
    else:
        raise TypeError(f"Unsupported Polsby-Popper trace JSON type in {path}: {type(raw)}")

    if not all(isinstance(record, dict) for record in records):
        raise TypeError(f"All Polsby-Popper trace records must be dicts: {path}")
    return records


def load_ensemble_scores(path: str | Path, stat: str, burnin: int) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Ensemble Polsby-Popper trace JSON not found: {source}")

    key = STAT_TO_KEY[stat]
    records = _trace_records(_read_json(source), path=source)

    rows: list[dict[str, Any]] = []
    for record in records:
        chain_id = str(record.get("chain_id", "unknown_chain"))

        if key not in record:
            raise KeyError(
                f"Could not find key {key!r} in chain record {chain_id!r}. "
                f"Available keys are: {sorted(record.keys())}"
            )

        scores = list(record[key])
        proposals = list(record.get("proposals", range(len(scores))))

        if len(proposals) != len(scores):
            raise ValueError(
                f"Chain {chain_id!r} has {len(proposals)} proposals but "
                f"{len(scores)} scores for {key!r}."
            )

        for proposal, score in zip(proposals, scores, strict=True):
            if proposal < burnin:
                continue
            if score is None:
                continue
            score_float = float(score)
            if not np.isfinite(score_float):
                continue
            rows.append(
                {
                    "chain_id": chain_id,
                    "proposal": proposal,
                    "score": score_float,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No ensemble scores remain after filtering/burn-in.")

    return df


def load_public_scores(
    path: str | Path | None, stat: str, *, metric_name: str = "Polsby-Popper"
) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["plan_id", "score"])

    source = Path(path)
    if not source.exists():
        print(f"Public {metric_name} score CSV not found; skipping: {source}")
        return pd.DataFrame(columns=["plan_id", "score"])

    df = pd.read_csv(source)
    stat_col = STAT_TO_SUMMARY_COL[stat]
    if stat_col in df.columns:
        score_col = stat_col
    elif "score" in df.columns:
        score_col = "score"
    else:
        raise ValueError(
            f"Public scores CSV must contain either {stat_col!r} or 'score': {source}"
        )

    if "plan_id" not in df.columns:
        df["plan_id"] = [f"public_{i + 1}" for i in range(len(df))]

    out = df[["plan_id", score_col]].copy()
    out = out.rename(columns={score_col: "score"})
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    return out.dropna(subset=["score"])


def discover_public_plan_files(public_plan_dir: str | Path) -> list[Path]:
    source = Path(public_plan_dir)
    if not source.exists():
        raise FileNotFoundError(f"Public plan directory not found: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"Public plan path is not a directory: {source}")

    files = sorted(
        p
        for p in source.rglob("*")
        if p.is_file()
        and p.suffix.lower() in PUBLIC_PLAN_EXTENSIONS
        and not p.name.startswith("~$")
    )
    if not files:
        raise ValueError(f"No public plan assignment files found in {source}")
    return files


def _format_public_summary(summary_df: pd.DataFrame, stat: str) -> pd.DataFrame:
    score_col = STAT_TO_SUMMARY_COL[stat]
    required = {"plan_id", "average", "minimum", "maximum"}
    missing = sorted(required - set(summary_df.columns))
    if missing:
        raise ValueError(f"Public plan summary is missing columns: {missing}")

    out = summary_df.copy()
    out["score"] = pd.to_numeric(out[score_col], errors="coerce")

    first_cols = [
        "plan_id",
        "score",
        "average",
        "minimum",
        "maximum",
        "n_districts",
        "matched_blocks",
        "total_blocks_in_shapefile",
        "matched_block_share",
        "matched_graph_nodes",
        "total_graph_nodes",
        "matched_graph_node_share",
        "score_method",
        "source_path",
    ]
    ordered_cols = [c for c in first_cols if c in out.columns]
    ordered_cols.extend(c for c in out.columns if c not in first_cols)
    return out[ordered_cols]


def compute_public_scores_from_blocks(
    *,
    public_plan_dir: str | Path,
    stat: str,
    block_shapefile: str | Path | None,
    block_col: str | None,
    projected_crs: str,
    assignment_block_col: str | None,
    assignment_district_col: str | None,
    expected_districts: int,
    public_scores_out: str | Path | None,
) -> pd.DataFrame:
    if block_shapefile is None:
        raise ValueError("Block-based public plan scoring requires --block-shapefile.")

    from atlas_posthoc_plus.workflows.compactness.official_polsby import (
        compute_plan_from_blocks,
        prepare_blocks,
    )

    files = discover_public_plan_files(public_plan_dir)
    print(f"Discovered {len(files):,} public plan assignment files in {public_plan_dir}")
    print("Scoring public plans from block geometry. This can be slow.")

    blocks = prepare_blocks(
        block_shapefile=block_shapefile,
        block_col=block_col,
        projected_crs=projected_crs,
    )

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for path in files:
        plan_id = path.stem
        try:
            summary, _district_rows = compute_plan_from_blocks(
                blocks=blocks,
                assignment_path=path,
                plan_label=plan_id,
                assignment_block_col=assignment_block_col,
                assignment_district_col=assignment_district_col,
                expected_districts=expected_districts,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append(
                {
                    "plan_id": plan_id,
                    "source_path": str(path),
                    "error": message,
                }
            )
            if len(failures) <= 10:
                print(f"  FAILED public plan {plan_id}: {message}")
            continue

        summary["plan_id"] = plan_id
        summary["source_path"] = str(path)
        summaries.append(summary)

    if not summaries:
        raise ValueError(
            f"No public plans could be scored from {public_plan_dir}. "
            "Check the assignment columns and district count."
        )

    summary_df = _format_public_summary(pd.DataFrame(summaries), stat=stat)

    if public_scores_out is not None:
        destination = Path(public_scores_out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(destination, index=False)
        print(f"Wrote public plan Polsby-Popper score cache to {destination}")

    print(
        f"Scored {len(summary_df):,}/{len(files):,} public plan files for "
        f"stat={stat!r}."
    )
    if failures:
        print(f"Skipped {len(failures):,} public plan files that could not be scored.")

    return summary_df[["plan_id", "score"]].copy()


def _node_to_district_with_missing_nodes(
    assignment: pd.Series,
    graph_nodes: pd.DataFrame,
    *,
    node_id_col: str,
) -> dict[Any, int | None]:
    out: dict[Any, int | None] = {}
    assigned = {int(node): int(dist) for node, dist in assignment.items()}

    for node_id in graph_nodes[node_id_col]:
        node_key = int(node_id)
        out[node_key] = assigned.get(node_key)

    return out


def compute_public_scores_from_graph(
    *,
    public_plan_dir: str | Path,
    stat: str,
    graph_json: str | Path | None,
    crosswalk_csv: str | Path | None,
    graph_key_col: str,
    node_id_col: str,
    area_col: str,
    border_length_col: str,
    edge_length_col: str | None,
    assignment_block_col: str | None,
    assignment_district_col: str | None,
    expected_districts: int,
    public_scores_out: str | Path | None,
) -> pd.DataFrame:
    if graph_json is None:
        raise ValueError("Graph-based public plan scoring requires --graph-json.")
    if crosswalk_csv is None:
        raise ValueError("Graph-based public plan scoring requires --crosswalk-csv.")

    from atlas_posthoc.audit.enacted import compute_reference_compactness
    from atlas_posthoc.io.graph_reader import read_graph_json
    from atlas_posthoc.observables.polsby_popper import (
        isoperimetric_to_polsby_popper,
        summarize_polsby_popper_scores,
    )
    from atlas_posthoc_plus.workflows.compactness.cut_edges import (
        official_assignment_from_block_plan,
        read_block_assignment_plan,
        read_crosswalk,
    )

    files = discover_public_plan_files(public_plan_dir)
    print(f"Discovered {len(files):,} public plan assignment files in {public_plan_dir}")
    print("Scoring public plans from graph geometry and block-to-precinct crosswalk.")

    graph_data = read_graph_json(graph_json)
    graph_nodes = pd.DataFrame(graph_data.nodes)
    if node_id_col not in graph_nodes.columns:
        raise ValueError(f"Graph nodes do not contain node id column {node_id_col!r}.")

    crosswalk = read_crosswalk(crosswalk_csv)

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for path in files:
        plan_id = path.stem
        try:
            block_plan = read_block_assignment_plan(
                path,
                block_col=assignment_block_col,
                district_col=assignment_district_col,
            )
            block_districts = int(block_plan["DISTRICT"].nunique())
            if block_districts != expected_districts:
                raise ValueError(
                    f"expected {expected_districts} districts, found {block_districts}"
                )

            with contextlib.redirect_stdout(io.StringIO()):
                assignment = official_assignment_from_block_plan(
                    block_plan,
                    crosswalk,
                    graph_nodes,
                    graph_key_col=graph_key_col,
                    node_id_col=node_id_col,
                )
            node_to_district = _node_to_district_with_missing_nodes(
                assignment,
                graph_nodes,
                node_id_col=node_id_col,
            )
            graph_districts = {
                int(dist) for dist in node_to_district.values() if dist is not None
            }
            if len(graph_districts) != expected_districts:
                raise ValueError(
                    f"expected {expected_districts} graph districts, "
                    f"found {len(graph_districts)}"
                )

            result = compute_reference_compactness(
                graph_data,
                node_to_district=node_to_district,
                node_id_col=node_id_col,
                area_col=area_col,
                border_length_col=border_length_col,
                edge_length_col=edge_length_col,
                n_districts=expected_districts,
                sort="none",
            )
            isoperimetric_scores = [row.score for row in result.district_totals]
            pp_scores = isoperimetric_to_polsby_popper(
                isoperimetric_scores,
                convention="p2_over_area",
                invalid_policy="raise",
                clip=False,
            )
            summary = summarize_polsby_popper_scores(pp_scores).to_dict()
            summary.update(
                {
                    "plan_id": plan_id,
                    "plan": plan_id,
                    "source_path": str(path),
                    "n_districts": int(result.n_districts_observed),
                    "matched_graph_nodes": int(assignment.index.nunique()),
                    "total_graph_nodes": int(len(graph_nodes)),
                    "matched_graph_node_share": float(assignment.index.nunique() / len(graph_nodes)),
                    "score_method": "graph",
                }
            )
            summaries.append(summary)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append(
                {
                    "plan_id": plan_id,
                    "source_path": str(path),
                    "error": message,
                }
            )
            if len(failures) <= 10:
                print(f"  FAILED public plan {plan_id}: {message}")
            continue

    if not summaries:
        raise ValueError(
            f"No public plans could be scored from {public_plan_dir}. "
            "Check the assignment columns, crosswalk, graph key, and district count."
        )

    summary_df = _format_public_summary(pd.DataFrame(summaries), stat=stat)

    if public_scores_out is not None:
        destination = Path(public_scores_out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(destination, index=False)
        print(f"Wrote public plan Polsby-Popper score cache to {destination}")

    print(
        f"Scored {len(summary_df):,}/{len(files):,} public plan files for "
        f"stat={stat!r}."
    )
    if "matched_graph_node_share" in summary_df.columns:
        match = pd.to_numeric(summary_df["matched_graph_node_share"], errors="coerce")
        print(
            "Matched graph-node share among scored public plans: "
            f"{match.min():.2%}–{match.max():.2%}."
        )
    if failures:
        print(f"Skipped {len(failures):,} public plan files that could not be scored.")

    return summary_df[["plan_id", "score"]].copy()


def load_or_compute_public_scores(args: argparse.Namespace) -> pd.DataFrame:
    if args.public_scores_csv is not None and Path(args.public_scores_csv).exists():
        return load_public_scores(args.public_scores_csv, stat=args.stat)

    if args.public_scores_csv is not None:
        print(f"Public Polsby-Popper score CSV not found: {args.public_scores_csv}")

    if (
        args.public_scores_out is not None
        and Path(args.public_scores_out).exists()
        and not args.refresh_public_scores
    ):
        print(f"Reusing public Polsby-Popper score cache: {args.public_scores_out}")
        return load_public_scores(args.public_scores_out, stat=args.stat)

    if args.public_plan_dir is not None:
        if args.public_score_method == "graph":
            return compute_public_scores_from_graph(
                public_plan_dir=args.public_plan_dir,
                stat=args.stat,
                graph_json=args.graph_json,
                crosswalk_csv=args.crosswalk_csv,
                graph_key_col=args.graph_key_col,
                node_id_col=args.node_id_col,
                area_col=args.area_col,
                border_length_col=args.border_length_col,
                edge_length_col=args.edge_length_col,
                assignment_block_col=args.assignment_block_col,
                assignment_district_col=args.assignment_district_col,
                expected_districts=args.expected_districts,
                public_scores_out=args.public_scores_out,
            )

        return compute_public_scores_from_blocks(
            public_plan_dir=args.public_plan_dir,
            stat=args.stat,
            block_shapefile=args.block_shapefile,
            block_col=args.block_col,
            projected_crs=args.projected_crs,
            assignment_block_col=args.assignment_block_col,
            assignment_district_col=args.assignment_district_col,
            expected_districts=args.expected_districts,
            public_scores_out=args.public_scores_out,
        )

    return pd.DataFrame(columns=["plan_id", "score"])


def _resolve_official_label_col(df: pd.DataFrame) -> str:
    for candidate in ("label", "plan", "name", "plan_label"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Official scores CSV must contain one of: label, plan, name, plan_label."
    )


def load_official_scores(
    path: str | Path | None, stat: str, *, metric_name: str = "Polsby-Popper"
) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["label", "score"])

    source = Path(path)
    if not source.exists():
        print(f"Official {metric_name} score CSV not found; skipping: {source}")
        return pd.DataFrame(columns=["label", "score"])

    df = pd.read_csv(source)
    label_col = _resolve_official_label_col(df)

    if "score" in df.columns:
        score_col = "score"
    else:
        score_col = STAT_TO_SUMMARY_COL[stat]
        if score_col not in df.columns:
            raise ValueError(
                f"Official scores CSV must contain 'score' or {score_col!r} "
                f"for stat={stat!r}. Available columns: {list(df.columns)}"
            )

    out = df[[label_col, score_col]].copy()
    out = out.rename(columns={label_col: "label", score_col: "score"})
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    return out.dropna(subset=["score"])


def add_reference_rug(
    ax,
    public_df: pd.DataFrame,
    official_df: pd.DataFrame,
    ymax: float,
    *,
    public_label: str,
    official_label: str,
) -> None:
    if not public_df.empty:
        ax.scatter(
            public_df["score"],
            np.zeros(len(public_df)),
            marker="o",
            s=24,
            color="red",
            alpha=0.70,
            edgecolors="none",
            label=public_label,
            zorder=7,
        )

    if not official_df.empty:
        if not paper_mode_enabled():
            ax.scatter(
                official_df["score"],
                np.zeros(len(official_df)),
                marker="x",
                s=95,
                color="black",
                linewidths=2.0,
                label=official_label,
                zorder=8,
            )

        label_offsets = [10, 26, 42]
        for i, row in official_df.reset_index(drop=True).iterrows():
            label = official_plan_label(str(row["label"]))
            style = official_plan_style(label, marker="x", color="black")
            if paper_mode_enabled():
                ax.plot(
                    row["score"], 0, label=label, **style,
                    markersize=7, markerfacecolor="none", markeredgewidth=1.3, zorder=8,
                )
            ax.annotate(
                label,
                xy=(row["score"], 0),
                xytext=(0, label_offsets[i % len(label_offsets)]),
                textcoords="offset points",
                ha="center",
                va="bottom",
                rotation=35,
                fontsize=9,
                color=style["color"],
                annotation_clip=False,
                arrowprops={
                    "arrowstyle": "-",
                    "linewidth": 0.8,
                    "color": style["color"],
                    "shrinkA": 0,
                    "shrinkB": 4,
                },
            )


def plot_histogram(
    ensemble_df: pd.DataFrame,
    public_df: pd.DataFrame,
    official_df: pd.DataFrame,
    outpath: str | Path,
    *,
    stat: str,
    bins: int,
    title: str | None,
    density: bool,
    ensemble_label: str,
    public_label: str,
    official_label: str,
    metric_name: str = "Polsby-Popper",
) -> None:
    scores = ensemble_df["score"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.hist(
        scores,
        bins=bins,
        density=density,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.7,
        label=ensemble_label,
    )

    ymax = ax.get_ylim()[1]
    add_reference_rug(
        ax,
        public_df,
        official_df,
        ymax,
        public_label=public_label,
        official_label=official_label,
    )

    mean_score = float(np.mean(scores))
    median_score = float(np.median(scores))

    ax.axvline(
        mean_score,
        linestyle="--",
        linewidth=1.4,
        color="gray",
        label=f"Ensemble mean = {mean_score:.3f}",
    )
    ax.axvline(
        median_score,
        linestyle=":",
        linewidth=1.6,
        color="gray",
        label=f"Ensemble median = {median_score:.3f}",
    )

    ax.set_ylim(0, ymax * 1.08)

    stat_label = f"{STAT_TO_SUMMARY_COL[stat].capitalize()} district {metric_name}"
    ax.set_xlabel(stat_label)
    ax.set_ylabel("Density" if density else "Number of saved plans")
    ax.set_title(title or f"Ohio congressional {stat_label.lower()} distribution")
    figure_legend(ax, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    destination = Path(outpath)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    destination = save_figure(fig, destination, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figure to {destination}")

    print("\nEnsemble score summary")
    print(ensemble_df["score"].describe().to_string())

    if not public_df.empty:
        print(f"\nLoaded {len(public_df)} public submitted plan scores.")
        print(public_df["score"].describe().to_string())

    if not official_df.empty:
        print(f"\nLoaded {len(official_df)} official plan scores.")
        print(official_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot an ensemble Polsby-Popper histogram from atlas-posthoc traces "
            "with optional public and official plan overlays."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--ensemble-pp-json",
        required=True,
        type=Path,
        help="atlas-posthoc diagnostics/polsby_popper_traces.json file.",
    )
    parser.add_argument(
        "--public-scores-csv",
        type=Path,
        default=None,
        help=(
            "Optional public-plan score CSV. Accepts plan_id,score or the "
            "public score cache with plan_id,average,minimum,maximum."
        ),
    )
    parser.add_argument(
        "--public-plan-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory of public submitted assignment files. Used to "
            "compute a public score cache when --public-scores-csv/cache is absent."
        ),
    )
    parser.add_argument(
        "--public-scores-out",
        type=Path,
        default=None,
        help="Where to write/reuse computed public plan Polsby-Popper scores.",
    )
    parser.add_argument(
        "--public-score-method",
        choices=["graph", "blocks"],
        default="graph",
        help=(
            "How to compute public plan scores when no public score CSV/cache "
            "exists. graph is much faster; blocks dissolves block geometries."
        ),
    )
    parser.add_argument(
        "--refresh-public-scores",
        action="store_true",
        help="Recompute public plan scores even if --public-scores-out already exists.",
    )
    parser.add_argument(
        "--graph-json",
        type=Path,
        default=None,
        help="Atlas graph JSON used for graph-based public plan scores.",
    )
    parser.add_argument(
        "--crosswalk-csv",
        type=Path,
        default=None,
        help="Block-to-graph/precinct crosswalk used for graph-based public plan scores.",
    )
    parser.add_argument(
        "--graph-key-col",
        default="ORIG_NAME",
        help="Graph node column matched to crosswalk precinct names.",
    )
    parser.add_argument("--node-id-col", default="id", help="Graph node id column.")
    parser.add_argument(
        "--area-col",
        default="area",
        help="Graph node area column for graph-based public compactness.",
    )
    parser.add_argument(
        "--border-length-col",
        default="border_length",
        help="Graph node external border-length column for graph-based compactness.",
    )
    parser.add_argument(
        "--edge-length-col",
        default=None,
        help="Graph adjacency edge length column; inferred when omitted.",
    )
    parser.add_argument(
        "--block-shapefile",
        type=Path,
        default=None,
        help="Block shapefile used when computing public plan scores.",
    )
    parser.add_argument("--block-col", default=None, help="Block GEOID column in the shapefile.")
    parser.add_argument(
        "--projected-crs",
        default="EPSG:5070",
        help="Projected CRS used when computing Polsby-Popper from public plans.",
    )
    parser.add_argument("--assignment-block-col", default=None)
    parser.add_argument("--assignment-district-col", default=None)
    parser.add_argument(
        "--expected-districts",
        type=int,
        default=15,
        help="Expected number of districts in each public submitted plan.",
    )
    parser.add_argument(
        "--official-scores-csv",
        type=Path,
        default=None,
        help=(
            "Optional official-plan score CSV. Accepts either label,score or "
            "the official-polsby summary CSV with plan,average,minimum,maximum."
        ),
    )
    parser.add_argument(
        "--stat",
        choices=["average", "mean", "min", "minimum", "max", "maximum"],
        default="average",
        help="Which Polsby-Popper statistic to plot.",
    )
    parser.add_argument(
        "--burnin",
        type=int,
        default=0,
        help="Discard ensemble samples with proposal number less than this.",
    )
    parser.add_argument("--bins", type=int, default=40, help="Number of histogram bins.")
    parser.add_argument("--density", action="store_true", help="Plot density instead of raw counts.")
    parser.add_argument("--title", default=None, help="Optional plot title.")
    parser.add_argument("--ensemble-label", default="Compliant ensemble plans")
    parser.add_argument("--public-label", default="Public submitted plans")
    parser.add_argument("--official-label", default="Official plans")
    parser.add_argument(
        "--out",
        "--output",
        dest="out",
        required=True,
        type=Path,
        help="Output PNG/PDF/SVG path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensemble_df = load_ensemble_scores(
        path=args.ensemble_pp_json,
        stat=args.stat,
        burnin=args.burnin,
    )
    public_df = load_or_compute_public_scores(args)
    official_df = load_official_scores(args.official_scores_csv, stat=args.stat)

    plot_histogram(
        ensemble_df=ensemble_df,
        public_df=public_df,
        official_df=official_df,
        outpath=args.out,
        stat=args.stat,
        bins=args.bins,
        title=args.title,
        density=args.density,
        ensemble_label=args.ensemble_label,
        public_label=args.public_label,
        official_label=args.official_label,
    )


if __name__ == "__main__":
    main()
