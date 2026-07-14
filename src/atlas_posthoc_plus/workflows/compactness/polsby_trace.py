#!/usr/bin/env python3
"""
Want min PP:
atlas-posthoc-plus run polsby-trace \
  --graph-json data/graph/OHpct20.json \
  --chain-files \
    outputs/chains/soft_constraint/congressional/run37_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run38_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run39_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run40_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
  --chain-ids run37 run38 run39 run40 \
  --official-reference-csv outputs/polsby_popper_traces_37_40/official_plan_polsby_popper_from_blocks_summary.csv \
  --metric min \
  --output-frequency 50000 \
  --total-proposals 50000000 \
  --max-maps-per-chain 1000 \
  --output outputs/polsby_popper_traces_37_40/oh_congressional_minimum_polsby_popper_trace_with_refs.png

Want avg PP:
atlas-posthoc-plus run polsby-trace \
  --graph-json data/graph/OHpct20.json \
  --chain-files \
    outputs/chains/soft_constraint/congressional/run37_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run38_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run39_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run40_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
  --chain-ids run37 run38 run39 run40 \
  --official-reference-csv outputs/polsby_popper_traces_37_40/official_plan_polsby_popper_from_blocks_summary.csv \
  --metric average \
  --output-frequency 50000 \
  --total-proposals 50000000 \
  --max-maps-per-chain 1000 \
  --output outputs/polsby_popper_traces_37_40/oh_congressional_average_polsby_popper_trace_with_refs.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

import numpy as np
import pandas as pd

import atlas_posthoc.pipeline.chain_observables as chain_observables
from atlas_posthoc.observables.compactness import extract_ordered_compactness_from_record
from atlas_posthoc.observables.polsby_popper import make_polsby_popper_traces
from atlas_posthoc_plus.io.atlas_records import iter_records
from atlas_posthoc_plus.io.chain_sweep import configured_sweep, iter_sweep_records


PLAN_STYLES = {
    "Enacted ('25)": {"color": "#9467bd", "linestyle": "-", "linewidth": 2.5},
    "Overturned ('21)": {"color": "#2ca02c", "linestyle": "--", "linewidth": 2.5},
    "Remedial ('22)": {"color": "#d62728", "linestyle": "-.", "linewidth": 2.5},
}

CHAIN_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def load_sweep_compactness_runs(
    chain_files: list[str | Path],
    chain_ids: list[str],
    *,
    n_districts: int,
    thin_step: int,
    max_maps_per_chain: int | None,
) -> dict[str, np.ndarray]:
    runs: dict[str, np.ndarray] = {}
    for chain_file, chain_id in zip(chain_files, chain_ids):
        rows: list[np.ndarray] = []
        for index, record in enumerate(iter_sweep_records(chain_file)):
            if index % thin_step != 0:
                continue
            if max_maps_per_chain is not None and len(rows) >= max_maps_per_chain:
                break
            result = extract_ordered_compactness_from_record(
                record,
                n_districts=n_districts,
                sort="ascending",
                allow_missing_scores=False,
            )
            rows.append(
                np.asarray(
                    [np.nan if value is None else value for value in result.ordered_scores],
                    dtype=float,
                )
            )
        if not rows:
            raise RuntimeError(f"no compactness records found for {chain_id}")
        runs[chain_id] = np.vstack(rows)
    return runs


def get_trace_values(trace: Any, metric: str) -> np.ndarray:
    metric = normalize_metric(metric)
    if metric == "average":
        return np.asarray(trace.average_polsby_popper, dtype=float)
    if metric == "minimum":
        return np.asarray(trace.minimum_polsby_popper, dtype=float)
    if metric == "maximum":
        return np.asarray(trace.maximum_polsby_popper, dtype=float)
    raise ValueError(f"unknown metric: {metric}")


def normalize_metric(metric: str) -> str:
    aliases = {
        "avg": "average",
        "mean": "average",
        "average": "average",
        "min": "minimum",
        "minimum": "minimum",
        "max": "maximum",
        "maximum": "maximum",
    }
    key = str(metric).strip().lower()
    if key not in aliases:
        raise ValueError(f"unknown metric: {metric!r}")
    return aliases[key]


def metric_label(metric: str) -> str:
    metric = normalize_metric(metric)
    if metric == "average":
        return "Average district Polsby-Popper"
    if metric == "minimum":
        return "Minimum district Polsby-Popper"
    if metric == "maximum":
        return "Maximum district Polsby-Popper"
    raise ValueError(f"unknown metric: {metric}")


def proposal_axis_for_trace(
    *,
    trace: Any,
    total_proposals: int | None,
    output_frequency: int,
    start_at_first_saved_step: bool,
) -> np.ndarray:
    """
    Return the x-axis for a trace.

    If total_proposals is supplied, spread the saved maps across the full run.
    This is the safest mode when the chain file contains a fixed number of saved
    plans from a known total proposal budget, e.g. 1000 saved plans over 50M
    proposals.

    Otherwise, use the proposal indices stored in the trace object.
    """
    n = len(trace.proposals)

    if total_proposals is None:
        return np.asarray(trace.proposals, dtype=float)

    if n == 0:
        return np.asarray([], dtype=float)

    if start_at_first_saved_step:
        # Example: 1000 saved maps, every 50,000 proposals, last map at 50M.
        return np.arange(1, n + 1, dtype=float) * float(output_frequency)

    if n == 1:
        return np.asarray([float(total_proposals)])

    # Example: 1001 saved maps including the initial state, from 0 to 50M.
    return np.linspace(0.0, float(total_proposals), n)


def plot_trace_with_references(
    *,
    traces: list[Any],
    reference_df: pd.DataFrame,
    metric: str,
    output: str | Path,
    title: str | None,
    y_limits: tuple[float, float] | None,
    total_proposals: int | None,
    output_frequency: int,
    start_at_first_saved_step: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.8))

    all_y: list[float] = []

    for i, trace in enumerate(traces):
        color = CHAIN_COLORS[i % len(CHAIN_COLORS)]
        x = proposal_axis_for_trace(
            trace=trace,
            total_proposals=total_proposals,
            output_frequency=output_frequency,
            start_at_first_saved_step=start_at_first_saved_step,
        )
        y = get_trace_values(trace, metric)
        mask = np.isfinite(x) & np.isfinite(y)

        all_y.extend(y[mask].tolist())

        ax.plot(
            x[mask],
            y[mask],
            color=color,
            linewidth=1.7,
            alpha=0.85,
            label=trace.chain_id,
        )

    plan_order = ["Enacted ('25)", "Overturned ('21)", "Remedial ('22)"]
    reference_df = reference_df.copy()
    reference_df["__order__"] = reference_df["plan"].map(
        {label: i for i, label in enumerate(plan_order)}
    ).fillna(999)
    reference_df = reference_df.sort_values("__order__")

    for _, row in reference_df.iterrows():
        label = str(row["plan"])
        style = PLAN_STYLES.get(label, {"color": "black", "linestyle": ":", "linewidth": 2.2})
        y = float(row[metric])
        all_y.append(y)

        ax.axhline(
            y,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=0.95,
            zorder=5,
        )

    ax.set_title(title or f"Congressional {metric} Polsby-Popper trace", fontsize=20, pad=12)
    ax.set_xlabel("Proposal", fontsize=15)
    ax.set_ylabel(metric_label(metric), fontsize=15)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    if y_limits is not None:
        ax.set_ylim(*y_limits)
    elif all_y:
        lo = min(all_y)
        hi = max(all_y)
        pad = max(0.01, 0.08 * (hi - lo))
        ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))

    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    chain_handles = [
        Line2D([0], [0], color=CHAIN_COLORS[i % len(CHAIN_COLORS)], lw=2.0, label=tr.chain_id)
        for i, tr in enumerate(traces)
    ]

    ref_handles = [
        Line2D(
            [0],
            [0],
            color=PLAN_STYLES.get(row["plan"], {"color": "black"})["color"],
            linestyle=PLAN_STYLES.get(row["plan"], {"linestyle": ":"})["linestyle"],
            lw=2.5,
            label=row["plan"],
        )
        for _, row in reference_df.iterrows()
    ]

    legend1 = ax.legend(
        handles=chain_handles,
        title="Chains",
        loc="upper right",
        frameon=True,
        fontsize=11,
        title_fontsize=11,
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=ref_handles,
        title="Official plans",
        loc="lower right",
        frameon=True,
        fontsize=11,
        title_fontsize=11,
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {output}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot congressional Polsby-Popper traces with precomputed official reference lines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--graph-json", default="data/graph/OHpct20.json")
    p.add_argument(
        "--chain-files",
        nargs="+",
        default=[
            "outputs/chains/soft_constraint/congressional/run37_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl",
            "outputs/chains/soft_constraint/congressional/run38_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl",
            "outputs/chains/soft_constraint/congressional/run39_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl",
            "outputs/chains/soft_constraint/congressional/run40_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl",
        ],
    )
    p.add_argument("--chain-ids", nargs="+", default=["run37", "run38", "run39", "run40"])
    p.add_argument("--n-districts", type=int, default=15)
    p.add_argument("--output-frequency", type=int, default=10000)
    p.add_argument("--thin-step", type=int, default=1)
    p.add_argument("--max-maps-per-chain", type=int, default=None)
    p.add_argument(
        "--total-proposals",
        type=int,
        default=None,
        help=(
            "Override the trace x-axis so the plotted proposal range spans this "
            "total budget. Useful when 1000 saved maps came from 50M proposals."
        ),
    )
    p.add_argument(
        "--start-at-first-saved-step",
        action="store_true",
        help=(
            "With --total-proposals, put the first saved map at output_frequency "
            "rather than at 0. Use this when the chain file does not include the "
            "initial state."
        ),
    )

    p.add_argument("--dem-col", default="G20PREDEM")
    p.add_argument("--rep-col", default="G20PREREP")
    p.add_argument("--atlas-key-col", default="NAME")
    p.add_argument("--compactness-convention", default="p2_over_area")

    p.add_argument(
        "--official-reference-csv",
        required=True,
        help="Summary CSV from compute_official_congressional_polsby_popper_from_blocks.py",
    )
    p.add_argument(
        "--metric",
        choices=["average", "avg", "mean", "minimum", "min", "maximum", "max"],
        default="average",
        help=(
            "Which plan-level Polsby-Popper statistic to plot. "
            "Use 'average' for mean district PP or 'min'/'minimum' for worst-district PP."
        ),
    )
    p.add_argument(
        "--output",
        default="outputs/polsby_popper_traces_37_40/oh_congressional_average_polsby_popper_trace_with_refs.png",
    )
    p.add_argument("--title", default=None)
    p.add_argument("--y-min", type=float, default=None)
    p.add_argument("--y-max", type=float, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.chain_ids) != len(args.chain_files):
        raise ValueError("--chain-ids must have the same length as --chain-files")

    args.metric = normalize_metric(args.metric)

    print("=" * 80)
    print("Ohio congressional Polsby-Popper trace from official reference CSV")
    print("=" * 80)

    if configured_sweep() is not None:
        print("\n[1/3] Loading compactness from the shared one-pass sweep...")
        compactness_runs = load_sweep_compactness_runs(
            args.chain_files,
            args.chain_ids,
            n_districts=args.n_districts,
            thin_step=args.thin_step,
            max_maps_per_chain=args.max_maps_per_chain,
        )
    else:
        print("\n[1/3] Parsing AtlasIO chains with atlas-posthoc...")
        chain_observables.iter_atlas_records = iter_records
        obs = chain_observables.build_multi_chain_observables(
            [Path(p) for p in args.chain_files],
            args.graph_json,
            chain_ids=args.chain_ids,
            key_cols=[args.atlas_key_col],
            dem_col=args.dem_col,
            rep_col=args.rep_col,
            n_districts=args.n_districts,
            output_frequency=args.output_frequency,
            thin_step=args.thin_step,
            require_compactness=True,
        )
        compactness_runs = obs.compactness_runs
        if args.max_maps_per_chain is not None:
            for chain_id in list(compactness_runs):
                compactness_runs[chain_id] = compactness_runs[chain_id][
                    : args.max_maps_per_chain
                ]

    for chain_id, arr in compactness_runs.items():
        print(
            f"  {chain_id}: compactness matrix {arr.shape}, "
            f"raw score range=({np.nanmin(arr):.3f}, {np.nanmax(arr):.3f})"
        )

    traces_n_maps = [arr.shape[0] for arr in compactness_runs.values()]
    if args.total_proposals is not None and traces_n_maps:
        unique_n_maps = sorted(set(traces_n_maps))
        print(f"  saved maps per chain: {unique_n_maps}")
        if len(unique_n_maps) == 1:
            n_maps = unique_n_maps[0]
            if args.start_at_first_saved_step:
                implied = args.total_proposals / n_maps
                print(
                    f"  total proposal override: {args.total_proposals:,}; "
                    f"first saved map treated as step {implied:,.0f}"
                )
            else:
                implied = args.total_proposals / max(n_maps - 1, 1)
                print(
                    f"  total proposal override: {args.total_proposals:,}; "
                    f"saved maps spread from 0 to {args.total_proposals:,}; "
                    f"implied interval {implied:,.0f}"
                )

    print("\n[2/3] Converting chain isoperimetric scores to Polsby-Popper...")
    traces = make_polsby_popper_traces(
        compactness_runs,
        output_frequency=args.output_frequency * args.thin_step,
        convention=args.compactness_convention,
        input_is_polsby_popper=False,
        invalid_policy="nan",
    )

    for tr in traces:
        vals = get_trace_values(tr, args.metric)
        print(
            f"  {tr.chain_id}: {args.metric} PP range=({np.nanmin(vals):.4f}, {np.nanmax(vals):.4f}), "
            f"n_maps={tr.n_maps}"
        )

    print("\n[3/3] Reading official reference CSV...")
    reference_df = pd.read_csv(args.official_reference_csv)

    required = {"plan", args.metric}
    missing = required - set(reference_df.columns)
    if missing:
        raise ValueError(
            f"Official reference CSV missing columns {sorted(missing)}. "
            f"Available columns: {list(reference_df.columns)}"
        )

    print(reference_df[["plan", "average", "minimum", "maximum"]].to_string(index=False))

    y_limits = None
    if args.y_min is not None or args.y_max is not None:
        if args.y_min is None or args.y_max is None:
            raise ValueError("Pass both --y-min and --y-max, or neither.")
        y_limits = (args.y_min, args.y_max)

    plot_trace_with_references(
        traces=traces,
        reference_df=reference_df,
        metric=args.metric,
        output=args.output,
        title=args.title,
        y_limits=y_limits,
        total_proposals=args.total_proposals,
        output_frequency=args.output_frequency * args.thin_step,
        start_at_first_saved_step=args.start_at_first_saved_step,
    )


if __name__ == "__main__":
    main()
