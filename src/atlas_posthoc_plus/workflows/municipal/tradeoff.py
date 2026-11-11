#!/usr/bin/env python3
"""
### Use this to process atlas-posthoc generated json diagnostics files! Not touching the raw chain files ###
-------
atlas-posthoc-plus run municipal-tradeoff \
  --traces-json outputs/municipal-compliant/municipal_split_traces_compliant.json \
  --reference-json outputs/oh_municipal_split_distributions_with_refs_reference_summaries.json \
  --output outputs/oh_municipal_split_tradeoff_with_refs.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from atlas_posthoc_plus.paper import save_figure

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

from atlas_posthoc_plus.workflows.municipal.geography import municipality_name_for_traces


DEFAULT_CHAIN_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
]

OFFICIAL_STYLES = {
    "Enacted ('25)": {
        "marker": "X",
        "color": "#9467bd",
        "label_dx": 0.35,
        "label_dy": 0.00025,
    },
    "Overturned ('21)": {
        "marker": "s",
        "color": "#2ca02c",
        "label_dx": 0.35,
        "label_dy": -0.00035,
    },
    "Remedial ('22)": {
        "marker": "^",
        "color": "#d62728",
        "label_dx": 0.35,
        "label_dy": 0.00055,
    },
}


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)


def require_keys(obj: dict[str, Any], keys: list[str], *, context: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise ValueError(
            f"{context} is missing required keys {missing}. "
            f"Available keys: {list(obj.keys())}"
        )


def clean_plan_label(label: str) -> str:
    """
    Keep labels consistent even if one workflow writes slightly different keys.
    """
    aliases = {
        "Enacted ('25)": "Enacted ('25)",
        "Enacted ('25')": "Enacted ('25)",
        "Enacted": "Enacted ('25)",
        "Overturned ('21)": "Overturned ('21)",
        "Overturned ('21')": "Overturned ('21)",
        "Overturned": "Overturned ('21)",
        "Remedial ('22)": "Remedial ('22)",
        "Remedial ('22')": "Remedial ('22)",
        "Remedial": "Remedial ('22)",
    }
    return aliases.get(label, label)


def normalize_reference_summaries(raw_refs: dict[str, Any]) -> dict[str, Any]:
    refs: dict[str, Any] = {}
    for label, summary in raw_refs.items():
        new_label = clean_plan_label(label)
        refs[new_label] = summary
    return refs


def get_chain_colors(traces: list[dict[str, Any]]) -> dict[str, str]:
    chain_ids = [str(item.get("chain_id", f"chain{i + 1}")) for i, item in enumerate(traces)]
    unique_chain_ids = []
    for chain_id in chain_ids:
        if chain_id not in unique_chain_ids:
            unique_chain_ids.append(chain_id)

    return {
        chain_id: DEFAULT_CHAIN_COLORS[i % len(DEFAULT_CHAIN_COLORS)]
        for i, chain_id in enumerate(unique_chain_ids)
    }


def compute_axis_limits(
    traces: list[dict[str, Any]],
    refs: dict[str, Any],
    *,
    y_key: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = []
    ys = []

    for item in traces:
        xs.extend(float(x) for x in item["n_cut_municipalities"])
        ys.extend(float(y) for y in item[y_key])

    for summary in refs.values():
        xs.append(float(summary["n_cut_municipalities"]))
        ys.append(float(summary[y_key]))

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_pad = max(1.0, 0.08 * (x_max - x_min))
    y_pad = max(0.0005, 0.12 * (y_max - y_min))

    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def plot_tradeoff(
    *,
    traces: list[dict[str, Any]],
    refs: dict[str, Any],
    output: str | Path,
    title: str,
    y_key: str,
    y_label: str,
    use_percent_axis: bool,
    label_references: bool,
) -> None:
    if not isinstance(traces, list) or not traces:
        raise ValueError("traces JSON must be a nonempty list.")

    for i, item in enumerate(traces):
        require_keys(
            item,
            ["chain_id", "n_cut_municipalities", y_key],
            context=f"trace item {i}",
        )

    for label, summary in refs.items():
        require_keys(
            summary,
            ["n_cut_municipalities", y_key],
            context=f"reference plan {label}",
        )

    chain_colors = get_chain_colors(traces)

    fig, ax = plt.subplots(figsize=(11.5, 7.2))

    # Ensemble cloud.
    for item in traces:
        chain_id = str(item.get("chain_id", "chain"))
        x = np.asarray(item["n_cut_municipalities"], dtype=float)
        y = np.asarray(item[y_key], dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)

        ax.scatter(
            x[mask],
            y[mask],
            s=34,
            facecolors="none",
            edgecolors=chain_colors.get(chain_id, "gray"),
            linewidths=1.4,
            alpha=0.38,
            label=chain_id,
            zorder=1,
        )

    # Official plan points.
    for label, style in OFFICIAL_STYLES.items():
        if label not in refs:
            continue

        x = float(refs[label]["n_cut_municipalities"])
        y = float(refs[label][y_key])

        ax.scatter(
            [x],
            [y],
            s=260,
            marker=style["marker"],
            color=style["color"],
            edgecolors="black",
            linewidths=1.4,
            zorder=10,
        )

    xlim, ylim = compute_axis_limits(traces, refs, y_key=y_key)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Reference labels after limits are fixed.
    if label_references:
        for label, style in OFFICIAL_STYLES.items():
            if label not in refs:
                continue

            x = float(refs[label]["n_cut_municipalities"])
            y = float(refs[label][y_key])

            ax.annotate(
                label,
                xy=(x, y),
                xytext=(x + style["label_dx"], y + style["label_dy"]),
                fontsize=12,
                fontweight="bold",
                color=style["color"],
                ha="left",
                va="center",
                arrowprops={
                    "arrowstyle": "->",
                    "lw": 1.3,
                    "color": style["color"],
                    "shrinkA": 5,
                    "shrinkB": 5,
                },
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "fc": "white",
                    "ec": style["color"],
                    "alpha": 0.92,
                },
                zorder=20,
            )

    ax.set_title(title, fontsize=22, pad=14)
    ax.set_xlabel(f"Number of cut {municipality_name_for_traces(traces)}s", fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)

    if use_percent_axis:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Separate chain and reference plan legends.
    chain_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.5,
            markersize=8,
            label=chain_id,
            alpha=0.8,
        )
        for chain_id, color in chain_colors.items()
    ]

    ref_handles = [
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            linestyle="None",
            markerfacecolor=style["color"],
            markeredgecolor="black",
            markeredgewidth=1.2,
            markersize=10,
            label=label,
        )
        for label, style in OFFICIAL_STYLES.items()
        if label in refs
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
        title="Reference plans",
        loc="lower right",
        frameon=True,
        fontsize=11,
        title_fontsize=11,
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    output = save_figure(fig, output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {output}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Overlay Enacted / Overturned / Remedial plans on an atlas-posthoc "
            "municipal split tradeoff scatter plot."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--traces-json",
        required=True,
        help="Path to atlas-posthoc municipal_split_traces.json.",
    )
    p.add_argument(
        "--reference-json",
        required=True,
        help="Path to reference plan municipal split summaries JSON.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output PNG path.",
    )
    p.add_argument(
        "--title",
        default="Municipal split tradeoff",
        help="Plot title.",
    )
    p.add_argument(
        "--y-metric",
        choices=["share", "population"],
        default="share",
        help="Use ousted population share or raw ousted population on the y-axis.",
    )
    p.add_argument(
        "--no-labels",
        action="store_true",
        help="Use large markers and legend only; do not annotate reference plans.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    traces = load_json(args.traces_json)
    refs = normalize_reference_summaries(load_json(args.reference_json))
    municipality_name = municipality_name_for_traces(traces)

    if args.y_metric == "share":
        y_key = "ousted_population_share"
        y_label = f"Ousted {municipality_name} population share"
        use_percent_axis = True
    else:
        y_key = "ousted_population"
        y_label = f"Ousted {municipality_name} population"
        use_percent_axis = False

    plot_tradeoff(
        traces=traces,
        refs=refs,
        output=args.output,
        title=args.title,
        y_key=y_key,
        y_label=y_label,
        use_percent_axis=use_percent_axis,
        label_references=not args.no_labels,
    )


if __name__ == "__main__":
    main()
