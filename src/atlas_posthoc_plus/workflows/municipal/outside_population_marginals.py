#!/usr/bin/env python3
"""Plot multi-ensemble marginal distributions of ousted MCD population."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from atlas_posthoc_plus.paper import save_figure

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

from atlas_posthoc_plus.workflows.municipal.marginal_helpers import (
    REFERENCE_LINE_STYLES,
    CollectionSeries,
    build_collection_series,
    combined_min_max,
    load_other_ensemble_series,
    load_reference_summaries,
    reference_values,
)
from atlas_posthoc_plus.workflows.municipal.tradeoff_weights import (
    DEFAULT_TRACES,
    load_all_traces,
)


def _draw_half_violins(ax, series: list[CollectionSeries], positions: np.ndarray) -> None:
    violin_data = []
    violin_positions = []
    violin_series = []

    for item, position in zip(series, positions):
        if item.values.size >= 2 and np.ptp(item.values) > 0:
            violin_data.append(item.values)
            violin_positions.append(position)
            violin_series.append(item)
        else:
            value = float(item.values[0])
            ax.plot(
                [value, value],
                [position - 0.30, position],
                color=item.color,
                linewidth=5.0,
                alpha=0.38,
                solid_capstyle="round",
                zorder=1,
            )

    if not violin_data:
        return

    parts = ax.violinplot(
        violin_data,
        positions=violin_positions,
        orientation="horizontal",
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, item, position in zip(parts["bodies"], violin_series, violin_positions):
        vertices = body.get_paths()[0].vertices
        vertices[:, 1] = np.minimum(vertices[:, 1], position)
        body.set_facecolor(item.color)
        body.set_edgecolor(item.color)
        body.set_alpha(0.36)
        body.set_linewidth(1.0)
        body.set_zorder(1)


def _draw_boxplots(ax, series: list[CollectionSeries], positions: np.ndarray) -> None:
    box_positions = positions + 0.16
    plot = ax.boxplot(
        [item.values for item in series],
        positions=box_positions,
        orientation="horizontal",
        widths=0.18,
        showfliers=False,
        patch_artist=True,
        manage_ticks=False,
        medianprops={"color": "black", "linewidth": 1.4},
        whiskerprops={"color": "#333333", "linewidth": 1.0},
        capprops={"color": "#333333", "linewidth": 1.0},
    )

    for patch, item in zip(plot["boxes"], series):
        patch.set_facecolor("white")
        patch.set_edgecolor(item.color)
        patch.set_linewidth(1.5)
        patch.set_alpha(0.95)

    medians = [float(np.median(item.values)) for item in series]
    ax.scatter(
        medians,
        box_positions,
        s=28,
        color=[item.color for item in series],
        edgecolors="black",
        linewidths=0.6,
        zorder=4,
    )


def plot_outside_population_marginals(
    *,
    traces: list[dict[str, Any]],
    refs: dict[str, Any],
    output: str | Path,
    metric_key: str,
    metric_label: str,
    use_percent_axis: bool,
    title: str,
    extra_series: list[CollectionSeries] | None = None,
) -> None:
    series = build_collection_series(traces, metric_key=metric_key)
    if extra_series:
        series = [*series, *extra_series]

    ref_values = reference_values(refs, metric_key=metric_key)
    positions = np.arange(len(series), 0, -1, dtype=float)

    fig_height = max(4.8, 1.0 + 0.72 * len(series))
    fig, ax = plt.subplots(figsize=(11.8, fig_height))

    _draw_half_violins(ax, series, positions)
    _draw_boxplots(ax, series, positions)

    ref_handles = []
    for label, value in ref_values.items():
        style = REFERENCE_LINE_STYLES.get(
            label,
            {"color": "#555555", "linestyle": ":", "linewidth": 2.0},
        )
        ax.axvline(value, alpha=0.96, zorder=5, **style)
        ref_handles.append(Line2D([0], [0], label=label, **style))

    x_min, x_max = combined_min_max(series, ref_values)
    pad = max(0.0005 if use_percent_axis else 1.0, 0.08 * (x_max - x_min))
    ax.set_xlim(x_min - pad, x_max + pad)

    ax.set_yticks(positions)
    ax.set_yticklabels([item.label for item in series])
    ax.set_xlabel(metric_label)
    fig.suptitle(title, fontsize=18, y=0.98)
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    if use_percent_axis:
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    if ref_handles:
        fig.legend(
            handles=ref_handles,
            title="Official plans",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.92),
            ncol=min(3, len(ref_handles)),
            frameon=True,
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84 if ref_handles else 0.92))
    output = save_figure(fig, output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot marginal distributions of ousted MCD population across municipal "
            "trace collections, with official-plan reference lines."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trace-collection",
        dest="trace_collection",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Collection label and municipal_split_traces.json path. May be repeated.",
    )
    parser.add_argument(
        "--traces-json",
        dest="trace_collection",
        action="append",
        metavar="LABEL=PATH",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--traces-json-mcd20", default=DEFAULT_TRACES["20"])
    parser.add_argument("--traces-json-mcd5", default=DEFAULT_TRACES["5"])
    parser.add_argument("--traces-json-mcd1", default=DEFAULT_TRACES["1"])
    parser.add_argument(
        "--reference-json",
        default="outputs/municipal-compliant/"
        "oh_municipal_split_histograms_with_refs_reference_summaries.json",
        help="Official-plan municipal split summary JSON.",
    )
    parser.add_argument(
        "--skip-official-plans",
        action="store_true",
        help="Do not draw official-plan reference lines.",
    )
    parser.add_argument(
        "--y-metric",
        choices=["share", "population"],
        default="share",
        help="Plot ousted MCD population share or raw population.",
    )
    parser.add_argument(
        "--include-other-ensemble",
        action="store_true",
        help="Add values from --other-ensemble-summary-json as an extra collection.",
    )
    parser.add_argument(
        "--other-ensemble-summary-json",
        default=None,
        help="Optional external ensemble municipal summary JSON.",
    )
    parser.add_argument(
        "--other-ensemble-label",
        default="Other ensemble",
        help="Label to use for --include-other-ensemble.",
    )
    parser.add_argument(
        "--include-other-cd-2020",
        action="store_true",
        help="Keep reference cd_2020 rows in the optional external ensemble.",
    )
    parser.add_argument(
        "--output",
        default="outputs/municipal/municipal-outside-population-marginals.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--title",
        default="Ousted MCD population marginal distributions",
        help="Plot title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    traces = load_all_traces(args)

    if args.y_metric == "share":
        metric_key = "ousted_population_share"
        metric_label = "Ousted MCD population share"
        use_percent_axis = True
    else:
        metric_key = "ousted_population"
        metric_label = "Ousted MCD population"
        use_percent_axis = False

    refs = {} if args.skip_official_plans else load_reference_summaries(args.reference_json)

    extra_series = []
    if args.include_other_ensemble:
        if args.other_ensemble_summary_json is None:
            raise ValueError("--include-other-ensemble requires --other-ensemble-summary-json.")
        other = load_other_ensemble_series(
            args.other_ensemble_summary_json,
            metric_key=metric_key,
            label=args.other_ensemble_label,
            style_index=len(traces),
            include_reference_cd_2020=args.include_other_cd_2020,
        )
        if other is not None:
            extra_series.append(other)

    plot_outside_population_marginals(
        traces=traces,
        refs=refs,
        output=args.output,
        metric_key=metric_key,
        metric_label=metric_label,
        use_percent_axis=use_percent_axis,
        title=args.title,
        extra_series=extra_series,
    )


if __name__ == "__main__":
    main()
