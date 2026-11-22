#!/usr/bin/env python3
"""Plot overlaid multi-ensemble histograms of cut municipality counts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from atlas_posthoc_plus.paper import figure_legend, official_plan_style, save_figure

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np

from atlas_posthoc_plus.workflows.municipal.geography import municipality_name_for_traces
from atlas_posthoc_plus.workflows.municipal.marginal_helpers import (
    REFERENCE_LINE_STYLES,
    CollectionSeries,
    build_collection_series as _build_collection_series,
    combined_min_max,
    load_other_ensemble_series,
    load_reference_summaries,
    reference_values,
)
from atlas_posthoc_plus.workflows.municipal.tradeoff_weights import (
    DEFAULT_TRACES,
    load_all_traces,
)


def build_collection_series(traces: list[dict[str, Any]]) -> list[CollectionSeries]:
    return _build_collection_series(traces, metric_key="n_cut_municipalities")


def integer_centered_bins(
    series: list[CollectionSeries],
    *,
    reference_values: list[float] | None = None,
) -> np.ndarray:
    refs = {} if reference_values is None else {str(i): value for i, value in enumerate(reference_values)}
    low, high = combined_min_max(series, refs)
    start = int(np.floor(low)) - 0.5
    stop = int(np.ceil(high)) + 1.0
    return np.arange(start, stop, 1.0)


def plot_cut_mcd_histograms(
    *,
    traces: list[dict[str, Any]],
    refs: dict[str, Any],
    output: str | Path,
    density: bool,
    title: str,
    extra_series: list[CollectionSeries] | None = None,
) -> None:
    series = build_collection_series(traces)
    if extra_series:
        series = [*series, *extra_series]

    ref_values = reference_values(refs, metric_key="n_cut_municipalities")
    bins = integer_centered_bins(series, reference_values=list(ref_values.values()))

    fig, ax = plt.subplots(figsize=(12.0, 6.4))

    collection_handles: list[Patch] = []
    for item in series:
        weights = None
        if density:
            weights = np.ones_like(item.values, dtype=float) / item.values.size

        ax.hist(
            item.values,
            bins=bins,
            weights=weights,
            color=item.color,
            alpha=0.34,
            edgecolor="none",
            label=item.label,
            zorder=1,
        )
        ax.hist(
            item.values,
            bins=bins,
            weights=weights,
            histtype="step",
            color=item.color,
            linewidth=1.5,
            zorder=2,
        )
        collection_handles.append(
            Patch(facecolor=item.color, edgecolor=item.color, alpha=0.34, label=item.label)
        )

    ref_handles = []
    for label, value in ref_values.items():
        style = official_plan_style(
            label,
            **REFERENCE_LINE_STYLES.get(
                label,
                {"color": "#555555", "linestyle": ":", "linewidth": 2.0},
            ),
        )
        ax.axvline(value, alpha=0.96, zorder=5, **style)
        ref_handles.append(Line2D([0], [0], label=label, **style))

    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel(f"Number of cut {municipality_name_for_traces(traces)}s")
    ax.set_ylabel("Share of plans" if density else "Count")
    fig.suptitle(title, fontsize=18, y=0.98)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if density:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    legend1 = figure_legend(
        ax,
        handles=collection_handles,
        title="Ensembles",
        loc="upper right",
        frameon=True,
    )
    ax.add_artist(legend1)
    if ref_handles:
        figure_legend(
            fig,
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
            "Plot overlaid marginal histograms of cut municipality counts across municipal "
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
        "--count",
        dest="density",
        action="store_false",
        default=True,
        help="Use raw counts instead of per-ensemble plan shares.",
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
        default="outputs/municipal/municipal-cut-mcd-histograms.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Plot title (defaults to the trace geography).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    traces = load_all_traces(args)
    refs = {} if args.skip_official_plans else load_reference_summaries(args.reference_json)

    extra_series = []
    if args.include_other_ensemble:
        if args.other_ensemble_summary_json is None:
            raise ValueError("--include-other-ensemble requires --other-ensemble-summary-json.")
        other = load_other_ensemble_series(
            args.other_ensemble_summary_json,
            metric_key="n_cut_municipalities",
            label=args.other_ensemble_label,
            style_index=len(traces),
            include_reference_cd_2020=args.include_other_cd_2020,
        )
        if other is not None:
            extra_series.append(other)

    plot_cut_mcd_histograms(
        traces=traces,
        refs=refs,
        output=args.output,
        density=args.density,
        title=(
            args.title if args.title is not None
            else f"Cut {municipality_name_for_traces(traces)} marginal distributions"
        ),
        extra_series=extra_series,
    )


if __name__ == "__main__":
    main()
