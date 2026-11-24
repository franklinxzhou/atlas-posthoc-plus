"""Compare municipal split distributions with official plans in paired histograms."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from atlas_posthoc_plus.config import ProjectConfig, load_config
from atlas_posthoc_plus.paper import (
    apply_paper_style,
    official_plan_label,
    official_plan_style,
    paper_mode_enabled,
    save_figure,
)
from atlas_posthoc_plus.workflows.municipal.geography import municipality_name_for_traces
from atlas_posthoc_plus.workflows.municipal.mct_diagnostics import PLOT_REFERENCE_LABELS
from atlas_posthoc_plus.workflows.municipal.tradeoff import load_json, normalize_reference_summaries

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator, PercentFormatter, StrMethodFormatter


def _is_mct(config: ProjectConfig) -> bool:
    return str(config.get("municipality.mode", "")).upper() == "MCT"


def preflight_paths(config: ProjectConfig) -> list[tuple[str, Path]]:
    namespace = Path(config.require("municipality.output_namespace"))
    if _is_mct(config):
        directory = config.path_value("outputs.mct_dir") or config.output_path(namespace)
        return [
            ("MCT ensemble plans", directory / "mct_plan_summary.csv"),
            ("MCT official plans", directory / "mct_official_plans.csv"),
        ]

    directory = config.output_path(namespace / "municipal-compliant")
    return [
        (
            "ensemble traces",
            config.path_value("inputs.traces_json")
            or directory / "municipal_split_traces_compliant.json",
        ),
        (
            "official plan summaries",
            config.path_value("inputs.reference_json")
            or directory / "oh_municipal_split_histograms_with_refs_reference_summaries.json",
        ),
    ]


def _mark_plans(ax: Axes, references: dict[str, float], edges: np.ndarray) -> None:
    grouped: dict[float, list[str]] = {}
    for label, value in references.items():
        if np.isfinite(value):
            grouped.setdefault(value, []).append(label)

    low = min(edges[0], *grouped) if grouped else edges[0]
    high = max(edges[-1], *grouped) if grouped else edges[-1]
    padding = 0.06 * (high - low)
    low, high = max(min(0, edges[0]), low - padding), high + padding
    ax.set_xlim(low, high)

    # Spread nearby labels without changing the values indicated by their arrows.
    margin = 0.16
    count = len(grouped)
    gap = min(0.30, (1 - 2 * margin) / max(count - 1, 1))
    previous = margin - gap
    for index, (value, labels) in enumerate(sorted(grouped.items())):
        target = (value - low) / (high - low)
        right = 1 - margin - gap * (count - index - 1)
        center = min(max(target, previous + gap), right)
        previous = center
        arrow_color = "black"
        if len(labels) == 1:
            arrow_color = official_plan_style(labels[0], color=arrow_color)["color"]
        if paper_mode_enabled():
            for mark_index, label in enumerate(labels):
                ax.plot(
                    value, 0,
                    **official_plan_style(label, marker="o", color="black"),
                    transform=ax.get_xaxis_transform(),
                    markersize=max(4, 8 - 2 * mark_index),
                    markerfacecolor="none",
                    clip_on=False,
                    zorder=6,
                )
        ax.annotate(
            "\n".join(official_plan_label(label) for label in labels),
            xy=(value, 0),
            xycoords=ax.get_xaxis_transform(),
            xytext=(center, 0.30),
            textcoords="axes fraction",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
            arrowprops={"arrowstyle": "->", "color": arrow_color, "lw": 0.9, "shrinkB": 0},
            zorder=5,
        )


def _histogram(ax: Axes, edges: np.ndarray, heights: np.ndarray) -> None:
    ax.bar(
        edges[:-1],
        heights,
        width=np.diff(edges),
        align="edge",
        color="0.72",
        edgecolor="0.20",
        linewidth=0.55,
    )
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_axisbelow(True)


def plot_tradeoff_histograms(
    *,
    cut_values: np.ndarray,
    ousted_values: np.ndarray,
    references: dict[str, tuple[float, float]],
    cut_label: str,
    ousted_label: str,
    share: bool,
    output: Path,
) -> None:
    """Draw the ousted-population density and integer cut frequencies."""
    cuts = np.asarray(cut_values, dtype=float)
    cuts = cuts[np.isfinite(cuts)]
    if cuts.size == 0:
        raise ValueError("No ensemble cut counts are available")
    ousted = np.asarray(ousted_values, dtype=float)
    ousted = ousted[np.isfinite(ousted)]

    apply_paper_style()
    fig, (population_ax, cut_ax) = plt.subplots(1, 2, figsize=(8.4, 3.1), layout="constrained")
    try:
        population_ax.set_xlabel(ousted_label)
        if ousted.size:
            edges = np.histogram_bin_edges(ousted, bins=12)
            if ousted.min() == ousted.max():
                value = ousted[0]
                half_width = 0.0005 if share else max(0.5, 0.01 * value)
                edges = np.array([max(0, value - half_width), value + half_width])
            density, _ = np.histogram(ousted, bins=edges, density=True)
            if share:
                population_ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
                population_ax.set_ylabel("Plan density\n(per percentage point)")
                density *= 0.01
            else:
                population_ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
                population_ax.set_ylabel("Plan density\n(per 100,000 people)")
                density *= 100_000
            _histogram(population_ax, edges, density)
            _mark_plans(
                population_ax, {label: values[1] for label, values in references.items()}, edges
            )
        else:
            population_ax.set(xticks=[], yticks=[])
            population_ax.text(
                0.5,
                0.5,
                "No represented population",
                transform=population_ax.transAxes,
                ha="center",
                va="center",
            )

        edges = np.arange(cuts.min() - 0.5, cuts.max() + 1.5, 1)
        frequencies, _ = np.histogram(cuts, bins=edges)
        cut_ax.set(xlabel=cut_label, ylabel="Share of sampled plans")
        _histogram(cut_ax, edges, frequencies / cuts.size)
        cut_ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=1))
        cut_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        _mark_plans(cut_ax, {label: values[0] for label, values in references.items()}, edges)

        destination = save_figure(fig, output)
        print(f"Wrote {destination}")
    finally:
        plt.close(fig)


def _plot_mct(ensemble_path: Path, official_path: Path, output_dir: Path, *, share: bool) -> None:
    ensemble = pd.read_csv(ensemble_path)
    official = pd.read_csv(official_path).set_index("plan")
    official = official.rename(index=PLOT_REFERENCE_LABELS)
    for category, units, cut_key, share_key, population_key in (
        (
            "mct",
            "municipal corporations\nand townships",
            "cut_mct_units",
            "ousted_share_represented",
            "deduplicated_ousted_population",
        ),
        (
            "municipal-corporation",
            "municipal corporations",
            "cut_municipal_corporations",
            "municipal_corporation_ousted_share",
            "municipal_corporation_ousted_population",
        ),
        (
            "township",
            "townships",
            "cut_townships",
            "township_ousted_share",
            "township_ousted_population",
        ),
    ):
        ousted_key = share_key if share else population_key
        ousted_label = {
            "mct": "Deduplicated ousted population",
            "municipal-corporation": "Ousted municipal corporation population",
            "township": "Ousted township population",
        }[category]
        if share:
            ousted_label += " share"
            if category == "mct":
                ousted_label += "\n(of represented population)"
        plot_tradeoff_histograms(
            cut_values=ensemble[cut_key].to_numpy(),
            ousted_values=ensemble[ousted_key].to_numpy(),
            references=dict(
                zip(
                    official.index,
                    official[[cut_key, ousted_key]].itertuples(index=False, name=None),
                )
            ),
            cut_label=f"Number of cut {units}",
            ousted_label=ousted_label,
            share=share,
            output=output_dir / f"{category}-tradeoff-histograms.pdf",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ousted-scale", choices=("share", "population"), default="share")
    args = parser.parse_args()
    config = load_config(args.config)
    (_, ensemble_path), (_, official_path) = preflight_paths(config)
    output_dir = args.output_dir or config.output_path(
        Path(config.require("municipality.output_namespace")) / "municipal"
    )
    share = args.ousted_scale == "share"
    if _is_mct(config):
        _plot_mct(ensemble_path, official_path, output_dir, share=share)
        return

    traces = load_json(ensemble_path)
    if not isinstance(traces, list) or not traces:
        raise ValueError("The ensemble traces must contain a nonempty JSON list")
    geography = municipality_name_for_traces(traces)
    refs = normalize_reference_summaries(load_json(official_path))
    cut_key = "n_cut_municipalities"
    ousted_key = "ousted_population_share" if share else "ousted_population"
    plot_tradeoff_histograms(
        cut_values=np.concatenate([trace[cut_key] for trace in traces]),
        ousted_values=np.concatenate([trace[ousted_key] for trace in traces]),
        references={label: (plan[cut_key], plan[ousted_key]) for label, plan in refs.items()},
        cut_label=f"Number of cut {geography}s",
        ousted_label=f"Ousted {geography} population{' share' if share else ''}",
        share=share,
        output=output_dir / "municipal-tradeoff-histograms.pdf",
    )


if __name__ == "__main__":
    main()
