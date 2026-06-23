#!/usr/bin/env python3

"""
RUN THIS ONLY AFTER GENERATING THE CSV FILES WITH 
the responsiveness workflow

Example:
atlas-posthoc-plus run responsiveness-zoom \
  --race-summary outputs/ohio_responsiveness_all_races/oh_responsiveness_race_summary.csv \
  --ensemble-seats outputs/ohio_responsiveness_all_races/oh_responsiveness_ensemble_seats.csv \
  --official-seats outputs/ohio_responsiveness_all_races/oh_responsiveness_official_plan_seats.csv \
  --output outputs/ohio_responsiveness_all_races/ohio_responsiveness_zoomed.png \
  --title "Ohio congressional responsiveness across statewide elections" \
  --n-districts 15 \
  --zoom-panels 4 \
  --max-races-per-zoom 6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

PLAN_STYLES = {
    "Enacted ('25)": {"color": "#9467bd"},
    "Overturned ('21)": {"color": "#2ca02c"},
    "Remedial ('22)": {"color": "#d62728"},
}

HIST_FACE = "#a9c7d9"
HIST_EDGE = "#465f70"
BASELINE_COLOR = "#666666"
PROPORTIONALITY_STYLE = dict(color="black", linestyle=(0, (7, 5)), linewidth=2.2, alpha=0.85)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot Ohio responsiveness ridgeline histograms from CSV outputs."
    )
    p.add_argument("--race-summary", required=True, help="Path to oh_responsiveness_race_summary.csv")
    p.add_argument("--ensemble-seats", required=True, help="Path to oh_responsiveness_ensemble_seats.csv")
    p.add_argument("--official-seats", required=True, help="Path to oh_responsiveness_official_plan_seats.csv")
    p.add_argument("--output", required=True, help="Output PNG/PDF path")
    p.add_argument("--title", default="Ohio congressional responsiveness across statewide elections")
    p.add_argument("--n-districts", type=int, default=15)
    p.add_argument("--zoom-panels", type=int, default=4)
    p.add_argument("--max-races-per-zoom", type=int, default=6)
    return p.parse_args()


def build_chunks(races: pd.DataFrame, zoom_panels: int, max_races_per_zoom: int) -> list[pd.DataFrame]:
    n = len(races)
    target_panels = max(1, min(zoom_panels, int(np.ceil(n / max_races_per_zoom)) if max_races_per_zoom > 0 else zoom_panels))
    target_panels = max(target_panels, int(np.ceil(n / max_races_per_zoom))) if max_races_per_zoom > 0 else target_panels
    target_panels = min(max(1, target_panels), n)
    idx_chunks = np.array_split(np.arange(n), target_panels)
    return [races.iloc[idx].copy() for idx in idx_chunks if len(idx) > 0]


def draw_ridgeline_panel(
    ax,
    races_sub: pd.DataFrame,
    ensemble: pd.DataFrame,
    official: pd.DataFrame,
    n_districts: int,
    *,
    show_labels: bool = True,
    show_proportionality: bool = False,
    dot_size: float = 72,
    label_fontsize: float = 10.0,
):
    if races_sub.empty:
        return

    yvals = races_sub["statewide_dem_share"].to_numpy(float)
    y_sorted = np.sort(np.unique(yvals))
    if len(y_sorted) > 1:
        gaps = np.diff(y_sorted)
        positive_gaps = gaps[gaps > 1e-9]
        if len(positive_gaps):
            ridge_height = min(0.0075, max(0.0025, 0.45 * np.percentile(positive_gaps, 25)))
        else:
            ridge_height = 0.0035
    else:
        ridge_height = 0.0035

    plan_offsets = {"Enacted ('25)": -0.10, "Overturned ('21)": 0.00, "Remedial ('22)": 0.10}
    edges = np.arange(-0.5, n_districts + 1.5, 1.0)

    for _, race in races_sub.iterrows():
        race_id = race["race_id"]
        y = float(race["statewide_dem_share"])
        sub = ensemble[ensemble["race_id"] == race_id]
        if sub.empty:
            continue

        counts = (
            sub["dem_seats"]
            .value_counts()
            .reindex(range(0, n_districts + 1), fill_value=0)
            .sort_index()
        )
        fractions = counts / max(counts.sum(), 1)
        max_fraction = float(fractions.max()) if len(fractions) else 0.0
        if max_fraction <= 0:
            continue
        heights = fractions.to_numpy(float) / max_fraction * ridge_height

        # baseline
        ax.hlines(y, xmin=-0.25, xmax=n_districts + 0.35, color=BASELINE_COLOR, linewidth=0.8, alpha=0.75, zorder=1)

        # connected histogram shape
        ax.stairs(
            values=y + heights,
            edges=edges,
            baseline=y,
            fill=True,
            facecolor=HIST_FACE,
            edgecolor=HIST_EDGE,
            linewidth=1.0,
            alpha=0.95,
            zorder=2,
        )

        # official plans as colored dots
        off = official[official["race_id"] == race_id]
        for plan_label, style in PLAN_STYLES.items():
            row = off[off["source"] == plan_label]
            if row.empty:
                continue
            x = float(row["dem_seats"].iloc[0]) + plan_offsets.get(plan_label, 0.0)
            ax.scatter(
                x,
                y,
                s=dot_size,
                color=style["color"],
                edgecolor="black",
                linewidth=0.6,
                zorder=5,
            )

        if show_labels:
            ax.text(
                n_districts + 0.42,
                y + 0.0003,
                race["abbr"],
                fontsize=label_fontsize,
                ha="left",
                va="center",
            )

    ymin = max(0.0, float(races_sub["statewide_dem_share"].min()) - ridge_height * 1.8)
    ymax = min(1.0, float(races_sub["statewide_dem_share"].max()) + ridge_height * 3.2)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.4, n_districts + 2.3)
    ax.set_xticks(range(0, n_districts + 1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(axis="both", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_proportionality:
        yline = np.linspace(ymin, ymax, 200)
        ax.plot(n_districts * yline, yline, zorder=0, **PROPORTIONALITY_STYLE)


def add_key(ax):
    key_lines = [
        "PR  - President",
        "USS - U.S. Senate",
        "GOV - Governor",
        "ATG - Attorney General",
        "AUD - Auditor",
        "SOS - Secretary of State",
        "TRE - Treasurer",
        "CJU - Chief Justice",
        "SC  - Supreme Court",
    ]
    ax.text(
        0.99,
        0.02,
        "\n".join(key_lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={"boxstyle": "square,pad=0.30", "facecolor": "white", "edgecolor": "0.45", "alpha": 0.96},
    )


def plot_from_csv(races: pd.DataFrame, ensemble: pd.DataFrame, official: pd.DataFrame, output: Path, title: str, n_districts: int, zoom_panels: int, max_races_per_zoom: int):
    races = races.copy().sort_values(["statewide_dem_share", "year", "race_id"]).reset_index(drop=True)
    chunks = build_chunks(races, zoom_panels=zoom_panels, max_races_per_zoom=max_races_per_zoom)
    n_zoom = len(chunks)

    fig = plt.figure(figsize=(15.5, max(10.5, 2.6 * n_zoom)))
    gs = GridSpec(n_zoom, 2, width_ratios=[1.10, 1.90], height_ratios=[1] * n_zoom, wspace=0.28, hspace=0.32, figure=fig)

    ax_over = fig.add_subplot(gs[:, 0])
    draw_ridgeline_panel(
        ax_over, races, ensemble, official, n_districts,
        show_labels=False, show_proportionality=True, dot_size=76, label_fontsize=9.5,
    )
    ax_over.set_title("Overview", fontsize=13)
    ax_over.set_xlabel("Democrats elected")
    ax_over.set_ylabel("Statewide Democratic vote share")

    handles = [
        Line2D([0], [0], color=HIST_EDGE, marker="s", linestyle="None", markerfacecolor=HIST_FACE,
               markeredgecolor=HIST_EDGE, markersize=10, label="Compliant ensemble histogram"),
    ]
    for label, style in PLAN_STYLES.items():
        handles.append(
            Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=style["color"], markeredgecolor="black", markersize=9, label=label)
        )
    ax_over.legend(handles=handles, loc="upper left", frameon=True)

    # Draw zoom panels covering all races.
    for i, chunk in enumerate(chunks):
        ax = fig.add_subplot(gs[i, 1])
        draw_ridgeline_panel(
            ax, chunk, ensemble, official, n_districts,
            show_labels=True, show_proportionality=False, dot_size=74, label_fontsize=11.0,
        )
        lo = chunk["statewide_dem_share"].min()
        hi = chunk["statewide_dem_share"].max()
        ax.set_title(f"Zoom {i+1}: vote-share range {lo:.1%} to {hi:.1%}", fontsize=12)
        if i == n_zoom - 1:
            ax.set_xlabel("Democrats elected")
        else:
            ax.set_xlabel("")
        ax.set_ylabel("")
        add_key(ax) if i == n_zoom - 1 else None

    fig.suptitle(title, fontsize=19, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    races = pd.read_csv(args.race_summary)
    ensemble = pd.read_csv(args.ensemble_seats)
    official = pd.read_csv(args.official_seats)
    output = Path(args.output)
    plot_from_csv(
        races, ensemble, official, output,
        title=args.title,
        n_districts=args.n_districts,
        zoom_panels=args.zoom_panels,
        max_races_per_zoom=args.max_races_per_zoom,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
