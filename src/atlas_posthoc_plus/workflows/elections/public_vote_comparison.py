"""Ordered public-plan vote-share boxes with the three official plans overlaid.

Render saved vectors without rescoring plans with::

    python -m atlas_posthoc_plus.workflows.elections.public_vote_comparison \
        --ordered-votes outputs/vote-comparison-adjusted/external_plan_ordered_vote_vectors.csv \
        --output outputs/vote-comparison-adjusted/oh_public_plans_ordered_vote_shares.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from atlas_posthoc.plotting.style import format_percent_axis, plot_style
from matplotlib.patches import Patch

from atlas_posthoc_plus.paper import OFFICIAL_PLAN_STYLES, save_figure
from atlas_posthoc_plus.workflows.elections.vote_comparison_plot import (
    POOLED_BOX_WIDTH,
    format_vote_comparison,
    plot_reference_plan,
)


def _vote_matrix(rows: pd.DataFrame, n_districts: int) -> pd.DataFrame:
    """Require one finite, correctly ordered vote vector per submitted plan."""
    if rows.empty:
        raise ValueError("No public submitted plans found in the ordered vote vectors")
    rows = rows.copy()
    rows["rank"] = pd.to_numeric(rows["rank"], errors="raise")
    rows["dem_share"] = pd.to_numeric(rows["dem_share"], errors="raise")
    if rows["plan_id"].isna().any() or rows.duplicated(["plan_id", "rank"]).any():
        raise ValueError("Each plan must have exactly one vote share per district rank")
    ranks = np.arange(1, n_districts + 1)
    matrix = rows.pivot(index="plan_id", columns="rank", values="dem_share")
    if not np.array_equal(matrix.columns.to_numpy(), ranks) or matrix.isna().any().any():
        raise ValueError(f"Each plan must contain all district ranks 1 through {n_districts}")
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("District Democratic vote shares must be finite and between 0 and 1")
    if (np.diff(values, axis=1) < 0).any():
        raise ValueError("District vote shares must increase from most Republican to most Democratic")
    return matrix


def plot_public_plan_comparison(
    external_long: pd.DataFrame,
    *,
    n_districts: int,
    output_path: Path,
    title: str = "Ohio congressional plans under the 2020 presidential vote",
    y_limits: tuple[float, float] | None = None,
) -> Path:
    """Summarize public plans descriptively; official plans never enter the boxes.

    Boxes span the middle 50% of submitted plans, with a median line, mean
    marker, and minimum-to-maximum whiskers. Each plan contributes once per rank.
    """
    required = {"plan_id", "plan_group", "plot_role", "rank", "dem_share"}
    missing = required.difference(external_long.columns)
    if missing:
        raise ValueError(f"Ordered vote vectors are missing columns: {sorted(missing)}")
    if n_districts < 1:
        raise ValueError("n_districts must be positive")
    is_public = external_long["plot_role"].str.lower() == "public"
    public = _vote_matrix(external_long.loc[is_public], n_districts).to_numpy(float)
    official_rows = external_long.loc[~is_public]
    officials = {}
    for label in OFFICIAL_PLAN_STYLES:
        rows = official_rows.loc[official_rows["plan_group"] == label]
        if rows["plan_id"].nunique() != 1:
            raise ValueError(f"Expected exactly one official plan labelled {label!r}")
        officials[label] = _vote_matrix(rows, n_districts).iloc[0].to_numpy(float)

    with plot_style() as style:
        fig, ax = plt.subplots(figsize=style.figsize_wide, facecolor="white")
        ax.set_facecolor("white")

    x = np.arange(1, n_districts + 1)
    face, edge = "#dce4e8", (0, 0, 0, 0.60)
    ax.boxplot(
        public,
        positions=x,
        widths=POOLED_BOX_WIDTH,
        whis=(0, 100),
        patch_artist=True,
        showfliers=False,
        showmeans=True,
        manage_ticks=False,
        boxprops={"facecolor": face, "edgecolor": edge, "linewidth": 0.8},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        medianprops={"color": "black", "linewidth": 1.0},
        meanprops={
            "marker": "o",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 3,
        },
    )
    ax.axhline(0.5, color="black", linewidth=0.8)
    handles = [Patch(facecolor=face, edgecolor=edge, label="Public submitted plans")]
    for index, (label, values) in enumerate(officials.items()):
        handles.append(plot_reference_plan(ax, x, values, label, index))

    values = np.concatenate([public.ravel(), *officials.values(), [0.5]])
    padding = max(0.015, float(np.ptp(values)) * 0.045)
    lo, hi = max(0, float(values.min()) - padding), min(1, float(values.max()) + padding)
    if y_limits is not None:
        lo, hi = min(lo, y_limits[0]), max(hi, y_limits[1])
    ax.set_ylim(lo, hi)
    format_percent_axis(ax)
    ax.set_title(title)
    format_vote_comparison(ax, x, handles=handles)
    fig.tight_layout()
    destination = save_figure(fig, output_path, dpi=300)
    plt.close(fig)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordered-votes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-districts", type=int, default=15)
    parser.add_argument("--title", default="Ohio congressional plans under the 2020 presidential vote")
    args = parser.parse_args()
    output = plot_public_plan_comparison(
        pd.read_csv(args.ordered_votes),
        n_districts=args.n_districts,
        output_path=args.output,
        title=args.title,
    )
    print(f"Wrote public-plan ordered vote-share plot: {output}")


if __name__ == "__main__":
    main()
