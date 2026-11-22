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
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from atlas_posthoc_plus.paper import figure_legend, official_plan_style, save_figure
from atlas_posthoc_plus.workflows.elections.responsiveness_zoom import (
    PLAN_STYLES,
    apply_figure_style,
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

    Boxes span the middle 50% of submitted plans, with a median line and
    minimum-to-maximum whiskers. Every plan contributes once at every rank.
    """
    required = {"plan_id", "plan_group", "plot_role", "rank", "dem_share"}
    missing = required.difference(external_long.columns)
    if missing:
        raise ValueError(f"Ordered vote vectors are missing columns: {sorted(missing)}")
    if n_districts < 1:
        raise ValueError("n_districts must be positive")
    is_public = external_long["plot_role"].str.lower() == "public"
    public = _vote_matrix(external_long.loc[is_public], n_districts)
    official_rows = external_long.loc[~is_public]
    officials = {}
    for label in PLAN_STYLES:
        rows = official_rows.loc[official_rows["plan_group"] == label]
        if rows["plan_id"].nunique() != 1:
            raise ValueError(f"Expected exactly one official plan labelled {label!r}")
        officials[label] = _vote_matrix(rows, n_districts).iloc[0].to_numpy(float)

    with plt.rc_context():
        apply_figure_style()
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        x = np.arange(1, n_districts + 1)
        face, edge = "#dce4e8", "#667680"
        ax.boxplot(
            public.to_numpy(float),
            positions=x,
            widths=0.55,
            whis=(0, 100),
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": face, "edgecolor": edge, "linewidth": 0.9},
            whiskerprops={"color": edge, "linewidth": 0.9},
            capprops={"color": edge, "linewidth": 0.9},
            medianprops={"color": "#263238", "linewidth": 1.3},
            zorder=2,
        )
        ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1.1, zorder=1)
        handles = [Patch(facecolor=face, edgecolor=edge,
                         label=f"Public submitted plans (n = {len(public)})")]
        line_styles = ["-", "--", "-."]
        marker_sizes = [7.2, 6.0, 4.8]
        for index, (label, values) in enumerate(officials.items()):
            style = official_plan_style(label, linestyle=line_styles[index], **PLAN_STYLES[label])
            line, = ax.plot(
                x, values,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.6,
                markersize=marker_sizes[index],
                markerfacecolor="white",
                markeredgewidth=1.2,
                label=label,
                zorder=3 + index,
            )
            handles.append(line)

        values = np.concatenate([public.to_numpy(float).ravel(), *officials.values(), [0.5]])
        padding = max(0.015, float(np.ptp(values)) * 0.045)
        lo, hi = max(0, float(values.min()) - padding), min(1, float(values.max()) + padding)
        if y_limits is not None:
            lo, hi = min(lo, y_limits[0]), max(hi, y_limits[1])
        ax.set_ylim(lo, hi)
        ax.set_yticks([tick for tick in ax.get_yticks() if lo <= tick <= hi])
        ax.set_xlim(0.4, n_districts + 0.6)
        ax.set_xticks(x)
        ax.set_xlabel("District rank, from most Republican to most Democratic")
        ax.set_ylabel("Democratic two-party vote share")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_title(title, pad=10)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
        figure_legend(ax, handles=handles, loc="upper left", frameon=False)
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
