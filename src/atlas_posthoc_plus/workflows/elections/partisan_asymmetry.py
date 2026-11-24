"""Compare partisan asymmetry in the same election's district margins across three plans."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from atlas_posthoc_plus.paper import (
    OFFICIAL_PLANS_TITLE,
    apply_paper_style,
    figure_legend,
    official_plan_style,
    paper_mode_enabled,
    save_figure,
    set_panel_title,
)

VOTE_COLUMNS = ["dem_votes", "rep_votes", "total_votes"]
PLAN_STYLES = [
    # Marker sizes and draw order keep plans visible when their margins coincide.
    {"marker": "o", "color": "#808080", "markersize": 8, "zorder": 3},
    {"marker": "s", "color": "#D99412", "markersize": 4, "zorder": 5},
    {"marker": "D", "color": "#23232A", "markersize": 6, "zorder": 4},
]


def read_margins(path: Path) -> pd.DataFrame:
    """Read district totals and compute all-vote D minus R margins in points."""
    votes = pd.read_csv(path, dtype={"plan": "string"}, keep_default_na=False)
    required = {"plan", "district", *VOTE_COLUMNS}
    if missing := required.difference(votes.columns):
        raise ValueError(f"District votes CSV is missing columns: {', '.join(sorted(missing))}")

    votes["plan"] = votes["plan"].str.strip()
    if votes["plan"].eq("").any() or votes["plan"].nunique() != len(PLAN_STYLES):
        raise ValueError("Expected three plans with nonempty names")
    districts = pd.to_numeric(votes["district"], errors="coerce")
    if not (np.isfinite(districts) & districts.gt(0) & districts.mod(1).eq(0)).all():
        raise ValueError("District numbers must be positive integers")
    votes["district"] = districts.astype(int)
    if votes.duplicated(["plan", "district"]).any():
        raise ValueError("Expected one row per plan and district")
    if not votes.groupby("district")["plan"].nunique().eq(len(PLAN_STYLES)).all():
        raise ValueError("Each district number must occur on all three plans")

    votes[VOTE_COLUMNS] = votes[VOTE_COLUMNS].apply(pd.to_numeric, errors="coerce").astype(float)
    if not (np.isfinite(votes[VOTE_COLUMNS]) & votes[VOTE_COLUMNS].ge(0)).to_numpy().all():
        raise ValueError("Vote counts must be finite and nonnegative")
    major_party_votes = votes["dem_votes"] + votes["rep_votes"]
    total_votes = votes["total_votes"]
    rounding_tolerance = total_votes * 1e-12
    if (total_votes <= 0).any() or (major_party_votes - total_votes > rounding_tolerance).any():
        raise ValueError("Total votes must be positive and include both major parties' votes")
    votes["margin_pp"] = 100 * ((votes["dem_votes"] - votes["rep_votes"]) / total_votes)
    return votes


def summarize_margins(votes: pd.DataFrame, close_pp: float) -> pd.DataFrame:
    """Count close wins and report the margin at the chamber majority cutoff."""
    records = []
    for plan, group in votes.groupby("plan", sort=False):
        margins = group["margin_pp"].sort_values(ascending=False)
        dem_wins = margins[margins > 0]
        rep_wins = -margins[margins < 0]
        dem_close = int((dem_wins <= close_pp).sum())
        rep_close = int((rep_wins <= close_pp).sum())
        dem_close_share = dem_close / len(dem_wins) if len(dem_wins) else np.nan
        rep_close_share = rep_close / len(rep_wins) if len(rep_wins) else np.nan
        majority = len(margins) // 2 + 1
        records.append(
            {
                "plan": plan,
                "districts": len(margins),
                "dem_wins": len(dem_wins),
                "rep_wins": len(rep_wins),
                "ties": int((margins == 0).sum()),
                "close_threshold_pp": close_pp,
                "dem_wins_within_threshold": dem_close,
                "rep_wins_within_threshold": rep_close,
                "dem_close_win_share": dem_close_share,
                "rep_close_win_share": rep_close_share,
                "dem_minus_rep_close_win_share_pp": 100 * (dem_close_share - rep_close_share),
                "closest_dem_win_pp": dem_wins.min(),
                "closest_rep_win_pp": rep_wins.min(),
                "seats_needed_for_majority": majority,
                "majority_seat_margin_pp": margins.iloc[majority - 1],
            }
        )
    return pd.DataFrame(records)


def plot_margins(
    votes: pd.DataFrame, *, close_pp: float, output: Path, title: str | None = None
) -> Path:
    apply_paper_style()
    districts = sorted(votes["district"].unique())
    rows = np.arange(len(districts))
    figure, (overview, zoom) = plt.subplots(
        1, 2, sharey=True, figsize=(9.8, max(3.6, 2.0 + len(districts) * 0.27)),
        gridspec_kw={"width_ratios": [1.5, 1]},
    )
    try:
        figure.subplots_adjust(left=0.08, right=0.98, bottom=0.19, top=0.85, wspace=0.13)
        ranges = votes.groupby("district")["margin_pp"].agg(["min", "max"]).loc[districts]
        overview.hlines(rows, ranges["min"], ranges["max"], color="#C8C8C8", linewidth=1.6)
        zoom.axvspan(-close_pp, 0, color="#FBF1EE", zorder=0)
        zoom.axvspan(0, close_pp, color="#EAF2FA", zorder=0)
        for axis in (overview, zoom):
            axis.set_axisbelow(True)
            axis.axvline(0, color="#292929", linewidth=1.0, zorder=2)
            axis.set_xlabel("Margin, Democratic minus Republican (percentage points)")
            axis.xaxis.set_major_locator(MaxNLocator(nbins=6, symmetric=axis is zoom))
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(axis="y", length=0)

        for (plan, group), style in zip(votes.groupby("plan", sort=False), PLAN_STYLES, strict=True):
            style = official_plan_style(plan, **style)
            margins = group.set_index("district").loc[districts, "margin_pp"]
            overview.plot(margins, rows, linestyle="none", label=plan, **style)
            near = margins.abs().le(close_pp).to_numpy()
            zoom.plot(margins[near], rows[near], linestyle="none", **style)

        overview.set_yticks(rows, labels=districts)
        overview.set_ylim(len(rows) - 0.5, -0.5)
        overview.set_ylabel("District number")
        low = min(-close_pp, ranges["min"].min())
        high = max(close_pp, ranges["max"].max())
        padding = (high - low) * 0.05
        overview.set_xlim(low - padding, high + padding)
        zoom.set_xlim(-close_pp, close_pp)
        if title:
            figure.suptitle(title, x=0.08, ha="left", fontsize=13)
        legend_placement = {"loc": "lower center"}
        if paper_mode_enabled():
            set_panel_title(overview, "Full range")
            set_panel_title(zoom, f"Margins within ±{close_pp:g} percentage points")
            legend_placement = {
                "loc": "upper center", "bbox_to_anchor": (0.5, 0.06), "title": OFFICIAL_PLANS_TITLE,
            }
        handles, labels = overview.get_legend_handles_labels()
        figure_legend(
            figure, handles, labels, ncol=len(PLAN_STYLES), frameon=False, **legend_placement,
        )
        return save_figure(figure, output, dpi=190, bbox_inches="tight")
    finally:
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--district-votes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--close-pp", type=float, default=5, help="Zoom and close-win threshold")
    parser.add_argument("--title")
    args = parser.parse_args()
    if not math.isfinite(args.close_pp) or not 0 < args.close_pp <= 100:
        parser.error("--close-pp must be greater than 0 and at most 100")
    try:
        votes = read_margins(args.district_votes)
    except ValueError as error:
        parser.error(str(error))
    summary = summarize_margins(votes, args.close_pp)
    destination = plot_margins(votes, close_pp=args.close_pp, output=args.output, title=args.title)
    summary_path = destination.with_name(f"{destination.stem}-summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {destination}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
