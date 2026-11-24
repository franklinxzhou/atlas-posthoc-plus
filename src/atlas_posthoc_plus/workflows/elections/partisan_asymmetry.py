"""Compare district margins and score partisan asymmetry from reciprocal seat curves."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox

from atlas_posthoc_plus.paper import (
    OFFICIAL_PLANS_TITLE,
    apply_paper_style,
    figure_legend,
    official_plan_style,
    paper_mode_enabled,
    save_figure,
)
from atlas_posthoc_plus.workflows.elections.partisan_swing import (
    chain_votes,
    ensemble_votes,
    official_votes,
    read_vote_table,
    write_swing_analysis,
)

VOTE_COLUMNS = ["dem_votes", "rep_votes", "total_votes"]
PLAN_LEGEND_LABELS = {
    "Enacted ('25)": "Enacted (25')",
    "Overturned ('21)": "Overturned (21')",
    "Remedial ('22)": "Remedial (22')",
}
PLAN_STYLES = [
    # Marker sizes and draw order keep plans visible when their margins coincide.
    {"color": "#808080", "markersize": 9, "zorder": 3},
    {"color": "#D99412", "markersize": 4, "zorder": 5},
    {"color": "#23232A", "markersize": 6.5, "zorder": 4},
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


def _zoom_limit(margins: pd.Series) -> float:
    """Round up the 75th percentile of absolute margins to a readable bound."""
    limit = max(1.0, float(margins.abs().quantile(0.75)))
    step = 0.5 * 10 ** math.floor(math.log10(limit))
    return min(100.0, math.ceil(limit / step) * step)


def _place_corner_legend(axis: Axes, legend: Legend) -> None:
    """Use a clear lower corner, adding bottom space only when both are occupied."""
    axis.figure.draw_without_rendering()
    pixels_per_point = axis.figure.dpi / 72
    obstacles = []
    for line in axis.lines:
        radius = (line.get_markersize() + line.get_markeredgewidth()) * pixels_per_point / 2
        for x, y in axis.transData.transform(line.get_xydata()):
            obstacles.append(Bbox.from_extents(x - radius, y - radius, x + radius, y + radius))
    for collection in axis.collections:
        for segment in collection.get_segments():
            bounds = Bbox.null()
            bounds.update_from_data_xy(axis.transData.transform(segment))
            obstacles.append(bounds)

    for location in ("lower left", "lower right"):
        legend.set_loc(location)
        box = legend.get_window_extent().padded(pixels_per_point)
        if not any(box.overlaps(obstacle) for obstacle in obstacles):
            return

    bottom, top = axis.get_ylim()
    legend_space = box.ymax - axis.bbox.ymin
    extra = (bottom - top) * legend_space / (axis.bbox.height - legend_space)
    axis.set_ylim(bottom + extra, top)


def plot_margins(
    votes: pd.DataFrame, *, output: Path, title: str | None = None
) -> tuple[Path, Path]:
    apply_paper_style()
    output = output.with_suffix(".pdf")
    zoom_output = output.with_stem(f"{output.stem}-zoom")
    districts = sorted(votes["district"].unique())
    rows = np.arange(len(districts))
    ranges = votes.groupby("district")["margin_pp"].agg(["min", "max"]).loc[districts]
    zoom_pp = _zoom_limit(votes["margin_pp"])
    low = min(-zoom_pp, ranges["min"].min())
    high = max(zoom_pp, ranges["max"].max())
    padding = (high - low) * 0.05
    panels = (
        (output, (low - padding, high + padding)),
        (zoom_output, (-zoom_pp, zoom_pp)),
    )
    for destination, limits in panels:
        is_zoom = destination == zoom_output
        figure, axis = plt.subplots(
            figsize=(6.0, max(3.2, 1.35 + len(rows) * 0.24)), layout="constrained",
            facecolor="white",
        )
        try:
            axis.set_facecolor("white")
            axis.grid(False, which="both")
            axis.spines[:].set_visible(True)
            axis.vlines(0, -0.5, len(rows) - 0.5, color="#555555", linewidth=0.8, zorder=2)
            axis.hlines(rows, ranges["min"], ranges["max"], color="#C8C8C8", linewidth=1.6)
            axis.set_xlabel("Margin, Democratic minus Republican (percentage points)")
            axis.set_ylabel("District number")
            axis.set_xlim(*limits)
            axis.set_yticks(rows, labels=districts)
            axis.set_ylim(len(rows) - 0.5, -0.5)
            axis.xaxis.set_major_locator(
                MaxNLocator(nbins=6, symmetric=is_zoom, steps=[1, 2, 2.5, 5, 10])
            )
            axis.tick_params(axis="y", length=0)

            for (plan, group), style in zip(
                votes.groupby("plan", sort=False), PLAN_STYLES, strict=True
            ):
                margins = group.set_index("district").loc[districts, "margin_pp"]
                visible = margins.between(*limits).to_numpy()
                axis.plot(
                    margins[visible], rows[visible], linestyle="none", marker="o", label=plan,
                    markeredgecolor="#303030", markeredgewidth=0.4,
                    **official_plan_style(plan, **style),
                )

            if title and not paper_mode_enabled():
                figure.suptitle(title)
            handles, labels = axis.get_legend_handles_labels()
            legend = figure_legend(
                axis, handles, labels, loc="lower left", alignment="left",
                title=OFFICIAL_PLANS_TITLE, framealpha=1, facecolor="white", edgecolor="#808892",
                fancybox=False, borderaxespad=0.45, borderpad=0.3, labelspacing=0.25,
                handlelength=1.1, handletextpad=0.45,
            )
            for label in legend.get_texts():
                label.set_text(PLAN_LEGEND_LABELS.get(label.get_text(), label.get_text()))
            legend.set_frame_on(True)
            legend.get_frame().set_linewidth(0.55)
            _place_corner_legend(axis, legend)
            save_figure(
                figure, destination, facecolor="white", bbox_inches="tight",
                preserve_legend_frames=True,
            )
        finally:
            plt.close(figure)
    return output, zoom_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--district-votes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Full-range PDF path")
    parser.add_argument("--close-pp", type=float, default=5, help="Close-win summary threshold")
    parser.add_argument("--title")
    parser.add_argument("--official-swing-votes", type=Path,
                        help="Optional district vote table for the three official swing curves")
    parser.add_argument("--ensemble-district-votes", type=Path, nargs="+",
                        help="Preaggregated vote CSVs containing every ensemble plan to score")
    parser.add_argument("--ensemble-node-votes", type=Path,
                        help="Already allocated graph-node vote CSV for scoring AtlasIO maps")
    parser.add_argument("--ensemble-graph", type=Path)
    parser.add_argument("--ensemble-chains", type=Path, nargs="+")
    args = parser.parse_args()
    if not math.isfinite(args.close_pp) or not 0 < args.close_pp <= 100:
        parser.error("--close-pp must be greater than 0 and at most 100")
    chain_args = (args.ensemble_node_votes, args.ensemble_graph, args.ensemble_chains)
    if any(chain_args) and not all(chain_args):
        parser.error("AtlasIO scoring requires --ensemble-node-votes, --ensemble-graph, and "
                     "--ensemble-chains")
    if args.ensemble_district_votes and all(chain_args):
        parser.error("Supply either ensemble district vote CSVs or AtlasIO inputs")
    try:
        votes = read_margins(args.district_votes)
        swing_votes = (
            read_vote_table(args.official_swing_votes, ["plan", "district"])
            if args.official_swing_votes else votes
        )
        ensemble = None
        if args.ensemble_district_votes:
            ensemble = ensemble_votes(args.ensemble_district_votes)
        elif all(chain_args):
            ensemble = chain_votes(args.ensemble_graph, args.ensemble_node_votes,
                                   args.ensemble_chains)
        swing_output = args.output.with_name(f"{args.output.stem}-swing")
        write_swing_analysis(swing_output, official_votes(swing_votes), ensemble)
    except ValueError as error:
        parser.error(str(error))
    summary = summarize_margins(votes, args.close_pp)
    destination, zoom = plot_margins(votes, output=args.output, title=args.title)
    summary_path = destination.with_name(f"{destination.stem}-summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {destination}")
    print(f"Wrote {zoom}")
    print(f"Wrote {summary_path}")
    print(f"Wrote swing PDFs and CSVs in {swing_output}")


if __name__ == "__main__":
    main()
