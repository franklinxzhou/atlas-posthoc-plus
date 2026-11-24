"""Render CEAH overviews and linked vote-share detail panels."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Circle, Patch, Rectangle, Wedge
from matplotlib.ticker import PercentFormatter
from matplotlib.transforms import Bbox

from atlas_posthoc_plus.paper import (
    PAPER_SANS_SERIF,
    apply_paper_style,
    figure_legend,
    paper_mode_enabled,
    save_figure,
)

PLAN_COLORS = {
    "Overturned ('21)": "#9367a3",
    "Remedial ('22)": "#429369",
    "Enacted ('25)": "#ce4840",
}
SEAT_LABEL = "Number of Democrats elected"
VOTE_LABEL = "Statewide two-party Democratic vote (%)"

INK = "#232a36"
FRAME = "#808892"
BASELINE = "#a1a7b0"
HIST_FACE = "#a5b3df"
HIST_EDGE = "#3c4c73"
ZOOM_EDGE = "#b94d46"
DOT_SIZE = 4.8
LABEL_SIZE = 8


def _panel_geometry(ax, shares, *, seat_range, overview, legend_headroom):
    """Set the panel's scale and return a single ridge height for its races."""
    span = max(float(np.ptp(shares)), 0.004)
    gaps = np.diff(np.unique(shares))
    height = min(0.008, 0.6 * np.percentile(gaps, 35)) if len(gaps) else 0.0016
    height = max(height, 0.022 * span)
    pad = max(0.085 * span, 0.8 * height)
    bottom = shares[0] - pad
    peak = shares[-1] + height
    height_pt = ax.get_position().height * ax.figure.get_figheight() * 72
    headroom = (peak - bottom) * legend_headroom / (height_pt - legend_headroom)
    ax.set_ylim(bottom, peak + max(pad, headroom))

    lo, hi = seat_range
    # Zoom panels leave room for a race code beside the outermost seat bin.
    width_pt = ax.get_position().width * ax.figure.get_figwidth() * 72
    bin_margin = 0.5 if overview else 0.64
    label_space_pt = 0 if overview else 34
    points_per_seat = (width_pt - label_space_pt) / (hi - lo + 2 * bin_margin)
    ax.set_xlim(lo - bin_margin, hi + bin_margin + label_space_pt / points_per_seat)
    tick_spacing = 12 if hi >= 10 else 8
    step = max(1, int(np.ceil(tick_spacing / points_per_seat)))
    ticks = list(range(lo, hi + 1, step))
    if ticks[-1] != hi:
        if len(ticks) > 1 and hi - ticks[-1] <= step / 2:
            ticks.pop()
        ticks.append(hi)
    ax.set_xticks(ticks)
    ax.tick_params(which="major", length=3)
    ax.tick_params(which="minor", length=1.6, color=FRAME)
    if hi - lo <= 30:
        ax.set_xticks(np.arange(lo, hi + 1), minor=True)
    ylo, yhi = ax.get_ylim()
    min_gap = (yhi - ylo) * 13 / height_pt
    ticks = []
    for share in np.unique(shares):
        if not ticks or share - ticks[-1] >= min_gap:
            ticks.append(share)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2, symbol=""))
    ax.tick_params(axis="both", color=FRAME, pad=3.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(FRAME)
        spine.set_linewidth(0.6)
    return height


def _plot_ridge(ax, seats, counts, share, height_per_count):
    occupied = np.flatnonzero(counts)
    start, stop = occupied[0], occupied[-1] + 1
    ax.stairs(
        share + height_per_count * counts[start:stop],
        np.arange(seats[start] - 0.5, seats[stop - 1] + 1.5),
        baseline=share,
        fill=True,
        facecolor=HIST_FACE,
        edgecolor=HIST_EDGE,
        linewidth=0.65,
        alpha=0.85,
        zorder=3,
    )


def _draw_plan_dots(ax, outcomes, y, *, compact):
    """Split one circle into equal sectors where official plans share an outcome."""
    by_seat = {}
    for label, color in PLAN_COLORS.items():
        if label in outcomes:
            by_seat.setdefault(outcomes[label], []).append(color)
    size = DOT_SIZE * (0.8 if compact else 1)
    for seat, colors in by_seat.items():
        dot = DrawingArea(size, size)
        radius = size / 2
        edges = np.linspace(90, 450, len(colors) + 1)
        for color, start, stop in zip(colors, edges[:-1], edges[1:]):
            dot.add_artist(Wedge(
                (radius, radius), radius, start, stop, facecolor=color,
                edgecolor="white" if len(colors) > 1 else "none", linewidth=0.28,
            ))
        dot.add_artist(Circle((radius, radius), radius, fill=False, edgecolor=INK, linewidth=0.45))
        badge = ax.add_artist(AnnotationBbox(
            dot, (seat, y), pad=0, frameon=False, zorder=5,
        ))
        yield badge


def _race_labels(ax, rows, obstacles):
    """Place codes at ridge ends, using the other side when the first would collide."""
    renderer = ax.figure.canvas.get_renderer()
    pixels_per_point = renderer.points_to_pixels(1)
    font = FontProperties(size=LABEL_SIZE, family=plt.rcParams["font.family"])
    frame = ax.get_window_extent().padded(-2 * pixels_per_point)
    # Try both sides at the actual vote share before checking small vertical shifts.
    shifts = [0]
    for shift in range(2, int(frame.height / pixels_per_point), 2):
        shifts.extend((-shift, shift))

    for code, start, stop, share in rows:
        width, height, _ = renderer.get_text_width_height_descent(code, font, ismath=False)
        height = max(height, LABEL_SIZE * pixels_per_point * 0.85)
        for shift in shifts:
            for x, direction in ((stop, 1), (start, -1), (stop, -1), (start, 1)):
                px, py = ax.transData.transform((x, share))
                px += direction * 3 * pixels_per_point
                py += shift * pixels_per_point
                box = Bbox.from_extents(
                    px - (width if direction < 0 else 0), py - height / 2,
                    px + (width if direction > 0 else 0), py + height / 2,
                ).padded(pixels_per_point)
                if not frame.contains(*box.min) or not frame.contains(*box.max):
                    continue
                if any(box.overlaps(obstacle) for obstacle in obstacles):
                    continue
                ax.annotate(
                    code, (x, share), xytext=(direction * 3, shift), textcoords="offset points",
                    ha="left" if direction > 0 else "right", va="center",
                    fontsize=LABEL_SIZE, color=INK, zorder=7,
                )
                obstacles.append(box)
                break
            else:
                continue
            break


def _data_box(ax, x0, y0, x1, y1, pad_pt=0):
    corners = ax.transData.transform([(x0, y0), (x1, y1)])
    return Bbox(corners).padded(ax.figure.dpi * pad_pt / 72)


def _draw_panel(
    ax, races, outcomes, official, *, seat_range, overview=False, detail_chunks=(), legend_headroom=0
):
    shares = races["statewide_dem_share"].to_numpy(float)
    lo, hi = seat_range
    seats = np.arange(lo, hi + 1)
    height = _panel_geometry(
        ax, shares, seat_range=seat_range, overview=overview, legend_headroom=legend_headroom,
    )
    label_rows = []
    obstacles = []
    renderer = ax.figure.canvas.get_renderer()
    for race, share in zip(races.itertuples(index=False), shares):
        reference = {
            label: seat for label, seat in official.get(race.race_id, {}).items()
            if lo <= seat <= hi
        }
        all_counts = outcomes.get(race.race_id, {})
        counts = {seat: count for seat, count in all_counts.items() if lo <= seat <= hi}
        occupied = counts.keys() | reference.values()
        start = min(occupied, default=seat_range[0]) - 0.5
        stop = max(occupied, default=seat_range[0]) + 0.5
        ridge_start, ridge_stop = start, stop
        if occupied:
            ax.hlines(share, start, stop, color=BASELINE, linewidth=0.55, zorder=2)
            obstacles.append(_data_box(ax, start, share, stop, share, pad_pt=0.4))

        if counts:
            ridge_start, ridge_stop = min(counts) - 0.5, max(counts) + 0.5
            heights = np.array([counts.get(seat, 0) for seat in seats])
            height_per_count = height / max(all_counts.values())
            _plot_ridge(ax, seats, heights, share, height_per_count)
            for seat, count in counts.items():
                peak = share + height_per_count * count
                obstacles.append(_data_box(ax, seat - 0.5, share, seat + 0.5, peak, pad_pt=0.4))

        dots = _draw_plan_dots(ax, reference, share, compact=bool(detail_chunks))
        for dot in dots:
            dot.update_positions(renderer)
            obstacles.append(dot.offsetbox.get_window_extent(renderer))
        label_rows.append((str(race.abbr), ridge_start, ridge_stop, share))

    # Keep the annotation itself clear of the red outline, without a background fill.
    x0, x1 = ax.get_xlim()
    for bounds in _zoom_ranges(ax, detail_chunks):
        for share in bounds:
            obstacles.append(_data_box(ax, x0, share, x1, share, pad_pt=0.5))
    legend = ax.get_legend()
    if legend is not None:
        obstacles.append(legend.get_window_extent(renderer).padded(2))
    _race_labels(ax, label_rows, obstacles)
    return obstacles


def _link_detail(fig, overview, detail, share_range):
    """Mark the election interval in the overview and link it to its detail panel."""
    lo, hi = share_range
    x0, x1 = overview.get_xlim()
    inset = (x1 - x0) * 0.012
    edge = x1 - inset
    overview.add_patch(Rectangle(
        (x0 + inset, lo), edge - x0 - inset, hi - lo,
        fill=False, edgecolor=ZOOM_EDGE, linewidth=0.75, zorder=0.5,
    ))
    for side, share in ((0, lo), (1, hi)):
        origin = overview.transData.transform((edge, share))
        target_x, target_y = detail.transAxes.transform((0, side))
        # Keep the sloped part of the connector outside the detail tick labels.
        elbow = (target_x - 29 * fig.dpi / 72, target_y)
        points = fig.transFigure.inverted().transform([origin, elbow, (target_x, target_y)])
        fig.add_artist(Line2D(
            *points.T, transform=fig.transFigure,
            color=ZOOM_EDGE, alpha=0.55, linewidth=0.55, zorder=-1,
        ))


def _zoom_ranges(overview, chunks):
    shares = [(chunk["statewide_dem_share"].min(), chunk["statewide_dem_share"].max())
              for chunk in chunks]
    bottom, top = overview.get_ylim()
    below = (top - bottom) * 0.012
    above = (top - bottom) * 0.035
    for i, (lo, hi) in enumerate(shares):
        lower_pad = min(below, 0.42 * (lo - shares[i + 1][1])) if i + 1 < len(shares) else below
        upper_pad = min(above, 0.42 * (shares[i - 1][0] - hi)) if i else above
        yield max(bottom, lo - lower_pad), min(top, hi + upper_pad)


def _add_legend(ax, obstacles, official, *, public, upper_left_only=False):
    handles = [Patch(
        facecolor=HIST_FACE, edgecolor=HIST_EDGE,
        label="Public submissions" if public else "Ensemble",
    )]
    present = {label for race in official.values() for label in race}
    handles.extend(
        Line2D(
            [], [], linestyle="none", marker="o", markersize=DOT_SIZE,
            markerfacecolor=color, markeredgecolor=INK, markeredgewidth=0.45, label=label,
        )
        for label, color in PLAN_COLORS.items() if label in present
    )
    options = {
        "handles": handles, "frameon": True, "framealpha": 1, "facecolor": "white",
        "edgecolor": FRAME, "fancybox": False, "borderaxespad": 0.25, "borderpad": 0.3,
        "labelspacing": 0.25, "handlelength": 0.9, "handletextpad": 0.45,
    }
    renderer = ax.figure.canvas.get_renderer()
    corners = ("upper left",) if upper_left_only else ("lower right", "upper left")
    for corner in corners:
        legend = figure_legend(ax, loc=corner, **options)
        legend.set_frame_on(True)
        legend.get_frame().set_linewidth(0.55)
        box = legend.get_window_extent(renderer).padded(2)
        if not any(box.overlaps(other) for other in obstacles):
            return 0.0
    return box.height / renderer.points_to_pixels(1) + 7


def _seat_counts(rows: pd.DataFrame, n_districts: int) -> dict[str, dict[int, int]]:
    counts = rows.loc[rows["dem_seats"].isin(range(n_districts + 1))]
    counts = counts.groupby("race_id", sort=False)["dem_seats"].value_counts()
    by_race: dict[str, dict[int, int]] = {}
    for (race_id, seat), count in counts.items():
        by_race.setdefault(race_id, {})[int(seat)] = int(count)
    return by_race


def seat_window(seats: Iterable[int], n_districts: int) -> tuple[int, int]:
    used = set(seats).intersection(range(n_districts + 1))
    if not used:
        return 0, n_districts
    return max(0, int(min(used)) - 1), int(max(used))


def render_ceah(
    races: pd.DataFrame,
    ensemble: pd.DataFrame,
    official: pd.DataFrame,
    output: Path,
    title: str | None,
    n_districts: int,
    *,
    chunks: tuple[pd.DataFrame, ...] = (),
    public_counts: Mapping[str, Mapping[int, int]] | None = None,
    detail_seat_windows: tuple[tuple[int, int], ...] = (),
) -> Path:
    """Render the CEAH design from computed or previously saved election rows."""
    if races.empty:
        raise ValueError("Cannot plot CEAH without races")
    if n_districts < 1:
        raise ValueError("Cannot plot CEAH without districts")
    public = public_counts is not None
    counts = public_counts if public else _seat_counts(ensemble, n_districts)
    reference: dict[str, dict[str, int]] = {}
    for row in official.itertuples(index=False):
        if row.source in PLAN_COLORS and row.dem_seats in range(n_districts + 1):
            reference.setdefault(row.race_id, {}).setdefault(row.source, int(row.dem_seats))

    zoomed = bool(chunks)
    overview_seats = (0, n_districts)
    if not detail_seat_windows:
        detail_seat_windows = tuple(
            seat_window(
                (
                    seat
                    for race_id in panel["race_id"]
                    for seat in counts.get(race_id, {}).keys() | reference.get(race_id, {}).values()
                ),
                n_districts,
            )
            for panel in chunks
        )
    heights = [max(1.0, 0.30 * len(chunk)) for chunk in chunks]
    width = 7.1 if zoomed else 4.8
    height = max(4.8, sum(heights) + 0.30 * len(chunks) + 0.8) if zoomed else max(
        4.8, 0.23 * len(races) + 1.2
    )
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": PAPER_SANS_SERIF,
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        apply_paper_style()
        fig = plt.figure(figsize=(width, height))
        try:
            gs = fig.add_gridspec(
                len(chunks) or 1, 2 if zoomed else 1,
                left=0.83 / width, right=1 - 0.20 / width,
                bottom=0.48 / height,
                top=1 - (0.5 if title and not paper_mode_enabled() else 0.12) / height,
                width_ratios=(0.95, 1.05) if zoomed else (1,),
                height_ratios=heights or (1,),
                wspace=0.28, hspace=0.29,
            )
            overview = fig.add_subplot(gs[:, 0])
            overview_obstacles = _draw_panel(
                overview, races, counts, reference,
                seat_range=overview_seats, overview=True, detail_chunks=chunks,
            )
            headroom = _add_legend(overview, overview_obstacles, reference, public=public)
            if headroom:
                overview.clear()
                _add_legend(overview, [], reference, public=public, upper_left_only=True)
                _draw_panel(
                    overview, races, counts, reference, seat_range=overview_seats,
                    overview=True, detail_chunks=chunks, legend_headroom=headroom,
                )
            overview.set_ylabel(VOTE_LABEL)
            overview.set_xlabel(SEAT_LABEL)
            details = zip(chunks, _zoom_ranges(overview, chunks), detail_seat_windows)
            for i, (chunk, share_range, seats) in enumerate(details, 1):
                detail = fig.add_subplot(gs[i - 1, 1])
                _draw_panel(
                    detail, chunk, counts, reference,
                    seat_range=seats,
                )
                if i == len(chunks):
                    detail.set_xlabel(SEAT_LABEL)
                _link_detail(fig, overview, detail, share_range)

            if title:
                fig.suptitle(title, y=1 - 0.04 / height, fontsize=13)
            return save_figure(
                fig, output, dpi=300, bbox_inches="tight", pad_inches=0.09,
                panel_labels=False, preserve_legend_frames=True,
            )
        finally:
            plt.close(fig)
