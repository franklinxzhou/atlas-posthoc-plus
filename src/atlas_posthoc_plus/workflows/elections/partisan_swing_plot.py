"""Manuscript plots for reciprocal seat curves and ensemble asymmetry areas."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from atlas_posthoc_plus.paper import (
    OFFICIAL_PLANS_TITLE,
    apply_paper_style,
    figure_legend,
    official_plan_label,
    official_plan_style,
    save_figure,
)
from atlas_posthoc_plus.workflows.elections.partisan_swing import (
    LOWER_VOTE_PCT,
    UPPER_VOTE_PCT,
    Plan,
    SwingResult,
)

INK = "#303030"
DEMOCRATIC = "#236992"
REPUBLICAN = "#AE4039"
GAP_FILL = "#CDCDD2"


def _panel():
    figure, axis = plt.subplots(figsize=(5.35, 3.5), layout="constrained", facecolor="white")
    axis.set_facecolor("white")
    axis.grid(False, which="both")
    axis.spines[:].set_visible(True)
    axis.spines[:].set_color(INK)
    return figure, axis


def _display_name(label: str) -> str:
    return re.sub(r"\('(\d{2})\)$", r"(\1')", official_plan_label(label))


def _boxed_legend(axis, *, title, loc, handles=None, labels=None):
    legend = figure_legend(
        axis, handles, labels, title=title, loc=loc, alignment="left", ncols=1,
        framealpha=1, facecolor="white", edgecolor="#808892", fancybox=False,
        borderaxespad=0.45, borderpad=0.3, labelspacing=0.25, handlelength=1.5,
        handletextpad=0.45,
    )
    legend.set_frame_on(True)
    legend.get_frame().set_linewidth(0.55)
    for label in legend.get_texts():
        label.set_text(_display_name(label.get_text()))


def plot_curves(outdir: Path, officials: list[tuple[Plan, SwingResult]]) -> None:
    apply_paper_style()
    seats = np.concatenate([curve for _, result in officials for curve in
                            (result.dem_seats, result.rep_seats)])
    halfway = officials[0][1].n_districts / 2
    low, high = min(seats.min(), halfway), max(seats.max(), halfway)
    pad = max(0.35, 0.08 * (high - low))
    for number, (plan, result) in enumerate(officials, start=1):
        figure, axis = _panel()
        try:
            edges = result.edges
            axis.axhline(halfway, color="#767676", linestyle="--", linewidth=0.65, zorder=1)
            axis.fill_between(
                edges, np.r_[result.dem_seats, result.dem_seats[-1]],
                np.r_[result.rep_seats, result.rep_seats[-1]], step="post",
                color=GAP_FILL, linewidth=0, label="Deviation from symmetry", zorder=2,
            )
            axis.stairs(result.dem_seats, edges, baseline=None, color=DEMOCRATIC, linewidth=1.55,
                        label="Democratic majority", zorder=3)
            axis.stairs(result.rep_seats, edges, baseline=None, color=REPUBLICAN, linewidth=1.35,
                        linestyle=(0, (4, 1.7)), label="Republican majority", zorder=4)
            axis.set(xlim=(LOWER_VOTE_PCT, UPPER_VOTE_PCT), ylim=(low - pad, high + pad),
                     xlabel="Statewide two-party vote for majority party (%)",
                     ylabel="Seats won by majority party")
            axis.set_xticks(np.arange(LOWER_VOTE_PCT, UPPER_VOTE_PCT + 1))
            axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
            handles, labels = axis.get_legend_handles_labels()
            # Put the party curves before the area swatch in the compact legend.
            _boxed_legend(axis, title=_display_name(plan.label), loc="best",
                          handles=handles[1:] + handles[:1], labels=labels[1:] + labels[:1])
            slug = re.sub(r"[^a-z0-9]+", "-", plan.label.lower()).strip("-")
            save_figure(figure, outdir / f"{number:02d}-{slug or 'plan'}.pdf",
                        facecolor="white", preserve_legend_frames=True)
        finally:
            plt.close(figure)


def plot_histogram(
    outdir: Path, areas: np.ndarray, officials: list[tuple[Plan, SwingResult]],
) -> None:
    apply_paper_style()
    low, high = float(areas.min()), float(areas.max())
    if high - low < 1e-9:
        low, high = max(0, low - 0.25), high + 0.25
    count, edges = np.histogram(areas, bins=min(40, max(8, int(np.sqrt(len(areas))))),
                                range=(low, high))
    with (outdir / "ensemble-histogram-bins.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["area_bin_start_seat_pp", "area_bin_end_seat_pp", "plans"])
        writer.writerows(zip(edges[:-1], edges[1:], count, strict=True))

    figure, axis = _panel()
    try:
        axis.bar(edges[:-1], count, width=np.diff(edges), align="edge", color="#CCD3DF",
                 edgecolor="#536174", linewidth=0.6, zorder=1)
        top = float(count.max())
        left = min(edges[0], *(result.area for _, result in officials))
        right = max(edges[-1], *(result.area for _, result in officials))
        for index, ((plan, result), color) in enumerate(zip(
            officials, ("#7A5195", "#0072B2", "#D99412"), strict=True,
        )):
            style = official_plan_style(plan.label, color=color)
            dot_y = top * (0.76 + index * 0.045)
            axis.vlines(result.area, 0, dot_y, **style, linewidth=0.85, zorder=2)
            axis.plot(result.area, dot_y, label=plan.label, **style, linestyle="none", marker="o",
                      markersize=(6.5, 4, 5.5)[index], markeredgecolor=INK, markeredgewidth=0.4,
                      zorder=3)
        pad = (right - left) * 0.045
        axis.set(xlim=(max(0, left - pad), right + pad), ylim=(0, top * 1.18),
                 xlabel="Area between curves (seats × percentage points)", ylabel="Ensemble plans")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
        axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        _boxed_legend(axis, title=OFFICIAL_PLANS_TITLE, loc="best")
        save_figure(figure, outdir / "ensemble-histogram.pdf", facecolor="white",
                    preserve_legend_frames=True)
    finally:
        plt.close(figure)
