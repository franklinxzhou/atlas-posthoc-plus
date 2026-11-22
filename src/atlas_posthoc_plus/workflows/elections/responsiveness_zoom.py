#!/usr/bin/env python3

"""
RUN THIS ONLY AFTER GENERATING THE CSV FILES WITH 
the responsiveness workflow

Example:
atlas-posthoc-plus run responsiveness-zoom \
  --race-summary outputs/ohio_responsiveness_all_races/oh_responsiveness_race_summary.csv \
  --ensemble-seats outputs/ohio_responsiveness_all_races/oh_responsiveness_ensemble_seats.csv \
  --official-seats outputs/ohio_responsiveness_all_races/oh_responsiveness_official_plan_seats.csv \
  --public-seats outputs/ohio_responsiveness_all_races/oh_responsiveness_public_plan_seats.csv \
  --overlay-public-plans \
  --output outputs/ohio_responsiveness_all_races/ohio_responsiveness_zoomed.png \
  --public-output outputs/ohio_responsiveness_all_races/ohio_responsiveness_zoomed-public-plans.png \
  --title "Ohio congressional responsiveness across statewide elections" \
  --n-districts 15 \
  --zoom-panels 4 \
  --max-races-per-zoom 6

Suggested manuscript caption guidance:
The ensemble and public-plan collections are rendered as separate figures with
the same overview and vote-share zoom ranges. The Place-5 figure shows the
ensemble frequency distribution; the companion public-plan figure summarizes
the 105 usable public plans. Each public marker represents a distinct integer
seat outcome; the zoom panels print the corresponding number of submitted
plans inside each fixed-size tile. Public plans are summarized descriptively
and are not interpreted as draws from a probability distribution. Symbols
identify the outcomes of the three official plans.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from atlas_posthoc_plus.paper import apply_paper_style, save_figure

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from matplotlib.transforms import ScaledTranslation

PLAN_STYLES = {
    # Purple/blue/orange remains distinguishable under common CVD modes;
    # marker shape provides a second channel when official outcomes coincide.
    "Enacted ('25)": {"color": "#7A5195", "marker": "o"},
    "Overturned ('21)": {"color": "#0072B2", "marker": "s"},
    "Remedial ('22)": {"color": "#E69F00", "marker": "D"},
}

HIST_FACE = "#a9c7d9"
HIST_EDGE = "#465f70"
BASELINE_COLOR = "#666666"
PROPORTIONALITY_STYLE = dict(color="black", linestyle=(0, (7, 5)), linewidth=2.2, alpha=0.85)
PUBLIC_COLOR = "#607d8b"
# Fixed tile area for every distinct public outcome; multiplicity never changes
# the glyph size.  This leaves room for three-digit zoom labels such as 105.
PUBLIC_MARKER_SIZE = 220.0


def _rc_font_size(name: str, fallback: float) -> float:
    value = plt.rcParams.get(name, fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def apply_figure_style() -> None:
    """Apply the role-mapped publication style used by this workflow."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlelocation": "left",
            "axes.titleweight": "normal",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": False,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    apply_paper_style()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot Ohio responsiveness ridgeline histograms from CSV outputs."
    )
    p.add_argument("--race-summary", required=True, help="Path to oh_responsiveness_race_summary.csv")
    p.add_argument("--ensemble-seats", required=True, help="Path to oh_responsiveness_ensemble_seats.csv")
    p.add_argument("--official-seats", required=True, help="Path to oh_responsiveness_official_plan_seats.csv")
    p.add_argument("--public-seats", default=None, help="Optional public-plan seat-count CSV")
    p.add_argument("--overlay-public-plans", action="store_true")
    p.add_argument("--output", required=True, help="Output PNG/PDF path")
    p.add_argument(
        "--public-output",
        default=None,
        help="Optional public-plan figure output; defaults to <output stem>-public-plans<suffix>",
    )
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


def public_multiplicities(
    public: pd.DataFrame | None,
    races: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse public plans to one row per race and integer seat outcome.

    The public submissions are a descriptive collection, so each row in the
    result is a count, not a probability or a smoothed density.  The statewide
    vote share comes from ``races`` so that both figure artifacts use exactly
    the same y coordinate for a given election.
    """
    columns = ["race_id", "dem_seats", "statewide_dem_share", "n_plans"]
    if public is None or public.empty:
        return pd.DataFrame(columns=columns)
    missing = {"race_id", "dem_seats"}.difference(public.columns)
    if missing:
        raise ValueError(
            "Public-plan seat data need columns race_id and dem_seats; "
            f"missing {sorted(missing)}"
        )

    clean = public[["race_id", "dem_seats"]].copy()
    clean["dem_seats"] = pd.to_numeric(clean["dem_seats"], errors="coerce")
    clean = clean.dropna(subset=["race_id", "dem_seats"])
    if clean.empty:
        return pd.DataFrame(columns=columns)
    if not np.isclose(clean["dem_seats"], np.round(clean["dem_seats"])).all():
        raise ValueError("Public-plan Democratic seat counts must be integers")
    clean["dem_seats"] = np.round(clean["dem_seats"]).astype(int)
    grouped = (
        clean.groupby(["race_id", "dem_seats"], as_index=False)
        .size()
        .rename(columns={"size": "n_plans"})
    )
    race_coords = races[["race_id", "statewide_dem_share"]].drop_duplicates("race_id")
    grouped = grouped.merge(race_coords, on="race_id", how="inner")
    return grouped[columns].sort_values(["statewide_dem_share", "race_id", "dem_seats"])


def _ridge_height(races: pd.DataFrame) -> float:
    yvals = races["statewide_dem_share"].dropna().to_numpy(float)
    y_sorted = np.sort(np.unique(yvals))
    if len(y_sorted) > 1:
        gaps = np.diff(y_sorted)
        positive_gaps = gaps[gaps > 1e-9]
        if len(positive_gaps):
            return min(0.0075, max(0.0025, 0.45 * np.percentile(positive_gaps, 25)))
    return 0.0035


def _axis_limits(races: pd.DataFrame, n_districts: int) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return shared x/y limits for the ensemble and public-plan panels."""
    if races.empty:
        return (-0.4, n_districts + 2.3), (0.0, 1.0)
    ridge_height = _ridge_height(races)
    ymin = max(0.0, float(races["statewide_dem_share"].min()) - ridge_height * 1.8)
    ymax = min(1.0, float(races["statewide_dem_share"].max()) + ridge_height * 3.2)
    return (-0.4, n_districts + 2.3), (ymin, ymax)


def draw_ridgeline_panel(
    ax,
    races_sub: pd.DataFrame,
    ensemble: pd.DataFrame,
    official: pd.DataFrame,
    n_districts: int,
    *,
    public: pd.DataFrame | None = None,
    show_public: bool = False,
    show_labels: bool = True,
    show_proportionality: bool = False,
    dot_size: float = 72,
    label_fontsize: float | None = None,
    axis_limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
):
    """Draw the packed Place-5 ensemble responsiveness display.

    ``public`` and ``show_public`` remain accepted for compatibility with
    earlier callers, but public plans are intentionally never drawn here.
    Public submissions belong in :func:`draw_public_multiplicity_panel`, where
    repeated integer seat outcomes can be represented by one marker per count.
    """
    if races_sub.empty:
        return

    del public, show_public
    if label_fontsize is None:
        label_fontsize = _rc_font_size("font.size", 9.0)

    ridge_height = _ridge_height(races_sub)

    edges = np.arange(-0.5, n_districts + 1.5, 1.0)

    for _, race in races_sub.iterrows():
        race_id = race["race_id"]
        y = float(race["statewide_dem_share"])
        sub = ensemble[ensemble["race_id"] == race_id]

        # baseline
        ax.hlines(y, xmin=-0.25, xmax=n_districts + 0.35, color=BASELINE_COLOR, linewidth=0.8, alpha=0.75, zorder=1)

        if not sub.empty:
            counts = (
                sub["dem_seats"]
                .value_counts()
                .reindex(range(0, n_districts + 1), fill_value=0)
                .sort_index()
            )
            fractions = counts / max(counts.sum(), 1)
            max_fraction = float(fractions.max()) if len(fractions) else 0.0
            if max_fraction > 0:
                heights = fractions.to_numpy(float) / max_fraction * ridge_height

                # Connected histogram shape.  Heights are normalized within
                # each election, retaining the existing packed display.
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
            x = float(row["dem_seats"].iloc[0])
            ax.scatter(
                x,
                y,
                s=dot_size,
                marker=style["marker"],
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

    if axis_limits is None:
        axis_limits = _axis_limits(races_sub, n_districts)
    xlim, ylim = axis_limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(range(0, n_districts + 1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(axis="both", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_proportionality:
        yline = np.linspace(ylim[0], ylim[1], 200)
        ax.plot(n_districts * yline, yline, zorder=0, **PROPORTIONALITY_STYLE)
        # Rotate in data coordinates so the label follows the line even when
        # the overview's aspect ratio changes with the number of zoom panels.
        label_y = (ylim[0] + ylim[1]) / 2
        ax.text(
            n_districts * label_y,
            label_y,
            "PROPORTIONAL",
            transform=ax.transData + ScaledTranslation(-7 / 72, 0, ax.figure.dpi_scale_trans),
            rotation=np.degrees(np.arctan2(1, n_districts)),
            transform_rotates_text=True,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=_rc_font_size("legend.fontsize", 8.0),
            fontweight="bold",
            color="black",
            zorder=6,
        )


def draw_public_multiplicity_panel(
    ax,
    races_sub: pd.DataFrame,
    official: pd.DataFrame,
    public: pd.DataFrame,
    n_districts: int,
    *,
    axis_limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
    show_labels: bool = True,
    show_count_labels: bool = True,
    label_fontsize: float | None = None,
) -> pd.DataFrame:
    """Draw one muted marker per public race/seat outcome.

    Every public outcome uses one constant-size muted tile.  In zoom panels,
    the exact multiplicity is printed inside the tile; the compact overview
    suppresses those labels.  Official plans are drawn at the same integer
    seat coordinate as public outcomes, using distinct outlined markers so no
    seat-coordinate jitter is introduced.
    """
    if races_sub.empty:
        return pd.DataFrame(columns=["race_id", "dem_seats", "statewide_dem_share", "n_plans"])
    if label_fontsize is None:
        label_fontsize = _rc_font_size("legend.fontsize", 8.0)
    if axis_limits is None:
        axis_limits = _axis_limits(races_sub, n_districts)
    xlim, ylim = axis_limits

    grouped = public_multiplicities(public, races_sub)
    for _, race in races_sub.iterrows():
        race_id = race["race_id"]
        y = float(race["statewide_dem_share"])
        ax.hlines(
            y,
            xmin=-0.25,
            xmax=n_districts + 0.35,
            color=BASELINE_COLOR,
            linewidth=0.8,
            alpha=0.75,
            zorder=1,
        )

        pub = grouped[grouped["race_id"] == race_id]
        if not pub.empty:
            xs = pub["dem_seats"].to_numpy(float)
            counts = pub["n_plans"].to_numpy(float)
            ax.scatter(
                xs,
                np.full(len(pub), y),
                s=np.full(len(pub), PUBLIC_MARKER_SIZE),
                marker="s",
                color=PUBLIC_COLOR,
                alpha=0.84,
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )
            if show_count_labels:
                for x, count in zip(xs, counts):
                    ax.text(
                        x,
                        y,
                        f"{int(count)}",
                        fontsize=label_fontsize,
                        color="white",
                        fontweight="bold",
                        ha="center",
                        va="center",
                        zorder=6,
                    )

        off = official[official["race_id"] == race_id]
        for plan_label, style in PLAN_STYLES.items():
            row = off[off["source"] == plan_label]
            if row.empty:
                continue
            ax.scatter(
                float(row["dem_seats"].iloc[0]),
                y,
                s=78,
                marker=style["marker"],
                facecolors="none",
                edgecolors=style["color"],
                linewidths=1.5,
                zorder=5,
            )

        if show_labels:
            ax.text(
                n_districts + 0.42,
                y + 0.0003,
                race["abbr"],
                fontsize=_rc_font_size("font.size", 9.0),
                ha="left",
                va="center",
            )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(range(0, n_districts + 1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(axis="both", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return grouped


def _official_legend_handles(*, outlined: bool) -> list[Line2D]:
    handles: list[Line2D] = []
    for label, style in PLAN_STYLES.items():
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                linestyle="None",
                markerfacecolor="none" if outlined else style["color"],
                markeredgecolor=style["color"] if outlined else "black",
                markeredgewidth=1.35 if outlined else 0.65,
                markersize=8.5,
                label=label,
            )
        )
    return handles


def _derived_public_output(output: Path) -> Path:
    """Return the companion public-plan filename for an ensemble output."""
    output = Path(output)
    return output.with_name(f"{output.stem}-public-plans{output.suffix}")


def _ensemble_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=HIST_EDGE,
            marker="s",
            linestyle="None",
            markerfacecolor=HIST_FACE,
            markeredgecolor=HIST_EDGE,
            markersize=10,
            label="Compliant ensemble histogram",
        ),
        *_official_legend_handles(outlined=False),
    ]


def _public_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor=PUBLIC_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.8,
            alpha=0.84,
            markersize=np.sqrt(PUBLIC_MARKER_SIZE),
            label="Submitted-plan outcome",
        ),
        *_official_legend_handles(outlined=True),
    ]


def _plot_ensemble_figure(
    races: pd.DataFrame,
    ensemble: pd.DataFrame,
    official: pd.DataFrame,
    chunks: list[pd.DataFrame],
    output: Path,
    title: str,
    n_districts: int,
) -> None:
    """Render the original overview-plus-zoom ensemble responsiveness figure."""
    if races.empty:
        raise ValueError("Cannot plot responsiveness panels without races")
    n_zoom = len(chunks)
    fig = plt.figure(figsize=(15.5, max(10.5, 2.6 * n_zoom)))
    gs = GridSpec(
        n_zoom,
        2,
        width_ratios=[1.10, 1.90],
        height_ratios=[1] * n_zoom,
        wspace=0.28,
        hspace=0.32,
        figure=fig,
    )

    ax_over = fig.add_subplot(gs[:, 0])
    draw_ridgeline_panel(
        ax_over,
        races,
        ensemble,
        official,
        n_districts,
        show_labels=False,
        show_proportionality=True,
        dot_size=76,
        axis_limits=_axis_limits(races, n_districts),
    )
    ax_over.set_title("Overview")
    ax_over.set_xlabel("Democrats elected")
    ax_over.set_ylabel("Statewide Democratic vote share")
    ax_over.legend(handles=_ensemble_legend_handles(), loc="upper left")

    # Draw zoom panels covering the same chunks used by the public companion.
    for i, chunk in enumerate(chunks):
        ax = fig.add_subplot(gs[i, 1])
        draw_ridgeline_panel(
            ax,
            chunk,
            ensemble,
            official,
            n_districts,
            show_labels=True,
            show_proportionality=False,
            dot_size=74,
            axis_limits=_axis_limits(chunk, n_districts),
        )
        lo = chunk["statewide_dem_share"].min()
        hi = chunk["statewide_dem_share"].max()
        ax.set_title(f"Zoom {i+1}: vote-share range {lo:.1%} to {hi:.1%}")
        if i == n_zoom - 1:
            ax.set_xlabel("Democrats elected")
        else:
            ax.set_xlabel("")
        ax.set_ylabel("")
        add_key(ax) if i == n_zoom - 1 else None

    fig.suptitle(title, y=0.995)
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.065,
        top=0.94,
        wspace=0.28,
        hspace=0.32,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_public_figure(
    races: pd.DataFrame,
    official: pd.DataFrame,
    public: pd.DataFrame,
    chunks: list[pd.DataFrame],
    output: Path,
    title: str,
    n_districts: int,
) -> None:
    """Render the companion public-plan overview-plus-zoom figure.

    The overview intentionally omits direct ``n`` labels so its compact left
    column remains readable.  Zoom panels carry the exact labels and are the
    authoritative view for individual multiplicity values.
    """
    if races.empty:
        raise ValueError("Cannot plot public responsiveness without races")
    n_zoom = len(chunks)
    fig = plt.figure(figsize=(15.5, max(10.5, 2.6 * n_zoom)))
    gs = GridSpec(
        n_zoom,
        2,
        width_ratios=[1.10, 1.90],
        height_ratios=[1] * n_zoom,
        wspace=0.28,
        hspace=0.32,
        figure=fig,
    )

    ax_over = fig.add_subplot(gs[:, 0])
    draw_public_multiplicity_panel(
        ax_over,
        races,
        official,
        public,
        n_districts,
        axis_limits=_axis_limits(races, n_districts),
        show_labels=False,
        show_count_labels=False,
    )
    ax_over.set_title("Overview")
    ax_over.set_xlabel("Democrats elected")
    ax_over.set_ylabel("Statewide Democratic vote share")
    ax_over.legend(
        handles=_public_legend_handles(),
        title="Zoom labels: Number of submitted plans",
        loc="upper left",
    )

    for i, chunk in enumerate(chunks):
        ax = fig.add_subplot(gs[i, 1])
        draw_public_multiplicity_panel(
            ax,
            chunk,
            official,
            public,
            n_districts,
            axis_limits=_axis_limits(chunk, n_districts),
            show_labels=True,
            show_count_labels=True,
        )
        lo = chunk["statewide_dem_share"].min()
        hi = chunk["statewide_dem_share"].max()
        ax.set_title(f"Zoom {i+1}: vote-share range {lo:.1%} to {hi:.1%}")
        if i == n_zoom - 1:
            ax.set_xlabel("Democrats elected")
        else:
            ax.set_xlabel("")
        ax.set_ylabel("")
        add_key(ax) if i == n_zoom - 1 else None

    fig.suptitle(f"{title} — public submitted plans", y=0.995)
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.065,
        top=0.94,
        wspace=0.28,
        hspace=0.32,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output, dpi=300, bbox_inches="tight")
    plt.close(fig)


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


def plot_from_csv(
    races: pd.DataFrame,
    ensemble: pd.DataFrame,
    official: pd.DataFrame,
    output: Path,
    title: str,
    n_districts: int,
    zoom_panels: int,
    max_races_per_zoom: int,
    *,
    public: pd.DataFrame | None = None,
    overlay_public_plans: bool = False,
    public_output: Path | None = None,
):
    apply_figure_style()
    races = races.copy().sort_values(["statewide_dem_share", "year", "race_id"]).reset_index(drop=True)
    if overlay_public_plans and public is None:
        raise ValueError("overlay_public_plans=True requires public seat data")
    chunks = build_chunks(races, zoom_panels=zoom_panels, max_races_per_zoom=max_races_per_zoom)
    output = Path(output)
    _plot_ensemble_figure(races, ensemble, official, chunks, output, title, n_districts)

    if overlay_public_plans:
        destination = Path(public_output) if public_output is not None else _derived_public_output(output)
        _plot_public_figure(
            races,
            official,
            public,
            chunks,
            destination,
            title,
            n_districts,
        )


def main():
    args = parse_args()
    races = pd.read_csv(args.race_summary)
    ensemble = pd.read_csv(args.ensemble_seats)
    official = pd.read_csv(args.official_seats)
    public = None
    if args.overlay_public_plans:
        if args.public_seats is None:
            raise ValueError("--overlay-public-plans requires --public-seats")
        public_path = Path(args.public_seats)
        if not public_path.exists():
            raise FileNotFoundError(f"Public-plan seats CSV not found: {public_path}")
        public = pd.read_csv(public_path)
    output = Path(args.output)
    public_output = Path(args.public_output) if args.public_output else None
    plot_from_csv(
        races, ensemble, official, output,
        title=args.title,
        n_districts=args.n_districts,
        zoom_panels=args.zoom_panels,
        max_races_per_zoom=args.max_races_per_zoom,
        public=public,
        overlay_public_plans=args.overlay_public_plans,
        public_output=public_output,
    )
    print(f"Wrote {output}")
    if args.overlay_public_plans:
        destination = public_output if public_output is not None else _derived_public_output(output)
        print(f"Wrote public-plan figure {destination}")


if __name__ == "__main__":
    main()
