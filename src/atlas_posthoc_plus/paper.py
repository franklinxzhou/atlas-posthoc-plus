from __future__ import annotations

import os
from pathlib import Path
from typing import Any

PAPER_MODE_ENV = "ATLAS_POSTHOC_PLUS_PAPER_MODE"
PANEL_LABEL_GID = "atlas-posthoc-plus-panel-label"


def paper_mode_enabled() -> bool:
    """Return whether the current workflow process is rendering manuscript figures."""
    return os.environ.get(PAPER_MODE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def apply_paper_style() -> None:
    """Apply the typography and PDF settings used by manuscript figures."""
    if not paper_mode_enabled():
        return

    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
            "mathtext.fontset": "stixsans",
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def paper_output_path(output: str | Path) -> Path:
    """Return the PDF destination used for a manuscript figure."""
    destination = Path(output)
    if paper_mode_enabled() and destination.suffix.lower() != ".pdf":
        return destination.with_suffix(".pdf")
    return destination


def _distinct_panel_axes(figure: Any) -> list[Any]:
    """Return visible, non-colorbar axes, deduplicating twinned axes."""
    panels: list[Any] = []
    occupied: list[tuple[float, float, float, float]] = []

    for axis in figure.axes:
        if not axis.get_visible() or axis.get_label() == "<colorbar>":
            continue
        if getattr(axis, "get_subplotspec", lambda: None)() is None:
            continue
        bounds = tuple(round(value, 6) for value in axis.get_position().bounds)
        if bounds in occupied:
            continue
        occupied.append(bounds)
        panels.append(axis)

    return panels


def _prepare_paper_figure(figure: Any) -> None:
    for axis in figure.axes:
        axis.set_title("")

    suptitle = getattr(figure, "_suptitle", None)
    if suptitle is not None:
        suptitle.set_text("")

    for artist in figure.findobj():
        if getattr(artist, "get_gid", lambda: None)() == PANEL_LABEL_GID:
            artist.remove()

    panels = _distinct_panel_axes(figure)
    if len(panels) not in {2, 4}:
        return
    if len(panels) == 2:
        first = panels[0].get_position()
        second = panels[1].get_position()
        overlap = max(
            0.0,
            min(first.y1, second.y1) - max(first.y0, second.y0),
        )
        if overlap < 0.75 * min(first.height, second.height):
            return

    for index, axis in enumerate(panels):
        label = axis.text(
            -0.10,
            1.02,
            f"({chr(ord('a') + index)})",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )
        label.set_gid(PANEL_LABEL_GID)


def save_figure(
    figure: Any,
    output: str | Path,
    **kwargs: Any,
) -> Path:
    """Save a figure, applying manuscript conventions when paper mode is active."""
    destination = paper_output_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if paper_mode_enabled():
        _prepare_paper_figure(figure)
        kwargs.pop("format", None)
        kwargs.setdefault("bbox_inches", "tight")
        figure.savefig(destination, format="pdf", **kwargs)
    else:
        figure.savefig(destination, **kwargs)

    return destination
