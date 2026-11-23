from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PAPER_MODE_ENV = "ATLAS_POSTHOC_PLUS_PAPER_MODE"
PANEL_LABEL_GID = "atlas-posthoc-plus-panel-label"
PANEL_TITLE_GID = "atlas-posthoc-plus-panel-title"
DATA_LABEL_GID = "atlas-posthoc-plus-data-label"

PAPER_FONT_FAMILY = "sans-serif"
PAPER_SANS_SERIF = ("Nimbus Sans", "DejaVu Sans")
PAPER_LABEL_SIZE = 10
PAPER_LEGEND_SIZE = 9
PAPER_TICK_SIZE = 8
PAPER_ANNOTATION_SIZE = 8
PAPER_DATA_LABEL_SIZE = 6
PAPER_PANEL_SIZE = 11
PAPER_PANEL_WEIGHT = "bold"
OFFICIAL_PLANS_TITLE = "Official plans"
PAPER_RC = {
    "font.family": PAPER_FONT_FAMILY,
    "font.sans-serif": PAPER_SANS_SERIF,
    "font.size": PAPER_LABEL_SIZE,
    "mathtext.fontset": "stixsans",
    "axes.titlesize": PAPER_LABEL_SIZE,
    "axes.labelsize": PAPER_LABEL_SIZE,
    "legend.fontsize": PAPER_LEGEND_SIZE,
    "legend.title_fontsize": PAPER_LEGEND_SIZE,
    "xtick.labelsize": PAPER_TICK_SIZE,
    "ytick.labelsize": PAPER_TICK_SIZE,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

# Color and a second visual channel identify Ohio reference plans across plots.
OFFICIAL_PLAN_STYLES = {
    "Enacted ('25)": {"color": "#7A5195", "marker": "o", "linestyle": "-"},
    "Overturned ('21)": {"color": "#0072B2", "marker": "s", "linestyle": "--"},
    "Remedial ('22)": {"color": "#E69F00", "marker": "D", "linestyle": "-."},
}
_OFFICIAL_PLAN_ALIASES = {
    alias.casefold(): f"{role} ('{short_year})"
    for role, year in (("Enacted", 2025), ("Overturned", 2021), ("Remedial", 2022))
    for short_year in [str(year)[-2:]]
    for alias in (
        role,
        f"{role} ('{short_year})",
        f"{role} ('{short_year}')",
        f"{role} ({year})",
        f"{year} {role}",
    )
}


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
    mpl.rcParams.update(PAPER_RC)


def _official_plan_name(label: str) -> str | None:
    return _OFFICIAL_PLAN_ALIASES.get(label.strip().replace("’", "'").casefold())


def official_plan_label(label: str) -> str:
    """Use the same display name for each known plan in paper figures."""
    if paper_mode_enabled():
        return _official_plan_name(label) or label
    return label


def official_plan_style(label: str, **style: Any) -> dict[str, Any]:
    """Apply the common palette to the style properties used by a plot."""
    if paper_mode_enabled():
        canonical = OFFICIAL_PLAN_STYLES.get(_official_plan_name(label) or "", {})
        style.update({key: canonical[key] for key in style.keys() & canonical.keys()})
    return style


def official_plan_styles(
    styles: dict[str, dict[str, Any]],
) -> Iterable[tuple[str, dict[str, Any]]]:
    return ((label, official_plan_style(label, **style)) for label, style in styles.items())


def figure_legend(
    owner: Any,
    handles: Iterable[Any] | None = None,
    labels: Iterable[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Set paper legend dimensions before Matplotlib lays out the figure."""
    paper = paper_mode_enabled()
    if paper:
        kwargs.update(frameon=False, fontsize=PAPER_LEGEND_SIZE, title_fontsize=PAPER_LEGEND_SIZE)
        if handles is None and labels is None:
            handles, labels = owner.get_legend_handles_labels()
        if handles is not None:
            handle_list = list(handles)
            entries = list(
                zip(
                    handle_list,
                    labels if labels is not None else (h.get_label() for h in handle_list),
                )
            )
            plan_order = {name: index for index, name in enumerate(OFFICIAL_PLAN_STYLES)}
            official = {
                index: plan_order[name]
                for index, (_, label) in enumerate(entries)
                if (name := _official_plan_name(label)) is not None
            }
            # Order plan entries consistently while keeping other entries in place.
            ordered = iter(sorted(official, key=official.__getitem__))
            entries = [
                entries[next(ordered)] if index in official else entry
                for index, entry in enumerate(entries)
            ]
            handles = [handle for handle, _ in entries]
            labels = [label for _, label in entries]
    if handles is not None:
        kwargs["handles"] = handles
    if labels is not None:
        kwargs["labels"] = labels
    legend = owner.legend(**kwargs)
    if paper:
        _style_legend(legend)
    return legend


def set_panel_title(axis: Any, title: str) -> None:
    """Mark a subplot heading that carries information needed to identify the panel."""
    if paper_mode_enabled():
        axis.set_title(official_plan_label(title), loc="left").set_gid(PANEL_TITLE_GID)
    else:
        axis.set_title(title)


def panel_tag(index: int) -> str:
    """Return (a), ..., (z), (aa), ... for a zero-based panel index."""
    value = index + 1
    letters = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        letters = chr(ord("a") + remainder) + letters
    return f"({letters})"


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


def _style_legend(legend: Any) -> None:
    legend.set_frame_on(False)
    entries = legend.get_texts()
    official_names = [_official_plan_name(text.get_text()) for text in entries]
    for text, name in zip(entries, official_names):
        text.set_fontsize(PAPER_LEGEND_SIZE)
        if name:
            text.set_text(name)
    title = legend.get_title()
    if title.get_text() == "Reference plans" or (entries and all(official_names)):
        legend.set_title(OFFICIAL_PLANS_TITLE)
    title.set_fontsize(PAPER_LEGEND_SIZE)


def _prepare_paper_figure(figure: Any) -> None:
    from matplotlib.axes import Axes
    from matplotlib.legend import Legend
    from matplotlib.text import Annotation, Text

    for text in figure.findobj(match=Text):
        if text.get_gid() == PANEL_LABEL_GID:
            text.remove()
            continue
        text.set_fontfamily(PAPER_FONT_FAMILY)
        text.set_math_fontfamily("stixsans")
        size = PAPER_DATA_LABEL_SIZE if text.get_gid() == DATA_LABEL_GID else PAPER_ANNOTATION_SIZE
        text.set_fontsize(size)
        if isinstance(text, Annotation) and (plan := _official_plan_name(text.get_text())):
            text.set_text(plan)
            text.set_color("#222222")

    for axis in figure.findobj(match=Axes):
        for title in (axis.title, axis._left_title, axis._right_title):
            if title.get_gid() == PANEL_TITLE_GID:
                title.set_fontsize(PAPER_LABEL_SIZE)
            else:
                title.set_text("")
        for dimension in (axis.xaxis, axis.yaxis):
            dimension.label.set_fontsize(PAPER_LABEL_SIZE)
            dimension.get_offset_text().set_fontsize(PAPER_TICK_SIZE)
        axis.tick_params(which="both", labelsize=PAPER_TICK_SIZE, direction="out")
        for spine in axis.spines.values():
            spine.set_linewidth(PAPER_RC["axes.linewidth"])

    for attribute in ("_supxlabel", "_supylabel"):
        label = getattr(figure, attribute, None)
        if label is not None:
            label.set_fontsize(PAPER_LABEL_SIZE)
    if figure._suptitle is not None:
        figure._suptitle.set_text("")

    for legend in figure.findobj(match=Legend):
        _style_legend(legend)

    panels = _distinct_panel_axes(figure)
    if len(panels) > 1:
        for index, axis in enumerate(panels):
            label = axis.annotate(
                panel_tag(index),
                (0, 1),
                xycoords="axes fraction",
                xytext=(-5, 6),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=PAPER_PANEL_SIZE,
                fontweight=PAPER_PANEL_WEIGHT,
                annotation_clip=False,
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
        import matplotlib as mpl

        with mpl.rc_context(PAPER_RC):
            _prepare_paper_figure(figure)
            kwargs.pop("format", None)
            kwargs.setdefault("bbox_inches", "tight")
            figure.savefig(destination, format="pdf", **kwargs)
    else:
        figure.savefig(destination, **kwargs)

    return destination
