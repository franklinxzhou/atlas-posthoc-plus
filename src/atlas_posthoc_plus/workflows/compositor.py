from __future__ import annotations

import argparse
import io
import math
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PageObject, PdfReader, PdfWriter, Transformation

from atlas_posthoc_plus.config import ConfigError, ProjectConfig, load_config
from atlas_posthoc_plus.paper import (
    PAPER_FONT_FAMILY,
    PAPER_PANEL_SIZE,
    PAPER_PANEL_WEIGHT,
    PAPER_SANS_SERIF,
    panel_tag,
)

SECTION = "workflow.figure-compositor"
SPACERS = {"#", "."}
POINTS_PER_INCH = 72.0


@dataclass(frozen=True)
class Panel:
    """One existing workflow figure placed on the composite canvas."""

    id: str
    workflow: str
    config_path: Path
    figure_path: Path
    page: int = 1
    tag: str | bool | None = None
    horizontal_align: str = "center"
    vertical_align: str = "center"


@dataclass(frozen=True)
class Margins:
    left: float
    right: float
    bottom: float
    top: float


@dataclass(frozen=True)
class Composition:
    panels: tuple[Panel, ...]
    design: tuple[tuple[str | None, ...], ...]
    widths: tuple[float, ...]
    heights: tuple[float, ...]
    width: float | None
    height: float | None
    margins: Margins
    column_gap: float
    row_gap: float
    tags: bool
    tag_size: float
    tag_weight: str
    tag_offset_x: float
    tag_offset_y: float
    font_family: str


@dataclass(frozen=True)
class PanelBox:
    panel: Panel
    left: float
    bottom: float
    width: float
    height: float


def _table(value: Any, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"configuration value {key!r} must be a TOML table")
    return value


def _number(
    value: Any,
    key: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"configuration value {key!r} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"configuration value {key!r} must be finite")
    if positive and result <= 0:
        raise ConfigError(f"configuration value {key!r} must be greater than zero")
    if nonnegative and result < 0:
        raise ConfigError(f"configuration value {key!r} cannot be negative")
    return result


def _integer(value: Any, key: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(
            f"configuration value {key!r} must be an integer >= {minimum}"
        )
    return value


def _alignment(value: Any, key: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        options = ", ".join(sorted(choices))
        raise ConfigError(f"configuration value {key!r} must be one of: {options}")
    return value


def _default_panel_id(index: int) -> str:
    """Return A..Z, AA..AZ, BA... identifiers."""
    value = index + 1
    result = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def _parse_panels(config: ProjectConfig, section: Mapping[str, Any]) -> tuple[Panel, ...]:
    raw_panels = section.get("panels")
    if not isinstance(raw_panels, list) or not raw_panels:
        raise ConfigError(
            f"configuration value {SECTION + '.panels'!r} must be a non-empty "
            "array of tables"
        )

    # Imported lazily to avoid a registry/config import cycle during CLI startup.
    from atlas_posthoc_plus.registry import WORKFLOW_ALIASES, WORKFLOWS

    panels: list[Panel] = []
    seen_ids: set[str] = set()
    for index, raw_panel in enumerate(raw_panels):
        key = f"{SECTION}.panels[{index}]"
        panel_data = _table(raw_panel, key)

        panel_id = panel_data.get("id", _default_panel_id(index))
        if not isinstance(panel_id, str) or not panel_id.strip():
            raise ConfigError(f"configuration value {key + '.id'!r} must be a string")
        panel_id = panel_id.strip()
        if panel_id in SPACERS:
            raise ConfigError(
                f"panel id {panel_id!r} is reserved for empty/spacer cells"
            )
        if panel_id in seen_ids:
            raise ConfigError(f"duplicate compositor panel id: {panel_id!r}")
        seen_ids.add(panel_id)

        workflow = panel_data.get("workflow")
        if not isinstance(workflow, str) or not workflow:
            raise ConfigError(
                f"configuration value {key + '.workflow'!r} is required"
            )
        if WORKFLOW_ALIASES.get(workflow, workflow) not in WORKFLOWS or workflow == "figure-compositor":
            choices = ", ".join(
                sorted(name for name in WORKFLOWS if name != "figure-compositor")
            )
            raise ConfigError(
                f"unknown source workflow {workflow!r}; use a registered workflow: "
                f"{choices}"
            )

        source_config_value = panel_data.get("config")
        if source_config_value is None or source_config_value == "":
            source_config_path = config.path
        elif isinstance(source_config_value, str):
            source_config_path = config.resolve_path(source_config_value)
        else:
            raise ConfigError(
                f"configuration value {key + '.config'!r} must be a path string"
            )
        source_config = load_config(source_config_path)

        figure_value = panel_data.get("figure")
        if not isinstance(figure_value, str) or not figure_value:
            raise ConfigError(
                f"configuration value {key + '.figure'!r} is required"
            )
        figure_path = source_config.resolve_path(figure_value)
        if figure_path.suffix.lower() != ".pdf":
            raise ConfigError(
                f"panel {panel_id!r} selects {figure_path.name!r}; compositor "
                "inputs must be PDF files so vector content is preserved"
            )

        page = _integer(panel_data.get("page", 1), key + ".page")
        tag = panel_data.get("tag")
        if tag is not None and not isinstance(tag, (str, bool)):
            raise ConfigError(
                f"configuration value {key + '.tag'!r} must be a string or boolean"
            )

        panels.append(
            Panel(
                id=panel_id,
                workflow=workflow,
                config_path=source_config.path,
                figure_path=figure_path,
                page=page,
                tag=tag,
                horizontal_align=_alignment(
                    panel_data.get("horizontal_align", "center"),
                    key + ".horizontal_align",
                    {"left", "center", "right"},
                ),
                vertical_align=_alignment(
                    panel_data.get("vertical_align", "center"),
                    key + ".vertical_align",
                    {"bottom", "center", "top"},
                ),
            )
        )
    return tuple(panels)


def _design_rows(value: Any) -> list[list[str]]:
    if isinstance(value, str):
        lines = [line.strip() for line in value.splitlines() if line.strip()]
    elif isinstance(value, list) and all(isinstance(line, str) for line in value):
        lines = [line.strip() for line in value if line.strip()]
    else:
        raise ConfigError(
            f"configuration value {SECTION + '.design'!r} must be a string "
            "or an array of strings"
        )
    if not lines:
        raise ConfigError("compositor design cannot be empty")

    # Compact character designs ("AAB") mirror patchwork. Whitespace-delimited
    # designs ("hero hero side") allow descriptive, multi-character panel IDs.
    rows = [
        line.split() if any(character.isspace() for character in line) else list(line)
        for line in lines
    ]
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ConfigError("every compositor design row must have the same width")
    return rows


def _validate_rectangular_areas(rows: Sequence[Sequence[str]]) -> None:
    identifiers = {
        identifier
        for row in rows
        for identifier in row
        if identifier not in SPACERS
    }
    for identifier in identifiers:
        cells = {
            (row_index, column_index)
            for row_index, row in enumerate(rows)
            for column_index, value in enumerate(row)
            if value == identifier
        }
        row_indices = [row for row, _ in cells]
        column_indices = [column for _, column in cells]
        rectangle = {
            (row, column)
            for row in range(min(row_indices), max(row_indices) + 1)
            for column in range(min(column_indices), max(column_indices) + 1)
        }
        if cells != rectangle:
            raise ConfigError(
                f"panel {identifier!r} must occupy one rectangular area in the design"
            )


def _parse_design(
    section: Mapping[str, Any], panels: Sequence[Panel]
) -> tuple[tuple[str | None, ...], ...]:
    raw_design = section.get("design")
    panel_ids = {panel.id for panel in panels}
    ncol_value = section.get("ncol")
    nrow_value = section.get("nrow")

    if raw_design is not None:
        if ncol_value is not None or nrow_value is not None:
            raise ConfigError("set either design or ncol/nrow, not both")
        rows = _design_rows(raw_design)
        _validate_rectangular_areas(rows)
        design_ids = {
            identifier
            for row in rows
            for identifier in row
            if identifier not in SPACERS
        }
        missing = panel_ids - design_ids
        unknown = design_ids - panel_ids
        if missing:
            raise ConfigError(
                "compositor design does not place panel(s): " + ", ".join(sorted(missing))
            )
        if unknown:
            raise ConfigError(
                "compositor design references unknown panel(s): "
                + ", ".join(sorted(unknown))
            )
        return tuple(
            tuple(None if identifier in SPACERS else identifier for identifier in row)
            for row in rows
        )

    panel_count = len(panels)
    ncol = (
        None
        if ncol_value is None
        else _integer(ncol_value, SECTION + ".ncol")
    )
    nrow = (
        None
        if nrow_value is None
        else _integer(nrow_value, SECTION + ".nrow")
    )
    if ncol is None and nrow is None:
        ncol = math.ceil(math.sqrt(panel_count))
        nrow = math.ceil(panel_count / ncol)
    elif ncol is None:
        assert nrow is not None
        ncol = math.ceil(panel_count / nrow)
    elif nrow is None:
        nrow = math.ceil(panel_count / ncol)
    elif ncol * nrow < panel_count:
        raise ConfigError(
            f"ncol * nrow provides {ncol * nrow} cells for {panel_count} panels"
        )

    identifiers = [panel.id for panel in panels]
    identifiers.extend([None] * (nrow * ncol - panel_count))
    return tuple(
        tuple(identifiers[row * ncol : (row + 1) * ncol])
        for row in range(nrow)
    )


def _ratios(value: Any, key: str, count: int) -> tuple[float, ...]:
    if value is None:
        return (1.0,) * count
    if not isinstance(value, list) or len(value) != count:
        raise ConfigError(
            f"configuration value {key!r} must contain exactly {count} numbers"
        )
    return tuple(
        _number(item, f"{key}[{index}]", positive=True)
        for index, item in enumerate(value)
    )


def parse_composition(config: ProjectConfig) -> Composition:
    """Parse and validate ``[workflow.figure-compositor]``."""
    section = _table(config.get(SECTION), SECTION)
    panels = _parse_panels(config, section)
    design = _parse_design(section, panels)
    nrows = len(design)
    ncols = len(design[0])

    margin = _number(
        section.get("margin", 0.25), SECTION + ".margin", nonnegative=True
    )
    margins = Margins(
        left=_number(
            section.get("margin_left", margin),
            SECTION + ".margin_left",
            nonnegative=True,
        ),
        right=_number(
            section.get("margin_right", margin),
            SECTION + ".margin_right",
            nonnegative=True,
        ),
        bottom=_number(
            section.get("margin_bottom", margin),
            SECTION + ".margin_bottom",
            nonnegative=True,
        ),
        top=_number(
            section.get("margin_top", margin),
            SECTION + ".margin_top",
            nonnegative=True,
        ),
    )
    gap = _number(section.get("gap", 0.15), SECTION + ".gap", nonnegative=True)
    column_gap = _number(
        section.get("column_gap", gap),
        SECTION + ".column_gap",
        nonnegative=True,
    )
    row_gap = _number(
        section.get("row_gap", gap), SECTION + ".row_gap", nonnegative=True
    )

    width = (
        _number(section["width"], SECTION + ".width", positive=True)
        if "width" in section else None
    )
    height = (
        _number(section["height"], SECTION + ".height", positive=True)
        if "height" in section
        else None
    )
    if width is not None and width <= margins.left + margins.right + column_gap * (ncols - 1):
        raise ConfigError("canvas width is too small for its margins and column gaps")
    if height is not None and height <= margins.top + margins.bottom + row_gap * (nrows - 1):
        raise ConfigError("canvas height is too small for its margins and row gaps")

    tags = section.get("tags", True)
    if not isinstance(tags, bool):
        raise ConfigError(f"configuration value {SECTION + '.tags'!r} must be boolean")

    tag_weight = section.get("tag_weight", PAPER_PANEL_WEIGHT)
    if not isinstance(tag_weight, str) or not tag_weight:
        raise ConfigError(
            f"configuration value {SECTION + '.tag_weight'!r} must be a string"
        )
    font_family = section.get("font_family", PAPER_FONT_FAMILY)
    if not isinstance(font_family, str) or not font_family:
        raise ConfigError(
            f"configuration value {SECTION + '.font_family'!r} must be a string"
        )

    return Composition(
        panels=panels,
        design=design,
        widths=_ratios(section.get("widths"), SECTION + ".widths", ncols),
        heights=_ratios(section.get("heights"), SECTION + ".heights", nrows),
        width=width,
        height=height,
        margins=margins,
        column_gap=column_gap,
        row_gap=row_gap,
        tags=tags,
        tag_size=_number(
            section.get("tag_size", PAPER_PANEL_SIZE), SECTION + ".tag_size", positive=True
        ),
        tag_weight=tag_weight,
        tag_offset_x=_number(
            section.get("tag_offset_x", 0.04),
            SECTION + ".tag_offset_x",
            nonnegative=True,
        ),
        tag_offset_y=_number(
            section.get("tag_offset_y", 0.04),
            SECTION + ".tag_offset_y",
            nonnegative=True,
        ),
        font_family=font_family,
    )


def _axis_segments(
    ratios: Sequence[float],
    available: float,
    start: float,
    gap: float,
) -> tuple[tuple[float, float], ...]:
    scale = available / sum(ratios)
    result: list[tuple[float, float]] = []
    cursor = start
    for ratio in ratios:
        length = ratio * scale
        result.append((cursor, length))
        cursor += length + gap
    return tuple(result)


def _panel_boxes(composition: Composition, width: float, height: float) -> tuple[PanelBox, ...]:
    nrows = len(composition.design)
    ncols = len(composition.design[0])
    available_width = (
        width
        - composition.margins.left
        - composition.margins.right
        - composition.column_gap * (ncols - 1)
    )
    available_height = (
        height
        - composition.margins.top
        - composition.margins.bottom
        - composition.row_gap * (nrows - 1)
    )
    columns = _axis_segments(
        composition.widths,
        available_width,
        composition.margins.left,
        composition.column_gap,
    )
    # Segment calculation starts at the bottom, so reverse the top-to-bottom
    # row ratios before mapping them back to design row indices.
    bottom_up_rows = _axis_segments(
        tuple(reversed(composition.heights)),
        available_height,
        composition.margins.bottom,
        composition.row_gap,
    )
    rows = tuple(reversed(bottom_up_rows))

    boxes: list[PanelBox] = []
    for panel in composition.panels:
        cells = [
            (row_index, column_index)
            for row_index, row in enumerate(composition.design)
            for column_index, identifier in enumerate(row)
            if identifier == panel.id
        ]
        row_indices = [row for row, _ in cells]
        column_indices = [column for _, column in cells]
        first_row, last_row = min(row_indices), max(row_indices)
        first_column, last_column = min(column_indices), max(column_indices)
        left = columns[first_column][0]
        right = columns[last_column][0] + columns[last_column][1]
        bottom = rows[last_row][0]
        top = rows[first_row][0] + rows[first_row][1]
        boxes.append(
            PanelBox(
                panel=panel,
                left=left,
                bottom=bottom,
                width=right - left,
                height=top - bottom,
            )
        )
    return tuple(boxes)


def panel_boxes(composition: Composition) -> tuple[PanelBox, ...]:
    """Calculate panel areas in inches, preserving panel declaration order."""
    if composition.width is not None and composition.height is not None:
        return _panel_boxes(composition, composition.width, composition.height)
    pages = [_read_source_page(panel) for panel in composition.panels]
    width, height, _, _ = _canvas_layout(composition, pages)
    return _panel_boxes(composition, width, height)


def _read_source_page(panel: Panel) -> PageObject:
    if not panel.figure_path.is_file():
        raise ConfigError(
            f"panel {panel.id!r} figure does not exist: {panel.figure_path}; "
            f"run {panel.workflow!r} with its source configuration in paper mode first"
        )
    try:
        reader = PdfReader(panel.figure_path)
    except Exception as exc:
        raise ConfigError(
            f"cannot read PDF for panel {panel.id!r}: {panel.figure_path}: {exc}"
        ) from exc
    if panel.page > len(reader.pages):
        raise ConfigError(
            f"panel {panel.id!r} requests page {panel.page}, but "
            f"{panel.figure_path} has {len(reader.pages)} page(s)"
        )
    page = reader.pages[panel.page - 1]
    if page.rotation:
        page.transfer_rotation_to_content()
    return page


def _aligned_offset(extra: float, alignment: str) -> float:
    if alignment in {"left", "bottom"}:
        return 0.0
    if alignment == "center":
        return extra / 2
    return extra


def _source_size(panel: Panel, page: PageObject) -> tuple[float, float]:
    """Read a PDF's visible width and height in inches."""
    source_box = page.cropbox
    inches_per_unit = float(page.user_unit) / POINTS_PER_INCH
    source_width = float(source_box.width) * inches_per_unit
    source_height = float(source_box.height) * inches_per_unit
    if source_width <= 0 or source_height <= 0:
        raise ConfigError(f"panel {panel.id!r} PDF page has an empty crop box")
    return source_width, source_height


def _canvas_layout(
    composition: Composition,
    pages: Sequence[PageObject],
) -> tuple[float, float, float, float]:
    """Resolve canvas dimensions, a common panel scale, and the tag gutter in inches."""
    source_sizes = [
        _source_size(panel, page) for panel, page in zip(composition.panels, pages, strict=True)
    ]
    tag_gutter = 0.0
    if any(
        _tag_label(panel, index, composition.tags) for index, panel in enumerate(composition.panels)
    ):
        tag_gutter = composition.tag_offset_y + 1.25 * composition.tag_size / POINTS_PER_INCH

    nrows, ncols = len(composition.design), len(composition.design[0])
    gutter_width = (
        composition.margins.left + composition.margins.right + (ncols - 1) * composition.column_gap
    )
    gutter_height = (
        composition.margins.top + composition.margins.bottom + (nrows - 1) * composition.row_gap
    )
    # Spanning cells contain their fixed intervening gaps as well as a fraction
    # of the usable canvas. Measure each contribution using zero and one inch.
    gap_boxes = _panel_boxes(composition, gutter_width, gutter_height)
    unit_boxes = _panel_boxes(composition, gutter_width + 1, gutter_height + 1)
    columns = [
        (source_width, unit.width - gap.width, gap.width)
        for (source_width, _), unit, gap in zip(source_sizes, unit_boxes, gap_boxes, strict=True)
    ]
    rows = [
        (source_height, unit.height - gap.height, gap.height - tag_gutter)
        for (_, source_height), unit, gap in zip(source_sizes, unit_boxes, gap_boxes, strict=True)
    ]

    scale = 1.0
    if composition.width is not None:
        available = composition.width - gutter_width
        scale = min(
            scale, min((available * fraction + gap) / size for size, fraction, gap in columns)
        )
    if composition.height is not None:
        available = composition.height - gutter_height
        scale = min(scale, min((available * fraction + gap) / size for size, fraction, gap in rows))
    if scale <= 0:
        raise ConfigError("canvas height is too small for its panel tags")

    width = composition.width
    if width is None:
        needed = max((size * scale - gap) / fraction for size, fraction, gap in columns)
        width = gutter_width + max(needed, 1 / POINTS_PER_INCH)
    height = composition.height
    if height is None:
        needed = max((size * scale - gap) / fraction for size, fraction, gap in rows)
        height = gutter_height + max(needed, 1 / POINTS_PER_INCH)
    return width, height, scale, tag_gutter


def _merge_panel(
    canvas: PageObject,
    box: PanelBox,
    page: PageObject,
    scale: float,
    tag_gutter: float,
) -> tuple[float, float]:
    source_box = page.cropbox
    source_width, source_height = _source_size(box.panel, page)
    target_x = box.left + _aligned_offset(
        box.width - source_width * scale, box.panel.horizontal_align
    )
    target_y = box.bottom + _aligned_offset(
        box.height - tag_gutter - source_height * scale, box.panel.vertical_align
    )

    transform = (
        Transformation()
        .translate(tx=-float(source_box.left), ty=-float(source_box.bottom))
        .scale(scale * float(page.user_unit))
        .translate(tx=target_x * POINTS_PER_INCH, ty=target_y * POINTS_PER_INCH)
    )
    canvas.merge_transformed_page(page, transform, over=True, expand=False)
    return target_x, target_y + source_height * scale


def _tag_label(panel: Panel, index: int, enabled: bool) -> str | None:
    if panel.tag is False:
        return None
    if isinstance(panel.tag, str):
        return panel.tag
    if panel.tag is True or enabled:
        return panel_tag(index)
    return None


def _tag_overlay(
    composition: Composition,
    panel_corners: Sequence[tuple[float, float]],
    width: float,
    height: float,
) -> PageObject | None:
    labels = [
        _tag_label(panel, index, composition.tags) for index, panel in enumerate(composition.panels)
    ]
    if not any(labels):
        return None

    import matplotlib as mpl
    from matplotlib.backends.backend_pdf import FigureCanvasPdf
    from matplotlib.figure import Figure

    with mpl.rc_context(
        {
            "font.family": composition.font_family,
            "font.sans-serif": PAPER_SANS_SERIF,
            "pdf.fonttype": 42,
            "savefig.bbox": None,
        }
    ):
        figure = Figure(figsize=(width, height))
        FigureCanvasPdf(figure)
        figure.patch.set_alpha(0)
        for (left, top), label in zip(panel_corners, labels, strict=True):
            if not label:
                continue
            x = (left + composition.tag_offset_x) / width
            y = (top + composition.tag_offset_y) / height
            figure.text(
                x,
                y,
                label,
                ha="left",
                va="bottom",
                fontsize=composition.tag_size,
                fontweight=composition.tag_weight,
                family=composition.font_family,
            )

        stream = io.BytesIO()
        figure.savefig(
            stream,
            format="pdf",
            transparent=True,
            bbox_inches=None,
            pad_inches=0,
        )
    stream.seek(0)
    return PdfReader(stream).pages[0]


def compose(composition: Composition, output: str | Path) -> Path:
    """Compose all source PDF pages onto one vector-preserving PDF page."""
    destination = Path(output).expanduser().resolve()
    if destination.suffix.lower() != ".pdf":
        raise ConfigError("figure-compositor output must use a .pdf extension")
    destination.parent.mkdir(parents=True, exist_ok=True)

    pages = [_read_source_page(panel) for panel in composition.panels]
    width, height, scale, tag_gutter = _canvas_layout(composition, pages)
    canvas = PageObject.create_blank_page(
        width=width * POINTS_PER_INCH,
        height=height * POINTS_PER_INCH,
    )
    boxes = _panel_boxes(composition, width, height)
    panel_corners = [
        _merge_panel(canvas, box, page, scale, tag_gutter)
        for box, page in zip(boxes, pages, strict=True)
    ]
    overlay = _tag_overlay(composition, panel_corners, width, height)
    if overlay is not None:
        canvas.merge_page(overlay, over=True, expand=False)

    writer = PdfWriter()
    writer.add_page(canvas)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.stem}-",
            suffix=".pdf",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            writer.write(stream)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def preflight_paths(config: ProjectConfig) -> list[tuple[str, Path]]:
    """Expose nested panel inputs to the generic ``doctor`` command."""
    composition = parse_composition(config)
    return [
        (f"panel {panel.id} ({panel.workflow})", panel.figure_path)
        for panel in composition.panels
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compose existing atlas-posthoc-plus workflow PDFs."
    )
    parser.add_argument("--config", required=True, help="Compositor project TOML.")
    parser.add_argument("--output", help="Destination PDF.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        composition = parse_composition(config)
        output = (
            config.output_path("figures/composite.pdf")
            if args.output is None
            else config.resolve_path(args.output)
        )
        destination = compose(composition, output)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
