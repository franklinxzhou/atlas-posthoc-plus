from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import pytest
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.figure import Figure
from pypdf import PdfReader

from atlas_posthoc_plus.config import ConfigError, load_config
from atlas_posthoc_plus.preflight import check_workflow
from atlas_posthoc_plus.registry import WORKFLOWS
from atlas_posthoc_plus.runner import prepare_workflow
from atlas_posthoc_plus.workflows.compositor import (
    compose,
    panel_boxes,
    parse_composition,
)


def _write_source_config(path: Path, root: str) -> None:
    path.write_text(
        f"""
[project]
root = {root!r}

[outputs]
root = "outputs"
""",
        encoding="utf-8",
    )


def _write_vector_pdf(path: Path, slope: float) -> None:
    with mpl.rc_context({"pdf.fonttype": 42}):
        figure = Figure(figsize=(2, 1.5))
        FigureCanvasPdf(figure)
        axis = figure.subplots()
        axis.plot([0, 1], [0, slope])
        axis.set_xlabel("x")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, format="pdf", bbox_inches="tight")


def test_registered_workflow_passes_its_project_config(tmp_path: Path) -> None:
    config_path = tmp_path / "composition.toml"
    config_path.write_text(
        """
[project]
root = "."
[outputs]
root = "outputs"
""",
        encoding="utf-8",
    )

    prepared = prepare_workflow(
        WORKFLOWS["figure-compositor"], load_config(config_path)
    )

    assert prepared.argv[:2] == ("--config", str(config_path.resolve()))
    assert prepared.argv[2:] == (
        "--output",
        str((tmp_path / "outputs/figures/composite.pdf").resolve()),
    )


def test_patchwork_design_supports_spans_spacers_and_ratios(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_config = tmp_path / "source.toml"
    _write_source_config(source_config, "source")
    config_path = tmp_path / "composition.toml"
    config_path.write_text(
        """
[project]
root = "."

[workflow.figure-compositor]
design = '''
AAB
#CB
'''
widths = [1, 2, 1]
heights = [1, 2]
width = 10
height = 6
margin = 0
gap = 0

[[workflow.figure-compositor.panels]]
id = "A"
workflow = "polsby-histogram"
config = "source.toml"
figure = "figures/a.pdf"

[[workflow.figure-compositor.panels]]
id = "B"
workflow = "responsiveness"
config = "source.toml"
figure = "figures/b.pdf"

[[workflow.figure-compositor.panels]]
id = "C"
workflow = "municipal-tradeoff"
config = "source.toml"
figure = "figures/c.pdf"
""",
        encoding="utf-8",
    )

    composition = parse_composition(load_config(config_path))
    boxes = {box.panel.id: box for box in panel_boxes(composition)}

    assert (boxes["A"].left, boxes["A"].bottom) == pytest.approx((0, 4))
    assert (boxes["A"].width, boxes["A"].height) == pytest.approx((7.5, 2))
    assert (boxes["B"].left, boxes["B"].bottom) == pytest.approx((7.5, 0))
    assert (boxes["B"].width, boxes["B"].height) == pytest.approx((2.5, 6))
    assert (boxes["C"].left, boxes["C"].bottom) == pytest.approx((2.5, 0))
    assert (boxes["C"].width, boxes["C"].height) == pytest.approx((5, 4))
    assert boxes["A"].panel.figure_path == (
        source_root / "figures/a.pdf"
    ).resolve()


def test_automatic_wrapping_leaves_trailing_spacer(tmp_path: Path) -> None:
    _write_source_config(tmp_path / "source.toml", ".")
    panel_tables = "\n".join(
        f"""
[[workflow.figure-compositor.panels]]
workflow = {workflow!r}
config = "source.toml"
figure = {f"{letter}.pdf"!r}
"""
        for letter, workflow in zip(
            ("a", "b", "c"),
            ("polsby-histogram", "responsiveness", "municipal-tradeoff"),
        )
    )
    config_path = tmp_path / "composition.toml"
    config_path.write_text(
        f"""
[project]
root = "."

[workflow.figure-compositor]
ncol = 2

{panel_tables}
""",
        encoding="utf-8",
    )

    composition = parse_composition(load_config(config_path))

    assert composition.design == (("A", "B"), ("C", None))


def test_design_rejects_non_rectangular_span(tmp_path: Path) -> None:
    _write_source_config(tmp_path / "source.toml", ".")
    config_path = tmp_path / "composition.toml"
    config_path.write_text(
        """
[project]
root = "."

[workflow.figure-compositor]
design = "AA\\nAB"

[[workflow.figure-compositor.panels]]
id = "A"
workflow = "polsby-histogram"
config = "source.toml"
figure = "a.pdf"

[[workflow.figure-compositor.panels]]
id = "B"
workflow = "responsiveness"
config = "source.toml"
figure = "b.pdf"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="rectangular"):
        parse_composition(load_config(config_path))


def test_composition_is_one_page_vector_pdf_with_automatic_tags(
    tmp_path: Path,
) -> None:
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    source_a.mkdir()
    source_b.mkdir()
    _write_source_config(tmp_path / "a.toml", "source-a")
    _write_source_config(tmp_path / "b.toml", "source-b")
    _write_vector_pdf(source_a / "figure.pdf", 1)
    _write_vector_pdf(source_b / "figure.pdf", -1)

    config_path = tmp_path / "composition.toml"
    config_path.write_text(
        """
[project]
root = "."

[workflow.figure-compositor]
nrow = 1
width = 8
height = 3
margin = 0.2
gap = 0.1

[[workflow.figure-compositor.panels]]
workflow = "polsby-histogram"
config = "a.toml"
figure = "figure.pdf"

[[workflow.figure-compositor.panels]]
workflow = "responsiveness"
config = "b.toml"
figure = "figure.pdf"
""",
        encoding="utf-8",
    )
    output = tmp_path / "combined.pdf"

    destination = compose(parse_composition(load_config(config_path)), output)
    raw_pdf = destination.read_bytes()
    reader = PdfReader(destination)

    assert len(reader.pages) == 1
    assert float(reader.pages[0].mediabox.width) == pytest.approx(8 * 72)
    assert float(reader.pages[0].mediabox.height) == pytest.approx(3 * 72)
    text = reader.pages[0].extract_text()
    assert "(a)" in text
    assert "(b)" in text
    assert b"/Subtype /Image" not in raw_pdf


def test_non_pdf_panel_is_rejected_with_vector_guidance(tmp_path: Path) -> None:
    _write_source_config(tmp_path / "source.toml", ".")
    config_path = tmp_path / "composition.toml"
    config_path.write_text(
        """
[project]
root = "."

[workflow.figure-compositor]

[[workflow.figure-compositor.panels]]
workflow = "polsby-histogram"
config = "source.toml"
figure = "figure.png"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="must be PDF"):
        parse_composition(load_config(config_path))


def test_doctor_checks_nested_panel_figure_paths(tmp_path: Path) -> None:
    _write_source_config(tmp_path / "source.toml", ".")
    config_path = tmp_path / "composition.toml"
    config_path.write_text(
        """
[project]
root = "."

[workflow.figure-compositor]

[[workflow.figure-compositor.panels]]
workflow = "polsby-histogram"
config = "source.toml"
figure = "missing.pdf"
""",
        encoding="utf-8",
    )

    checks = check_workflow(
        load_config(config_path), WORKFLOWS["figure-compositor"]
    )

    panel_check = next(check for check in checks if check.label.startswith("panel A"))
    assert panel_check.ok is False
    assert panel_check.detail == str((tmp_path / "missing.pdf").resolve())
