from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
from pypdf import PdfReader

from atlas_posthoc_plus.config import ConfigError, load_config
from atlas_posthoc_plus.paper import (
    PANEL_LABEL_GID,
    PAPER_MODE_ENV,
    apply_paper_style,
    figure_legend,
    official_plan_style,
    save_figure,
    set_panel_title,
)
from atlas_posthoc_plus.registry import WorkflowSpec
from atlas_posthoc_plus.runner import _subprocess_environment, prepare_workflow


def test_paper_mode_saves_pdf_without_titles_and_adds_panel_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PAPER_MODE_ENV, "1")
    apply_paper_style()

    figure, axes = plt.subplots(1, 2)
    figure.suptitle("Figure title")
    axes[0].set_title("Left title")
    axes[1].set_title("Left-aligned subtitle", loc="left")
    axes[1].set_title("Right-aligned subtitle", loc="right")
    axes[0].plot([0, 1], [0, 1])
    axes[1].plot([0, 1], [1, 0])

    destination = save_figure(figure, tmp_path / "figure.png", dpi=72)

    assert destination == tmp_path / "figure.pdf"
    assert destination.read_bytes().startswith(b"%PDF")
    assert figure._suptitle.get_text() == ""
    assert all(
        not axis.get_title(loc=location)
        for axis in axes
        for location in ("left", "center", "right")
    )
    assert [
        artist.get_text()
        for artist in figure.findobj()
        if getattr(artist, "get_gid", lambda: None)() == PANEL_LABEL_GID
    ] == ["(a)", "(b)"]
    assert mpl.rcParams["font.sans-serif"][0] == "Nimbus Sans"
    plt.close(figure)


def test_paper_mode_labels_three_panels_and_keeps_plan_headings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PAPER_MODE_ENV, "1")
    figure, axes = plt.subplots(1, 3)
    set_panel_title(axes[0], "2025 enacted")
    set_panel_title(axes[1], "Overturned (2021)")
    set_panel_title(axes[2], "Remedial ('22)")
    axes[0].set_title("Descriptive title", loc="center")

    save_figure(figure, tmp_path / "first.svg")
    output = save_figure(figure, tmp_path / "three-panels.svg")

    assert [
        artist.get_text()
        for artist in figure.findobj()
        if getattr(artist, "get_gid", lambda: None)() == PANEL_LABEL_GID
    ] == ["(a)", "(b)", "(c)"]
    text = PdfReader(output).pages[0].extract_text()
    assert all(plan in text for plan in ("Enacted ('25)", "Overturned ('21)", "Remedial ('22)"))
    assert "Descriptive title" not in text
    plt.close(figure)


def test_paper_mode_labels_two_stacked_panels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PAPER_MODE_ENV, "1")
    figure, _ = plt.subplots(2, 1)

    save_figure(figure, tmp_path / "stacked.png")

    assert [
        artist.get_text()
        for artist in figure.findobj()
        if getattr(artist, "get_gid", lambda: None)() == PANEL_LABEL_GID
    ] == ["(a)", "(b)"]
    plt.close(figure)


def test_paper_mode_normalizes_explicit_font_sizes_and_official_legend(tmp_path, monkeypatch):
    monkeypatch.setenv(PAPER_MODE_ENV, "1")
    figure, axis = plt.subplots()
    axis.set_xlabel("Count", fontsize=16)
    axis.tick_params(labelsize=13)
    annotation = axis.annotate("Sample", (0.5, 0.5), fontsize=12, fontfamily="serif")
    for plan in ("Remedial", "2021 overturned", "Enacted (2025)"):
        axis.plot(
            [0, 1],
            [0, 1],
            label=plan,
            **official_plan_style(
                plan,
                color="black",
                marker="x",
                linestyle=":",
            ),
        )
    legend = figure_legend(axis, title="Reference plans", fontsize=11, frameon=True)

    save_figure(figure, tmp_path / "styled.pdf")

    assert axis.xaxis.label.get_fontsize() == 10
    assert all(tick.get_fontsize() == 8 for tick in axis.get_xticklabels())
    assert annotation.get_fontsize() == 8
    assert annotation.get_fontfamily() == ["sans-serif"]
    assert legend.get_title().get_text() == "Official plans"
    assert legend.get_title().get_fontsize() == 9
    assert not legend.get_frame_on()
    assert [text.get_text() for text in legend.get_texts()] == [
        "Enacted ('25)",
        "Overturned ('21)",
        "Remedial ('22)",
    ]
    assert all(text.get_fontsize() == 9 for text in legend.get_texts())
    assert [line.get_color() for line in legend.get_lines()] == ["#7A5195", "#0072B2", "#E69F00"]
    assert [line.get_marker() for line in legend.get_lines()] == ["o", "s", "D"]
    plt.close(figure)


def test_prepare_workflow_propagates_configured_paper_mode(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[project]
root = "."

[figures]
paper_mode = true
""",
        encoding="utf-8",
    )
    spec = WorkflowSpec("test", "example.workflow", "test", "Example", ())

    prepared = prepare_workflow(spec, load_config(config_path))
    environment = _subprocess_environment(paper_mode=prepared.paper_mode)

    assert prepared.paper_mode is True
    assert prepared.display_command().startswith(f"{PAPER_MODE_ENV}=1 ")
    assert environment[PAPER_MODE_ENV] == "1"


def test_paper_mode_must_be_boolean(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[project]
root = "."

[figures]
paper_mode = "yes"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="must be a boolean"):
        prepare_workflow(
            WorkflowSpec("test", "example.workflow", "test", "Example", ()),
            load_config(config_path),
        )
