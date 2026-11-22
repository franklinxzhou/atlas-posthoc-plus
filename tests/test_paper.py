from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from atlas_posthoc_plus.config import ConfigError, load_config
from atlas_posthoc_plus.paper import (
    PANEL_LABEL_GID,
    PAPER_MODE_ENV,
    apply_paper_style,
    save_figure,
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
    assert mpl.rcParams["font.sans-serif"][0] == "Helvetica Neue"
    plt.close(figure)


def test_paper_mode_does_not_label_three_panel_figure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PAPER_MODE_ENV, "1")
    figure, _ = plt.subplots(1, 3)

    save_figure(figure, tmp_path / "three-panels.svg")

    assert not [
        artist
        for artist in figure.findobj()
        if getattr(artist, "get_gid", lambda: None)() == PANEL_LABEL_GID
    ]
    assert (tmp_path / "three-panels.pdf").is_file()
    plt.close(figure)


def test_paper_mode_does_not_label_two_stacked_panels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PAPER_MODE_ENV, "1")
    figure, _ = plt.subplots(2, 1)

    save_figure(figure, tmp_path / "stacked.png")

    assert not [
        artist
        for artist in figure.findobj()
        if getattr(artist, "get_gid", lambda: None)() == PANEL_LABEL_GID
    ]
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
