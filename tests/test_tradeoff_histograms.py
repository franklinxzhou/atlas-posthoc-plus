from pathlib import Path

import numpy as np
import pytest

from atlas_posthoc_plus.workflows.municipal import tradeoff_histograms


@pytest.mark.parametrize("share", [True, False])
def test_tradeoff_histograms_normalize_marginals_and_keep_exact_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, share: bool
) -> None:
    def inspect_figure(figure, output):
        population, cuts = figure.axes
        widths = np.array([bar.get_width() for bar in population.patches])
        heights = np.array([bar.get_height() for bar in population.patches])
        assert (widths * heights).sum() == pytest.approx(0.01 if share else 100_000)
        assert [bar.get_height() for bar in cuts.patches] == pytest.approx([0.25, 0, 0.75])
        assert [bar.get_x() for bar in cuts.patches] == [-0.5, 0.5, 1.5]
        assert cuts.get_xlim()[0] <= -0.5
        assert {label.xy for label in cuts.texts} == {(0, 0), (4, 0)}
        assert "Enacted" in cuts.texts[0].get_text()
        assert "Remedial" in cuts.texts[0].get_text()
        assert {label.xy for label in population.texts} == {
            (reference_a, 0),
            (reference_b, 0),
        }
        return output

    reference_a, reference_b = (0.02, 0.07) if share else (2_000, 7_000)
    monkeypatch.setattr(tradeoff_histograms, "save_figure", inspect_figure)
    tradeoff_histograms.plot_tradeoff_histograms(
        cut_values=np.array([0, 2, 2, 2]),
        ousted_values=np.linspace(reference_a, reference_b, 4),
        references={
            "Enacted ('25)": (0, reference_a),
            "Remedial ('22)": (0, reference_b),
            "Overturned ('21)": (4, reference_b),
        },
        cut_label="Number of cut places",
        ousted_label="Ousted population",
        share=share,
        output=tmp_path / "tradeoff.pdf",
    )


def test_tradeoff_histograms_handles_an_unrepresented_category(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def inspect_figure(figure, output):
        population, cuts = figure.axes
        assert population.texts[0].get_text() == "No represented population"
        assert all(tick == int(tick) for tick in cuts.get_xticks())
        assert cuts.patches[0].get_height() == 1
        return output

    monkeypatch.setattr(tradeoff_histograms, "save_figure", inspect_figure)
    tradeoff_histograms.plot_tradeoff_histograms(
        cut_values=np.array([0, 0]),
        ousted_values=np.array([np.nan, np.nan]),
        references={"Enacted ('25)": (0, np.nan)},
        cut_label="Number of cut townships",
        ousted_label="Ousted township population share",
        share=True,
        output=tmp_path / "tradeoff.pdf",
    )
