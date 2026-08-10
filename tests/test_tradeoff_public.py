from pathlib import Path

from matplotlib.legend import Legend

from atlas_posthoc_plus.workflows.municipal import tradeoff_public


def test_plot_tradeoff_can_pool_chains_under_custom_label(
    tmp_path: Path, monkeypatch
) -> None:
    captured = {}

    def capture_figure(fig, path, **kwargs):
        captured["figure"] = fig
        return Path(path)

    monkeypatch.setattr(tradeoff_public, "save_figure", capture_figure)

    traces = [
        {
            "chain_id": "run501",
            "n_cut_municipalities": [20, 22],
            "ousted_population_share": [0.04, 0.05],
        },
        {
            "chain_id": "run502",
            "n_cut_municipalities": [24, 26],
            "ousted_population_share": [0.045, 0.055],
        },
    ]
    refs = {
        label: {
            "n_cut_municipalities": 28 + i,
            "ousted_population_share": 0.035 + i * 0.002,
        }
        for i, label in enumerate(tradeoff_public.OFFICIAL_STYLES)
    }

    tradeoff_public.plot_tradeoff(
        traces=traces,
        refs=refs,
        public_summaries=[],
        output=tmp_path / "tradeoff.pdf",
        title="Municipal split tradeoff",
        y_key="ousted_population_share",
        y_label="Ousted municipal population share",
        use_percent_axis=True,
        label_references=False,
        label_public=False,
        pool_chains=True,
        ensemble_label="Place-5 ensemble",
    )

    ax = captured["figure"].axes[0]
    ensemble_legend = next(
        artist
        for artist in ax.artists
        if isinstance(artist, Legend) and artist.get_title().get_text() == "Ensemble"
    )
    assert [text.get_text() for text in ensemble_legend.get_texts()] == [
        "Place-5 ensemble"
    ]
    assert len(ax.collections[0].get_offsets()) == 4

    all_legend_text = [
        text.get_text()
        for legend in [*ax.artists, ax.get_legend()]
        if isinstance(legend, Legend)
        for text in legend.get_texts()
    ]
    assert "run501" not in all_legend_text
    assert "run502" not in all_legend_text
