import pytest

from atlas_posthoc_plus.workflows.elections import partisan_asymmetry
from atlas_posthoc_plus.workflows.elections.partisan_asymmetry import (
    read_margins,
    summarize_margins,
)
from atlas_posthoc_plus.workflows.elections.recast_precinct_votes import recast_votes


@pytest.fixture
def district_votes(tmp_path):
    # 2026 General Election data in Ohio in preparation
    source = tmp_path / "district-votes.csv"
    source.write_text(
        "plan,district,dem_votes,rep_votes,total_votes\n"
        "A,1,60,30,100\n"
        "A,2,46,44,100\n"
        "A,3,47,43,100\n"
        "A,4,37,53,100\n"
        "B,1,60,30,100\n"
        "B,2,44,46,100\n"
        "B,3,39,51,100\n"
        "B,4,47,43,100\n"
        "C,1,54,36,100\n"
        "C,2,44,46,100\n"
        "C,3,45,45,100\n"
        "C,4,47,43,100\n",
        encoding="utf-8",
    )
    return read_margins(source)


def test_summary_preserves_close_wins_and_all_vote_denominator(district_votes):
    summary = summarize_margins(district_votes, close_pp=5)

    assert summary["dem_wins"].tolist() == [3, 2, 2]
    assert summary["ties"].tolist() == [0, 0, 1]
    assert summary["dem_wins_within_threshold"].tolist() == [2, 1, 1]
    assert summary["rep_wins_within_threshold"].tolist() == [0, 1, 1]
    assert summary["dem_minus_rep_close_win_share_pp"].tolist() == pytest.approx([200 / 3, 0, -50])
    assert summary["majority_seat_margin_pp"].tolist() == pytest.approx([2, -2, 0])
    district_two = district_votes.loc[district_votes["district"] == 2, "margin_pp"].tolist()
    assert district_two == pytest.approx([2, -2, -2])


def test_zoom_preserves_district_rows_and_margins(tmp_path, monkeypatch, district_votes):
    panels = {}

    def capture_panel(figure, output, **kwargs):
        (axis,) = figure.axes
        panels[output] = axis
        return output

    monkeypatch.setattr(partisan_asymmetry, "save_figure", capture_panel)
    full_pdf, zoom_pdf = partisan_asymmetry.plot_margins(
        district_votes, output=tmp_path / "figure.pdf"
    )

    assert (full_pdf, zoom_pdf) == (tmp_path / "figure.pdf", tmp_path / "figure-zoom.pdf")
    overview, zoom = panels[full_pdf], panels[zoom_pdf]
    assert zoom.get_xlim() == (-20, 20)
    assert zoom.get_ylim() == overview.get_ylim() == (3.5, -0.5)
    assert [tick.get_text() for tick in zoom.get_yticklabels()] == ["1", "2", "3", "4"]
    main_markers = [line for line in overview.lines if line.get_linestyle() == "None"]
    zoom_markers = [line for line in zoom.lines if line.get_linestyle() == "None"]
    assert len(main_markers) == len(zoom_markers) == 3
    for main, detail in zip(main_markers, zoom_markers, strict=True):
        visible = abs(main.get_xdata()) <= 20
        assert detail.get_xdata() == pytest.approx(main.get_xdata()[visible])
        assert detail.get_ydata() == pytest.approx(main.get_ydata()[visible])


def test_recast_preserves_split_precinct_pieces_and_audits_missing_blocks(tmp_path):
    precincts = tmp_path / "precincts.csv"
    precincts.write_text(
        "county,precinct,source_district,real_precinct,dem_votes,rep_votes,other_votes\n"
        "01,P,1,true,60,20,0\n"
        "01,Q,1,true,20,60,10\n"
        "01,Missing,1,true,0,10,0\n"
        "01,Admin,1,false,20,18,10\n"
        "01,P,2,true,10,80,4\n"
    )
    blocks = tmp_path / "blocks.csv"
    blocks.write_text(
        "block_id,county,precinct,source_district,vap,population,land_area,alternate,other\n"
        "b1,01,P,1,3,3,1,1,1\n"
        "b2,01,P,1,1,1,1,2,1\n"
        "b3,01,Q,1,0,5,1,2,2\n"
        "b4,01,P,2,0,0,3,1,2\n"
    )
    totals, audit = recast_votes(
        precincts, blocks, {"Source": "source_district", "Alternate": "alternate", "Other": "other"}
    )
    districts = totals.set_index(["plan", "district"])
    assert districts.loc[("Source", 1), ["dem_votes", "rep_votes", "other_votes"]].tolist() == [
        100, 96, 20
    ]
    assert districts.loc[("Alternate", 1), ["dem_votes", "rep_votes", "other_votes"]].tolist() == [
        66.25, 98, 4
    ]
    assert audit["unallocated_rep_votes"].tolist() == [12, 0]
    assert totals.groupby("plan")["total_votes"].sum().tolist() == [310, 310, 310]
