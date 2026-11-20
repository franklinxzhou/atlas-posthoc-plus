import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader

from atlas_posthoc_plus.io.graph import read_graph_frame
from atlas_posthoc_plus.workflows.municipal.mct_geography import (
    MEMBERSHIPS,
    GeographySettings,
    classify_geography,
    load_membership,
)
from atlas_posthoc_plus.workflows.municipal.mct_metrics import SplitScorer


def score(rows: list, *, details: bool = False) -> tuple:
    pieces = pd.DataFrame(rows, columns=["district", "population", *MEMBERSHIPS])
    pieces["position"] = np.arange(len(pieces))
    ids = sorted(set(pieces[MEMBERSHIPS].to_numpy().ravel()) - {""})
    units = pd.DataFrame(
        {
            "unit_id": ids,
            "kind": ["township" if uid.startswith("T") else "city" for uid in ids],
        }
    )
    scorer = SplitScorer(pieces, units, statewide_population=pieces["population"].sum())
    return scorer.score(pieces["district"].to_numpy(), details=details)


@pytest.mark.parametrize(
    ("rows", "cities", "townships", "ousted"),
    [
        ([(1, 20, "C", "T"), (30, 0, "C", "T")], 0, 0, 0),
        ([(1, 20, "C", ""), (2, 5, "C", "")], 1, 0, 5),
        ([(1, 20, "", "T"), (2, 5, "", "T")], 0, 1, 5),
        ([(1, 20, "C", ""), (2, 5, "C", ""), (1, 30, "", "T"), (2, 7, "", "T")], 1, 1, 12),
        ([(1, 20, "", "")], 0, 0, 0),
    ],
)
def test_unsplit_single_cut_and_disjoint_units(rows, cities, townships, ousted):
    summary, _ = score(rows)
    assert summary["cut_cities"] == cities
    assert summary["cut_townships"] == townships
    assert summary["cut_mct_units"] == cities + townships
    assert summary["deduplicated_ousted_population"] == ousted
    assert summary["naive_ousted_population"] == ousted
    if not cities and not townships and not rows[0][2] and not rows[0][3]:
        assert summary["ousted_share_represented"] is None


def test_four_cases_with_different_dominants():
    summary, details = score(
        [
            (1, 10, "C", "T"),  # township only
            (2, 8, "C", "T"),  # municipal corporation only
            (3, 5, "C", "T"),  # both
            (3, 7, "C", "T"),  # a second piece with those memberships
            (1, 20, "C", ""),  # neither
            (2, 25, "", "T"),  # neither
            (4, 4, "", ""),  # neither, outside the represented population
        ],
        details=True,
    )
    assert details.set_index("unit_id")["dominant_district"].to_dict() == {"C": 1, "T": 2}
    assert summary["population_ousted_from_neither"] == 49
    assert summary["population_ousted_from_municipal_corporation_only"] == 8
    assert summary["population_ousted_from_township_only"] == 10
    assert summary["population_ousted_from_both"] == 12
    assert summary["municipal_corporation_ousted_population"] == 20
    assert summary["township_ousted_population"] == 22
    assert summary["cut_municipal_corporations"] == 1
    assert summary["municipal_corporation_ousted_share"] == pytest.approx(20 / 50)
    assert summary["township_ousted_share"] == pytest.approx(22 / 55)
    assert summary["naive_ousted_population"] == 42
    assert summary["deduplicated_ousted_population"] == 30
    assert summary["double_counted_overlap"] == 12
    assert summary["ousted_share_statewide"] == pytest.approx(30 / 79)
    assert summary["ousted_share_represented"] == pytest.approx(30 / 75)
    assert (
        summary["township_ousted_population"]
        <= summary["deduplicated_ousted_population"]
        <= summary["naive_ousted_population"]
    )
    assert summary["deduplicated_ousted_population"] <= summary["represented_population"]


def test_ties_use_numeric_district_order_and_overlap_is_counted_once():
    summary, details = score([(10, 500, "C", "T"), (2, 500, "C", "T")], details=True)
    assert details["dominant_district"].tolist() == [2, 2]
    assert details["ousted_population"].tolist() == [500, 500]
    assert summary["naive_ousted_population"] == 1000
    assert summary["deduplicated_ousted_population"] == 500


def _classification(source: str, pairs: list[tuple[str, str, str]]) -> pd.DataFrame:
    width = 5 if source == "PLACE" else 8
    return pd.DataFrame(
        [
            {
                "GEOID": f"39{i:0{width}}",
                "NAME": source,
                "LSADC": code,
                "CLASSFP": classfp,
                "FUNCSTAT": funcstat,
            }
            for i, (classfp, code, funcstat) in enumerate(pairs, start=1)
        ]
    )


def test_lsadc_classification_and_independent_cousub_guard():
    places = _classification(
        "PLACE",
        [
            ("C1", "25", "A"),
            ("C2", "47", "A"),
            ("C5", "25", "B"),
            ("C6", "25", "C"),
            ("U1", "57", "S"),
            ("U2", "57", "S"),
            ("U5", "57", "S"),
            ("C1", "99", "A"),
            ("C3", "25", "A"),
        ],
    )
    place = classify_geography(places, "PLACE", township_lsadc="47")
    assert place["kind"].tolist() == ["city", "village", "city", "city", "", "", "", "", "city"]
    assert place["review"].ne("").tolist() == [False] * 7 + [True, True]

    subdivisions = _classification(
        "COUSUB",
        [
            ("C2", "25", "A"),
            ("C5", "47", "A"),
            ("T1", "47", "A"),
            ("T5", "47", "B"),
            ("T9", "47", "N"),
            ("Z5", "47", "S"),
            ("T1", "44", "A"),
            ("T5", "44", "G"),
        ],
    )
    recoded = classify_geography(subdivisions, "COUSUB", township_lsadc="47")
    standard = classify_geography(subdivisions, "COUSUB", township_lsadc="44")
    assert recoded["accepted"].tolist() == [False, False, True, True, False, False, False, False]
    assert standard["accepted"].tolist() == [False] * 6 + [True, True]


def _pl_row(
    level: str,
    geoid: str,
    population: int,
    *,
    county: str = "",
    place: str = "",
    place_class: str = "",
    cousub: str = "",
    cousub_class: str = "",
    name: str = "",
    lsadc: str = "",
    funcstat: str = "",
) -> str:
    fields = [""] * 97
    for index, value in {
        0: "PLST",
        1: "OH",
        2: level,
        9: geoid,
        12: "39",
        14: county,
        17: cousub,
        18: cousub_class,
        29: place,
        30: place_class,
        87: name,
        88: funcstat,
        90: str(population),
        94: lsadc,
    }.items():
        fields[index] = value
    return "|".join(fields)


@pytest.fixture
def inputs(tmp_path: Path) -> dict:
    counties = ["39001"] * 3 + ["39003"] + ["39001"] * 4
    blocks = [county + f"{i + 1:010}" for i, county in enumerate(counties)]
    populations = [10, 8, 5, 20, 25, 9, 4, 0]
    pd.DataFrame(
        {
            "GEOID20": blocks[:-1],
            "NAME": [f"n{i}" for i in range(7)],
            "POP20": populations[:-1],
        }
    ).to_csv(tmp_path / "crosswalk.csv", index=False)
    place_codes = ["00001"] * 4 + ["99999", "00002", "00003", "00001"]
    place_classes = ["C6"] * 4 + ["99", "C1", "U1", "C6"]
    cousub_codes = ["00001"] * 5 + ["00002", "00003", "00001"]
    cousub_classes = ["T1"] * 3 + ["C2", "T1", "C5", "Z5", "T1"]
    pl_rows = [_pl_row("040", "39", sum(populations))]
    for geoid, pop, name, classfp, code, status in [
        ("3900001", 43, "Test city", "C6", "25", "A"),
        ("3900002", 9, "Test village", "C1", "47", "A"),
        ("3900003", 4, "Test CDP", "U1", "57", "S"),
    ]:
        pl_rows.append(
            _pl_row(
                "160",
                geoid,
                pop,
                place=geoid[2:],
                place_class=classfp,
                lsadc=code,
                funcstat=status,
                name=name,
            )
        )
    for geoid, pop, name, classfp, code, status in [
        ("3900100001", 48, "Test township", "T1", "44", "A"),
        ("3900300001", 20, "Test city", "C2", "25", "F"),
        ("3900100002", 9, "Test village", "C5", "47", "F"),
        ("3900100003", 4, "Test statistical subdivision", "Z5", "22", "S"),
    ]:
        pl_rows.append(
            _pl_row(
                "060",
                geoid,
                pop,
                county=geoid[2:5],
                cousub=geoid[5:],
                cousub_class=classfp,
                lsadc=code,
                funcstat=status,
                name=name,
            )
        )
    for geoid, pop, place, place_class, cousub, cousub_class in zip(
        blocks,
        populations,
        place_codes,
        place_classes,
        cousub_codes,
        cousub_classes,
    ):
        pl_rows.append(
            _pl_row(
                "750",
                geoid,
                pop,
                county=geoid[2:5],
                place=place,
                place_class=place_class,
                cousub=cousub,
                cousub_class=cousub_class,
            )
        )
    census_path = tmp_path / "ohgeo2020.pl"
    census_path.write_text("\n".join(pl_rows) + "\n")
    with ZipFile(tmp_path / "oh2020.pl.zip", "w") as archive:
        archive.write(census_path, arcname=census_path.name)
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": i, "NAME": f"n{i}", "COUNTY": county, "POP20": pop}
                    for i, (county, pop) in enumerate(zip(counties[:-1], populations[:-1]))
                ]
            }
        )
    )
    compliant = [1, 2, 1, 3, 2, 1, 2]
    noncompliant = [1, 2, 3, 1, 2, 3, 3]
    for label, districts in (
        ("overturned", compliant),
        ("remedial", noncompliant),
        ("enacted", [1, 1, 1, 3, 1, 2, 2]),
    ):
        assignments = pd.DataFrame({"Block": blocks[:-1], "DistrictID": districts})
        if label == "enacted":
            assignments.to_excel(tmp_path / "enacted.xlsx", index=False)
        else:
            assignments.to_csv(tmp_path / f"{label}.csv", index=False)
    lines = [
        json.dumps(
            {
                "name": f"plan {index}",
                "weight": index + 1,
                "districting": {
                    json.dumps([f"n{i}"]): district for i, district in enumerate(values)
                },
            }
        )
        for index, values in enumerate([compliant, [], noncompliant, [], compliant])
    ]
    header = "header\nheader\nheader\n"
    (tmp_path / "one.jsonl").write_text(header + "\n".join(lines))
    first_plan = json.loads(lines[0])
    first_plan["districting"][json.dumps(["n1"])] = 1
    first_plan["districting"][json.dumps(["n2"])] = 1
    (tmp_path / "two.jsonl").write_text(header + "\n".join([json.dumps(first_plan), *lines[1:]]))
    return {
        "graph_path": graph_path,
        "graph": read_graph_frame(graph_path),
        "census_path": census_path,
        "crosswalk_path": tmp_path / "crosswalk.csv",
        "cache_dir": tmp_path / "cache",
        "settings": GeographySettings(township_lsadc="44", expected_population=sum(populations)),
    }


def test_county_components_and_membership_cache(inputs):
    whole, first = load_membership(**inputs)
    assert not first["cache_hit"]
    assert load_membership(**inputs)[1]["cache_hit"]
    assert set(whole.units["unit_id"]) == {
        "COUSUB:3900100001",
        "PLACE:3900001",
        "PLACE:3900002",
    }
    city = whole.units.loc[whole.units["kind"].eq("city")].iloc[0]
    assert city["county_geoids"] == "39001|39003"
    assert (
        whole.classification.set_index(["source", "geoid"]).at[("COUSUB", "3900300001"), "decision"]
        == "incorporated COUSUB"
    )
    assert whole.blocks.loc[whole.blocks["population"].eq(4), MEMBERSHIPS].iloc[0].tolist() == [
        "",
        "",
    ]

    settings = replace(inputs["settings"], unit_definition="county-component")
    split, second = load_membership(**{**inputs, "settings": settings})
    assert second["cache_key"] != first["cache_key"]
    assert {"PLACE:3900001:39001", "PLACE:3900001:39003"} <= set(split.units["unit_id"])
    assignment = np.array([1, 1, 1, 2, 1, 1, 1])
    assert (
        SplitScorer(whole.graph_pieces, whole.units, statewide_population=81).score(
            assignment,
        )[0]["cut_cities"]
        == 1
    )
    assert (
        SplitScorer(split.graph_pieces, split.units, statewide_population=81).score(
            assignment,
        )[0]["cut_cities"]
        == 0
    )

    inputs["census_path"].write_text(inputs["census_path"].read_text() + "\n")
    assert load_membership(**inputs)[1]["cache_key"] != first["cache_key"]


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("duplicate", "Duplicate positive block/graph-node"),
        ("missing", "missing positive-population blocks"),
        ("mass", "Block-to-graph population is not conserved"),
        ("node", "Graph-node population is not conserved"),
        ("split", "multiple graph nodes"),
    ],
)
def test_crosswalk_validation(inputs, case, message):
    path = inputs["crosswalk_path"]
    table = pd.read_csv(path)
    if case == "duplicate":
        table = pd.concat([table, table.iloc[:1]])
    elif case == "missing":
        table = table.iloc[1:]
    elif case == "mass":
        table.loc[0, "POP20"] -= 1
    elif case == "node":
        table.loc[[0, 1], "NAME"] = ["n1", "n0"]
    else:
        table.loc[0, "POP20"] -= 1
        table.loc[len(table)] = [table.loc[0, "GEOID20"], "n1", 1]
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_membership(**inputs)
    if case == "split":
        inputs["graph"].loc[[0, 1], "POP20"] = [9, 9]
        inputs["graph_path"].write_text(json.dumps({"nodes": inputs["graph"].to_dict("records")}))
        settings = replace(inputs["settings"], split_block_policy="allocated")
        membership, _ = load_membership(**{**inputs, "settings": settings})
        assert membership.graph_pieces["population"].sum() == 81
        assert membership.allocation_audit["split_positive_blocks"] == 1


def test_conflicting_and_incomplete_census_membership(inputs):
    path = inputs["census_path"]
    original = path.read_text().splitlines()
    positive_block = original[-2]

    path.write_text("\n".join(original + [positive_block]))
    with pytest.raises(ValueError, match="duplicate Census block identifiers"):
        load_membership(**inputs)

    path.write_text("\n".join(original[:-2] + original[-1:]))
    with pytest.raises(ValueError, match="integer block counts"):
        load_membership(**inputs)

    fields = positive_block.split("|")
    fields[29] = "91234"
    path.write_text("\n".join(original[:-2] + ["|".join(fields), original[-1]]))
    with pytest.raises(ValueError, match="unknown PLACE Census GEOIDs"):
        load_membership(**inputs)

    fields[29], fields[30] = "00003", "C1"
    path.write_text("\n".join(original[:-2] + ["|".join(fields), original[-1]]))
    with pytest.raises(ValueError, match="PLACE class codes disagree"):
        load_membership(**inputs)

    fields[29], fields[30] = "00002", "C1"
    path.write_text("\n".join(original[:-2] + ["|".join(fields), original[-1]]))
    with pytest.raises(ValueError, match="PLACE population is not conserved"):
        load_membership(**inputs)


def _cli(tmp_path: Path, *, mct_mode: bool = False) -> subprocess.CompletedProcess[str]:
    config = tmp_path / "config.toml"
    graph = "unused.json" if mct_mode else "graph.json"
    municipality = (
        """
[municipality]
mode = "MCT"
graph_json = "graph.json"
output_namespace = "joint"
"""
        if mct_mode
        else ""
    )
    config.write_text(f"""
[project]
root = "."
{municipality}
[inputs]
graph_json = "{graph}"
crosswalk_csv = "crosswalk.csv"
[chains]
files = ["one.jsonl", "two.jsonl"]
ids = ["chain-a", "chain-b"]
[plans]
overturned = "overturned.csv"
remedial = "remedial.csv"
enacted = "enacted.xlsx"
[analysis]
n_districts = 3
max_districts_per_county = 2
max_maps_per_chain = 2
thin_step = 2
[outputs]
root = "results"
[workflow.mct-diagnostics]
census_geography_header = "oh2020.pl.zip"
expected_population = 81
township_lsadc = "44"
ensemble_unit_details = true
""")
    source = Path(__file__).resolve().parents[1] / "src"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "atlas_posthoc_plus.cli",
            "-c",
            str(config),
            "--paper-mode",
            *(("suite", "municipal") if mct_mode else ("run", "mct-diagnostics")),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(source)},
    )


@pytest.mark.parametrize("mct_mode", [False, True], ids=["standalone", "municipal-suite"])
def test_workflow_cli_outputs_are_comparable(inputs, tmp_path, mct_mode):
    result = _cli(tmp_path, mct_mode=mct_mode)
    assert result.returncode == 0, result.stdout + result.stderr
    results = tmp_path / "results"
    if mct_mode:
        manifest = json.loads((results / "suite-municipal-manifest.json").read_text())
        assert [result["workflow"] for result in manifest["results"]] == ["mct-diagnostics"]
    out = results / ("joint" if mct_mode else "mct")
    expected = {
        "mct_traces.json",
        "mct_plan_summary.csv",
        "mct_chain_summary.csv",
        "mct_official_plans.csv",
        "mct_unit_details.csv",
        "mct_geography_classification.csv",
        "mct_geography_classification_summary.csv",
        "mct_diagnostics.pdf",
        "mct_naive_vs_deduplicated.pdf",
        "mct_municipal_corporation_tradeoff.pdf",
        "mct_township_tradeoff.pdf",
        "mct_metadata.json",
    }
    assert expected <= {p.name for p in out.iterdir()}
    plans = pd.read_csv(out / "mct_plan_summary.csv")
    official = pd.read_csv(out / "mct_official_plans.csv")
    assert plans["chain_id"].tolist() == ["chain-a", "chain-b"]
    assert plans["map_index"].tolist() == [0, 0]
    # The official 2021 assignment is identical at block and graph resolution.
    metrics = official.columns[2:]
    assert plans.loc[0, metrics].to_dict() == official.loc[0, metrics].to_dict()
    assert official.loc[1, "population_ousted_from_both"] == 5
    assert official.loc[1, "double_counted_overlap"] == 5
    assert official.loc[1, "naive_ousted_population"] == 28
    assert official.loc[1, "deduplicated_ousted_population"] == 23
    assert official.loc[1, "cut_municipal_corporations"] == 1
    assert official.loc[1, "municipal_corporation_represented_population"] == 52
    assert official.loc[1, "township_represented_population"] == 48
    assert official.loc[1, "municipal_corporation_ousted_share"] == pytest.approx(13 / 52)
    assert official.loc[1, "township_ousted_share"] == pytest.approx(15 / 48)
    assert official["plan"].tolist() == ["2021 overturned", "2022 remedial", "2025 enacted"]
    chain_summary = pd.read_csv(out / "mct_chain_summary.csv")
    assert chain_summary["examined"].tolist() == [2, 2]
    assert chain_summary["noncompliant_maps"].tolist() == [1, 1]
    traces = json.loads((out / "mct_traces.json").read_text())["chains"]
    assert [trace["plans"][0]["chain_id"] for trace in traces] == ["chain-a", "chain-b"]
    details = pd.read_csv(out / "mct_unit_details.csv")
    assert details.groupby("plan_type")["unit_id"].count().to_dict() == {
        "ensemble": 6,
        "official": 9,
    }
    pdf = PdfReader(out / "mct_diagnostics.pdf")
    labels = pdf.pages[0].extract_text()
    assert "Number of cut Municipal Corporations and Townships" in labels
    assert "Deduplicated ousted population" in labels
    assert all(label in labels for label in ("Enacted ('25)", "Overturned ('21)", "Remedial ('22)"))
    assert len(pdf.pages[0].images) == 0
    comparison = PdfReader(out / "mct_naive_vs_deduplicated.pdf")
    assert "Municipal Corporations and Townships" in comparison.pages[0].extract_text()
    assert len(comparison.pages[0].images) == 0
    for category, title in (
        ("municipal_corporation", "Municipal Corporations"),
        ("township", "Townships"),
    ):
        tradeoff = PdfReader(out / f"mct_{category}_tradeoff.pdf").pages[0]
        labels = tradeoff.extract_text()
        assert f"Number of cut {title}" in labels
        assert "Ousted population share" in labels
        assert "Reference plans" in labels and "chain-a" in labels
        assert len(tradeoff.images) == 0


@pytest.mark.parametrize("conflict", [False, True])
def test_workflow_rejects_incomplete_or_conflicting_official_plan(inputs, tmp_path, conflict):
    path = tmp_path / "overturned.csv"
    rows = pd.read_csv(path)
    if conflict:
        rows = pd.concat([rows, rows.iloc[:1].assign(DistrictID=2)])
    else:
        rows = rows.iloc[1:]
    rows.to_csv(path, index=False)
    result = _cli(tmp_path)
    assert result.returncode != 0
    assert ("Conflicting official assignments" if conflict else "missing 1 populated blocks") in (
        result.stderr
    )
    assert not (tmp_path / "results" / "mct" / "mct_plan_summary.csv").exists()


def test_workflow_rejects_incomplete_chain_assignment(inputs, tmp_path):
    path = tmp_path / "one.jsonl"
    lines = path.read_text().splitlines()
    lines[3] = json.dumps({"districting": {"n0": 1}})
    path.write_text("\n".join(lines))
    result = _cli(tmp_path)
    assert result.returncode != 0
    assert "chain-a, line 4: Chain assignment is missing 6 graph nodes" in result.stderr
    assert not (tmp_path / "results" / "mct" / "mct_plan_summary.csv").exists()
