import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from atlas_posthoc_plus.io import chain_sweep
from atlas_posthoc_plus.workflows.elections import responsiveness as resp
from atlas_posthoc_plus.workflows.elections.responsiveness_cache import input_signatures


@pytest.fixture
def inputs(tmp_path):
    # IDs are in reverse order and both kinds of precinct keys are different.
    nodes = [
        {"id": 99 - i, "NAME": f"node-{i}", "ORIG_NAME": f"precinct-{i}",
         "COUNTY": "split" if i < 3 else f"county-{i // 2}"}
        for i in range(30)
    ]
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps({"nodes": nodes, "adjacency": [[] for _ in nodes]}))
    target_path = tmp_path / "target.shp"
    gpd.GeoDataFrame(
        {"NAME": [node["ORIG_NAME"] for node in nodes]},
        geometry=[box(i, 0, i + 1, 1) for i in range(30)], crs="EPSG:26917",
    ).to_file(target_path)
    election_path = tmp_path / "election.shp"
    gpd.GeoDataFrame(
        {"G24PREDHAR": [13 + 8 * i for i in range(15)], "G24PRERTRU": [69] * 15,
         "G24USSDBRO": [141 - 6 * i for i in range(15)], "G24USSRMOR": [89] * 15},
        geometry=[box(2 * i, 0, 2 * i + 2, 1) for i in range(15)], crs="EPSG:26917",
    ).to_file(election_path)
    chain_path = tmp_path / "run999.jsonl"
    assignments = [
        [i // 2 + 1 for i in range(30)],
        [i % 15 + 1 for i in range(30)],
        [15 - i // 2 for i in range(30)],
    ]
    raw = [
        json.dumps({"name": f"fixture{i}", "districting": dict(zip(keys, labels))})
        for i, (keys, labels) in enumerate(zip(
            ([json.dumps([node["NAME"]]) for node in nodes],
             [node["NAME"] for node in nodes], [str(node["id"]) for node in nodes]),
            assignments,
        ))
    ]
    chain_path.write_text('"Atlas"\n{}\n{}\n' + "\n".join(
        [raw[0], "{bad json", raw[1], '{"districting":{"node-0":1}}', raw[2]]
    ) + "\n")

    crosswalk = tmp_path / "crosswalk.csv"
    block_ids = [f"39000000000{i:04}" for i in range(30)]
    pd.DataFrame({"BLOCK": block_ids, "NAME": [node["ORIG_NAME"] for node in nodes],
                  "POP20": [100] * 30}).to_csv(crosswalk, index=False)
    baf_paths = []
    for i in range(3):
        path = tmp_path / f"plan-{i}.csv"
        pd.DataFrame({"BLOCK": block_ids, "DISTRICT": assignments[i]}).to_csv(path, index=False)
        baf_paths.append(path)
    args = SimpleNamespace(
        graph_json=graph_path, target_precincts=target_path, target_key_col="NAME",
        graph_target_key_col="ORIG_NAME", area_crs="EPSG:26917", node_id_col="id",
        election_layer=[f"2024={election_path}"], race_ids=["G24PRE", "G24USS"],
        election_cache_mode="use", election_cache_dir=tmp_path / "cache",
        chain_files=[chain_path], atlas_key_col="NAME", county_col="COUNTY",
        max_districts_per_county=2, max_maps_per_chain=4, no_compliance_filter=False,
        n_districts=15,
    )
    return args, assignments, crosswalk, baf_paths


def legacy_totals(votes, assignment, cols):
    """The original per-race merge and compensated sum."""
    return votes[["node_id", *cols]].merge(
        assignment.rename("DISTRICT").rename_axis("node_id").reset_index(),
        on="node_id", how="inner",
    ).groupby("DISTRICT")[cols].sum()


def test_allocation_voting_and_compliance_match_legacy(inputs, capsys):
    args, assignments, crosswalk, baf_paths = inputs
    graph = resp.read_graph_nodes(args.graph_json)
    progress = resp.Progress(0)
    votes, validation, races = resp.preprocess_elections(args, graph, progress)
    assert votes["node_id"].tolist() == graph["id"].tolist()
    expected = np.column_stack([
        np.repeat((13 + 8 * np.arange(15)) / 2, 2), np.repeat(69 / 2, 30),
        np.repeat((141 - 6 * np.arange(15)) / 2, 2), np.repeat(89 / 2, 30),
    ])
    np.testing.assert_array_equal(votes.iloc[:, 1:].to_numpy(), expected)
    dem = expected[:, ::2].sum(axis=0)
    rep = expected[:, 1::2].sum(axis=0)
    np.testing.assert_array_equal(validation["dem_source_total"], dem)
    np.testing.assert_array_equal(validation["rep_source_total"], rep)
    np.testing.assert_array_equal(validation["dem_allocated_total"], dem)
    np.testing.assert_array_equal(validation["rep_allocated_total"], rep)
    np.testing.assert_array_equal(validation["dem_allocation_ratio"], [1, 1])
    np.testing.assert_array_equal(validation["rep_allocation_ratio"], [1, 1])
    np.testing.assert_array_equal(validation["statewide_dem_share_source"], dem / (dem + rep))
    np.testing.assert_array_equal(validation["statewide_dem_share_allocated"], dem / (dem + rep))

    summaries, failures = [], []
    kept = list(resp.iter_assignments(args, graph, progress, summaries, failures))
    assert [item["map_index"] for item in kept] == [1, 4]
    assert summaries[0]["maps_seen"] == 5
    assert summaries[0]["maps_retained"] == 2
    assert summaries[0]["invalid_maps"] == 2
    assert summaries[0]["parse_failures"] == 1
    assert summaries[0]["compliance_exclusions"] == 1
    assert summaries[0]["worst_single_county_split"] == 3
    reference_splits = [
        graph.assign(district=values).groupby("COUNTY")["district"].nunique()
        for values in assignments
    ]
    assert [int((sizes > 2).sum()) for sizes in reference_splits] == [0, 1, 0]
    assert summaries[0]["avg_double_crosses"] == pytest.approx(1 / 3)
    assert [item["stage"] for item in failures] == ["assignment", "parse"]
    assert failures[1]["line_number"] == 5
    assert "parser excluded line 5" in capsys.readouterr().out

    plans = resp.load_officials(
        graph, resp.read_crosswalk(crosswalk), *baf_paths, "ORIG_NAME", "id", None, None,
        progress=progress,
    )
    for plan, expected_assignment in zip(plans, assignments):
        assert plan["assignment"].reindex(graph["id"]).tolist() == expected_assignment
    columns = list(votes.columns[1:])
    totals = resp.DistrictVoteTotals(votes, columns)
    ensemble, official, summary = resp.evaluate(
        kept, plans, votes, races, progress=progress, n_districts=15,
    )
    for i, race in enumerate(races):
        cols = columns[2 * i:2 * i + 2]
        race_id = race["race_id"]
        expected_ensemble, expected_official = [], []
        for item in kept + plans:
            assignment = item["assignment"]
            if isinstance(assignment, np.ndarray):
                assignment = pd.Series(assignment, index=graph["id"])
            legacy = legacy_totals(votes, assignment, cols)
            actual = totals.for_assignment(item["assignment"])[cols]
            np.testing.assert_array_equal(actual.index, legacy.index)
            np.testing.assert_array_equal(actual.to_numpy(), legacy.to_numpy())
            wins = int((legacy[cols[0]] > legacy[cols[1]]).sum())
            target = expected_ensemble if item["source"] == "ensemble" else expected_official
            target.append(wins)
        assert ensemble.loc[ensemble["race_id"] == race_id, "dem_seats"].tolist() == expected_ensemble
        assert official.loc[official["race_id"] == race_id, "dem_seats"].tolist() == expected_official
        assert summary.loc[i, "statewide_dem_share"] == dem[i] / (dem[i] + rep[i])
        assert ensemble.loc[ensemble["race_id"] == race_id, "dem_seats"].mean() == np.mean(expected_ensemble)


def test_fractional_near_ties_and_partial_reference_match_legacy():
    # Include a tie at a boundary where ordinary sequential addition differs.
    votes = pd.DataFrame({"node_id": [30, 10, 9, 1, 2, 4, 7, 8],
                          "dem": [1e16, 1.0, 1.0, 0.21, 0.30, 0.04, 0.75, 0.2],
                          "rep": [1e16, 0.0, 2.0, 0.17, 0.23, 0.14, 0.59, 0.9]})
    columns = ["dem", "rep"]
    assignment = pd.Series([5, 1, 5, 1, 5, 7, 7], index=[9, 1, 30, 2, 10, 4, 7])
    expected = legacy_totals(votes, assignment, columns)
    actual = resp.DistrictVoteTotals(votes, columns).for_assignment(assignment)
    np.testing.assert_array_equal(actual.to_numpy(), expected.to_numpy())


def test_validated_cache_reuse_rebuild_bypass_and_race_errors(inputs, monkeypatch, capsys):
    args, *_ = inputs
    graph = resp.read_graph_nodes(args.graph_json)
    progress = resp.Progress(0)
    original = resp.preprocess_elections(args, graph, progress)
    first_cache, = args.election_cache_dir.glob("*.npz")
    alloc = resp.allocate_layer

    def forbidden(*_args, **_kwargs):
        pytest.fail("Reused cache recomputed the overlay")

    monkeypatch.setattr(resp, "allocate_layer", forbidden)
    hit = resp.preprocess_elections(args, graph, progress)
    pd.testing.assert_frame_equal(original[0], hit[0], check_exact=True)
    pd.testing.assert_frame_equal(original[1], hit[1], check_exact=True)
    assert original[2] == hit[2]
    assert "Election cache hit" in capsys.readouterr().out
    monkeypatch.setattr(resp, "allocate_layer", alloc)

    first_cache.write_bytes(b"corrupt NPZ")
    rebuilt = resp.preprocess_elections(args, graph, progress)
    pd.testing.assert_frame_equal(original[0], rebuilt[0], check_exact=True)
    assert "rejecting election cache" in capsys.readouterr().out
    args.election_cache_mode = "rebuild"
    resp.preprocess_elections(args, graph, progress)
    assert "Election cache rebuild" in capsys.readouterr().out

    args.election_cache_mode = "off"
    args.race_ids = ["G24PRE"]
    assert len(resp.preprocess_elections(args, graph, progress)[2]) == 1
    assert list(args.election_cache_dir.glob("*.npz")) == [first_cache]
    args.race_ids = ["G24JUS1"]
    with pytest.raises(ValueError, match="missing election columns"):
        resp.preprocess_elections(args, graph, progress)
    args.race_ids = ["G20PRE"]
    with pytest.raises(ValueError, match="Race IDs unavailable: G20PRE"):
        resp.preprocess_elections(args, graph, progress)


def test_shapefile_fingerprint_and_input_invalidation(inputs):
    args, *_ = inputs
    graph = resp.read_graph_nodes(args.graph_json)
    progress = resp.Progress(0)
    resp.preprocess_elections(args, graph, progress)
    path = Path(args.election_layer[0].split("=", 1)[1])
    assert any(item["path"].endswith(".dbf") for item in input_signatures(path))
    dbf = path.with_suffix(".dbf")
    stat = dbf.stat()
    updated = dbf.read_bytes().replace(b" 13", b" 14", 1)
    assert updated != dbf.read_bytes()
    dbf.write_bytes(updated)
    os.utime(dbf, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    _, validation, _ = resp.preprocess_elections(args, graph, progress)
    assert validation["dem_source_total"].iloc[0] == sum(13 + 8 * i for i in range(15)) + 1
    assert len(list(args.election_cache_dir.glob("*.npz"))) == 2


def test_concurrent_election_cache_has_one_writer(inputs):
    args, *_ = inputs
    values_path = args.graph_json.parent / "args.json"
    values_path.write_text(json.dumps(vars(args), default=str))
    program = """
import json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
from atlas_posthoc_plus.workflows.elections import responsiveness as r
settings = Path(sys.argv[1])
args = SimpleNamespace(**json.loads(settings.read_text()))
(settings.parent / f"ready-{os.getpid()}").touch()
deadline = time.monotonic() + 15
while len(list(settings.parent.glob("ready-*"))) < 2:
    if time.monotonic() > deadline:
        raise TimeoutError("Second cache process failed to start")
    time.sleep(0.05)
original = r.allocate_layer
def delayed(*a, **kw):
    time.sleep(1.5)
    return original(*a, **kw)
r.allocate_layer = delayed
votes, _, _ = r.preprocess_elections(args, r.read_graph_nodes(args.graph_json), r.Progress(0))
print('totals', votes.iloc[:, 1:].sum().tolist(), flush=True)
"""
    command = [sys.executable, "-c", program, str(values_path)]
    jobs = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True) for _ in range(2)]
    logs = [job.communicate(timeout=30)[0] for job in jobs]
    assert [job.returncode for job in jobs] == [0, 0], logs
    assert sum("Election cache miss" in log for log in logs) == 1
    assert sum("Election cache hit" in log for log in logs) == 1
    assert sum("Waiting for shared election cache lock" in log for log in logs) == 1
    assert logs[0].splitlines()[-1] == logs[1].splitlines()[-1]
    assert len(list(args.election_cache_dir.iterdir())) == 2


def test_valid_suite_sweep_is_reused_and_short_sweep_is_not_trusted(inputs, monkeypatch):
    args, assignments, *_ = inputs
    chain = args.chain_files[0]
    chain.write_text('"Atlas"\n{}\n{}\n' + "\n".join(
        json.dumps({"name": f"map{i}", "districting": {
            f"node-{n}": district for n, district in enumerate(labels)
        }}) for i, labels in enumerate(assignments)
    ) + "\n")
    cache = chain.parent / "sweep"
    chain_sweep.materialize_chain_sweep(chain, args.graph_json, cache, max_records=3)
    monkeypatch.setenv(chain_sweep.SWEEP_DIR_ENV, str(cache))
    monkeypatch.setenv(chain_sweep.SWEEP_LIMIT_ENV, "3")
    args.max_maps_per_chain = 3
    graph = resp.read_graph_nodes(args.graph_json)
    progress = resp.Progress(0)
    original = resp.iter_map_records

    def forbidden(*_args, **_kwargs):
        pytest.fail("A valid sweep was reparsed")

    monkeypatch.setattr(resp, "iter_map_records", forbidden)
    summaries = []
    result = list(resp.iter_assignments(args, graph, progress, summaries, []))
    assert [item["map_index"] for item in result] == [1, 3]
    assert summaries[0]["invalid_maps"] == 0
    monkeypatch.setattr(resp, "iter_map_records", original)
    monkeypatch.setenv(chain_sweep.SWEEP_LIMIT_ENV, "1")
    result = list(resp.iter_assignments(args, graph, progress, [], []))
    assert [item["map_index"] for item in result] == [1, 3]


def test_uncovered_area_is_reported_without_renormalizing(inputs, capsys):
    args, *_ = inputs
    target = gpd.read_file(args.target_precincts)
    target.loc[29, "geometry"] = box(100, 0, 101, 1)
    target.to_file(args.target_precincts)
    graph = resp.read_graph_nodes(args.graph_json)
    votes, validation, _ = resp.preprocess_elections(args, graph, resp.Progress(0))
    np.testing.assert_array_equal(votes.iloc[-1, 1:].to_numpy(), np.zeros(4))
    row = validation.iloc[0]
    assert row.dem_source_total - row.dem_allocated_total == (13 + 8 * 14) / 2
    assert row.rep_source_total - row.rep_allocated_total == 69 / 2
    assert "WARNING: G24PRE allocation" in capsys.readouterr().out


def test_cli_selects_races_and_replots_existing_csvs(inputs):
    args, _, crosswalk, baf_paths = inputs
    root = args.graph_json.parent
    config = root / "fixture.toml"
    _, election_path = args.election_layer[0].split("=", 1)
    config.write_text(f"""
[project]
root = {json.dumps(str(root))}
[analysis]
n_districts = 15
max_maps_per_chain = 4
[chains]
files = [{json.dumps(str(args.chain_files[0]))}]
[inputs]
graph_json = {json.dumps(str(args.graph_json))}
target_precincts = {json.dumps(str(args.target_precincts))}
crosswalk_csv = {json.dumps(str(crosswalk))}
[elections.layers]
2024 = {json.dumps(election_path)}
[plans]
enacted = {json.dumps(str(baf_paths[0]))}
overturned = {json.dumps(str(baf_paths[1]))}
remedial = {json.dumps(str(baf_paths[2]))}
[outputs]
root = "out"
responsiveness_election_cache = "cache"
[workflow.responsiveness]
race_ids = ["G24PRE", "G24USS"]
[workflow.responsiveness-zoom]
race_ids = ["G24PRE"]
""")
    cli = [sys.executable, "-m", "atlas_posthoc_plus.cli",
           "--config", str(config), "--paper-mode", "run"]
    result = subprocess.run([*cli, "responsiveness"], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    out = root / "out" / "responsiveness"
    summary_path = out / "oh_responsiveness_race_summary.csv"
    assert pd.read_csv(summary_path)["race_id"].tolist() == ["G24PRE", "G24USS"]
    assert (out / "ohio_responsiveness_all_statewide_races.pdf").stat().st_size > 0
    official = pd.read_csv(out / "oh_responsiveness_official_plan_seats.csv")
    assert set(official["source"]) == set(resp.PLAN_STYLES)
    assert len(pd.read_csv(out / "oh_responsiveness_excluded_maps.csv")) == 2

    result = subprocess.run(
        [*cli, "responsiveness-zoom", "--", "--race-ids", "G24USS",
         "--output", str(out / "selected.png")], text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Loaded zoom inputs for 1 races" in result.stdout
    assert (out / "selected.pdf").stat().st_size > 0
    assert pd.read_csv(summary_path)["race_id"].tolist() == ["G24PRE", "G24USS"]


def test_cache_normalizes_numeric_string_node_ids(inputs):
    args, *_ = inputs
    source = json.loads(args.graph_json.read_text())
    for node in source["nodes"]:
        node["alternate_id"] = str(node["id"] + 1000)
    args.graph_json.write_text(json.dumps(source))
    args.node_id_col = "alternate_id"
    graph = resp.read_graph_nodes(args.graph_json)
    cold, _, _ = resp.preprocess_elections(args, graph, resp.Progress(0))
    warm, _, _ = resp.preprocess_elections(args, graph, resp.Progress(0))
    assert cold["node_id"].tolist() == [node["id"] + 1000 for node in source["nodes"]]
    pd.testing.assert_frame_equal(cold, warm, check_exact=True)
