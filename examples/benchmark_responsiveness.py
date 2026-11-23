"""Benchmark the responsiveness stages against a revision before the optimization."""

import argparse
import importlib.util
import json
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from atlas_posthoc_plus.workflows.elections import responsiveness as current

CASES = (
    "legacy-election", "cold-election", "warm-election", "legacy-load", "stream-load",
    "legacy-votes", "stream-votes", "stream-votes-full",
)


def prepare(root, nodes, maps):
    rng = np.random.default_rng(172)
    size = int(np.ceil(np.sqrt(nodes)))
    graph = pd.DataFrame({
        "id": np.arange(nodes)[::-1],
        "NAME": [f"node-{i}" for i in range(nodes)],
        "ORIG_NAME": [f"precinct-{i}" for i in range(nodes)],
        "COUNTY": np.arange(nodes) % 88,
    })
    (root / "graph.json").write_text(json.dumps({"nodes": graph.to_dict("records")}))

    def rectangles(offset):
        return [box(i % size + offset, i // size + offset,
                    i % size + 1 + offset, i // size + 1 + offset) for i in range(nodes)]

    gpd.GeoDataFrame({"NAME": graph["ORIG_NAME"]}, geometry=rectangles(0),
                     crs="EPSG:26917").to_file(root / "target.shp")
    geometry = rectangles(0.1)
    layers = []
    for year, manifest in current.RACE_MANIFEST.items():
        values = {candidates[0]: rng.integers(0, 500, nodes)
                  for _, _, dem, rep in manifest for candidates in (dem, rep)}
        path = root / f"{year}.shp"
        gpd.GeoDataFrame(values, geometry=geometry, crs="EPSG:26917").to_file(path)
        layers.append(f"{year}={path}")

    chain_path = root / "run999.jsonl"
    with chain_path.open("w") as stream:
        stream.write('"Atlas"\n{}\n{}\n')
        for i in range(maps):
            stream.write(json.dumps({"name": f"step{i}", "districting": {
                f"node-{n}": 1 + (n+i) % 15 for n in range(nodes)
            }}) + "\n")

    args = dict(
        graph_json=root / "graph.json", target_precincts=root / "target.shp", target_key_col="NAME",
        graph_target_key_col="ORIG_NAME", area_crs="EPSG:26917", election_layer=layers,
        race_ids=None, election_cache_dir=root / "cache", election_cache_mode="use",
        chain_files=[chain_path], atlas_key_col="NAME", node_id_col="id", county_col="COUNTY",
        max_districts_per_county=2, max_maps_per_chain=maps, no_compliance_filter=True,
        n_districts=15,
    )
    (root / "args.json").write_text(json.dumps(args, default=str))


def worker(root, case):
    args = SimpleNamespace(**json.loads((root / "args.json").read_text()))
    spec = importlib.util.spec_from_file_location("baseline", root / "baseline.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    graph = current.read_graph_nodes(args.graph_json)
    progress = current.Progress(0)

    if case == "legacy-election":
        def run():
            target = baseline.load_target_geoms(graph, args.target_precincts, "NAME", "ORIG_NAME", "id")
            votes = pd.DataFrame({"node_id": graph["id"]})
            validations = []
            for year, path in baseline.parse_layers(args.election_layer).items():
                allocated, validation, _ = baseline.allocate_layer(year, path, target, args.area_crs)
                votes = votes.merge(allocated, on="node_id", how="left")
                validations.append(validation)
            return votes.fillna(0.0), pd.concat(validations, ignore_index=True)
    elif case in {"cold-election", "warm-election"}:
        def run():
            return current.preprocess_elections(args, graph, progress)
    elif case == "legacy-load":
        def run():
            return baseline.load_assignments(
                args.chain_files, graph, "NAME", "id", "COUNTY", 2,
                args.max_maps_per_chain, True,
            )
    elif case == "stream-load":
        def run():
            return sum(1 for _ in current.iter_assignments(args, graph, progress, [], []))
    else:
        votes, _, races = current.preprocess_elections(args, graph, progress)
        count = 20_000 if case == "stream-votes-full" else args.max_maps_per_chain
        labels = np.arange(len(graph)) % 15

        def items():
            for i in range(count):
                districts = (labels + i) % 15 + 1
                assignment = (
                    pd.Series(districts, index=graph["id"]).sort_index()
                    if case == "legacy-votes" else districts
                )
                yield {"source": "ensemble", "chain_id": "run999", "map_index": i + 1,
                       "proposal": f"step{i}", "assignment": assignment}

        if case == "legacy-votes":
            retained = list(items())

            def run():
                return baseline.evaluate(retained, [], votes, races)
        else:
            def run():
                return current.evaluate(items(), [], votes, races, progress=progress, n_districts=15)

    started = perf_counter()
    result = run()  # Keep outputs live through the peak-memory measurement.
    elapsed = perf_counter() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mib = rss / (2**20 if sys.platform == "darwin" else 2**10)
    print(f"BENCH {case:19s} {elapsed:8.3f} s   {rss_mib:8.1f} MiB", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="d39e008",
                        help="Git revision containing the old responsiveness module")
    parser.add_argument("--nodes", type=int, default=9_000)
    parser.add_argument("--maps", type=int, default=100)
    parser.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(Path(args.worker[0]), args.worker[1])
        return
    if args.nodes < 15 or args.maps <= 0:
        parser.error("benchmark needs at least 15 nodes and one map")

    repo = Path(__file__).resolve().parents[1]
    module = "src/atlas_posthoc_plus/workflows/elections/responsiveness.py"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "baseline.py").write_bytes(subprocess.check_output(
            ["git", "show", f"{args.baseline}:{module}"], cwd=repo,
        ))
        prepare(root, args.nodes, args.maps)
        for case in CASES:
            process = subprocess.run(
                [sys.executable, __file__, "--worker", str(root), case],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            if process.returncode:
                print(process.stdout)
                process.check_returncode()
            print(process.stdout.splitlines()[-1], flush=True)


if __name__ == "__main__":
    main()
