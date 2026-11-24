"""Export and plot Reock scores from AtlasIO assignments and projected unit geometry."""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Callable, Iterable
from functools import lru_cache
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import numpy as np
import pandas as pd
from atlas_posthoc.io.atlas_reader import parse_step
from pyproj import CRS
from shapely import union_all

from atlas_posthoc_plus.io.atlas_records import iter_map_records
from atlas_posthoc_plus.io.graph import read_graph_frame
from atlas_posthoc_plus.paper import paper_output_path
from atlas_posthoc_plus.workflows.compactness.cut_edges import (
    atlasio_districting_to_series,
    chain_id_from_path,
    normalize_atlas_key,
)
from atlas_posthoc_plus.workflows.compactness.official_polsby import (
    PLAN_FILES,
    dissolve_plan_blocks,
    prepare_blocks,
)
from atlas_posthoc_plus.workflows.compactness.polsby_histogram import (
    STAT_TO_SUMMARY_COL,
    discover_public_plan_files,
    load_official_scores,
    load_public_scores,
    plot_histogram,
)
from atlas_posthoc_plus.workflows.compactness.reock import reock_score

STAT_FUNCTIONS = {"average": np.mean, "minimum": np.min, "maximum": np.max}
DISTRICT_COLUMNS = ("source", "plan_id", "chain_id", "map_index", "proposal", "district", "score")
WriteDistrictRows = Callable[[Iterable[tuple[object, ...]]], None]


def prepare_units(
    path: str | Path,
    graph: pd.DataFrame,
    *,
    unit_key_col: str,
    graph_key_col: str,
    node_id_col: str,
    projected_crs: CRS,
) -> gpd.GeoSeries:
    """Return projected geometries indexed by graph node ID."""
    units = gpd.read_file(path)
    if units.crs is None:
        raise ValueError(f"Reock unit layer has no CRS: {path}")
    if unit_key_col not in units:
        raise ValueError(f"Reock unit layer has no key column {unit_key_col!r}: {path}")
    if graph_key_col not in graph:
        raise ValueError(f"Graph has no Reock geometry key column {graph_key_col!r}.")

    graph_keys = graph[graph_key_col].map(normalize_atlas_key)
    if graph_keys.eq("").any() or graph_keys.duplicated().any():
        raise ValueError(f"Graph geometry keys in {graph_key_col!r} must be nonempty and unique.")

    unit_keys = units[unit_key_col].map(normalize_atlas_key)
    matched = unit_keys.isin(graph_keys)
    geometries = units.geometry.loc[matched].to_crs(projected_crs).rename("geometry")
    geometries.index = unit_keys.loc[matched]
    missing = graph_keys[~graph_keys.isin(geometries.index)]
    if not missing.empty:
        raise ValueError(
            f"Reock unit layer is missing geometries for {len(missing)} graph nodes. "
            f"Example keys: {missing.head().tolist()}"
        )
    if not (geometries.is_valid & geometries.area.gt(0)).all():
        raise ValueError(f"Matched Reock unit geometries must be valid with positive area: {path}")
    if geometries.index.has_duplicates:
        geometries = geometries.to_frame().dissolve(level=0).geometry

    geometries = geometries.loc[graph_keys].copy()
    geometries.index = graph[node_id_col].astype(int)
    return geometries


def score_ensemble(
    args: argparse.Namespace,
    graph: pd.DataFrame,
    geometries: gpd.GeoSeries,
    write_district_rows: WriteDistrictRows,
) -> pd.DataFrame:
    stat = STAT_TO_SUMMARY_COL[args.stat]
    summarize = STAT_FUNCTIONS[stat]

    @lru_cache(maxsize=256)
    def score_district(positions: tuple[int, ...]) -> float:
        # Identical district memberships recur across saved plans.
        return reock_score(union_all(geometries.array.take(positions)))

    rows = []
    chain_ids = args.chain_ids or [chain_id_from_path(path) for path in args.chain_files]
    for path, chain_id in zip(args.chain_files, chain_ids, strict=True):
        print(f"Scoring Reock for {chain_id}: {path}", flush=True)
        sampled = islice(enumerate(iter_map_records(path)), 0, None, args.thin_step)
        for map_index, record in islice(sampled, args.max_maps_per_chain):
            proposal = parse_step(record["name"])
            if proposal is None:
                proposal = map_index * args.output_frequency
            if proposal < args.burnin:
                continue
            districts = atlasio_districting_to_series(
                record["districting"],
                graph,
                key_col=args.atlas_key_col,
                node_id_col=args.node_id_col,
            ).reindex(geometries.index)
            if districts.isna().any() or districts.nunique() != args.expected_districts:
                raise ValueError(
                    f"{chain_id} at proposal {proposal} needs a complete assignment to "
                    f"{args.expected_districts} districts."
                )
            scores = {
                district: score_district(tuple(positions))
                for district, positions in districts.groupby(districts).indices.items()
            }
            write_district_rows(
                ("ensemble", None, chain_id, map_index, proposal, district, score)
                for district, score in scores.items()
            )
            rows.append(
                {
                    "chain_id": chain_id,
                    "map_index": map_index,
                    "proposal": proposal,
                    "stat": stat,
                    "score": summarize(list(scores.values())),
                }
            )

    if not rows:
        raise ValueError("No ensemble Reock scores remain after filtering/burn-in.")
    return pd.DataFrame(rows)


def reference_scores(
    args: argparse.Namespace, projected_crs: CRS, write_district_rows: WriteDistrictRows
) -> tuple[pd.DataFrame, pd.DataFrame]:
    public = load_public_scores(args.public_scores_csv, args.stat, metric_name="Reock")
    official = load_official_scores(args.official_scores_csv, args.stat, metric_name="Reock")
    official_plans = (
        [(label, getattr(args, key)) for label, key in PLAN_FILES if getattr(args, key)]
        if not args.official_scores_csv
        else []
    )
    public_plans = (
        discover_public_plan_files(args.public_plan_dir)
        if args.public_plan_dir and not args.public_scores_csv
        else []
    )
    if not official_plans and not public_plans:
        return public, official
    if not args.block_shapefile:
        raise ValueError("Scoring Reock from block assignments requires --block-shapefile.")

    blocks = prepare_blocks(
        block_shapefile=args.block_shapefile, block_col=args.block_col, projected_crs=projected_crs
    )
    summarize = STAT_FUNCTIONS[STAT_TO_SUMMARY_COL[args.stat]]

    def score_plan(source: str, label: str, path: str | Path) -> float:
        districts, _ = dissolve_plan_blocks(
            blocks=blocks,
            assignment_path=path,
            plan_label=label,
            assignment_block_col=args.assignment_block_col,
            assignment_district_col=args.assignment_district_col,
            expected_districts=args.expected_districts,
        )
        scores = districts.geometry.map(reock_score)
        write_district_rows(
            (source, label, None, None, None, district, score)
            for district, score in zip(districts["__district__"], scores, strict=True)
        )
        return float(summarize(scores))

    if official_plans:
        official = pd.DataFrame(
            {"label": label, "score": score_plan("official", label, path)}
            for label, path in official_plans
        )
    if public_plans:
        public = pd.DataFrame(
            {"plan_id": path.stem, "score": score_plan("public", path.stem, path)}
            for path in public_plans
        )
    return public, official


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-json", required=True)
    parser.add_argument("--chain-files", nargs="+", required=True)
    parser.add_argument("--chain-ids", nargs="+")
    parser.add_argument("--unit-shapefile", required=True)
    parser.add_argument("--unit-key-col", default="NAME")
    parser.add_argument("--graph-geometry-key", default="NAME")
    parser.add_argument("--atlas-key-col", default="NAME")
    parser.add_argument("--node-id-col", default="id")
    parser.add_argument("--projected-crs", default="EPSG:5070")
    parser.add_argument("--expected-districts", type=int, default=15)
    parser.add_argument("--output-frequency", type=int, default=50_000)
    parser.add_argument("--thin-step", type=int, default=1)
    parser.add_argument("--max-maps-per-chain", type=int)
    parser.add_argument("--public-scores-csv")
    parser.add_argument("--public-plan-dir")
    parser.add_argument("--official-scores-csv")
    parser.add_argument("--block-shapefile")
    parser.add_argument("--block-col")
    parser.add_argument("--assignment-block-col")
    parser.add_argument("--assignment-district-col")
    parser.add_argument("--enacted-baf")
    parser.add_argument("--overturned-baf")
    parser.add_argument("--remedial-baf")
    parser.add_argument("--stat", choices=STAT_TO_SUMMARY_COL, default="average")
    parser.add_argument("--burnin", type=int, default=0, help="First proposal to include.")
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--density", action="store_true")
    parser.add_argument("--title")
    parser.add_argument("--ensemble-label", default="Ensemble")
    parser.add_argument("--public-label", default="Public submitted plans")
    parser.add_argument("--official-label", default="Official plans")
    parser.add_argument("--out", default="outputs/reock/reock-histogram.pdf")
    args = parser.parse_args()
    if args.chain_ids is not None and len(args.chain_ids) != len(args.chain_files):
        parser.error("--chain-ids must have the same length as --chain-files")
    if args.thin_step <= 0 or args.output_frequency <= 0:
        parser.error("--thin-step and --output-frequency must be positive")
    if args.max_maps_per_chain is not None and args.max_maps_per_chain <= 0:
        parser.error("--max-maps-per-chain must be positive")
    return args


def publish_outputs(staged: Path, output_dir: Path) -> None:
    # Serialize replacement of the bundle when runs share an output directory.
    paths = sorted(staged.iterdir())
    destinations = [output_dir / path.name for path in paths]
    lock_fd = None
    try:
        if os.name == "posix":
            import fcntl

            lock_fd = os.open(output_dir, os.O_RDONLY)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        for destination in destinations:
            if destination.is_dir():
                raise IsADirectoryError(f"Reock output path is a directory: {destination}")
        for path, destination in zip(paths, destinations, strict=True):
            path.replace(destination)
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
    for destination in destinations:
        print(f"Wrote {destination}")


def main() -> None:
    args = parse_args()
    projected_crs = CRS.from_user_input(args.projected_crs)
    if not projected_crs.is_projected:
        raise ValueError("Reock requires a projected CRS for area and distance.")
    graph = read_graph_frame(args.graph_json, node_id_col=args.node_id_col)
    geometries = prepare_units(
        args.unit_shapefile,
        graph,
        unit_key_col=args.unit_key_col,
        graph_key_col=args.graph_geometry_key,
        node_id_col=args.node_id_col,
        projected_crs=projected_crs,
    )
    figure = paper_output_path(args.out)
    output_dir = figure.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stat = STAT_TO_SUMMARY_COL[args.stat]
    with TemporaryDirectory(dir=output_dir) as temp_dir:
        staged = Path(temp_dir)
        with (staged / "district-scores.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(DISTRICT_COLUMNS)
            ensemble = score_ensemble(args, graph, geometries, writer.writerows)
            public, official = reference_scores(args, projected_crs, writer.writerows)

        ensemble.to_csv(staged / "ensemble-scores.csv", index=False)
        official.assign(stat=stat).to_csv(staged / "official-summary.csv", index=False)
        pdf = output_dir / "reock-histogram.pdf"
        plot_histogram(
            ensemble,
            public,
            official,
            pdf,
            stat=args.stat,
            bins=args.bins,
            density=args.density,
            title=args.title or f"{stat.capitalize()} district Reock distribution",
            ensemble_label=args.ensemble_label,
            public_label=args.public_label,
            official_label=args.official_label,
            metric_name="Reock",
            additional_outpath=figure if figure != pdf else None,
            staging_dir=staged,
        )
        publish_outputs(staged, output_dir)


if __name__ == "__main__":
    main()
