#!/usr/bin/env python3

"""
atlas-posthoc-plus run responsiveness \
  --graph-json data/graph/OHpct20.json \
  --target-precincts data/derived/pctSF_OH/pctSF_OH.shp \
  --target-key-col NAME \
  --graph-target-key-col ORIG_NAME \
  --chain-files \
    outputs/chains/soft_constraint/congressional/run37_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run38_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run39_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
    outputs/chains/soft_constraint/congressional/run40_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl \
  --election-layer 2016=data/raw/oh_vest_16/oh_vest_16.shp \
  --election-layer 2018=data/raw/oh_vest_18/oh_vest_18.shp \
  --election-layer 2020=data/raw/oh_vest_20/oh_vest_20.shp \
  --election-layer 2022=data/raw/oh_gen_2022_prec/oh_2022_gen_prec_st.shp \
  --election-layer 2024=data/raw/oh_2024_gen_prec/oh_2024_gen_all_prec/oh_2024_gen_all_prec.shp \
  --crosswalk-csv outputs/oh_block_to_precinct_crosswalk.csv \
  --enacted-baf "inputs/plans/October 31 2025 CD BAF.xlsx" \
  --overturned-baf "inputs/plans/SubSB258-Block-Assignment-File.xlsx" \
  --remedial-baf "inputs/plans/March 2 2022 CD BAF.xlsx" \
  --output-dir outputs/ohio_responsiveness_all_races \
  --max-maps-per-chain 1001
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from itertools import islice
from pathlib import Path
from time import perf_counter
from typing import Iterable

from atlas_posthoc_plus import __version__
from atlas_posthoc_plus.paper import figure_legend, official_plan_styles, save_figure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from atlas_posthoc_plus.io.atlas_records import ParseSummary, iter_map_records
from atlas_posthoc_plus.io.chain_sweep import (
    DenseDistricting,
    audited_sweep_size,
    county_split_sizes,
    configured_sweep,
    iter_sweep_records,
)

from atlas_posthoc_plus.workflows.elections.responsiveness_cache import (
    cached_allocation,
    input_signatures,
)
from atlas_posthoc_plus.workflows.elections.responsiveness_progress import Progress

PLAN_STYLES = {
    "Enacted ('25)": {"color": "#9467bd", "marker": "o"},
    "Overturned ('21)": {"color": "#2ca02c", "marker": "o"},
    "Remedial ('22)": {"color": "#d62728", "marker": "o"},
}
PUBLIC_SOURCE = "Public submitted plans"

RACE_MANIFEST = {
    2016: [
        ("G16PRE", "2016 President", ["G16PREDCLI"], ["G16PRERTRU"]),
        ("G16USS", "2016 U.S. Senate", ["G16USSDSTR"], ["G16USSRPOR"]),
    ],
    2018: [
        ("G18USS", "2018 U.S. Senate", ["G18USSDBRO"], ["G18USSRREN"]),
        ("G18GOV", "2018 Governor", ["G18GOVDCOR"], ["G18GOVRDEW"]),
        ("G18ATG", "2018 Attorney General", ["G18ATGDDET"], ["G18ATGRYOS"]),
        ("G18AUD", "2018 Auditor", ["G18AUDDSPA"], ["G18AUDRFAB"]),
        ("G18SOS", "2018 Secretary of State", ["G18SOSDCLY"], ["G18SOSRLAR"]),
        ("G18TRE", "2018 Treasurer", ["G18TREDRIC"], ["G18TRERSPR"]),
    ],
    2020: [
        ("G20PRE", "2020 President", ["G20PREDBID"], ["G20PRERTRU", "GG20PRERTRU"]),
    ],
    2022: [
        ("G22ATG", "2022 Attorney General", ["G22ATGDCRO"], ["G22ATGRYOS"]),
        ("G22AUD", "2022 Auditor", ["G22AUDDSAP"], ["G22AUDRFAB"]),
        ("G22CJU", "2022 Chief Justice", ["G22CJUDBRU"], ["G22CJURKEN"]),
        ("G22GOV", "2022 Governor", ["G22GOVDWHA"], ["G22GOVRDEW"]),
        ("G22JUS1", "2022 Supreme Court 1/1", ["G22JUSDJAM"], ["G22JUSRFIS"]),
        ("G22JUS2", "2022 Supreme Court 1/2", ["G22JUSDZAY"], ["G22JUSRDEW"]),
        ("G22SOS", "2022 Secretary of State", ["G22SOSDCLA"], ["G22SOSRLAR"]),
        ("G22TRE", "2022 Treasurer", ["G22TREDSCH"], ["G22TRERSPR"]),
        ("G22USS", "2022 U.S. Senate", ["G22USSDRYA"], ["G22USSRVAN"]),
    ],
    2024: [
        ("G24PRE", "2024 President", ["G24PREDHAR"], ["G24PRERTRU"]),
        ("G24USS", "2024 U.S. Senate", ["G24USSDBRO"], ["G24USSRMOR"]),
        ("G24JUS1", "2024 Supreme Court 1/1", ["G24JUSDDON"], ["G24JUSRSHA"]),
        ("G24JUS2", "2024 Supreme Court 1/2", ["G24JUSDSTE"], ["G24JUSRDET"]),
        ("G24JUSU", "2024 Supreme Court unexpired", ["G24JUSDFOR"], ["G24JUSRHAW"]),
    ],
}

def canon(x): return re.sub(r"[^a-z0-9]", "", str(x).lower())

def norm(x):
    if pd.isna(x): return ""
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") else s

def norm_block(x):
    s = norm(x)
    return s.zfill(15) if s else ""

def norm_atlas_key(x):
    if isinstance(x, (list, tuple)):
        return norm(x[0]) if len(x) == 1 else "|".join(norm(v) for v in x)
    s = norm(x)
    try:
        obj = json.loads(s)
    except Exception:
        return s
    if isinstance(obj, list):
        return norm(obj[0]) if len(obj) == 1 else "|".join(norm(v) for v in obj)
    if isinstance(obj, str): return norm(obj)
    return s

def resolve(columns: Iterable[str], candidates: Iterable[str | None], label: str) -> str:
    columns = list(columns)
    d = {canon(c): c for c in columns}
    for cand in candidates:
        if cand is None: continue
        if canon(cand) in d: return d[canon(cand)]
    for cand in candidates:
        if cand is None: continue
        key = canon(cand)
        for k, v in d.items():
            if k.startswith(key): return v
    raise ValueError(f"Could not resolve {label} from {list(candidates)}; available={columns}")

def pick(columns, candidates):
    d = {canon(c): c for c in columns}
    for cand in candidates:
        if canon(cand) in d: return d[canon(cand)]
    return None

def dist_label(x):
    if x is None: raise ValueError("district label is None")
    if isinstance(x, (int, np.integer)): return int(x)
    if isinstance(x, float):
        if np.isfinite(x) and abs(x - round(x)) < 1e-9: return int(round(x))
        raise ValueError(f"bad district label {x}")
    s = norm(x)
    if s.lstrip("-").isdigit(): return int(s)
    m = re.search(r"\d+", s)
    if m: return int(m.group())
    raise ValueError(f"cannot convert district label {x!r}")

def read_vector(path):
    import geopandas as gpd
    return gpd.read_file(path)

def read_graph_nodes(path):
    content = Path(path).read_bytes()
    obj = json.loads(content)
    nodes = obj.get("nodes") if isinstance(obj.get("nodes"), list) else obj.get("node_attributes")
    if not isinstance(nodes, list): raise ValueError("graph JSON needs nodes or node_attributes")
    df = pd.DataFrame(nodes).copy()
    for c in ["id", "NAME", "ORIG_NAME", "COUNTY"]:
        if c not in df.columns: raise ValueError(f"graph missing {c}; columns={list(df.columns)}")
    df["id"] = pd.to_numeric(df["id"], errors="raise").astype(int)
    for c in ["NAME", "ORIG_NAME", "COUNTY"]: df[c] = df[c].map(norm)
    df.attrs["source_sha256"] = hashlib.sha256(content).hexdigest()
    return df

def load_target_geoms(graph_nodes, target_precincts, target_key_col, graph_target_key_col, node_id_col):
    gdf = read_vector(target_precincts)
    if target_key_col not in gdf.columns: raise ValueError(f"{target_key_col} missing in target layer")
    if graph_target_key_col not in graph_nodes.columns: raise ValueError(f"{graph_target_key_col} missing in graph")
    gdf = gdf[[target_key_col, "geometry"]].copy()
    gdf[target_key_col] = gdf[target_key_col].map(norm)
    lookup = graph_nodes[[node_id_col, graph_target_key_col]].copy()
    lookup[graph_target_key_col] = lookup[graph_target_key_col].map(norm)
    gdf = gdf.merge(lookup, left_on=target_key_col, right_on=graph_target_key_col, how="inner")
    n_graph, n_match = graph_nodes[node_id_col].nunique(), gdf[node_id_col].nunique()
    print(f"Target geometry matched {n_match:,} / {n_graph:,} graph nodes", flush=True)
    if n_match < 0.95 * n_graph: print("WARNING: target geometry match below 95%", flush=True)
    gdf = gdf.rename(columns={node_id_col: "node_id"})
    gdf["node_id"] = gdf["node_id"].astype(int)
    return gdf[["node_id", "geometry"]].copy()

def parse_layers(items):
    out = {}
    for item in items:
        y, p = item.split("=", 1)
        out[int(y)] = Path(p)
    return out

def requested_races(race_ids, available):
    """Validate an explicit selection and return its IDs as a set."""
    if not race_ids:
        return set(available)
    requested = set(race_ids)
    missing = requested - set(available)
    if missing:
        raise ValueError(
            f"Race IDs unavailable: {', '.join(sorted(missing))}. "
            f"Available: {', '.join(sorted(available))}"
        )
    return requested


def infer_races(year, columns, race_ids=None, *, progress):
    races = []
    for race_id, label, dem_cands, rep_cands in RACE_MANIFEST[year]:
        if race_ids and race_id not in race_ids:
            continue
        dc, rc = pick(columns, dem_cands), pick(columns, rep_cands)
        if dc is None or rc is None:
            message = f"{race_id} ({label}): dem {dem_cands}->{dc}, rep {rep_cands}->{rc}"
            if race_ids:
                raise ValueError(f"Requested race is missing election columns: {message}")
            progress.log(f"WARNING: skipping unavailable manifest race {message}")
            continue
        races.append({
            "year": year, "race_id": race_id, "label": label,
            "dem_col_source": dc, "rep_col_source": rc,
            "dem_col_graph": f"{race_id}DEM", "rep_col_graph": f"{race_id}REP",
        })
    return races


def area_geometry(frame, area_crs):
    if frame.crs is None:
        raise ValueError("Vector layer has no CRS")
    result = frame.to_crs(area_crs)
    result["geometry"] = result.geometry.buffer(0)
    return result


def spatial_crosswalk(source, projected_target, area_crs, *, progress, label):
    import geopandas as gpd

    with progress.stage(f"{label} source geometry projection and repair"):
        src = area_geometry(source[["geometry"]].reset_index(drop=True), area_crs)
        src["_source_id"] = np.arange(len(src), dtype=int)
        src["_src_area"] = src.geometry.area
        src = src[src["_src_area"] > 0].copy()
    with progress.stage(
        f"{label} spatial intersection ({len(src):,} source × {len(projected_target):,} target geometries)"
    ):
        inter = gpd.overlay(src, projected_target, how="intersection", keep_geom_type=False)
        if inter.empty:
            raise ValueError("spatial overlay produced no intersections")
        inter["_inter_area"] = inter.geometry.area
        inter = inter[inter["_inter_area"] > 0].copy()
        if inter.empty:
            raise ValueError("spatial overlay produced no positive-area intersections")
        inter["weight"] = inter["_inter_area"] / inter["_src_area"]
        progress.log(f"{label}: {len(inter):,} positive-area intersections")
    return inter[["_source_id", "node_id", "weight"]]


def allocation_metrics(sd, sr, ad, ar):
    return {
        "dem_source_total": sd,
        "rep_source_total": sr,
        "dem_allocated_total": ad,
        "rep_allocated_total": ar,
        "dem_allocation_ratio": ad / sd if sd else np.nan,
        "rep_allocation_ratio": ar / sr if sr else np.nan,
        "statewide_dem_share_source": sd / (sd + sr) if sd + sr > 0 else np.nan,
        "statewide_dem_share_allocated": ad / (ad + ar) if ad + ar > 0 else np.nan,
    }


def allocate_layer(year, election_path, target_geoms, area_crs, race_ids=None, *, progress, label):
    with progress.stage(f"{label} reading election layer {election_path}"):
        src = read_vector(election_path)
        races = infer_races(year, src.columns, race_ids, progress=progress)
        if not races:
            progress.log(f"WARNING: election layer {year} has no usable statewide races")
            return pd.DataFrame(), pd.DataFrame(), []
        needed = sorted({r[key] for r in races for key in ("dem_col_source", "rep_col_source")})
        for column in needed:
            numeric = pd.to_numeric(src[column], errors="coerce")
            invalid = int((numeric.isna() & src[column].notna()).sum())
            if invalid:
                progress.log(f"WARNING: {label} {column} contains {invalid:,} nonnumeric "
                             "vote values treated as zero")
            src[column] = numeric.fillna(0.0)
        progress.log(f"{label}: {len(src):,} source rows, races: "
                     + ", ".join(r["race_id"] for r in races))
    cw = spatial_crosswalk(src, target_geoms, area_crs, progress=progress, label=label)
    with progress.stage(f"{label} allocating node votes and validating {len(races)} races"):
        values = src[needed].copy()
        values["_source_id"] = np.arange(len(values), dtype=int)
        overlaps = cw.merge(values, on="_source_id", how="left")
        # Use the same products and groupby sums as allocating each D/R pair.
        weighted = {r[graph_col]: overlaps[r[source_col]] * overlaps["weight"]
                    for r in races for graph_col, source_col in (
                        ("dem_col_graph", "dem_col_source"), ("rep_col_graph", "rep_col_source")
                    )}
        allocated = pd.DataFrame(weighted).groupby(overlaps["node_id"]).sum()
        validation = []
        for race in races:
            ds, rs = race["dem_col_source"], race["rep_col_source"]
            dg, rg = race["dem_col_graph"], race["rep_col_graph"]
            validation.append({
                "year": year, "race_id": race["race_id"], "label": race["label"],
                "dem_source_col": ds, "rep_source_col": rs,
                **allocation_metrics(float(src[ds].sum()), float(src[rs].sum()),
                                     float(allocated[dg].sum()), float(allocated[rg].sum())),
            })
        allocated = allocated.reindex(sorted(target_geoms["node_id"].unique()), fill_value=0.0)
    return allocated.reset_index(), pd.DataFrame(validation), races


def preprocess_elections(args, graph, progress):
    layers = parse_layers(args.election_layer)
    unknown_years = layers.keys() - RACE_MANIFEST.keys()
    if unknown_years:
        progress.log(f"WARNING: no statewide race definitions for election years {sorted(unknown_years)}")
    definitions = {race[0]: (year, *race[1:])
                   for year in layers for race in RACE_MANIFEST.get(year, ())}
    selected = requested_races(args.race_ids, definitions)
    if not selected:
        raise ValueError("The configured layers have no statewide race definitions")
    layers = {year: path for year, path in sorted(layers.items())
              if any(race[0] in selected for race in RACE_MANIFEST.get(year, ()))}
    graph_ids = graph[args.node_id_col].to_numpy(dtype=np.int64)
    if not graph_ids.size or not pd.Index(graph_ids).is_unique:
        raise ValueError("Graph node IDs must be nonempty and unique")

    def decode(node_ids, matrix, metadata):
        races = metadata["races"]
        race_ids = [r["race_id"] for r in races]
        columns = [r[key] for r in races for key in ("dem_col_graph", "rep_col_graph")]
        if (not races or len(set(race_ids)) != len(race_ids) or set(race_ids) - selected
                or (args.race_ids and set(race_ids) != selected)):
            raise ValueError("Allocated race IDs do not match the election selection")
        if (node_ids.dtype != graph_ids.dtype
                or not np.array_equal(node_ids, graph_ids)
                or matrix.dtype != np.float64
                or matrix.shape != (len(graph_ids), len(columns))
                or not np.isfinite(matrix).all()):
            raise ValueError("Allocated vote matrix is invalid")
        validation = pd.DataFrame(metadata["validation"])
        if validation["race_id"].tolist() != race_ids:
            raise ValueError("Allocation validation does not cover the allocated races")
        for i, race in enumerate(races):
            race_id = race["race_id"]
            year, label, dem_candidates, rep_candidates = definitions[race_id]
            row = validation.iloc[i]
            if (race["year"] != year or race["label"] != label or row["year"] != year
                    or row["label"] != label or race["dem_col_graph"] != f"{race_id}DEM"
                    or race["rep_col_graph"] != f"{race_id}REP"
                    or race["dem_col_source"] != row["dem_source_col"]
                    or race["rep_col_source"] != row["rep_source_col"]
                    or canon(row["dem_source_col"]) not in map(canon, dem_candidates)
                    or canon(row["rep_source_col"]) not in map(canon, rep_candidates)):
                raise ValueError(f"Invalid allocation metadata for race {race_id}")
            sd, sr = float(row["dem_source_total"]), float(row["rep_source_total"])
            ad, ar = float(row["dem_allocated_total"]), float(row["rep_allocated_total"])
            if not np.isfinite([sd, sr, ad, ar]).all() or not np.allclose(
                [ad, ar], matrix[:, 2*i:2*i+2].sum(axis=0), rtol=1e-12, atol=1e-8
            ):
                raise ValueError(f"Inconsistent allocated statewide totals for race {race_id}")
            for metric, expected in allocation_metrics(sd, sr, ad, ar).items():
                actual = np.nan if pd.isna(row[metric]) else row[metric]
                if not np.isclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True):
                    raise ValueError(f"Invalid allocation validation metric {metric} for {race_id}")

        metrics = list(allocation_metrics(0, 0, 0, 0))
        validation[metrics] = validation[metrics].astype(float)
        votes = pd.DataFrame(matrix, columns=columns)
        votes.insert(0, "node_id", node_ids)
        return votes, validation, races

    def build():
        with progress.stage(f"Reading and projecting target geometry {args.target_precincts}"):
            target = load_target_geoms(
                graph, args.target_precincts, args.target_key_col,
                args.graph_target_key_col, args.node_id_col,
            )
            target = area_geometry(target, args.area_crs)
        votes = pd.DataFrame({"node_id": graph_ids})
        validations, races = [], []
        for index, (year, path) in enumerate(layers.items(), start=1):
            allocated, validation, layer_races = allocate_layer(
                year, path, target, args.area_crs, args.race_ids,
                progress=progress, label=f"Election layer {index}/{len(layers)} ({year})",
            )
            if layer_races:
                votes = votes.merge(allocated, on="node_id", how="left", validate="one_to_one")
                validations.append(validation)
                races.extend(layer_races)
        if not races:
            raise ValueError("No statewide races have usable election columns")
        validation = pd.concat(validations, ignore_index=True)
        metadata = {"races": races, "validation": validation.astype(object).where(
            validation.notna(), None
        ).to_dict("records")}
        matrix = votes.drop(columns="node_id").fillna(0.0).to_numpy(dtype=np.float64)
        return decode(graph_ids, matrix, metadata), graph_ids, matrix, metadata

    def identify():
        import geopandas as gpd
        import pyproj
        import shapely

        dependency_versions = {}
        for package in ("geopandas", "shapely", "pyproj", "pyogrio", "fiona", "pandas", "numpy"):
            try:
                dependency_versions[package] = version(package)
            except PackageNotFoundError:
                dependency_versions[package] = None
        graph_signatures = input_signatures(args.graph_json)
        if graph_signatures[0]["sha256"] != graph.attrs["source_sha256"]:
            raise RuntimeError("Graph JSON changed after it was loaded")
        return {
            "algorithm": "geopandas-intersection-buffer0-source-area-v1",
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "package_version": __version__,
            "dependencies": dependency_versions,
            "geos": shapely.geos_version_string,
            "proj": pyproj.proj_version_str,
            "io_engine": gpd.options.io_engine,
            "graph": graph_signatures,
            "target": input_signatures(args.target_precincts),
            "layers": {str(year): input_signatures(path) for year, path in layers.items()},
            "election_definitions": json.loads(json.dumps(
                {race_id: definitions[race_id] for race_id in sorted(selected)}
            )),
            "strict_selection": bool(args.race_ids),
            "area_crs": pyproj.CRS.from_user_input(args.area_crs).to_wkt(),
            "target_key_col": args.target_key_col,
            "graph_target_key_col": args.graph_target_key_col,
            "node_id_col": args.node_id_col,
        }

    if args.election_cache_mode == "off" or args.election_cache_dir is None:
        progress.log("Shared election cache disabled")
        result, *_ = build()
    else:
        result = cached_allocation(Path(args.election_cache_dir), args.election_cache_mode,
                                   identify, build, decode, progress)
    _, validation, races = result
    missing = selected - {race["race_id"] for race in races}
    if missing:
        progress.log(f"WARNING: manifest races with unavailable columns: {', '.join(sorted(missing))}")
    for row in validation.itertuples(index=False):
        share_change = row.statewide_dem_share_allocated - row.statewide_dem_share_source
        discrepancy = not np.isfinite(share_change) or not np.allclose(
            [row.dem_source_total, row.rep_source_total],
            [row.dem_allocated_total, row.rep_allocated_total],
            rtol=1e-9, atol=1e-6,
        )
        progress.log(
            f"{'WARNING: ' if discrepancy else ''}{row.race_id} allocation D "
            f"{row.dem_source_total:,.6f} → {row.dem_allocated_total:,.6f}, R "
            f"{row.rep_source_total:,.6f} → {row.rep_allocated_total:,.6f}, "
            f"statewide D share change {share_change:+.8%}"
        )
    return result


def chain_id_from_path(path):
    m = re.search(r"(run\d+)", Path(path).name)
    return m.group(1) if m else Path(path).stem

class AssignmentIndex:
    def __init__(self, graph, atlas_key_col, node_id_col, county_col):
        self.node_ids = graph[node_id_col].to_numpy(dtype=np.int64)
        self.normalize_key = lru_cache(maxsize=4 * len(graph))(norm_atlas_key)
        name_positions = {}
        for i, raw_key in enumerate(graph[atlas_key_col]):
            key = self.normalize_key(raw_key)
            if key in name_positions:
                raise ValueError(f"Duplicate graph key for {atlas_key_col}: {key!r}")
            name_positions[key] = i
        # The canonical parser flattens AtlasIO assignments to dicts. Keep the
        # legacy precedence of numeric node IDs over coincident precinct names.
        self.key_positions = name_positions | {
            str(node_id): i for i, node_id in enumerate(self.node_ids)
        }
        counties, labels = pd.factorize(graph[county_col])
        self.county_positions = counties
        self.county_count = len(labels)
        self.county_nodes = None if (counties >= 0).all() else counties >= 0

    def districts(self, districting):
        if isinstance(districting, DenseDistricting):
            if not np.array_equal(districting.node_ids, self.node_ids):
                raise ValueError("Sweep node IDs or graph order differ from responsiveness graph")
            return districting.districts

        districts = np.empty(len(self.node_ids), dtype=np.int64)
        assigned = np.zeros(len(self.node_ids), dtype=bool)
        for key, label in districting.items():
            position = self.key_positions.get(self.normalize_key(key))
            if position is None:
                raise ValueError(f"Unmatched districting key {key!r}")
            district = dist_label(label)
            if assigned[position] and districts[position] != district:
                raise ValueError(f"Conflicting assignments for graph node {self.node_ids[position]}")
            districts[position] = district
            assigned[position] = True
        if not assigned.all():
            raise ValueError(f"Assignment is missing {len(assigned) - assigned.sum():,} graph nodes")
        return districts

    def compliance(self, districts, max_per_county):
        counties = self.county_positions
        if self.county_nodes is not None:
            districts = districts[self.county_nodes]
            counties = counties[self.county_nodes]
        splits = county_split_sizes(districts, counties, self.county_count)
        return int(np.count_nonzero(splits > max_per_county)), int(splits.max(initial=0))


def iter_assignments(args, graph, progress, summaries, excluded):
    index = AssignmentIndex(graph, args.atlas_key_col, args.node_id_col, args.county_col)
    for chain_number, chain in enumerate(args.chain_files, start=1):
        chain_id = chain_id_from_path(chain)
        summary = ParseSummary(path=Path(chain), chain_id=chain_id)
        sweep_size = audited_sweep_size(
            chain, args.graph_json, max_records=args.max_maps_per_chain,
            node_id_col=args.node_id_col, key_col=args.atlas_key_col, county_col=args.county_col,
            max_districts_per_county=args.max_districts_per_county,
        )
        if sweep_size is None:
            if configured_sweep() is not None:
                progress.log(f"{chain_id}: rereading source for complete parse diagnostics")
            records = iter_map_records(chain, summary=summary)
            action = "reading AtlasIO and checking compliance"
        else:
            records = (
                {"districting": record.districting, "name": record.name,
                 "line_number": record.line_number}
                for record in iter_sweep_records(chain)
            )
            action = "reading shared assignments and compliance"
        limit = args.max_maps_per_chain
        total = sweep_size if sweep_size is not None else limit
        total_text = f"/{total:,}" if total is not None else ""
        seen = kept = invalid = filtered = 0
        dcs, worsts = [], []
        label = f"chain {chain_number}/{len(args.chain_files)} {chain_id}: {action} ({chain})"
        try:
            with progress.stage(label):
                for seen, record in enumerate(islice(records, limit), start=1):
                    item = {"source": "ensemble", "chain_id": chain_id, "map_index": seen,
                            "proposal": record["name"], "line_number": record["line_number"]}
                    try:
                        raw = record["districting"]
                        districts = index.districts(raw)
                        count = int(pd.unique(districts).size)
                        if count != args.n_districts:
                            raise ValueError(f"Expected {args.n_districts} districts, found {count}")
                        if isinstance(raw, DenseDistricting):
                            dc, worst = (raw.compliance[key]
                                         for key in ("double_cross_count", "worst_split"))
                        else:
                            dc, worst = index.compliance(districts, args.max_districts_per_county)
                    except (ValueError, TypeError, OverflowError) as exc:
                        invalid += 1
                        excluded.append({**item, "stage": "assignment", "reason": str(exc)})
                        if invalid <= 5:
                            progress.log(f"WARNING: {chain_id} map {seen} excluded: {exc}")
                    else:
                        dcs.append(dc)
                        worsts.append(worst)
                        if args.no_compliance_filter or dc == 0:
                            kept += 1
                            yield {**item, "assignment": districts}
                        else:
                            filtered += 1
                    if seen % 250 == 0:
                        progress.log(f"{chain_id}: {seen:,}{total_text} parsed, {kept:,} kept, "
                                     f"{filtered:,} compliance exclusions, {invalid:,} invalid")
                if sweep_size is not None and seen != sweep_size:
                    raise ValueError(f"Expected {sweep_size:,} cached maps, read {seen:,}")
        except (OSError, ValueError) as exc:
            line = summary.lines_seen if sweep_size is None else None
            excluded.append({"source": "ensemble", "chain_id": chain_id, "line_number": line,
                             "stage": "read_error", "reason": str(exc)})
            location = f", line {line}" if line is not None else ""
            raise RuntimeError(f"Cannot finish reading {chain_id} ({chain}){location}: {exc}") from exc
        finally:
            parse_failures = summary.records_seen - summary.records_yielded
            for error in summary.errors:
                progress.log(f"WARNING: {chain_id} parser excluded {error}")
                line = re.match(r"line (\d+):", error)
                excluded.append({"source": "ensemble", "chain_id": chain_id,
                                 "line_number": int(line[1]) if line else None,
                                 "stage": "parse", "reason": error})
            maps_seen = seen + parse_failures
            summaries.append({
                "chain_id": chain_id, "maps_seen": maps_seen, "maps_retained": kept,
                "invalid_maps": invalid + parse_failures,
                "parse_failures": parse_failures, "compliance_exclusions": filtered,
                "retained_fraction": kept / max(maps_seen, 1),
                "avg_double_crosses": float(np.mean(dcs)) if dcs else np.nan,
                "max_double_crosses": max(dcs, default=0),
                "worst_single_county_split": max(worsts, default=0),
            })
            progress.log(f"{chain_id}: {seen:,}{total_text} parsed, {kept:,} kept, {filtered:,} "
                         f"compliance exclusions, {invalid:,} invalid assignments, "
                         f"{parse_failures:,} parser exclusions")
            if limit is not None and seen < limit:
                progress.log(f"WARNING: {chain_id} supplied {seen:,} of {limit:,} requested maps")


def read_crosswalk(path):
    cw = pd.read_csv(path, dtype=str)
    b = resolve(cw.columns, ["GEOID20", "BLOCK", "Block", "GEOID"], "crosswalk block")
    p = resolve(cw.columns, ["NAME", "ORIG_NAME", "PRECINCT"], "crosswalk precinct")
    w = resolve(cw.columns, ["POP20", "ALLOCATED_POP20", "weight", "share"], "crosswalk weight")
    out = cw[[b, p, w]].copy().rename(columns={b: "GEOID20", p: "PRECINCT_NAME", w: "CW_WEIGHT"})
    out["GEOID20"] = out["GEOID20"].map(norm_block)
    out["PRECINCT_NAME"] = out["PRECINCT_NAME"].map(norm)
    out["CW_WEIGHT"] = pd.to_numeric(out["CW_WEIGHT"], errors="coerce").fillna(0.0)
    return out[(out["GEOID20"] != "") & (out["PRECINCT_NAME"] != "")].copy()

def read_baf(path, block_col=None, district_col=None):
    path = Path(path)
    df = pd.read_excel(path, dtype=str) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path, dtype=str)
    b = resolve(df.columns, [block_col, "BLOCK", "Block", "GEOID20", "GEOID", "block_geoid", "block_id"], "block")
    d = resolve(df.columns, [district_col, "DistrictID", "DistrictID:1", "DISTRICT", "District", "CD", "CONG_DIST", "assignment"], "district")
    out = df[[b, d]].copy()
    out[b] = out[b].map(norm_block)
    out[d] = out[d].map(dist_label)
    out = out[(out[b] != "") & out[d].notna()].drop_duplicates(subset=[b], keep="last")
    return out.rename(columns={b: "GEOID20", d: "DISTRICT"})[["GEOID20", "DISTRICT"]]

def official_assignment(block_plan, cw, graph_nodes, graph_key_col, node_id_col):
    merged = block_plan.merge(cw, on="GEOID20", how="inner")
    if merged.empty: raise ValueError("No overlap between official block plan and crosswalk")
    grouped = (merged.groupby(["PRECINCT_NAME", "DISTRICT"], as_index=False)["CW_WEIGHT"].sum()
               .sort_values(["PRECINCT_NAME", "CW_WEIGHT", "DISTRICT"], ascending=[True, False, True]))
    pa = grouped.drop_duplicates("PRECINCT_NAME", keep="first")[["PRECINCT_NAME", "DISTRICT"]]
    lookup = graph_nodes[[node_id_col, graph_key_col]].copy()
    lookup[graph_key_col] = lookup[graph_key_col].map(norm)
    matched = pa.rename(columns={"PRECINCT_NAME": graph_key_col}).merge(lookup, on=graph_key_col, how="inner")
    print(f"  official precinct assignment matched {matched[node_id_col].nunique():,} / "
          f"{graph_nodes[node_id_col].nunique():,} graph nodes", flush=True)
    s = pd.Series(matched["DISTRICT"].astype(int).to_numpy(), index=matched[node_id_col].astype(int))
    s.index.name = node_id_col
    return s[~s.index.duplicated(keep="first")].sort_index().astype(int)

def load_officials(graph_nodes, cw, enacted, overturned, remedial,
                   official_key_col, node_id_col, block_col, district_col, *, progress):
    out = []
    plans = {"Enacted ('25)": enacted, "Overturned ('21)": overturned, "Remedial ('22)": remedial}
    for number, (label, path) in enumerate(plans.items(), start=1):
        with progress.stage(f"Loading official plan {number}/{len(plans)} {label}: {path}"):
            assn = official_assignment(
                read_baf(path, block_col, district_col), cw, graph_nodes, official_key_col, node_id_col
            )
        out.append({"source": label, "chain_id": "", "map_index": "",
                    "proposal": label, "assignment": assn})
    return out


def load_publics(
    graph_nodes,
    cw,
    public_plan_dir,
    official_key_col,
    node_id_col,
    block_col,
    district_col,
    expected_districts,
    *,
    progress,
):
    source = Path(public_plan_dir)
    if not source.exists():
        raise FileNotFoundError(f"Public plan directory not found: {source}")
    paths = sorted(source.glob("*.csv"))
    if not paths:
        raise ValueError(f"No public plan CSV files found in {source}")

    graph_node_count = graph_nodes[node_id_col].nunique()
    out, manifest_rows = [], []
    for number, path in enumerate(paths, start=1):
        row = {"plan_id": path.stem, "path": str(path), "status": "failed", "reason": ""}
        try:
            block_plan = read_baf(path, block_col, district_col)
            block_districts = int(block_plan["DISTRICT"].nunique())
            if block_districts != expected_districts:
                raise ValueError(
                    f"expected {expected_districts} districts, found {block_districts}"
                )
            assn = official_assignment(
                block_plan, cw, graph_nodes, official_key_col, node_id_col
            )
            graph_districts = int(assn.nunique())
            matched_share = len(assn) / max(graph_node_count, 1)
            if graph_districts != expected_districts:
                raise ValueError(
                    f"graph assignment has {graph_districts} districts, "
                    f"expected {expected_districts}"
                )
            if matched_share < 0.95:
                raise ValueError(
                    f"matched graph-node share {matched_share:.1%} is below 95%"
                )
            out.append(
                {
                    "source": PUBLIC_SOURCE,
                    "chain_id": "",
                    "map_index": "",
                    "proposal": path.stem,
                    "assignment": assn,
                }
            )
            row.update(
                {
                    "status": "ok",
                    "matched_graph_nodes": len(assn),
                    "matched_graph_node_share": matched_share,
                    "n_districts": graph_districts,
                }
            )
        except Exception as exc:
            row["reason"] = f"{type(exc).__name__}: {exc}"
            progress.log(f"WARNING: skipping public plan {path.name}: {row['reason']}")
        manifest_rows.append(row)
        if number % 25 == 0:
            progress.log(f"Loaded {number:,}/{len(paths):,} public plans")

    progress.log(f"Public plans accepted for responsiveness: {len(out)} / {len(paths)}")
    return out, pd.DataFrame(manifest_rows)

def race_abbrev(race_id):
    """Compact labels for the right side of the histogram plot."""
    s = str(race_id)
    if not s.startswith("G") or len(s) < 6:
        return s
    yy = s[1:3]
    code = s[3:]
    mapping = {
        "PRE": "PR",
        "USS": "USS",
        "GOV": "GOV",
        "ATG": "ATG",
        "AUD": "AUD",
        "SOS": "SOS",
        "TRE": "TRE",
        "CJU": "CJU",
        "JUS1": "SC1",
        "JUS2": "SC2",
        "JUSU": "SCU",
    }
    return yy + mapping.get(code, code)


class DistrictVoteTotals:
    def __init__(self, votes, columns):
        self.node_index = pd.Index(votes["node_id"])
        self.votes = votes[columns]

    def for_assignment(self, assignment):
        # Preserve the vote-table order used by the old left-side inner join,
        # and pandas' compensated groupby sum used for seat comparisons.
        if isinstance(assignment, pd.Series):
            positions = assignment.index.get_indexer(self.node_index)
            covered = positions >= 0
            return self.votes.loc[covered].groupby(assignment.to_numpy()[positions[covered]]).sum()
        return self.votes.groupby(assignment).sum()


def evaluate(assignments, officials, votes, races, *, progress, n_districts,
             expected_maps=None, excluded=None):
    columns = [r[key] for r in races for key in ("dem_col_graph", "rep_col_graph")]
    totals = DistrictVoteTotals(votes, columns)
    race_rows = []
    for race in races:
        dg, rg = race["dem_col_graph"], race["rep_col_graph"]
        td, tr = float(votes[dg].sum()), float(votes[rg].sum())
        race_rows.append({
            "year": race["year"], "race_id": race["race_id"],
            "abbr": race_abbrev(race["race_id"]), "label": race["label"],
            "dem_col": dg, "rep_col": rg,
            "statewide_dem_share": td / (td + tr) if td + tr > 0 else np.nan,
            "statewide_dem_votes": td, "statewide_rep_votes": tr,
        })
    race_summary = pd.DataFrame(race_rows)

    def seats(item):
        grouped = totals.for_assignment(item["assignment"])
        if len(grouped) != n_districts:
            raise ValueError(f"{item['proposal']}: expected {n_districts} districts with graph "
                             f"nodes, found {len(grouped)}")
        values = grouped.to_numpy()
        return np.count_nonzero(values[:, ::2] > values[:, 1::2], axis=0)

    def frame(metadata, counts):
        num_maps = len(metadata)
        if not num_maps:
            counts = np.empty((0, len(races)), dtype=int)
        data = {key: np.repeat(race_summary[key].to_numpy(), num_maps)
                for key in ("year", "race_id", "label", "statewide_dem_share")}
        data.update({key: np.tile(np.asarray([item[i] for item in metadata], dtype=object),
                                  len(races))
                     for i, key in enumerate(("source", "chain_id", "map_index", "proposal"))})
        data["dem_seats"] = np.asarray(counts).T.ravel()
        return pd.DataFrame(data)

    refs, ref_counts = [], []
    with progress.stage("Evaluating official and requested public plans"):
        for number, item in enumerate(officials, start=1):
            refs.append((item["source"], item["chain_id"], item["map_index"], item["proposal"]))
            ref_counts.append(seats(item))
            if item["source"] != PUBLIC_SOURCE or number % 25 == 0:
                progress.log(f"Reference plan {number:,}/{len(officials):,}: {item['proposal']}")

    ensemble, ens_counts = [], []
    aggregation_seconds = 0.0
    with progress.stage(f"Streaming ensemble maps and aggregating votes across {len(races)} races",
                        heartbeat=False):
        for number, item in enumerate(assignments, start=1):
            started = perf_counter()
            try:
                counts = seats(item)
            except (ValueError, TypeError, IndexError) as exc:
                if excluded is None:
                    raise
                excluded.append({key: value for key, value in item.items() if key != "assignment"}
                                | {"stage": "evaluation", "reason": str(exc)})
                progress.log(f"WARNING: {item['chain_id']} map {item['map_index']} "
                             f"excluded during evaluation: {exc}")
            else:
                ens_counts.append(counts)
                ensemble.append((item["source"], item["chain_id"], item["map_index"],
                                 item["proposal"]))
            aggregation_seconds += perf_counter() - started
            if number % 250 == 0:
                total = f" of {expected_maps:,} requested" if expected_maps is not None else ""
                progress.log(f"Aggregated {len(ensemble):,}/{number:,} eligible maps{total} "
                             f"({aggregation_seconds:.1f}s in vote aggregation)")
        progress.log(f"Aggregated {len(ensemble):,} retained maps "
                     f"({aggregation_seconds:.1f}s in vote aggregation)")
    return frame(ensemble, ens_counts), frame(refs, ref_counts), race_summary


def plot(ensemble, official, races, output, title, n_districts):
    """
    NC-style horizontal histograms.

    Each statewide race is drawn at its statewide Democratic vote share.
    The ensemble is a small histogram over Democratic-won district counts.
    The official plans are overlaid as colored circular dots.
    """
    races = races.copy()
    if "abbr" not in races.columns:
        races["abbr"] = races["race_id"].map(race_abbrev)

    races = races.sort_values(["statewide_dem_share", "year", "race_id"]).reset_index(drop=True)
    fig_height = max(7.2, 0.34 * len(races) + 2.5)
    fig, ax = plt.subplots(figsize=(11.8, fig_height))

    yvals = races["statewide_dem_share"].dropna().to_numpy(float)
    if len(yvals) > 1:
        gaps = np.diff(np.sort(np.unique(yvals)))
        gaps = gaps[gaps > 1e-9]
        if len(gaps):
            ridge_height = min(0.008, max(0.0022, 0.35 * np.percentile(gaps, 25)))
        else:
            ridge_height = 0.004
    else:
        ridge_height = 0.004
    ridge_height = max(ridge_height, 0.0028)

    hist_face = "#b9d7e8"
    hist_edge = "#34495e"
    plan_offsets = {"Enacted ('25)": -0.08, "Overturned ('21)": 0.00, "Remedial ('22)": 0.08}

    for _, race in races.iterrows():
        race_id = race["race_id"]
        y = float(race["statewide_dem_share"])
        sub = ensemble[ensemble["race_id"] == race_id]
        if sub.empty:
            continue

        counts = sub["dem_seats"].value_counts().reindex(range(0, n_districts + 1), fill_value=0).sort_index()
        fractions = counts / max(counts.sum(), 1)
        max_fraction = float(fractions.max()) if len(fractions) else 0.0
        if max_fraction <= 0:
            continue
        heights = fractions / max_fraction * ridge_height

        ax.hlines(y, xmin=-0.15, xmax=n_districts + 0.35, color="black", linewidth=0.45, alpha=0.55, zorder=1)

        for seats, h in heights.items():
            if h <= 0:
                continue
            ax.bar(
                seats,
                h,
                width=0.78,
                bottom=y,
                align="center",
                color=hist_face,
                edgecolor=hist_edge,
                linewidth=0.7,
                alpha=0.88,
                zorder=2,
            )

        off = official[official["race_id"] == race_id]
        for plan_label, style in official_plan_styles(PLAN_STYLES):
            row = off[off["source"] == plan_label]
            if row.empty:
                continue
            x = float(row["dem_seats"].iloc[0]) + plan_offsets.get(plan_label, 0.0)
            ax.scatter(
                x,
                y,
                s=72,
                marker="o",
                color=style["color"],
                edgecolor="black",
                linewidth=0.55,
                zorder=5,
            )

        ax.text(n_districts + 0.48, y + 0.08 * ridge_height, race["abbr"], fontsize=9.5, va="center", ha="left")

    ymin = max(0.0, float(races["statewide_dem_share"].min()) - 0.012)
    ymax = min(1.0, float(races["statewide_dem_share"].max()) + 0.014)

    # Proportionality reference: seats = statewide two-party Democratic share * number of districts.
    yline = np.linspace(ymin, ymax, 100)
    ax.plot(n_districts * yline, yline, color="black", linestyle=(0, (7, 5)), linewidth=2.0, alpha=0.85, zorder=0)

    key_lines = [
        "PR - President",
        "USS - U.S. Senate",
        "GOV - Governor",
        "ATG - Attorney General",
        "AUD - Auditor",
        "SOS - Secretary of State",
        "TRE - Treasurer",
        "CJU - Chief Justice",
        "SC - Supreme Court",
    ]
    ax.text(
        0.985,
        0.02,
        "\n".join(key_lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
        bbox={"boxstyle": "square,pad=0.35", "facecolor": "white", "edgecolor": "0.45", "alpha": 0.92},
    )

    ax.set_xlim(-0.4, n_districts + 2.6)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(range(0, n_districts + 1))
    ax.set_xlabel("Number of Democratic-won congressional districts")
    ax.set_ylabel("Statewide two-party Democratic vote share")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D([0], [0], color=hist_edge, marker="s", linestyle="None", markerfacecolor=hist_face,
               markeredgecolor=hist_edge, markersize=10, label="Compliant ensemble histogram"),
    ]
    for label, style in official_plan_styles(PLAN_STYLES):
        handles.append(Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=style["color"],
                              markeredgecolor="black", markersize=8.5, label=label))
    figure_legend(ax, handles=handles, loc="upper left", frameon=True)

    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output = save_figure(fig, output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figure: {output}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--graph-json", required=True)
    p.add_argument("--target-precincts", required=True, help="Geometry layer for OHpct20 graph nodes")
    p.add_argument("--target-key-col", default="NAME")
    p.add_argument("--graph-target-key-col", default="ORIG_NAME")
    p.add_argument("--chain-files", nargs="+", required=True)
    p.add_argument("--election-layer", action="append", required=True, help="YEAR=path/to/RDH/file.shp")
    p.add_argument("--crosswalk-csv", required=True)
    p.add_argument("--enacted-baf", required=True)
    p.add_argument("--overturned-baf", required=True)
    p.add_argument("--remedial-baf", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--node-id-col", default="id")
    p.add_argument("--atlas-key-col", default="NAME")
    p.add_argument("--official-key-col", default="ORIG_NAME")
    p.add_argument("--county-col", default="COUNTY")
    p.add_argument("--n-districts", type=int, default=15)
    p.add_argument("--area-crs", default="EPSG:26917")
    p.add_argument("--max-districts-per-county", type=int, default=2)
    p.add_argument("--max-maps-per-chain", type=int, default=None)
    p.add_argument("--no-compliance-filter", action="store_true")
    p.add_argument("--assignment-block-col", default=None)
    p.add_argument("--assignment-district-col", default=None)
    p.add_argument("--overlay-public-plans", action="store_true")
    p.add_argument("--public-plan-dir", default=None)
    p.add_argument("--public-seats-output", default=None)
    p.add_argument("--race-ids", nargs="+", metavar="ID",
                   help="Evaluate only these statewide race IDs, e.g. G20PRE G24USS")
    p.add_argument("--election-cache-dir", type=Path,
                   help="Shared directory for validated election allocations")
    p.add_argument("--election-cache-mode", choices=("use", "rebuild", "off"), default="use",
                   help="Reuse, rebuild, or bypass the shared election allocation cache")
    p.add_argument("--progress-interval", type=float, default=60.0, metavar="SECONDS",
                   help="Wall and CPU time heartbeat interval (0 disables heartbeats)")
    p.add_argument("--title", default="Ohio congressional responsiveness across statewide elections")
    return p.parse_args()

def main():
    args = parse_args()
    if args.election_cache_mode == "rebuild" and args.election_cache_dir is None:
        raise ValueError("--election-cache-mode rebuild requires --election-cache-dir")
    if not np.isfinite(args.progress_interval) or args.progress_interval < 0:
        raise ValueError("--progress-interval must be finite and nonnegative")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_plot = Path(args.output) if args.output else (
        outdir / "ohio_responsiveness_all_statewide_races.png"
    )
    progress = Progress(args.progress_interval)
    with progress.stage(f"Reading graph {args.graph_json}"):
        graph = read_graph_nodes(args.graph_json)
        progress.log(f"Loaded {len(graph):,} graph nodes")

    votes, validation, races = preprocess_elections(args, graph, progress)
    with progress.stage("Writing allocated vote, validation, and race manifest CSVs"):
        votes.to_csv(outdir / "oh_graph_node_votes_all_statewide_races.csv", index=False)
        validation.to_csv(outdir / "oh_election_allocation_validation.csv", index=False)
        pd.DataFrame(races).to_csv(outdir / "oh_statewide_race_manifest_used.csv", index=False)

    with progress.stage(f"Reading reference-plan block crosswalk {args.crosswalk_csv}"):
        crosswalk = read_crosswalk(args.crosswalk_csv)
    officials = load_officials(
        graph, crosswalk, args.enacted_baf, args.overturned_baf, args.remedial_baf,
        args.official_key_col, args.node_id_col, args.assignment_block_col,
        args.assignment_district_col, progress=progress,
    )
    publics = []
    if args.overlay_public_plans:
        if args.public_plan_dir is None:
            raise ValueError("--overlay-public-plans requires --public-plan-dir")
        with progress.stage(f"Loading public plans {args.public_plan_dir}"):
            publics, public_manifest = load_publics(
                graph, crosswalk, args.public_plan_dir, args.official_key_col,
                args.node_id_col, args.assignment_block_col, args.assignment_district_col,
                args.n_districts, progress=progress,
            )
            public_manifest.to_csv(
                outdir / "oh_responsiveness_public_plan_manifest.csv", index=False
            )

    summaries, excluded = [], []
    assignments = iter_assignments(args, graph, progress, summaries, excluded)
    expected_maps = (args.max_maps_per_chain * len(args.chain_files)
                     if args.max_maps_per_chain is not None else None)
    try:
        ens, evaluated_refs, race_summary = evaluate(
            assignments, officials + publics, votes, races, progress=progress,
            n_districts=args.n_districts, expected_maps=expected_maps, excluded=excluded,
        )
    finally:
        assignments.close()
        with progress.stage("Writing chain compliance and map exclusion CSVs"):
            failures = pd.DataFrame(
                excluded, columns=["source", "chain_id", "map_index", "proposal", "line_number",
                                   "stage", "reason"],
            )
            evaluation_counts = failures[failures["stage"] == "evaluation"]["chain_id"].value_counts()
            for row in summaries:
                evaluation_failures = int(evaluation_counts.get(row["chain_id"], 0))
                row["evaluation_exclusions"] = evaluation_failures
                row["invalid_maps"] += evaluation_failures
                row["maps_retained"] -= evaluation_failures
                row["retained_fraction"] = row["maps_retained"] / max(row["maps_seen"], 1)
            compliance = pd.DataFrame(summaries)
            compliance.to_csv(outdir / "oh_responsiveness_compliance_summary.csv", index=False)
            failures.to_csv(outdir / "oh_responsiveness_excluded_maps.csv", index=False)
            if summaries:
                progress.log("Chain counts:\n" + compliance.to_string(index=False))

    if ens.empty:
        raise RuntimeError("No ensemble assignments retained")
    off = evaluated_refs[evaluated_refs["source"].isin(PLAN_STYLES)].copy()
    with progress.stage("Writing ensemble, reference, and race summary CSVs"):
        ens.to_csv(outdir / "oh_responsiveness_ensemble_seats.csv", index=False)
        off.to_csv(outdir / "oh_responsiveness_official_plan_seats.csv", index=False)
        if args.overlay_public_plans:
            public_seats = evaluated_refs[evaluated_refs["source"] == PUBLIC_SOURCE]
            public_seats_output = (
                Path(args.public_seats_output) if args.public_seats_output
                else outdir / "oh_responsiveness_public_plan_seats.csv"
            )
            public_seats_output.parent.mkdir(parents=True, exist_ok=True)
            public_seats.to_csv(public_seats_output, index=False)
            progress.log(f"Wrote public-plan seats: {public_seats_output}")
        race_summary.to_csv(outdir / "oh_responsiveness_race_summary.csv", index=False)

    with progress.stage("Generating responsiveness figure"):
        plot(ens, off, race_summary, output_plot, args.title, args.n_districts)
    progress.log(f"Completed responsiveness: {outdir}")


if __name__ == "__main__":
    main()
