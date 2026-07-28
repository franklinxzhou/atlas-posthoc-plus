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

import argparse, gzip, json, re
from pathlib import Path
from typing import Any, Iterable

from atlas_posthoc_plus.paper import save_figure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from atlas_posthoc_plus.io.atlas_records import iter_map_records
from atlas_posthoc_plus.io.chain_sweep import (
    cached_assignment_series,
    cached_compliance,
    configured_sweep,
)

PLAN_STYLES = {
    "Enacted ('25)": {"color": "#9467bd", "marker": "o"},
    "Overturned ('21)": {"color": "#2ca02c", "marker": "o"},
    "Remedial ('22)": {"color": "#d62728", "marker": "o"},
}

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
    with Path(path).open() as f: obj = json.load(f)
    nodes = obj.get("nodes") if isinstance(obj.get("nodes"), list) else obj.get("node_attributes")
    if not isinstance(nodes, list): raise ValueError("graph JSON needs nodes or node_attributes")
    df = pd.DataFrame(nodes).copy()
    for c in ["id", "NAME", "ORIG_NAME", "COUNTY"]:
        if c not in df.columns: raise ValueError(f"graph missing {c}; columns={list(df.columns)}")
    df["id"] = pd.to_numeric(df["id"], errors="raise").astype(int)
    for c in ["NAME", "ORIG_NAME", "COUNTY"]: df[c] = df[c].map(norm)
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
    print(f"Target geometry matched {n_match:,} / {n_graph:,} graph nodes")
    if n_match < 0.95 * n_graph: print("WARNING: target geometry match below 95%")
    gdf = gdf.rename(columns={node_id_col: "node_id"})
    gdf["node_id"] = gdf["node_id"].astype(int)
    return gdf[["node_id", "geometry"]].copy()

def parse_layers(items):
    out = {}
    for item in items:
        y, p = item.split("=", 1)
        out[int(y)] = Path(p)
    return out

def infer_races(year, columns):
    races = []
    for race_id, label, dem_cands, rep_cands in RACE_MANIFEST.get(year, []):
        dc, rc = pick(columns, dem_cands), pick(columns, rep_cands)
        if dc and rc:
            races.append({
                "year": year, "race_id": race_id, "label": label,
                "dem_col_source": dc, "rep_col_source": rc,
                "dem_col_graph": f"{race_id}DEM", "rep_col_graph": f"{race_id}REP",
            })
        else:
            print(f"  Skipping {label}: dem {dem_cands}->{dc}, rep {rep_cands}->{rc}")
    return races

def spatial_crosswalk(source, target, area_crs):
    import geopandas as gpd
    if source.crs is None: raise ValueError("election layer has no CRS")
    if target.crs is None: raise ValueError("target layer has no CRS")
    src = source.reset_index(drop=True).copy()
    src["_source_id"] = np.arange(len(src), dtype=int)
    src = src.to_crs(area_crs)
    tgt = target.to_crs(area_crs)
    src["geometry"] = src.geometry.buffer(0)
    tgt["geometry"] = tgt.geometry.buffer(0)
    src["_src_area"] = src.geometry.area
    src = src[src["_src_area"] > 0].copy()
    inter = gpd.overlay(src[["_source_id", "_src_area", "geometry"]],
                        tgt[["node_id", "geometry"]],
                        how="intersection", keep_geom_type=False)
    if inter.empty: raise ValueError("spatial overlay produced no intersections")
    inter["_inter_area"] = inter.geometry.area
    inter = inter[inter["_inter_area"] > 0].copy()
    inter["weight"] = inter["_inter_area"] / inter["_src_area"]
    return inter[["_source_id", "node_id", "weight"]]

def allocate_layer(year, election_path, target_geoms, area_crs):
    print(f"\nReading election layer {year}: {election_path}")
    src = read_vector(election_path)
    races = infer_races(year, src.columns)
    if not races:
        return pd.DataFrame({"node_id": target_geoms["node_id"]}), pd.DataFrame(), []
    needed = sorted(set([r["dem_col_source"] for r in races] + [r["rep_col_source"] for r in races]))
    for c in needed: src[c] = pd.to_numeric(src[c], errors="coerce").fillna(0.0)
    print(f"  Found {len(races)} statewide D/R races:")
    for r in races: print(f"    {r['label']}: {r['dem_col_source']} / {r['rep_col_source']}")
    print("  Building spatial crosswalk...")
    cw = spatial_crosswalk(src, target_geoms, area_crs)
    values = src[needed].copy()
    values["_source_id"] = np.arange(len(values), dtype=int)
    long = cw.merge(values, on="_source_id", how="left")
    allocated = pd.DataFrame({"node_id": sorted(target_geoms["node_id"].unique())})
    val_rows = []
    for r in races:
        ds, rs = r["dem_col_source"], r["rep_col_source"]
        dg, rg = r["dem_col_graph"], r["rep_col_graph"]
        long[dg], long[rg] = long[ds] * long["weight"], long[rs] * long["weight"]
        grouped = long.groupby("node_id", as_index=False)[[dg, rg]].sum()
        allocated = allocated.merge(grouped, on="node_id", how="left")
        sd, sr = float(src[ds].sum()), float(src[rs].sum())
        ad, ar = float(grouped[dg].sum()), float(grouped[rg].sum())
        val_rows.append({
            "year": year, "race_id": r["race_id"], "label": r["label"],
            "dem_source_col": ds, "rep_source_col": rs,
            "dem_source_total": sd, "rep_source_total": sr,
            "dem_allocated_total": ad, "rep_allocated_total": ar,
            "dem_allocation_ratio": ad / sd if sd else np.nan,
            "rep_allocation_ratio": ar / sr if sr else np.nan,
            "statewide_dem_share_source": sd / (sd + sr) if sd + sr > 0 else np.nan,
            "statewide_dem_share_allocated": ad / (ad + ar) if ad + ar > 0 else np.nan,
        })
    return allocated.fillna(0.0), pd.DataFrame(val_rows), races

def open_maybe_gzip(path):
    path = Path(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open("r")

def maybe_json_loads(x):
    while isinstance(x, str):
        s = x.strip()
        if not s or not (s.startswith("{") or s.startswith("[") or s.startswith('"')): return x
        try: x = json.loads(s)
        except json.JSONDecodeError: return x
    return x

def iter_json_values(line):
    dec, idx, n = json.JSONDecoder(), 0, len(line)
    while idx < n:
        while idx < n and line[idx].isspace(): idx += 1
        if idx >= n: break
        obj, end = dec.raw_decode(line, idx)
        yield obj
        idx = end

def iter_atlasio_records(path):
    yield from iter_map_records(path)

def chain_id_from_path(path):
    m = re.search(r"(run\d+)", Path(path).name)
    return m.group(1) if m else Path(path).stem

def atlasio_assignment(districting, graph_nodes, key_col, node_id_col):
    cached = cached_assignment_series(districting, node_id_col=node_id_col)
    if cached is not None:
        return cached.sort_index()

    node_ids = graph_nodes[node_id_col].astype(int).tolist()
    name_to_id = {norm_atlas_key(row[key_col]): int(row[node_id_col]) for _, row in graph_nodes.iterrows()}
    districting = maybe_json_loads(districting)
    if isinstance(districting, list) and districting:
        if all(isinstance(x, dict) and len(x) == 1 for x in districting):
            out, unmatched = {}, []
            for item in districting:
                k, d = next(iter(item.items()))
                node_id = name_to_id.get(norm_atlas_key(k))
                if node_id is None: unmatched.append(norm_atlas_key(k)); continue
                out[node_id] = dist_label(d)
            if unmatched: raise ValueError(f"{len(unmatched)} unmatched Atlas keys; examples={unmatched[:5]}")
            if len(out) != len(graph_nodes): raise ValueError(f"matched {len(out)} but graph has {len(graph_nodes)}")
            s = pd.Series(out, dtype=int); s.index.name = node_id_col; return s.sort_index()
        if len(districting) == len(graph_nodes) and not isinstance(districting[0], (list, tuple, dict, set)):
            s = pd.Series([dist_label(x) for x in districting], index=node_ids, dtype=int)
            s.index.name = node_id_col; return s.sort_index()
    if isinstance(districting, dict):
        graph_id_strings = {str(x) for x in node_ids}
        out = {}
        for k, d in districting.items():
            key = norm_atlas_key(k)
            node_id = int(key) if str(key) in graph_id_strings else name_to_id.get(key)
            if node_id is None: raise ValueError(f"districting key {k!r} unmatched")
            out[node_id] = dist_label(d)
        s = pd.Series(out, dtype=int); s.index.name = node_id_col; return s.sort_index()
    raise ValueError(f"unsupported districting type {type(districting)}")

def compliance(assignment, graph_nodes, node_id_col, county_col, max_per_county):
    cached = cached_compliance(assignment)
    if cached is not None:
        return cached

    a = assignment.rename("DISTRICT").rename_axis(node_id_col).reset_index()
    a[node_id_col] = pd.to_numeric(a[node_id_col], errors="coerce")
    nodes = graph_nodes[[node_id_col, county_col]].copy()
    nodes[node_id_col] = pd.to_numeric(nodes[node_id_col], errors="coerce")
    assigned = nodes.merge(a, on=node_id_col, how="left")
    by_county = assigned.dropna(subset=["DISTRICT"]).groupby(county_col)["DISTRICT"].nunique()
    double_cross = int((by_county > max_per_county).sum())
    return {"compliant": double_cross == 0, "double_cross_count": double_cross,
            "worst_split": int(by_county.max()) if len(by_county) else 0}

def load_assignments(chain_files, graph_nodes, atlas_key_col, node_id_col, county_col, max_per_county, max_maps, no_filter):
    assignments, rows = [], []
    for chain in chain_files:
        chain_id = chain_id_from_path(chain)
        action = "Analyzing shared assignments" if configured_sweep() else "Sweeping chain"
        print(f"\n{action} {chain_id}: {chain}")
        seen = kept = invalid = 0
        dcs, worsts = [], []
        for rec in iter_atlasio_records(chain):
            seen += 1
            if max_maps is not None and seen > max_maps: break
            try:
                assn = atlasio_assignment(rec["districting"], graph_nodes, atlas_key_col, node_id_col)
                comp = compliance(assn, graph_nodes, node_id_col, county_col, max_per_county)
                dcs.append(comp["double_cross_count"]); worsts.append(comp["worst_split"])
                if (not no_filter) and (not comp["compliant"]): continue
                assignments.append({"source": "ensemble", "chain_id": chain_id, "map_index": seen,
                                    "proposal": rec.get("name", seen), "assignment": assn})
                kept += 1
            except Exception as exc:
                invalid += 1
                if invalid <= 5: print(f"  FAILED map {seen}: {type(exc).__name__}: {exc}")
        rows.append({"chain_id": chain_id, "maps_seen": seen, "maps_retained": kept,
                     "invalid_maps": invalid, "retained_fraction": kept / max(seen, 1),
                     "avg_double_crosses": float(np.mean(dcs)) if dcs else np.nan,
                     "max_double_crosses": int(np.max(dcs)) if dcs else 0,
                     "worst_single_county_split": int(np.max(worsts)) if worsts else 0})
        print(f"  maps_seen={seen}\n  maps_retained={kept}\n  invalid_maps={invalid}")
    if not assignments: raise RuntimeError("No ensemble assignments retained")
    return assignments, pd.DataFrame(rows)

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
    print(f"  official precinct assignment matched {matched[node_id_col].nunique():,} / {graph_nodes[node_id_col].nunique():,} graph nodes")
    s = pd.Series(matched["DISTRICT"].astype(int).to_numpy(), index=matched[node_id_col].astype(int))
    s.index.name = node_id_col
    return s[~s.index.duplicated(keep="first")].sort_index().astype(int)

def load_officials(graph_nodes, crosswalk_csv, enacted, overturned, remedial, official_key_col, node_id_col, block_col, district_col):
    cw = read_crosswalk(crosswalk_csv)
    out = []
    for label, path in {"Enacted ('25)": enacted, "Overturned ('21)": overturned, "Remedial ('22)": remedial}.items():
        print(f"\nReading official plan: {label}")
        assn = official_assignment(read_baf(path, block_col, district_col), cw, graph_nodes, official_key_col, node_id_col)
        out.append({"source": label, "chain_id": "", "map_index": "", "proposal": label, "assignment": assn})
    return out

def dem_seats(assignment, votes, dem_col, rep_col, node_id_col="node_id"):
    a = assignment.rename("DISTRICT").rename_axis(node_id_col).reset_index()
    a[node_id_col] = pd.to_numeric(a[node_id_col], errors="coerce").astype(int)
    df = votes[[node_id_col, dem_col, rep_col]].merge(a, on=node_id_col, how="inner")
    g = df.groupby("DISTRICT")[[dem_col, rep_col]].sum()
    return int((g[dem_col] > g[rep_col]).sum())

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


def evaluate(assignments, officials, votes, races):
    ens_rows, off_rows, race_rows = [], [], []
    for r in races:
        dg, rg = r["dem_col_graph"], r["rep_col_graph"]
        if dg not in votes.columns or rg not in votes.columns: continue
        td, tr = float(votes[dg].sum()), float(votes[rg].sum())
        share = td / (td + tr) if td + tr > 0 else np.nan
        race_rows.append({"year": r["year"], "race_id": r["race_id"], "abbr": race_abbrev(r["race_id"]),
                          "label": r["label"], "dem_col": dg, "rep_col": rg,
                          "statewide_dem_share": share,
                          "statewide_dem_votes": td, "statewide_rep_votes": tr})
        for item in assignments + officials:
            row = {"year": r["year"], "race_id": r["race_id"], "label": r["label"],
                   "statewide_dem_share": share, "source": item["source"],
                   "chain_id": item["chain_id"], "map_index": item["map_index"],
                   "proposal": item["proposal"], "dem_seats": dem_seats(item["assignment"], votes, dg, rg)}
            (ens_rows if item["source"] == "ensemble" else off_rows).append(row)
    return pd.DataFrame(ens_rows), pd.DataFrame(off_rows), pd.DataFrame(race_rows)

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
        for plan_label, style in PLAN_STYLES.items():
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
    ax.grid(axis="both", alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D([0], [0], color=hist_edge, marker="s", linestyle="None", markerfacecolor=hist_face,
               markeredgecolor=hist_edge, markersize=10, label="Compliant ensemble histogram"),
    ]
    for label, style in PLAN_STYLES.items():
        handles.append(Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=style["color"],
                              markeredgecolor="black", markersize=8.5, label=label))
    ax.legend(handles=handles, loc="upper left", frameon=True)

    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output = save_figure(fig, output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figure: {output}")


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
    p.add_argument("--title", default="Ohio congressional responsiveness across statewide elections")
    return p.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    output_plot = Path(args.output) if args.output else outdir / "ohio_responsiveness_all_statewide_races.png"

    print("="*80); print("Ohio responsiveness plot: all statewide D/R races"); print("="*80)

    graph = read_graph_nodes(args.graph_json)
    target = load_target_geoms(graph, args.target_precincts, args.target_key_col, args.graph_target_key_col, args.node_id_col)

    votes = pd.DataFrame({"node_id": graph[args.node_id_col].astype(int)})
    validations, races_all = [], []
    for year, path in sorted(parse_layers(args.election_layer).items()):
        alloc, val, races = allocate_layer(year, path, target, args.area_crs)
        votes = votes.merge(alloc, on="node_id", how="left")
        validations.append(val); races_all.extend(races)
    votes = votes.fillna(0.0)

    validations_df = pd.concat(validations, ignore_index=True) if validations else pd.DataFrame()
    votes.to_csv(outdir / "oh_graph_node_votes_all_statewide_races.csv", index=False)
    validations_df.to_csv(outdir / "oh_election_allocation_validation.csv", index=False)
    pd.DataFrame(races_all).to_csv(outdir / "oh_statewide_race_manifest_used.csv", index=False)

    assignments, comp = load_assignments(args.chain_files, graph, args.atlas_key_col, args.node_id_col, args.county_col,
                                         args.max_districts_per_county, args.max_maps_per_chain, args.no_compliance_filter)
    comp.to_csv(outdir / "oh_responsiveness_compliance_summary.csv", index=False)
    print(comp.to_string(index=False))

    officials = load_officials(graph, args.crosswalk_csv, args.enacted_baf, args.overturned_baf, args.remedial_baf,
                               args.official_key_col, args.node_id_col, args.assignment_block_col, args.assignment_district_col)

    ens, off, race_summary = evaluate(assignments, officials, votes, races_all)
    ens.to_csv(outdir / "oh_responsiveness_ensemble_seats.csv", index=False)
    off.to_csv(outdir / "oh_responsiveness_official_plan_seats.csv", index=False)
    race_summary.to_csv(outdir / "oh_responsiveness_race_summary.csv", index=False)

    plot(ens, off, race_summary, output_plot, args.title, args.n_districts)
    print("\nDone."); print(f"Output directory: {outdir}")

if __name__ == "__main__":
    main()
