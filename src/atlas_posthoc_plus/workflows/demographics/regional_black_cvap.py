#!/usr/bin/env python3
"""Plot district-level African American CVAP sorting within fixed regions.

For each plan, region, and intersecting congressional district, the workflow
calculates a regional location quotient: the African American CVAP share of
the district-region intersection divided by the African American CVAP share of
the region.  It pairs that quantity with the African American CVAP share of the
whole congressional district.  This keeps districts as the observational unit
without requiring plans to touch each CUA or MSA the same number of times.

The ensemble calculation uses the exact CUA/MSA dictionaries attached to the
statewide graph.  Official plans are evaluated directly from block assignment
files so precincts split by an official congressional boundary are not forced
into a single district.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atlas_posthoc_plus.paper import (
    DATA_LABEL_GID,
    apply_paper_style,
    figure_legend,
    official_plan_styles,
    save_figure,
)

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from atlas_posthoc_plus.io.atlas_records import iter_map_records
from atlas_posthoc_plus.io.chain_sweep import configured_sweep
from atlas_posthoc_plus.io.graph import read_graph_frame
from atlas_posthoc_plus.workflows.demographics.ranked_vote_cvap import (
    atlasio_districting_to_series,
    build_node_key_lookup,
    chain_id_from_path,
    county_compliance_summary_for_assignment,
    read_block_assignment_excel,
    read_cvap_file,
)


DUKE_NAVY = "#012169"
DUKE_NAVY_LIGHT = "#7A99AC"
GRID_COLOR = "#D8DDE3"
ENSEMBLE_LABEL = __import__("os").environ.get("ATLAS_POSTHOC_ENSEMBLE_LABEL", "Ensemble")
DEFAULT_MINIMUM_REGION_CVAP_SHARE = 0.01

PLAN_STYLES = {
    "Enacted ('25)": {"color": "#7A5195", "marker": "o"},
    "Overturned ('21)": {"color": "#0072B2", "marker": "s"},
    "Remedial ('22)": {"color": "#E69F00", "marker": "D"},
}


def apply_figure_style() -> None:
    """Apply the role-mapped publication style used by the regional figures."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": False,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    apply_paper_style()


@dataclass(frozen=True)
class RegionData:
    geography: str
    manifest: pd.DataFrame
    node_ids: np.ndarray
    regional_population: np.ndarray
    regional_cvap: np.ndarray
    regional_black_cvap: np.ndarray


@dataclass(frozen=True)
class NodeDemographics:
    node_ids: np.ndarray
    cvap: np.ndarray
    african_american_cvap: np.ndarray


@dataclass(frozen=True)
class RankedPlan:
    black_fraction: np.ndarray
    touching: np.ndarray
    rows: list[dict[str, Any]]


def _validate_node_demographics(
    demographics: NodeDemographics,
    expected_node_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    node_ids = np.asarray(demographics.node_ids, dtype=int)
    if not np.array_equal(node_ids, np.asarray(expected_node_ids, dtype=int)):
        raise ValueError("Node demographics do not match regional graph node order")
    cvap = np.asarray(demographics.cvap, dtype=float)
    african_american = np.asarray(demographics.african_american_cvap, dtype=float)
    if cvap.shape != node_ids.shape or african_american.shape != node_ids.shape:
        raise ValueError("Node demographic arrays must match node_ids")
    if not np.isfinite(cvap).all() or not np.isfinite(african_american).all():
        raise ValueError("Node demographics must be finite")
    if np.any(cvap < 0) or np.any(african_american < 0):
        raise ValueError("Node demographics must be nonnegative")
    # The CVAP special tabulation contains independently estimated small-area
    # cells, so a racial estimate can occasionally exceed the corresponding
    # total at block scale. District and regional aggregation greatly reduces
    # this noise; rejecting such source rows would silently bias the totals.
    return cvap, african_american


def _regional_location_quotient(
    *,
    intersection_cvap: float,
    intersection_african_american_cvap: float,
    region_cvap: float,
    region_african_american_cvap: float,
) -> float:
    """Return the intersection-to-region African American CVAP location quotient."""
    if intersection_cvap <= 0 or region_cvap <= 0 or region_african_american_cvap <= 0:
        return float("nan")
    return (intersection_african_american_cvap / intersection_cvap) / (
        region_african_american_cvap / region_cvap
    )


def _normalize_code(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(5) if text else ""


def _normalize_block_id(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(15) if text else ""


def _mapping(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        raw = value
    elif isinstance(value, str) and value.strip():
        decoded = json.loads(value)
        if not isinstance(decoded, dict):
            raise ValueError(f"Expected a region dictionary, found {type(decoded)}")
        raw = decoded
    elif value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    else:
        raise ValueError(f"Expected a region dictionary, found {type(value)}")
    return {_normalize_code(key): float(amount) for key, amount in raw.items()}


def read_region_manifest(path: str | Path, geography: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": str})
    required = {"code", "name", "role", "scope"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{geography} manifest is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["code"] = frame["code"].map(_normalize_code)
    if frame["code"].duplicated().any():
        raise ValueError(f"{geography} manifest contains duplicate region codes")
    frame["geography"] = geography
    return frame.reset_index(drop=True)


def read_region_graph(
    graph_path: str | Path,
    manifest_path: str | Path,
    geography: str,
) -> tuple[pd.DataFrame, RegionData]:
    graph = read_graph_frame(graph_path).copy()
    manifest = read_region_manifest(manifest_path, geography)
    total_field = f"{geography}_CVAP_TOT23"
    black_field = f"{geography}_CVAP_BLK23"
    required = {"id", "NAME", "COUNTY", geography, total_field, black_field}
    missing = required.difference(graph.columns)
    if missing:
        raise ValueError(f"{geography} graph is missing columns: {sorted(missing)}")

    graph["id"] = pd.to_numeric(graph["id"], errors="raise").astype(int)
    codes = manifest["code"].tolist()
    regional_population = np.zeros((len(graph), len(codes)), dtype=float)
    regional_cvap = np.zeros((len(graph), len(codes)), dtype=float)
    regional_black = np.zeros_like(regional_cvap)
    for node_position, (population_raw, total_raw, black_raw) in enumerate(
        zip(graph[geography], graph[total_field], graph[black_field], strict=True)
    ):
        population_map = _mapping(population_raw)
        total_map = _mapping(total_raw)
        black_map = _mapping(black_raw)
        for region_position, code in enumerate(codes):
            regional_population[node_position, region_position] = population_map.get(code, 0.0)
            regional_cvap[node_position, region_position] = total_map.get(code, 0.0)
            regional_black[node_position, region_position] = black_map.get(code, 0.0)

    if np.any(regional_black.sum(axis=0) <= 0):
        empty = manifest.loc[regional_black.sum(axis=0) <= 0, "code"].tolist()
        raise ValueError(
            f"{geography} graph has no African American CVAP for regions: {empty}"
        )

    data = RegionData(
        geography=geography,
        manifest=manifest,
        node_ids=graph["id"].to_numpy(int),
        regional_population=regional_population,
        regional_cvap=regional_cvap,
        regional_black_cvap=regional_black,
    )
    return graph, data


def verify_graph_alignment(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    left = reference[["id", "NAME", "COUNTY"]].reset_index(drop=True)
    right = candidate[["id", "NAME", "COUNTY"]].reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("CUA and MSA graphs do not have identical node order and identity")


def load_node_demographics(
    graph: pd.DataFrame,
    *,
    blocks_path: str | Path,
    cvap_path: str | Path,
    cvap_total_col: str,
    black_cvap_col: str,
) -> tuple[NodeDemographics, pd.DataFrame]:
    """Aggregate block CVAP to the exact graph nodes used by ensemble assignments."""
    blocks = gpd.read_file(blocks_path, ignore_geometry=True)
    required = {"GEOID20", "NAME", "ENACTED_UPPER", "ENACTED_LOWER"}
    missing = required.difference(blocks.columns)
    if missing:
        raise ValueError(f"Regional block file is missing columns: {sorted(missing)}")
    blocks = blocks.copy()
    blocks["GEOID20"] = blocks["GEOID20"].map(_normalize_block_id)
    blocks["graph_name"] = (
        blocks["NAME"].astype(str)
        + "__SD"
        + pd.to_numeric(blocks["ENACTED_UPPER"], errors="raise").astype(int).astype(str)
        + "__HD"
        + pd.to_numeric(blocks["ENACTED_LOWER"], errors="raise").astype(int).astype(str)
    )

    cvap = read_cvap_file(cvap_path)
    required_cvap = {"GEOID20", cvap_total_col, black_cvap_col}
    missing_cvap = required_cvap.difference(cvap.columns)
    if missing_cvap:
        raise ValueError(f"CVAP file is missing columns: {sorted(missing_cvap)}")
    full_blocks = blocks[["GEOID20", "graph_name"]].merge(
        cvap[["GEOID20", cvap_total_col, black_cvap_col]],
        on="GEOID20",
        how="left",
        validate="one_to_one",
    )
    full_blocks[[cvap_total_col, black_cvap_col]] = full_blocks[
        [cvap_total_col, black_cvap_col]
    ].fillna(0.0)
    full_blocks = full_blocks.rename(
        columns={
            cvap_total_col: "district_cvap",
            black_cvap_col: "district_african_american_cvap",
        }
    )

    grouped = (
        full_blocks.groupby("graph_name", as_index=False)[
            ["district_cvap", "district_african_american_cvap"]
        ]
        .sum()
        .set_index("graph_name")
    )
    missing_nodes = set(graph["NAME"]).difference(grouped.index)
    if missing_nodes:
        examples = sorted(missing_nodes)[:3]
        raise ValueError(f"Block demographics omit {len(missing_nodes)} graph nodes: {examples}")
    aligned = grouped.reindex(graph["NAME"])
    demographics = NodeDemographics(
        node_ids=graph["id"].to_numpy(int),
        cvap=aligned["district_cvap"].to_numpy(float),
        african_american_cvap=aligned["district_african_american_cvap"].to_numpy(float),
    )
    _validate_node_demographics(demographics, graph["id"].to_numpy(int))
    return demographics, full_blocks[
        ["GEOID20", "district_cvap", "district_african_american_cvap"]
    ]


def rank_assignment(
    assignment: pd.Series,
    region_data: RegionData,
    *,
    node_demographics: NodeDemographics,
    n_districts: int,
    source: str,
    chain_id: str = "",
    record_index: int | str = "",
    proposal: Any = "",
    minimum_region_cvap_share: float = DEFAULT_MINIMUM_REGION_CVAP_SHARE,
) -> RankedPlan:
    if not 0 <= minimum_region_cvap_share < 1:
        raise ValueError("minimum_region_cvap_share must lie in [0, 1)")
    labels = assignment.reindex(region_data.node_ids)
    if labels.isna().any():
        raise ValueError(f"Assignment omits {int(labels.isna().sum())} graph nodes")
    labels = labels.astype(int).to_numpy()
    districts, inverse = np.unique(labels, return_inverse=True)
    if len(districts) != n_districts:
        raise ValueError(f"Expected {n_districts} districts, found {len(districts)}")
    node_cvap, node_african_american_cvap = _validate_node_demographics(
        node_demographics,
        region_data.node_ids,
    )
    district_cvap = np.bincount(
        inverse,
        weights=node_cvap,
        minlength=n_districts,
    )
    district_african_american_cvap = np.bincount(
        inverse,
        weights=node_african_american_cvap,
        minlength=n_districts,
    )

    n_regions = len(region_data.manifest)
    black_fraction = np.zeros((n_regions, n_districts), dtype=float)
    touching = np.zeros((n_regions, n_districts), dtype=bool)
    rows: list[dict[str, Any]] = []

    for region_position, region in region_data.manifest.iterrows():
        population_by_district = np.bincount(
            inverse,
            weights=region_data.regional_population[:, region_position],
            minlength=n_districts,
        )
        total_by_district = np.bincount(
            inverse,
            weights=region_data.regional_cvap[:, region_position],
            minlength=n_districts,
        )
        black_by_district = np.bincount(
            inverse,
            weights=region_data.regional_black_cvap[:, region_position],
            minlength=n_districts,
        )
        region_total = float(total_by_district.sum())
        region_black_total = float(black_by_district.sum())
        region_population = float(population_by_district.sum())
        if region_population <= 0 or region_total <= 0 or region_black_total <= 0:
            raise ValueError(f"No regional CVAP for {region_data.geography} {region['code']}")

        region_cvap_share = total_by_district / region_total
        location_quotient = np.array(
            [
                _regional_location_quotient(
                    intersection_cvap=float(total),
                    intersection_african_american_cvap=float(black),
                    region_cvap=region_total,
                    region_african_american_cvap=region_black_total,
                )
                for total, black in zip(total_by_district, black_by_district, strict=True)
            ],
            dtype=float,
        )
        substantive = (population_by_district > 0) & (
            region_cvap_share >= minimum_region_cvap_share
        )
        substantive_positions = np.flatnonzero(substantive)
        substantive_order = substantive_positions[
            np.lexsort(
                (
                    districts[substantive_positions],
                    location_quotient[substantive_positions],
                )
            )
        ]
        rank_by_position = {
            int(position): rank
            for rank, position in enumerate(substantive_order, start=1)
        }
        # Retain the legacy most-to-least regional-AA ordering in the compact
        # arrays; tidy rows now carry the explicit LQ rank used by the figure.
        order = np.lexsort(
            (districts, -total_by_district, -population_by_district, -black_by_district)
        )
        touching_count = int((population_by_district > 0).sum())
        substantive_count = int(substantive.sum())
        for descending_rank, district_position in enumerate(order, start=1):
            population = float(population_by_district[district_position])
            total = float(total_by_district[district_position])
            black = float(black_by_district[district_position])
            black_share = black / region_black_total
            total_share = float(region_cvap_share[district_position])
            intersection_black_share = black / total if total > 0 else 0.0
            black_fraction[region_position, descending_rank - 1] = black_share
            touching[region_position, descending_rank - 1] = population > 0
            if population > 0:
                full_total = float(district_cvap[district_position])
                full_black = float(district_african_american_cvap[district_position])
                is_substantive = bool(substantive[district_position])
                rows.append(
                    {
                        "geography": region_data.geography,
                        "region_code": region["code"],
                        "region_name": region["name"],
                        "role": region["role"],
                        "scope": region["scope"],
                        "source": source,
                        "chain_id": chain_id,
                        "record_index": record_index,
                        "proposal": proposal,
                        "n_touching_districts": touching_count,
                        "n_substantive_districts": substantive_count,
                        "substantive_intersection": is_substantive,
                        "regional_rank": rank_by_position.get(int(district_position), 0),
                        "rank": rank_by_position.get(int(district_position), 0),
                        "district": int(districts[district_position]),
                        "regional_population": population,
                        "regional_cvap": total,
                        "regional_african_american_cvap": black,
                        "fraction_region_population": population / region_population,
                        "fraction_region_cvap": total_share,
                        "fraction_region_african_american_cvap": black_share,
                        "intersection_african_american_cvap_share": intersection_black_share,
                        "regional_location_quotient": float(
                            location_quotient[district_position]
                        ),
                        "district_cvap": full_total,
                        "district_african_american_cvap": full_black,
                        "district_african_american_cvap_share": (
                            full_black / full_total if full_total > 0 else float("nan")
                        ),
                    }
                )

        if not np.isclose(black_fraction[region_position].sum(), 1.0, atol=1e-10):
            raise AssertionError(
                f"Rank shares do not sum to one for {region_data.geography} {region['code']}"
            )

    return RankedPlan(black_fraction=black_fraction, touching=touching, rows=rows)


def build_ensemble_rankings(
    chain_files: list[str | Path],
    graph: pd.DataFrame,
    region_data: dict[str, RegionData],
    *,
    node_demographics: NodeDemographics,
    atlas_key_col: str,
    node_id_col: str,
    county_col: str,
    n_districts: int,
    max_districts_per_county: int,
    filter_county_compliant: bool,
    max_maps_per_chain: int | None,
    minimum_region_cvap_share: float = DEFAULT_MINIMUM_REGION_CVAP_SHARE,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    matrices: dict[str, list[np.ndarray]] = {key: [] for key in region_data}
    touching: dict[str, list[np.ndarray]] = {key: [] for key in region_data}
    rows: list[dict[str, Any]] = []
    compliance_rows: list[dict[str, Any]] = []
    node_lookup = build_node_key_lookup(graph, key_col=atlas_key_col, node_id_col=node_id_col)

    for chain_file in chain_files:
        chain_id = chain_id_from_path(chain_file)
        maps_seen = used = skipped_noncompliant = failed = 0
        action = "Analyzing shared assignments" if configured_sweep() else "Reading chain"
        print(f"{action}: {chain_file}")
        for record_index, record in enumerate(iter_map_records(chain_file)):
            if max_maps_per_chain is not None and maps_seen >= max_maps_per_chain:
                break
            maps_seen += 1
            try:
                assignment = atlasio_districting_to_series(
                    record["districting"],
                    graph,
                    key_col=atlas_key_col,
                    node_id_col=node_id_col,
                    node_lookup=node_lookup,
                )
                compliance = county_compliance_summary_for_assignment(
                    assignment,
                    graph,
                    node_id_col=node_id_col,
                    county_col=county_col,
                    max_districts_per_county=max_districts_per_county,
                )
                if filter_county_compliant and not compliance["compliant"]:
                    skipped_noncompliant += 1
                    continue

                for geography, data in region_data.items():
                    ranked = rank_assignment(
                        assignment,
                        data,
                        node_demographics=node_demographics,
                        n_districts=n_districts,
                        source=ENSEMBLE_LABEL,
                        chain_id=chain_id,
                        record_index=record_index,
                        proposal=record.get("name", record_index),
                        minimum_region_cvap_share=minimum_region_cvap_share,
                    )
                    matrices[geography].append(ranked.black_fraction)
                    touching[geography].append(ranked.touching)
                    rows.extend(ranked.rows)
                used += 1
            except Exception as exc:
                failed += 1
                if failed <= 5:
                    print(f"  FAILED record {record_index}: {type(exc).__name__}: {exc}")

        compliance_rows.append(
            {
                "chain_id": chain_id,
                "chain_path": str(chain_file),
                "maps_seen": maps_seen,
                "used_compliant_maps": used,
                "skipped_noncompliant": skipped_noncompliant,
                "failed_maps": failed,
                "filter_county_compliant": filter_county_compliant,
                "max_districts_per_county": max_districts_per_county,
            }
        )
        print(
            f"  maps seen={maps_seen:,}; used={used:,}; "
            f"noncompliant={skipped_noncompliant:,}; failed={failed:,}"
        )

    if not any(matrices.values()) or not next(iter(matrices.values())):
        raise RuntimeError("No ensemble maps were usable after filtering")
    return (
        {key: np.stack(value) for key, value in matrices.items()},
        {key: np.stack(value) for key, value in touching.items()},
        pd.DataFrame(rows),
        pd.DataFrame(compliance_rows),
    )


def load_block_regions(
    *,
    blocks_path: str | Path,
    cvap_path: str | Path,
    cbsa_path: str | Path,
    region_data: dict[str, RegionData],
    cvap_total_col: str,
    black_cvap_col: str,
) -> dict[str, pd.DataFrame]:
    blocks = gpd.read_file(blocks_path)
    required = {"GEOID20", "UACE20", "POP20", "geometry"}
    missing = required.difference(blocks.columns)
    if missing:
        raise ValueError(f"Regional block file is missing columns: {sorted(missing)}")
    blocks = blocks.copy()
    blocks["GEOID20"] = blocks["GEOID20"].map(_normalize_block_id)
    cvap = read_cvap_file(cvap_path)
    required_cvap = {"GEOID20", cvap_total_col, black_cvap_col}
    missing_cvap = required_cvap.difference(cvap.columns)
    if missing_cvap:
        raise ValueError(f"CVAP file is missing columns: {sorted(missing_cvap)}")
    blocks = blocks.merge(
        cvap[["GEOID20", cvap_total_col, black_cvap_col]],
        on="GEOID20",
        how="left",
        validate="one_to_one",
    )
    blocks[[cvap_total_col, black_cvap_col]] = blocks[
        [cvap_total_col, black_cvap_col]
    ].fillna(0.0)

    cua_codes = set(region_data["CUA"].manifest["code"])
    cua = blocks["UACE20"].map(_normalize_code)

    cbsa = gpd.read_file(cbsa_path)
    code_column = "GEOID" if "GEOID" in cbsa.columns else "CBSAFP"
    cbsa[code_column] = cbsa[code_column].map(_normalize_code)
    msa_codes = set(region_data["MSA"].manifest["code"])
    cbsa = cbsa.loc[cbsa[code_column].isin(msa_codes), [code_column, "geometry"]]
    if set(cbsa[code_column]) != msa_codes:
        raise ValueError("CBSA file does not contain every requested MSA")
    points = blocks[["geometry"]].copy()
    points["geometry"] = blocks.geometry.representative_point()
    joined = gpd.sjoin(points.to_crs(cbsa.crs), cbsa, how="left", predicate="within")
    if not joined.index.is_unique:
        raise ValueError("A block point matched more than one MSA")
    msa = joined[code_column].fillna("").reindex(blocks.index).map(_normalize_code)

    result: dict[str, pd.DataFrame] = {}
    for geography, codes in (("CUA", cua), ("MSA", msa)):
        allowed = set(region_data[geography].manifest["code"])
        out = pd.DataFrame(
            {
                "GEOID20": blocks["GEOID20"],
                "region_code": codes.where(codes.isin(allowed), ""),
                "regional_population": pd.to_numeric(blocks["POP20"], errors="coerce").fillna(0.0),
                "regional_cvap": blocks[cvap_total_col].astype(float),
                "regional_black_cvap": blocks[black_cvap_col].astype(float),
            }
        )
        result[geography] = out.loc[out["region_code"] != ""].reset_index(drop=True)

        graph_totals = region_data[geography].regional_black_cvap.sum(axis=0)
        block_totals = (
            result[geography]
            .groupby("region_code")["regional_black_cvap"]
            .sum()
            .reindex(region_data[geography].manifest["code"], fill_value=0)
            .to_numpy(float)
        )
        if not np.allclose(graph_totals, block_totals, atol=0.5):
            raise ValueError(
                f"{geography} graph and block African American-CVAP totals disagree"
            )
    return result


def rank_official_block_plan(
    block_plan: pd.DataFrame,
    block_regions: pd.DataFrame,
    region_data: RegionData,
    *,
    full_block_demographics: pd.DataFrame,
    source: str,
    n_districts: int,
    minimum_region_cvap_share: float = DEFAULT_MINIMUM_REGION_CVAP_SHARE,
) -> RankedPlan:
    if not 0 <= minimum_region_cvap_share < 1:
        raise ValueError("minimum_region_cvap_share must lie in [0, 1)")
    districts = np.sort(block_plan["DISTRICT"].unique())
    if len(districts) != n_districts:
        raise ValueError(f"{source}: expected {n_districts} districts, found {len(districts)}")
    merged = block_regions.merge(block_plan, on="GEOID20", how="left", validate="many_to_one")
    if merged["DISTRICT"].isna().any():
        missing = int(merged.loc[merged["DISTRICT"].isna(), "GEOID20"].nunique())
        raise ValueError(f"{source}: official plan omits {missing} regional blocks")
    merged["DISTRICT"] = merged["DISTRICT"].astype(int)
    grouped = (
        merged.groupby(["region_code", "DISTRICT"], as_index=False)[
            ["regional_population", "regional_cvap", "regional_black_cvap"]
        ]
        .sum()
    )
    full = full_block_demographics.merge(
        block_plan,
        on="GEOID20",
        how="left",
        validate="one_to_one",
    )
    if full["DISTRICT"].isna().any():
        missing = int(full.loc[full["DISTRICT"].isna(), "GEOID20"].nunique())
        raise ValueError(f"{source}: official plan omits {missing} demographic blocks")
    full["DISTRICT"] = full["DISTRICT"].astype(int)
    full_by_district = (
        full.groupby("DISTRICT")[
            ["district_cvap", "district_african_american_cvap"]
        ]
        .sum()
        .reindex(districts, fill_value=0)
    )

    n_regions = len(region_data.manifest)
    black_fraction = np.zeros((n_regions, n_districts), dtype=float)
    touching = np.zeros((n_regions, n_districts), dtype=bool)
    rows: list[dict[str, Any]] = []
    for region_position, region in region_data.manifest.iterrows():
        part = grouped.loc[grouped["region_code"] == region["code"]].set_index("DISTRICT")
        population = part["regional_population"].reindex(districts, fill_value=0).to_numpy(float)
        total = part["regional_cvap"].reindex(districts, fill_value=0).to_numpy(float)
        black = part["regional_black_cvap"].reindex(districts, fill_value=0).to_numpy(float)
        region_population = float(population.sum())
        region_total = float(total.sum())
        region_black_total = float(black.sum())
        region_cvap_share = total / region_total
        location_quotient = np.array(
            [
                _regional_location_quotient(
                    intersection_cvap=float(intersection_total),
                    intersection_african_american_cvap=float(intersection_black),
                    region_cvap=region_total,
                    region_african_american_cvap=region_black_total,
                )
                for intersection_total, intersection_black in zip(total, black, strict=True)
            ],
            dtype=float,
        )
        substantive = (population > 0) & (region_cvap_share >= minimum_region_cvap_share)
        substantive_positions = np.flatnonzero(substantive)
        substantive_order = substantive_positions[
            np.lexsort(
                (districts[substantive_positions], location_quotient[substantive_positions])
            )
        ]
        rank_by_position = {
            int(position): rank
            for rank, position in enumerate(substantive_order, start=1)
        }
        order = np.lexsort((districts, -total, -population, -black))
        touching_count = int((population > 0).sum())
        substantive_count = int(substantive.sum())
        for descending_rank, district_position in enumerate(order, start=1):
            district_population = float(population[district_position])
            district_total = float(total[district_position])
            district_black = float(black[district_position])
            share = district_black / region_black_total
            black_fraction[region_position, descending_rank - 1] = share
            touching[region_position, descending_rank - 1] = district_population > 0
            if district_population > 0:
                full_total = float(
                    full_by_district.iloc[district_position]["district_cvap"]
                )
                full_black = float(
                    full_by_district.iloc[district_position][
                        "district_african_american_cvap"
                    ]
                )
                is_substantive = bool(substantive[district_position])
                rows.append(
                    {
                        "geography": region_data.geography,
                        "region_code": region["code"],
                        "region_name": region["name"],
                        "role": region["role"],
                        "scope": region["scope"],
                        "source": source,
                        "chain_id": "",
                        "record_index": "",
                        "proposal": "",
                        "n_touching_districts": touching_count,
                        "n_substantive_districts": substantive_count,
                        "substantive_intersection": is_substantive,
                        "regional_rank": rank_by_position.get(int(district_position), 0),
                        "rank": rank_by_position.get(int(district_position), 0),
                        "district": int(districts[district_position]),
                        "regional_population": district_population,
                        "regional_cvap": district_total,
                        "regional_african_american_cvap": district_black,
                        "fraction_region_population": district_population / region_population,
                        "fraction_region_cvap": float(region_cvap_share[district_position]),
                        "fraction_region_african_american_cvap": share,
                        "intersection_african_american_cvap_share": (
                            district_black / district_total if district_total > 0 else 0.0
                        ),
                        "regional_location_quotient": float(
                            location_quotient[district_position]
                        ),
                        "district_cvap": full_total,
                        "district_african_american_cvap": full_black,
                        "district_african_american_cvap_share": (
                            full_black / full_total if full_total > 0 else float("nan")
                        ),
                    }
                )
        if not np.isclose(black_fraction[region_position].sum(), 1.0, atol=1e-10):
            raise AssertionError(f"Official shares do not sum to one for {region['code']}")
    return RankedPlan(black_fraction=black_fraction, touching=touching, rows=rows)


def build_official_rankings(
    plan_paths: dict[str, str | Path],
    block_regions: dict[str, pd.DataFrame],
    region_data: dict[str, RegionData],
    *,
    full_block_demographics: pd.DataFrame,
    n_districts: int,
    assignment_block_col: str | None,
    assignment_district_col: str | None,
    minimum_region_cvap_share: float = DEFAULT_MINIMUM_REGION_CVAP_SHARE,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    matrices = {geography: {} for geography in region_data}
    touching = {geography: {} for geography in region_data}
    rows: list[dict[str, Any]] = []
    for label, path in plan_paths.items():
        print(f"Reading official plan: {label}")
        plan = read_block_assignment_excel(
            path,
            block_col=assignment_block_col,
            district_col=assignment_district_col,
        )
        for geography, data in region_data.items():
            ranked = rank_official_block_plan(
                plan,
                block_regions[geography],
                data,
                full_block_demographics=full_block_demographics,
                source=label,
                n_districts=n_districts,
                minimum_region_cvap_share=minimum_region_cvap_share,
            )
            matrices[geography][label] = ranked.black_fraction
            touching[geography][label] = ranked.touching
            rows.extend(ranked.rows)
    return matrices, touching, pd.DataFrame(rows)


def _short_name(name: str) -> str:
    return str(name).split(",", 1)[0].split("-", 1)[0]


def _official_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=1.5,
            markersize=6.5,
            label=label,
        )
        for label, style in official_plan_styles(PLAN_STYLES)
    ]


def _weighted_quantile(
    values: np.ndarray,
    quantiles: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return weighted quantiles using centred empirical probability mass."""
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.ndim != 1 or values.size != weights.size:
        raise ValueError("values and weights must be equally sized one-dimensional arrays")
    if np.any((quantiles < 0) | (quantiles > 1)):
        raise ValueError("quantiles must lie in [0, 1]")
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return np.full(quantiles.shape, np.nan, dtype=float)
    values = values[finite]
    weights = weights[finite]
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    probabilities = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return np.interp(quantiles, probabilities, values, left=values[0], right=values[-1])


def _matched_interval(
    rows: pd.DataFrame,
    metric: str,
    k: int,
    rank: int,
) -> np.ndarray:
    """Summarize the same ranked district among ensemble plan-regions with k pieces."""
    matched = rows.loc[
        (rows["n_substantive_districts"].astype(int) == int(k))
        & (rows["regional_rank"].astype(int) == int(rank))
    ].copy()
    if matched.empty:
        return np.full(5, np.nan, dtype=float)
    values = matched[metric].to_numpy(float)
    weights = 1.0 / matched["n_substantive_districts"].to_numpy(float)
    return _weighted_quantile(values, np.array([0.05, 0.25, 0.50, 0.75, 0.95]), weights)


def _rank_offsets(k: int) -> np.ndarray:
    """Return small vertical offsets that expose coincident district markers."""
    if k <= 1:
        return np.zeros(1, dtype=float)
    return np.linspace(0.32, -0.32, k)


def _row_layout(
    manifest: pd.DataFrame,
    official_rows: pd.DataFrame,
) -> tuple[dict[tuple[str, str], float], list[float], list[str], list[float]]:
    """Lay out three official-plan rows per region with visible group gaps."""
    positions: dict[tuple[str, str], float] = {}
    ticks: list[float] = []
    labels: list[str] = []
    separators: list[float] = []
    cursor = 0.0
    for region_position, region in manifest.iterrows():
        if region_position:
            separators.append(cursor - 0.45)
            cursor += 0.65
        for source in PLAN_STYLES:
            plan_rows = official_rows.loc[
                (official_rows["region_code"] == region["code"])
                & (official_rows["source"] == source)
            ]
            k = int(plan_rows["n_substantive_districts"].iloc[0]) if not plan_rows.empty else 0
            y = -cursor
            positions[(str(region["code"]), source)] = y
            ticks.append(y)
            short_source = source.replace(" ('", " ’").replace("')", "")
            labels.append(f"{short_source}   k={k}")
            cursor += 1.0
    return positions, ticks, labels, separators


def _draw_reference_interval(ax, y: float, interval: np.ndarray) -> None:
    if not np.isfinite(interval).all():
        return
    q05, q25, median, q75, q95 = interval
    ax.plot([q05, q95], [y, y], color=DUKE_NAVY_LIGHT, linewidth=1.2, zorder=1)
    ax.plot([q25, q75], [y, y], color=DUKE_NAVY, linewidth=3.2, alpha=0.58, zorder=2)
    ax.plot(
        [median],
        [y],
        marker="|",
        markersize=7,
        markeredgewidth=1.1,
        color=DUKE_NAVY,
        zorder=3,
    )


def plot_geography(
    *,
    geography: str,
    region_data: RegionData,
    rows: pd.DataFrame,
    output: str | Path,
) -> Path:
    apply_figure_style()
    required = {
        "geography",
        "region_code",
        "source",
        "substantive_intersection",
        "regional_rank",
        "n_substantive_districts",
        "regional_location_quotient",
        "district_african_american_cvap_share",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Plot rows are missing columns: {sorted(missing)}")
    plotted = rows.loc[
        (rows["geography"] == geography) & rows["substantive_intersection"].astype(bool)
    ].copy()
    if plotted.empty:
        raise ValueError(f"No substantive {geography} intersections to plot")
    ensemble_rows = plotted.loc[plotted["source"] == ENSEMBLE_LABEL]
    official_rows = plotted.loc[plotted["source"].isin(PLAN_STYLES)]
    positions, y_ticks, y_labels, separators = _row_layout(
        region_data.manifest,
        official_rows,
    )
    fig, (lq_ax, share_ax) = plt.subplots(
        1,
        2,
        figsize=(7.25, 8.7),
        sharey=True,
        gridspec_kw={"left": 0.285, "right": 0.985, "bottom": 0.065, "top": 0.90, "wspace": 0.12},
    )
    metrics = (
        (
            lq_ax,
            "regional_location_quotient",
            "Concentration relative to the metro",
        ),
        (
            share_ax,
            "district_african_american_cvap_share",
            "African American share of district CVAP",
        ),
    )
    for _, region in region_data.manifest.iterrows():
        region_code = str(region["code"])
        region_ensemble = ensemble_rows.loc[ensemble_rows["region_code"] == region_code]
        for source, style in official_plan_styles(PLAN_STYLES):
            plan_rows = official_rows.loc[
                (official_rows["region_code"] == region_code)
                & (official_rows["source"] == source)
            ].sort_values("regional_rank")
            if plan_rows.empty:
                continue
            k = int(plan_rows["n_substantive_districts"].iloc[0])
            if len(plan_rows) != k:
                raise ValueError(
                    f"{region['name']} {source}: expected {k} substantive rows, "
                    f"found {len(plan_rows)}"
                )
            y = positions[(region_code, source)]
            rank_offsets = _rank_offsets(k)
            for ax, metric, _ in metrics:
                values = plan_rows[metric].to_numpy(float)
                point_y = y + rank_offsets
                for rank, reference_y in zip(
                    plan_rows["regional_rank"].astype(int),
                    point_y,
                    strict=True,
                ):
                    _draw_reference_interval(
                        ax,
                        reference_y,
                        _matched_interval(region_ensemble, metric, k, int(rank)),
                    )
                ax.scatter(
                    values,
                    point_y,
                    s=52,
                    marker=style["marker"],
                    facecolors="white",
                    edgecolors=style["color"],
                    linewidths=1.45,
                    zorder=5,
                )
                for value, point_position, rank in zip(
                    values,
                    point_y,
                    plan_rows["regional_rank"].astype(int),
                    strict=True,
                ):
                    ax.text(
                        value,
                        point_position,
                        str(rank),
                        color=style["color"],
                        fontsize=5.5,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        zorder=6,
                    ).set_gid(DATA_LABEL_GID)

    lower = min(y_ticks) - 0.55
    upper = max(y_ticks) + 0.55
    for ax, _, xlabel in metrics:
        ax.set_ylim(lower, upper)
        ax.set_xlabel(xlabel, labelpad=6)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.55, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", left=False)
        for separator in separators:
            ax.axhline(separator, color="#E4E7EA", linewidth=0.55, zorder=0)
    lq_ax.set_xlim(0.0, 2.55)
    lq_ax.set_xticks(np.arange(0.0, 2.51, 0.5))
    lq_ax.axvline(1.0, color="#555555", linewidth=0.8, linestyle=(0, (4, 3)), zorder=0)
    lq_ax.text(
        1.0,
        upper + 0.18,
        "same as metro",
        color="#555555",
        fontsize=7,
        ha="center",
        va="bottom",
        clip_on=False,
    )
    share_ax.set_xlim(0.0, 0.50)
    share_ax.set_xticks(np.arange(0.0, 0.51, 0.10))
    share_ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    lq_ax.set_yticks(y_ticks, labels=y_labels)
    share_ax.tick_params(axis="y", labelleft=False)

    for region_position, (_, region) in enumerate(region_data.manifest.iterrows()):
        first_y = positions[(str(region["code"]), next(iter(PLAN_STYLES)))]
        suffix = "†" if str(region["role"]).lower() == "supplementary" else ""
        scope = " (Ohio portion)" if str(region["scope"]).lower() == "ohio portion" else ""
        lq_ax.text(
            -0.64,
            first_y + 0.50,
            f"{_short_name(region['name'])}{suffix}{scope}",
            transform=lq_ax.get_yaxis_transform(),
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=DUKE_NAVY_LIGHT,
            linewidth=1.2,
            label="Same-k, same-rank ensemble: middle 90%",
        ),
        Line2D(
            [0],
            [0],
            color=DUKE_NAVY,
            linewidth=5,
            alpha=0.58,
            label="Same-k, same-rank ensemble: middle 50%",
        ),
        *_official_handles(),
    ]
    figure_legend(
        fig,
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.57, 0.99),
        ncol=3,
        handletextpad=0.5,
        columnspacing=1.1,
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    destination = save_figure(fig, output, dpi=300)
    plt.close(fig)
    print(f"Wrote {destination}")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot aligned district-level CUA and MSA comparisons of African "
            "American CVAP location quotients and full-district CVAP share."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--chain-files", nargs="+", required=True)
    parser.add_argument("--cua-graph-json", required=True)
    parser.add_argument("--msa-graph-json", required=True)
    parser.add_argument("--cua-manifest", required=True)
    parser.add_argument("--msa-manifest", required=True)
    parser.add_argument("--region-blocks", required=True)
    parser.add_argument("--cbsa-file", required=True)
    parser.add_argument("--cvap-file", required=True)
    parser.add_argument("--enacted-baf", required=True)
    parser.add_argument("--overturned-baf", required=True)
    parser.add_argument("--remedial-baf", required=True)
    parser.add_argument("--output-cua", required=True)
    parser.add_argument("--output-msa", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--compliance-summary-csv", default=None)
    parser.add_argument("--atlas-key-col", default="NAME")
    parser.add_argument("--node-id-col", default="id")
    parser.add_argument("--county-col", default="COUNTY")
    parser.add_argument("--cvap-total-col", default="CVAP_TOT23")
    parser.add_argument(
        "--african-american-cvap-col",
        dest="black_cvap_col",
        default="CVAP_BLK23",
    )
    parser.add_argument("--assignment-block-col", default=None)
    parser.add_argument("--assignment-district-col", default=None)
    parser.add_argument("--n-districts", type=int, default=15)
    parser.add_argument("--max-districts-per-county", type=int, default=2)
    parser.add_argument("--max-maps-per-chain", type=int, default=None)
    parser.add_argument(
        "--minimum-region-cvap-share",
        type=float,
        default=DEFAULT_MINIMUM_REGION_CVAP_SHARE,
        help=(
            "Minimum fraction of a region's CVAP assigned to a district for "
            "that intersection to enter the figure; all intersections remain in the audit CSV."
        ),
    )
    parser.add_argument("--no-filter-compliant", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cua_graph, cua_data = read_region_graph(
        args.cua_graph_json,
        args.cua_manifest,
        "CUA",
    )
    msa_graph, msa_data = read_region_graph(
        args.msa_graph_json,
        args.msa_manifest,
        "MSA",
    )
    verify_graph_alignment(cua_graph, msa_graph)
    region_data = {"CUA": cua_data, "MSA": msa_data}

    node_demographics, full_block_demographics = load_node_demographics(
        cua_graph,
        blocks_path=args.region_blocks,
        cvap_path=args.cvap_file,
        cvap_total_col=args.cvap_total_col,
        black_cvap_col=args.black_cvap_col,
    )

    ensemble, ensemble_touching, ensemble_rows, compliance = build_ensemble_rankings(
        args.chain_files,
        cua_graph,
        region_data,
        node_demographics=node_demographics,
        atlas_key_col=args.atlas_key_col,
        node_id_col=args.node_id_col,
        county_col=args.county_col,
        n_districts=args.n_districts,
        max_districts_per_county=args.max_districts_per_county,
        filter_county_compliant=not args.no_filter_compliant,
        max_maps_per_chain=args.max_maps_per_chain,
        minimum_region_cvap_share=args.minimum_region_cvap_share,
    )

    block_regions = load_block_regions(
        blocks_path=args.region_blocks,
        cvap_path=args.cvap_file,
        cbsa_path=args.cbsa_file,
        region_data=region_data,
        cvap_total_col=args.cvap_total_col,
        black_cvap_col=args.black_cvap_col,
    )
    plan_paths = {
        "Enacted ('25)": args.enacted_baf,
        "Overturned ('21)": args.overturned_baf,
        "Remedial ('22)": args.remedial_baf,
    }
    official, official_touching, official_rows = build_official_rankings(
        plan_paths,
        block_regions,
        region_data,
        full_block_demographics=full_block_demographics,
        n_districts=args.n_districts,
        assignment_block_col=args.assignment_block_col,
        assignment_district_col=args.assignment_district_col,
        minimum_region_cvap_share=args.minimum_region_cvap_share,
    )

    audit_rows = pd.concat([ensemble_rows, official_rows], ignore_index=True)
    plot_geography(
        geography="CUA",
        region_data=cua_data,
        rows=audit_rows,
        output=args.output_cua,
    )
    plot_geography(
        geography="MSA",
        region_data=msa_data,
        rows=audit_rows,
        output=args.output_msa,
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    audit_rows.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")

    compliance_path = (
        Path(args.compliance_summary_csv)
        if args.compliance_summary_csv
        else output_csv.with_name(f"{output_csv.stem}-compliance.csv")
    )
    compliance_path.parent.mkdir(parents=True, exist_ok=True)
    compliance.to_csv(compliance_path, index=False)
    print(f"Wrote {compliance_path}")


if __name__ == "__main__":
    main()
