"""Measure how plans change the partisan district context of city residents.

For each configured Census Place, a plan is represented by one coordinate per
overlapping graph unit: the Democratic vote share of the district containing
that unit.  Coordinates are weighted by the fraction of city population in the
unit.  The workflow compares every plan with the coordinate-wise ensemble mean
using a population-normalized signed displacement.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from atlas_posthoc_plus.paper import save_figure

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from atlas_posthoc_plus.workflows.municipal.district_rank_compliant import (
    atlasio_districting_to_series,
    compute_vote_share_by_district,
    county_compliance_summary,
    iter_atlasio_map_records,
    maybe_json_loads,
    normalize_block_id,
    normalize_token,
    read_block_assignment_excel,
    read_graph_nodes,
)
from atlas_posthoc_plus.workflows.municipal.geography import (
    _read_spatial_or_table,
    _resolve_column,
    add_municipality_arguments,
    geography_from_args,
)
from atlas_posthoc_plus.workflows.municipal.place_partisanship import (
    PLAN_COLORS,
    apply_figure_style,
    read_place_names,
)


DEFAULT_CITIES = (
    "Cleveland",
    "Columbus",
    "Cincinnati",
    "Toledo",
    "Akron",
    "Dayton",
)

PLAN_LINESTYLES = {
    "Enacted ('25)": "-",
    "Overturned ('21)": "--",
    "Remedial ('22)": ":",
}

# Accept old configuration values, but every mode now writes only the signed panel.
OUTPUT_MODES = ("signed", "duo", "both")


@dataclass(frozen=True)
class CityContext:
    municipality: str
    place_name: str
    population: float
    unit_ids: np.ndarray
    weights: np.ndarray


def add_population_adjusted_votes(
    graph_nodes: pd.DataFrame,
    *,
    orig_name_col: str = "ORIG_NAME",
    pop_col: str = "POP20",
    dem_col: str = "G20PREDEM",
    rep_col: str = "G20PREREP",
    dem_output_col: str = "_CITY_CONTEXT_DEM",
    rep_output_col: str = "_CITY_CONTEXT_REP",
) -> pd.DataFrame:
    """Allocate one copy of each original unit's votes across split nodes."""
    nodes = graph_nodes.copy()
    grouped = nodes.groupby(orig_name_col, as_index=False).agg(
        _group_population=(pop_col, "sum"),
        _group_dem=(dem_col, "first"),
        _group_rep=(rep_col, "first"),
        _group_size=(orig_name_col, "size"),
    )
    nodes = nodes.merge(grouped, on=orig_name_col, how="left", validate="many_to_one")
    population_share = np.where(
        nodes["_group_population"] > 0,
        nodes[pop_col] / nodes["_group_population"],
        1 / nodes["_group_size"],
    )
    nodes[dem_output_col] = nodes["_group_dem"] * population_share
    nodes[rep_output_col] = nodes["_group_rep"] * population_share
    return nodes.drop(
        columns=["_group_population", "_group_dem", "_group_rep", "_group_size"]
    )


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return text or "city"


def build_city_contexts(
    population_rows: pd.DataFrame,
    place_names: pd.DataFrame,
    requested_cities: Sequence[str],
) -> list[CityContext]:
    """Resolve requested city names/keys and normalize within-city weights."""
    required = {"unit_id", "municipality", "population"}
    missing = required - set(population_rows.columns)
    if missing:
        raise ValueError(f"population rows are missing columns {sorted(missing)}")

    names = place_names[["municipality", "place_name"]].copy()
    names["municipality"] = names["municipality"].astype(str)
    names["place_name"] = names["place_name"].astype(str)
    by_key = {value.casefold(): value for value in names["municipality"]}
    by_name: dict[str, list[str]] = {}
    for row in names.itertuples(index=False):
        by_name.setdefault(row.place_name.casefold(), []).append(row.municipality)

    rows = population_rows.copy()
    rows["municipality"] = rows["municipality"].astype(str)
    rows["population"] = pd.to_numeric(rows["population"], errors="raise")
    contexts: list[CityContext] = []
    seen: set[str] = set()
    for requested in requested_cities:
        token = str(requested).strip()
        key = by_key.get(token.casefold())
        if key is None:
            matches = by_name.get(token.casefold(), [])
            if len(matches) > 1:
                raise ValueError(
                    f"Place name {token!r} is ambiguous; use one of the keys {matches}"
                )
            key = matches[0] if matches else None
        if key is None:
            raise ValueError(f"Configured city {token!r} was not found in the Place layer")
        if key in seen:
            continue
        seen.add(key)

        city_rows = rows[(rows["municipality"] == key) & (rows["population"] > 0)].copy()
        city_rows = city_rows.groupby("unit_id", as_index=False)["population"].sum()
        if city_rows.empty:
            raise ValueError(f"Configured city {token!r} has no represented graph population")
        population = float(city_rows["population"].sum())
        weights = city_rows["population"].to_numpy(float) / population
        display = names.loc[names["municipality"] == key, "place_name"].iloc[0]
        contexts.append(
            CityContext(
                municipality=key,
                place_name=display,
                population=population,
                unit_ids=city_rows["unit_id"].to_numpy(),
                weights=weights,
            )
        )
    if not contexts:
        raise ValueError("At least one city must be configured")
    return contexts


def context_vectors_for_assignment(
    assignment: pd.Series,
    graph_nodes: pd.DataFrame,
    contexts: Sequence[CityContext],
    *,
    node_id_col: str = "id",
    dem_col: str = "G20PREDEM",
    rep_col: str = "G20PREREP",
) -> dict[str, np.ndarray]:
    """Return each city's district-Democratic-share vector for one plan."""
    district_share = compute_vote_share_by_district(
        assignment,
        graph_nodes,
        node_id_col=node_id_col,
        dem_col=dem_col,
        rep_col=rep_col,
    )
    output: dict[str, np.ndarray] = {}
    for city in contexts:
        districts = assignment.reindex(city.unit_ids)
        if districts.isna().any():
            missing = city.unit_ids[districts.isna().to_numpy()][:5].tolist()
            raise ValueError(
                f"Plan is missing {city.place_name} graph units; examples: {missing}"
            )
        vector = districts.map(district_share).to_numpy(float)
        if not np.isfinite(vector).all():
            raise ValueError(f"Plan produced undefined district vote shares for {city.place_name}")
        output[city.municipality] = vector
    return output


def build_ensemble_context_matrices(
    chain_files: Sequence[str | Path],
    graph_nodes: pd.DataFrame,
    contexts: Sequence[CityContext],
    *,
    atlas_key_col: str = "NAME",
    node_id_col: str = "id",
    county_col: str = "COUNTY",
    dem_col: str = "G20PREDEM",
    rep_col: str = "G20PREREP",
    max_districts_per_county: int = 2,
    max_maps_per_chain: int | None = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    """Collect city vectors for every compliant ensemble plan."""
    values = {city.municipality: [] for city in contexts}
    plan_rows: list[dict[str, Any]] = []
    compliance_rows: list[dict[str, Any]] = []

    for chain_file in map(Path, chain_files):
        match = re.search(r"(run\d+)", chain_file.stem)
        chain_id = match.group(1) if match else chain_file.stem
        seen = retained = invalid = 0
        double_crosses: list[int] = []
        worst_splits: list[int] = []
        print(f"Reading chain: {chain_file}")

        for record in iter_atlasio_map_records(chain_file):
            if max_maps_per_chain is not None and seen >= max_maps_per_chain:
                break
            seen += 1
            try:
                assignment = atlasio_districting_to_series(
                    maybe_json_loads(record["districting"]),
                    graph_nodes,
                    key_col=atlas_key_col,
                    node_id_col=node_id_col,
                )
                compliance = county_compliance_summary(
                    assignment,
                    graph_nodes,
                    node_id_col=node_id_col,
                    county_col=county_col,
                    max_districts_per_county=max_districts_per_county,
                )
                double_crosses.append(int(compliance["double_cross_count"]))
                worst_splits.append(int(compliance["worst_split"]))
                if not compliance["compliant"]:
                    continue

                vectors = context_vectors_for_assignment(
                    assignment,
                    graph_nodes,
                    contexts,
                    node_id_col=node_id_col,
                    dem_col=dem_col,
                    rep_col=rep_col,
                )
                ensemble_index = len(plan_rows)
                for city in contexts:
                    values[city.municipality].append(vectors[city.municipality])
                plan_rows.append(
                    {
                        "ensemble_index": ensemble_index,
                        "chain_id": chain_id,
                        "map_index": seen,
                        "record_name": record.get("name"),
                    }
                )
                retained += 1
            except Exception as exc:
                invalid += 1
                if invalid <= 5:
                    print(f"  FAILED map {seen}: {type(exc).__name__}: {exc}")

        compliance_rows.append(
            {
                "chain_id": chain_id,
                "chain_path": str(chain_file.resolve()),
                "maps_seen": seen,
                "compliant_maps": retained,
                "invalid_maps": invalid,
                "compliant_fraction": retained / max(seen, 1),
                "average_double_crosses": (
                    float(np.mean(double_crosses)) if double_crosses else np.nan
                ),
                "maximum_double_crosses": max(double_crosses, default=0),
                "worst_single_county_split": max(worst_splits, default=0),
            }
        )
        print(f"  maps_seen={seen}; compliant_maps={retained}; invalid_maps={invalid}")

    if not plan_rows:
        raise RuntimeError("No compliant ensemble plans were retained")
    matrices = {
        city.municipality: np.vstack(values[city.municipality]) for city in contexts
    }
    return matrices, pd.DataFrame(plan_rows), pd.DataFrame(compliance_rows)


def read_block_context_rows(
    block_metadata: str | Path,
    graph_nodes: pd.DataFrame,
    contexts: Sequence[CityContext],
    *,
    mapping_column: str = "PLACE",
    node_id_col: str = "id",
    graph_orig_name_col: str = "ORIG_NAME",
    graph_upper_col: str = "ENACTED_UPPER",
    graph_lower_col: str = "ENACTED_LOWER",
    block_col: str | None = None,
    block_name_col: str | None = None,
    block_upper_col: str | None = None,
    block_lower_col: str | None = None,
    block_pop_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map blocks to graph units and retain exact within-block city population."""
    blocks = _read_spatial_or_table(Path(block_metadata))
    block_col = _resolve_column(
        blocks.columns,
        [block_col, "GEOID20", "BLOCK", "GEOID", "BLOCKID"],
        label="block id",
    )
    block_name_col = _resolve_column(
        blocks.columns,
        [block_name_col, "NAME", "ORIG_NAME", "PRECINCT"],
        label="block precinct name",
    )
    block_upper_col = _resolve_column(
        blocks.columns,
        [block_upper_col, "ENACTED_UPPER", "ENACTED_UP"],
        label="enacted upper district",
    )
    block_lower_col = _resolve_column(
        blocks.columns,
        [block_lower_col, "ENACTED_LOWER", "ENACTED_LO"],
        label="enacted lower district",
    )
    block_pop_col = _resolve_column(
        blocks.columns,
        [block_pop_col, "POP20", "POPULATION", "POP"],
        label="block population",
    )
    if mapping_column not in blocks.columns:
        raise ValueError(
            f"Block metadata is missing Place population mapping {mapping_column!r}"
        )

    component_lookup: dict[tuple[str, str, str], int] = {}
    graph_orig_names: dict[int, str] = {}
    graph_columns = [
        node_id_col,
        graph_orig_name_col,
        graph_upper_col,
        graph_lower_col,
    ]
    for row in graph_nodes[graph_columns].itertuples(index=False, name=None):
        node_id, original_name, upper, lower = row
        node_id = int(node_id)
        original_name = normalize_token(original_name)
        graph_orig_names[node_id] = original_name
        for component in original_name.split(","):
            key = (
                normalize_token(component),
                normalize_token(upper),
                normalize_token(lower),
            )
            previous = component_lookup.get(key)
            if previous is not None and previous != node_id:
                raise ValueError(f"Graph component key {key!r} maps to multiple nodes")
            component_lookup[key] = node_id

    selected = blocks[
        [
            block_col,
            block_name_col,
            block_upper_col,
            block_lower_col,
            block_pop_col,
            mapping_column,
        ]
    ].copy()
    selected["GEOID20"] = selected[block_col].map(normalize_block_id)
    selected["block_population"] = pd.to_numeric(
        selected[block_pop_col], errors="coerce"
    ).fillna(0.0)
    def graph_unit_for_block(name: Any, upper: Any, lower: Any) -> int | None:
        upper = normalize_token(upper)
        lower = normalize_token(lower)
        matches = {
            component_lookup.get((normalize_token(component), upper, lower))
            for component in normalize_token(name).split(",")
        }
        matches.discard(None)
        return next(iter(matches)) if len(matches) == 1 else None

    selected["unit_id"] = [
        graph_unit_for_block(name, upper, lower)
        for name, upper, lower in zip(
            selected[block_name_col],
            selected[block_upper_col],
            selected[block_lower_col],
        )
    ]
    represented = selected["block_population"] > 0
    if selected.loc[represented, "unit_id"].isna().any():
        examples = selected.loc[
            represented & selected["unit_id"].isna(),
            [block_name_col, block_upper_col, block_lower_col],
        ].head()
        raise ValueError(
            "Positive-population blocks could not be mapped to graph units; "
            f"examples: {examples.to_dict('records')}"
        )
    selected = selected.dropna(subset=["unit_id"])
    selected["unit_id"] = selected["unit_id"].astype(int)
    selected["graph_orig_name"] = selected["unit_id"].map(graph_orig_names)
    block_rows = selected[
        ["GEOID20", "unit_id", "graph_orig_name", "block_population"]
    ].copy()

    requested = {city.municipality for city in contexts}
    city_rows: list[dict[str, Any]] = []
    for row in selected[
        ["GEOID20", "unit_id", mapping_column]
    ].itertuples(index=False, name=None):
        block_id, unit_id, raw_mapping = row
        mapping = maybe_json_loads(raw_mapping)
        if not isinstance(mapping, Mapping):
            raise ValueError(f"Block {block_id} has invalid {mapping_column} mapping")
        for municipality, population in mapping.items():
            municipality = str(municipality)
            value = float(population)
            if municipality in requested and value > 0:
                city_rows.append(
                    {
                        "GEOID20": block_id,
                        "unit_id": unit_id,
                        "municipality": municipality,
                        "city_population": value,
                    }
                )
    city_blocks = pd.DataFrame(city_rows)
    observed = city_blocks.groupby("municipality")["city_population"].sum()
    for city in contexts:
        if not np.isclose(observed.get(city.municipality, 0.0), city.population):
            raise ValueError(
                f"Block and graph populations disagree for {city.place_name}: "
                f"{observed.get(city.municipality, 0.0)} versus {city.population}"
            )
    return block_rows, city_blocks


def official_vote_share_by_district(
    block_plan: pd.DataFrame,
    block_rows: pd.DataFrame,
    graph_nodes: pd.DataFrame,
    *,
    graph_orig_name_col: str = "ORIG_NAME",
    dem_col: str = "G20PREDEM",
    rep_col: str = "G20PREREP",
) -> pd.Series:
    """Allocate one copy of graph votes to official districts through blocks."""
    original_votes = graph_nodes.groupby(graph_orig_name_col, as_index=False).agg(
        dem_votes=(dem_col, "first"),
        rep_votes=(rep_col, "first"),
    )
    original_votes = original_votes.rename(
        columns={graph_orig_name_col: "graph_orig_name"}
    )
    rows = block_rows.merge(block_plan, on="GEOID20", how="left", validate="one_to_one")
    if rows.loc[rows["block_population"] > 0, "DISTRICT"].isna().any():
        raise ValueError("Official plan omits positive-population graph blocks")
    totals = rows.groupby("graph_orig_name")["block_population"].transform("sum")
    counts = rows.groupby("graph_orig_name")["GEOID20"].transform("size")
    rows["allocation_share"] = np.where(
        totals > 0, rows["block_population"] / totals, 1 / counts
    )
    rows = rows.merge(original_votes, on="graph_orig_name", how="left")
    rows["dem_allocated"] = rows["dem_votes"] * rows["allocation_share"]
    rows["rep_allocated"] = rows["rep_votes"] * rows["allocation_share"]
    district_votes = rows.groupby("DISTRICT")[["dem_allocated", "rep_allocated"]].sum()
    denominator = district_votes["dem_allocated"] + district_votes["rep_allocated"]
    return district_votes["dem_allocated"] / denominator


def build_official_context_vectors(
    official_plans: Mapping[str, str | Path],
    block_rows: pd.DataFrame,
    city_block_rows: pd.DataFrame,
    graph_nodes: pd.DataFrame,
    contexts: Sequence[CityContext],
    *,
    dem_col: str = "G20PREDEM",
    rep_col: str = "G20PREREP",
    assignment_block_col: str | None = None,
    assignment_district_col: str | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    for label, path in official_plans.items():
        block_plan = read_block_assignment_excel(
            path,
            block_col=assignment_block_col,
            district_col=assignment_district_col,
        )
        district_share = official_vote_share_by_district(
            block_plan, block_rows, graph_nodes, dem_col=dem_col, rep_col=rep_col
        )
        assigned = city_block_rows.merge(
            block_plan, on="GEOID20", how="left", validate="many_to_one"
        )
        if assigned["DISTRICT"].isna().any():
            raise ValueError(f"Official plan {label} omits city-population blocks")
        assigned["district_dem_share"] = assigned["DISTRICT"].map(district_share)
        assigned["weighted_context"] = (
            assigned["city_population"] * assigned["district_dem_share"]
        )
        node_context = assigned.groupby(["municipality", "unit_id"]).agg(
            weighted_context=("weighted_context", "sum"),
            city_population=("city_population", "sum"),
        )
        node_context["district_dem_share"] = (
            node_context["weighted_context"] / node_context["city_population"]
        )
        output[label] = {}
        for city in contexts:
            values = node_context["district_dem_share"].reindex(
                pd.MultiIndex.from_product(
                    [[city.municipality], city.unit_ids],
                    names=["municipality", "unit_id"],
                )
            )
            if values.isna().any():
                raise ValueError(f"Official plan {label} omits {city.place_name} units")
            output[label][city.municipality] = values.to_numpy(float)
    return output


def normalized_context_shift(
    vectors: np.ndarray,
    centroid: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return the population-weighted signed deviation from the ensemble mean.

    ``weights`` must sum to one, so the result remains in Democratic-share
    units rather than population-times-share units.
    """
    vectors = np.asarray(vectors, dtype=float)
    centroid = np.asarray(centroid, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    if vectors.shape[1] != centroid.size or centroid.size != weights.size:
        raise ValueError("vectors, centroid, and city weights must share coordinates")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("city population weights must sum to one")
    delta = vectors - centroid[None, :]
    return np.sum(delta * weights[None, :], axis=1)


def _midrank_percentile(reference: np.ndarray, value: float) -> float:
    ties = np.isclose(reference, value, atol=1e-12)
    less = (reference < value) & ~ties
    return float((less.sum() + 0.5 * ties.sum()) / len(reference))


def summarize_context_metrics(
    contexts: Sequence[CityContext],
    ensemble_matrices: Mapping[str, np.ndarray],
    ensemble_plans: pd.DataFrame,
    official_vectors: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build ensemble, official, and centroid audit tables."""
    ensemble_tables: list[pd.DataFrame] = []
    official_rows: list[dict[str, Any]] = []
    centroid_rows: list[dict[str, Any]] = []

    for city in contexts:
        matrix = np.asarray(ensemble_matrices[city.municipality], dtype=float)
        centroid = matrix.mean(axis=0)
        ens_direction = normalized_context_shift(
            matrix, centroid, city.weights
        )
        table = ensemble_plans.copy()
        table.insert(0, "municipality", city.municipality)
        table.insert(1, "place_name", city.place_name)
        table["signed_displacement"] = ens_direction
        ensemble_tables.append(table)

        for unit_id, weight, mean_share in zip(
            city.unit_ids, city.weights, centroid, strict=True
        ):
            centroid_rows.append(
                {
                    "municipality": city.municipality,
                    "place_name": city.place_name,
                    "unit_id": unit_id,
                    "city_population": city.population,
                    "city_population_weight": weight,
                    "ensemble_mean_district_dem_share": mean_share,
                }
            )

        for label, plan_vectors in official_vectors.items():
            vector = np.asarray(plan_vectors[city.municipality], dtype=float)
            direction = normalized_context_shift(
                vector, centroid, city.weights
            )
            direction_value = float(direction[0])
            signed_percentile = _midrank_percentile(ens_direction, direction_value)
            if direction_value > 0:
                directional_tail = (1 + int((ens_direction >= direction_value).sum())) / (
                    len(ens_direction) + 1
                )
                interpretation = "packing"
            elif direction_value < 0:
                directional_tail = (1 + int((ens_direction <= direction_value).sum())) / (
                    len(ens_direction) + 1
                )
                interpretation = "cracking"
            else:
                directional_tail = 1.0
                interpretation = "neutral"
            official_rows.append(
                {
                    "municipality": city.municipality,
                    "place_name": city.place_name,
                    "city_population": city.population,
                    "graph_unit_count": len(city.unit_ids),
                    "official_plan": label,
                    "signed_displacement": direction_value,
                    "signed_ensemble_percentile": signed_percentile,
                    "directional_tail_probability": directional_tail,
                    "direction_interpretation": interpretation,
                    "ensemble_plan_count": len(ens_direction),
                }
            )

    return (
        pd.concat(ensemble_tables, ignore_index=True),
        pd.DataFrame(official_rows),
        pd.DataFrame(centroid_rows),
    )


def _official_plan_handles(official: pd.DataFrame) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=PLAN_COLORS.get(row.official_plan, "#555555"),
            linestyle=PLAN_LINESTYLES.get(row.official_plan, "-"),
            lw=1.8,
            label=row.official_plan,
        )
        for row in official.itertuples(index=False)
    ]


def _plot_official_plan_lines(
    axis: Any,
    official: pd.DataFrame,
    value_column: str,
) -> list[Line2D]:
    for row in official.itertuples(index=False):
        label = row.official_plan
        axis.axvline(
            getattr(row, value_column),
            color=PLAN_COLORS.get(label, "#555555"),
            linestyle=PLAN_LINESTYLES.get(label, "-"),
            lw=1.8,
            ymax=0.84,
        )
    return _official_plan_handles(official)


def _plot_signed_context_panel(
    axis: Any,
    ensemble_direction: np.ndarray,
    official: pd.DataFrame,
    *,
    bins: int,
    ensemble_weights: np.ndarray,
    show_ylabel: bool,
    show_legend: bool,
) -> None:
    """Draw the signed-displacement histogram and official comparisons."""
    direction_limit = max(
        float(np.max(np.abs(ensemble_direction))),
        float(official["signed_displacement"].abs().max()),
        0.01,
    ) * 1.08
    direction_edges = np.linspace(-direction_limit, direction_limit, bins + 1)
    axis.axvspan(-direction_limit, 0, color="#E69F00", alpha=0.08, lw=0)
    axis.axvspan(0, direction_limit, color="#0072B2", alpha=0.08, lw=0)
    axis.hist(
        ensemble_direction,
        bins=direction_edges,
        weights=ensemble_weights,
        color="#B8C4CE",
        edgecolor="white",
        linewidth=0.5,
    )
    axis.axvline(0, color="#555555", lw=0.8)
    axis.set_xlim(-direction_limit, direction_limit)
    axis.set_xlabel("Signed shift from ensemble mean district context")
    if show_ylabel:
        axis.set_ylabel("Fraction of compliant ensemble plans")
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.text(
        0.02,
        0.96,
        r"$\leftarrow$ more Republican" "\n(cracking)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#8A5A00",
    )
    axis.text(
        0.98,
        0.96,
        "more Democratic " r"$\rightarrow$" "\n(packing)",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#005A8D",
    )
    handles = _plot_official_plan_lines(axis, official, "signed_displacement")
    if show_legend:
        axis.legend(handles=handles, loc="center right")


def plot_city_context_signed(
    city: CityContext,
    ensemble: pd.DataFrame,
    official: pd.DataFrame,
    output: str | Path,
    *,
    bins: int = 35,
) -> Path:
    """Plot the signed partisan-context shift as a standalone city figure."""
    apply_figure_style()
    ensemble_direction = ensemble["signed_displacement"].to_numpy(float)
    ensemble_weights = np.full(len(ensemble_direction), 1 / len(ensemble_direction))

    figure, axis = plt.subplots(figsize=(5.0, 3.8))
    _plot_signed_context_panel(
        axis,
        ensemble_direction,
        official.copy(),
        bins=bins,
        ensemble_weights=ensemble_weights,
        show_ylabel=True,
        show_legend=True,
    )
    figure.subplots_adjust(top=0.97, bottom=0.16, left=0.15, right=0.97)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output = save_figure(figure, output)
    plt.close(figure)
    return output


def plot_city_context_outputs(
    city: CityContext,
    ensemble: pd.DataFrame,
    official: pd.DataFrame,
    output_dir: str | Path,
    *,
    output_mode: str = "signed",
    bins: int = 35,
) -> dict[str, Path]:
    """Write one signed city-context figure, including for legacy modes."""
    if output_mode not in OUTPUT_MODES:
        raise ValueError(
            f"output_mode must be one of {', '.join(OUTPUT_MODES)}, found {output_mode!r}"
        )
    output_dir = Path(output_dir)
    stem = _slug(city.place_name)
    return {
        "signed": plot_city_context_signed(
            city,
            ensemble,
            official,
            output_dir / f"{stem}-city-partisan-context.png",
            bins=bins,
        )
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare official and ensemble plans in each city's normalized partisan "
            "district-context vector space."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--graph-json", required=True)
    parser.add_argument("--chain-files", nargs="+", required=True)
    parser.add_argument("--block-metadata", required=True)
    parser.add_argument("--enacted-baf", required=True)
    parser.add_argument("--overturned-baf", required=True)
    parser.add_argument("--remedial-baf", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cities", nargs="+", default=list(DEFAULT_CITIES))
    parser.add_argument("--atlas-key-col", default="NAME")
    parser.add_argument("--node-id-col", default="id")
    parser.add_argument("--county-col", default="COUNTY")
    parser.add_argument("--pop-col", default="POP20")
    parser.add_argument("--dem-col", default="G20PREDEM")
    parser.add_argument("--rep-col", default="G20PREREP")
    parser.add_argument("--max-districts-per-county", type=int, default=2)
    parser.add_argument("--max-maps-per-chain", type=int, default=None)
    parser.add_argument("--assignment-block-col", default=None)
    parser.add_argument("--assignment-district-col", default=None)
    parser.add_argument("--block-col", default=None)
    parser.add_argument("--block-name-col", default=None)
    parser.add_argument("--block-upper-col", default=None)
    parser.add_argument("--block-lower-col", default=None)
    parser.add_argument("--block-pop-col", default=None)
    parser.add_argument("--place-name-column", default=None)
    parser.add_argument("--bins", type=int, default=35)
    parser.add_argument(
        "--output-mode",
        choices=OUTPUT_MODES,
        default="signed",
        help="Write only the signed panel; legacy duo/both values are aliases for signed",
    )
    add_municipality_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    geography = geography_from_args(args, graph_json=args.graph_json)
    if geography.mode != "PLACE":
        raise ValueError("municipal-city-partisan-context requires municipality.mode=PLACE")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Place graph and city population weights...")
    graph_nodes = read_graph_nodes(args.graph_json, geography)
    population_rows = geography.graph_population_rows(
        graph_nodes, node_id_col=args.node_id_col, pop_col=args.pop_col
    )
    names = read_place_names(geography, place_name_column=args.place_name_column)
    contexts = build_city_contexts(population_rows, names, args.cities)
    block_rows, city_block_rows = read_block_context_rows(
        args.block_metadata,
        graph_nodes,
        contexts,
        mapping_column=geography.mapping_column,
        node_id_col=args.node_id_col,
        block_col=args.block_col,
        block_name_col=args.block_name_col,
        block_upper_col=args.block_upper_col,
        block_lower_col=args.block_lower_col,
        block_pop_col=args.block_pop_col,
    )
    adjusted_dem_col = "_CITY_CONTEXT_DEM"
    adjusted_rep_col = "_CITY_CONTEXT_REP"
    graph_nodes = add_population_adjusted_votes(
        graph_nodes,
        pop_col=args.pop_col,
        dem_col=args.dem_col,
        rep_col=args.rep_col,
        dem_output_col=adjusted_dem_col,
        rep_output_col=adjusted_rep_col,
    )

    print("Computing compliant-ensemble city context vectors...")
    matrices, ensemble_plans, compliance = build_ensemble_context_matrices(
        args.chain_files,
        graph_nodes,
        contexts,
        atlas_key_col=args.atlas_key_col,
        node_id_col=args.node_id_col,
        county_col=args.county_col,
        dem_col=adjusted_dem_col,
        rep_col=adjusted_rep_col,
        max_districts_per_county=args.max_districts_per_county,
        max_maps_per_chain=args.max_maps_per_chain,
    )

    print("Computing official-plan city context vectors...")
    official_vectors = build_official_context_vectors(
        {
            "Enacted ('25)": args.enacted_baf,
            "Overturned ('21)": args.overturned_baf,
            "Remedial ('22)": args.remedial_baf,
        },
        block_rows,
        city_block_rows,
        graph_nodes,
        contexts,
        dem_col=args.dem_col,
        rep_col=args.rep_col,
        assignment_block_col=args.assignment_block_col,
        assignment_district_col=args.assignment_district_col,
    )

    ensemble, official, centroids = summarize_context_metrics(
        contexts, matrices, ensemble_plans, official_vectors
    )
    ensemble_path = output_dir / "city-partisan-context-ensemble.csv"
    official_path = output_dir / "city-partisan-context-official.csv"
    centroid_path = output_dir / "city-partisan-context-centroids.csv"
    compliance_path = output_dir / "city-partisan-context-ensemble-compliance.csv"
    ensemble.to_csv(ensemble_path, index=False)
    official.to_csv(official_path, index=False)
    centroids.to_csv(centroid_path, index=False)
    compliance.to_csv(compliance_path, index=False)
    geography.write_metadata(output_dir / "municipal-geography.json")

    for city in contexts:
        city_ensemble = ensemble[ensemble["municipality"] == city.municipality]
        city_official = official[official["municipality"] == city.municipality]
        outputs = plot_city_context_outputs(
            city,
            city_ensemble,
            city_official,
            output_dir,
            output_mode=args.output_mode,
            bins=args.bins,
        )
        for output in outputs.values():
            print(f"Wrote {output}")
    print(f"Wrote {ensemble_path}")
    print(f"Wrote {official_path}")
    print(f"Wrote {centroid_path}")
    print(f"Wrote {compliance_path}")


if __name__ == "__main__":
    main()
