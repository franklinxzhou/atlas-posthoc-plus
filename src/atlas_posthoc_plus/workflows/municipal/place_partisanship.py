"""Compare official Place fragmentation with place-specific ensemble distributions.

For each Census Place, this workflow fixes the place-wide two-party Democratic
vote share and computes the share of its population outside its largest district
under every compliant ensemble plan and each official plan.  The ensemble is
summarized separately for every Place, so a large city's official value is
compared with plans facing the same geography rather than with other cities.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from atlas_posthoc_plus.workflows.municipal.district_rank_compliant import (
    atlasio_districting_to_series,
    county_compliance_summary,
    iter_atlasio_map_records,
    maybe_json_loads,
    read_block_assignment_excel,
    read_graph_nodes,
)
from atlas_posthoc_plus.workflows.municipal.geography import (
    MunicipalGeography,
    _read_spatial_or_table,
    _resolve_column,
    add_municipality_arguments,
    geography_from_args,
)


PLAN_COLORS = {
    "Enacted ('25)": "#7A5195",
    "Overturned ('21)": "#0072B2",
    "Remedial ('22)": "#E69F00",
}


def apply_figure_style() -> None:
    """Apply the role-based publication style used by this workflow."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlelocation": "left",
            "axes.titleweight": "normal",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": False,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def compute_place_ousted_shares(
    assignment: pd.Series,
    population_rows: pd.DataFrame,
    *,
    require_complete_assignment: bool = False,
) -> pd.DataFrame:
    """Return one fragmentation record per Place for one districting.

    ``population_rows`` must contain ``unit_id``, ``municipality``, and
    ``population``.  Ousted share is one minus the population share in the
    Place's largest district.  Unsplit Places therefore receive zero.
    """
    required = {"unit_id", "municipality", "population"}
    missing_columns = required - set(population_rows.columns)
    if missing_columns:
        raise ValueError(f"population rows are missing columns {sorted(missing_columns)}")

    rows = population_rows[list(required)].copy()
    rows["population"] = pd.to_numeric(rows["population"], errors="raise")
    rows = rows[rows["population"] > 0]
    if require_complete_assignment:
        missing_units = set(rows["unit_id"]) - set(assignment.index)
        if missing_units:
            examples = sorted(missing_units, key=str)[:5]
            raise ValueError(
                f"{len(missing_units)} positive-population Place units are missing "
                f"district assignments; examples: {examples}"
            )

    assigned = rows.merge(
        assignment.rename("district").rename_axis("unit_id").reset_index(),
        on="unit_id",
        how="inner",
    )
    pieces = (
        assigned.groupby(["municipality", "district"], as_index=False)["population"]
        .sum()
        .rename(columns={"population": "piece_population"})
    )
    if pieces.empty:
        return pd.DataFrame(
            columns=[
                "municipality",
                "place_population",
                "n_districts",
                "largest_piece_population",
                "ousted_population",
                "ousted_population_share",
            ]
        )

    result = pieces.groupby("municipality").agg(
        place_population=("piece_population", "sum"),
        n_districts=("district", "nunique"),
        largest_piece_population=("piece_population", "max"),
    )
    result["ousted_population"] = (
        result["place_population"] - result["largest_piece_population"]
    )
    result["ousted_population_share"] = (
        result["ousted_population"] / result["place_population"]
    )
    return result.reset_index()


def compute_place_vote_shares(
    graph_nodes: pd.DataFrame,
    geography: MunicipalGeography,
    *,
    node_id_col: str = "id",
    pop_col: str = "POP20",
    dem_col: str = "G20PREDEM",
    rep_col: str = "G20PREREP",
) -> pd.DataFrame:
    """Population-allocate graph-node votes to Places and aggregate them.

    A graph node's votes are allocated to each represented Place in proportion
    to the Place population within that node.  This matches the population
    allocation available in the Place graph while avoiding double-counting the
    full precinct vote total for nodes that intersect more than one Place.
    """
    rows = geography.graph_population_rows(
        graph_nodes, node_id_col=node_id_col, pop_col=pop_col
    )
    node_values = graph_nodes[[node_id_col, pop_col, dem_col, rep_col]].rename(
        columns={node_id_col: "unit_id", pop_col: "node_population"}
    )
    allocated = rows.merge(node_values, on="unit_id", how="left", validate="many_to_one")
    allocated["node_population"] = pd.to_numeric(
        allocated["node_population"], errors="coerce"
    ).fillna(0.0)
    allocated["allocation_share"] = np.where(
        allocated["node_population"] > 0,
        allocated["population"] / allocated["node_population"],
        0.0,
    )
    if (allocated["allocation_share"] > 1 + 1e-9).any():
        raise ValueError("Place population exceeds graph-node population")

    for source, target in ((dem_col, "dem_votes"), (rep_col, "rep_votes")):
        allocated[target] = (
            pd.to_numeric(allocated[source], errors="coerce").fillna(0.0)
            * allocated["allocation_share"]
        )

    result = allocated.groupby("municipality", as_index=False).agg(
        place_population=("population", "sum"),
        dem_votes=("dem_votes", "sum"),
        rep_votes=("rep_votes", "sum"),
    )
    denominator = result["dem_votes"] + result["rep_votes"]
    result["dem_two_party_share"] = result["dem_votes"] / denominator.replace(0, np.nan)
    return result


def read_place_names(
    geography: MunicipalGeography,
    *,
    place_name_column: str | None = None,
) -> pd.DataFrame:
    """Read a key-to-display-name table from the configured Place layer."""
    if geography.mode != "PLACE" or geography.place_shapefile is None:
        raise ValueError("place names require PLACE mode and a Place layer")
    places = _read_spatial_or_table(geography.place_shapefile)
    key_col = geography._resolve_place_key_column(places)
    name_col = _resolve_column(
        places.columns,
        [place_name_column, "NAME", "NAME20", "NAMELSAD", key_col],
        label="Place name",
    )
    result = places[[key_col, name_col]].copy()
    result.columns = ["municipality", "place_name"]
    result["municipality"] = result["municipality"].astype(str).str.strip()
    result["place_name"] = result["place_name"].astype(str).str.strip()
    return result.drop_duplicates("municipality", keep="first")


def build_ensemble_matrix(
    chain_files: Sequence[str | Path],
    graph_nodes: pd.DataFrame,
    population_rows: pd.DataFrame,
    *,
    atlas_key_col: str = "NAME",
    node_id_col: str = "id",
    county_col: str = "COUNTY",
    max_districts_per_county: int = 2,
    max_maps_per_chain: int | None = None,
) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    """Build Place-by-plan ousted-share matrix for compliant ensemble maps."""
    place_ids = sorted(population_rows["municipality"].astype(str).unique())
    place_index = pd.Index(place_ids, name="municipality")
    values: list[np.ndarray] = []
    compliance_rows: list[dict[str, Any]] = []

    for chain_file in map(Path, chain_files):
        match = re.search(r"(run\d+)", chain_file.stem)
        chain_id = match.group(1) if match else chain_file.stem
        seen = retained = invalid = 0
        double_crosses: list[int] = []
        worst_splits: list[int] = []

        print(f"Reading chain: {chain_file}")
        for record in iter_atlasio_map_records(chain_file):
            seen += 1
            if max_maps_per_chain is not None and seen > max_maps_per_chain:
                break
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

                metrics = compute_place_ousted_shares(
                    assignment,
                    population_rows,
                    require_complete_assignment=True,
                ).set_index("municipality")
                aligned = metrics["ousted_population_share"].reindex(place_index)
                if aligned.isna().any():
                    missing = aligned[aligned.isna()].index.tolist()[:5]
                    raise ValueError(f"compliant plan omitted Places {missing}")
                values.append(aligned.to_numpy(float))
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

    if not values:
        raise RuntimeError("No compliant ensemble plans were retained")
    return place_ids, np.vstack(values), pd.DataFrame(compliance_rows)


def summarize_ensemble(place_ids: Sequence[str], matrix: np.ndarray) -> pd.DataFrame:
    """Calculate place-specific ensemble quantiles."""
    if matrix.ndim != 2 or matrix.shape[1] != len(place_ids):
        raise ValueError("ensemble matrix columns must match place_ids")
    quantiles = np.quantile(matrix, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)
    return pd.DataFrame(
        {
            "municipality": list(place_ids),
            "ensemble_q05": quantiles[0],
            "ensemble_q25": quantiles[1],
            "ensemble_median": quantiles[2],
            "ensemble_q75": quantiles[3],
            "ensemble_q95": quantiles[4],
            "ensemble_plan_count": matrix.shape[0],
        }
    )


def summarize_official_plans(
    official_plans: Mapping[str, str | Path],
    block_population_rows: pd.DataFrame,
    place_ids: Sequence[str],
    ensemble_matrix: np.ndarray,
    *,
    assignment_block_col: str | None = None,
    assignment_district_col: str | None = None,
) -> pd.DataFrame:
    """Compute official Place metrics, residuals, and ensemble percentiles."""
    place_index = pd.Index(place_ids, name="municipality")
    output: list[pd.DataFrame] = []
    for label, path in official_plans.items():
        block_plan = read_block_assignment_excel(
            path,
            block_col=assignment_block_col,
            district_col=assignment_district_col,
        )
        assignment = pd.Series(
            block_plan["DISTRICT"].to_numpy(),
            index=block_plan["GEOID20"].to_numpy(),
            dtype=int,
        )
        metrics = compute_place_ousted_shares(
            assignment,
            block_population_rows,
            require_complete_assignment=True,
        ).set_index("municipality")
        aligned = metrics.reindex(place_index)
        if aligned["ousted_population_share"].isna().any():
            missing = aligned[aligned["ousted_population_share"].isna()].index.tolist()[:5]
            raise ValueError(f"official plan {label} omitted Places {missing}")

        official = aligned["ousted_population_share"].to_numpy(float)
        tie_mask = np.isclose(ensemble_matrix, official[None, :], atol=1e-12)
        less = ((ensemble_matrix < official[None, :]) & ~tie_mask).sum(axis=0)
        ties = tie_mask.sum(axis=0)
        aligned = aligned.reset_index()
        aligned["official_plan"] = label
        aligned["official_ensemble_percentile"] = (
            less + 0.5 * ties
        ) / ensemble_matrix.shape[0]
        output.append(aligned)
    return pd.concat(output, ignore_index=True)


def build_analysis_table(
    official: pd.DataFrame,
    ensemble: pd.DataFrame,
    votes: pd.DataFrame,
    names: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge official, ensemble, vote, and display-name records."""
    result = official.merge(ensemble, on="municipality", how="left", validate="many_to_one")
    result = result.merge(
        votes[
            [
                "municipality",
                "dem_votes",
                "rep_votes",
                "dem_two_party_share",
            ]
        ],
        on="municipality",
        how="left",
        validate="many_to_one",
    )
    if names is not None:
        result = result.merge(names, on="municipality", how="left", validate="many_to_one")
    else:
        result["place_name"] = result["municipality"]
    result["place_name"] = result["place_name"].fillna(result["municipality"])
    result["official_minus_ensemble_median"] = (
        result["ousted_population_share"] - result["ensemble_median"]
    )
    columns = [
        "official_plan",
        "municipality",
        "place_name",
        "place_population",
        "dem_votes",
        "rep_votes",
        "dem_two_party_share",
        "n_districts",
        "largest_piece_population",
        "ousted_population",
        "ousted_population_share",
        "ensemble_q05",
        "ensemble_q25",
        "ensemble_median",
        "ensemble_q75",
        "ensemble_q95",
        "ensemble_plan_count",
        "official_ensemble_percentile",
        "official_minus_ensemble_median",
    ]
    return result[columns].sort_values(
        ["official_plan", "dem_two_party_share", "place_population"],
        na_position="last",
    )


def _plot_rows(summary: pd.DataFrame, minimum_population: float) -> pd.DataFrame:
    rows = summary[
        (summary["place_population"] >= minimum_population)
        & summary["dem_two_party_share"].notna()
    ].copy()
    if rows.empty:
        raise ValueError(
            f"No Places with votes meet minimum population {minimum_population:,.0f}"
        )
    return rows


def _point_sizes(population: pd.Series) -> np.ndarray:
    values = population.to_numpy(float)
    maximum = max(float(np.nanmax(values)), 1.0)
    return 14.0 + 54.0 * np.sqrt(values / maximum)


def _label_places(ax: plt.Axes, rows: pd.DataFrame, *, value_col: str, count: int) -> None:
    if count <= 0:
        return
    selected = rows.nlargest(count, "place_population").copy()
    extreme = rows.loc[[rows[value_col].abs().idxmax()]]
    selected = pd.concat([selected, extreme]).drop_duplicates("municipality")
    selected = selected.sort_values("dem_two_party_share").reset_index(drop=True)
    near_zero = 0.08 if value_col == "official_minus_ensemble_median" else 0.025
    for index, row in selected.iterrows():
        x = float(row["dem_two_party_share"])
        y = float(row[value_col])
        # Labels point into the panel, and near-zero cities are staggered upward
        # rather than alternating below the visible data range.
        right_side = x >= 0.68
        offset_x = -5 if right_side else 5
        offset_y = (
            7 + 10 * (index % 3)
            if abs(y) < near_zero
            else (8 if index % 2 == 0 else -10)
        )
        ax.annotate(
            str(row["place_name"]),
            (x, y),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=7,
            color="#3F3F3F",
            ha="right" if right_side else "left",
            va="bottom" if offset_y > 0 else "top",
            arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.5},
            annotation_clip=True,
        )


def plot_place_ousted_share_by_partisanship(
    summary: pd.DataFrame,
    output: str | Path,
    *,
    minimum_population: float = 25_000,
    label_count: int = 3,
) -> Path:
    """Plot official Place values against each Place's ensemble IQR."""
    apply_figure_style()
    rows = _plot_rows(summary, minimum_population)
    plans = list(dict.fromkeys(rows["official_plan"]))
    upper = min(
        1.0,
        max(
            float(rows[["ousted_population_share", "ensemble_q95"]].max().max())
            * 1.12,
            0.05,
        ),
    )
    figure, axes = plt.subplots(
        1,
        len(plans),
        figsize=(4.2 * len(plans), 4.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for index, (ax, plan) in enumerate(zip(axes[0], plans)):
        panel = rows[rows["official_plan"] == plan].copy()
        color = PLAN_COLORS.get(plan, "#6A3D9A")
        x = panel["dem_two_party_share"].to_numpy(float)
        ax.vlines(
            x,
            panel["ensemble_q25"],
            panel["ensemble_q75"],
            color="#B9B9B9",
            alpha=0.65,
            linewidth=1.0,
            zorder=1,
        )
        ax.scatter(
            x,
            panel["ensemble_median"],
            marker="_",
            s=42,
            linewidths=1.0,
            color="#5F5F5F",
            zorder=2,
        )
        ax.scatter(
            x,
            panel["ousted_population_share"],
            s=_point_sizes(panel["place_population"]),
            color=color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.82,
            zorder=3,
        )
        _label_places(
            ax,
            panel,
            value_col="ousted_population_share",
            count=label_count,
        )
        ax.set_title(plan)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.015 * upper, upper)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.margins(0.04)
        if index == 0:
            ax.set_ylabel("Place population outside its largest district")
            ax.legend(
                handles=[
                    Line2D([0], [0], color="#B9B9B9", lw=2, label="Ensemble middle 50%"),
                    Line2D(
                        [0],
                        [0],
                        color="#5F5F5F",
                        marker="_",
                        linestyle="None",
                        markersize=8,
                        label="Ensemble median",
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="white",
                        markerfacecolor="#555555",
                        markeredgecolor="white",
                        markersize=6,
                        label="Official plan",
                    ),
                ],
                loc="upper left",
            )
        ax.set_xlabel("Place Democratic two-party vote share")

    figure.suptitle(
        "Official plans compared with each Place's ensemble distribution",
        x=0.5,
        ha="center",
        fontsize=10,
    )
    figure.subplots_adjust(top=0.84, bottom=0.13, left=0.08, right=0.98, wspace=0.08)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    return output


def plot_place_ousted_share_residual_by_partisanship(
    summary: pd.DataFrame,
    output: str | Path,
    *,
    minimum_population: float = 25_000,
    label_count: int = 3,
) -> Path:
    """Plot official-minus-ensemble-median Place fragmentation."""
    apply_figure_style()
    rows = _plot_rows(summary, minimum_population)
    plans = list(dict.fromkeys(rows["official_plan"]))
    limit = max(
        float(rows["official_minus_ensemble_median"].abs().max()) * 1.12,
        0.05,
    )
    figure, axes = plt.subplots(
        1,
        len(plans),
        figsize=(4.2 * len(plans), 4.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for index, (ax, plan) in enumerate(zip(axes[0], plans)):
        panel = rows[rows["official_plan"] == plan].copy()
        color = PLAN_COLORS.get(plan, "#6A3D9A")
        ax.axhline(0, color="#555555", lw=0.8, zorder=1)
        ax.scatter(
            panel["dem_two_party_share"],
            panel["official_minus_ensemble_median"],
            s=_point_sizes(panel["place_population"]),
            color=color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.82,
            zorder=2,
        )
        _label_places(
            ax,
            panel,
            value_col="official_minus_ensemble_median",
            count=label_count,
        )
        ax.set_title(plan)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-limit, limit)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.margins(x=0.04)
        ax.set_xlabel("Place Democratic two-party vote share")
        if index == 0:
            ax.set_ylabel("Official minus ensemble-median ousted share")

    figure.suptitle(
        "Official Place fragmentation relative to its ensemble expectation",
        x=0.5,
        ha="center",
        fontsize=10,
    )
    figure.subplots_adjust(top=0.84, bottom=0.13, left=0.08, right=0.98, wspace=0.08)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare official Place ousted-population shares with Place-specific "
            "compliant-ensemble distributions and Democratic vote share."
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
    parser.add_argument("--atlas-key-col", default="NAME")
    parser.add_argument("--node-id-col", default="id")
    parser.add_argument("--county-col", default="COUNTY")
    parser.add_argument("--pop-col", default="POP20")
    parser.add_argument("--dem-col", default="G20PREDEM")
    parser.add_argument("--rep-col", default="G20PREREP")
    parser.add_argument("--max-districts-per-county", type=int, default=2)
    parser.add_argument("--max-maps-per-chain", type=int, default=None)
    parser.add_argument("--block-col", default=None)
    parser.add_argument("--county-col-block", default=None)
    parser.add_argument("--pop-col-block", default=None)
    parser.add_argument("--assignment-block-col", default=None)
    parser.add_argument("--assignment-district-col", default=None)
    parser.add_argument("--place-name-column", default=None)
    parser.add_argument("--minimum-place-population", type=float, default=25_000)
    parser.add_argument("--label-count", type=int, default=3)
    add_municipality_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    geography = geography_from_args(args, graph_json=args.graph_json)
    if geography.mode != "PLACE":
        raise ValueError("municipal-place-partisanship requires municipality.mode=PLACE")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Place graph and vote totals...")
    graph_nodes = read_graph_nodes(args.graph_json, geography)
    graph_population_rows = geography.graph_population_rows(
        graph_nodes,
        node_id_col=args.node_id_col,
        pop_col=args.pop_col,
    )
    votes = compute_place_vote_shares(
        graph_nodes,
        geography,
        node_id_col=args.node_id_col,
        pop_col=args.pop_col,
        dem_col=args.dem_col,
        rep_col=args.rep_col,
    )
    names = read_place_names(geography, place_name_column=args.place_name_column)

    print("Computing compliant-ensemble Place distributions...")
    place_ids, ensemble_matrix, compliance = build_ensemble_matrix(
        args.chain_files,
        graph_nodes,
        graph_population_rows,
        atlas_key_col=args.atlas_key_col,
        node_id_col=args.node_id_col,
        county_col=args.county_col,
        max_districts_per_county=args.max_districts_per_county,
        max_maps_per_chain=args.max_maps_per_chain,
    )
    ensemble = summarize_ensemble(place_ids, ensemble_matrix)

    print("Loading block Place geography and official plans...")
    block_nodes = geography.read_block_nodes(
        args.block_metadata,
        block_col=args.block_col,
        county_col=args.county_col_block,
        pop_col=args.pop_col_block,
    )
    block_population_rows = geography.graph_population_rows(
        pd.DataFrame(block_nodes), node_id_col="id", pop_col="POP20"
    )
    official = summarize_official_plans(
        {
            "Enacted ('25)": args.enacted_baf,
            "Overturned ('21)": args.overturned_baf,
            "Remedial ('22)": args.remedial_baf,
        },
        block_population_rows,
        place_ids,
        ensemble_matrix,
        assignment_block_col=args.assignment_block_col,
        assignment_district_col=args.assignment_district_col,
    )
    summary = build_analysis_table(official, ensemble, votes, names)

    summary_path = output_dir / "place-partisanship-summary.csv"
    compliance_path = output_dir / "place-partisanship-ensemble-compliance.csv"
    summary.to_csv(summary_path, index=False)
    compliance.to_csv(compliance_path, index=False)
    geography.write_metadata(output_dir / "municipal-geography.json")

    interval_output = output_dir / "place-ousted-share-by-partisanship.png"
    residual_output = output_dir / "place-ousted-share-residual-by-partisanship.png"
    plot_place_ousted_share_by_partisanship(
        summary,
        interval_output,
        minimum_population=args.minimum_place_population,
        label_count=args.label_count,
    )
    plot_place_ousted_share_residual_by_partisanship(
        summary,
        residual_output,
        minimum_population=args.minimum_place_population,
        label_count=args.label_count,
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {compliance_path}")
    print(f"Wrote {interval_output}")
    print(f"Wrote {residual_output}")


if __name__ == "__main__":
    main()
