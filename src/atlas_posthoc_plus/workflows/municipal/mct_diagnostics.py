"""Compare municipal corporation and township splits across AtlasIO and official plans."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np
import pandas as pd
from atlas_posthoc.io.atlas_reader import canonicalize_unit_key, infer_chain_id

from atlas_posthoc_plus.io.atlas_records import iter_records
from atlas_posthoc_plus.io.graph import read_graph_frame
from atlas_posthoc_plus.paper import apply_paper_style, save_figure
from atlas_posthoc_plus.workflows.municipal.compliant_histograms import (
    county_compliance_summary_for_assignment,
)
from atlas_posthoc_plus.workflows.municipal.mct_geography import (
    OHIO_POPULATION_2020,
    GeographySettings,
    load_membership,
)
from atlas_posthoc_plus.workflows.municipal.mct_metrics import SplitScorer
from atlas_posthoc_plus.workflows.municipal.tradeoff_weights import read_assignment_table

OFFICIAL_PLANS = (
    ("2021 overturned", "overturned"),
    ("2022 remedial", "remedial"),
    ("2025 enacted", "enacted"),
)
PRIMARY_METRIC = "deduplicated_ousted_population"
NAIVE_METRIC = "naive_ousted_population"


def _validate_districts(districts: np.ndarray, count: int) -> None:
    if np.any(districts < 0) or len(np.unique(districts)) != count:
        raise ValueError(f"Expected {count} distinct nonnegative integer district labels")


def _official_assignment(path: Path, blocks: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    assignment, _ = read_assignment_table(
        path, block_col=args.assignment_block_col, district_col=args.assignment_district_col,
        strict=True,
    )
    unknown = sorted(set(assignment).difference(blocks.index))
    if unknown:
        raise ValueError(f"{path} has IDs outside the 2020 block population table: {unknown[:5]}")
    assigned = blocks.index.to_series().map(assignment)
    missing = assigned.isna() & blocks["population"].gt(0)
    if missing.any():
        examples = blocks.index[missing][:5].tolist()
        raise ValueError(f"{path} is missing {int(missing.sum())} populated blocks: {examples}")
    districts = assigned.fillna("-1").astype(int).to_numpy()
    _validate_districts(districts[districts != -1], args.n_districts)
    return districts


def _chain_context(graph: pd.DataFrame, args: argparse.Namespace) -> tuple[dict, dict]:
    key_col = args.atlas_key_col or args.graph_key_col
    node_ids = graph[args.node_id_col].astype(str)
    key_to_id = {}
    for key, node in zip(graph[key_col], node_ids):
        if not isinstance(key, (list, tuple)) and (pd.isna(key) or not str(key).strip()):
            raise ValueError(f"Missing Atlas key for graph node {node}")
        canonical = canonicalize_unit_key(key)
        if canonical in key_to_id:
            raise ValueError(f"Duplicate graph Atlas key: {canonical}")
        key_to_id[canonical] = node
    if node_ids.duplicated().any() or graph[args.county_col].isna().any():
        raise ValueError("Graph must have unique node IDs and county labels for every node")
    node_attrs = {
        node: {args.county_col: county}
        for node, county in zip(node_ids, graph[args.county_col])
    }
    return key_to_id, node_attrs


def _chain_assignment(
    raw: dict, key_to_id: dict, node_attrs: dict, count: int,
) -> tuple[dict, np.ndarray]:
    assignment = {}
    for key, district in raw.items():
        node = key_to_id.get(key)
        if node is None and len(key) == 1 and key[0] in node_attrs:
            node = key[0]
        if node is None or node in assignment:
            raise ValueError(f"Unknown or ambiguous graph assignment key: {key}")
        assignment[node] = district
    if len(assignment) != len(node_attrs):
        missing = sorted(node_attrs.keys() - assignment.keys())
        raise ValueError(f"Chain assignment is missing {len(missing)} graph nodes: {missing[:5]}")
    districts = np.fromiter((assignment[node] for node in node_attrs), dtype=np.int64)
    _validate_districts(districts, count)
    return assignment, districts


def _append_details(path: Path, identity: dict, details: pd.DataFrame) -> None:
    details = details.assign(**identity)[[*identity, *details.columns]]
    details.to_csv(path, mode="a", header=not path.exists(), index=False)


def _score_chains(
    args: argparse.Namespace, graph: pd.DataFrame, scorer: SplitScorer, details_path: Path,
) -> tuple[list[dict], list[dict]]:
    key_to_id, node_attrs = _chain_context(graph, args)
    ids = args.chain_ids if args.chain_ids is not None else [infer_chain_id(p) for p in args.chains]
    if len(ids) != len(args.chains) or len(set(ids)) != len(ids) or any(not i.strip() for i in ids):
        raise ValueError("Provide one unique, nonempty chain ID per chain file")

    traces, chain_summaries = [], []
    for chain_id, path in zip(ids, args.chains):
        plans, examined = [], 0
        records = iter_records(path, chain_id=chain_id, skip_bad_lines=False)
        thinned = islice(records, 0, None, args.thin_step)
        for record in islice(thinned, args.max_maps_per_chain):
            examined += 1
            try:
                assignment, districts = _chain_assignment(
                    record.assignment, key_to_id, node_attrs, args.n_districts,
                )
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"{chain_id}, line {record.line_number}: {exc}") from exc
            compliance = county_compliance_summary_for_assignment(
                assignment, node_attrs, county_col=args.county_col,
                max_districts_per_county=args.max_districts_per_county,
            )
            if not compliance["compliant"]:
                continue
            identity = {
                "chain_id": chain_id, "map_index": record.map_index,
                "line_number": record.line_number, "plan": record.name,
            }
            metrics, details = scorer.score(districts, details=args.ensemble_unit_details)
            plans.append({**identity, "weight": record.weight, **metrics})
            if details is not None:
                _append_details(details_path, {"plan_type": "ensemble", **identity}, details)

        values = np.array([plan[PRIMARY_METRIC] for plan in plans])
        chain_summaries.append({
            "chain_id": chain_id, "chain_file": str(path), "examined": examined,
            "compliant_maps": len(plans), "noncompliant_maps": examined - len(plans),
            "mean_deduplicated_ousted_population": float(values.mean()) if len(values) else None,
            "median_deduplicated_ousted_population": (
                float(np.median(values)) if len(values) else None
            ),
        })
        traces.append({"chain_id": chain_id, "chain_file": str(path), "plans": plans})
        print(f"{chain_id}: retained {len(plans):,} of {examined:,} examined maps", flush=True)
    return traces, chain_summaries


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _plot(path: Path, plans: pd.DataFrame, references: pd.DataFrame) -> None:
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    apply_paper_style()
    figure, (histogram, comparison) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    units = "Municipal Corporations and Townships"
    deduplicated_label = f"Deduplicated ousted population\n{units}"
    naive_label = f"Conventional summed ousted population\n{units}"
    histogram.hist(plans[PRIMARY_METRIC], bins=30, color="#4b7194")
    histogram.set(xlabel=deduplicated_label, ylabel="Compliant ensemble plans")

    plotted_columns = [PRIMARY_METRIC, NAIVE_METRIC]
    limits = pd.concat([plans[plotted_columns], references[plotted_columns]]).to_numpy()
    low, high = float(limits.min()), float(limits.max())
    comparison.axline((low, low), slope=1, color="0.60", linewidth=0.9, linestyle="--", zorder=0)
    comparison.scatter(plans[NAIVE_METRIC], plans[PRIMARY_METRIC], s=9, color="#4b7194", alpha=0.3)
    margin = max((high - low) * 0.06, 1)
    comparison.set(
        xlabel=naive_label, ylabel=deduplicated_label,
        xlim=(max(0, low - margin), high + margin), ylim=(max(0, low - margin), high + margin),
        aspect="equal",
    )
    for i, (ref, color, marker) in enumerate(zip(
        references.itertuples(), ("#944c1f", "#91396e", "#297561"), ("o", "s", "D"),
    )):
        height = 0.98 - 0.09 * i
        histogram.axvline(
            getattr(ref, PRIMARY_METRIC), ymin=height - 0.07, ymax=height,
            color=color, linewidth=2, label=ref.plan,
        )
        comparison.scatter(
            getattr(ref, NAIVE_METRIC), getattr(ref, PRIMARY_METRIC),
            s=31, marker=marker, color=color, edgecolor="white", linewidth=0.5, zorder=3,
        )
    histogram.legend(loc="upper left", bbox_to_anchor=(0, 1.13), ncols=3, frameon=False)
    for axis in (histogram.xaxis, histogram.yaxis, comparison.xaxis, comparison.yaxis):
        axis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    save_figure(figure, path)
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    if (
        min(args.thin_step, args.n_districts, args.max_districts_per_county) < 1
        or (args.max_maps_per_chain is not None and args.max_maps_per_chain < 1)
    ):
        raise ValueError("Thinning, district counts, and map limits must be positive")
    graph = read_graph_frame(args.graph_json, node_id_col=args.node_id_col)
    settings = GeographySettings(**{f.name: getattr(args, f.name) for f in fields(GeographySettings)})
    membership, metadata = load_membership(
        graph_path=args.graph_json, graph=graph, blocks_path=args.block_metadata,
        place_path=args.place_shapefile, cousub_path=args.mcd_shapefile,
        place_baf=args.place_block_assignments, cousub_baf=args.cousub_block_assignments,
        crosswalk_path=args.crosswalk_csv, cache_dir=args.output_dir / ".membership-cache",
        settings=settings,
    )
    audit = membership.classification
    audit_totals = audit.groupby(
        ["source", "classfp", "lsadc", "funcstat", "accepted", "decision", "review"],
        as_index=False,
    ).agg(records=("geoid", "size"), population=("population", "sum"))
    print(audit_totals.to_string(index=False), flush=True)
    population = float(settings.expected_population)
    graph_scorer = SplitScorer(membership.graph_pieces, membership.units, statewide_population=population)
    block_pieces = membership.blocks.assign(position=np.arange(len(membership.blocks)))
    official_scorer = SplitScorer(block_pieces, membership.units, statewide_population=population)

    with TemporaryDirectory(dir=args.output_dir) as temp_dir:
        out = Path(temp_dir)
        details_path = out / "mct_unit_details.csv"
        official_plans = []
        for plan, option in OFFICIAL_PLANS:
            path = getattr(args, option)
            assignment = _official_assignment(path, membership.blocks, args)
            metrics, details = official_scorer.score(assignment, details=True)
            identity = {"plan": plan, "assignment_file": str(path)}
            official_plans.append({**identity, **metrics})
            _append_details(
                details_path,
                {"plan_type": "official", "chain_id": "", "map_index": "", "line_number": "",
                 "plan": plan},
                details,
            )

        traces, summaries = _score_chains(args, graph, graph_scorer, details_path)
        plans = pd.DataFrame([plan for trace in traces for plan in trace["plans"]])
        if plans.empty:
            raise ValueError("No compliant ensemble plans were available")
        references = pd.DataFrame(official_plans)
        plans.to_csv(out / "mct_plan_summary.csv", index=False)
        references.to_csv(out / "mct_official_plans.csv", index=False)
        pd.DataFrame(summaries).to_csv(out / "mct_chain_summary.csv", index=False)
        audit.to_csv(out / "mct_geography_classification.csv", index=False)
        audit_totals.to_csv(out / "mct_geography_classification_summary.csv", index=False)
        _write_json(out / "mct_traces.json", {"chains": traces})
        _plot(out / "mct_diagnostics.pdf", plans, references)
        _write_json(out / "mct_metadata.json", {
            **metadata, "allocation": membership.allocation_audit,
            "total_blocks": len(membership.blocks),
            "positive_blocks": int(membership.blocks["population"].gt(0).sum()),
            "statewide_population": population,
            "represented_population": official_scorer.represented_population,
            "positive_population_cuts": True,
            "dominant_district_order": "ascending numeric label, with a 1e-9 population tie tolerance",
            "thinning": args.thin_step, "n_districts": args.n_districts,
            "max_maps_per_chain": args.max_maps_per_chain,
            "max_districts_per_county": args.max_districts_per_county,
            "ensemble_unit_details": args.ensemble_unit_details,
        })
        for path in sorted(out.iterdir(), key=lambda file: file.name == "mct_metadata.json"):
            path.replace(args.output_dir / path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "graph-json", "crosswalk-csv", "block-metadata", "place-shapefile", "mcd-shapefile",
        "place-block-assignments", "cousub-block-assignments", "output-dir",
        "overturned", "remedial", "enacted",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--chains", type=Path, nargs="+", required=True)
    parser.add_argument("--chain-ids", nargs="+")
    parser.add_argument("--node-id-col", default="id")
    parser.add_argument("--atlas-key-col")
    parser.add_argument("--county-col", default="COUNTY")
    parser.add_argument("--graph-key-col", default="NAME")
    parser.add_argument("--graph-pop-col", default="POP20")
    parser.add_argument("--block-id-col", default="GEOID20")
    parser.add_argument("--block-pop-col", default="POP20")
    parser.add_argument("--crosswalk-block-col", default="GEOID20")
    parser.add_argument("--crosswalk-key-col", default="NAME")
    parser.add_argument("--crosswalk-pop-col", default="POP20")
    parser.add_argument("--assignment-block-col")
    parser.add_argument("--assignment-district-col")
    parser.add_argument("--expected-population", type=int, default=OHIO_POPULATION_2020)
    parser.add_argument("--township-lsadc", default="47", choices=("44", "47"))
    parser.add_argument("--unit-definition", default="whole", choices=("whole", "county-component"))
    parser.add_argument("--split-block-policy", default="reject", choices=("reject", "allocated"))
    parser.add_argument("--n-districts", type=int, default=15)
    parser.add_argument("--max-districts-per-county", type=int, default=2)
    parser.add_argument("--thin-step", type=int, default=1)
    parser.add_argument("--max-maps-per-chain", type=int)
    parser.add_argument("--ensemble-unit-details", action="store_true")
    return parser.parse_args()


def main() -> None:
    matplotlib.use("Agg")
    run(parse_args())


if __name__ == "__main__":
    main()
