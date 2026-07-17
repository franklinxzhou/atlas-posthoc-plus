from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from atlas_posthoc_plus.config import ConfigError, ProjectConfig

BindingKind = Literal["value", "path", "values", "paths", "flag", "path_mapping"]


@dataclass(frozen=True)
class OutputDefault:
    relative: str


@dataclass(frozen=True)
class MunicipalityOutputDefault:
    relative: str


@dataclass(frozen=True)
class Binding:
    flag: str
    key: str
    kind: BindingKind = "value"
    required: bool = False
    default: Any = None


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    module: str
    category: str
    summary: str
    bindings: tuple[Binding, ...]
    legacy: bool = False
    required_municipality_mode: Literal["COUSUB", "PLACE"] | None = None

    def build_args(self, config: ProjectConfig) -> list[str]:
        if self.required_municipality_mode is not None:
            mode = str(config.require("municipality.mode")).upper()
            if mode != self.required_municipality_mode:
                raise ConfigError(
                    f"workflow {self.name!r} requires municipality.mode="
                    f"{self.required_municipality_mode}, found {mode}"
                )
        if any(binding.key == "municipality.mode" for binding in self.bindings):
            mode = str(config.require("municipality.mode")).upper()
            if mode not in {"COUSUB", "PLACE"}:
                raise ConfigError("municipality.mode must be 'COUSUB' or 'PLACE'")
            if mode == "PLACE" and config.get("municipality.place_shapefile") in {
                None,
                "",
            }:
                raise ConfigError(
                    f"workflow {self.name!r} requires "
                    "'municipality.place_shapefile' in PLACE mode"
                )

        args: list[str] = []
        for binding in self.bindings:
            value = config.get(binding.key)
            if value is None:
                value = binding.default
            if isinstance(value, OutputDefault):
                value = config.output_path(value.relative)
            if isinstance(value, MunicipalityOutputDefault):
                namespace = config.require("municipality.output_namespace")
                value = config.output_path(Path(str(namespace)) / value.relative)

            if value is None or value == "" or value == [] or value == {}:
                if binding.required:
                    raise ConfigError(
                        f"workflow {self.name!r} requires configuration value "
                        f"{binding.key!r}"
                    )
                continue

            if binding.kind == "flag":
                if bool(value):
                    args.append(binding.flag)
                continue

            if binding.kind == "path":
                value = config.resolve_path(value)
                args.extend([binding.flag, str(value)])
                continue

            if binding.kind == "paths":
                if not isinstance(value, list):
                    raise ConfigError(f"{binding.key!r} must be a list")
                args.append(binding.flag)
                args.extend(str(config.resolve_path(item)) for item in value)
                continue

            if binding.kind == "values":
                if not isinstance(value, list):
                    raise ConfigError(f"{binding.key!r} must be a list")
                args.append(binding.flag)
                args.extend(str(item) for item in value)
                continue

            if binding.kind == "path_mapping":
                if not isinstance(value, dict):
                    raise ConfigError(f"{binding.key!r} must be a TOML table")
                for label, path in value.items():
                    args.extend([binding.flag, f"{label}={config.resolve_path(path)}"])
                continue

            args.extend([binding.flag, str(value)])

        extra = config.get(f"workflow.{self.name}.extra_args", [])
        if not isinstance(extra, list):
            raise ConfigError(f"workflow.{self.name}.extra_args must be a list")
        args.extend(str(item) for item in extra)
        return args


GRAPH = Binding("--graph-json", "inputs.graph_json", "path", required=True)
GRAPH_PATH = Binding("--graph-path", "inputs.graph_json", "path", required=True)
MUNICIPAL_GRAPH = Binding(
    "--graph-json", "municipality.graph_json", "path", required=True
)
MUNICIPAL_GRAPH_PATH = Binding(
    "--graph-path", "municipality.graph_json", "path", required=True
)
MUNICIPALITY_BINDINGS = (
    Binding("--municipality-mode", "municipality.mode", required=True),
    Binding("--mapping-column", "municipality.mapping_column", default="PLACE"),
    Binding("--scalar-column", "municipality.scalar_column", default="MCD"),
    Binding("--place-shapefile", "municipality.place_shapefile", "path"),
    Binding("--place-key-column", "municipality.place_key_column"),
    Binding("--municipality-graph-json", "municipality.graph_json", "path"),
    Binding("--output-namespace", "municipality.output_namespace", required=True),
)
CHAINS = Binding("--chain-files", "chains.files", "paths", required=True)
CHAINS_ALT = Binding("--chains", "chains.files", "paths", required=True)
CHAIN_IDS = Binding("--chain-ids", "chains.ids", "values")
CROSSWALK = Binding("--crosswalk-csv", "inputs.crosswalk_csv", "path", required=True)
CROSSWALK_ALT = Binding("--crosswalk", "inputs.crosswalk_csv", "path", required=True)
ENACTED = Binding("--enacted-baf", "plans.enacted", "path", required=True)
OVERTURNED = Binding("--overturned-baf", "plans.overturned", "path", required=True)
REMEDIAL = Binding("--remedial-baf", "plans.remedial", "path", required=True)
ENACTED_ALT = Binding("--enacted", "plans.enacted", "path", required=True)
OVERTURNED_ALT = Binding("--overturned", "plans.overturned", "path", required=True)
REMEDIAL_ALT = Binding("--remedial", "plans.remedial", "path", required=True)
BLOCK_METADATA = Binding("--block-metadata", "inputs.block_metadata", "path", required=True)
MCD_SHAPE = Binding("--mcd-shapefile", "inputs.mcd_shapefile", "path")
MAX_MAPS = Binding("--max-maps-per-chain", "analysis.max_maps_per_chain")
MAX_COUNTY = Binding(
    "--max-districts-per-county", "analysis.max_districts_per_county", default=2
)
N_DISTRICTS = Binding("--n-districts", "analysis.n_districts", default=15)


WORKFLOWS: dict[str, WorkflowSpec] = {}


def _register(spec: WorkflowSpec) -> None:
    WORKFLOWS[spec.name] = spec


_register(
    WorkflowSpec(
        "build-crosswalk",
        "atlas_posthoc_plus.workflows.setup.crosswalk",
        "setup",
        "Build a population-weighted census-block to graph-precinct crosswalk.",
        (
            Binding("--blocks", "inputs.blocks", "path", required=True),
            Binding("--precincts", "inputs.precincts", "path", required=True),
            Binding(
                "--output",
                "outputs.crosswalk_csv",
                "path",
                default=OutputDefault("crosswalk/oh_block_to_precinct_crosswalk.csv"),
            ),
            Binding("--block-geoid-col", "columns.block_id", default="GEOID20"),
            Binding("--block-pop-col", "columns.block_population", default="POP20"),
            Binding("--precinct-name-col", "columns.precinct_name", default="NAME"),
            Binding("--area-crs", "analysis.area_crs", default="EPSG:26917"),
        ),
    )
)

_register(
    WorkflowSpec(
        "vote-comparison",
        "atlas_posthoc_plus.workflows.standard.vote_comparison",
        "standard",
        "Compare compliant ensemble ordered Democratic vote shares with reference plans.",
        (
            GRAPH_PATH,
            CHAINS_ALT,
            CHAIN_IDS,
            Binding("--plan-manifest", "inputs.plan_manifest", "path", required=True),
            CROSSWALK_ALT,
            Binding(
                "--output-dir",
                "outputs.vote_comparison_dir",
                "path",
                default=OutputDefault("vote-comparison"),
            ),
            N_DISTRICTS,
            MAX_COUNTY,
            MAX_MAPS,
        ),
    )
)

_register(
    WorkflowSpec(
        "vote-comparison-adjusted",
        "atlas_posthoc_plus.workflows.standard.vote_comparison_adjusted",
        "standard",
        "Vote-share comparison with ORIG_NAME-adjusted external-plan vote allocation.",
        (
            GRAPH_PATH,
            CHAINS_ALT,
            CHAIN_IDS,
            Binding("--plan-manifest", "inputs.plan_manifest", "path", required=True),
            CROSSWALK_ALT,
            Binding(
                "--output-dir",
                "outputs.vote_comparison_adjusted_dir",
                "path",
                default=OutputDefault("vote-comparison-adjusted"),
            ),
            N_DISTRICTS,
            MAX_COUNTY,
            MAX_MAPS,
            Binding("--external-vote-name-col", "columns.official_key", default="ORIG_NAME"),
            Binding("--pop-col", "columns.population", default="POP20"),
        ),
    )
)

_register(
    WorkflowSpec(
        "municipal-histograms-compliant",
        "atlas_posthoc_plus.workflows.municipal.compliant_histograms",
        "standard",
        "Compute compliant-chain municipal split traces and reference-plan histograms.",
        (
            MUNICIPAL_GRAPH_PATH,
            *MUNICIPALITY_BINDINGS,
            CHAINS_ALT,
            CHAIN_IDS,
            Binding(
                "--output-dir",
                "outputs.municipal_compliant_dir",
                "path",
                default=MunicipalityOutputDefault("municipal-compliant"),
            ),
            Binding(
                "--output-traces-json",
                "outputs.municipal_compliant_traces",
                "path",
                default=MunicipalityOutputDefault(
                    "municipal-compliant/municipal_split_traces_compliant.json"
                ),
            ),
            Binding(
                "--reference-json",
                "outputs.municipal_compliant_references",
                "path",
                default=MunicipalityOutputDefault(
                    "municipal-compliant/"
                    "oh_municipal_split_histograms_with_refs_reference_summaries.json"
                ),
            ),
            N_DISTRICTS,
            MAX_COUNTY,
            MAX_MAPS,
            BLOCK_METADATA,
            MCD_SHAPE,
            ENACTED_ALT,
            OVERTURNED_ALT,
            REMEDIAL_ALT,
            Binding("--block-col", "columns.block_id"),
            Binding("--county-col-block", "columns.block_county"),
            Binding("--pop-col-block", "columns.block_population"),
            Binding("--mcd-shape-county-col", "columns.mcd_county"),
            Binding("--mcd-shape-mcd-col", "columns.mcd_id"),
            Binding("--ousted-scale", "analysis.ousted_scale", default="share"),
        ),
    )
)

_register(
    WorkflowSpec(
        "municipal-histograms",
        "atlas_posthoc_plus.workflows.municipal.histograms",
        "municipal",
        "Plot pooled municipal-split histograms from atlas-posthoc trace JSON.",
        (
            *MUNICIPALITY_BINDINGS,
            Binding(
                "--traces-json",
                "inputs.traces_json",
                "path",
                default=MunicipalityOutputDefault(
                    "municipal-compliant/municipal_split_traces_compliant.json"
                ),
            ),
            BLOCK_METADATA,
            MCD_SHAPE,
            ENACTED_ALT,
            OVERTURNED_ALT,
            REMEDIAL_ALT,
            Binding("--block-col", "columns.block_id"),
            Binding("--county-col", "columns.block_county"),
            Binding("--pop-col", "columns.block_population"),
            Binding("--mcd-shape-county-col", "columns.mcd_county"),
            Binding("--mcd-shape-mcd-col", "columns.mcd_id"),
            Binding(
                "--output",
                "outputs.municipal_histogram",
                "path",
                default=MunicipalityOutputDefault("municipal/municipal-histograms.png"),
            ),
            Binding(
                "--reference-json",
                "outputs.reference_json",
                "path",
                default=MunicipalityOutputDefault("municipal/reference-summaries.json"),
            ),
        ),
    )
)

_register(
    WorkflowSpec(
        "municipal-distributions",
        "atlas_posthoc_plus.workflows.municipal.distributions",
        "municipal",
        "Plot chain-wise municipal-split distributions with reference plans.",
        (
            *MUNICIPALITY_BINDINGS,
            Binding(
                "--traces-json",
                "inputs.traces_json",
                "path",
                default=MunicipalityOutputDefault(
                    "municipal-compliant/municipal_split_traces_compliant.json"
                ),
            ),
            BLOCK_METADATA,
            MCD_SHAPE,
            ENACTED_ALT,
            OVERTURNED_ALT,
            REMEDIAL_ALT,
            Binding("--block-col", "columns.block_id"),
            Binding("--county-col", "columns.block_county"),
            Binding("--pop-col", "columns.block_population"),
            Binding("--mcd-shape-county-col", "columns.mcd_county"),
            Binding("--mcd-shape-mcd-col", "columns.mcd_id"),
            Binding(
                "--output",
                "outputs.municipal_distributions",
                "path",
                default=MunicipalityOutputDefault("municipal/municipal-distributions.png"),
            ),
        ),
    )
)

_register(
    WorkflowSpec(
        "municipal-tradeoff",
        "atlas_posthoc_plus.workflows.municipal.tradeoff",
        "municipal",
        "Plot cut-municipality versus ousted-population tradeoff.",
        (
            Binding(
                "--traces-json",
                "inputs.traces_json",
                "path",
                default=MunicipalityOutputDefault(
                    "municipal-compliant/municipal_split_traces_compliant.json"
                ),
            ),
            Binding(
                "--reference-json",
                "inputs.reference_json",
                "path",
                default=MunicipalityOutputDefault(
                    "municipal-compliant/"
                    "oh_municipal_split_histograms_with_refs_reference_summaries.json"
                ),
            ),
            Binding(
                "--output",
                "outputs.municipal_tradeoff",
                "path",
                default=MunicipalityOutputDefault("municipal/municipal-tradeoff.png"),
            ),
            Binding("--y-metric", "analysis.ousted_scale", default="share"),
        ),
    )
)

_register(
    WorkflowSpec(
        "municipal-outside-population-marginals",
        "atlas_posthoc_plus.workflows.municipal.outside_population_marginals",
        "municipal",
        "Compare ousted MCD population marginals across trace collections.",
        (
            Binding(
                "--trace-collection",
                "inputs.municipal_trace_collections",
                "path_mapping",
            ),
            Binding("--traces-json-mcd20", "inputs.traces_json_mcd20", "path"),
            Binding("--traces-json-mcd5", "inputs.traces_json_mcd5", "path"),
            Binding("--traces-json-mcd1", "inputs.traces_json_mcd1", "path"),
            Binding(
                "--reference-json",
                "inputs.reference_json",
                "path",
                default=MunicipalityOutputDefault(
                    "municipal-compliant/"
                    "oh_municipal_split_histograms_with_refs_reference_summaries.json"
                ),
            ),
            Binding(
                "--output",
                "outputs.municipal_outside_population_marginals",
                "path",
                default=MunicipalityOutputDefault(
                    "municipal/municipal-outside-population-marginals.png"
                ),
            ),
            Binding(
                "--y-metric",
                "workflow.municipal-outside-population-marginals.y_metric",
                default="share",
            ),
            Binding(
                "--include-other-ensemble",
                "workflow.municipal-outside-population-marginals.include_other_ensemble",
                "flag",
            ),
            Binding("--other-ensemble-summary-json", "inputs.other_summary_json", "path"),
            Binding(
                "--skip-official-plans",
                "workflow.municipal-outside-population-marginals.skip_official_plans",
                "flag",
            ),
        ),
    )
)

_register(
    WorkflowSpec(
        "municipal-cut-mcd-histograms",
        "atlas_posthoc_plus.workflows.municipal.cut_mcd_histograms",
        "municipal",
        "Compare overlaid cut-MCD histograms across trace collections.",
        (
            Binding(
                "--trace-collection",
                "inputs.municipal_trace_collections",
                "path_mapping",
            ),
            Binding("--traces-json-mcd20", "inputs.traces_json_mcd20", "path"),
            Binding("--traces-json-mcd5", "inputs.traces_json_mcd5", "path"),
            Binding("--traces-json-mcd1", "inputs.traces_json_mcd1", "path"),
            Binding(
                "--reference-json",
                "inputs.reference_json",
                "path",
                default=MunicipalityOutputDefault(
                    "municipal-compliant/"
                    "oh_municipal_split_histograms_with_refs_reference_summaries.json"
                ),
            ),
            Binding(
                "--output",
                "outputs.municipal_cut_mcd_histograms",
                "path",
                default=MunicipalityOutputDefault(
                    "municipal/municipal-cut-mcd-histograms.png"
                ),
            ),
            Binding(
                "--include-other-ensemble",
                "workflow.municipal-cut-mcd-histograms.include_other_ensemble",
                "flag",
            ),
            Binding("--other-ensemble-summary-json", "inputs.other_summary_json", "path"),
            Binding(
                "--skip-official-plans",
                "workflow.municipal-cut-mcd-histograms.skip_official_plans",
                "flag",
            ),
        ),
    )
)

_register(
    WorkflowSpec(
        "municipal-tradeoff-public",
        "atlas_posthoc_plus.workflows.municipal.tradeoff_public",
        "municipal",
        "Add validated citizen-submitted plans to the municipal tradeoff plot.",
        (
            *MUNICIPALITY_BINDINGS,
            Binding(
                "--traces-json",
                "inputs.traces_json",
                "path",
                default=MunicipalityOutputDefault(
                    "municipal-compliant/municipal_split_traces_compliant.json"
                ),
            ),
            Binding(
                "--reference-json",
                "inputs.reference_json",
                "path",
                default=MunicipalityOutputDefault(
                    "municipal-compliant/"
                    "oh_municipal_split_histograms_with_refs_reference_summaries.json"
                ),
            ),
            Binding("--public-plan-dir", "inputs.public_plan_dir", "path", required=True),
            BLOCK_METADATA,
            MCD_SHAPE,
            Binding(
                "--output",
                "outputs.municipal_tradeoff_public",
                "path",
                default=MunicipalityOutputDefault("municipal/municipal-tradeoff-public.png"),
            ),
            Binding(
                "--public-summary-json",
                "outputs.public_summary_json",
                "path",
                default=MunicipalityOutputDefault("municipal/public-plan-summaries.json"),
            ),
            Binding(
                "--public-manifest-out",
                "outputs.public_manifest_csv",
                "path",
                default=MunicipalityOutputDefault("municipal/public-plan-manifest.csv"),
            ),
            Binding("--expected-districts", "analysis.n_districts", default=15),
            Binding("--block-col", "columns.block_id"),
            Binding("--county-col-block", "columns.block_county"),
            Binding("--pop-col-block", "columns.block_population"),
            Binding("--mcd-shape-county-col", "columns.mcd_county"),
            Binding("--mcd-shape-mcd-col", "columns.mcd_id"),
        ),
    )
)

_register(
    WorkflowSpec(
        "municipal-tradeoff-weights",
        "atlas_posthoc_plus.workflows.municipal.tradeoff_weights",
        "municipal",
        "Compare municipal tradeoffs across arbitrary trace collections.",
        (
            *MUNICIPALITY_BINDINGS,
            Binding(
                "--trace-collection",
                "inputs.municipal_trace_collections",
                "path_mapping",
            ),
            Binding("--traces-json-mcd20", "inputs.traces_json_mcd20", "path"),
            Binding("--traces-json-mcd5", "inputs.traces_json_mcd5", "path"),
            Binding("--traces-json-mcd1", "inputs.traces_json_mcd1", "path"),
            Binding("--official-overturned", "plans.overturned", "path"),
            Binding("--official-remedial", "plans.remedial", "path"),
            Binding("--official-enacted", "plans.enacted", "path"),
            Binding(
                "--official-summary-json",
                "outputs.official_summary_json",
                "path",
                default=MunicipalityOutputDefault("municipal/official-plan-summaries.json"),
            ),
            Binding("--public-plan-dir", "inputs.public_plan_dir", "path"),
            Binding(
                "--public-summary-json",
                "outputs.public_summary_json",
                "path",
                default=MunicipalityOutputDefault("municipal/public-plan-summaries.json"),
            ),
            Binding(
                "--public-manifest-out",
                "outputs.public_manifest_csv",
                "path",
                default=MunicipalityOutputDefault("municipal/public-plan-manifest.csv"),
            ),
            Binding("--block-metadata", "inputs.block_metadata", "path"),
            MCD_SHAPE,
            Binding("--block-col", "columns.block_id"),
            Binding("--county-col-block", "columns.block_county"),
            Binding("--pop-col-block", "columns.block_population"),
            Binding("--mcd-shape-county-col", "columns.mcd_county"),
            Binding("--mcd-shape-mcd-col", "columns.mcd_id"),
            Binding("--expected-districts", "analysis.n_districts", default=15),
            Binding(
                "--output",
                "outputs.municipal_tradeoff_weights",
                "path",
                default=MunicipalityOutputDefault("municipal/municipal-tradeoff-weights.png"),
            ),
            Binding(
                "--reuse-official-summary",
                "workflow.municipal-tradeoff-weights.reuse_official_summary",
                "flag",
            ),
            Binding(
                "--reuse-public-summary",
                "workflow.municipal-tradeoff-weights.reuse_public_summary",
                "flag",
            ),
            Binding(
                "--warning-text",
                "workflow.municipal-tradeoff-weights.warning_text",
            ),
        ),
    )
)

_register(
    WorkflowSpec(
        "municipal-tradeoff-weights-external",
        "atlas_posthoc_plus.workflows.municipal.tradeoff_weights_external",
        "external",
        "Compare municipal trace collections and optionally overlay the external ensemble.",
        (
            *MUNICIPALITY_BINDINGS,
            Binding(
                "--trace-collection",
                "inputs.municipal_trace_collections",
                "path_mapping",
            ),
            Binding("--traces-json-mcd20", "inputs.traces_json_mcd20", "path"),
            Binding("--traces-json-mcd5", "inputs.traces_json_mcd5", "path"),
            Binding("--traces-json-mcd1", "inputs.traces_json_mcd1", "path"),
            Binding("--official-overturned", "plans.overturned", "path"),
            Binding("--official-remedial", "plans.remedial", "path"),
            Binding("--official-enacted", "plans.enacted", "path"),
            Binding(
                "--official-summary-json",
                "outputs.official_summary_json",
                "path",
                default=MunicipalityOutputDefault("municipal/official-plan-summaries.json"),
            ),
            Binding("--public-plan-dir", "inputs.public_plan_dir", "path"),
            Binding(
                "--public-summary-json",
                "outputs.public_summary_json",
                "path",
                default=MunicipalityOutputDefault("municipal/public-plan-summaries.json"),
            ),
            Binding(
                "--public-manifest-out",
                "outputs.public_manifest_csv",
                "path",
                default=MunicipalityOutputDefault("municipal/public-plan-manifest.csv"),
            ),
            Binding("--block-metadata", "inputs.block_metadata", "path"),
            MCD_SHAPE,
            Binding("--block-col", "columns.block_id"),
            Binding("--county-col-block", "columns.block_county"),
            Binding("--pop-col-block", "columns.block_population"),
            Binding("--mcd-shape-county-col", "columns.mcd_county"),
            Binding("--mcd-shape-mcd-col", "columns.mcd_id"),
            Binding("--expected-districts", "analysis.n_districts", default=15),
            Binding("--other-ensemble-summary-json", "inputs.other_summary_json", "path"),
            Binding(
                "--output",
                "outputs.municipal_tradeoff_weights_external",
                "path",
                default=MunicipalityOutputDefault("municipal/municipal-tradeoff-weights-external.png"),
            ),
            Binding(
                "--include-other-ensemble",
                "workflow.municipal-tradeoff-weights-external.include_other_ensemble",
                "flag",
            ),
            Binding(
                "--reuse-official-summary",
                "workflow.municipal-tradeoff-weights-external.reuse_official_summary",
                "flag",
            ),
            Binding(
                "--reuse-public-summary",
                "workflow.municipal-tradeoff-weights-external.reuse_public_summary",
                "flag",
            ),
            Binding(
                "--warning-text",
                "workflow.municipal-tradeoff-weights-external.warning_text",
            ),
        ),
        legacy=True,
    )
)

_register(
    WorkflowSpec(
        "municipal-place-partisanship",
        "atlas_posthoc_plus.workflows.municipal.place_partisanship",
        "municipal",
        "Compare official Place fragmentation with place-specific ensemble distributions.",
        (
            MUNICIPAL_GRAPH,
            *MUNICIPALITY_BINDINGS,
            CHAINS,
            BLOCK_METADATA,
            ENACTED,
            OVERTURNED,
            REMEDIAL,
            Binding(
                "--output-dir",
                "outputs.municipal_place_partisanship_dir",
                "path",
                default=MunicipalityOutputDefault("municipal-place-partisanship"),
            ),
            MAX_COUNTY,
            MAX_MAPS,
            Binding("--block-col", "columns.block_id"),
            Binding("--county-col-block", "columns.block_county"),
            Binding("--pop-col-block", "columns.block_population"),
            Binding("--place-name-column", "municipality.place_name_column"),
            Binding(
                "--minimum-place-population",
                "workflow.municipal-place-partisanship.minimum_place_population",
                default=25000,
            ),
            Binding(
                "--label-count",
                "workflow.municipal-place-partisanship.label_count",
                default=3,
            ),
        ),
        required_municipality_mode="PLACE",
    )
)

_register(
    WorkflowSpec(
        "municipal-city-partisan-context",
        "atlas_posthoc_plus.workflows.municipal.city_partisan_context",
        "municipal",
        "Compare each city's official partisan district context with its ensemble.",
        (
            MUNICIPAL_GRAPH,
            *MUNICIPALITY_BINDINGS,
            CHAINS,
            BLOCK_METADATA,
            ENACTED,
            OVERTURNED,
            REMEDIAL,
            Binding(
                "--output-dir",
                "outputs.municipal_city_partisan_context_dir",
                "path",
                default=MunicipalityOutputDefault("municipal-city-partisan-context"),
            ),
            MAX_COUNTY,
            MAX_MAPS,
            Binding("--block-col", "columns.block_id"),
            Binding("--block-pop-col", "columns.block_population"),
            Binding("--place-name-column", "municipality.place_name_column"),
            Binding(
                "--cities",
                "workflow.municipal-city-partisan-context.cities",
                "values",
                default=[
                    "Cleveland",
                    "Columbus",
                    "Cincinnati",
                    "Toledo",
                    "Akron",
                    "Dayton",
                ],
            ),
            Binding(
                "--bins",
                "workflow.municipal-city-partisan-context.bins",
                default=35,
            ),
        ),
        required_municipality_mode="PLACE",
    )
)

_register(
    WorkflowSpec(
        "district-rank-municipal",
        "atlas_posthoc_plus.workflows.municipal.district_rank",
        "municipal",
        "Create one municipal-burden histogram per vote-share-ranked district.",
        (
            CHAINS,
            MUNICIPAL_GRAPH,
            *MUNICIPALITY_BINDINGS,
            CROSSWALK,
            BLOCK_METADATA,
            MCD_SHAPE,
            ENACTED,
            OVERTURNED,
            REMEDIAL,
            Binding(
                "--output-dir",
                "outputs.district_rank_municipal_dir",
                "path",
                default=MunicipalityOutputDefault("district-rank-municipal"),
            ),
            N_DISTRICTS,
        ),
    )
)

_register(
    WorkflowSpec(
        "district-rank-municipal-compliant",
        "atlas_posthoc_plus.workflows.municipal.district_rank_compliant",
        "standard",
        "District-rank municipal burden histograms after county-compliance filtering.",
        (
            MUNICIPAL_GRAPH,
            *MUNICIPALITY_BINDINGS,
            CHAINS,
            CROSSWALK,
            BLOCK_METADATA,
            MCD_SHAPE,
            ENACTED,
            OVERTURNED,
            REMEDIAL,
            Binding(
                "--output-dir",
                "outputs.district_rank_municipal_compliant_dir",
                "path",
                default=MunicipalityOutputDefault("district-rank-municipal-compliant"),
            ),
            N_DISTRICTS,
            MAX_COUNTY,
            MAX_MAPS,
        ),
    )
)

_register(
    WorkflowSpec(
        "cut-edges-compliant",
        "atlas_posthoc_plus.workflows.compactness.cut_edges",
        "standard",
        "Plot graph cut-edge counts for compliant ensemble and official plans.",
        (
            GRAPH,
            CHAINS,
            CROSSWALK,
            ENACTED,
            OVERTURNED,
            REMEDIAL,
            Binding(
                "--output-dir",
                "outputs.cut_edges_dir",
                "path",
                default=OutputDefault("cut-edges"),
            ),
            MAX_COUNTY,
            MAX_MAPS,
        ),
    )
)

_register(
    WorkflowSpec(
        "official-polsby",
        "atlas_posthoc_plus.workflows.compactness.official_polsby",
        "compactness",
        "Compute official-plan district and summary Polsby-Popper values from blocks.",
        (
            Binding("--block-shapefile", "inputs.block_shapefile", "path", required=True),
            ENACTED,
            OVERTURNED,
            REMEDIAL,
            Binding("--expected-districts", "analysis.n_districts", default=15),
            Binding(
                "--summary-output",
                "outputs.polsby_summary_csv",
                "path",
                default=OutputDefault("polsby/official-summary.csv"),
            ),
            Binding(
                "--district-output",
                "outputs.polsby_district_csv",
                "path",
                default=OutputDefault("polsby/official-by-district.csv"),
            ),
        ),
    )
)

_register(
    WorkflowSpec(
        "polsby-trace",
        "atlas_posthoc_plus.workflows.compactness.polsby_trace",
        "compactness",
        "Plot ensemble average/minimum/maximum Polsby-Popper traces with references.",
        (
            GRAPH,
            CHAINS,
            CHAIN_IDS,
            Binding(
                "--official-reference-csv",
                "inputs.polsby_reference_csv",
                "path",
                default=OutputDefault("polsby/official-summary.csv"),
            ),
            Binding("--metric", "workflow.polsby-trace.metric", default="average"),
            Binding("--output-frequency", "analysis.output_frequency", default=50_000),
            Binding("--total-proposals", "analysis.total_proposals"),
            MAX_MAPS,
            Binding(
                "--output",
                "outputs.polsby_trace",
                "path",
                default=OutputDefault("polsby/polsby-trace.png"),
            ),
        ),
    )
)

_register(
    WorkflowSpec(
        "polsby-histogram",
        "atlas_posthoc_plus.workflows.compactness.polsby_histogram",
        "compactness",
        "Plot a Polsby-Popper histogram with optional public and official plan overlays.",
        (
            Binding("--ensemble-pp-json", "inputs.polsby_traces_json", "path", required=True),
            Binding("--public-scores-csv", "inputs.public_polsby_scores_csv", "path"),
            Binding("--public-plan-dir", "inputs.public_plan_dir", "path"),
            Binding(
                "--public-scores-out",
                "outputs.public_polsby_scores_csv",
                "path",
                default=OutputDefault("polsby/public-plan-scores.csv"),
            ),
            Binding(
                "--public-score-method",
                "workflow.polsby-histogram.public_score_method",
                default="graph",
            ),
            Binding(
                "--refresh-public-scores",
                "workflow.polsby-histogram.refresh_public_scores",
                "flag",
            ),
            Binding("--graph-json", "inputs.graph_json", "path"),
            Binding("--crosswalk-csv", "inputs.crosswalk_csv", "path"),
            Binding("--graph-key-col", "columns.official_key", default="ORIG_NAME"),
            Binding("--node-id-col", "workflow.polsby-histogram.node_id_col", default="id"),
            Binding("--area-col", "workflow.polsby-histogram.area_col", default="area"),
            Binding(
                "--border-length-col",
                "workflow.polsby-histogram.border_length_col",
                default="border_length",
            ),
            Binding("--edge-length-col", "workflow.polsby-histogram.edge_length_col"),
            Binding("--block-shapefile", "inputs.block_shapefile", "path"),
            Binding("--block-col", "columns.block_id"),
            Binding("--projected-crs", "analysis.polsby_projected_crs", default="EPSG:5070"),
            Binding("--expected-districts", "analysis.n_districts", default=15),
            Binding("--assignment-block-col", "workflow.polsby-histogram.assignment_block_col"),
            Binding(
                "--assignment-district-col",
                "workflow.polsby-histogram.assignment_district_col",
            ),
            Binding(
                "--official-scores-csv",
                "inputs.official_polsby_summary_csv",
                "path",
                default=OutputDefault("polsby/official-summary.csv"),
            ),
            Binding("--stat", "workflow.polsby-histogram.stat", default="average"),
            Binding("--burnin", "workflow.polsby-histogram.burnin", default=0),
            Binding("--bins", "workflow.polsby-histogram.bins", default=40),
            Binding("--density", "workflow.polsby-histogram.density", "flag"),
            Binding("--title", "workflow.polsby-histogram.title"),
            Binding("--ensemble-label", "workflow.polsby-histogram.ensemble_label"),
            Binding("--public-label", "workflow.polsby-histogram.public_label"),
            Binding("--official-label", "workflow.polsby-histogram.official_label"),
            Binding(
                "--out",
                "outputs.polsby_histogram",
                "path",
                default=OutputDefault("polsby/polsby-histogram.png"),
            ),
        ),
    )
)

_register(
    WorkflowSpec(
        "ranked-vote-cvap",
        "atlas_posthoc_plus.workflows.demographics.ranked_vote_cvap",
        "demographics",
        "Compare ranked district Democratic vote share and Black CVAP.",
        (
            CHAINS,
            GRAPH,
            Binding("--cvap-file", "inputs.cvap_file", "path", required=True),
            CROSSWALK,
            ENACTED,
            OVERTURNED,
            REMEDIAL,
            Binding(
                "--output",
                "outputs.ranked_vote_cvap",
                "path",
                default=OutputDefault("demographics/ranked-vote-cvap.png"),
            ),
            Binding(
                "--output-csv",
                "outputs.ranked_vote_cvap_csv",
                "path",
                default=OutputDefault("demographics/ranked-vote-cvap.csv"),
            ),
            MAX_COUNTY,
            N_DISTRICTS,
            MAX_MAPS,
        ),
    )
)

_register(
    WorkflowSpec(
        "responsiveness",
        "atlas_posthoc_plus.workflows.elections.responsiveness",
        "elections",
        "Allocate statewide elections and plot seat responsiveness across races.",
        (
            GRAPH,
            Binding("--target-precincts", "inputs.target_precincts", "path", required=True),
            CHAINS,
            Binding("--election-layer", "elections.layers", "path_mapping", required=True),
            CROSSWALK,
            ENACTED,
            OVERTURNED,
            REMEDIAL,
            Binding(
                "--output-dir",
                "outputs.responsiveness_dir",
                "path",
                default=OutputDefault("responsiveness"),
            ),
            MAX_COUNTY,
            MAX_MAPS,
            N_DISTRICTS,
        ),
    )
)

_register(
    WorkflowSpec(
        "responsiveness-zoom",
        "atlas_posthoc_plus.workflows.elections.responsiveness_zoom",
        "elections",
        "Create zoomed responsiveness panels from previously generated CSVs.",
        (
            Binding(
                "--race-summary",
                "inputs.race_summary",
                "path",
                default=OutputDefault("responsiveness/oh_responsiveness_race_summary.csv"),
            ),
            Binding(
                "--ensemble-seats",
                "inputs.ensemble_seats",
                "path",
                default=OutputDefault("responsiveness/oh_responsiveness_ensemble_seats.csv"),
            ),
            Binding(
                "--official-seats",
                "inputs.official_seats",
                "path",
                default=OutputDefault(
                    "responsiveness/oh_responsiveness_official_plan_seats.csv"
                ),
            ),
            Binding(
                "--output",
                "outputs.responsiveness_zoom",
                "path",
                default=OutputDefault("responsiveness/responsiveness-zoom.png"),
            ),
            N_DISTRICTS,
        ),
    )
)

_register(
    WorkflowSpec(
        "inspect-external-ensemble",
        "atlas_posthoc_plus.workflows.external.inspect_ensemble",
        "external",
        "Inspect an external wide ensemble CSV and GeoJSON before processing.",
        (
            Binding("--csv", "inputs.other_csv", "path", required=True),
            Binding("--geojson", "inputs.other_geojson", "path", required=True),
            Binding(
                "--out-dir",
                "outputs.external_probe_dir",
                "path",
                default=OutputDefault("external/probe"),
            ),
        ),
        legacy=True,
    )
)

_register(
    WorkflowSpec(
        "summarize-external-ensemble",
        "atlas_posthoc_plus.workflows.external.summarize_ensemble",
        "external",
        "Compute municipal split summaries for the external ensemble.",
        (
            Binding("--csv", "inputs.other_csv", "path", required=True),
            Binding("--geojson", "inputs.other_geojson", "path", required=True),
            Binding("--expected-districts", "analysis.n_districts", default=15),
            Binding(
                "--output-json",
                "outputs.external_summary_json",
                "path",
                default=OutputDefault("external/municipal-summaries.json"),
            ),
            Binding(
                "--output-csv",
                "outputs.external_summary_csv",
                "path",
                default=OutputDefault("external/municipal-summaries.csv"),
            ),
        ),
        legacy=True,
    )
)


PIPELINES: dict[str, tuple[str, ...]] = {
    "standard": (
        "vote-comparison-adjusted",
        "municipal-histograms-compliant",
        "district-rank-municipal-compliant",
        "cut-edges-compliant",
        "official-polsby",
        "polsby-trace",
        "ranked-vote-cvap",
        "responsiveness",
    ),
    "municipal": (
        "municipal-histograms",
        "municipal-distributions",
        "municipal-tradeoff",
        "municipal-outside-population-marginals",
        "municipal-cut-mcd-histograms",
        "municipal-tradeoff-public",
        "municipal-place-partisanship",
        "municipal-city-partisan-context",
    ),
    "external": (
        "inspect-external-ensemble",
        "summarize-external-ensemble",
        "municipal-tradeoff-weights-external",
    ),
}

PIPELINES["full"] = (
    *PIPELINES["standard"],
    *PIPELINES["municipal"],
    "responsiveness-zoom",
)

SUITES = PIPELINES


def get_workflow(name: str) -> WorkflowSpec:
    try:
        return WORKFLOWS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(WORKFLOWS))
        raise ConfigError(f"unknown workflow {name!r}; choose from: {choices}") from exc
