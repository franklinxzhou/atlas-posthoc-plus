# atlas-posthoc-plus

`atlas-posthoc-plus` is a self-contained Python package for producing the
project-specific plots and analysis tables that sit on top of
[`atlas-posthoc`](https://github.com/franklinxzhou/atlas-posthoc).

The original loose scripts have been migrated into importable modules under
`src/atlas_posthoc_plus/workflows/`. Running the package does **not** import,
discover, or execute code from `oh-gerrymandering/scripts`.

## Package structure

```text
src/
└── atlas_posthoc_plus/
    ├── cli.py                         # command-line interface
    ├── config.py                      # TOML loading and path resolution
    ├── io/
    │   └── atlas_compat.py             # legacy reader API backed by atlas-posthoc
    ├── registry.py                    # workflow and suite definitions
    ├── runner.py                      # execute one packaged workflow
    ├── suite.py                       # execute suites and write manifests
    └── workflows/
        ├── setup/
        │   └── crosswalk.py
        ├── standard/
        │   ├── vote_comparison.py
        │   └── vote_comparison_adjusted.py
        ├── municipal/
        │   ├── compliant_histograms.py
        │   ├── cut_mcd_histograms.py
        │   ├── district_rank.py
        │   ├── district_rank_compliant.py
        │   ├── distributions.py
        │   ├── histograms.py
        │   ├── marginal_helpers.py
        │   ├── outside_population_marginals.py
        │   ├── tradeoff.py
        │   ├── tradeoff_public.py
        │   ├── tradeoff_weights.py
        │   └── tradeoff_weights_external.py
        ├── compactness/
        │   ├── cut_edges.py
        │   ├── official_polsby.py
        │   ├── polsby_histogram.py
        │   └── polsby_trace.py
        ├── demographics/
        │   └── ranked_vote_cvap.py
        ├── elections/
        │   ├── responsiveness.py
        │   └── responsiveness_zoom.py
        └── external/
            ├── inspect_ensemble.py
            └── summarize_ensemble.py
```

`atlas-posthoc` remains the preferred implementation for reusable AtlasIO
parsing, graph loading, municipal-split observables, compactness observables,
and plotting primitives. `atlas-posthoc-plus` contains the Ohio/project-specific
orchestration and specialized analyses.

## Installation

For development with the adjacent `atlas-posthoc` checkout:

```bash
cd /Users/franklinzhou/Documents/Work/Duke/Gerrymandering
python3 -m pip install -e ./atlas-posthoc
python3 -m pip install -e ./atlas-posthoc-plus
```

The commands `atlas-posthoc-plus` and `aph-plus` are equivalent.

## Configuration

All workflows use one TOML configuration:

```bash
atlas-posthoc-plus init analysis.toml
```

Paths are resolved relative to `[project].root`. Input files can live anywhere;
the package does not require a particular repository layout.

The checked-in `atlas-posthoc-plus.toml` is a ready-to-edit Ohio configuration
using the four `run37`–`run40` chains.

Important sections:

- `[chains]`: AtlasIO chain files and chain labels.
- `[analysis]`: district count, compliance rule, thinning, and proposal count.
- `[inputs]`: graph, crosswalk, block/MCD geometry, CVAP, and derived files.
- `[municipality]`: explicit COUSUB or PLACE graph/schema bindings and output namespace.
- `[plans]`: enacted, overturned, and remedial block-assignment files.
- `[elections.layers]`: election layers keyed by year.
- `[outputs]`: the root directory for generated results.
- `[workflow.NAME]`: workflow-specific options and extra CLI arguments.

Municipal analyses use one of two explicit arms. The checked-in configuration
retains the baseline:

```toml
[municipality]
mode = "COUSUB"
graph_json = "data/graph/OHpct20.json"
scalar_column = "MCD"
output_namespace = "cousub"
```

Use a separate configuration for the PLACE comparison:

```toml
[municipality]
mode = "PLACE"
graph_json = "data/graph/OH_pct20.json"
mapping_column = "PLACE"
place_shapefile = "data/raw/tl_2024_39_place"
output_namespace = "place"

[inputs]
# This is the block-geometry checkpoint used by graph generation. Using the
# raw TIGER block polygons changes the represented denominator by 26 people.
block_metadata = "data/derived/cblocksMergeMultiPcts_OH.gpkg"
```

PLACE graph nodes are exploded into weighted Place-population pieces. Official,
remedial, and public plans assign Census blocks to the same Place layer using
each block's interior representative point; blocks outside every Place are
excluded from the PLACE population denominator. The adapter matches the Place
layer key to the selected graph automatically: current sampler graphs use
`GEOID`, while older name-keyed graphs continue to use `NAME`. Set
`place_key_column` explicitly only to override that inference. Municipality-
scoped default outputs are written below `outputs/<output_namespace>/`, and
cache metadata prevents cross-arm reuse.

## Running one analysis

List the available workflow modules:

```bash
atlas-posthoc-plus list
```

Validate an analysis before a long run:

```bash
atlas-posthoc-plus doctor vote-comparison-adjusted
```

Show the exact packaged-module command:

```bash
atlas-posthoc-plus show vote-comparison-adjusted
```

Run it:

```bash
atlas-posthoc-plus run vote-comparison-adjusted
```

Pass a one-off override after `--`:

```bash
atlas-posthoc-plus run cut-edges-compliant -- --max-maps-per-chain 25
```

The resulting command invokes a module such as:

```bash
python -m atlas_posthoc_plus.workflows.compactness.cut_edges ...
```

It never invokes an external script file.

## Producing a suite

Preview the main suite:

```bash
atlas-posthoc-plus suite standard --dry-run
```

Run it:

```bash
atlas-posthoc-plus suite standard
```

Available suites:

- `standard`: vote comparison, compliant municipal analysis, district-rank
  municipal plots, cut edges, Polsby-Popper, CVAP, and responsiveness.
- `municipal`: derived municipal histograms, distributions, tradeoff,
  multi-ensemble marginal plots, public-plan comparison, and Place-partisanship
  diagnostics.
- `external`: inspect and summarize the optional outside ensemble.
- `full`: `standard`, then `municipal`, then the zoomed responsiveness figure.

Each non-dry suite writes `suite-<name>-manifest.json` under `outputs.root`. The
manifest records every command, return code, run time, configuration path, and
overall success status.

Suites sweep each configured AtlasIO chain once into graph-indexed assignments
under `outputs.root/.atlas-chain-cache`. Atlas key decoding and county-compliance
evaluation happen only in that pass; raw-chain workflows reuse the dense
assignments and attached compliance results. The sweep is rebuilt automatically
when a chain, graph, or relevant filter setting changes, then the cache directory
is deleted automatically after the suite finishes (including after a workflow
failure). Standalone `run` commands continue to read the source chain directly.

Use `--keep-going` to continue after a failed workflow:

```bash
atlas-posthoc-plus suite full --keep-going
```

## Output dependencies

The configured suites connect multi-stage analyses automatically:

- `municipal-histograms-compliant` creates the compliant municipal trace JSON
  and reference-plan summary JSON used by the derived municipal workflows.
- `official-polsby` creates the reference CSV used by `polsby-trace`.
- `polsby-histogram` consumes `atlas-posthoc`'s
  `diagnostics/polsby_popper_traces.json` and can overlay the `official-polsby`
  summary CSV. If `inputs.public_plan_dir`, `inputs.graph_json`, and
  `inputs.crosswalk_csv` are configured, it also computes and caches public-plan
  Polsby-Popper scores.
- `responsiveness` creates the CSV files used by `responsiveness-zoom`.

Explicit paths in `[inputs]` override these generated defaults.

## Polsby-Popper histogram

To plot an ensemble compactness histogram from `atlas-posthoc` output:

```toml
[inputs]
polsby_traces_json = "atlas-posthoc-output/diagnostics/polsby_popper_traces.json"
public_plan_dir = "inputs/plans/public"         # optional, for public overlays
graph_json = "inputs/graph.json"                # needed to score public plans
crosswalk_csv = "inputs/block_to_precinct.csv"  # needed to score public plans
# public_polsby_scores_csv = "outputs/polsby/public-plan-scores.csv"
# Optional: use this only if you already have a precomputed public score CSV.

[workflow.polsby-histogram]
stat = "average"  # average, minimum, or maximum
burnin = 0
bins = 40
public_score_method = "graph"  # or "blocks" for slower block-geometry scoring
refresh_public_scores = false
```

Then run:

```bash
atlas-posthoc-plus run polsby-histogram
```

The official overlay can use either a legacy `label,score` CSV or the
`official-polsby` summary CSV with `plan,average,minimum,maximum` columns.
Public submitted plans are discovered from `inputs.public_plan_dir`, mapped to
graph nodes through the block-to-precinct crosswalk, scored with
`atlas-posthoc`'s graph compactness audit, and cached at
`outputs/polsby/public-plan-scores.csv` by default. Re-running the workflow
reuses that cache unless `refresh_public_scores = true`. Set
`public_score_method = "blocks"` only when you want slower block-geometry
scoring.

## Comparing arbitrary municipal trace collections

`municipal-tradeoff-weights`, `municipal-tradeoff-weights-external`,
`municipal-outside-population-marginals`, and `municipal-cut-mcd-histograms`
can compare any number of `atlas-posthoc` municipal trace outputs. In the TOML
file, use a label-to-path table:

```toml
[inputs.municipal_trace_collections]
"Congress base chain" = "atlas-posthoc/base/diagnostics/municipal_split_traces.json"
"Higher MCD penalty" = "atlas-posthoc/mcd20/diagnostics/municipal_split_traces.json"
"External lab ensemble" = "atlas-posthoc/external/diagnostics/municipal_split_traces.json"
```

The table key is used as the legend label. The same thing can be passed directly
on the command line:

```bash
atlas-posthoc-plus run municipal-tradeoff-weights -- \
  --trace-collection "Congress base chain=atlas-posthoc/base/diagnostics/municipal_split_traces.json" \
  --trace-collection "Higher MCD penalty=atlas-posthoc/mcd20/diagnostics/municipal_split_traces.json"
```

For the requested marginal views, use:

```bash
atlas-posthoc-plus run municipal-outside-population-marginals
atlas-posthoc-plus run municipal-cut-mcd-histograms
```

## Place fragmentation and partisanship

`municipal-place-partisanship` compares every official Place ousted-population
share with the distribution for that same Place across county-compliant ensemble
plans. It requires the PLACE graph arm and raw chain assignments, because the
aggregate municipal trace JSON does not retain place-level values.

Configure the PLACE arm with the graph and block checkpoint used to create it:

```toml
[municipality]
mode = "PLACE"
graph_json = "data/graph/OH_pct20.json"
mapping_column = "PLACE"
place_shapefile = "data/raw/tl_2024_39_place/tl_2024_39_place.shp"
output_namespace = "place"

[inputs]
graph_json = "data/graph/OH_pct20.json"
block_metadata = "data/derived/cblocksMergeMultiPcts_OH.gpkg"

[workflow.municipal-place-partisanship]
minimum_place_population = 25000
label_count = 3
```

The configured chains must also have been sampled from `OH_pct20.json`. Then run:

```bash
atlas-posthoc-plus run municipal-place-partisanship
```

The workflow writes two named figures under the PLACE output namespace:

- `place-ousted-share-by-partisanship.png`: the official value, ensemble median,
  and ensemble middle 50% for each Place.
- `place-ousted-share-residual-by-partisanship.png`: official ousted share minus
  the place-specific ensemble median, centered at zero.

It also writes `place-partisanship-summary.csv` with the five ensemble quantiles,
official empirical percentile, vote allocation, and residual for every Place and
official plan. Graph-node votes are allocated to Places in proportion to the Place
population represented inside each node.

## City partisan context

`municipal-city-partisan-context` asks whether official plans place residents of
selected cities into unusually Democratic or Republican districts. For each city,
it represents a plan by the Democratic two-party share of the district containing
each overlapping graph unit. Graph-unit coordinates are weighted by their city
population and the weights sum to one, so the resulting metrics are average
vote-share shifts per city resident rather than totals that grow with city size.

Configure the cities and run the workflow with the same PLACE graph and chains:

```toml
[workflow.municipal-city-partisan-context]
cities = ["Cleveland", "Columbus", "Cincinnati", "Toledo", "Akron", "Dayton"]
bins = 35
```

```bash
atlas-posthoc-plus run municipal-city-partisan-context
```

The workflow writes one two-panel figure per city. The left panel shows each
plan's population-normalized absolute distance from the ensemble centroid; the
right shows its signed shift. Positive shifts place city residents in more
Democratic district contexts than the ensemble expectation and are
packing, while negative shifts are cracking. These are
directional diagnostics, not by themselves proof of intent. CSV outputs retain
the ensemble distributions, official-plan percentiles and tail probabilities,
and the population weights and centroid coordinates used in the calculations.
For graph units created by state-legislative splits, votes are population-
allocated so each original precinct contributes exactly once. Official block
assignments are projected through the same graph units using the exact Place
population recorded for each block.

The older `inputs.traces_json_mcd20`, `inputs.traces_json_mcd5`, and
`inputs.traces_json_mcd1` keys are still supported as compatibility shortcuts.

If `reuse_official_summary` or `reuse_public_summary` is enabled but the cached
JSON does not exist yet, the workflow recomputes the missing summary from the
configured plan/block inputs and writes it under `outputs.root`. To make a
collection-only plot without official-plan overlays, pass `--skip-official-plans`
after `--`.

## Workflow inventory

| Workflow | Packaged module | Result |
|---|---|---|
| `build-crosswalk` | `workflows.setup.crosswalk` | Block-to-precinct crosswalk |
| `vote-comparison` | `workflows.standard.vote_comparison` | Ordered vote-share comparison |
| `vote-comparison-adjusted` | `workflows.standard.vote_comparison_adjusted` | Ordered vote-share comparison |
| `municipal-histograms-compliant` | `workflows.municipal.compliant_histograms` | Compliant traces and reference histograms |
| `district-rank-municipal` | `workflows.municipal.district_rank` | Municipal burden by ranked district |
| `district-rank-municipal-compliant` | `workflows.municipal.district_rank_compliant` | Municipal burden by ranked district |
| `cut-edges-compliant` | `workflows.compactness.cut_edges` | Cut-edge distribution |
| `official-polsby` | `workflows.compactness.official_polsby` | Official-plan compactness CSVs |
| `polsby-trace` | `workflows.compactness.polsby_trace` | Compactness trace with references |
| `polsby-histogram` | `workflows.compactness.polsby_histogram` | Compactness histogram with public/official overlays |
| `ranked-vote-cvap` | `workflows.demographics.ranked_vote_cvap` | Vote/CVAP ranked comparison |
| `responsiveness` | `workflows.elections.responsiveness` | Multi-election responsiveness |
| `responsiveness-zoom` | `workflows.elections.responsiveness_zoom` | Zoomed responsiveness panels |
| `municipal-histograms` | `workflows.municipal.histograms` | Pooled municipal histograms |
| `municipal-distributions` | `workflows.municipal.distributions` | Chain-wise municipal distributions |
| `municipal-tradeoff` | `workflows.municipal.tradeoff` | Cut/ousted-population tradeoff |
| `municipal-outside-population-marginals` | `workflows.municipal.outside_population_marginals` | Multi-ensemble ousted-population marginals |
| `municipal-cut-mcd-histograms` | `workflows.municipal.cut_mcd_histograms` | Multi-ensemble cut-MCD histograms |
| `municipal-tradeoff-public` | `workflows.municipal.tradeoff_public` | Public-plan tradeoff |
| `municipal-tradeoff-weights` | `workflows.municipal.tradeoff_weights` | Multi-collection municipal tradeoff |
| `municipal-tradeoff-weights-external` | `workflows.municipal.tradeoff_weights_external` | Multi-collection tradeoff with an external ensemble |
| `municipal-place-partisanship` | `workflows.municipal.place_partisanship` | Place-specific ensemble quartiles and partisan fragmentation residuals |
| `municipal-city-partisan-context` | `workflows.municipal.city_partisan_context` | Population-normalized city partisan-context distances and signed shifts |
| `inspect-external-ensemble` | `workflows.external.inspect_ensemble` | External-ensemble input inspection |
| `summarize-external-ensemble` | `workflows.external.summarize_ensemble` | External-ensemble municipal summaries |

## Development checks

```bash
pytest -q
python3 -m compileall -q src tests
python3 -m pip wheel . --no-deps
```

## AI Usage Acknowledgement

We acknowledge the use of OpenAI Codex 5.6 Sol, under the direction and rigorous human reviewing of Franklin, in bringing loose scripts to this package and enabling the one-line call of analysis/plotting suites. The functionality of this package is from a handful of loose scripts in `oh-gerrymandering` repository. 
