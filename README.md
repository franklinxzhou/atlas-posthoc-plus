[![Test CI](https://github.com/franklinxzhou/atlas-posthoc-plus/actions/workflows/test-ci.yml/badge.svg)](https://github.com/franklinxzhou/atlas-posthoc-plus/actions/workflows/test-ci.yml)

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
        ├── compositor.py              # vector PDF figure composition
        ├── setup/
        │   └── crosswalk.py
        ├── standard/
        │   ├── vote_comparison.py
        │   └── vote_comparison_adjusted.py
        ├── municipal/
        │   ├── city_partisan_context.py
        │   ├── compliant_histograms.py
        │   ├── cut_mcd_histograms.py
        │   ├── district_rank.py
        │   ├── district_rank_compliant.py
        │   ├── distributions.py
        │   ├── geography.py
        │   ├── histograms.py
        │   ├── marginal_helpers.py
        │   ├── outside_population_marginals.py
        │   ├── place_partisanship.py
        │   ├── tradeoff.py
        │   ├── tradeoff_public.py
        │   ├── tradeoff_weights.py
        │   └── tradeoff_weights_external.py
        ├── compactness/
        │   ├── cut_edges.py
        │   ├── official_polsby.py
        │   ├── polsby_histogram.py
        │   ├── polsby_trace.py
        │   ├── reock.py
        │   └── reock_histogram.py
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

For development, run these commands from the directory containing both the
`atlas-posthoc` and `atlas-posthoc-plus` checkouts:

```bash
python3 -m pip install -e ./atlas-posthoc
python3 -m pip install -e ./atlas-posthoc-plus
```

The commands `atlas-posthoc-plus` and `aph-plus` are equivalent.

## Configuration

All workflows use one TOML configuration:

```bash
atlas-posthoc-plus init analysis.toml
```

The generated template uses placeholder paths and a COUSUB baseline. Point
`[project].root` at your analysis project, then update the input paths and chain
IDs for the workflows you want to run. A relative `project.root` is resolved
from the TOML file's directory. Other relative paths are resolved from the
project root. Pass `--config analysis.toml` before the command to use the
generated file. The CLI defaults to `atlas-posthoc-plus.toml`.

The checked-in `atlas-posthoc-plus.toml` assumes a sibling `oh-gerrymandering`
checkout with the Ohio data and the four `run37`–`run40` chains.

Important sections:

- `[chains]`: AtlasIO chain files and chain labels.
- `[analysis]`: district count, compliance rule, thinning, and proposal count.
- `[inputs]`: graph, crosswalk, block/MCD geometry, CVAP, and derived files.
- `[municipality]`: explicit COUSUB or PLACE graph/schema bindings and output namespace.
- `[plans]`: enacted, overturned, and remedial block-assignment files.
- `[elections.layers]`: election layers keyed by year.
- `[outputs]`: the root directory for generated results.
- `[figures]`: optional manuscript-oriented figure rendering.
- `[workflow.NAME]`: workflow-specific options and extra CLI arguments.

### Paper mode

The reference source for arXiv 2203.06552 uses PDF figure assets. Inspection of
those assets shows Helvetica Neue/Helvetica text (with STIX for mathematical
glyphs), while the manuscript itself loads `newtxmath`. Enable the same
manuscript-oriented figure treatment globally with:

```toml
[figures]
paper_mode = true
```

You can also override the configuration for one invocation:

```bash
atlas-posthoc-plus --paper-mode run municipal-histograms
atlas-posthoc-plus --no-paper-mode suite standard
```

In paper mode, every plotting workflow:

- removes axes titles and figure-level titles;
- uses Helvetica Neue with portable Helvetica/Arial/sans-serif fallbacks;
- writes a vector `.pdf`, even when the configured output name ends in `.png`;
- embeds TrueType text in the PDF; and
- labels two- and four-panel figures `(a)`, `(b)`, `(c)`, and `(d)` at the
  upper-left of each panel.

Axis labels, tick labels, legends, and annotations remain intact so each figure
still identifies its marks without relying on the manuscript caption.

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
place_shapefile = "data/raw/tl_2024_39_place/tl_2024_39_place.shp"
output_namespace = "place"

[inputs]
graph_json = "data/graph/OH_pct20.json"
# This is the block-geometry checkpoint used by graph generation. Using the
# raw TIGER block polygons changes the represented denominator by 26 people.
block_metadata = "data/derived/cblocksMergeMultiPcts_OH.gpkg"
```

Update `[chains].files` and `[chains].ids` to use chains sampled on that PLACE
graph. PLACE graph nodes are exploded into weighted Place-population pieces.
Official, remedial, and public plans assign Census blocks to Places represented
in the selected graph using each block's interior representative point. The
adapter checks that the assigned population totals agree with the graph. It
automatically matches the Place layer's `GEOID` or `NAME` field to the graph's
keys. Set `place_key_column` to override the inferred field. Municipality-scoped
default outputs are written under `outputs.root` in the `output_namespace`
subdirectory, and cache metadata prevents cross-arm reuse.

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

To hide which chain produced each box and show one pooled ensemble instead:

```toml
[workflow.vote-comparison-adjusted]
pool_chains = true
ensemble_label = "Place-5 ensemble"
overlay_public_plans = true
```

The pooling affects only the figure; the per-chain compliance summary and
`compliant_ensemble_ordered_vote_vectors.csv` retain their chain IDs. With
`overlay_public_plans = true`, every valid CSV under `inputs.public_plan_dir`
is evaluated using the same election and drawn as small red dots at its ordered
district ranks.

For a one-off pooled PDF without changing the TOML:

```bash
atlas-posthoc-plus --paper-mode run vote-comparison-adjusted -- \
  --pool-chains --ensemble-label "Place-5 ensemble"
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

## Composing figures

The registered `figure-compositor` workflow combines existing paper-mode PDF
figures without rasterizing them. Every panel names its source using a canonical
workflow inventory name and may point to a different project TOML:

```toml
[outputs]
figure_compositor = "outputs/manuscript/structural-comparison.pdf"

[workflow.figure-compositor]
design = '''
AAB
#CB
'''
widths = [1, 1, 0.85]
heights = [1, 1.15]
width = 10
height = 6.5
margin = 0.25
gap = 0.15

[[workflow.figure-compositor.panels]]
id = "A"
workflow = "polsby-histogram"
config = "configs/baseline.toml"
figure = "outputs/polsby/polsby-histogram.pdf"

[[workflow.figure-compositor.panels]]
id = "B"
workflow = "responsiveness"
config = "configs/baseline.toml"
figure = "outputs/responsiveness/ohio_responsiveness_all_statewide_races.pdf"

[[workflow.figure-compositor.panels]]
id = "C"
workflow = "municipal-tradeoff"
config = "configs/place.toml"
figure = "outputs/place/municipal/municipal-tradeoff.pdf"
```

Save this example as `compose.toml` in your project root and replace the
example `config` paths with your source TOML files. The example
figure paths assume each source uses `outputs.root = "outputs"` and the PLACE
source uses `municipality.output_namespace = "place"`. Generate the source
figures with paper mode enabled, then validate and compose:

```bash
atlas-posthoc-plus --config compose.toml doctor figure-compositor
atlas-posthoc-plus --config compose.toml run figure-compositor
```

In a compact design, repeated IDs span cells and `#` or `.` creates a spacer.
Whitespace-delimited rows may be used for multi-character IDs. Omit `design`
and set `ncol` or `nrow` for automatic row-major wrapping; `ncol = 1` stacks
panels and `nrow = 1` places them side by side. `widths` and `heights` set
relative grid sizes. The compositor centers each source page without changing
its aspect ratio, applies consistent margins and gaps, and adds `(a)`, `(b)`,
... in panel declaration order.

Panel `figure` paths are resolved against the `project.root` in that panel's
source `config`; omit `config` to use the compositor's TOML. Optional panel
keys include one-based `page`, `horizontal_align` (`left`, `center`, `right`),
`vertical_align` (`bottom`, `center`, `top`), and `tag` (a custom string or
`false`). Global controls include `tags`, `tag_size`, `tag_weight`,
`tag_offset_x`, `tag_offset_y`, `font_family`, `column_gap`, `row_gap`, and
per-edge `margin_left`, `margin_right`, `margin_top`, and `margin_bottom`.
Inputs and output must be PDF files so source vector marks, text, and fonts are
preserved.

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
  multi-ensemble marginal plots, public-plan comparison, and Place/city
  partisanship diagnostics.
- `external`: inspect and summarize the optional outside ensemble.
- `full`: `standard`, both compactness histograms, `municipal`, and the zoomed responsiveness figure.

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
- `polsby-trace` exports the plan scores used by `polsby-histogram`. The
  histogram can overlay the `official-polsby` summary CSV and compute and cache
  public-plan scores when the plan directory, graph, and crosswalk are configured.
- `reock-histogram` uses the shared chain sweep and the polygon layer for the
  graph nodes. It can score reference plans from the configured block assignments.
- `responsiveness` creates the CSV files used by `responsiveness-zoom`. To add
  public plans to the zoomed figure, set
  `workflow.responsiveness.overlay_public_plans = true`. The upstream workflow
  then creates `responsiveness/oh_responsiveness_public_plan_seats.csv`. The
  zoom workflow reads it and writes two separate overview-plus-zoom source figures:
  the existing `responsiveness/responsiveness-zoom.png` ensemble figure and a
  companion `responsiveness/responsiveness-zoom-public-plans.png` public-plan
  figure. Public plans with the same election and integer Democratic seat
  count are collapsed into one constant-size muted tile; exact counts are
  labelled inside the tiles in the zoom axes (the compact overview suppresses
  those labels). The public legend identifies a “Submitted-plan outcome” tile
  and notes that zoom labels give the number of submitted plans. The seat
  coordinate is never jittered or smoothed. Set
  `outputs.responsiveness_public_zoom` to choose another public figure path.

For a clearer public-plan comparison, `vote-comparison-adjusted` also writes
`vote-comparison-adjusted/oh_public_plans_ordered_vote_shares.png` when
`workflow.vote-comparison-adjusted.overlay_public_plans = true`. This figure
uses the same election as the vote comparison (2020 presidential by default).
Districts are sorted within each plan from most Republican to most Democratic.
Each box summarizes the submitted plans at that rank: the box spans the middle
50%, the line marks the median, and whiskers span the full range. The three
official plans are overlaid with distinct colors, line styles, and marker
shapes; they do not contribute to the boxes. A dashed line marks the 50% winning
threshold. Set `outputs.public_vote_comparison` to choose another output path.

The checked-in `figure-compositor` configuration combines the paper-mode PDF
ensemble responsiveness figure as panel `(a)` with this public-plan ordered
vote-share figure as panel `(b)` without rasterizing them, and writes
`responsiveness/responsiveness-zoom-composite.pdf`:

```bash
atlas-posthoc-plus --paper-mode run responsiveness-zoom
atlas-posthoc-plus --paper-mode run vote-comparison-adjusted
atlas-posthoc-plus doctor figure-compositor
atlas-posthoc-plus run figure-compositor
```

To redraw only the public boxplot from saved ordered vote vectors:

```bash
python -m atlas_posthoc_plus.workflows.elections.public_vote_comparison \
  --ordered-votes outputs/vote-comparison-adjusted/external_plan_ordered_vote_vectors.csv \
  --output outputs/vote-comparison-adjusted/oh_public_plans_ordered_vote_shares.png
```

These saved vectors must include the public plans; generate them once with the
public overlay enabled. The boxes describe submissions, not a neutral ensemble
or a probability of drawing a plan.

Explicit paths in `[inputs]` override these generated defaults.

The public-plan overlays for both vote comparison and responsiveness can be
enabled together with:

```toml
[workflow.vote-comparison-adjusted]
overlay_public_plans = true

[workflow.responsiveness]
overlay_public_plans = true
```

For responsiveness, run `responsiveness` before `responsiveness-zoom` at least
once after enabling the option so the public-plan seat-count CSV exists. The
public figure's zoom labels give the number of submitted plans; these
descriptive counts are not probabilities or densities.

For `municipal-tradeoff-public`, sampled plans can be pooled into one visual
series so the plot does not reveal which chain produced each point:

```toml
[workflow.municipal-tradeoff-public]
pool_chains = true
ensemble_label = "Place-5 ensemble"
```

## Polsby-Popper histogram

Run `polsby-trace` to generate `polsby/polsby-popper-traces.json` under
`outputs.root`, or configure a precomputed `atlas-posthoc` trace JSON:

```toml
[inputs]
# polsby_traces_json = "atlas-posthoc-output/diagnostics/polsby_popper_traces.json"
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
If you customize `outputs.polsby_traces_json` for the trace workflow, point
`inputs.polsby_traces_json` at that path for the histogram.
Public submitted plans are discovered from `inputs.public_plan_dir`, mapped to
graph nodes through the block-to-precinct crosswalk, scored with
`atlas-posthoc`'s graph compactness audit, and cached at
`outputs/polsby/public-plan-scores.csv` by default. Re-running the workflow
reuses that cache unless `refresh_public_scores = true`. Set
`public_score_method = "blocks"` only when you want slower block-geometry
scoring.

## Reock histogram

`reock-histogram` reads `chains.files` and `inputs.graph_json`. Set
`inputs.precincts` to the polygon layer that represents the graph's nodes.
`columns.precinct_name` selects the geometry key in that layer and
`graph_geometry_key` selects the matching graph column. Both default to `NAME`.
Every graph node needs a unique key and matching geometry. Multiple features
with the same geometry key are dissolved before scoring. Reock uses a projected
CRS for both area and distance. Set `analysis.reock_projected_crs` to override
the default `EPSG:5070`.

```toml
[workflow.reock-histogram]
stat = "average"  # average, minimum, or maximum across districts
burnin = 0       # first proposal to include
thin_step = 1
bins = 40
```

```bash
atlas-posthoc-plus run reock-histogram
```

The figure is saved to `reock/reock-histogram.png` under `outputs.root`.
The workflow scores any configured `[plans]` assignments from
`inputs.block_shapefile`. To score submitted block plans, set
`inputs.reock_public_plan_dir`. To use precomputed Reock overlays, set
`inputs.public_reock_scores_csv` and/or `inputs.official_reock_scores_csv`.
These CSVs use the same schemas as the Polsby-Popper overlays.

### Comparison with Greg's scripts

- **Polsby-Popper.** Both implementations use `4πA/P²`. Greg's graph-style
  calculation sums unit areas, adds each exterior border once, and adds
  each cut border to both districts. The package's graph-based public scoring
  applies the same rule after mapping block plans to graph nodes through a
  weighted crosswalk. It skips edges touching unmatched nodes. The ensemble
  trace converts the sampler's stored `P²/A` to Polsby-Popper. The official
  and optional block-based public calculations measure area and perimeter after
  dissolving projected block polygons, so agreement with graph scores depends
  on matching assignments, geometry, and measurement conventions.
- **Greg's border reader.** A descending FID pair `(a, b)` is stored as
  `(b, b)` in `getFidPairToData`. The scoring loop treats that pair as internal
  and omits the boundary. On a two-district fixture, correctly ordered borders
  matched the dissolved scores. Reversing one cut edge changed the first
  district's score from 0.698 to 1.005 in Greg's implementation.
- **Joining and dissolving geometry.** Greg maps GEOIDs to feature IDs, fills
  a plan array by FID, and then dissolves the cluster layer by district in its
  source CRS. Correct alignment depends on the FIDs matching shapefile row
  positions. The package's block workflow joins assignments by normalized
  GEOID, reports the match rate, checks the district count, and dissolves
  after projecting. The Reock ensemble workflow joins the unit layer to graph
  nodes by key and requires geometry for every node.
- **Reock.** The new workflow uses Greg's ratio of dissolved district area
  to the area of the minimum enclosing circle of its convex hull. Shapely/GEOS
  supplies the circle radius, and the area is calculated as `πr²`. The scores
  matched Greg's circle helper on a rectangle, a triangle, a multipart district,
  and a polygon with a hole. Greg's `pairDownReock` keeps plans only when every
  district clears a threshold. The histogram includes all sampled plans after
  thinning and burn-in, preserving the low-score tail.

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

The older `inputs.traces_json_mcd20`, `inputs.traces_json_mcd5`, and
`inputs.traces_json_mcd1` keys are still supported as compatibility shortcuts.

If `reuse_official_summary` or `reuse_public_summary` is enabled but the cached
JSON does not exist yet, the workflow recomputes the missing summary from the
configured plan/block inputs and writes it under `outputs.root`. To make a
collection-only plot without official-plan overlays, pass `--skip-official-plans`
after `--`.

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

## City-level district partisanship

`municipal-city-partisan-context` compares the districts containing residents of
selected cities under official and simulated plans. Configure the cities and run
it with the same PLACE graph and chains:

```toml
[workflow.municipal-city-partisan-context]
cities = ["Cleveland", "Columbus", "Cincinnati", "Toledo", "Akron", "Dayton"]
bins = 35
output_mode = "signed"
```

```bash
atlas-posthoc-plus run municipal-city-partisan-context
```

Each city produces one signed-shift panel at `<city>-city-partisan-context.png`,
with the population-weighted ensemble statistic, cracking and packing shading,
and official-plan reference lines. Paper mode uses the same stem with vector
PDF output. `output_mode` defaults to `"signed"`; legacy `"duo"` and `"both"`
settings also produce only this signed figure. The absolute-distance panel and
its CSV statistics have been removed. The workflow also writes CSV audit tables. See
[City-level district partisanship methodology](docs/municipal-city-partisan-context.md)
for the formulas, population weighting, vote allocation, official-plan
projection, interpretation, and output definitions.

The `regional-african-american-cvap` workflow writes separate CUA and MSA
district-level forest plots. Each metro has one row per official plan, and the
row states `k`, its number of substantive touching districts. Numbered official
symbols are the actual districts, ranked within that plan and metro from lower
to higher African American CVAP location quotient. The two aligned columns show
regional concentration and whole-district African American CVAP share. Each
navy interval is computed from ensemble plans with the same `k` and the same
rank as its official symbol; thin and thick segments show the middle 90% and
50%. By default, intersections containing less than 1% of a region's CVAP
remain in the audit CSV but do not enter the figure;
configure `workflow.regional-african-american-cvap.minimum_region_cvap_share`
to change that threshold.

Enable the six commented regional inputs in either example config to run this
workflow. Set `cua_graph_json` and `msa_graph_json` to graphs with the same node
IDs and order as the configured chains, and supply their matching
`cua_manifest` and `msa_manifest` CSVs. Set `region_blocks` to a block layer
with `GEOID20`, `UACE20`, and `POP20` fields and `cbsa_file` to the CBSA geometry
layer covering the manifest's metro areas.

## Workflow inventory

| Workflow | Packaged module | Result |
|---|---|---|
| `figure-compositor` | `workflows.compositor` | Vector-preserving multi-figure PDF composition |
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
| `reock-histogram` | `workflows.compactness.reock_histogram` | Reock histogram with public/official overlays |
| `ranked-vote-cvap` | `workflows.demographics.ranked_vote_cvap` | Vote/CVAP ranked comparison |
| `regional-african-american-cvap` | `workflows.demographics.regional_black_cvap` | District-level CUA/MSA African American-CVAP location quotients |
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
| `municipal-city-partisan-context` | `workflows.municipal.city_partisan_context` | Population-normalized signed shifts in city district partisanship |
| `inspect-external-ensemble` | `workflows.external.inspect_ensemble` | External-ensemble input inspection |
| `summarize-external-ensemble` | `workflows.external.summarize_ensemble` | External-ensemble municipal summaries |

## Development checks

```bash
pytest -q
python3 -m compileall -q src tests
python3 -m pip wheel . --no-deps
```

## AI Usage Acknowledgement

We acknowledge the use of OpenAI Codex 6.1 Sol to help turn scripts from the
`oh-gerrymandering` repository into this package and make its analysis and
plotting suites runnable with a single command. Franklin directed and reviewed
the work.
