EXAMPLE_CONFIG = """\
[project]
# All relative input paths are resolved from this directory.
root = "."

[chains]
files = [
  "inputs/chains/run1.jsonl.gz",
  "inputs/chains/run2.jsonl.gz",
  "inputs/chains/run3.jsonl.gz",
  "inputs/chains/run4.jsonl.gz",
]
ids = ["run1", "run2", "run3", "run4"]

[analysis]
n_districts = 15
max_districts_per_county = 2
max_maps_per_chain = 1001
output_frequency = 50000
total_proposals = 50000000
ousted_scale = "share"
area_crs = "EPSG:26917"
polsby_projected_crs = "EPSG:5070"

[inputs]
graph_json = "inputs/graph.json"
crosswalk_csv = "inputs/block_to_precinct_crosswalk.csv"
plan_manifest = "inputs/plan_manifest.csv"
blocks = "inputs/blocks.shp"
precincts = "inputs/precincts.shp"
block_metadata = "inputs/blocks.shp"
block_shapefile = "inputs/blocks.shp"
mcd_shapefile = "inputs/mcd.shp"
target_precincts = "inputs/precincts.shp"
cvap_file = "inputs/cvap.zip"
public_plan_dir = "inputs/plans/public"
other_csv = "inputs/external/ensemble.csv"
other_geojson = "inputs/external/units.geojson"
polsby_traces_json = "outputs/atlas-posthoc/diagnostics/polsby_popper_traces.json"
# Optional: use this only if you already have a precomputed public-plan score CSV.
# public_polsby_scores_csv = "outputs/polsby/public-plan-scores.csv"

[inputs.municipal_trace_collections]
"MCD weight 20" = "inputs/traces/mcd20.json"
"MCD weight 5" = "inputs/traces/mcd5.json"
"MCD weight 1" = "inputs/traces/mcd1.json"

[plans]
enacted = "inputs/plans/enacted.xlsx"
overturned = "inputs/plans/overturned.xlsx"
remedial = "inputs/plans/remedial.xlsx"

[columns]
block_id = "GEOID20"
block_county = "COUNTYFP20"
block_population = "POP20"
precinct_name = "NAME"
population = "POP20"
official_key = "ORIG_NAME"
mcd_county = "COUNTYFP"
mcd_id = "COUSUBFP"

[elections.layers]
2016 = "inputs/elections/2016.shp"
2018 = "inputs/elections/2018.shp"
2020 = "inputs/elections/2020.shp"
2022 = "inputs/elections/2022.shp"
2024 = "inputs/elections/2024.shp"

[outputs]
root = "outputs"

[workflow.polsby-trace]
metric = "average"

[workflow.polsby-histogram]
stat = "average"
burnin = 0
bins = 40
density = false
public_score_method = "graph"
refresh_public_scores = false

[workflow.municipal-tradeoff-weights]
reuse_official_summary = true
reuse_public_summary = true

[workflow.municipal-place-partisanship]
minimum_place_population = 25000
label_count = 3

[workflow.municipal-tradeoff-weights-external]
include_other_ensemble = true
reuse_official_summary = true
reuse_public_summary = true
"""
