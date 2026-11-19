EXAMPLE_CONFIG = """\
[project]
# Relative paths use this root, which is relative to this TOML file.
# Replace the example paths and chain IDs with your project's values.
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

# Use COUSUB with a graph whose MCD column holds municipality codes.
# For PLACE workflows, use matching PLACE chains, set mode to "PLACE", set
# mapping_column to the graph's Place mapping column (usually "PLACE"), and
# set place_shapefile to the matching Census Place layer. Change both graph
# paths, set inputs.block_metadata to the graph-generation block checkpoint,
# and choose a distinct output_namespace such as "place".
[municipality]
mode = "COUSUB"
graph_json = "inputs/graph.json"
scalar_column = "MCD"
output_namespace = "cousub"
# mapping_column = "PLACE"
# place_shapefile = "inputs/places.shp"

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
# Optional override for the traces exported by polsby-trace.
# polsby_traces_json = "outputs/atlas-posthoc/diagnostics/polsby_popper_traces.json"
# Optional: use this only if you already have a precomputed public-plan score CSV.
# public_polsby_scores_csv = "outputs/polsby/public-plan-scores.csv"

# For regional-african-american-cvap, supply these six additional inputs.
# Both graphs must match the node IDs and order used by chains.files.
# cua_graph_json = "inputs/regions/cua-graph.json"
# msa_graph_json = "inputs/regions/msa-graph.json"
# cua_manifest = "inputs/regions/cua-manifest.csv"
# msa_manifest = "inputs/regions/msa-manifest.csv"
# region_blocks = "inputs/regions/blocks-with-urban-area-codes.gpkg"
# cbsa_file = "inputs/regions/cbsa.shp"

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

[figures]
# Set true for title-free Helvetica Neue figures saved as vector PDF.
paper_mode = false

# Standalone workflow. Match the 2020 Census block universe and the exact
# node names and allocations used by the graph in inputs.graph_json.
[workflow.mct-diagnostics]
block_metadata = "inputs/census-2020/oh-block-population.csv"
place_layer = "inputs/census-2020/oh-place.shp"
cousub_layer = "inputs/census-2020/oh-cousub.shp"
place_block_assignments = "inputs/census-2020/BlockAssign_ST39_OH_INCPLACE_CDP.txt"
cousub_block_assignments = "inputs/census-2020/BlockAssign_ST39_OH_MCD.txt"
township_lsadc = "44"  # Standard 2020 Ohio Census code. Use "47" for recoded township data.
unit_definition = "whole"  # Or "county-component".
split_block_policy = "reject"  # Or "allocated" for graph-specific population allocations.
ensemble_unit_details = false

[workflow.polsby-trace]
metric = "average"

[workflow.polsby-histogram]
stat = "average"
burnin = 0
bins = 40
density = false
public_score_method = "graph"
refresh_public_scores = false

[workflow.reock-histogram]
stat = "average"
burnin = 0
bins = 40

[workflow.municipal-tradeoff-weights]
reuse_official_summary = true
reuse_public_summary = true

# These two workflows require municipality.mode = "PLACE".
[workflow.municipal-place-partisanship]
minimum_place_population = 25000
label_count = 3

[workflow.municipal-city-partisan-context]
cities = ["Cleveland", "Columbus", "Cincinnati", "Toledo", "Akron", "Dayton"]
bins = 35
output_mode = "signed"

[workflow.municipal-tradeoff-weights-external]
include_other_ensemble = true
reuse_official_summary = true
reuse_public_summary = true

# Reusable PDF composition example. Uncomment after generating the source
# workflows in paper mode. Replace the example config paths with your source
# TOML files. Each figure path is relative to its source config's project.root.
#
# [workflow.figure-compositor]
# design = '''
# AAB
# #CB
# '''
# widths = [1, 1, 0.85]
# heights = [1, 1]
# width = 10
# height = 6.5
#
# [[workflow.figure-compositor.panels]]
# id = "A"
# workflow = "polsby-histogram"
# config = "configs/baseline.toml"
# figure = "outputs/polsby/polsby-histogram.pdf"
#
# [[workflow.figure-compositor.panels]]
# id = "B"
# workflow = "responsiveness"
# config = "configs/baseline.toml"
# figure = "outputs/responsiveness/ohio_responsiveness_all_statewide_races.pdf"
#
# [[workflow.figure-compositor.panels]]
# id = "C"
# workflow = "municipal-tradeoff"
# config = "configs/place.toml"
# figure = "outputs/place/municipal/municipal-tradeoff.pdf"
"""
