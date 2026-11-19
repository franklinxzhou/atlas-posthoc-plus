# Municipal Corporations and Townships

The standalone `mct-diagnostics` workflow compares compliant AtlasIO ensemble plans with the
2021 overturned, 2022 remedial, and 2025 enacted congressional plans. It uses the same Census
block populations and joint membership for both sets of plans.

## Run on the Ohio ensembles

Use the graph, chains, and official block assignment files already named in the project TOML.
Download the Ohio 2020 redistricting archive and set
`workflow.mct-diagnostics.census_geography_header` to that ZIP or its extracted
`ohgeo2020.pl` member. The example TOML selects the codes in the original Census release.
[Ohio 2020 P.L. 94-171 data](https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/Ohio/oh2020.pl.zip)

```toml
[workflow.mct-diagnostics]
census_geography_header = "data/raw/oh2020.pl.zip"
township_lsadc = "44"
unit_definition = "whole"
split_block_policy = "reject"
ensemble_unit_details = false
```

Point `inputs.crosswalk_csv` at the allocation used to construct the chosen
`inputs.graph_json`. The default crosswalk columns are `GEOID20,NAME,POP20`. Each row
reports a 2020 block's population allocated to one graph node identified by its unique
`NAME`. The three crosswalk columns and the graph and AtlasIO key columns can be
overridden in the workflow table. The three `[plans]` files accept the existing XLSX and CSV
block assignment formats. For custom headers, set `assignment_block_col` and
`assignment_district_col` in the workflow table.

```console
aph-plus -c atlas-posthoc-plus.toml doctor mct-diagnostics
aph-plus -c atlas-posthoc-plus.toml --paper-mode run mct-diagnostics
```

The workflow uses `analysis.n_districts`, `analysis.max_districts_per_county`,
`analysis.thin_step` (default 1), and `analysis.max_maps_per_chain`. The limit
applies to examined maps after thinning and before county filtering. Retained maps
carry their chain ID, source line, index, name, and weight. The figures give each
retained map equal weight. The county filter follows the existing ensemble county rule.

## Classification and unit definitions

The P.L. 94-171 geographic header provides the source units and fixed block membership.
Summary levels 160 and 060 describe PLACE and COUSUB entities. Summary level 750 gives each
block's PLACE and COUSUB, county, and population. Published PLACE and COUSUB population totals
are checked against the block sums. [2020 State Redistricting Data technical documentation](https://www2.census.gov/programs-surveys/decennial/2020/technical-documentation/complete-tech-docs/summary-file/2020Census_PL94_171Redistricting_StatesTechDoc_English.pdf)

- PLACE LSADC `25` gives cities and `47` gives villages. Ohio's Census-designated places
  have LSADC `57` and are excluded. Incorporated PLACE records can have class
  `C1`, `C2`, `C5`, `C6`, or `C7`.
- COUSUB townships need the configured township LSADC and a township class of `T1`, `T2`,
  or `T5`. The class check resolves shared LSADC codes and excludes independent
  municipal copies (`C2`, `C5`), inactive townships (`T9`), and nonfunctioning or
  statistical records such as `Z1` or `Z9`. In the original Ohio Census release,
  townships use `44`, including `T5` records with functional status `C`.
  Set `township_lsadc = "47"` for a source in which township records have been recoded
  to `47`. Independent COUSUB villages retain their separate class and remain excluded.
  If the selected code yields no eligible townships, the workflow fails with the code
  mismatch. [Census LSAD codes](https://www2.census.gov/geo/pdfs/reference/LSADCodes.pdf),
  [Census class definitions](https://www.census.gov/library/reference/code-lists/class-codes.html)
- The two classification CSVs record the inclusion decision, source class, LSADC, functional
  status, name, GEOID, and population. The grouped audit is also printed and labels
  unexpected combinations for review.

`whole` groups every incorporated PLACE by its statewide Census GEOID. This measures
statewide municipal cohesion. `county-component` adds the county GEOID to the PLACE key.
That option follows the county-specific focus of Article XIX, §2(B)(4) and §2(C)(2).
COUSUB GEOIDs already encode the county. Disconnected territory is pooled within each chosen
key, and zero-population territory does not create a cut in these population diagnostics.
Article XIX §2(C) refers to contiguous territory, and §2(B)(4)(a) has a separate enclosure
rule for population thresholds. The workflow measures population splits and does not
implement those legal tests. [Ohio Constitution, Article XIX, §2](https://codes.ohio.gov/ohio-constitution/section-19.2)

## Population and overlap

Each positive-population block carries one PLACE and one COUSUB identifier at most. A block
can belong to both eligible categories, either one, or neither. Each unit's dominant district
is the one with the greatest population in the unit, with ties broken by the lowest numeric
district label. The implementation treats gaps within `1e-9` of a person as ties for
fractional allocations.

Every resident is in one of four cases for a plan.

| Ousted from a municipal corporation | Ousted from a township | Counted in the deduplicated total |
| --- | --- | --- |
| No | No | No |
| Yes | No | Yes |
| No | Yes | Yes |
| Yes | Yes | Yes, once |

The conventional municipal corporation total includes the second and fourth rows.
The conventional township total includes the third and fourth rows. Adding these two
totals counts the fourth row twice. The deduplicated total is their population union,
and the overlap is the population in the fourth row. The statewide share uses
11,799,448 residents for the original Census release. The represented share uses
the population with at least one eligible membership. Residents with no eligible
memberships are counted in the statewide denominator and in the first row of the
table. [2020 Census resident population](https://www2.census.gov/programs-surveys/decennial/2020/data/apportionment/apportionment-2020-table02.pdf)

The older PLACE graph dictionaries and COUSUB node scalars lack joint block membership.
The Census header supplies this directly. The crosswalk supplies block-to-node
assignments only when its keys and populations agree with the graph. The project's
`build-crosswalk` command generates area-weighted overlaps, and an original precinct
can map to several sampler nodes. For these cases, export the block, final graph-node key,
and allocated population from the graph-generation checkpoint. Multiple positive
allocations per block require `split_block_policy = "allocated"`. Use that setting with
a verified allocation. If the underlying checkpoint uses area weights, the ensemble
result inherits that sub-block population approximation. Block-based official plans
use the single district label in their assignment file.

Each Census block sum and graph-node sum is checked, using an absolute tolerance of
`1e-5` person for allocation roundoff. Duplicate positive block/node rows, missing
positive blocks, inconsistent Census identifiers, and conflicting official assignments
fail validation. Official assignments may omit zero-population blocks. If an official
file uses a different block vintage or divides a populated 2020 block between districts,
an authoritative allocation to 2020 blocks or finer population pieces is needed.
The workflow requires one district per 2020 block for official plans.

## Outputs and cost

The default destination is `outputs/mct/`. The per-chain JSON traces, ensemble plan
CSV, chain summary, and official-plan CSV carry the scalar metrics. The unit details
CSV includes the three official plans by default. Enabling `ensemble_unit_details`
adds a per-unit row for each retained ensemble plan and can create a large file.
The classification CSVs and metadata record the geography and allocation checks.
The main vector PDF uses the PLACE and COUSUB histogram layout for cut units and
deduplicated ousted population, including the same three official reference lines
and legend. A second vector PDF compares conventional summed and deduplicated
population using the same official-plan colors.

The workflow caches validated population and membership tables under
`.membership-cache` in the output directory. Census source, graph, crosswalk,
and geographic settings determine the cache key. Regenerate the crosswalk if its
source geometry or population allocation changes. The ensemble scorer aggregates
blocks with the same graph node and joint membership once, then scores each
assignment with NumPy arrays. It reads chain records sequentially and retains
the scalar diagnostics without writing a second assignment archive.
