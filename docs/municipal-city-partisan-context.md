# City-level district partisanship methodology

The `municipal-city-partisan-context` workflow compares how official and
simulated congressional plans assign the residents of selected Census Places to
districts with different Democratic two-party vote shares. It produces two
statistics for each city and plan: a population-normalized absolute distance and
a population-normalized signed shift.

## Notation

For a city, let $i=1,\ldots,M$ index the graph units that contain some of its
population. Define:

- $c_i$: city population represented in graph unit $i$;
- $C=\sum_i c_i$: total represented city population;
- $w_i=c_i/C$: graph-unit population weight, so $\sum_i w_i=1$;
- $a_p(i)$: the district containing graph unit $i$ under plan $p$;
- $v_{pd}$: Democratic two-party vote share of district $d$ under plan $p$;
- $x_{pi}=v_{p,a_p(i)}$: Democratic share of the district containing unit $i$.

Thus, a plan is represented by the vector

$$
x_p=(x_{p1},x_{p2},\ldots,x_{pM}).
$$

For $N$ county-compliant ensemble plans, the ensemble mean for graph unit $i$
is

$$
\bar{x}_i=\frac{1}{N}\sum_{p=1}^{N}x_{pi}.
$$

The ensemble mean is calculated separately for each city because cities overlap
different graph units and have different within-unit population weights.

## Population-normalized absolute distance

For plan $p$, the absolute distance from the ensemble mean is

$$
L_p=\sum_{i=1}^{M}w_i\lvert x_{pi}-\bar{x}_i\rvert.
$$

This is the city-population-weighted average absolute difference between the
plan's district Democratic vote shares and their ensemble averages. Because the
weights sum to one, the statistic remains in vote-share units and does not grow
mechanically with city population. The left histogram shows $L_p$ for the
compliant ensemble plans and marks each official plan's value.

## Population-normalized signed shift

The signed statistic is

$$
S_p=\sum_{i=1}^{M}w_i(x_{pi}-\bar{x}_i).
$$

A positive value means that city residents are assigned, on average, to more
Democratic districts than in the ensemble; the figure labels this direction
`packing`. A negative value means that they are assigned to more Republican
districts; the figure labels this direction `cracking`. The signed histogram
shows $S_p$ for the compliant ensemble plans and marks each official plan's
value. It appears as the right panel of the combined figure and can also be
written as a standalone, city-titled figure with its own y-axis label and
official-plan legend.

These labels describe the direction of the result. They do not by themselves
establish intent, a general mechanism for treating municipalities, or the
electoral effect of the plan.

## Vote allocation for simulated plans

Some graph units are pieces of the same original precinct created to preserve
enacted state-legislative boundaries. Those pieces can carry duplicate copies of
the original precinct's raw votes. Before calculating congressional district
vote shares, the workflow keeps one copy of the original precinct's Democratic
and Republican votes and distributes it across its graph-unit pieces in
proportion to their population. This prevents split graph units from duplicating
votes.

For each simulated plan, the adjusted graph-unit votes are summed within each
congressional district. The Democratic two-party share is then

$$
v_{pd}=\frac{D_{pd}}{D_{pd}+R_{pd}},
$$

where $D_{pd}$ and $R_{pd}$ are the adjusted Democratic and Republican vote
totals in district $d$.

## Official-plan projection

Official plans assign Census blocks rather than ensemble graph units. The
workflow therefore:

1. maps each block back to its graph unit using its precinct name and enacted
   state Senate and House districts;
2. allocates one copy of each original precinct's votes across its blocks in
   proportion to block population;
3. sums those votes by the official block-assignment district to calculate
   official district Democratic shares; and
4. uses the Place-population mapping stored for every block to determine the
   district Democratic share experienced by the city population in each graph
   unit.

If an official congressional boundary splits one graph unit, its official value
for city $c$ is the city-population-weighted mean across the blocks in that
unit:

$$
x_{oi}=\frac{\sum_{b\in i}c_b\,v_{o,a_o(b)}}{\sum_{b\in i}c_b},
$$

where $c_b$ is the city's population in block $b$. This preserves the exact
Place population represented in the graph while allowing official block
assignments to differ within a graph unit.

## Ensemble comparisons

The official-plan CSV reports:

- a midrank ensemble percentile for the absolute distance;
- a midrank ensemble percentile for the signed shift; and
- a one-sided directional tail probability for the observed sign.

For an official value $z$ and ensemble values $z_1,\ldots,z_N$, the midrank
percentile is

$$
\frac{\#\{z_p<z\}+\tfrac{1}{2}\#\{z_p=z\}}{N}.
$$

The directional tail probability uses an add-one correction. For a positive
signed shift $S_o$, it is

$$
\frac{1+\#\{S_p\ge S_o\}}{N+1},
$$

with the inequality reversed for a negative signed shift.

## Outputs

The workflow writes the following files under
`<outputs.root>/<output_namespace>/municipal-city-partisan-context/`:

- `<city>-city-partisan-context.png`: the two ensemble histograms with official
  plan reference lines;
- `<city>-city-partisan-context-signed.png`: the standalone signed-shift
  histogram with symmetric cracking/packing directions and official-plan
  reference lines;
- `city-partisan-context-ensemble.csv`: both statistics for every retained
  ensemble plan and city;
- `city-partisan-context-official.csv`: official statistics, percentiles, tail
  probabilities, and the `packing`/`cracking` direction;
- `city-partisan-context-centroids.csv`: each graph unit's city-population
  weight and ensemble mean district Democratic share;
- `city-partisan-context-ensemble-compliance.csv`: chain-level compliance
  accounting; and
- `municipal-geography.json`: the graph and Place-layer provenance.

Choose which figure files are written with:

```toml
[workflow.municipal-city-partisan-context]
output_mode = "duo" # "duo", "signed", or "both"
```

`"duo"` is the default and preserves the original behavior. `"signed"` writes
only the standalone signed-shift figure, while `"both"` writes both figure
forms. In paper mode, the same stems are written as vector PDFs through the
package's normal figure-saving convention. The CSV and provenance outputs are
written in every mode.
