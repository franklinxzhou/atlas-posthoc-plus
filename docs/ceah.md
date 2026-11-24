# CEAH plot

Run these commands from your analysis project. They use
`atlas-posthoc-plus.toml` and the default output paths. For another config,
put `--config your-analysis.toml` before `run` and use that config's paths.

Run `atlas-posthoc-plus run ceah` to compute election results and draw the
standalone plot. Run `atlas-posthoc-plus run ceah-zoom` to redraw the saved
results with linked vote-share detail panels. Each stepped histogram shows
relative ensemble frequencies for one election and is scaled to its own peak.
The baseline is the statewide two-party Democratic vote share. Official-plan
markers sit at the corresponding integer seat count. Plans that elect the
same number of Democrats appear as colored sectors of one dot. Race codes
sit beside their histograms, using both sides where the rows are close.
The boxed legend sits inside the overview, at bottom right when there is
room and at top left otherwise. Nimbus Sans is preferred when installed.

The default figure names are `responsiveness/ceah.png` and
`responsiveness/ceah-zoom.png` under `outputs.root`. The previous command
names `responsiveness` and `responsiveness-zoom` are accepted as aliases.
Their TOML settings and existing CSV files still work. Update any external
references to the default figure filenames.

## Find race IDs

Open `outputs/responsiveness/oh_statewide_race_manifest_used.csv` and look at
the `race_id` column. You can print the IDs from a terminal with:

```bash
cut -d, -f2 outputs/responsiveness/oh_statewide_race_manifest_used.csv | tail -n +2
```

For a first run, see the [built-in race manifest](../src/atlas_posthoc_plus/workflows/elections/responsiveness.py).
For example, `G20PRE` is 2020 President, `G24PRE` is 2024 President, and
`G24USS` is 2024 U.S. Senate. Each race needs its corresponding election layer.

## Try different figures from existing CSVs

Start with one race.

```bash
atlas-posthoc-plus run ceah-zoom -- \
  --race-ids G20PRE \
  --output outputs/responsiveness/2020-president.png
```

Then compare a few races. Pass the IDs after one `--race-ids` flag, separated
by spaces. Use a separate output filename for each view.

```bash
atlas-posthoc-plus run ceah-zoom -- \
  --race-ids G20PRE G24PRE G24USS \
  --zoom-panels 2 --max-races-per-zoom 2 \
  --output outputs/responsiveness/president-and-senate.png
```

The zoom command reads the saved CSVs. It reports an error if a requested ID
is missing from the race summary. Isolated elections appear once in the
overview. A race gets a detail view when its nearest vote share is within
5.5% of the overview's vote-share span. The overview seat axis runs from
0 through the number of districts. Each zoom panel uses the seat range
observed for its own races. Set `--zoom-panels` to target a number
of detail panels and `--max-races-per-zoom` to limit their size. Elections
with identical vote shares stay together. Add `--paper-mode` before `run`
to save a vector PDF.
The red outlines all span the same width in the overview. Their heights
follow the vote-share ranges they link to detail panels.
If you also plot public submissions, the companion figure uses the same
histogram design and shows the distribution of submitted seat counts.
Use `--public-output` to give that figure its own filename.

To see the resolved command before running it, try:

```bash
atlas-posthoc-plus run ceah-zoom --dry-run -- --race-ids G20PRE G24USS
```

## Compute only the selected races

Copy your analysis TOML and give the copy a separate `[outputs]` root, such as
`outputs/responsiveness-subset`. If the config specifies individual CSV or
figure paths, point those at the new directory as well.

```bash
cp atlas-posthoc-plus.toml selected-races.toml
# Edit the output paths in selected-races.toml before running.
atlas-posthoc-plus --config selected-races.toml run ceah -- \
  --race-ids G20PRE G24USS
atlas-posthoc-plus --config selected-races.toml run ceah-zoom
```

The first command allocates votes and evaluates plans for the requested races.
The second plots all the races present in the new CSVs. To save a selection
in TOML, add `race_ids = ["G20PRE", "G24USS"]` under
`[workflow.ceah]` for computation or `[workflow.ceah-zoom]`
for plotting.
