# Select races for responsiveness

Run these commands from your analysis project. They use
`atlas-posthoc-plus.toml` and the default output paths. For another config,
put `--config your-analysis.toml` before `run` and use that config's paths.

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
atlas-posthoc-plus run responsiveness-zoom -- \
  --race-ids G20PRE \
  --output outputs/responsiveness/2020-president.png
```

Then compare a few races. Pass the IDs after one `--race-ids` flag, separated
by spaces. Use a separate output filename for each view.

```bash
atlas-posthoc-plus run responsiveness-zoom -- \
  --race-ids G20PRE G24PRE G24USS \
  --zoom-panels 2 --max-races-per-zoom 2 \
  --output outputs/responsiveness/president-and-senate.png
```

The zoom command reads the saved CSVs. It reports an error if a requested ID
is missing from the race summary. Add `--paper-mode` before `run` to save a PDF.
If you also plot public submissions, use `--public-output` to give that companion
figure its own filename.

To see the resolved command before running it, try:

```bash
atlas-posthoc-plus run responsiveness-zoom --dry-run -- --race-ids G20PRE G24USS
```

## Compute only the selected races

Copy your analysis TOML and give the copy a separate `[outputs]` root, such as
`outputs/responsiveness-subset`. If the config specifies individual CSV or
figure paths, point those at the new directory as well.

```bash
cp atlas-posthoc-plus.toml selected-races.toml
# Edit the output paths in selected-races.toml before running.
atlas-posthoc-plus --config selected-races.toml run responsiveness -- \
  --race-ids G20PRE G24USS
atlas-posthoc-plus --config selected-races.toml run responsiveness-zoom
```

The first command allocates votes and evaluates plans for the requested races.
The second plots all the races present in the new CSVs. To save a selection
in TOML, add `race_ids = ["G20PRE", "G24USS"]` under
`[workflow.responsiveness]` for computation or `[workflow.responsiveness-zoom]`
for plotting.
