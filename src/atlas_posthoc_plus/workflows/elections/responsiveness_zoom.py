"""Redraw saved CEAH election results with linked vote-share detail panels."""
from __future__ import annotations

import argparse
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from atlas_posthoc_plus.workflows.elections.ceah_plot import PLAN_COLORS, render_ceah, seat_window
from atlas_posthoc_plus.workflows.elections.responsiveness import plot_races, requested_races
from atlas_posthoc_plus.workflows.elections.responsiveness_progress import Progress


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Draw a CEAH plot with linked detail panels from saved CSVs."
    )
    p.add_argument("--race-summary", required=True, help="Path to oh_responsiveness_race_summary.csv")
    p.add_argument("--ensemble-seats", required=True, help="Path to oh_responsiveness_ensemble_seats.csv")
    p.add_argument("--official-seats", required=True, help="Path to oh_responsiveness_official_plan_seats.csv")
    p.add_argument("--public-seats", default=None, help="Optional public-plan seat-count CSV")
    p.add_argument("--overlay-public-plans", action="store_true")
    p.add_argument("--output", required=True, help="Output PNG/PDF path")
    p.add_argument(
        "--public-output",
        default=None,
        help="Optional public-plan figure output; defaults to <output stem>-public-plans<suffix>",
    )
    p.add_argument("--title", default=None)
    p.add_argument("--n-districts", type=int, default=15)
    p.add_argument("--race-ids", nargs="+", metavar="ID",
                   help="Plot only these race IDs from the existing CSVs")
    p.add_argument("--zoom-panels", type=int, default=4)
    p.add_argument("--max-races-per-zoom", type=int, default=6)
    return p.parse_args()


def build_chunks(
    races: pd.DataFrame, zoom_panels: int, max_races_per_zoom: int
) -> list[pd.DataFrame]:
    """Zoom races whose vote shares are too close for the overview alone."""
    if len(races) < 2:
        return []
    shares = races["statewide_dem_share"].to_numpy(float)
    close = np.diff(shares) <= 0.055 * np.ptp(shares)
    crowded = np.flatnonzero(np.r_[False, close] | np.r_[close, False])
    if not len(crowded):
        return []

    # A detail range must not span a race that was left in the overview only.
    breaks = np.flatnonzero(np.diff(crowded) > 1) + 1
    regions = np.split(crowded, breaks)
    chunks = []
    for region in regions:
        part = shares[region]
        target = int(np.round(zoom_panels * len(region) / len(crowded)))
        minimum = int(np.ceil(len(region) / max_races_per_zoom)) if max_races_per_zoom > 0 else 1
        n_panels = min(max(1, target, minimum), max(minimum, len(region) // 2))
        edges = {
            part.searchsorted(group[-1], side="right") for group in np.array_split(part, n_panels)
        }
        offset = region[0]
        chunks.extend(races.iloc[offset + lo:offset + hi] for lo, hi in pairwise(sorted({0, *edges})))
    return chunks


def public_multiplicities(
    public: pd.DataFrame | None,
    races: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse public plans to one row per race and integer seat outcome.

    The public submissions are a descriptive collection, so each row in the
    result is a count, not a probability or a smoothed density.  The statewide
    vote share comes from ``races`` so that both figure artifacts use exactly
    the same y coordinate for a given election.
    """
    columns = ["race_id", "dem_seats", "statewide_dem_share", "n_plans"]
    if public is None or public.empty:
        return pd.DataFrame(columns=columns)
    missing = {"race_id", "dem_seats"}.difference(public.columns)
    if missing:
        raise ValueError(
            "Public-plan seat data need columns race_id and dem_seats; "
            f"missing {sorted(missing)}"
        )

    clean = public[["race_id", "dem_seats"]].copy()
    clean["dem_seats"] = pd.to_numeric(clean["dem_seats"], errors="coerce")
    clean = clean.loc[np.isfinite(clean["dem_seats"])].dropna(subset=["race_id"])
    if clean.empty:
        return pd.DataFrame(columns=columns)
    if not np.isclose(clean["dem_seats"], np.round(clean["dem_seats"])).all():
        raise ValueError("Public-plan Democratic seat counts must be integers")
    clean["dem_seats"] = np.round(clean["dem_seats"]).astype(int)
    grouped = (
        clean.groupby(["race_id", "dem_seats"], as_index=False)
        .size()
        .rename(columns={"size": "n_plans"})
    )
    race_coords = races[["race_id", "statewide_dem_share"]].drop_duplicates("race_id")
    grouped = grouped.merge(race_coords, on="race_id", how="inner")
    return grouped[columns].sort_values(["statewide_dem_share", "race_id", "dem_seats"])


def plot_from_csv(
    races: pd.DataFrame,
    ensemble: pd.DataFrame,
    official: pd.DataFrame,
    output: Path,
    title: str | None,
    n_districts: int,
    zoom_panels: int,
    max_races_per_zoom: int,
    *,
    public: pd.DataFrame | None = None,
    overlay_public_plans: bool = False,
    public_output: Path | None = None,
) -> None:
    races = plot_races(races)
    if overlay_public_plans and public is None:
        raise ValueError("overlay_public_plans=True requires public seat data")

    race_ids = races["race_id"]
    ensemble = ensemble.loc[ensemble["race_id"].isin(race_ids)]
    official = official.loc[official["race_id"].isin(race_ids)]
    grouped = public_multiplicities(public if overlay_public_plans else None, races)
    chunks = tuple(reversed(build_chunks(races, zoom_panels, max_races_per_zoom)))
    grouped = grouped.loc[grouped["dem_seats"].isin(range(n_districts + 1))]
    coordinates = ["race_id", "dem_seats"]
    positions = pd.concat([
        ensemble[coordinates].drop_duplicates(),
        official.loc[official["source"].isin(PLAN_COLORS), coordinates],
        grouped[coordinates],
    ])
    detail_seat_windows = tuple(
        seat_window(
            positions.loc[positions["race_id"].isin(panel["race_id"]), "dem_seats"],
            n_districts,
        )
        for panel in chunks
    )
    output = Path(output)
    destination = render_ceah(
        races, ensemble, official, output, title, n_districts,
        chunks=chunks, detail_seat_windows=detail_seat_windows,
    )
    print(f"Wrote figure: {destination}", flush=True)

    if overlay_public_plans:
        counts = {
            race_id: dict(zip(rows["dem_seats"], rows["n_plans"]))
            for race_id, rows in grouped.groupby("race_id", sort=False)
        }
        public_output = public_output or output.with_name(
            f"{output.stem}-public-plans{output.suffix}"
        )
        destination = render_ceah(
            races, ensemble, official, public_output, title, n_districts,
            chunks=chunks, public_counts=counts, detail_seat_windows=detail_seat_windows,
        )
        print(f"Wrote figure: {destination}", flush=True)


def main():
    args = parse_args()
    progress = Progress()
    with progress.stage("Reading CEAH CSVs"):
        races = pd.read_csv(args.race_summary)
        selected = requested_races(args.race_ids, races["race_id"])
        if not selected:
            raise ValueError("Race summary contains no races")
        races = races[races["race_id"].isin(selected)]
        ensemble = pd.read_csv(args.ensemble_seats)
        official = pd.read_csv(args.official_seats)
        public = None
        if args.overlay_public_plans:
            if args.public_seats is None:
                raise ValueError("--overlay-public-plans requires --public-seats")
            public = pd.read_csv(args.public_seats)
        progress.log(f"Loaded CEAH inputs for {len(races)} races")
    output = Path(args.output)
    public_output = Path(args.public_output) if args.public_output else None
    with progress.stage("Generating CEAH plots with zoom"):
        plot_from_csv(
            races, ensemble, official, output,
            title=args.title, n_districts=args.n_districts,
            zoom_panels=args.zoom_panels, max_races_per_zoom=args.max_races_per_zoom,
            public=public, overlay_public_plans=args.overlay_public_plans,
            public_output=public_output,
        )


if __name__ == "__main__":
    main()
