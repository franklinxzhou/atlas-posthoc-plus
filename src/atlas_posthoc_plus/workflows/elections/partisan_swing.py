"""Uniform-swing seat curves and exact areas between the reciprocal curves."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

LOWER_VOTE_PCT = 50.0
UPPER_VOTE_PCT = 55.0
PARTY_VOTES = ["dem_votes", "rep_votes"]
MAP_ID = ["chain_id", "map_index"]


def read_vote_table(path: Path, identifiers: list[str]) -> pd.DataFrame:
    """Read district or node votes while preserving the text identifiers."""
    votes = pd.read_csv(path, dtype={key: "string" for key in [*identifiers, "race_id"]},
                        keep_default_na=False)
    if missing := set(identifiers).difference(votes):
        raise ValueError(f"{path}: missing columns {', '.join(sorted(missing))}")
    return votes


def prepare_vote_table(
    votes: pd.DataFrame, identifiers: list[str], *, counts_required: bool = False,
) -> pd.DataFrame:
    """Validate input rows once before grouping them into plans or races."""
    votes = votes.copy()
    if "race_id" not in votes:
        votes["race_id"] = "election"
    keys = [*identifiers, "race_id"]
    for key in keys:
        votes[key] = votes[key].astype("string").str.strip()
    if votes.empty or votes[keys].isna().any(axis=None) or votes[keys].eq("").any(axis=None):
        raise ValueError(f"Expected nonempty rows and identifiers in {', '.join(keys)}")
    if votes.duplicated(keys).any():
        raise ValueError(f"Expected one row per combination of {', '.join(keys)}")

    if set(PARTY_VOTES).issubset(votes) or counts_required:
        numeric = PARTY_VOTES
        upper = np.inf
    else:
        numeric = ["dem_share", "statewide_dem_share"]
        upper = 1.0
    if missing := set(numeric).difference(votes):
        raise ValueError(f"Missing vote columns {', '.join(sorted(missing))}")
    votes[numeric] = votes[numeric].apply(pd.to_numeric, errors="coerce").astype(float)
    values = votes[numeric].to_numpy()
    if not (np.isfinite(values) & (values >= 0) & (values <= upper)).all():
        raise ValueError(f"Invalid values in {', '.join(numeric)}")
    return votes[[*keys, *numeric]]


@dataclass(frozen=True)
class Plan:
    label: str
    chain_id: str = ""
    map_index: str = ""

    @property
    def csv_key(self) -> tuple[str, str, str]:
        return self.label, self.chain_id, self.map_index


@dataclass(frozen=True)
class SwingResult:
    edges: np.ndarray
    dem_seats: np.ndarray
    rep_seats: np.ndarray
    n_districts: int
    statewide_shares: dict[str, float]
    thresholds: list[tuple]

    @property
    def segment_areas(self) -> np.ndarray:
        return np.diff(self.edges) * np.abs(self.dem_seats - self.rep_seats)

    @property
    def area(self) -> float:
        return float(self.segment_areas.sum())


def swing_result(votes: pd.DataFrame) -> SwingResult:
    """Apply equal additive swing to district two-party shares at fixed turnout."""
    statewide_shares = {}
    thresholds = []
    dem_thresholds, rep_thresholds = [], []
    districts = frozenset(votes["district"])
    for race, frame in votes.groupby("race_id", sort=True):
        if frozenset(frame["district"]) != districts:
            raise ValueError(f"{race}: each election on a plan must cover the same districts")
        if set(PARTY_VOTES).issubset(frame):
            dem, rep = frame[PARTY_VOTES].to_numpy().T
            turnout = dem + rep
            if not np.isfinite(turnout.sum()) or (turnout <= 0).any():
                raise ValueError(f"{race}: each district must have finite, positive two-party turnout")
            shares = dem / turnout
            statewide = float(dem.sum() / turnout.sum())
        else:
            shares = frame["dem_share"].to_numpy()
            statewide = float(frame["statewide_dem_share"].iloc[0])
            if not np.allclose(frame["statewide_dem_share"], statewide, rtol=0, atol=1e-10):
                raise ValueError(f"{race}: expected one statewide share across the district rows")
            if not shares.min() - 1e-10 <= statewide <= shares.max() + 1e-10:
                raise ValueError(f"{race}: statewide share is outside the district share range")
            dem = rep = [None] * len(frame)
        statewide_shares[race] = statewide
        dem_flip = 50 + 100 * (statewide - shares)
        rep_flip = 100 - dem_flip
        dem_thresholds.extend(dem_flip)
        rep_thresholds.extend(rep_flip)
        thresholds.extend(
            (race, district, d, r, share, statewide, d_flip, r_flip)
            for district, d, r, share, d_flip, r_flip in zip(
                frame["district"], dem, rep, shares, dem_flip, rep_flip, strict=True,
            )
        )

    all_flips = np.asarray(dem_thresholds + rep_thresholds)
    internal = all_flips[(LOWER_VOTE_PCT < all_flips) & (all_flips < UPPER_VOTE_PCT)]
    edges = np.unique(np.r_[LOWER_VOTE_PCT, internal, UPPER_VOTE_PCT])
    # Counts on open intervals give the exact area. Ties at jumps have zero width.
    race_count = len(statewide_shares)
    dem_seats = np.searchsorted(np.sort(dem_thresholds), edges[:-1], side="right") / race_count
    rep_seats = np.searchsorted(np.sort(rep_thresholds), edges[:-1], side="right") / race_count
    return SwingResult(edges, dem_seats, rep_seats, len(districts), statewide_shares, thresholds)


def official_votes(votes: pd.DataFrame) -> Iterator[tuple[Plan, pd.DataFrame]]:
    votes = prepare_vote_table(votes, ["plan", "district"])
    for plan, frame in votes.groupby("plan", sort=False):
        yield Plan(plan), frame


def _check_comparable(result: SwingResult, reference: SwingResult) -> None:
    if result.n_districts != reference.n_districts:
        raise ValueError(f"Expected {reference.n_districts} districts, found {result.n_districts}")
    if result.statewide_shares.keys() != reference.statewide_shares.keys():
        raise ValueError("All plans must cover the same election IDs")
    for race, share in result.statewide_shares.items():
        if abs(share - reference.statewide_shares[race]) > 1e-6:
            raise ValueError(f"{race}: statewide two-party share differs across plans")


def _score_and_write(
    outdir: Path, group: str, plans: Iterable[tuple[Plan, pd.DataFrame]],
    *, reference: SwingResult | None = None,
) -> Iterator[tuple[Plan, SwingResult]]:
    """Write interval-level and district-level audit data without retaining ensemble curves."""
    seen = set()
    key = ["plan", *MAP_ID]
    with (
        (outdir / f"{group}-thresholds.csv").open("w", newline="", encoding="utf-8") as threshold_file,
        (outdir / f"{group}-curves.csv").open("w", newline="", encoding="utf-8") as curve_file,
        (outdir / f"{group}-scores.csv").open("w", newline="", encoding="utf-8") as score_file,
    ):
        threshold_writer, curve_writer, score_writer = map(
            csv.writer, (threshold_file, curve_file, score_file),
        )
        threshold_writer.writerow([*key, "race_id", "district", *PARTY_VOTES, "dem_share",
                                   "statewide_dem_share", "dem_flip_vote_pct", "rep_flip_vote_pct"])
        curve_writer.writerow([*key, "vote_pct_start", "vote_pct_end", "mean_dem_seats",
                               "mean_rep_seats", "abs_seat_gap", "area_seat_pp"])
        score_writer.writerow([*key, "districts", "elections", "race_ids",
                               "vote_pct_start", "vote_pct_end",
                               "area_seat_pp", "mean_abs_seat_gap"])
        for plan, votes in plans:
            identity = (plan.chain_id, plan.map_index) if plan.chain_id else plan.label
            if identity in seen:
                raise ValueError(f"Duplicate {group} plan {identity}")
            seen.add(identity)
            try:
                result = swing_result(votes)
                if reference is not None:
                    _check_comparable(result, reference)
            except ValueError as error:
                raise ValueError(f"{group} plan {identity}: {error}") from error
            if reference is None:
                reference = result
            threshold_writer.writerows((*plan.csv_key, *threshold) for threshold in result.thresholds)
            contributions = result.segment_areas
            curve_writer.writerows(
                (*plan.csv_key, lo, hi, dem, rep, abs(dem - rep), contribution)
                for lo, hi, dem, rep, contribution in zip(
                    result.edges[:-1], result.edges[1:], result.dem_seats, result.rep_seats,
                    contributions, strict=True,
                )
            )
            area = float(contributions.sum())
            score_writer.writerow([*plan.csv_key, result.n_districts, len(result.statewide_shares),
                                   "|".join(result.statewide_shares),
                                   LOWER_VOTE_PCT, UPPER_VOTE_PCT, area,
                                   area / (UPPER_VOTE_PCT - LOWER_VOTE_PCT)])
            yield plan, result
    if not seen:
        raise ValueError(f"No {group} plans to score")


def write_swing_analysis(
    output: Path, official: Iterable[tuple[Plan, pd.DataFrame]],
) -> Path:
    from atlas_posthoc_plus.workflows.elections.partisan_swing_plot import plot_curves

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}-") as temp:
        working = Path(temp)
        officials = list(_score_and_write(working, "official", official))
        if len(officials) != 3:
            raise ValueError("Expected three official plans for the swing analysis")
        plot_curves(working, officials)
        output.mkdir(exist_ok=True)
        for file in working.iterdir():
            file.replace(output / file.name)
    return output
