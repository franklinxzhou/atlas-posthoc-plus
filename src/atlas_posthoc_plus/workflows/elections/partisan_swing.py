"""Uniform-swing seat curves and exact areas between the reciprocal curves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LOWER_VOTE_PCT = 50.0
UPPER_VOTE_PCT = 55.0
PARTY_VOTES = ["dem_votes", "rep_votes"]


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
