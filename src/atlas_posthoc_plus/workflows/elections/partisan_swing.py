"""Uniform-swing seat curves and exact areas between the reciprocal curves."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from atlas_posthoc_plus.io.atlas_records import iter_map_records

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


def ensemble_votes(paths: Iterable[Path]) -> Iterator[tuple[Plan, pd.DataFrame]]:
    identifiers = [*MAP_ID, "district"]
    for path in paths:
        votes = prepare_vote_table(read_vote_table(path, identifiers), identifiers)
        for (chain, index), frame in votes.groupby(MAP_ID, sort=False):
            yield Plan("", chain, index), frame


def chain_votes(
    graph_path: Path, node_votes_path: Path, paths: Iterable[Path],
) -> Iterator[tuple[Plan, pd.DataFrame]]:
    """Stream every saved AtlasIO record and aggregate the same node votes on each map."""
    from atlas_posthoc_plus.workflows.elections.responsiveness import (
        AssignmentIndex,
        DistrictVoteTotals,
        chain_id_from_path,
        read_graph_nodes,
    )

    graph = read_graph_nodes(graph_path)
    assignment_index = AssignmentIndex(graph, "NAME", "id", "COUNTY")
    votes = prepare_vote_table(read_vote_table(node_votes_path, ["node_id"]), ["node_id"],
                               counts_required=True)
    wanted_nodes = pd.Index(assignment_index.node_ids.astype(str), name="node_id")
    if not wanted_nodes.is_unique:
        raise ValueError("Duplicate node IDs in the ensemble graph")
    by_race = votes.pivot(index="node_id", columns="race_id", values=PARTY_VOTES)
    if set(by_race.index) != set(wanted_nodes) or by_race.isna().any(axis=None):
        raise ValueError("Node vote CSV must include every graph node in every election")
    by_race = by_race.reindex(wanted_nodes)
    columns = list(by_race.columns)
    races = by_race.columns.get_level_values("race_id").unique()
    totals = DistrictVoteTotals(by_race.reset_index(), columns)
    for path in paths:
        chain = chain_id_from_path(path)
        count = 0
        for count, record in enumerate(iter_map_records(path, skip_bad_json=False), start=1):
            try:
                districts = assignment_index.districts(record["districting"])
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError(f"{chain} map {count}: {error}") from error
            grouped = totals.for_assignment(districts).rename_axis(index="district")
            frame = pd.concat({race: grouped.xs(race, axis=1, level="race_id") for race in races},
                              names=["race_id"]).reset_index()
            yield Plan(record["name"], chain, str(count)), frame
        if not count:
            raise ValueError(f"No ensemble maps in {path}")


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
    ensemble: Iterable[tuple[Plan, pd.DataFrame]] | None = None,
) -> Path:
    from atlas_posthoc_plus.workflows.elections.partisan_swing_plot import (
        plot_curves,
        plot_histogram,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}-") as temp:
        working = Path(temp)
        officials = list(_score_and_write(working, "official", official))
        if len(officials) != 3:
            raise ValueError("Expected three official plans for the swing analysis")
        plot_curves(working, officials)
        if ensemble is not None:
            scores = _score_and_write(working, "ensemble", ensemble, reference=officials[0][1])
            areas = np.fromiter((result.area for _, result in scores), dtype=float)
            plot_histogram(working, areas, officials)
        output.mkdir(exist_ok=True)
        for file in working.iterdir():
            file.replace(output / file.name)
    return output
