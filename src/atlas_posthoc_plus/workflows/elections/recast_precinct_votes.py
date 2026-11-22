"""Recast precinct-by-district returns on three maps using matched census blocks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CONTEST_KEY = ["county", "source_district"]
PIECE_KEY = [*CONTEST_KEY, "precinct"]
PARTIES = ["dem_votes", "rep_votes", "other_votes"]
WEIGHTS = ["vap", "population", "land_area"]


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    if missing := required.difference(data.columns):
        raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
    if data.empty:
        raise ValueError(f"{path.name} has no rows")
    return data[sorted(required)].apply(lambda column: column.str.strip())


def _read_numbers(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    numbers = data[columns].apply(pd.to_numeric, errors="coerce")
    if not (np.isfinite(numbers) & numbers.ge(0)).to_numpy().all():
        raise ValueError(f"Expected finite, nonnegative numbers in {', '.join(columns)}")
    return numbers


def _set_districts(data: pd.DataFrame, columns: list[str]) -> None:
    districts = _read_numbers(data, columns)
    if not (districts.gt(0) & districts.mod(1).eq(0)).to_numpy().all():
        raise ValueError(f"Expected positive integer districts in {', '.join(columns)}")
    data[columns] = districts.astype(int)


def recast_votes(
    precinct_path: Path, block_path: Path, plans: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    district_columns = list(plans.values())
    if len(plans) != 3 or len(set(district_columns)) != 3 or "source_district" not in district_columns:
        raise ValueError("Specify three distinct plans and district columns including source_district")

    pieces = _read_csv(precinct_path, {*PIECE_KEY, *PARTIES, "real_precinct"})
    blocks = _read_csv(block_path, {*PIECE_KEY, *WEIGHTS, *district_columns, "block_id"})
    if not set(district_columns).isdisjoint({"block_id", "county", "precinct", *WEIGHTS}):
        raise ValueError("Plan assignments must use district columns separate from block identifiers and weights")
    _set_districts(pieces, ["source_district"])
    _set_districts(blocks, district_columns)
    pieces[PARTIES] = _read_numbers(pieces, PARTIES).astype(float)
    blocks[WEIGHTS] = _read_numbers(blocks, WEIGHTS).astype(float)

    flags = pieces["real_precinct"].str.lower().map(
        {"1": True, "true": True, "y": True, "yes": True,
         "0": False, "false": False, "n": False, "no": False}
    )
    if flags.isna().any():
        raise ValueError("real_precinct must be a true or false flag")
    if pieces["county"].eq("").any() or pieces.loc[flags, "precinct"].eq("").any():
        raise ValueError("Returns require county IDs and real precinct rows require precinct IDs")
    if blocks[["block_id", "county", "precinct"]].eq("").to_numpy().any():
        raise ValueError("Every block requires a block, county, and precinct ID")
    if blocks["block_id"].duplicated().any():
        raise ValueError("Expected one assignment row per census block")

    actual = pieces.groupby(CONTEST_KEY)[PARTIES].sum()
    administrative = pieces.loc[~flags].groupby(CONTEST_KEY)[PARTIES].sum()
    real = pieces.loc[flags].groupby(PIECE_KEY)[PARTIES].sum()
    real_totals = real.groupby(level=CONTEST_KEY).transform("sum")
    precinct_shares = real.div(real_totals.where(real_totals.gt(0))).fillna(0)
    pools = real.reset_index()[CONTEST_KEY].merge(
        administrative.reset_index(), on=CONTEST_KEY, how="left", validate="many_to_one"
    )
    pools.index = real.index
    augmented = real + precinct_shares * pools[PARTIES].fillna(0)

    # Eligibility depends on both the precinct and the district in use on election day.
    joined = blocks.merge(augmented.reset_index(), on=PIECE_KEY, how="inner", validate="many_to_one")
    weight_totals = joined.groupby(PIECE_KEY)[WEIGHTS].transform("sum")
    shares = joined[WEIGHTS].div(weight_totals.where(weight_totals.gt(0)))
    chosen_share = shares.bfill(axis="columns").iloc[:, 0].fillna(0)
    joined[PARTIES] = joined[PARTIES].mul(chosen_share, axis="index")

    allocated = joined.groupby(CONTEST_KEY)[PARTIES].sum().reindex(actual.index, fill_value=0)
    unallocated = actual - allocated
    unallocated = unallocated.mask(unallocated.abs() < 1e-7, 0)
    reconciliation = pd.concat(
        [actual.add_prefix("reported_"), allocated.add_prefix("recast_"),
         unallocated.add_prefix("unallocated_")],
        axis="columns",
    ).reset_index()

    plan_totals = []
    for label, column in plans.items():
        district_totals = joined.groupby(column)[PARTIES].sum().reindex(
            sorted(blocks[column].unique()), fill_value=0
        )
        district_totals["total_votes"] = district_totals[PARTIES].sum(axis="columns")
        district_totals["plan"] = label
        plan_totals.append(district_totals.rename_axis("district").reset_index())
    columns = ["plan", "district", *PARTIES, "total_votes"]
    return pd.concat(plan_totals, ignore_index=True)[columns], reconciliation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precinct-votes", required=True, type=Path)
    parser.add_argument("--block-crosswalk", required=True, type=Path)
    parser.add_argument("--plans", nargs=3, required=True, metavar="LABEL=COLUMN")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan_pairs = [tuple(value.split("=", 1)) for value in args.plans]
    if any(len(pair) != 2 or not all(part.strip() for part in pair) for pair in plan_pairs):
        parser.error("--plans requires three LABEL=COLUMN entries")
    plans = {label.strip(): column.strip() for label, column in plan_pairs}
    try:
        totals, audit = recast_votes(args.precinct_votes, args.block_crosswalk, plans)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit_path = args.output.with_name(f"{args.output.stem}-reconciliation.csv")
    totals.to_csv(args.output, index=False)
    audit.to_csv(audit_path, index=False)
    missing = audit[[f"unallocated_{party}" for party in PARTIES]].sum().sum()
    print(f"Wrote {args.output}")
    print(f"Wrote {audit_path}")
    print(f"Votes left unallocated: {missing:,.2f}")


if __name__ == "__main__":
    main()
