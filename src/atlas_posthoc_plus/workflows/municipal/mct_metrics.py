"""Population-based cuts and the union of ousted residents."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from atlas_posthoc_plus.workflows.municipal.mct_geography import MEMBERSHIPS


@dataclass(frozen=True)
class OustedPopulation:
    """The four disjoint cases, over the full assigned population."""

    neither: float
    municipal_only: float
    township_only: float
    both: float

    @property
    def union(self) -> float:
        return self.municipal_only + self.township_only + self.both


class SplitScorer:
    """Reuse fixed arrays for population pieces indexed by assignment position."""

    def __init__(self, pieces: pd.DataFrame, units: pd.DataFrame, *, statewide_population: float):
        self.units = units
        self.statewide_population = statewide_population
        self.n_units = len(units)
        unit_ids = pd.Index(units["unit_id"])
        represented = pieces[MEMBERSHIPS].ne("").any(axis=1) & pieces["population"].gt(0)
        pieces = pieces.loc[represented]
        self.population = pieces["population"].to_numpy(dtype=float)
        self.represented_population = float(self.population.sum())
        self.assignment_positions = pieces["position"].to_numpy(dtype=np.intp)
        self.unit_index = np.column_stack([
            unit_ids.get_indexer(pieces[column]) for column in MEMBERSHIPS
        ])
        self.has_membership = self.unit_index >= 0
        self.member_units = self.unit_index[self.has_membership]
        self.member_population = np.broadcast_to(
            self.population[:, None], self.unit_index.shape,
        )[self.has_membership]
        self.kinds = units["kind"].to_numpy()

    def score(
        self, assignment: np.ndarray, *, details: bool = False,
    ) -> tuple[dict[str, int | float | None], pd.DataFrame | None]:
        district_labels, district_codes = np.unique(assignment, return_inverse=True)
        districts = district_codes[self.assignment_positions]
        n_districts = len(district_labels)

        # Each membership is counted in its unit's totals. Only the final union
        # is summed over disjoint population pieces.
        member_districts = np.broadcast_to(districts[:, None], self.unit_index.shape)
        totals = np.bincount(
            self.member_units * n_districts + member_districts[self.has_membership],
            weights=self.member_population,
            minlength=self.n_units * n_districts,
        ).reshape(self.n_units, n_districts)
        # District labels are numerically sorted. Treat sub-nano-person gaps
        # from adding fractional crosswalk allocations as ties.
        tied = (totals > 0) & (totals >= totals.max(axis=1, keepdims=True) - 1e-9)
        dominant = np.argmax(tied, axis=1)
        dominant_with_absent = np.append(dominant, -1)
        flags = self.has_membership & (districts[:, None] != dominant_with_absent[self.unit_index])
        cases = np.bincount(
            flags[:, 0].astype(np.intp) + 2 * flags[:, 1],
            weights=self.population,
            minlength=4,
        )
        cases[0] += max(0.0, self.statewide_population - self.represented_population)
        ousted = OustedPopulation(*map(float, cases))
        n_represented_districts = (totals > 0).sum(axis=1)
        cuts = n_represented_districts > 1
        summary = {
            "cut_cities": int(np.count_nonzero(cuts & (self.kinds == "city"))),
            "cut_villages": int(np.count_nonzero(cuts & (self.kinds == "village"))),
            "cut_townships": int(np.count_nonzero(cuts & (self.kinds == "township"))),
            "cut_mct_units": int(cuts.sum()),
            "municipal_corporation_ousted_population": ousted.municipal_only + ousted.both,
            "township_ousted_population": ousted.township_only + ousted.both,
            "naive_ousted_population": ousted.union + ousted.both,
            "deduplicated_ousted_population": ousted.union,
            "double_counted_overlap": ousted.both,
            "population_ousted_from_neither": ousted.neither,
            "population_ousted_from_municipal_corporation_only": ousted.municipal_only,
            "population_ousted_from_township_only": ousted.township_only,
            "population_ousted_from_both": ousted.both,
            "statewide_population": self.statewide_population,
            "represented_population": self.represented_population,
            "ousted_share_statewide": ousted.union / self.statewide_population,
            "ousted_share_represented": (
                ousted.union / self.represented_population if self.represented_population else None
            ),
        }
        unit_details = None
        if details:
            unit_details = self.units.assign(
                population=totals.sum(axis=1),
                dominant_district=pd.array(
                    district_labels[dominant], dtype="Int64",
                ),
                represented_districts=n_represented_districts,
                ousted_population=(
                    totals.sum(axis=1) - totals[np.arange(self.n_units), dominant]
                ),
            )
            unit_details.loc[n_represented_districts == 0, "dominant_district"] = pd.NA
        return summary, unit_details
