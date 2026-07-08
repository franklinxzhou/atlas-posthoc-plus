from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from atlas_posthoc_plus.workflows.municipal.tradeoff import normalize_reference_summaries
from atlas_posthoc_plus.workflows.municipal.tradeoff_weights import collection_style, load_json


@dataclass(frozen=True)
class CollectionSeries:
    collection: str
    label: str
    style_index: int
    color: str
    values: np.ndarray


REFERENCE_LINE_STYLES = {
    "Enacted ('25)": {"color": "#9467bd", "linestyle": "-", "linewidth": 2.4},
    "Overturned ('21)": {"color": "#2ca02c", "linestyle": "--", "linewidth": 2.4},
    "Remedial ('22)": {"color": "#d62728", "linestyle": "-.", "linewidth": 2.4},
}


def finite_array(values: Any, *, context: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError(f"{context} has no finite values.")
    return arr


def build_collection_series(
    traces: list[dict[str, Any]],
    *,
    metric_key: str,
) -> list[CollectionSeries]:
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for i, item in enumerate(traces):
        if metric_key not in item:
            raise ValueError(
                f"trace item {i} is missing {metric_key!r}; "
                f"available keys: {sorted(item.keys())}"
            )

        collection = str(item.get("collection", item.get("mcd_weight", f"collection-{i + 1}")))
        label = str(item.get("collection_label", collection))
        style_index = int(item.get("collection_style_index", len(order)))

        if collection not in grouped:
            style = collection_style(collection, style_index=style_index, label=label)
            grouped[collection] = {
                "label": label,
                "style_index": style_index,
                "color": style["color"],
                "values": [],
            }
            order.append(collection)

        grouped[collection]["values"].extend(
            finite_array(item[metric_key], context=f"trace item {i} {metric_key}").tolist()
        )

    series = [
        CollectionSeries(
            collection=collection,
            label=str(grouped[collection]["label"]),
            style_index=int(grouped[collection]["style_index"]),
            color=str(grouped[collection]["color"]),
            values=finite_array(
                grouped[collection]["values"],
                context=f"collection {grouped[collection]['label']} {metric_key}",
            ),
        )
        for collection in order
    ]
    if not series:
        raise ValueError("No trace collections were available to plot.")
    return series


def load_reference_summaries(path: str | Path) -> dict[str, Any]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise ValueError("--reference-json must contain a dict keyed by official plan label.")
    return normalize_reference_summaries(raw)


def reference_values(refs: dict[str, Any], *, metric_key: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for label, summary in refs.items():
        if metric_key not in summary:
            raise ValueError(f"reference plan {label} is missing {metric_key!r}.")
        value = float(summary[metric_key])
        if np.isfinite(value):
            out[label] = value
    return out


def load_other_ensemble_series(
    path: str | Path,
    *,
    metric_key: str,
    label: str,
    style_index: int,
    include_reference_cd_2020: bool = False,
) -> CollectionSeries | None:
    raw = load_json(path)
    if not isinstance(raw, list):
        raise ValueError("--other-ensemble-summary-json must contain a list of summaries.")

    values = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if item.get("status", "ok") != "ok":
            continue
        if not include_reference_cd_2020 and bool(item.get("is_reference_cd_2020", False)):
            continue
        if metric_key in item:
            value = float(item[metric_key])
            if np.isfinite(value):
                values.append(value)

    if not values:
        return None

    return CollectionSeries(
        collection="other_ensemble",
        label=label,
        style_index=style_index,
        color="#7f7f7f",
        values=finite_array(values, context=f"{label} {metric_key}"),
    )


def combined_min_max(
    series: list[CollectionSeries],
    refs: dict[str, float],
) -> tuple[float, float]:
    values = []
    for item in series:
        values.extend(float(v) for v in item.values)
    values.extend(refs.values())
    if not values:
        raise ValueError("No finite values are available to set axis limits.")
    return min(values), max(values)
