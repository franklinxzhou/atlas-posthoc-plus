#!/usr/bin/env python3
"""Plot a Polsby-Popper histogram with optional public and official overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAT_TO_KEY = {
    "average": "average_polsby_popper",
    "mean": "average_polsby_popper",
    "min": "minimum_polsby_popper",
    "minimum": "minimum_polsby_popper",
    "max": "maximum_polsby_popper",
    "maximum": "maximum_polsby_popper",
}

STAT_TO_SUMMARY_COL = {
    "average": "average",
    "mean": "average",
    "min": "minimum",
    "minimum": "minimum",
    "max": "maximum",
    "maximum": "maximum",
}

STAT_LABEL = {
    "average": "Average district Polsby-Popper",
    "mean": "Average district Polsby-Popper",
    "min": "Minimum district Polsby-Popper",
    "minimum": "Minimum district Polsby-Popper",
    "max": "Maximum district Polsby-Popper",
    "maximum": "Maximum district Polsby-Popper",
}


def _read_json(path: Path) -> Any:
    with path.open("r") as stream:
        return json.load(stream)


def _trace_records(raw: Any, *, path: Path) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict):
        for key in ("traces", "chains", "chain_traces"):
            value = raw.get(key)
            if isinstance(value, list):
                records = value
                break
        else:
            raise ValueError(
                f"Could not find Polsby-Popper trace records in {path}. "
                f"Available keys: {list(raw.keys())}"
            )
    else:
        raise TypeError(f"Unsupported Polsby-Popper trace JSON type in {path}: {type(raw)}")

    if not all(isinstance(record, dict) for record in records):
        raise TypeError(f"All Polsby-Popper trace records must be dicts: {path}")
    return records


def load_ensemble_scores(path: str | Path, stat: str, burnin: int) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Ensemble Polsby-Popper trace JSON not found: {source}")

    key = STAT_TO_KEY[stat]
    records = _trace_records(_read_json(source), path=source)

    rows: list[dict[str, Any]] = []
    for record in records:
        chain_id = str(record.get("chain_id", "unknown_chain"))

        if key not in record:
            raise KeyError(
                f"Could not find key {key!r} in chain record {chain_id!r}. "
                f"Available keys are: {sorted(record.keys())}"
            )

        scores = list(record[key])
        proposals = list(record.get("proposals", range(len(scores))))

        if len(proposals) != len(scores):
            raise ValueError(
                f"Chain {chain_id!r} has {len(proposals)} proposals but "
                f"{len(scores)} scores for {key!r}."
            )

        for proposal, score in zip(proposals, scores, strict=True):
            if proposal < burnin:
                continue
            if score is None:
                continue
            score_float = float(score)
            if not np.isfinite(score_float):
                continue
            rows.append(
                {
                    "chain_id": chain_id,
                    "proposal": proposal,
                    "score": score_float,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No ensemble scores remain after filtering/burn-in.")

    return df


def load_public_scores(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["plan_id", "score"])

    source = Path(path)
    if not source.exists():
        print(f"Public Polsby-Popper score CSV not found; skipping: {source}")
        return pd.DataFrame(columns=["plan_id", "score"])

    df = pd.read_csv(source)
    if "score" not in df.columns:
        raise ValueError(f"Public scores CSV must contain a 'score' column: {source}")

    if "plan_id" not in df.columns:
        df["plan_id"] = [f"public_{i + 1}" for i in range(len(df))]

    out = df[["plan_id", "score"]].copy()
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    return out.dropna(subset=["score"])


def _resolve_official_label_col(df: pd.DataFrame) -> str:
    for candidate in ("label", "plan", "name", "plan_label"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Official scores CSV must contain one of: label, plan, name, plan_label."
    )


def load_official_scores(path: str | Path | None, stat: str) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["label", "score"])

    source = Path(path)
    if not source.exists():
        print(f"Official Polsby-Popper score CSV not found; skipping: {source}")
        return pd.DataFrame(columns=["label", "score"])

    df = pd.read_csv(source)
    label_col = _resolve_official_label_col(df)

    if "score" in df.columns:
        score_col = "score"
    else:
        score_col = STAT_TO_SUMMARY_COL[stat]
        if score_col not in df.columns:
            raise ValueError(
                f"Official scores CSV must contain 'score' or {score_col!r} "
                f"for stat={stat!r}. Available columns: {list(df.columns)}"
            )

    out = df[[label_col, score_col]].copy()
    out = out.rename(columns={label_col: "label", score_col: "score"})
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    return out.dropna(subset=["score"])


def add_reference_rug(
    ax,
    public_df: pd.DataFrame,
    official_df: pd.DataFrame,
    ymax: float,
    *,
    public_label: str,
    official_label: str,
) -> None:
    public_y = -0.045 * ymax
    official_y = -0.105 * ymax

    if not public_df.empty:
        ax.scatter(
            public_df["score"],
            np.full(len(public_df), public_y),
            marker="o",
            s=24,
            color="red",
            alpha=0.70,
            edgecolors="none",
            clip_on=False,
            label=public_label,
            zorder=5,
        )

    if not official_df.empty:
        ax.scatter(
            official_df["score"],
            np.full(len(official_df), official_y),
            marker="x",
            s=95,
            color="black",
            linewidths=2.0,
            clip_on=False,
            label=official_label,
            zorder=6,
        )

        label_offsets = [-0.060, -0.095, -0.130]
        for i, row in official_df.reset_index(drop=True).iterrows():
            y_text = official_y + label_offsets[i % len(label_offsets)] * ymax
            ax.annotate(
                str(row["label"]),
                xy=(row["score"], official_y),
                xytext=(row["score"], y_text),
                textcoords="data",
                ha="center",
                va="top",
                rotation=35,
                fontsize=9,
                annotation_clip=False,
                arrowprops={
                    "arrowstyle": "-",
                    "linewidth": 0.8,
                    "color": "black",
                    "shrinkA": 0,
                    "shrinkB": 4,
                },
            )


def plot_histogram(
    ensemble_df: pd.DataFrame,
    public_df: pd.DataFrame,
    official_df: pd.DataFrame,
    outpath: str | Path,
    *,
    stat: str,
    bins: int,
    title: str | None,
    density: bool,
    ensemble_label: str,
    public_label: str,
    official_label: str,
) -> None:
    scores = ensemble_df["score"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.hist(
        scores,
        bins=bins,
        density=density,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.7,
        label=ensemble_label,
    )

    ymax = ax.get_ylim()[1]
    add_reference_rug(
        ax,
        public_df,
        official_df,
        ymax,
        public_label=public_label,
        official_label=official_label,
    )

    mean_score = float(np.mean(scores))
    median_score = float(np.median(scores))

    ax.axvline(
        mean_score,
        linestyle="--",
        linewidth=1.4,
        color="gray",
        label=f"Ensemble mean = {mean_score:.3f}",
    )
    ax.axvline(
        median_score,
        linestyle=":",
        linewidth=1.6,
        color="gray",
        label=f"Ensemble median = {median_score:.3f}",
    )

    ax.set_ylim(-0.28 * ymax, ymax * 1.05)

    stat_label = STAT_LABEL[stat]
    ax.set_xlabel(stat_label)
    ax.set_ylabel("Density" if density else "Number of saved plans")
    ax.set_title(title or f"Ohio congressional {stat_label.lower()} distribution")
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    destination = Path(outpath)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figure to {destination}")

    print("\nEnsemble score summary")
    print(ensemble_df["score"].describe().to_string())

    if not public_df.empty:
        print(f"\nLoaded {len(public_df)} public submitted plan scores.")
        print(public_df["score"].describe().to_string())

    if not official_df.empty:
        print(f"\nLoaded {len(official_df)} official plan scores.")
        print(official_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot an ensemble Polsby-Popper histogram from atlas-posthoc traces "
            "with optional public and official plan overlays."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--ensemble-pp-json",
        required=True,
        type=Path,
        help="atlas-posthoc diagnostics/polsby_popper_traces.json file.",
    )
    parser.add_argument(
        "--public-scores-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns plan_id,score for public submitted plans.",
    )
    parser.add_argument(
        "--official-scores-csv",
        type=Path,
        default=None,
        help=(
            "Optional official-plan score CSV. Accepts either label,score or "
            "the official-polsby summary CSV with plan,average,minimum,maximum."
        ),
    )
    parser.add_argument(
        "--stat",
        choices=["average", "mean", "min", "minimum", "max", "maximum"],
        default="average",
        help="Which Polsby-Popper statistic to plot.",
    )
    parser.add_argument(
        "--burnin",
        type=int,
        default=0,
        help="Discard ensemble samples with proposal number less than this.",
    )
    parser.add_argument("--bins", type=int, default=40, help="Number of histogram bins.")
    parser.add_argument("--density", action="store_true", help="Plot density instead of raw counts.")
    parser.add_argument("--title", default=None, help="Optional plot title.")
    parser.add_argument("--ensemble-label", default="Compliant ensemble plans")
    parser.add_argument("--public-label", default="Public submitted plans")
    parser.add_argument("--official-label", default="Official plans")
    parser.add_argument(
        "--out",
        "--output",
        dest="out",
        required=True,
        type=Path,
        help="Output PNG/PDF/SVG path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensemble_df = load_ensemble_scores(
        path=args.ensemble_pp_json,
        stat=args.stat,
        burnin=args.burnin,
    )
    public_df = load_public_scores(args.public_scores_csv)
    official_df = load_official_scores(args.official_scores_csv, stat=args.stat)

    plot_histogram(
        ensemble_df=ensemble_df,
        public_df=public_df,
        official_df=official_df,
        outpath=args.out,
        stat=args.stat,
        bins=args.bins,
        title=args.title,
        density=args.density,
        ensemble_label=args.ensemble_label,
        public_label=args.public_label,
        official_label=args.official_label,
    )


if __name__ == "__main__":
    main()
