"""Build joint 2020 Ohio block memberships and reconcile their graph allocation."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import geopandas as gpd
import numpy as np
import pandas as pd

from atlas_posthoc_plus.workflows.municipal.geography import _resolve_dataset_path

OHIO_POPULATION_2020 = 11_799_448
POPULATION_TOLERANCE = 1e-5
MEMBERSHIPS = ["municipal_corporation", "township"]
_CACHE_SCHEMA = 1


@dataclass(frozen=True)
class GeographySettings:
    township_lsadc: str = "47"
    unit_definition: str = "whole"
    split_block_policy: str = "reject"
    expected_population: int = OHIO_POPULATION_2020
    block_id_col: str = "GEOID20"
    block_pop_col: str = "POP20"
    graph_key_col: str = "NAME"
    graph_pop_col: str = "POP20"
    crosswalk_block_col: str = "GEOID20"
    crosswalk_key_col: str = "NAME"
    crosswalk_pop_col: str = "POP20"


@dataclass(frozen=True)
class Membership:
    blocks: pd.DataFrame
    graph_pieces: pd.DataFrame
    units: pd.DataFrame
    classification: pd.DataFrame
    allocation_audit: dict


def _read_attributes(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return gpd.read_file(path, ignore_geometry=True)


def _column(table: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in table:
            return table[name]
    raise ValueError(f"Missing Census field. Expected one of {names}; found {list(table)}")


def _codes(values: pd.Series, *, width: int, label: str, allow_empty: bool = False) -> pd.Series:
    codes = values.fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    valid = codes.str.fullmatch(rf"[0-9]{{1,{width}}}")
    if allow_empty:
        valid |= codes.eq("")
    if not valid.all():
        raise ValueError(f"Invalid {label}: {codes[~valid].head().tolist()}")
    return codes.where(codes.eq(""), codes.str.zfill(width))


def ohio_block_ids(values: pd.Series) -> pd.Series:
    ids = _codes(values, width=15, label="2020 block GEOID")
    if not ids.str.startswith("39").all():
        raise ValueError("Block GEOIDs must be 15-digit Ohio Census identifiers")
    return ids


def _population(values: pd.Series, *, label: str) -> pd.Series:
    population = pd.to_numeric(values, errors="raise").astype(float)
    if not np.isfinite(population).all() or (population < 0).any():
        raise ValueError(f"{label} must be finite and nonnegative")
    return population


def _unique_index(table: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    if table[column].duplicated().any() or table[column].eq("").any():
        raise ValueError(f"Missing or duplicate {label} identifiers")
    return table.set_index(column)


def classify_geography(
    table: pd.DataFrame, source: str, *, township_lsadc: str,
) -> pd.DataFrame:
    """Use LSADC and flag or resolve conflicts visible in source metadata."""
    width = 7 if source == "PLACE" else 10
    geoid = _codes(_column(table, "GEOID20", "GEOID"), width=width, label=f"{source} GEOID")
    if not geoid.str.startswith("39").all():
        raise ValueError(f"{source} contains non-Ohio GEOIDs")
    lsadc = _codes(_column(table, "LSADC", "LSAD20", "LSAD"), width=2, label="LSADC")
    classfp = _column(table, "CLASSFP20", "CLASSFP", f"{source}CC").fillna("").str.strip()
    funcstat = _column(table, "FUNCSTAT20", "FUNCSTAT").fillna("").str.strip()
    name = _column(table, "NAMELSAD20", "NAMELSAD", "NAME20", "NAME")
    if (classfp.eq("") | funcstat.eq("")).any():
        raise ValueError(f"{source} contains missing CLASSFP or FUNCSTAT metadata")

    kind = pd.Series("", index=table.index, dtype=str)
    review = pd.Series("", index=table.index, dtype=str)
    reason = pd.Series("other LSADC", index=table.index, dtype=str)
    if source == "PLACE":
        kind = lsadc.map({"25": "city", "47": "village"}).fillna("")
        reason.loc[classfp.str.startswith("U")] = "Census-designated place"
        conflict = kind.ne("") & ~classfp.isin(["C1", "C2", "C5", "C6", "C7"])
        review.loc[conflict] = "PLACE class differs from expected incorporated classes"
        review.loc[kind.eq("") & classfp.isin(["C1", "C2", "C5", "C6", "C7"])] = (
            "Incorporated PLACE has a different LSADC"
        )
    else:
        township_class = classfp.isin(["T1", "T2", "T5"])
        selected_code = lsadc.eq(township_lsadc)
        kind.loc[selected_code & township_class] = "township"
        reason.loc[classfp.str.startswith("C")] = "incorporated COUSUB"
        reason.loc[classfp.eq("T9") | classfp.str.startswith("Z")] = (
            "inactive or statistical COUSUB"
        )
        review.loc[township_class & ~selected_code] = "Township class has a different LSADC"
        review.loc[selected_code & ~township_class] = "Selected LSADC has a non-township class"

    reason.loc[kind.ne("")] = "included"
    return _unique_index(pd.DataFrame({
        "geoid": geoid, "source": source, "name": name, "kind": kind,
        "classfp": classfp, "lsadc": lsadc, "funcstat": funcstat,
        "accepted": kind.ne(""), "decision": reason, "review": review,
    }), "geoid", source)


def _read_census_baf(path: Path, source: str, blocks: pd.Index) -> pd.Series:
    table = pd.read_csv(path, sep="|", dtype=str, keep_default_na=False)
    block = ohio_block_ids(_column(table, "BLOCKID"))
    if source == "PLACE":
        code = _codes(_column(table, "PLACEFP"), width=5, label="PLACEFP", allow_empty=True)
        geoid = ("39" + code).where(~code.isin(["", "00000"]), "")
    else:
        code = _codes(_column(table, "COUSUBFP"), width=5, label="COUSUBFP", allow_empty=True)
        county = _codes(_column(table, "COUNTYFP"), width=3, label="COUNTYFP", allow_empty=True)
        if (county.ne("") & county.ne(block.str[2:5])).any() or (
            county.eq("") & code.ne("")
        ).any():
            raise ValueError("COUSUB BAF county does not match its block GEOID")
        # 00000 can be a Census record for an undefined county subdivision.
        geoid = (block.str[:5] + code).where(code.ne(""), "")
    rows = _unique_index(pd.DataFrame({"block": block, "geoid": geoid}), "block", source)
    if len(blocks.difference(rows.index)) or len(rows.index.difference(blocks)):
        raise ValueError(f"{source} Census BAF and population table have different block universes")
    return rows["geoid"].reindex(blocks)


def _graph_pieces(
    blocks: pd.DataFrame, graph: pd.DataFrame, crosswalk_path: Path, settings: GeographySettings,
) -> tuple[pd.DataFrame, dict]:
    graph_key = graph[settings.graph_key_col].astype("string").fillna("").str.strip()
    graph_pop = _population(graph[settings.graph_pop_col], label="Graph population")
    nodes = _unique_index(pd.DataFrame({
        "key": graph_key, "position": np.arange(len(graph)), "population": graph_pop,
    }), "key", "graph crosswalk key")

    crosswalk = pd.read_csv(crosswalk_path, dtype=str, keep_default_na=False)
    rows = pd.DataFrame({
        "block": ohio_block_ids(crosswalk[settings.crosswalk_block_col]),
        "key": crosswalk[settings.crosswalk_key_col].str.strip(),
        "population": _population(crosswalk[settings.crosswalk_pop_col], label="Crosswalk population"),
    })
    rows = rows.loc[rows["population"].gt(0)].copy()
    if rows.duplicated(["block", "key"]).any():
        raise ValueError("Duplicate positive block/graph-node crosswalk matches")
    for column, index in (("key", nodes.index), ("block", blocks.index)):
        missing = sorted(set(rows[column]).difference(index))
        if missing:
            raise ValueError(f"Crosswalk references unknown {column}: {missing[:5]}")

    allocation = rows.groupby("block")["population"].agg(["sum", "size"]).reindex(
        blocks.index, fill_value=0,
    )
    if (blocks["population"].gt(0) & allocation["size"].eq(0)).any():
        raise ValueError("Crosswalk is missing positive-population blocks")
    _check_population(blocks["population"], allocation["sum"], "Block-to-graph")
    split = allocation["size"].gt(1)
    if split.any() and settings.split_block_policy == "reject":
        raise ValueError(
            f"{int(split.sum())} populated blocks have multiple graph nodes. "
            "Supply the graph-generation allocation or explicitly use split_block_policy='allocated'."
        )

    # Restore exact per-block mass within the input roundoff tolerance.
    rows["population"] *= (
        rows["block"].map(blocks["population"]) / rows["block"].map(allocation["sum"])
    )
    _check_population(
        nodes["population"], rows.groupby("key")["population"].sum().reindex(nodes.index, fill_value=0),
        "Graph-node",
    )
    rows["position"] = rows["key"].map(nodes["position"])
    for column in MEMBERSHIPS:
        rows[column] = rows["block"].map(blocks[column])
    pieces = rows.groupby(["position", *MEMBERSHIPS], as_index=False)["population"].sum()
    return pieces, {
        "positive_crosswalk_rows": len(rows),
        "split_positive_blocks": int(split.sum()),
        "population_in_split_blocks": float(blocks.loc[split, "population"].sum()),
        "graph_pieces": len(pieces),
    }


def _check_population(expected: pd.Series, actual: pd.Series, label: str) -> None:
    bad = ~np.isclose(expected, actual, rtol=0, atol=POPULATION_TOLERANCE)
    if bad.any():
        differences = (actual[bad] - expected[bad]).head().to_dict()
        raise ValueError(f"{label} population is not conserved. Differences: {differences}")


def _build_membership(
    blocks_path: Path, place_path: Path, cousub_path: Path, place_baf: Path, cousub_baf: Path,
    crosswalk_path: Path, graph: pd.DataFrame, settings: GeographySettings,
) -> Membership:
    table = _read_attributes(blocks_path)
    blocks = _unique_index(pd.DataFrame({
        "block": ohio_block_ids(table[settings.block_id_col]),
        "population": _population(table[settings.block_pop_col], label="Block population"),
    }), "block", "Census block")
    if (
        settings.expected_population <= 0
        or np.any(blocks["population"] % 1)
        or blocks["population"].sum() != settings.expected_population
    ):
        raise ValueError(
            f"Expected {settings.expected_population:,} residents in integer block counts; "
            f"found {blocks['population'].sum():,.6f}"
        )
    county = blocks.index.to_series().str[:5]
    audits, unit_tables = [], []
    for source, path, baf, column in (
        ("PLACE", place_path, place_baf, MEMBERSHIPS[0]),
        ("COUSUB", cousub_path, cousub_baf, MEMBERSHIPS[1]),
    ):
        records = classify_geography(
            _read_attributes(path), source, township_lsadc=settings.township_lsadc,
        )
        geoids = _read_census_baf(baf, source, blocks.index)
        missing = sorted(set(geoids).difference(records.index, {""}))
        if missing:
            raise ValueError(f"{source} BAF references unknown Census GEOIDs: {missing[:5]}")
        totals = blocks["population"].groupby(geoids).sum()
        records["population"] = totals.reindex(records.index, fill_value=0)
        audits.append(records.reset_index())
        eligible = records.loc[records["accepted"]]
        if eligible.empty:
            raise ValueError(
                f"No eligible {source} records with township_lsadc={settings.township_lsadc!r}. "
                "Standard 2020 Ohio COUSUB uses LSADC 44 for townships and 47 for villages."
            )
        accepted = geoids.isin(eligible.index)
        unit_id = source + ":" + geoids
        if source == "PLACE" and settings.unit_definition == "county-component":
            unit_id += ":" + county
        blocks[column] = unit_id.where(accepted, "")
        components = pd.DataFrame({
            "unit_id": unit_id[accepted], "geoid": geoids[accepted], "county_geoid": county[accepted],
        }).drop_duplicates()
        units = components.groupby(["unit_id", "geoid"], as_index=False).agg(
            county_geoids=("county_geoid", "|".join),
        )
        unit_tables.append(units.join(eligible, on="geoid").drop(columns="population"))

    pieces, allocation_audit = _graph_pieces(blocks, graph, crosswalk_path, settings)
    return Membership(
        blocks, pieces, pd.concat(unit_tables, ignore_index=True).sort_values("unit_id"),
        pd.concat(audits, ignore_index=True), allocation_audit,
    )


def _source_signature(path: Path) -> list[dict]:
    files = [path]
    if path.suffix.lower() == ".shp":
        files.extend(
            sorted(p for p in path.parent.glob(f"{path.stem}.*") if p != path and p.is_file())
        )
    signatures = []
    for file in files:
        stat = file.stat()
        signatures.append({"file": str(file.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return signatures


def load_membership(
    *, graph_path: Path, graph: pd.DataFrame, blocks_path: Path, place_path: Path,
    cousub_path: Path, place_baf: Path, cousub_baf: Path, crosswalk_path: Path,
    cache_dir: Path, settings: GeographySettings,
) -> tuple[Membership, dict]:
    """Reuse validated membership when all source file signatures and settings match."""
    datasets = {
        "blocks": _resolve_dataset_path(blocks_path),
        "place": _resolve_dataset_path(place_path),
        "cousub": _resolve_dataset_path(cousub_path),
        "place_baf": place_baf, "cousub_baf": cousub_baf, "crosswalk": crosswalk_path,
    }
    sources = {"graph": graph_path, **datasets}
    metadata = {
        "schema": _CACHE_SCHEMA, "census_vintage": 2020, "settings": asdict(settings),
        "sources": {name: _source_signature(path) for name, path in sources.items()},
    }
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    cache_path = cache_dir / f"{digest}.pickle"
    if cache_path.exists():
        try:
            with cache_path.open("rb") as stream:
                membership = pickle.load(stream)
            if isinstance(membership, Membership):
                return membership, {**metadata, "cache_hit": True, "cache_key": digest}
        except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError):
            pass

    membership = _build_membership(*datasets.values(), graph, settings)
    if metadata["sources"] != {name: _source_signature(path) for name, path in sources.items()}:
        raise ValueError("A membership input changed during preparation; rerun with stable inputs")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=cache_dir, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            pickle.dump(membership, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.close()
            temporary.replace(cache_path)
        finally:
            temporary.unlink(missing_ok=True)
    return membership, {**metadata, "cache_hit": False, "cache_key": digest}
