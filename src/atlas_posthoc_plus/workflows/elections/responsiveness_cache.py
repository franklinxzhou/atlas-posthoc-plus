"""Shared, content-addressed cache for responsiveness election allocation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def input_signatures(path: Path) -> list[dict]:
    """Hash a local dataset and the files a shapefile reader can use alongside it."""
    path = Path(path)
    files = {path}
    if path.suffix.lower() == ".gpkg":
        # A writable GeoPackage can carry committed data in a SQLite sidecar.
        for suffix in ("-wal", "-journal"):
            sidecar = Path(f"{path}{suffix}")
            if sidecar.is_file():
                files.add(sidecar)
    if path.suffix.lower() == ".shp":
        prefix = path.stem.lower() + "."
        files.update(
            sibling for sibling in path.parent.iterdir()
            if sibling.name.lower().startswith(prefix) and sibling.is_file()
        )
    results = []
    for source in sorted(files):
        with source.open("rb") as stream:
            before = os.fstat(stream.fileno())
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = source.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise RuntimeError(f"Election cache input changed while hashing: {source}")
        results.append({
            "path": str(source.absolute()),
            "resolved_path": str(source.resolve()),
            "bytes": after.st_size,
            "sha256": digest,
        })
    return results


@contextmanager
def cache_lock(path, progress):
    import fcntl

    # Keep the lock inode in place. The kernel releases an acquired lock when a
    # job exits, including after a forced termination.
    with path.open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            with progress.stage(f"Waiting for shared election cache lock {path.name}"):
                while True:
                    time.sleep(0.5)
                    try:
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        continue
        yield


def cached_allocation(cache_dir, mode, identify, build, decode, progress):
    """Read or publish one atomic archive with metadata and lossless vote arrays."""
    with progress.stage("Fingerprinting election allocation inputs"):
        identity = identify()
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"responsiveness-{digest}.npz"

    with cache_lock(path.with_suffix(".lock"), progress):
        if mode == "use" and path.is_file():
            with progress.stage(f"Reading election cache {path.name}"):
                try:
                    with np.load(path, allow_pickle=False) as data:
                        metadata = json.loads(data["metadata"].tobytes())
                        if metadata["identity"] != identity:
                            raise ValueError("input metadata differs")
                        result = decode(data["node_ids"], data["votes"], metadata)
                except (OSError, EOFError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
                    progress.log(f"WARNING: rejecting election cache {path.name}: {exc}; rebuilding")
                else:
                    progress.log(f"Election cache hit: {path}")
                    return result

        progress.log(f"Election cache {'rebuild' if mode == 'rebuild' else 'miss'}: {path}")
        allocated, node_ids, votes, metadata = build()
        with progress.stage("Verifying election inputs and publishing cache"):
            if identify() != identity:
                raise RuntimeError("Election allocation inputs changed during preprocessing")
            metadata["identity"] = identity
            serialized = json.dumps(metadata, allow_nan=False).encode()
            with tempfile.NamedTemporaryFile(dir=cache_dir, prefix=path.stem, delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    np.savez(stream, metadata=np.frombuffer(serialized, dtype=np.uint8),
                             node_ids=node_ids, votes=votes)
                    stream.flush()
                    os.fsync(stream.fileno())
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
        return allocated
