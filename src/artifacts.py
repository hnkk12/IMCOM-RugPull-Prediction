"""Reproducible and crash-safe artifact helpers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


def _replace(tmp: Path, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, path)


def atomic_parquet(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, **kwargs)
    _replace(tmp, path)


def atomic_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, **kwargs)
    _replace(tmp, path)


def atomic_json(value, path: str | Path, **kwargs) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(value, **kwargs), encoding="utf-8")
    _replace(tmp, path)


def sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def parquet_record(path: str | Path, root: str | Path | None = None) -> dict:
    import pyarrow.parquet as pq

    path = Path(path)
    meta = pq.ParquetFile(path).metadata
    label = path.relative_to(root).as_posix() if root else path.as_posix()
    return {
        "path": label,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "rows": meta.num_rows,
        "columns": meta.num_columns,
    }


def write_parquet_manifest(paths, output: str | Path, root: str | Path | None = None) -> list[dict]:
    records = [parquet_record(p, root) for p in sorted(map(Path, paths))]
    atomic_json(records, output, indent=2)
    return records


def write_file_manifest(paths, output: str | Path, root: str | Path | None = None) -> list[dict]:
    root = Path(root) if root else None
    records = []
    for path in sorted(map(Path, paths)):
        records.append({"path": path.relative_to(root).as_posix() if root else path.as_posix(),
                        "bytes": path.stat().st_size, "sha256": sha256(path)})
    atomic_json(records, output, indent=2)
    return records
