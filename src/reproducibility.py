"""Generate data/result hashes and an exact runtime-environment record."""
from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys

from src.artifacts import atomic_json, write_file_manifest, write_parquet_manifest
from src.config import ROOT, load_config, resolve


PACKAGES = ["numpy", "pandas", "pyarrow", "scikit-learn", "xgboost", "lightgbm",
            "shap", "matplotlib", "joblib", "pycryptodome", "requests"]


def run(cfg: dict):
    raw, processed, results = (resolve(cfg["paths"][k]) for k in ("raw", "processed", "results"))
    raw_files = [raw / "pairs.parquet", raw / "contracts.parquet", *(raw / "events").glob("*.parquet")]
    processed_files = list(processed.glob("*.parquet"))
    excluded = {results / "manifest.json", results / "environment.json"}
    result_files = [p for p in results.rglob("*") if p.is_file()
                    and "legacy_pre_improvements" not in p.parts and p not in excluded]
    write_parquet_manifest(raw_files, raw / "manifest.json", root=ROOT)
    write_parquet_manifest(processed_files, processed / "manifest.json", root=ROOT)
    write_file_manifest(result_files, results / "manifest.json", root=ROOT)
    env = {"python": sys.version, "platform": platform.platform(),
           "packages": {p: importlib.metadata.version(p) for p in PACKAGES}}
    atomic_json(env, results / "environment.json", indent=2)
    return len(raw_files), len(processed_files), len(result_files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/common.yaml")
    a = ap.parse_args()
    print("manifest counts (raw, processed, results):", run(load_config(a.config)))


if __name__ == "__main__":
    main()
