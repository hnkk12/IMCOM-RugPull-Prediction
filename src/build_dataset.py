"""data/raw/*.parquet -> data/processed/{labels,dataset_N{n}h}.parquet

    python -m src.build_dataset --config experiments/common.yaml [--jobs 8]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.artifacts import atomic_csv, atomic_parquet, write_parquet_manifest
from src.config import load_config, resolve
from src.contract_features import contract_features
from src.decode import load_decoded, load_metadata
from src.features import build_features
from src.label import assign_labels, detect_rugs
from src.split import _ts


def validate_dataset(ds: pd.DataFrame, n_hours: float, h_days: float) -> None:
    """Fail before publishing a malformed processed artifact."""
    if ds.pair.duplicated().any():
        raise ValueError("processed dataset contains duplicate pair rows")
    if not set(ds.y.unique()).issubset({0, 1}):
        raise ValueError("labels must be binary")
    if not ds.status.isin(["pos", "neg"]).all():
        raise ValueError("training dataset contains excluded label statuses")
    if not np.allclose(ds.t_end - ds.t0, n_hours * 3600):
        raise ValueError("t_end does not match the configured observation window")
    pos = ds.y.eq(1)
    if not ((ds.loc[pos, "rug_ts"] > ds.loc[pos, "t_end"]) &
            (ds.loc[pos, "rug_ts"] <= ds.loc[pos, "t_end"] + h_days * 86400)).all():
        raise ValueError("positive rows violate the prediction-horizon definition")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/common.yaml")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--data-root", help="use <root>/raw, <root>/processed (e.g. data/synthetic)")
    ap.add_argument("--output-dir", help="write processed artifacts here without changing the raw-data path")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.data_root:
        cfg["paths"].update(raw=f"{a.data_root}/raw", processed=f"{a.data_root}/processed")
    if a.output_dir:
        cfg["paths"]["processed"] = a.output_dir
    raw, out = resolve(cfg["paths"]["raw"]), resolve(cfg["paths"]["processed"])
    out.mkdir(parents=True, exist_ok=True)
    L, S = cfg["label"], cfg["study"]

    pairs, contracts = load_metadata(raw)
    tables = load_decoded(raw, pairs)
    n_events = sum(len(v) for v in tables.values())
    print(f"pairs={len(pairs):,} decoded_events={n_events:,} contracts={len(contracts):,}")

    rugs = detect_rugs(tables["pool_ev"], L["drop_threshold"], L["mev_factor"], L["min_pre_reserve_eth"],
                       L["cumulative"], L["cum_window_h"])
    atomic_parquet(rugs, out / "rugs.parquet")
    cutoff = _ts(S["data_cutoff"]) if S.get("data_cutoff") else int(time.time())
    cfeat = contract_features(contracts, pairs) if len(contracts) else None

    summary = []
    for n in cfg["windows_hours"]:
        lab = assign_labels(rugs, pairs, n, L["H_days"], S["pool_horizon_days"], cutoff, L["min_init_weth"])
        summary.append(lab.status.value_counts().rename(f"N={n}h"))
        ds = build_features(lab, pairs, tables, n, n_jobs=a.jobs)
        if cfeat is not None:
            ds = ds.merge(cfeat, on="token", how="left")
        validate_dataset(ds, n, L["H_days"])
        atomic_parquet(ds, out / f"dataset_N{n}h.parquet")
        print(f"N={n}h: {len(ds):,} pools, positive rate {ds.y.mean():.3f}")

    s = pd.concat(summary, axis=1).fillna(0).astype(int)
    atomic_csv(s, out / "label_status.csv")
    write_parquet_manifest(out.glob("*.parquet"), out / "manifest.json", root=resolve("."))
    print(s)


if __name__ == "__main__":
    main()
