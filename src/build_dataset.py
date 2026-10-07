"""data/raw/*.parquet -> data/processed/{labels,dataset_N{n}h}.parquet

    python -m src.build_dataset --config experiments/common.yaml [--jobs 8]
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from src.config import load_config, resolve
from src.contract_features import contract_features
from src.decode import decode, load_raw
from src.features import build_features
from src.label import assign_labels, detect_rugs
from src.split import _ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/common.yaml")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--data-root", help="use <root>/raw, <root>/processed (e.g. data/synthetic)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.data_root:
        cfg["paths"].update(raw=f"{a.data_root}/raw", processed=f"{a.data_root}/processed")
    raw, out = resolve(cfg["paths"]["raw"]), resolve(cfg["paths"]["processed"])
    out.mkdir(parents=True, exist_ok=True)
    L, S = cfg["label"], cfg["study"]

    pairs, events, contracts = load_raw(raw)
    print(f"pairs={len(pairs):,} events={len(events):,} contracts={len(contracts):,}")
    tables = decode(events, pairs)
    del events

    rugs = detect_rugs(tables["pool_ev"], L["drop_threshold"], L["mev_factor"], L["min_pre_reserve_eth"],
                       L["cumulative"], L["cum_window_h"])
    rugs.to_parquet(out / "rugs.parquet")
    cutoff = _ts(S["data_cutoff"]) if S.get("data_cutoff") else int(time.time())
    cfeat = contract_features(contracts, pairs) if len(contracts) else None

    summary = []
    for n in cfg["windows_hours"]:
        lab = assign_labels(rugs, pairs, n, L["H_days"], S["pool_horizon_days"], cutoff, L["min_init_weth"])
        summary.append(lab.status.value_counts().rename(f"N={n}h"))
        ds = build_features(lab, pairs, tables, n, n_jobs=a.jobs)
        if cfeat is not None:
            ds = ds.merge(cfeat, on="token", how="left")
        ds.to_parquet(out / f"dataset_N{n}h.parquet")
        print(f"N={n}h: {len(ds):,} pools, positive rate {ds.y.mean():.3f}")

    s = pd.concat(summary, axis=1).fillna(0).astype(int)
    s.to_csv(out / "label_status.csv")
    print(s)


if __name__ == "__main__":
    main()
