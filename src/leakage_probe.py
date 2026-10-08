"""Same-cohort probe for label-dependent observation-window leakage.

For positives the oracle window ends at the future rug event; for negatives it
ends at the label horizon. The feature is intentionally invalid for deployment.
Its purpose is to quantify how much discrimination can be manufactured by the
baseline's label-dependent snapshot rule on the new dataset.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.artifacts import atomic_csv
from src.config import load_config, resolve
from src.split import make_split
from src.train import best_threshold, bootstrap_intervals, evaluate, fit_model


def run(cfg: dict) -> pd.DataFrame:
    proc, results = resolve(cfg["paths"]["processed"]), resolve(cfg["paths"]["results"])
    seed, h_days = cfg["train"]["seed"], cfg["label"]["H_days"]
    rows = []
    protocols = {
        "random": {"kind": "random", "val_size": 0.15, "test_size": 0.15},
        "temporal": {"kind": "temporal", "train_end": "2022-01-01", "val_end": "2022-07-01",
                     "test_end": None, "purge": True},
    }
    for n in cfg["windows_hours"]:
        df = pd.read_parquet(proc / f"dataset_N{n}h.parquet")
        df["oracle_window_h"] = np.where(df.y.eq(1), (df.rug_ts - df.t0) / 3600,
                                          n + h_days * 24)
        for protocol, spec in protocols.items():
            tr, va, te = make_split(df, spec, h_days, seed)
            cols = ["oracle_window_h"]
            model = fit_model("xgb", tr[cols], tr.y, va[cols], va.y, seed)
            pva = model.predict_proba(va[cols])[:, 1]
            pte = model.predict_proba(te[cols])[:, 1]
            threshold = best_threshold(va.y.to_numpy(), pva)
            y = te.y.to_numpy()
            rows.append({"probe": "label_dependent_window", "protocol": protocol,
                         "N_hours": n, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                         **evaluate(y, pte, threshold),
                         **bootstrap_intervals(y, pte, threshold,
                                               cfg["train"].get("bootstrap_samples", 0), seed + n)})
    out = pd.DataFrame(rows)
    atomic_csv(out, results / "leakage_probe.csv", index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/common.yaml")
    a = ap.parse_args()
    print(run(load_config(a.config)).to_string(index=False))


if __name__ == "__main__":
    main()
