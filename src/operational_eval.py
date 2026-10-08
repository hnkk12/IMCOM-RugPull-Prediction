"""Measure end-to-end coverage after accounting for rugs before prediction time.

The ordinary recall in metrics.csv is conditional on a pool surviving until t0+N.
This report additionally divides true positives by all rugs occurring before or
within the target horizon, making the cost of waiting operationally visible.
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.artifacts import atomic_csv
from src.config import load_config, resolve
from src.decode import to_unix
from src.label import assign_labels
from src.split import _ts


def run(cfg: dict, setup: str = "C_no_token_accounting", model: str = "lgbm") -> pd.DataFrame:
    raw = resolve(cfg["paths"]["raw"])
    proc = resolve(cfg["paths"]["processed"])
    results = resolve(cfg["paths"]["results"])
    pairs = pd.read_parquet(raw / "pairs.parquet")
    for c in ("created_ts", "token_deployed_ts"):
        pairs[c] = to_unix(pairs[c])
    rugs = pd.read_parquet(proc / "rugs.parquet")
    split = cfg["split"]
    if split.get("kind") != "temporal":
        raise ValueError("operational coverage requires a temporal setup")
    test_start = _ts(split["val_end"])
    test_end = _ts(split["test_end"]) if split.get("test_end") else float("inf")
    L, S = cfg["label"], cfg["study"]
    rows = []
    for n in cfg["windows_hours"]:
        cutoff = _ts(S["data_cutoff"]) if S.get("data_cutoff") else int(
            pd.Timestamp.now(tz="UTC").timestamp())
        lab = assign_labels(rugs, pairs, n, L["H_days"], S["pool_horizon_days"],
                            cutoff, L["min_init_weth"])
        test = lab[lab.t0.ge(test_start) & lab.t0.lt(test_end)]
        early = int(test.status.eq("early_rug").sum())
        eligible = int(test.status.eq("pos").sum())
        pred = pd.read_parquet(results / f"setup_{setup}" / f"pred_N{n}h_{model}.parquet")
        threshold = float(pred.threshold.iloc[0]) if "threshold" in pred else 0.5
        tp = int(((pred.y == 1) & (pred.p >= threshold)).sum())
        denom = early + eligible
        rows.append({"setup": setup, "model": model, "N_hours": n, "early_rugs": early,
                     "eligible_positive_rugs": eligible, "detected_positive_rugs": tp,
                     "max_achievable_coverage": eligible / denom if denom else float("nan"),
                     "end_to_end_coverage": tp / denom if denom else float("nan")})
    out = pd.DataFrame(rows)
    atomic_csv(out, results / "operational_coverage.csv", index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/setup_C_no_token_accounting.yaml")
    ap.add_argument("--setup", default="C_no_token_accounting")
    ap.add_argument("--model", default="lgbm")
    a = ap.parse_args()
    print(run(load_config(a.config), a.setup, a.model).to_string(index=False))


if __name__ == "__main__":
    main()
