"""Automated processed-data quality report for paper and pipeline audits."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.artifacts import atomic_csv
from src.config import load_config, resolve
from src.train import GROUPS, feature_cols


def run(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    proc, results = resolve(cfg["paths"]["processed"]), resolve(cfg["paths"]["results"])
    summary, detail = [], []
    groups = list(GROUPS)
    bounded = ("_share", "_ratio", "_hhi")
    for n in cfg["windows_hours"]:
        df = pd.read_parquet(proc / f"dataset_N{n}h.parquet")
        cols = feature_cols(df, groups)
        numeric = df[cols].astype("float64")
        finite = np.isfinite(numeric.to_numpy()) | numeric.isna().to_numpy()
        summary.append({"N_hours": n, "rows": len(df), "columns": len(df.columns),
                        "duplicate_pairs": int(df.pair.duplicated().sum()),
                        "positive_rate": float(df.y.mean()),
                        "nonfinite_values": int((~finite).sum()),
                        "features_missing_over_20pct": int((numeric.isna().mean() > .2).sum()),
                        "invalid_token_accounting": int(df.get("token_accounting_valid",
                                                               pd.Series(1, index=df.index)).eq(0).sum())})
        for c in cols:
            s = numeric[c]
            invalid_bounded = 0
            if c.endswith(bounded) and c not in ("sell_buy_vol_ratio", "post_launch_mint_ratio",
                                                  "burn_weth_ratio"):
                invalid_bounded = int(((s < -1e-9) | (s > 1 + 1e-9)).sum())
            detail.append({"N_hours": n, "feature": c, "missing_rate": float(s.isna().mean()),
                           "min": float(s.min()) if s.notna().any() else np.nan,
                           "median": float(s.median()) if s.notna().any() else np.nan,
                           "max": float(s.max()) if s.notna().any() else np.nan,
                           "invalid_bounded_values": invalid_bounded})
    a, b = pd.DataFrame(summary), pd.DataFrame(detail)
    atomic_csv(a, results / "data_quality_summary.csv", index=False)
    atomic_csv(b, results / "data_quality_features.csv", index=False)
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/common.yaml")
    a = ap.parse_args()
    summary, _ = run(load_config(a.config))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
