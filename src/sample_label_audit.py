"""Create reproducible, stratified and blinded sheets for independent label review."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import atomic_csv
from src.config import load_config, resolve


REVIEW_COLUMNS = ["audit_id", "pair", "token", "t0", "init_weth", "pair_url", "token_url",
                  "reviewer_label", "confidence", "evidence_notes"]


def _has_review_data(path: Path) -> bool:
    if not path.exists():
        return False
    df = pd.read_csv(path, dtype=str).fillna("")
    return any(df.get(c, pd.Series(dtype=str)).str.strip().ne("").any()
               for c in ("reviewer_label", "confidence", "evidence_notes"))


def run(cfg: dict, per_group: int, seed: int, overwrite: bool = False) -> pd.DataFrame:
    results = resolve(cfg["paths"]["results"])
    review_paths = [results / "label_audit_reviewer_1.csv", results / "label_audit_reviewer_2.csv"]
    if not overwrite and any(_has_review_data(p) for p in review_paths):
        raise FileExistsError("review data already exist; pass --overwrite only to discard them intentionally")

    pairs = pd.read_parquet(resolve(cfg["paths"]["raw"]) / "pairs.parquet")
    rugs = pd.read_parquet(resolve(cfg["paths"]["processed"]) / "rugs.parquet")
    df = pairs.merge(rugs, on="pair", how="left", suffixes=("", "_rug"))
    df["audit_group"] = np.where(df.rug_type.eq("burn"), "rug_burn",
                                  np.where(df.rug_type.eq("swap"), "rug_swap", "no_detected_rug"))
    df["cohort_year"] = pd.to_datetime(df.created_ts, utc=True).dt.year
    sampled = pd.concat(
        [g.sample(min(per_group, len(g)), random_state=seed)
         for _, g in df.groupby(["audit_group", "cohort_year"], sort=True)],
        ignore_index=True,
    ).sort_values(["audit_group", "cohort_year", "pair"], kind="stable").reset_index(drop=True)
    sampled.insert(0, "audit_id", [f"AUD{i:04d}" for i in range(1, len(sampled) + 1)])
    sampled["pair_url"] = "https://etherscan.io/address/" + sampled.pair
    sampled["token_url"] = "https://etherscan.io/token/" + sampled.token
    if "rug_tx" in sampled:
        sampled["rug_tx_url"] = sampled.rug_tx.map(
            lambda x: f"https://etherscan.io/tx/{x}" if isinstance(x, str) and x else "")
    sampled["adjudicated_label"] = ""
    sampled["adjudication_notes"] = ""

    master_cols = [c for c in ["audit_id", "audit_group", "cohort_year", "pair", "token", "t0",
                               "init_weth", "rug_ts", "rug_type", "rug_drop", "rug_tx", "pair_url",
                               "token_url", "rug_tx_url", "adjudicated_label", "adjudication_notes"]
                   if c in sampled]
    master = sampled[master_cols]
    atomic_csv(master, results / "label_audit_sample.csv", index=False)

    visible = sampled[[c for c in REVIEW_COLUMNS if c not in
                       {"reviewer_label", "confidence", "evidence_notes"}]].copy()
    visible["reviewer_label"] = ""
    visible["confidence"] = ""
    visible["evidence_notes"] = ""
    for i, path in enumerate(review_paths, start=1):
        sheet = visible.sample(frac=1, random_state=seed + i).reset_index(drop=True)
        atomic_csv(sheet[REVIEW_COLUMNS], path, index=False)
    return master


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/common.yaml")
    ap.add_argument("--per-group-year", type=int, default=25)
    ap.add_argument("--overwrite", action="store_true",
                    help="discard existing reviewer entries and regenerate all sheets")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = run(cfg, a.per_group_year, cfg["train"]["seed"], a.overwrite)
    print(out.audit_group.value_counts().to_string())
    print("Created blinded reviewer sheets: results/label_audit_reviewer_1.csv and _2.csv")


if __name__ == "__main__":
    main()
