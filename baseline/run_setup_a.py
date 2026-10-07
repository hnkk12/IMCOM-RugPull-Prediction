"""Setup A: reproduce the baseline on Dataset_v1.9.csv + leakage probes.

    python baseline/run_setup_a.py

Protocol (as in the baseline): 18 features, random stratified split. We report
5-fold stratified CV (mean +- std) for XGBoost and an MLP with the baseline ANN's
layer sizes (read from ann97.h5 when h5py is available, MinMax-scaled inputs as in predict.py).

Leakage probes (same CV):
  lifecycle_only : only active_period (first -> last transaction of the whole pool life)
  window_only    : only the length of the window the baseline used to compute features
                   (positives: up to the rug, negatives: a fixed/long period)
  repo18_no_time : the 18 features minus the ones normalized by lifetime (per_week, mean_period)
plus repo18_temporal: train on older pools, test on newer ones (70/30 by creation date).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "setup_A"
SEED = 42

REPO18 = ["mint_count_per_week", "burn_count_per_week", "mint_ratio", "swap_ratio", "burn_ratio",
          "mint_mean_period", "swap_mean_period", "burn_mean_period", "swap_in_per_week", "swap_out_per_week",
          "swap_rate", "lp_avg", "lp_std", "lp_creator_holding_ratio", "lp_lock_ratio", "token_burn_ratio",
          "token_creator_holding_ratio", "number_of_token_creation_of_creator"]
TIME_NORMALIZED = [c for c in REPO18 if c.endswith("_per_week") or c.endswith("_mean_period")]


def load() -> pd.DataFrame:
    d = pd.read_csv(HERE / "Dataset_v1.9.csv", encoding="utf-8-sig")
    lab = pd.read_csv(HERE / "2. Feature" / "4. Labeling File" / "Labeling_v3.3.csv",
                      encoding="utf-8-sig", low_memory=False)
    lab = lab[["id", "active_period", "last_transaction_timestamp", "feature_timestamp", "createdAtTimestamp"]]
    d = d.merge(lab.drop_duplicates("id"), on="id", how="left")
    d["y"] = d.Label.astype(str).str.upper().eq("TRUE").astype(int)
    first_tx = d.last_transaction_timestamp - d.active_period
    d["window_only"] = pd.to_numeric(d.feature_timestamp, errors="coerce") - first_tx
    d["lifecycle_only"] = d.active_period
    d["created"] = pd.to_datetime(d.createdAtTimestamp, errors="coerce")
    return d


def ann_hidden_layers() -> tuple[int, ...]:
    try:
        import h5py
        with h5py.File(HERE / "ann97.h5", "r") as f:
            cfg = json.loads(f.attrs["model_config"])
        units = [l["config"]["units"] for l in cfg["config"]["layers"] if l["class_name"] == "Dense"]
        return tuple(units[:-1]) or (64, 32)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not read ann97.h5 ({e}); using (64, 32)")
        return (64, 32)


def models(hidden):
    return {
        "xgb": lambda: XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, tree_method="hist", random_state=SEED),
        "ann": lambda: make_pipeline(MinMaxScaler(), MLPClassifier(hidden_layer_sizes=hidden, max_iter=60, batch_size=512,
                                                                   early_stopping=True, n_iter_no_change=8,
                                                                   random_state=SEED)),
    }


def scores(y, p) -> dict:
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "f1": f1_score(y, p >= 0.5), "accuracy": accuracy_score(y, p >= 0.5)}


def cv(df, cols, make, k=5) -> dict:
    X, y = df[cols].fillna(0).to_numpy(), df.y.to_numpy()
    res = []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=SEED).split(X, y):
        m = make()
        m.fit(X[tr], y[tr])
        res.append(scores(y[te], m.predict_proba(X[te])[:, 1]))
    r = pd.DataFrame(res)
    return {**{f"{c}": r[c].mean() for c in r}, **{f"{c}_std": r[c].std() for c in r}}


def temporal(df, cols, make, frac=0.7) -> dict:
    d = df.dropna(subset=["created"]).sort_values("created")
    cut = int(len(d) * frac)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    m = make()
    m.fit(tr[cols].fillna(0).to_numpy(), tr.y.to_numpy())
    r = scores(te.y.to_numpy(), m.predict_proba(te[cols].fillna(0).to_numpy())[:, 1])
    r["split_date"] = str(te.created.iloc[0].date())
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"rows={len(df):,}  positives={df.y.mean():.1%}  (majority-class accuracy = {max(df.y.mean(), 1 - df.y.mean()):.3f})")
    print("median feature window (hours) by label:")
    print((df.groupby("y").window_only.median() / 3600).round(2).to_string())

    hidden = ann_hidden_layers()
    M = models(hidden)
    rows = []
    runs = [("repo18", REPO18, ["xgb", "ann"]),
            ("lifecycle_only", ["lifecycle_only"], ["xgb"]),
            ("window_only", ["window_only"], ["xgb"]),
            ("repo18_no_time", [c for c in REPO18 if c not in TIME_NORMALIZED], ["xgb"])]
    for name, cols, ms in runs:
        for m in ms:
            r = {"experiment": name, "model": m, "protocol": "random_5fold", **cv(df, cols, M[m])}
            rows.append(r)
            print(f"{name:16s} {m:4s} ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['pr_auc']:.4f}  F1={r['f1']:.4f}  acc={r['accuracy']:.4f}")
    for m in ["xgb", "ann"]:
        r = {"experiment": "repo18_temporal", "model": m, "protocol": "temporal_70_30", **temporal(df, REPO18, M[m])}
        rows.append(r)
        print(f"repo18_temporal  {m:4s} ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['pr_auc']:.4f}  F1={r['f1']:.4f}  (test from {r['split_date']})")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "metrics.csv", index=False)
    (OUT / "notes.json").write_text(json.dumps({"ann_hidden_layers": hidden, "n": len(df),
                                                "pos_rate": float(df.y.mean())}, indent=2))
    print(f"\nsaved -> {OUT / 'metrics.csv'}")


if __name__ == "__main__":
    main()
