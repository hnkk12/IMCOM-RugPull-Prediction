"""Train + evaluate one setup for every window N.

    python -m src.train --config experiments/setup_C.yaml

Writes results/<setup>/metrics.csv, predictions, and SHAP summaries (if train.shap),
and upserts results/summary.csv so setups can be compared side by side.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import load_config, resolve
from src.contract_features import CONTRACT_FEATURES
from src.features import NEW_FEATURES, REPO_FEATURES
from src.split import make_split

GROUPS = {"repo": REPO_FEATURES, "new": NEW_FEATURES, "contract": CONTRACT_FEATURES}


def feature_cols(df: pd.DataFrame, groups: list[str]) -> list[str]:
    cols = [c for g in groups for c in GROUPS[g]]
    missing = [c for c in cols if c not in df]
    if missing:
        print(f"  [warn] {len(missing)} feature(s) not in dataset, skipped: {missing[:5]}...")
    return list(dict.fromkeys(c for c in cols if c in df))


def _signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def fit_model(name: str, Xtr, ytr, Xva, yva, seed: int):
    if name == "lgbm":
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
        m = LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
              callbacks=[early_stopping(150, verbose=False), log_evaluation(0)])
        return m
    if name == "xgb":
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                          tree_method="hist", eval_metric="aucpr", early_stopping_rounds=150, random_state=seed)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return m
    if name == "logreg":
        m = make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(_signed_log), StandardScaler(),
                          LogisticRegression(max_iter=5000, class_weight="balanced"))
        m.fit(Xtr, ytr)
        return m
    raise ValueError(name)


def best_threshold(y, p) -> float:
    """Threshold maximizing F1 on validation (applied unchanged to test)."""
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 199)))
    f1s = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))]) if len(grid) else 0.5


def evaluate(y, p, thr) -> dict:
    yhat = p >= thr
    single = len(np.unique(y)) < 2
    return {"roc_auc": np.nan if single else roc_auc_score(y, p),
            "pr_auc": np.nan if single else average_precision_score(y, p),
            "f1": f1_score(y, yhat, zero_division=0), "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0), "brier": brier_score_loss(y, p),
            "threshold": thr, "pos_rate": float(np.mean(y))}


def shap_report(model, X: pd.DataFrame, out_prefix, sample: int, seed: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    Xs = X.sample(min(sample, len(X)), random_state=seed)
    sv = shap.TreeExplainer(model).shap_values(Xs)
    if isinstance(sv, list):
        sv = sv[1]
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    imp = pd.Series(np.abs(sv).mean(0), index=Xs.columns).sort_values(ascending=False)
    imp.rename("mean_abs_shap").to_csv(f"{out_prefix}.csv")
    shap.summary_plot(sv, Xs, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", dpi=150)
    plt.close()


def run(cfg: dict):
    name = cfg["name"]
    proc = resolve(cfg["paths"]["processed"])
    out = resolve(cfg["paths"]["results"]) / f"setup_{name}"
    out.mkdir(parents=True, exist_ok=True)
    T, seed = cfg["train"], cfg["train"]["seed"]
    rows = []
    for n in cfg["windows_hours"]:
        path = proc / f"dataset_N{n}h.parquet"
        if not path.exists():
            print(f"[skip] {path} not found - run src.build_dataset first")
            continue
        df = pd.read_parquet(path)
        cols = feature_cols(df, cfg["features"])
        # XGBoost works in float32: inf and huge ratios (x / 1e-12) become missing / clipped
        df[cols] = df[cols].astype("float64").replace([np.inf, -np.inf], np.nan).clip(-1e30, 1e30)
        tr, va, te = make_split(df, cfg["split"], cfg["label"]["H_days"], seed)
        print(f"[{name}] N={n}h  train={len(tr):,} ({tr.y.mean():.2%} pos)  val={len(va):,}  "
              f"test={len(te):,} ({te.y.mean():.2%} pos)  features={len(cols)}")
        if min(len(tr), len(va), len(te)) == 0 or tr.y.nunique() < 2:
            print("  [skip] empty split - check split dates against your data range")
            continue
        for mname in T["models"]:
            m = fit_model(mname, tr[cols], tr.y, va[cols], va.y, seed)
            pva, pte = m.predict_proba(va[cols])[:, 1], m.predict_proba(te[cols])[:, 1]
            thr = best_threshold(va.y.to_numpy(), pva)
            r = {"setup": name, "N_hours": n, "model": mname, "n_train": len(tr), "n_val": len(va),
                 "n_test": len(te), "n_features": len(cols), **evaluate(te.y.to_numpy(), pte, thr)}
            rows.append(r)
            print(f"  {mname:7s} ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['pr_auc']:.4f}  F1={r['f1']:.4f}")
            te[["pair", "t0", "y"]].assign(p=pte).to_parquet(out / f"pred_N{n}h_{mname}.parquet")
            if T.get("shap") and mname in ("lgbm", "xgb"):
                shap_report(m, te[cols], out / f"shap_N{n}h_{mname}", T.get("shap_sample", 5000), seed)

    res = pd.DataFrame(rows)
    res.to_csv(out / "metrics.csv", index=False)
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    summary = resolve(cfg["paths"]["results"]) / "summary.csv"
    old = pd.read_csv(summary) if summary.exists() else pd.DataFrame()
    if len(old):
        old = old[old.setup.astype(str) != str(name)]
    pd.concat([old, res], ignore_index=True).to_csv(summary, index=False)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", help="use <root>/processed and <root>/results (e.g. data/synthetic)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.data_root:
        cfg["paths"].update(processed=f"{a.data_root}/processed", results=f"{a.data_root}/results")
    run(cfg)


if __name__ == "__main__":
    main()
