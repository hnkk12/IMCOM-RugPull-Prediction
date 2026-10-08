"""Train + evaluate one setup for every window N.

    python -m src.train --config experiments/setup_C_no_token_accounting.yaml

Writes results/<setup>/metrics.csv, predictions, and SHAP summaries (if train.shap),
and upserts results/summary.csv so setups can be compared side by side.
"""
from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.artifacts import atomic_csv, atomic_json, atomic_parquet
from src.config import load_config, resolve
from src.contract_features import CONTRACT_FEATURES
from src.features import NEW_FEATURES, REPO_FEATURES, ROBUST_FEATURES, ROBUST_REPO_FEATURES
from src.split import make_splits

GROUPS = {"repo": REPO_FEATURES, "repo_robust": ROBUST_REPO_FEATURES,
          "new": NEW_FEATURES, "robust": ROBUST_FEATURES,
          "contract": CONTRACT_FEATURES}


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
    precision, recall, thresholds = precision_recall_curve(y, p)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[int(np.nanargmax(f1))])


def calibration_error(y, p, n_bins: int = 10) -> float:
    """Expected calibration error using fixed-width probability bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        keep = ids == b
        if keep.any():
            ece += keep.mean() * abs(float(np.mean(y[keep])) - float(np.mean(p[keep])))
    return ece


def evaluate(y, p, thr) -> dict:
    yhat = p >= thr
    single = len(np.unique(y)) < 2
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"roc_auc": np.nan if single else roc_auc_score(y, p),
            "pr_auc": np.nan if single else average_precision_score(y, p),
            "f1": f1_score(y, yhat, zero_division=0),
            "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0),
            "specificity": tn / max(tn + fp, 1),
            "balanced_accuracy": balanced_accuracy_score(y, yhat),
            "brier": brier_score_loss(y, p), "ece": calibration_error(y, p),
            "threshold": thr, "pos_rate": float(np.mean(y)), "pr_baseline": float(np.mean(y)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def bootstrap_intervals(y, p, thr, n_boot: int, seed: int) -> dict:
    """Stratified bootstrap CIs, preserving the observed class prevalence."""
    if n_boot <= 0 or len(np.unique(y)) < 2:
        return {}
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == cls) for cls in (0, 1)]
    keep = ("roc_auc", "pr_auc", "f1", "brier", "ece")
    draws = {k: [] for k in keep}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        r = evaluate(y[idx], p[idx], thr)
        for k in keep:
            draws[k].append(r[k])
    out = {}
    for k, values in draws.items():
        out[f"{k}_ci_low"], out[f"{k}_ci_high"] = np.quantile(values, [0.025, 0.975])
    return out


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


def calibration_report(y, p, out_prefix, n_bins: int = 10):
    """Save reliability-curve points and a compact calibration plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    observed, predicted = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    atomic_csv(pd.DataFrame({"mean_predicted_probability": predicted,
                             "observed_positive_rate": observed}),
               f"{out_prefix}.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="0.5", label="ideal")
    ax.plot(predicted, observed, marker="o", label="model")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed positive rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=150)
    plt.close(fig)


def run(cfg: dict):
    name = cfg["name"]
    proc = resolve(cfg["paths"]["processed"])
    out = resolve(cfg["paths"]["results"]) / f"setup_{name}"
    out.mkdir(parents=True, exist_ok=True)
    T, seed = cfg["train"], cfg["train"]["seed"]
    rows = []
    datasets = {}
    for n in cfg["windows_hours"]:
        path = proc / f"dataset_N{n}h.parquet"
        if not path.exists():
            print(f"[skip] {path} not found - run src.build_dataset first")
            continue
        datasets[n] = pd.read_parquet(path)

    if cfg.get("cohort", {}).get("common_across_windows") and datasets:
        common = set.intersection(*(set(df.pair) for df in datasets.values()))
        datasets = {n: df[df.pair.isin(common)].copy() for n, df in datasets.items()}
        print(f"[{name}] common cohort across windows: {len(common):,} pools")

    for n, df in datasets.items():
        cols = feature_cols(df, cfg["features"])
        # XGBoost works in float32: inf and huge ratios (x / 1e-12) become missing / clipped
        df[cols] = df[cols].astype("float64").replace([np.inf, -np.inf], np.nan).clip(-1e30, 1e30)
        for fold, tr, va, te in make_splits(df, cfg["split"], cfg["label"]["H_days"], seed):
            print(f"[{name}/{fold}] N={n}h  train={len(tr):,} ({tr.y.mean():.2%} pos)  val={len(va):,}  "
                  f"test={len(te):,} ({te.y.mean():.2%} pos)  features={len(cols)}")
            if min(len(tr), len(va), len(te)) == 0 or tr.y.nunique() < 2 or va.y.nunique() < 2:
                print("  [skip] empty/single-class split - check split dates against your data range")
                continue
            suffix = f"N{n}h" if fold == "main" else f"N{n}h_{fold}"
            for mi, mname in enumerate(T["models"]):
                m = fit_model(mname, tr[cols], tr.y, va[cols], va.y, seed)
                pva, pte = m.predict_proba(va[cols])[:, 1], m.predict_proba(te[cols])[:, 1]
                thr = best_threshold(va.y.to_numpy(), pva)
                yte = te.y.to_numpy()
                r = {"setup": name, "fold": fold, "N_hours": n, "model": mname,
                     "n_train": len(tr), "n_val": len(va), "n_test": len(te), "n_features": len(cols),
                     **evaluate(yte, pte, thr),
                     **bootstrap_intervals(yte, pte, thr, T.get("bootstrap_samples", 0),
                                           seed + 100 * int(n) + mi)}
                rows.append(r)
                print(f"  {mname:7s} ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['pr_auc']:.4f}  F1={r['f1']:.4f}")
                pred = te[["pair", "t0", "y"]].copy().assign(p=pte, threshold=thr, fold=fold)
                atomic_parquet(pred, out / f"pred_{suffix}_{mname}.parquet")
                if T.get("calibration_plots", True) and fold == "main":
                    calibration_report(yte, pte, out / f"calibration_{suffix}_{mname}")
                if T.get("save_models", True):
                    model_path = out / f"model_{suffix}_{mname}.joblib"
                    tmp = model_path.with_name(model_path.name + ".tmp")
                    joblib.dump(m, tmp)
                    os.replace(tmp, model_path)
                if T.get("shap") and fold == "main" and mname in ("lgbm", "xgb"):
                    shap_report(m, te[cols], out / f"shap_{suffix}_{mname}",
                                T.get("shap_sample", 5000), seed)

    res = pd.DataFrame(rows)
    atomic_csv(res, out / "metrics.csv", index=False)
    atomic_json(cfg, out / "config.json", indent=2, default=str)
    summary = resolve(cfg["paths"]["results"]) / "summary.csv"
    old = pd.read_csv(summary) if summary.exists() else pd.DataFrame()
    if len(old):
        old = old[old.setup.astype(str) != str(name)]
    atomic_csv(pd.concat([old, res], ignore_index=True), summary, index=False)
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
