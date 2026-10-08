"""Generate publication-oriented tables and figures from saved results."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from src.artifacts import atomic_csv
from src.config import load_config, resolve


def _savefig(fig, path: str | Path, dpi: int = 200) -> None:
    """Publish a complete figure atomically."""
    path = Path(path)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    fig.savefig(tmp, dpi=dpi, bbox_inches="tight")
    os.replace(tmp, path)
    plt.close(fig)


def _box(ax, x, y, w, h, text, color):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                           facecolor=color, edgecolor="0.25", linewidth=1)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def run(cfg: dict):
    results = resolve(cfg["paths"]["results"])
    processed = resolve(cfg["paths"]["processed"])
    out = results / "paper_artifacts"
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(results / "summary.csv")

    main_names = ["B_no_token_accounting", "C_no_token_accounting", "C_repo_only",
                  "D_no_token_accounting"]
    main = summary[(summary.model == "lgbm") & summary.setup.isin(main_names)].copy()
    main["setup"] = pd.Categorical(main.setup, categories=main_names, ordered=True)
    main = main.sort_values(["setup", "N_hours"])
    atomic_csv(main, out / "table_main_lgbm.csv", index=False)
    rolling = summary[summary.setup.eq("C_rolling")].sort_values(["N_hours", "fold"])
    atomic_csv(rolling, out / "table_rolling_temporal.csv", index=False)
    cold = summary[(summary.model == "lgbm") & summary.setup.isin(
        ["D_no_token_accounting", "D_coldstart", "D_template_coldstart"])]
    atomic_csv(cold.sort_values(["setup", "N_hours"]), out / "table_cold_start.csv", index=False)
    sensitivity = summary[
        summary.setup.isin(["C_no_token_accounting", "C_label95", "C_cumulative99"]) &
        summary.model.eq("lgbm")].sort_values(["setup", "N_hours"])
    atomic_csv(sensitivity, out / "table_label_sensitivity_models.csv", index=False)
    protocol = pd.DataFrame([
        {"protocol": "A", "feature_window": "label-dependent/full lifecycle",
         "split": "random 5-fold", "purge": False, "role": "prior-study replication / leakage probe"},
        {"protocol": "B", "feature_window": "fixed [t0, t0+N]",
         "split": "stratified random", "purge": False, "role": "random-split optimism"},
        {"protocol": "C", "feature_window": "fixed [t0, t0+N]",
         "split": "chronological train/val/test", "purge": True, "role": "primary deployment-like evaluation"},
    ])
    atomic_csv(protocol, out / "table_protocol_comparison.csv", index=False)
    quality_path = results / "data_quality_summary.csv"
    if quality_path.exists():
        atomic_csv(pd.read_csv(quality_path), out / "table_data_quality.csv", index=False)
    audit_path = results / "label_audit_metrics.csv"
    if audit_path.exists():
        atomic_csv(pd.read_csv(audit_path), out / "table_label_audit.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True)
    display = {"B_no_token_accounting": "B: random", "C_no_token_accounting": "C: temporal",
               "C_repo_only": "C: repo-only", "D_no_token_accounting": "D: C + contract"}
    for setup in main_names:
        d = main[main.setup.eq(setup)].sort_values("N_hours")
        for ax, metric, label in zip(axes, ("roc_auc", "pr_auc"), ("ROC-AUC", "PR-AUC")):
            low, high = d[f"{metric}_ci_low"], d[f"{metric}_ci_high"]
            ax.errorbar(d.N_hours, d[metric], yerr=np.vstack([d[metric] - low, high - d[metric]]),
                        marker="o", capsize=3, label=display[setup])
            ax.set(xlabel="Observation window N (hours)", ylabel=label, xticks=[1, 6, 24])
            ax.grid(alpha=0.25)
    primary = main[main.setup.eq("C_no_token_accounting")].sort_values("N_hours")
    axes[1].plot(primary.N_hours, primary.pr_baseline, "--", color="0.35", label="C: prevalence")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, out / "figure_main_performance.png")

    op = pd.read_csv(results / "operational_coverage.csv")
    atomic_csv(op, out / "table_operational_coverage.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(op.N_hours, op.max_achievable_coverage, marker="o", label="maximum achievable")
    ax.plot(op.N_hours, op.end_to_end_coverage, marker="o", label="model realized")
    ax.set(xlabel="Observation window N (hours)", ylabel="Coverage of target-period rugs",
           xticks=[1, 6, 24], ylim=(0, max(op.max_achievable_coverage) * 1.15))
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); _savefig(fig, out / "figure_operational_coverage.png")

    cohort = summary[(summary.model == "lgbm") &
                     summary.setup.isin(["C_no_token_accounting", "C_common"])].copy()
    cohort = cohort.sort_values(["setup", "N_hours"])
    atomic_csv(cohort, out / "table_landmark_vs_common_cohort.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for setup, label in (("C_no_token_accounting", "landmark population"),
                         ("C_common", "common survivor cohort")):
        d = cohort[cohort.setup.eq(setup)].sort_values("N_hours")
        ax.plot(d.N_hours, d.pr_auc, marker="o", label=label)
    ax.set(xlabel="Observation window N (hours)", ylabel="PR-AUC", xticks=[1, 6, 24])
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); _savefig(fig, out / "figure_landmark_vs_common.png")

    # Cohort attrition: the same sampled population reaches different landmark risk sets.
    status = pd.read_csv(processed / "label_status.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _box(ax, .02, .39, .16, .22, "Sampled WETH pairs\n23,400", "#dbeafe")
    _box(ax, .25, .39, .18, .22, "Liquidity + ≥0.1 ETH\n22,072", "#dcfce7")
    ax.annotate("", xy=(.25, .50), xytext=(.18, .50), arrowprops={"arrowstyle": "->"})
    ax.text(.215, .34, "−803 no liquidity\n−525 dust", ha="center", fontsize=8)
    colors = ["#fef3c7", "#fde68a", "#fcd34d"]
    for i, (n, y) in enumerate(zip((1, 6, 24), (.72, .43, .14))):
        col = f"N={n}h"
        early, censored = int(status.loc["early_rug", col]), int(status.loc["censored", col])
        pos, neg = int(status.loc["pos", col]), int(status.loc["neg", col])
        _box(ax, .55, y, .18, .16, f"N={n}h landmark\n−{early:,} early rugs\n−{censored:,} censored", colors[i])
        _box(ax, .82, y, .16, .16, f"Dataset: {pos + neg:,}\n{pos:,} pos / {neg:,} neg", "#e0e7ff")
        ax.annotate("", xy=(.55, y + .08), xytext=(.43, .50), arrowprops={"arrowstyle": "->"})
        ax.annotate("", xy=(.82, y + .08), xytext=(.73, y + .08), arrowprops={"arrowstyle": "->"})
    ax.set_title("Cohort construction and landmark-specific attrition", fontsize=13)
    _savefig(fig, out / "figure_cohort_construction.png")

    # Conceptual timeline used by labeling and purged temporal evaluation.
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.set_xlim(-.05, 1.05); ax.set_ylim(-.15, 1.05); ax.axis("off")
    y = .58
    ax.annotate("", xy=(1, y), xytext=(0, y), arrowprops={"arrowstyle": "->", "lw": 1.5})
    for x, label in ((0, "$t_0$\nfirst liquidity"), (.3, "$t_0+N$\nprediction"),
                     (.82, "$t_0+N+H$\nlabel horizon")):
        ax.plot([x, x], [y - .08, y + .08], color="black")
        ax.text(x, y - .14, label, ha="center", va="top", fontsize=9)
    ax.plot([0, .3], [y + .12, y + .12], lw=8, solid_capstyle="butt", color="#60a5fa")
    ax.text(.15, y + .2, "feature window [t₀, t₀+N]", ha="center", fontsize=9)
    ax.plot([.3, .82], [y + .12, y + .12], lw=8, solid_capstyle="butt", color="#f59e0b")
    ax.text(.56, y + .2, "target horizon (t₀+N, t₀+N+H]", ha="center", fontsize=9)
    ax.scatter([.18], [y], marker="x", s=70, color="#dc2626", zorder=3)
    ax.text(.18, y - .31, "early rug\nexcluded", ha="center", color="#991b1b", fontsize=9)
    ax.scatter([.55], [y], marker="*", s=110, color="#dc2626", zorder=3)
    ax.text(.55, y - .31, "positive rug", ha="center", color="#991b1b", fontsize=9)
    ax.axvline(.7, ymin=.49, ymax=.92, ls="--", color="0.35")
    ax.text(.7, .98, "split boundary", ha="center", fontsize=9)
    ax.text(.86, .9, "purge a training row if its\nlabel horizon crosses the boundary",
            ha="center", va="center", fontsize=8.5)
    ax.text(.9, .25, "censor if observation ends\nbefore the label horizon", ha="center", fontsize=8.5)
    ax.set_title("Leakage-resistant landmark labeling and temporal purge", fontsize=13)
    _savefig(fig, out / "figure_landmark_timeline.png")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/common.yaml")
    a = ap.parse_args()
    print(run(load_config(a.config)))


if __name__ == "__main__":
    main()
