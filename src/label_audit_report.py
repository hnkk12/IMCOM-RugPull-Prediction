"""Validate independent reviews and report label-audit agreement and error rates."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.artifacts import atomic_csv, atomic_json
from src.config import load_config, resolve


LABELS = {"rug", "not_rug", "uncertain"}


def _normalise(series: pd.Series) -> pd.Series:
    aliases = {"yes": "rug", "no": "not_rug", "non_rug": "not_rug", "not rug": "not_rug",
               "unknown": "uncertain", "?": "uncertain"}
    out = series.fillna("").astype(str).str.strip().str.lower()
    return out.replace(aliases)


def build_report(master: pd.DataFrame, reviewer_1: pd.DataFrame, reviewer_2: pd.DataFrame,
                 population_counts: dict[str, int] | None = None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    for name, df in (("master", master), ("reviewer_1", reviewer_1), ("reviewer_2", reviewer_2)):
        if "audit_id" not in df or df.audit_id.duplicated().any():
            raise ValueError(f"{name} must contain unique audit_id values")
    ids = set(master.audit_id)
    if set(reviewer_1.audit_id) != ids or set(reviewer_2.audit_id) != ids:
        raise ValueError("master and reviewer sheets do not contain the same audit_id set")

    r1 = reviewer_1[["audit_id", "reviewer_label"]].rename(columns={"reviewer_label": "reviewer_1"})
    r2 = reviewer_2[["audit_id", "reviewer_label"]].rename(columns={"reviewer_label": "reviewer_2"})
    merged = master.merge(r1, on="audit_id").merge(r2, on="audit_id")
    merged["reviewer_1"] = _normalise(merged.reviewer_1)
    merged["reviewer_2"] = _normalise(merged.reviewer_2)
    merged["adjudicated_label"] = _normalise(
        merged.get("adjudicated_label", pd.Series("", index=merged.index)))

    invalid = sorted((set(merged.reviewer_1) | set(merged.reviewer_2) |
                      set(merged.adjudicated_label)) - LABELS - {""})
    if invalid:
        raise ValueError(f"invalid labels: {invalid}; allowed values are {sorted(LABELS)}")

    complete = merged.reviewer_1.isin(LABELS) & merged.reviewer_2.isin(LABELS)
    reviewed = merged[complete].copy()
    reviewed["reviewer_agreement"] = reviewed.reviewer_1.eq(reviewed.reviewer_2)
    reviewed["consensus_label"] = np.where(
        reviewed.reviewer_agreement, reviewed.reviewer_1, reviewed.adjudicated_label)
    reviewed["resolved"] = reviewed.consensus_label.isin(LABELS)

    if len(reviewed):
        agreement = float(reviewed.reviewer_agreement.mean())
        kappa = float(cohen_kappa_score(reviewed.reviewer_1, reviewed.reviewer_2,
                                        labels=sorted(LABELS)))
    else:
        agreement = kappa = None

    rows = []
    for group, g in merged.groupby("audit_group", sort=True):
        done = reviewed[reviewed.audit_group.eq(group)]
        resolved = done[done.resolved & done.consensus_label.ne("uncertain")]
        r1 = g[g.reviewer_1.isin({"rug", "not_rug"})]
        rug_n = int(resolved.consensus_label.eq("rug").sum())
        n = len(resolved)
        rows.append({
            "audit_group": group,
            "population_n": (population_counts or {}).get(group),
            "sample_n": len(g),
            "double_reviewed_n": len(done),
            "resolved_binary_n": n,
            "consensus_rug_n": rug_n,
            "consensus_not_rug_n": int(resolved.consensus_label.eq("not_rug").sum()),
            "estimated_rug_rate": rug_n / n if n else np.nan,
            "reviewer_1_binary_n": len(r1),
            "reviewer_1_rug_n": int(r1.reviewer_1.eq("rug").sum()),
            "reviewer_1_rug_rate": float(r1.reviewer_1.eq("rug").mean()) if len(r1) else np.nan,
        })
    strata = pd.DataFrame(rows)

    detected = strata[strata.audit_group.isin(["rug_burn", "rug_swap"])].dropna(
        subset=["estimated_rug_rate"])
    if len(detected) and detected.population_n.notna().all() and detected.population_n.sum():
        precision = float(np.average(detected.estimated_rug_rate, weights=detected.population_n))
    else:
        precision = None
    no_detect = strata[strata.audit_group.eq("no_detected_rug")]
    undetected_rate = (float(no_detect.estimated_rug_rate.iloc[0])
                       if len(no_detect) and pd.notna(no_detect.estimated_rug_rate.iloc[0]) else None)
    r1_detected = strata[strata.audit_group.isin(["rug_burn", "rug_swap"])].dropna(
        subset=["reviewer_1_rug_rate"])
    if len(r1_detected) and r1_detected.population_n.notna().all() and r1_detected.population_n.sum():
        r1_precision = float(np.average(r1_detected.reviewer_1_rug_rate,
                                        weights=r1_detected.population_n))
    else:
        r1_precision = None
    r1_no_detect_rate = (float(no_detect.reviewer_1_rug_rate.iloc[0])
                         if len(no_detect) and pd.notna(no_detect.reviewer_1_rug_rate.iloc[0]) else None)

    disagreements = reviewed[~reviewed.reviewer_agreement][
        [c for c in ["audit_id", "pair", "token", "audit_group", "reviewer_1", "reviewer_2",
                     "adjudicated_label", "adjudication_notes"] if c in reviewed]
    ]
    report = {
        "sample_n": len(merged),
        "double_reviewed_n": int(complete.sum()),
        "double_review_progress": float(complete.mean()),
        "raw_reviewer_agreement": agreement,
        "cohen_kappa_three_class": kappa,
        "unresolved_disagreements_n": int((~reviewed.reviewer_agreement & ~reviewed.resolved).sum()),
        "detected_label_precision_population_weighted": precision,
        "detected_label_false_positive_rate_population_weighted": None if precision is None else 1 - precision,
        "no_detected_stratum_manual_rug_rate": undetected_rate,
        "reviewer_1_complete_n": int(merged.reviewer_1.isin(LABELS).sum()),
        "reviewer_1_detected_precision_population_weighted": r1_precision,
        "reviewer_1_detected_false_positive_rate_population_weighted": (
            None if r1_precision is None else 1 - r1_precision),
        "reviewer_1_no_detected_stratum_rug_rate": r1_no_detect_rate,
        "reviewer_1_provenance": "AI-assisted expert rubric; not an independent human review",
        "note": ("Stratified audit estimates are label-rule diagnostics, not classifier performance. "
                 "The no-detected rate is not a conventional false-negative rate."),
    }
    return report, strata, disagreements


def _population_counts(cfg: dict) -> dict[str, int]:
    pairs = pd.read_parquet(resolve(cfg["paths"]["raw"]) / "pairs.parquet", columns=["pair"])
    rugs = pd.read_parquet(resolve(cfg["paths"]["processed"]) / "rugs.parquet",
                           columns=["pair", "rug_type"])
    merged = pairs.merge(rugs, on="pair", how="left")
    groups = np.where(merged.rug_type.eq("burn"), "rug_burn",
                      np.where(merged.rug_type.eq("swap"), "rug_swap", "no_detected_rug"))
    return pd.Series(groups).value_counts().astype(int).to_dict()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/common.yaml")
    ap.add_argument("--allow-incomplete", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    results = Path(resolve(cfg["paths"]["results"]))
    master = pd.read_csv(results / "label_audit_sample.csv", dtype=str).fillna("")
    r1 = pd.read_csv(results / "label_audit_reviewer_1.csv", dtype=str).fillna("")
    r2 = pd.read_csv(results / "label_audit_reviewer_2.csv", dtype=str).fillna("")
    report, strata, disagreements = build_report(master, r1, r2, _population_counts(cfg))
    atomic_json(report, results / "label_audit_report.json", indent=2)
    atomic_csv(strata, results / "label_audit_metrics.csv", index=False)
    atomic_csv(disagreements, results / "label_audit_disagreements.csv", index=False)
    print(pd.Series(report).to_string())
    if report["double_reviewed_n"] < report["sample_n"] and not a.allow_incomplete:
        raise SystemExit("Audit is incomplete; use --allow-incomplete only for a progress report.")


if __name__ == "__main__":
    main()
