# Paper readiness checklist

## Recommended framing

**Leakage-Resistant Early Rug-Pull Detection on Uniswap V2: Temporal Evaluation and the Cost of Delayed Observation**

The primary contribution is the evaluation protocol, not a claim that a new classifier is universally best.
Use neutral wording such as "optimistically biased under a non-deployment evaluation" rather than calling prior
results fake. Setup A is a replication on the prior dataset; the same-dataset leakage probe is the cleaner causal
evidence for label-dependent observation-window leakage.

## Research questions

1. How strongly does a label-dependent observation window inflate reported discrimination?
2. How much optimism remains under a fixed window when pools are randomly rather than temporally split?
3. Do the new early on-chain features add value beyond the windowed baseline features?
4. How does the 1/6/24-hour landmark trade additional information against rugs that have already happened?

## Completed computational evidence

- [x] Reproduce prior Setup A and run direct lifecycle/window leakage probes.
- [x] Same-dataset label-dependent-window leakage probe under random and temporal splits.
- [x] Fixed-window clean Setup B and purged temporal clean Setup C.
- [x] New-feature and contract-feature ablations.
- [x] Common-cohort comparison across N=1/6/24 hours.
- [x] Four expanding-window temporal folds.
- [x] Creator and bytecode-template cold-start checks.
- [x] Stratified-bootstrap confidence intervals, confusion counts, Brier score and ECE.
- [x] Reliability curves, saved models/configs/predictions and SHA-256 manifests.
- [x] Automated duplicate, missingness, finite-value and bounded-feature quality report.
- [x] Robustness ablation excluding token-accounting-dependent holder shares.
- [x] Label sensitivity for 95%, 97.5%, 99% and cumulative-99% rules.
- [x] End-to-end operational coverage including early rugs.
- [x] Twenty-three automated tests, including threshold-boundary and audit-report edge cases.

## Writing status

The project is ready to draft. Use `paper/WRITING_BRIEF.md`, `paper/LITERATURE_MATRIX.md`,
`paper/VENUE_AND_CLAIMS.md` and `paper/references.bib`. Results and figures were regenerated after the
99% floating-point boundary fix; do not reuse numbers from earlier notes.

## Human/external evidence still required before submission

- [x] One AI-assisted expert pass completed on all 300 blinded samples: population-weighted positive-label
      precision 95.98%, false-positive rate 4.02%; 39 uncertain cases. Do not call this a human review.
- [ ] A second independent human reviewer completes `results/label_audit_reviewer_2.csv` if the final paper
      will claim human validation or inter-rater reliability; then adjudicate and report Cohen's κ.
- [ ] If feasible, compare the audit sample with TokenSniffer or another independent source.
- [ ] Verify and extend LP-locker addresses; rerun only if coverage changes materially.
- [x] Literature matrix covers characterization, tabular detection, static/graph methods and 2026 concurrent work.
- [x] Baseline provenance resolved: software artifact only; no verified paper/DOI or IMCOM relationship.
- [x] Primary target selected provisionally: Blockchain: Research and Applications; re-check its live Guide
      for Authors and Scopus quartile at submission.

## Required paper figures/tables

- [x] Cohort construction and label attrition: `results/paper_artifacts/figure_cohort_construction.png`.
- [x] Landmark/horizon/censoring/purge timeline: `results/paper_artifacts/figure_landmark_timeline.png`.
- [x] Setup A/B/C comparison: `results/paper_artifacts/table_protocol_comparison.csv`.
- [x] Main performance with 95% CI and temporal-test PR baselines: `figure_main_performance.png`.
- [x] Common-cohort versus landmark results: `figure_landmark_vs_common.png`.
- [x] Maximum achievable and realized operational coverage: `figure_operational_coverage.png`.
- [x] Label sensitivity, rolling temporal, cold-start and data-quality tables under `results/paper_artifacts/`.
- [x] SHAP summaries for the clean primary model under `results/setup_C_no_token_accounting/`.

## Reproduction commands

```powershell
python -m pytest
python -m src.build_dataset --jobs 8
python -m src.train --config experiments/setup_B_no_token_accounting.yaml
python -m src.train --config experiments/setup_C_no_token_accounting.yaml
python -m src.train --config experiments/setup_C_repo_only.yaml
python -m src.train --config experiments/setup_D_no_token_accounting.yaml
python -m src.train --config experiments/setup_C_common.yaml
python -m src.train --config experiments/setup_C_rolling.yaml
python -m src.train --config experiments/setup_D_coldstart.yaml
python -m src.train --config experiments/setup_D_template_coldstart.yaml
python -m src.leakage_probe
python -m src.operational_eval
python -m src.label_sensitivity
python -m src.build_dataset --config experiments/setup_C_label95.yaml --jobs 8
python -m src.train --config experiments/setup_C_label95.yaml
python -m src.build_dataset --config experiments/setup_C_cumulative99.yaml --jobs 8
python -m src.train --config experiments/setup_C_cumulative99.yaml
python -m src.sample_label_audit
python -m src.expert_label_audit --reviewer 1
python -m src.label_audit_report --allow-incomplete
python -m src.data_quality
python -m src.paper_artifacts
python -m src.reproducibility
```

Never distribute `.env` or the unlicensed baseline directory as supplementary material.
