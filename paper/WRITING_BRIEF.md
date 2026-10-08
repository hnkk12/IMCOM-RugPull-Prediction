# Paper writing brief

## Status and positioning

The evidence package is ready for drafting. The manuscript is not yet submission-ready if it claims
two independent human raters: reviewer 2 is intentionally blank. The current audit is explicitly an
AI-assisted expert on-chain review.

Provisional title:

> **Leakage-Resistant Landmark Evaluation for Early Rug-Pull Detection on Uniswap V2**

Primary target: **Blockchain: Research and Applications**. Frame the work as a measurement and
evaluation study, not as a new classifier or a universal state-of-the-art claim.

## Central thesis

Strong discrimination scores do not by themselves establish an effective early-warning system.
Label-dependent observation windows and random splits produce optimistic estimates, while later
landmarks remove many already-rugged pools from the eligible cohort. A valid evaluation must report
the decision time, censoring, temporal split, risk-set construction and end-to-end coverage together.

## Research questions

1. How much optimism is introduced by label-dependent/full-lifecycle feature windows?
2. Under a fixed landmark, how much does random splitting overstate chronological generalization?
3. Do expanded on-chain and contract features improve on a clean reconstruction of repository features?
4. How do 1/6/24-hour landmarks trade information gain against eligible-cohort and operational coverage?

## Dataset and protocol facts

- 23,400 sampled new-token WETH pools, October 2020–December 2023.
- 24,886,580 decoded raw events; token deployed at most seven days before pool creation.
- Operational rug-like collapse: first Burn, or filtered Swap, removing at least 99% of WETH reserve.
- The final build contains 18,161 detected events (16,374 Burn; 1,787 Swap). A `1e-12` ratio tolerance
  includes exact 99% boundary events that binary floating-point had previously excluded.
- Decision landmarks N = 1, 6 and 24 hours; 30-day target horizon; 45-day observation support.
- Primary model: LightGBM with 47 robust early on-chain features; token-accounting-dependent holder
  features are excluded because 29.7–37.1% of rows fail ledger consistency.
- Primary split: chronological train/validation/test with label-horizon purge.

## Results to report

For Setup C, test ROC-AUC (PR-AUC) is:

- N=1h: **0.848 (0.860)**, 95% CI PR-AUC 0.850–0.869, test prevalence 50.7%.
- N=6h: **0.830 (0.796)**, 95% CI PR-AUC 0.781–0.812, test prevalence 40.8%.
- N=24h: **0.791 (0.653)**, 95% CI PR-AUC 0.630–0.677, test prevalence 35.1%.

The fixed-window random Setup B reaches PR-AUC 0.950/0.928/0.858. A label-dependent-window probe on
the same dataset reaches temporal PR-AUC 0.973/0.990/0.982, isolating observation-window leakage.

On a common 8,891-pool cohort, PR-AUC rises from 0.531 to 0.599 to 0.653 with longer observation,
showing that additional information helps when the population is held fixed. In contrast, maximum
achievable operational coverage falls from 37.5% to 24.3% to 17.3%; realized end-to-end coverage is
30.6%, 16.7% and 12.8%.

Expanded robust features outperform the 16 reconstructed repository features by 0.037/0.061/0.086
PR-AUC at 1/6/24h. Contract features add only 0.009/0.001/0.006 PR-AUC and should be described as a
small, inconsistent gain. Rolling folds show temporal drift, especially for later N=24h folds.

The label audit's first pass reviewed 300/300 pools and estimates population-weighted positive-rule
precision at 95.98% (4.02% false positives), excluding uncertain judgments. It contains 39 uncertain
cases and no confirmed missed event in the 100-pool no-detected stratum. State its AI-assisted
provenance and do not report Cohen's κ until reviewer 2 is independently completed.

## Contribution statements

1. A fixed-landmark, censored and purged chronological evaluation protocol for Uniswap V2 early warning.
2. A controlled same-dataset demonstration of label-dependent-window and random-split optimism.
3. A multi-landmark analysis separating information gain from changing risk-set composition.
4. End-to-end coverage, rolling temporal, common-cohort, cold-start, feature and label robustness evidence.
5. A reproducible artifact package with saved predictions/models, confidence intervals, figures and hashes.

## Manuscript structure

1. Introduction: deployment question, evaluation failure modes, contributions.
2. Related work: characterization; tabular/time-window methods; static/graph methods; concurrent
   leakage-resistant and cross-chain work.
3. Problem formulation: t0, landmark, horizon, censoring, operational label and estimands.
4. Data and cohort: source, sampling, filtering, event decoding, label audit and ethics.
5. Features and models: repo reconstruction, robust extensions, contract ablation and calibration.
6. Evaluation: A/B/C protocols, purge, metrics, confidence intervals and robustness designs.
7. Results: leakage; main performance; common cohort; operational coverage; ablations/sensitivity.
8. Discussion: performance-versus-coverage paradox, drift, intended use and limits of on-chain intent inference.
9. Threats to validity and conclusion.

## Non-negotiable limitations

- The sample is conditional and its rug prevalence is not Uniswap-wide prevalence.
- The label detects rug-like liquidity collapse, not malicious intent or legal fraud.
- Honeypots, soft rugs and activity beyond the collection horizon are not fully captured.
- Contract selectors and creator identities are approximate; locker coverage is incomplete.
- Direct metric comparisons with other papers are invalid without harmonized labels/cohorts.
- Setup A's source is an unlicensed software artifact with no verified associated publication.

## Artifact map

- Main values: `results/paper_artifacts/table_main_lgbm.csv`
- Protocol/leakage: `results/leakage_probe.csv` and `table_protocol_comparison.csv`
- Operational analysis: `table_operational_coverage.csv` and `figure_operational_coverage.png`
- Robustness: rolling, common-cohort, cold-start, label-sensitivity and data-quality tables under
  `results/paper_artifacts/`
- Audit: `results/label_audit_report.json`, `label_audit_metrics.csv` and reviewer sheets
- Integrity: `results/manifest.json`, `data/raw/manifest.json`, `data/processed/manifest.json`
