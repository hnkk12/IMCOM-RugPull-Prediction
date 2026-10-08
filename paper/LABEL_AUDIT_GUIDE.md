# Label-audit protocol

## Purpose and claim boundary

The audit checks whether the automated rule correctly identifies an **on-chain rug-like liquidity
collapse** within the collected horizon. On-chain logs alone generally cannot establish malicious
intent, identity, or fraud in the legal sense. The paper must preserve this distinction.

The master sample contains 300 pools, stratified by calendar year and automated group: 100 burn,
100 swap and 100 no-detected-rug pools. Reviewers use the separately shuffled files
`results/label_audit_reviewer_1.csv` and `results/label_audit_reviewer_2.csv`. They must not open
`label_audit_sample.csv`, model predictions, or each other's sheet until both reviews are frozen.

## Allowed labels

- `rug`: clear action-linked removal of at least 99% of WETH liquidity from a reserve of practical
  size, by Burn or by a non-MEV-like Swap.
- `not_rug`: no qualifying collapse in the 45-day collected horizon, or the largest observed
  action-linked collapse is clearly below the operational threshold.
- `uncertain`: borderline threshold, tiny pre-event reserve, incomplete/ambiguous action linkage,
  liquidity migration, or evidence that requires context unavailable from the logs.

Record `confidence` as `high`, `medium`, or `low`. In `evidence_notes`, include the decisive
transaction, action, WETH reserve before/after, percentage drop, and the reason for uncertainty.

## Review and adjudication

1. Each reviewer completes their own sheet independently.
2. Run `python -m src.label_audit_report --allow-incomplete` only to check progress.
3. After both sheets are frozen, run the report and inspect `results/label_audit_disagreements.csv`.
4. Resolve disagreements without changing the two original labels. Enter only the final value in
   `adjudicated_label` and rationale in `adjudication_notes` of `label_audit_sample.csv`.
5. Run `python -m src.label_audit_report` without `--allow-incomplete` and report raw agreement,
   three-class Cohen's κ, unresolved count, and population-weighted rule precision/false-positive rate.

Because sampling is balanced across strata rather than proportional to the dataset, never report
the raw pooled 300-row rate as a population estimate. The reporting script weights burn and swap
strata by their population sizes. The rug rate in `no_detected_rug` is a missed-event diagnostic,
not a conventional classifier false-negative rate.

## Current expert pass

`src.expert_label_audit` can fill one sheet from a prespecified on-chain rubric without reading the
automatic group. Its provenance is **AI-assisted expert review**. It must not be described as an
independent human rater and cannot support a human inter-rater reliability claim by itself.

