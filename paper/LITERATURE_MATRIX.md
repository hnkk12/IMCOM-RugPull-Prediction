# Literature matrix and novelty boundary

Last verified: 2026-10-08. Peer-reviewed work is separated from preprints. Dataset sizes and metrics
below are useful for positioning, not for direct performance comparison because labels, chains,
cohorts and observation windows differ.

## Core peer-reviewed literature

| Work | Scope / evidence | Prediction design | Relationship to this project |
|---|---|---|---|
| Xia et al. (2021), [Trade or Trick?](https://doi.org/10.1145/3491051) | Uniswap V2 transactions; scam characterization and detection | Association heuristics plus ML over transaction/contract features | Foundational Uniswap scam study; does not establish this project's fixed absolute landmarks, censoring and purged temporal comparison. |
| Mazorra et al. (2022), [Do Not Rug on Me](https://doi.org/10.3390/math10060949) | 27,588 Uniswap V2 tokens; heavily imbalanced labels | ML scenarios at arbitrary blocks and fixed elapsed time | Closest early tabular predecessor; motivates careful reconstruction of what information is available at decision time. |
| Nguyen et al. (2023), [Rug-pull malicious token detection](https://doi.org/10.1145/3579375.3579385) | Transaction and engineered token features | Supervised feature selection/classification | Relevant tabular baseline; not evidence that the BOBAI GitHub artifact has an associated paper. |
| Igarashi and Matsuura (2024), [Static analysis before deployment](https://doi.org/10.1007/978-3-031-69231-4_13) | Bytecode/opcode features | Pre-deployment static detection | Complementary extreme-early setting; motivates the contract-feature ablation rather than replacing behavioral landmarks. |
| Qiao et al. (2024), [Feature-fused graph classification](https://doi.org/10.1007/978-981-97-3203-6_4) | Transaction graph plus manual/temporal features | Graph classification | Stronger representation family; our contribution is evaluation validity and operational coverage, not graph-model novelty. |
| Wu et al. (2024), [TokenScout](https://doi.org/10.1145/3658644.3690234) | 214,084 ERC-20 tokens and a 2023 deployment | Temporal GNN for several scam types | State-of-the-art graph/deployment reference. It makes a generic “first early detection” claim untenable for this paper. |
| Huynh et al. (2025), [Serial Scammers and Attack of the Clones](https://doi.org/10.1145/3696410.3714919) | Uniswap/PancakeSwap scammer networks and contract clones | Cluster and flow analysis | Supports creator/template cold-start tests and warns that random splits may share scam families. |
| Srifa et al. (2025), [Rug pull detection using transaction data](https://doi.org/10.1016/j.bcra.2025.100275) | 7,450 Uniswap V3 tokens/pools; time-window features | Multiple equal-duration observation windows | The closest windowed study. Our distinct focus is Uniswap V2, 1/6/24-hour landmarks, label censoring, purged temporal splits, common cohorts and end-to-end coverage. |
| Kalacheva et al. (2026), [The rise of meme coins](https://doi.org/10.1016/j.bcra.2025.100336) | Uniswap V2/V3 and five-minute early-trading features | Early ML plus economic/trading analysis | Demonstrates that “short-window early prediction” is no longer novel alone; the defensible novelty is protocol auditing and the coverage frontier. |

## Recent preprints to cite as concurrent work

| Work | Key overlap | Safe distinction |
|---|---|---|
| Shoaei et al. (2026), [TM-RUGPULL](https://arxiv.org/abs/2602.21529) | Explicit leakage-resistant, multimodal dataset with expert labels | Uses the first half of each project lifespan; ours uses fixed absolute decision landmarks and quantifies already-rugged exclusions. |
| Shoaei et al. (2026), [LROO Rug Pull Detector](https://arxiv.org/abs/2603.11324) | Leakage-aware on-chain + OSINT prediction and calibration | Smaller hand-labeled multimodal cohort; ours is a 23,400-pool on-chain measurement study with temporal/common-cohort/coverage diagnostics. |
| Cao et al. (2026), [BSC wash-trading warning](https://arxiv.org/abs/2603.13830) | Early warning, PR-AUC and lead time | BSC meme tokens and wash-trading features; different chain, evidence and scale. |
| Li et al. (2026), [Catching the Rug](https://arxiv.org/abs/2608.20271) | Five-minute early prediction on 6.4M Solana tokens | Solana launchpads and cross-platform generalization; cite as contemporary cross-chain evidence, not a direct benchmark. |

## Defensible research gap

The safe gap is not “no prior work detects rug pulls early” and not “the first leakage-resistant
dataset.” Prior work already covers time windows, temporal graphs, static pre-deployment features and,
in 2026 preprints, explicit leakage resistance. The defensible gap is a controlled measurement of how
evaluation choices change conclusions on a single Uniswap V2 cohort:

1. label-dependent/full-lifecycle versus fixed landmark windows;
2. random versus purged chronological splits on the same task;
3. landmark-specific versus common-cohort comparisons;
4. classifier metrics versus end-to-end achievable coverage after early-rug exclusion;
5. creator/template cold starts, rolling temporal folds, label sensitivity and feature-accounting checks.

The main claim should therefore be: **apparently strong early-warning performance can coexist with
substantially lower deployment coverage, and protocol choices must be reported as part of the result.**

## Baseline provenance

Setup A comes from the public software artifact
[`kangmyoungseok/RugPull-Prediction-AI`](https://github.com/kangmyoungseok/RugPull-Prediction-AI).
The repository identifies a KITRI BoB 10th project but supplies no paper, DOI or license. Searches of
the official [IMCOM 2022 program](https://imcom.org/2022/img/IMCOM2022_ConferenceProgram.pdf) and
[proceedings contents](https://www.proceedings.com/content/062/062555webtoc.pdf) found no matching rug-pull paper. Cite the
repository as software only unless its authors provide a verified publication.

