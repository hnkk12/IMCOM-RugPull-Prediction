"""Sensitivity audit for rug thresholds, cumulative detection and dust cutoffs."""
from __future__ import annotations

import argparse
import time

import pandas as pd

from src.artifacts import atomic_csv
from src.config import load_config, resolve
from src.decode import load_decoded, load_metadata
from src.label import assign_labels, detect_rugs
from src.split import _ts


VARIANTS = [
    ("single_95", 0.95, False),
    ("single_975", 0.975, False),
    ("single_99", 0.99, False),
    ("cumulative_99", 0.99, True),
]


def run(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = resolve(cfg["paths"]["raw"])
    results = resolve(cfg["paths"]["results"])
    pairs, _ = load_metadata(raw)
    tables = load_decoded(raw, pairs)
    L, S = cfg["label"], cfg["study"]
    rug_sets, rows = {}, []
    cutoff = _ts(S["data_cutoff"]) if S.get("data_cutoff") else int(time.time())
    for name, threshold, cumulative in VARIANTS:
        rugs = detect_rugs(tables["pool_ev"], threshold, L["mev_factor"],
                           L["min_pre_reserve_eth"], cumulative, L["cum_window_h"])
        rug_sets[name] = set(rugs.loc[rugs.rug_ts.notna(), "pair"])
        for min_weth in (0.1, 0.5, 1.0):
            for n in cfg["windows_hours"]:
                lab = assign_labels(rugs, pairs, n, L["H_days"], S["pool_horizon_days"],
                                    cutoff, min_weth)
                counts = lab.status.value_counts()
                rows.append({"variant": name, "drop_threshold": threshold,
                             "cumulative": cumulative, "min_init_weth": min_weth,
                             "N_hours": n, **{f"n_{k}": int(v) for k, v in counts.items()}})
    ref = rug_sets["single_99"]
    agreement = []
    for name, values in rug_sets.items():
        union = ref | values
        agreement.append({"variant": name, "n_rugs": len(values),
                          "added_vs_single_99": len(values - ref),
                          "missing_vs_single_99": len(ref - values),
                          "jaccard_vs_single_99": len(ref & values) / len(union) if union else 1.0})
    detail, agree = pd.DataFrame(rows), pd.DataFrame(agreement)
    atomic_csv(detail, results / "label_sensitivity.csv", index=False)
    atomic_csv(agree, results / "label_sensitivity_agreement.csv", index=False)
    return detail, agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/common.yaml")
    a = ap.parse_args()
    _, agreement = run(load_config(a.config))
    print(agreement.to_string(index=False))


if __name__ == "__main__":
    main()
