"""Populate one blinded audit sheet from an explicit on-chain expert rubric.

This is an AI-assisted computational review, not a substitute for a second independent human rater.
It reconstructs action-linked WETH reserve collapses directly from raw logs and never reads the
automatic audit group or stored rug label.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import atomic_csv
from src.config import load_config, resolve
from src.decode import decode, to_unix


def _event_evidence(pool: pd.DataFrame, first_mint: pd.Series) -> tuple[str, str, str]:
    pool = pool.sort_values(["ts", "block_number", "log_index"], kind="stable")
    sync = pool[pool.event.eq("sync")][["ts", "tx", "log_index", "weth"]].copy()
    sync["prev"] = sync.weth.shift(1).fillna(0.0)
    actions = pool[~pool.event.eq("sync")][["tx", "log_index", "event", "tok_in"]].copy()
    actions["log_index"] -= 1
    sync = sync.merge(actions.rename(columns={"event": "action"}),
                      on=["tx", "log_index"], how="left")
    sync["action"] = sync.action.fillna("sync")
    sync["drop"] = np.where(sync.prev > 0, 1 - sync.weth / sync.prev, np.nan)
    actionable = sync[sync.action.isin(["burn", "swap"]) & (sync.prev >= 0.1)].copy()
    if actionable.empty:
        return "not_rug", "medium", "No Burn/Swap-linked WETH collapse from >=0.1 ETH in collected horizon."

    best = actionable.sort_values("drop", ascending=False).iloc[0]
    init_tok = float(first_mint.tok) if pd.notna(first_mint.tok) else np.nan
    swap_factor = (float(best.tok_in) / init_tok
                   if best.action == "swap" and pd.notna(best.tok_in) and init_tok > 0 else np.nan)
    qualifies = best["drop"] >= 0.99 and (best.action == "burn" or swap_factor > 5)
    tx = str(best.tx)
    detail = (f"max action-linked drop={best['drop']:.4%}; action={best.action}; "
              f"WETH {best.prev:.6g}->{best.weth:.6g}; tx={tx}")
    if best.action == "swap" and pd.notna(swap_factor):
        detail += f"; token-in/init={swap_factor:.3g}x"
    if qualifies:
        return "rug", "high", detail
    if best["drop"] >= 0.95:
        return "uncertain", "low", detail + "; near-threshold collapse requires contextual adjudication."
    return "not_rug", "high", detail + "; below the prespecified 99% operational threshold."


def review(sheet: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    pairs = pd.read_parquet(raw_dir / "pairs.parquet")
    for c in ["created_ts", "token_deployed_ts"]:
        pairs[c] = to_unix(pairs[c])
    wanted = set(sheet.pair.str.lower())
    meta = pairs[pairs.pair.str.lower().isin(wanted)].copy()
    if len(meta) != len(wanted):
        raise ValueError(f"missing metadata for {len(wanted) - len(meta)} audited pools")

    chunks = []
    columns = ["address", "role", "ts", "block_number", "transaction_hash", "log_index",
               "event", "addr1", "addr2", "v0", "v1", "v2", "v3"]
    for path in sorted((raw_dir / "events").glob("*.parquet")):
        part = pd.read_parquet(path, columns=columns)
        part = part[part.role.eq("pool") & part.address.str.lower().isin(wanted)]
        if len(part):
            chunks.append(part)
    if not chunks:
        raise ValueError("no raw pool events found for audit sample")
    pool_ev = decode(pd.concat(chunks, ignore_index=True), meta)["pool_ev"]

    evidence = {}
    for pair, g in pool_ev.groupby("pair", sort=False):
        mints = g[g.event.eq("mint")].sort_values(["ts", "block_number", "log_index"])
        evidence[pair] = (("uncertain", "low", "No Mint event; operational label cannot be verified.")
                          if mints.empty else _event_evidence(g, mints.iloc[0]))

    out = sheet.copy()
    for idx, row in out.iterrows():
        label, confidence, notes = evidence.get(
            row.pair, ("uncertain", "low", "No decoded event evidence found."))
        out.at[idx, "reviewer_label"] = label
        out.at[idx, "confidence"] = confidence
        out.at[idx, "evidence_notes"] = "AI-assisted on-chain review: " + notes
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/common.yaml")
    ap.add_argument("--reviewer", type=int, choices=[1, 2], default=1)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    results = Path(resolve(cfg["paths"]["results"]))
    path = results / f"label_audit_reviewer_{a.reviewer}.csv"
    sheet = pd.read_csv(path, dtype=str).fillna("")
    if sheet.reviewer_label.str.strip().ne("").any() and not a.overwrite:
        raise SystemExit(f"{path} already contains labels; pass --overwrite to replace them")
    out = review(sheet, Path(resolve(cfg["paths"]["raw"])))
    atomic_csv(out, path, index=False)
    print(out.reviewer_label.value_counts().to_string())
    print("Provenance: AI-assisted expert rubric; do not describe this as an independent human review.")


if __name__ == "__main__":
    main()
