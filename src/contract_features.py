"""Static contract features for setup D, from BigQuery `contracts.function_sighashes`.

These are known before trading starts, so they are leak-free for any N.
An optional Slither pass on verified source can be added later; it needs the
Etherscan API (ETHERSCAN_API_KEY) and only covers verified contracts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from Crypto.Hash import keccak

from src.constants import RISKY_FUNCTIONS


def selector(signature: str) -> str:
    k = keccak.new(digest_bits=256)
    k.update(signature.encode())
    return "0x" + k.hexdigest()[:8]


SELECTORS = {flag: {selector(s) for s in sigs} for flag, sigs in RISKY_FUNCTIONS.items()}
CONTRACT_FEATURES = list(SELECTORS) + ["n_functions", "bytecode_size", "bytecode_dup_count",
                                      "bytecode_missing"]


def contract_features(contracts: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """One row per token. bytecode_dup_count = earlier tokens (by pair creation) with identical bytecode,
    i.e. how often this exact contract template was already deployed."""
    rows = []
    for token, sigs, size in zip(contracts.token, contracts.function_sighashes, contracts.bytecode_size):
        s = set(sigs) if sigs is not None else set()
        missing = bool(pd.isna(size) or size <= 0)
        r = {"token": token, "n_functions": len(s), "bytecode_size": size,
             "bytecode_missing": int(missing)}
        r.update({flag: int(bool(s & sel)) for flag, sel in SELECTORS.items()})
        rows.append(r)
    base_cols = ["token"] + [c for c in CONTRACT_FEATURES if c != "bytecode_dup_count"]
    out = pd.DataFrame(rows, columns=base_cols)

    if "bytecode_hash" in contracts:
        order = (pairs.groupby("token").created_ts.min().rename("first_ts").reset_index()
                 .merge(contracts[["token", "bytecode_hash", "bytecode_size"]], on="token")
                 .sort_values("first_ts"))
        valid = order.bytecode_hash.notna() & order.bytecode_size.fillna(0).gt(0)
        order["bytecode_dup_count"] = np.nan
        order.loc[valid, "bytecode_dup_count"] = order.loc[valid].groupby("bytecode_hash").cumcount()
        out = out.merge(order[["token", "bytecode_hash", "bytecode_dup_count"]], on="token", how="left")
    else:
        out["bytecode_hash"] = None
        out["bytecode_dup_count"] = np.nan
    if "code_at_prediction_time" in contracts:
        out = out.merge(contracts[["token", "code_at_prediction_time"]], on="token", how="left")
    return out
