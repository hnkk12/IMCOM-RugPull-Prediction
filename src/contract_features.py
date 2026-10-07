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
CONTRACT_FEATURES = list(SELECTORS) + ["n_functions", "bytecode_size", "bytecode_dup_count"]


def contract_features(contracts: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """One row per token. bytecode_dup_count = earlier tokens (by pair creation) with identical bytecode,
    i.e. how often this exact contract template was already deployed."""
    rows = []
    for token, sigs, size in zip(contracts.token, contracts.function_sighashes, contracts.bytecode_size):
        s = set(sigs) if sigs is not None else set()
        r = {"token": token, "n_functions": len(s), "bytecode_size": size}
        r.update({flag: int(bool(s & sel)) for flag, sel in SELECTORS.items()})
        rows.append(r)
    out = pd.DataFrame(rows, columns=["token"] + CONTRACT_FEATURES[:-1])

    if "bytecode_hash" in contracts:
        order = (pairs.groupby("token").created_ts.min().rename("first_ts").reset_index()
                 .merge(contracts[["token", "bytecode_hash"]], on="token").sort_values("first_ts"))
        order["bytecode_dup_count"] = order.groupby("bytecode_hash").cumcount()
        out = out.merge(order[["token", "bytecode_dup_count"]], on="token", how="left")
    else:
        out["bytecode_dup_count"] = np.nan
    return out
