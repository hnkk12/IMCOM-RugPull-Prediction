"""Turn raw `events` rows (address, role, event, addr1, addr2, v0..v3) into tidy tables.

pool_ev  : Sync/Mint/Burn/Swap of each pair, oriented to (WETH, token) and WETH in ETH units
lp_tr    : LP-token Transfer events of each pair
tok_tr   : token Transfer events
own_ev   : token OwnershipTransferred events
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WEI = 1e18
ORDER = ["ts", "block_number", "log_index"]


def to_unix(col: pd.Series) -> pd.Series:
    """Datetime column -> int seconds, independent of the stored resolution (ns/us/s)."""
    dt = pd.to_datetime(col, utc=True)
    return ((dt - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).astype("int64")


def load_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_parquet(raw_dir / "pairs.parquet")
    for c in ["created_ts", "token_deployed_ts"]:
        pairs[c] = to_unix(pairs[c])
    files = sorted((raw_dir / "events").glob("*.parquet"))
    events = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    cpath = raw_dir / "contracts.parquet"
    contracts = pd.read_parquet(cpath) if cpath.exists() else pd.DataFrame(columns=["token"])
    return pairs, events, contracts


def load_metadata(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the small pair/contract tables without materializing all raw events."""
    pairs = pd.read_parquet(raw_dir / "pairs.parquet")
    for c in ["created_ts", "token_deployed_ts"]:
        pairs[c] = to_unix(pairs[c])
    cpath = raw_dir / "contracts.parquet"
    contracts = pd.read_parquet(cpath) if cpath.exists() else pd.DataFrame(columns=["token"])
    return pairs, contracts


def load_decoded(raw_dir: Path, pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Decode monthly parquet files incrementally to keep peak memory bounded."""
    chunks: dict[str, list[pd.DataFrame]] = {k: [] for k in ("pool_ev", "lp_tr", "tok_tr", "own_ev")}
    files = sorted((raw_dir / "events").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no event parquet files under {raw_dir / 'events'}")
    for f in files:
        part = decode(pd.read_parquet(f), pairs)
        for key in chunks:
            chunks[key].append(part[key])
    out = {}
    for key, frames in chunks.items():
        out[key] = pd.concat(frames, ignore_index=True)
        frames.clear()
    return out


def decode(events: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # Sorting happens once, in-place, when feature indexes are built. Avoiding a
    # whole-table sort here materially lowers peak RAM for the 25M-event dataset.
    ev = events

    pool = ev[ev.role == "pool"].rename(columns={"address": "pair"}, copy=False)
    pool = pool.merge(pairs[["pair", "token_is_token0"]], on="pair", how="inner")
    t0 = pool.token_is_token0.to_numpy()
    # WETH is token1 when the token is token0
    w_idx, k_idx = np.where(t0, 1, 0), np.where(t0, 0, 1)
    v = pool[["v0", "v1", "v2", "v3"]].to_numpy(dtype="float64")
    rows = np.arange(len(pool))
    is_swap = (pool.event == "swap").to_numpy()
    out = pd.DataFrame({
        "pair": pool.pair.to_numpy(),
        "ts": pool.ts.to_numpy("int64"),
        "block_number": pool.block_number.to_numpy("int64"),
        "log_index": pool.log_index.to_numpy("int64"),
        "tx": pool.transaction_hash.to_numpy(),
        "event": pool.event.to_numpy(),
        # sync: reserves, mint/burn: amounts (both laid out as v0=token0, v1=token1)
        "weth": np.where(is_swap, np.nan, v[rows, w_idx] / WEI),
        "tok": np.where(is_swap, np.nan, v[rows, k_idx]),
        # swap: v0=amount0In v1=amount1In v2=amount0Out v3=amount1Out
        "weth_in": np.where(is_swap, v[rows, w_idx] / WEI, np.nan),
        "tok_in": np.where(is_swap, v[rows, k_idx], np.nan),
        "weth_out": np.where(is_swap, v[rows, w_idx + 2] / WEI, np.nan),
        "tok_out": np.where(is_swap, v[rows, k_idx + 2], np.nan),
        "sender": pool.addr1.to_numpy(),
        "to": pool.addr2.to_numpy(),
    })

    tr_cols = {"addr1": "frm", "addr2": "to", "v0": "value"}
    keep = ["ts", "block_number", "log_index", "frm", "to", "value"]
    lp = ev[(ev.role == "lp") & (ev.event == "transfer")].rename(columns={"address": "pair", **tr_cols})
    tok = ev[(ev.role == "token") & (ev.event == "transfer")].rename(columns={"address": "token", **tr_cols})
    own = ev[(ev.role == "token") & (ev.event == "ownership")].rename(
        columns={"address": "token", "addr1": "prev_owner", "addr2": "new_owner"})
    return {
        "pool_ev": out,
        "lp_tr": lp[["pair"] + keep].reset_index(drop=True),
        "tok_tr": tok[["token"] + keep].reset_index(drop=True),
        "own_ev": own[["token", "ts", "block_number", "log_index", "prev_owner", "new_owner"]].reset_index(drop=True),
    }
