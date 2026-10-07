"""Rug-pull labeling on crawled events + censoring for early detection.

Rule (kept from the baseline repo, re-implemented on Sync reserves):
  a pool is rugged at the first Burn or Swap that removes >= `drop_threshold` (99%)
  of the WETH reserve in a single event. Swap-driven drops count only when the
  token amount swapped in exceeds `mev_factor` x the initial token liquidity
  (the baseline's filter against MEV/sandwich false positives).
Optional ablation `cumulative`: reserve falls below (1 - drop_threshold) of its
running max within `cum_window_h` hours (catches rugs split over several txs).

Uniswap V2 emits Sync right before Mint/Burn/Swap inside the same call, so the
event at (tx, log_index + 1) tells which action produced each Sync.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOUR, DAY = 3600, 86400


def detect_rugs(pool_ev: pd.DataFrame, drop_threshold: float = 0.99, mev_factor: float = 5.0,
                min_pre_reserve_eth: float = 0.0, cumulative: bool = False,
                cum_window_h: float = 1.0) -> pd.DataFrame:
    """One row per pair: t0 (first liquidity add), init_weth, init_tok, rug_ts, rug_type, rug_drop."""
    ev = pool_ev.sort_values(["pair", "ts", "block_number", "log_index"], kind="stable")

    mints = ev[ev.event == "mint"]
    first = mints.groupby("pair", sort=False).agg(t0=("ts", "first"), init_tok=("tok", "first"),
                                                  init_weth_mint=("weth", "first"))

    sync = ev[ev.event == "sync"][["pair", "ts", "tx", "log_index", "weth"]].copy()
    sync["prev"] = sync.groupby("pair", sort=False).weth.shift(1).fillna(0.0)
    act = ev[ev.event != "sync"][["tx", "log_index", "event", "tok_in"]].copy()
    act["log_index"] -= 1
    sync = sync.merge(act.rename(columns={"event": "action"}), on=["tx", "log_index"], how="left")
    sync["action"] = sync.action.fillna("sync")

    # reserve right after the first mint = initial liquidity
    sync = sync.merge(first[["t0", "init_tok"]], left_on="pair", right_index=True, how="left")
    init_res = (sync[sync.action == "mint"].groupby("pair", sort=False).weth.first().rename("init_weth"))

    drop_ok = (sync.prev > max(min_pre_reserve_eth, 0.0)) & (sync.weth <= (1 - drop_threshold) * sync.prev)
    not_mev = (sync.action != "swap") | (sync.tok_in > mev_factor * sync.init_tok)
    cand = sync[drop_ok & sync.action.isin(["burn", "swap"]) & not_mev]
    rug = cand.groupby("pair", sort=False).first()[["ts", "action", "weth", "prev"]]
    rug = pd.DataFrame({"rug_ts": rug.ts, "rug_type": rug.action,
                        "rug_drop": 1 - rug.weth / rug.prev})

    if cumulative:
        s = sync.set_index(pd.to_datetime(sync.ts, unit="s"))
        rmax = (s.groupby("pair", sort=False).weth.rolling(f"{int(cum_window_h * 3600)}s").max()
                .reset_index(level=0, drop=True).to_numpy())
        hit = (rmax > max(min_pre_reserve_eth, 0.0)) & (s.weth.to_numpy() <= (1 - drop_threshold) * rmax)
        c = sync[hit].groupby("pair", sort=False).first()
        c = pd.DataFrame({"rug_ts": c.ts, "rug_type": "cumulative", "rug_drop": np.nan})
        both = pd.concat([rug, c]).reset_index().rename(columns={"index": "pair"})
        rug = both.sort_values("rug_ts").groupby("pair").first()

    out = first.join(init_res, how="left").join(rug, how="left")
    out["init_weth"] = out.init_weth.fillna(out.init_weth_mint)
    return out.drop(columns="init_weth_mint").reset_index().rename(columns={"index": "pair"})


def assign_labels(rugs: pd.DataFrame, pairs: pd.DataFrame, n_hours: float, h_days: float,
                  pool_horizon_days: float, data_cutoff: int, min_init_weth: float = 0.0) -> pd.DataFrame:
    """Label for early detection at t0 + N.

    status: pos      rug in (t0+N, t0+N+H]
            neg      no rug up to t0+N+H (horizon fully observed)
            early_rug rug at or before t0+N (excluded: too late to predict)
            censored  t0+N+H beyond what was crawled
            dust      initial liquidity < min_init_weth
            no_liquidity  pair never received a Mint
    """
    df = pairs[["pair", "token", "created_ts", "token_creator"]].merge(rugs, on="pair", how="left")
    t_end = df.t0 + n_hours * HOUR
    h_end = t_end + h_days * DAY
    obs_end = np.minimum(df.created_ts + pool_horizon_days * DAY, data_cutoff)

    status = np.select(
        [df.t0.isna(),
         df.init_weth.fillna(0) < min_init_weth,
         df.rug_ts <= t_end,
         h_end > obs_end,
         (df.rug_ts > t_end) & (df.rug_ts <= h_end)],
        ["no_liquidity", "dust", "early_rug", "censored", "pos"],
        default="neg")
    df["t_end"] = t_end
    df["status"] = status
    df["y"] = (status == "pos").astype(int)
    df["n_hours"] = n_hours
    return df
