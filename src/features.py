"""Leak-free feature extraction: only events with ts <= t0 + N are visible.

extract_features(pool, t0, n_hours) works on one pool (a `PoolData` bundle).
build_features(...) runs it for every labeled pool and adds the cross-pool
creator-history features, which also only look at the past.

Feature groups (used by experiments/*.yaml):
  repo     : windowed re-implementation of the baseline's 18 features
             (counts instead of "per week", offsets / N instead of whole-life mean period)
  new      : concentration (HHI), liquidity dynamics, sniping, ownership, token age, ...
  contract : static bytecode features (setup D, see contract_features.py)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.constants import BURN_ADDRESSES, DEAD, LP_LOCKERS, ZERO

HOUR = 3600
EPS = 1e-12

REPO_FEATURES = [
    "n_mint", "n_burn", "mint_ratio", "swap_ratio", "burn_ratio",
    "mint_mean_offset", "swap_mean_offset", "burn_mean_offset",
    "n_buy", "n_sell", "buy_sell_rate", "lp_avg", "lp_std",
    "lp_creator_share", "lp_locked_share", "token_burned_share",
    "creator_token_share", "creator_prev_tokens",
]
NEW_FEATURES = [
    "n_swap", "events_per_hour", "buy_vol_eth", "sell_vol_eth", "net_flow_eth", "sell_buy_vol_ratio",
    "unique_buyers", "first_swap_delay", "swaps_in_launch_block", "frac_swaps_first_5min",
    "init_weth", "end_weth", "max_weth", "reserve_change", "max_drawdown",
    "price_change", "price_vol", "burn_weth_ratio", "n_lp_providers",
    "lp_holder_count", "lp_hhi", "lp_top1_share", "lp_burned_share",
    "token_holder_count", "token_hhi", "token_top1_share", "token_top10_share",
    "pair_token_share", "contract_self_share", "post_launch_mint_ratio",
    "n_token_transfers", "unique_token_receivers",
    "token_age_h", "owner_renounced", "n_owner_changes", "creator_is_lp_provider",
    "creator_prev_rugs",
]
TOKEN_ACCOUNTING_FEATURES = [
    "token_burned_share", "creator_token_share", "token_holder_count", "token_hhi",
    "token_top1_share", "token_top10_share", "pair_token_share", "contract_self_share",
]
ROBUST_REPO_FEATURES = [c for c in REPO_FEATURES if c not in TOKEN_ACCOUNTING_FEATURES]
ROBUST_FEATURES = [c for c in REPO_FEATURES + NEW_FEATURES if c not in TOKEN_ACCOUNTING_FEATURES]


class _Index:
    """O(1) slicing of a table pre-sorted by (key, ts)."""

    def __init__(self, df: pd.DataFrame, key: str):
        if df.attrs.get("index_sorted_by") != key:
            df.sort_values([key, "ts", "block_number", "log_index"], kind="stable",
                           inplace=True, ignore_index=True)
            df.attrs["index_sorted_by"] = key
        self.df = df
        keys, start = np.unique(self.df[key].to_numpy(), return_index=True)
        end = np.r_[start[1:], len(self.df)]
        self.pos = {k: (s, e) for k, s, e in zip(keys, start, end)}

    def get(self, k) -> pd.DataFrame:
        s, e = self.pos.get(k, (0, 0))
        return self.df.iloc[s:e]


@dataclass
class PoolData:
    pair: str
    token: str
    created_ts: int
    token_deployed_ts: int
    token_creator: str | None
    pool_ev: pd.DataFrame
    lp_tr: pd.DataFrame
    tok_tr: pd.DataFrame
    own_ev: pd.DataFrame


def _balances(tr: pd.DataFrame, skip_credit_from: set[str] = frozenset()) -> pd.Series:
    """Net balance per address from Transfer rows. Mints (from 0x0) are not debited;
    credits to 0x0 coming from `skip_credit_from` (e.g. the pair redeeming LP) are ignored."""
    if tr.empty:
        return pd.Series(dtype="float64")
    frm, to, v = tr.frm.to_numpy(), tr.to.to_numpy(), tr.value.to_numpy("float64")
    v = np.nan_to_num(v)
    credit_ok = ~((to == ZERO) & np.isin(frm, list(skip_credit_from) + [ZERO]))
    debit_ok = frm != ZERO
    addr = np.r_[to[credit_ok], frm[debit_ok]]
    amt = np.r_[v[credit_ok], -v[debit_ok]]
    return pd.Series(amt).groupby(addr).sum()


def _hhi(x: np.ndarray) -> float:
    x = x[x > 0]
    if x.size == 0:
        return np.nan
    s = x / x.sum()
    return float((s ** 2).sum())


def _offset(ts: np.ndarray, t0: int, w: float) -> float:
    return float(((ts - t0) / w).mean()) if ts.size else 0.0


def extract_features(p: PoolData, t0: int, n_hours: float) -> dict:
    w = n_hours * HOUR
    t_end = t0 + w
    f: dict[str, float] = {}

    # ---------------- pool activity ----------------
    ev = p.pool_ev[p.pool_ev.ts.to_numpy() <= t_end]
    kind = ev.event.to_numpy()
    ts = ev.ts.to_numpy()
    mint, burn, swap, sync = (kind == "mint"), (kind == "burn"), (kind == "swap"), (kind == "sync")
    n_mint, n_burn, n_swap = int(mint.sum()), int(burn.sum()), int(swap.sum())
    n_tx = max(n_mint + n_burn + n_swap, 1)
    f.update(n_mint=n_mint, n_burn=n_burn, n_swap=n_swap,
             mint_ratio=n_mint / n_tx, burn_ratio=n_burn / n_tx, swap_ratio=n_swap / n_tx,
             mint_mean_offset=_offset(ts[mint], t0, w), burn_mean_offset=_offset(ts[burn], t0, w),
             swap_mean_offset=_offset(ts[swap], t0, w), events_per_hour=n_tx / n_hours)

    sw = ev[swap]
    weth_in = np.nan_to_num(sw.weth_in.to_numpy("float64"))
    weth_out = np.nan_to_num(sw.weth_out.to_numpy("float64"))
    buy = weth_in > 0
    n_buy, n_sell = int(buy.sum()), int((~buy).sum())
    buy_vol, sell_vol = float(weth_in[buy].sum()), float(weth_out[~buy].sum())
    sw_ts = sw.ts.to_numpy()
    launch_block = ev.block_number.to_numpy()[mint][0] if n_mint else -1
    f.update(n_buy=n_buy, n_sell=n_sell, buy_sell_rate=n_buy / max(n_sell, 1),
             buy_vol_eth=buy_vol, sell_vol_eth=sell_vol, net_flow_eth=buy_vol - sell_vol,
             sell_buy_vol_ratio=sell_vol / (buy_vol + EPS),
             unique_buyers=int(pd.unique(sw.to.to_numpy()[buy]).size),
             first_swap_delay=float((sw_ts.min() - t0) / w) if n_swap else 1.0,
             swaps_in_launch_block=int((sw.block_number.to_numpy() == launch_block).sum()),
             frac_swaps_first_5min=float((sw_ts - t0 <= 300).sum() / max(n_swap, 1)))

    sy = ev[sync]
    res = sy.weth.to_numpy("float64")
    tok_res = sy.tok.to_numpy("float64")
    if res.size:
        cummax = np.maximum.accumulate(res)
        init = float(res[res > 0][0]) if (res > 0).any() else np.nan
        price = np.where(tok_res > 0, res / np.where(tok_res > 0, tok_res, 1), np.nan)
        price = price[np.isfinite(price) & (price > 0)]
        lr = np.diff(np.log(price)) if price.size > 1 else np.array([])
        f.update(init_weth=init, end_weth=float(res[-1]), max_weth=float(res.max()),
                 reserve_change=float(res[-1] / init) if init else np.nan,
                 max_drawdown=float(np.max(1 - res / np.where(cummax > 0, cummax, 1))),
                 price_change=float(price[-1] / price[0]) if price.size else np.nan,
                 price_vol=float(lr.std()) if lr.size > 1 else 0.0)
    else:
        f.update(init_weth=np.nan, end_weth=np.nan, max_weth=np.nan, reserve_change=np.nan,
                 max_drawdown=np.nan, price_change=np.nan, price_vol=np.nan)
    minted = float(np.nansum(ev.weth.to_numpy("float64")[mint]))
    burned = float(np.nansum(ev.weth.to_numpy("float64")[burn]))
    f["burn_weth_ratio"] = burned / (minted + EPS)

    # ---------------- LP token holders ----------------
    lp = p.lp_tr[p.lp_tr.ts.to_numpy() <= t_end]
    lp_mints = lp[(lp.frm.to_numpy() == ZERO) & (lp.to.to_numpy() != ZERO)]
    lp_creator = lp_mints.to.iloc[0] if len(lp_mints) else None
    f["n_lp_providers"] = int(lp_mints.to.nunique())
    bal = _balances(lp, skip_credit_from={p.pair}).drop(labels=[p.pair], errors="ignore")
    bal = bal[bal > 0]
    total = float(bal.sum())
    if total > 0:
        burned_lp = float(bal.reindex(list(BURN_ADDRESSES)).fillna(0).sum())
        locked_lp = float(bal.reindex(list(LP_LOCKERS)).fillna(0).sum())
        # MINIMUM_LIQUIDITY is minted 0x0 -> 0x0 and skipped by _balances, so this is creator-burned LP only
        holders = bal.drop(labels=list(BURN_ADDRESSES), errors="ignore")
        shares = holders.to_numpy() / total
        f.update(lp_holder_count=int((shares > 1e-9).sum()), lp_hhi=_hhi(holders.to_numpy()),
                 lp_top1_share=float(shares.max()) if shares.size else 0.0,
                 lp_avg=100.0 / max(len(holders), 1), lp_std=float(np.std(shares * 100)) if shares.size else 0.0,
                 lp_creator_share=float(bal.get(lp_creator, 0.0) / total) if lp_creator else 0.0,
                 lp_locked_share=locked_lp / total, lp_burned_share=burned_lp / total)
    else:
        f.update(lp_holder_count=0, lp_hhi=np.nan, lp_top1_share=np.nan, lp_avg=np.nan, lp_std=np.nan,
                 lp_creator_share=np.nan, lp_locked_share=np.nan, lp_burned_share=np.nan)

    # ---------------- token holders ----------------
    tk = p.tok_tr[p.tok_tr.ts.to_numpy() <= t_end]
    creator = p.token_creator if isinstance(p.token_creator, str) and p.token_creator else lp_creator
    frm, to = tk.frm.to_numpy(), tk.to.to_numpy()
    val = np.nan_to_num(tk.value.to_numpy("float64"))
    minted_all = float(val[frm == ZERO].sum())
    minted_post = float(val[(frm == ZERO) & (tk.ts.to_numpy() > t0)].sum())
    tb = _balances(tk)
    burned_balance = max(float(tb.get(ZERO, 0.0)), 0.0) + max(float(tb.get(DEAD, 0.0)), 0.0)
    supply = minted_all - burned_balance
    live_balances = tb.drop(labels=list(BURN_ADDRESSES), errors="ignore")
    tolerance = max(abs(supply) * 1e-6, 1e-6)
    accounting_valid = (supply > 0 and (live_balances >= -tolerance).all()
                        and abs(float(live_balances.sum()) - supply) <= tolerance)
    f["token_accounting_valid"] = int(accounting_valid)
    if accounting_valid:
        circ = tb.drop(labels=list(BURN_ADDRESSES) + [p.pair, p.token], errors="ignore")
        circ = circ[circ > 0]
        top = np.sort(circ.to_numpy())[::-1]
        f.update(token_holder_count=int(circ.size), token_hhi=_hhi(circ.to_numpy()),
                 token_top1_share=float(top[:1].sum() / supply), token_top10_share=float(top[:10].sum() / supply),
                 pair_token_share=float(max(tb.get(p.pair, 0.0), 0.0) / supply),
                 contract_self_share=float(max(tb.get(p.token, 0.0), 0.0) / supply),
                 token_burned_share=float(burned_balance / max(minted_all, EPS)),
                 creator_token_share=float(max(tb.get(creator, 0.0), 0.0) / supply) if creator else np.nan)
    else:
        f.update(token_holder_count=np.nan, token_hhi=np.nan, token_top1_share=np.nan,
                 token_top10_share=np.nan, pair_token_share=np.nan, contract_self_share=np.nan,
                 token_burned_share=np.nan, creator_token_share=np.nan)
    in_win = tk.ts.to_numpy() >= t0
    f.update(post_launch_mint_ratio=minted_post / (minted_all - minted_post + EPS),
             n_token_transfers=int(in_win.sum()), unique_token_receivers=int(pd.unique(to[in_win]).size),
             token_age_h=(p.created_ts - p.token_deployed_ts) / HOUR,
             creator_is_lp_provider=int(bool(creator) and creator == lp_creator))

    # ---------------- ownership ----------------
    ow = p.own_ev[p.own_ev.ts.to_numpy() <= t_end]
    f["owner_renounced"] = int(len(ow) > 0 and ow.new_owner.iloc[-1] in BURN_ADDRESSES)
    f["n_owner_changes"] = int(len(ow))
    return f


def creator_history(lab: pd.DataFrame) -> pd.DataFrame:
    """creator_prev_tokens: earlier pairs by the same creator (by created_ts).
    creator_prev_rugs: earlier pairs by the same creator whose rug happened before t_end.
    Uses only information available at prediction time (the baseline counted future tokens too)."""
    df = lab[["pair", "creator", "created_ts", "t_end"]].copy()
    hist = lab[["creator", "created_ts", "rug_ts"]].dropna(subset=["creator"])
    out_tokens, out_rugs = [], []
    by_creator = {c: g.sort_values("created_ts") for c, g in hist.groupby("creator")}
    for c, cts, tend in zip(df.creator, df.created_ts, df.t_end):
        g = by_creator.get(c)
        if g is None:
            out_tokens.append(0); out_rugs.append(0); continue
        prev = g[g.created_ts.to_numpy() < cts]
        out_tokens.append(len(prev))
        out_rugs.append(int((prev.rug_ts.to_numpy() <= tend).sum()))
    df["creator_prev_tokens"] = out_tokens
    df["creator_prev_rugs"] = out_rugs
    return df[["pair", "creator_prev_tokens", "creator_prev_rugs"]]


def build_features(lab: pd.DataFrame, pairs: pd.DataFrame, tables: dict[str, pd.DataFrame],
                   n_hours: float, statuses=("pos", "neg"), n_jobs: int = 1) -> pd.DataFrame:
    """lab: output of label.assign_labels for this N. Returns one row per kept pool."""
    keep = lab[lab.status.isin(statuses)].copy()
    if keep.empty:
        raise ValueError(f"no pools with status in {statuses}; label status counts: "
                         f"{lab.status.value_counts().to_dict()}")
    meta = pairs.set_index("pair")
    idx = {"pool_ev": _Index(tables["pool_ev"], "pair"), "lp_tr": _Index(tables["lp_tr"], "pair"),
           "tok_tr": _Index(tables["tok_tr"], "token"), "own_ev": _Index(tables["own_ev"], "token")}

    def one(pair, t0):
        m = meta.loc[pair]
        pdata = PoolData(pair=pair, token=m.token, created_ts=int(m.created_ts),
                         token_deployed_ts=int(m.token_deployed_ts), token_creator=m.token_creator,
                         pool_ev=idx["pool_ev"].get(pair), lp_tr=idx["lp_tr"].get(pair),
                         tok_tr=idx["tok_tr"].get(m.token), own_ev=idx["own_ev"].get(m.token))
        return {"pair": pair, **extract_features(pdata, int(t0), n_hours)}

    items = list(zip(keep.pair, keep.t0))
    if n_jobs == 1:
        rows = [one(p, t) for p, t in items]
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs, backend="threading", batch_size=256)(delayed(one)(p, t) for p, t in items)
    feats = pd.DataFrame(rows)
    # columns already in the label table (e.g. init_weth) win, so merge does not create _x/_y copies
    feats = feats.drop(columns=[c for c in feats.columns if c != "pair" and c in keep.columns])

    # history is computed over all labeled pools (any status), but only looks backwards in time
    hist = creator_history(lab.assign(creator=lab.token_creator))
    return keep.merge(feats, on="pair").merge(hist, on="pair", how="left")
