"""Synthetic Uniswap V2 data in the exact format exported by src/bq_fetch.py.

Used by the tests and by `python -m tests.synthetic` to smoke-test the whole
pipeline before spending BigQuery credits. Scam pools get rugged by an LP burn
(or a large dump swap) at a known time, stored in `truth`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import WETH, ZERO, LP_LOCKERS

WEI = 1e18
LOCKER = sorted(LP_LOCKERS)[0]


def _addr(rng) -> str:
    return "0x" + "".join(rng.choice(list("0123456789abcdef"), 40))


class _Pool:
    def __init__(self, pair, token, tok0):
        self.pair, self.token, self.tok0 = pair, token, tok0
        self.rows, self.block, self.li = [], 0, 0
        self.w, self.k = 0.0, 0.0  # reserves (WETH in ETH, token raw)

    def _tx(self, ts):
        self.block = max(self.block + 1, int(ts // 12))
        self.li = 0
        return f"0x{self.pair[2:10]}{self.block:x}{len(self.rows):x}"

    def _log(self, ts, tx, address, role, event, a1=None, a2=None, v=(np.nan,) * 4):
        self.rows.append(dict(address=address, role=role, ts=int(ts), block_number=self.block,
                              transaction_hash=tx, log_index=self.li, event=event, addr1=a1, addr2=a2,
                              v0=v[0], v1=v[1], v2=v[2], v3=v[3]))
        self.li += 1

    def _pair_vals(self, w_eth, k):  # (token0, token1) ordering
        return (k, w_eth * WEI) if self.tok0 else (w_eth * WEI, k)

    def _sync(self, ts, tx):
        self._log(ts, tx, self.pair, "pool", "sync", v=(*self._pair_vals(self.w, self.k), np.nan, np.nan))

    def token_transfer(self, ts, frm, to, value, tx=None):
        tx = tx or self._tx(ts)
        self._log(ts, tx, self.token, "token", "transfer", frm, to, (value, np.nan, np.nan, np.nan))

    def lp_transfer(self, ts, frm, to, value, tx=None):
        tx = tx or self._tx(ts)
        self._log(ts, tx, self.pair, "lp", "transfer", frm, to, (value, np.nan, np.nan, np.nan))

    def add_liquidity(self, ts, who, w_eth, k, lp_amount, first=False):
        tx = self._tx(ts)
        self.token_transfer(ts, who, self.pair, k, tx)
        if first:
            self.lp_transfer(ts, ZERO, ZERO, 1000.0, tx)
        self.lp_transfer(ts, ZERO, who, lp_amount, tx)
        self.w += w_eth
        self.k += k
        self._sync(ts, tx)
        self._log(ts, tx, self.pair, "pool", "mint", "0xrouter", None, (*self._pair_vals(w_eth, k), np.nan, np.nan))

    def remove_liquidity(self, ts, who, frac, lp_amount):
        tx = self._tx(ts)
        dw, dk = self.w * frac, self.k * frac
        self.lp_transfer(ts, who, self.pair, lp_amount, tx)
        self.lp_transfer(ts, self.pair, ZERO, lp_amount, tx)
        self.token_transfer(ts, self.pair, who, dk, tx)
        self.w -= dw
        self.k -= dk
        self._sync(ts, tx)
        self._log(ts, tx, self.pair, "pool", "burn", "0xrouter", who, (*self._pair_vals(dw, dk), np.nan, np.nan))

    def swap(self, ts, who, buy: bool, amount):
        """buy: amount = ETH in; sell: amount = token in."""
        tx = self._tx(ts)
        if buy:
            out = self.k * amount / (self.w + amount)
            self.token_transfer(ts, self.pair, who, out, tx)
            self.w += amount
            self.k -= out
            ins, outs = (0.0, amount * WEI) if self.tok0 else (amount * WEI, 0.0), (out, 0.0) if self.tok0 else (0.0, out)
        else:
            out = self.w * amount / (self.k + amount)
            self.token_transfer(ts, who, self.pair, amount, tx)
            self.k += amount
            self.w -= out
            ins, outs = (amount, 0.0) if self.tok0 else (0.0, amount), (0.0, out * WEI) if self.tok0 else (out * WEI, 0.0)
        self._sync(ts, tx)
        self._log(ts, tx, self.pair, "pool", "swap", "0xrouter", who, (ins[0], ins[1], outs[0], outs[1]))


def make_raw(n_pools: int = 600, seed: int = 0, start="2021-01-01", end="2023-06-01"):
    rng = np.random.default_rng(seed)
    t_start = datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp()
    t_stop = datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp()
    creators = [_addr(rng) for _ in range(n_pools // 4)]
    scammers = set(creators[: len(creators) // 3])
    pairs, rows, truth, contracts = [], [], [], []
    for i in range(n_pools):
        creator = creators[rng.integers(len(creators))]
        scam = creator in scammers and rng.random() < 0.85
        pair, token = _addr(rng), _addr(rng)
        tok0 = token < WETH
        created = rng.uniform(t_start, t_stop)
        deployed = created - rng.uniform(60, 3 * 3600)
        p = _Pool(pair, token, tok0)
        supply = 1e27
        p.token_transfer(deployed, ZERO, creator, supply)
        t0 = created + rng.uniform(10, 600)
        lp_amt = 1e18
        p.add_liquidity(t0, creator, rng.uniform(1, 20), supply * (0.95 if scam else 0.6), lp_amt, first=True)
        if not scam and rng.random() < 0.7:
            p.lp_transfer(t0 + 120, creator, LOCKER, lp_amt * 0.9)
        if not scam:
            for _ in range(rng.integers(5, 40)):  # airdrop / team distribution
                p.token_transfer(t0 + rng.uniform(0, 3600), creator, _addr(rng), supply * 0.005)
        if scam and rng.random() < 0.3:
            p.rows.append(dict(address=token, role="token", ts=int(t0 + 30), block_number=p.block,
                               transaction_hash="0xown", log_index=99, event="ownership",
                               addr1=creator, addr2=creator, v0=np.nan, v1=np.nan, v2=np.nan, v3=np.nan))
        elif not scam:
            p.rows.append(dict(address=token, role="token", ts=int(t0 + 300), block_number=p.block,
                               transaction_hash="0xown2", log_index=99, event="ownership",
                               addr1=creator, addr2=ZERO, v0=np.nan, v1=np.nan, v2=np.nan, v3=np.nan))

        rug_ts = None
        if scam:
            rug_ts = t0 + (rng.uniform(600, 3000) if rng.random() < 0.2 else rng.uniform(2 * 3600, 20 * 86400))
        n_swaps = rng.integers(10, 120)
        horizon = 40 * 86400
        times = np.sort(t0 + rng.exponential(6 * 3600 if scam else 2 * 86400, n_swaps))
        if scam:  # snipers right at launch
            times = np.sort(np.r_[times, t0 + rng.uniform(0, 60, rng.integers(3, 10))])
        buyers = [_addr(rng) for _ in range(30)]
        for t in times:
            if t > created + horizon or (rug_ts and t >= rug_ts):
                break
            b = buyers[rng.integers(len(buyers))]
            if rng.random() < (0.75 if scam else 0.55) or p.k <= 0:
                p.swap(t, b, True, rng.uniform(0.01, 0.5))
            else:
                p.swap(t, b, False, p.k * rng.uniform(0.001, 0.01))
        if rug_ts:
            if rng.random() < 0.8:
                p.remove_liquidity(rug_ts, creator, 0.9995, lp_amt)
            else:
                p.swap(rug_ts, creator, False, p.k * 10_000)
        rows += p.rows
        pairs.append(dict(pair=pair, token=token, token_is_token0=tok0, token0=token if tok0 else WETH,
                          token1=WETH if tok0 else token,
                          created_ts=pd.Timestamp(created, unit="s", tz="UTC"), created_block=int(created // 12),
                          created_tx="0x" + pair[2:], token_deployed_ts=pd.Timestamp(deployed, unit="s", tz="UTC"),
                          token_deployed_block=int(deployed // 12), token_creator=creator,
                          cohort_date=pd.Timestamp(created, unit="s").date()))
        truth.append(dict(pair=pair, t0=int(t0), scam=scam, rug_ts=int(rug_ts) if rug_ts else None))
        sigs = ["0x8da5cb5b", "0xa9059cbb"] + (["0xb515566a", "0x40c10f19"] if scam and rng.random() < 0.6 else [])
        contracts.append(dict(token=token, function_sighashes=sigs, is_erc20=True,
                              bytecode_size=int(rng.integers(3000, 20000)), bytecode_hash=f"h{int(scam)}{rng.integers(5)}"))
    ev = pd.DataFrame(rows)
    return pd.DataFrame(pairs), ev, pd.DataFrame(contracts), pd.DataFrame(truth)


def write_raw(raw_dir: Path, **kw):
    pairs, ev, contracts, truth = make_raw(**kw)
    (raw_dir / "events").mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(raw_dir / "pairs.parquet")
    contracts.to_parquet(raw_dir / "contracts.parquet")
    ev.to_parquet(raw_dir / "events" / "synthetic.parquet")
    truth.to_parquet(raw_dir / "truth.parquet")
    return pairs, ev, contracts, truth


if __name__ == "__main__":
    from src.config import resolve
    out = resolve("data/synthetic/raw")
    p, e, _, t = write_raw(out)
    print(f"wrote {len(p)} pairs, {len(e):,} events, {t.scam.mean():.0%} scam -> {out}")
