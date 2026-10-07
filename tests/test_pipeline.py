import math
import time

import numpy as np
import pandas as pd
import pytest

from src.bq_fetch import hexf, render
from src.decode import decode, to_unix
from src.features import PoolData, _Index, build_features, extract_features
from src.label import assign_labels, detect_rugs
from src.split import temporal_split
from tests.synthetic import make_raw

HOUR, DAY = 3600, 86400


@pytest.fixture(scope="module")
def data():
    pairs, ev, contracts, truth = make_raw(n_pools=300, seed=1)
    pairs = pairs.copy()
    for c in ["created_ts", "token_deployed_ts"]:
        pairs[c] = to_unix(pairs[c])
    tables = decode(ev, pairs)
    rugs = detect_rugs(tables["pool_ev"])
    return pairs, tables, rugs, truth


def _py_hexf(word: str) -> float:
    """Python mirror of the SQL produced by hexf()."""
    parts = [(1, 4, 240), (5, 15, 180), (20, 15, 120), (35, 15, 60), (50, 15, 0)]
    return sum(int(word[s - 1:s - 1 + n], 16) * 2.0 ** e for s, n, e in parts)


@pytest.mark.parametrize("v", [0, 1, 10**18, 123456789 * 10**30, 2**112 - 1, 2**256 - 1])
def test_hexf_matches_uint256(v):
    w = f"{v:064x}"
    assert math.isclose(_py_hexf(w), float(v), rel_tol=1e-12)
    assert "SAFE_CAST" in hexf("w0")


def test_sql_renders_without_placeholders():
    for f in ["01_pairs.sql", "02_events.sql", "03_contracts.sql"]:
        sql = render(f, "proj.ds")
        code = "\n".join(l for l in sql.splitlines() if not l.lstrip().startswith("--"))
        assert "{" not in code and "HEXF(" not in code, f


def test_rug_times_exact(data):
    _, _, rugs, truth = data
    m = truth.rename(columns={"rug_ts": "true_rug"}).merge(rugs, on="pair")
    assert (m.t0_x == m.t0_y).all()
    assert (m.true_rug.fillna(-1).astype(int) == m.rug_ts.fillna(-1).astype(int)).all()


def test_label_statuses(data):
    pairs, _, rugs, _ = data
    lab = assign_labels(rugs, pairs, n_hours=1, h_days=30, pool_horizon_days=45,
                        data_cutoff=int(time.time()), min_init_weth=0.1)
    early = lab[lab.status == "early_rug"]
    assert (early.rug_ts <= early.t0 + HOUR).all()
    pos = lab[lab.status == "pos"]
    assert ((pos.rug_ts > pos.t_end) & (pos.rug_ts <= pos.t_end + 30 * DAY)).all()
    # a cutoff in the past censors everything that has not been observed for N + H
    cens = assign_labels(rugs, pairs, 1, 30, 45, data_cutoff=int(pairs.created_ts.min()) + 10 * DAY)
    assert (cens.status.isin(["censored", "early_rug", "dust", "no_liquidity", "pos"])).all()


@pytest.mark.parametrize("n_hours", [1, 6, 24])
def test_features_ignore_the_future(data, n_hours):
    """Core leakage test: features must not change when events after t0+N are deleted."""
    pairs, tables, rugs, _ = data
    idx = {k: _Index(v, "pair" if k in ("pool_ev", "lp_tr") else "token") for k, v in tables.items()}
    meta = pairs.set_index("pair")
    for pair in rugs.pair[:60]:
        m = meta.loc[pair]
        t0 = int(rugs.set_index("pair").t0[pair])
        t_end = t0 + n_hours * HOUR

        def bundle(cut):
            g = lambda k, key: idx[k].get(key)
            f = (lambda d: d[d.ts <= t_end]) if cut else (lambda d: d)
            return PoolData(pair, m.token, int(m.created_ts), int(m.token_deployed_ts), m.token_creator,
                            f(g("pool_ev", pair)), f(g("lp_tr", pair)), f(g("tok_tr", m.token)),
                            f(g("own_ev", m.token)))

        a = extract_features(bundle(False), t0, n_hours)
        b = extract_features(bundle(True), t0, n_hours)
        for k in a:
            assert (a[k] == b[k]) or (pd.isna(a[k]) and pd.isna(b[k])), (pair, k, a[k], b[k])


def test_creator_history_only_looks_back(data):
    pairs, tables, rugs, _ = data
    lab = assign_labels(rugs, pairs, 6, 30, 45, int(time.time()), 0.1)
    ds = build_features(lab, pairs, tables, 6)
    # brute force over *all* pairs (the creator's earlier pools may be filtered out of ds)
    for _, r in ds.iterrows():
        prev = lab[(lab.token_creator == r.token_creator) & (lab.created_ts < r.created_ts)]
        assert r.creator_prev_tokens == len(prev)
        assert r.creator_prev_rugs == int((prev.rug_ts <= r.t_end).sum())
    assert (ds.creator_prev_rugs <= ds.creator_prev_tokens).all()


def test_temporal_split_purge():
    t = pd.Timestamp("2022-01-01", tz="UTC").timestamp()
    df = pd.DataFrame({"t0": [t - 40 * DAY, t - 5 * DAY, t + DAY, t + 200 * DAY],
                       "t_end": [t - 40 * DAY + HOUR, t - 5 * DAY + HOUR, t + DAY + HOUR, t + 200 * DAY + HOUR],
                       "y": [0, 1, 0, 1]})
    tr, va, te = temporal_split(df, "2022-01-01", "2022-07-01", h_days=30)
    assert len(tr) == 1  # the pool 5 days before the boundary is purged (its label horizon crosses it)
    assert len(va) == 1 and len(te) == 1


def test_etherscan_parsing():
    from src.constants import TOPIC_SWAP
    from src.etherscan_fetch import log_row, selectors_from_bytecode
    # PUSH4 0xa9059cbb, PUSH32 containing a fake 0x63 byte that must be skipped
    code = "0x63a9059cbb7f" + "63deadbeef" + "00" * 27 + "638da5cb5b"
    assert selectors_from_bytecode(code) == ["0x8da5cb5b", "0xa9059cbb"]
    words = "".join(f"{v:064x}" for v in (0, 5 * 10**18, 123, 0))
    l = {"address": "0xABC", "topics": [TOPIC_SWAP, "0x" + "0" * 24 + "1" * 40, "0x" + "0" * 24 + "2" * 40],
         "data": "0x" + words, "blockNumber": "0x10", "timeStamp": "0x5f5e100", "logIndex": "0x",
         "transactionHash": "0xt"}
    r = log_row(l, "pool")
    assert r["event"] == "swap" and r["log_index"] == 0 and r["block_number"] == 16
    assert r["v1"] == 5e18 and r["v2"] == 123 and r["addr1"] == "0x" + "1" * 40 and r["address"] == "0xabc"


def test_to_unix_any_resolution():
    for unit in ["ns", "us", "s"]:
        col = pd.Series(pd.to_datetime([1_650_000_000], unit="s", utc=True)).astype(f"datetime64[{unit}, UTC]")
        assert to_unix(col).iloc[0] == 1_650_000_000
