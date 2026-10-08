import math
import time

import numpy as np
import pandas as pd
import pytest

from src.bq_fetch import hexf, render
from src.contract_features import contract_features
from src.constants import ZERO
from src.decode import decode, to_unix
from src.features import (ROBUST_REPO_FEATURES, TOKEN_ACCOUNTING_FEATURES, PoolData, _Index,
                          build_features, extract_features)
from src.label import assign_labels, detect_rugs
from src.label_audit_report import build_report
from src.split import make_split, temporal_split
from src.train import best_threshold, evaluate
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


def test_observed_positive_is_not_censored():
    """An observed event is positive even when the full negative follow-up is unavailable."""
    t0 = 1_700_000_000
    pairs = pd.DataFrame({"pair": ["p"], "token": ["t"], "created_ts": [t0],
                          "token_creator": ["c"]})
    rugs = pd.DataFrame({"pair": ["p"], "t0": [t0], "init_tok": [1.0], "init_weth": [1.0],
                         "rug_ts": [t0 + 2 * DAY], "rug_type": ["burn"], "rug_drop": [0.999]})
    lab = assign_labels(rugs, pairs, n_hours=1, h_days=30, pool_horizon_days=45,
                        data_cutoff=t0 + 3 * DAY, min_init_weth=0.1)
    assert lab.status.iloc[0] == "pos"


def test_exact_drop_threshold_survives_float_roundoff():
    pair, tx0, tx1 = "p", "init", "rug"
    ev = pd.DataFrame([
        {"pair": pair, "ts": 1, "block_number": 1, "log_index": 1, "tx": tx0,
         "event": "sync", "weth": 1.0, "tok": 100.0, "tok_in": np.nan},
        {"pair": pair, "ts": 1, "block_number": 1, "log_index": 2, "tx": tx0,
         "event": "mint", "weth": 1.0, "tok": 100.0, "tok_in": np.nan},
        {"pair": pair, "ts": 2, "block_number": 2, "log_index": 1, "tx": tx1,
         "event": "sync", "weth": 0.010000000000000009, "tok": 1.0, "tok_in": np.nan},
        {"pair": pair, "ts": 2, "block_number": 2, "log_index": 2, "tx": tx1,
         "event": "burn", "weth": 0.99, "tok": 99.0, "tok_in": np.nan},
    ])
    out = detect_rugs(ev)
    assert out.rug_type.iloc[0] == "burn"
    assert out.rug_drop.iloc[0] == pytest.approx(0.99)


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


def test_template_coldstart_removes_seen_contract_family():
    t = pd.Timestamp("2022-01-01", tz="UTC").timestamp()
    df = pd.DataFrame({"t0": [t - 60 * DAY, t + DAY, t + 200 * DAY, t + 201 * DAY],
                       "t_end": [t - 60 * DAY, t + DAY, t + 200 * DAY, t + 201 * DAY],
                       "y": [0, 1, 0, 1], "bytecode_hash": ["seen", "val", "seen", "new"]})
    spec = {"kind": "temporal", "train_end": "2022-01-01", "val_end": "2022-07-01",
            "purge": False, "drop_seen_templates": True}
    _, _, te = make_split(df, spec, h_days=30, seed=42)
    assert te.bytecode_hash.tolist() == ["new"]


def test_empty_bytecode_is_not_counted_as_reused_template():
    contracts = pd.DataFrame({"token": ["a", "b", "c"], "function_sighashes": [[], [], []],
                              "bytecode_size": [0, 0, 10], "bytecode_hash": [None, None, "hash"]})
    pairs = pd.DataFrame({"token": ["a", "b", "c"], "created_ts": [1, 2, 3]})
    out = contract_features(contracts, pairs).set_index("token")
    assert out.loc[["a", "b"], "bytecode_dup_count"].isna().all()
    assert out.loc[["a", "b"], "bytecode_missing"].eq(1).all()
    assert out.loc["c", "bytecode_dup_count"] == 0


def test_threshold_and_extended_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    threshold = best_threshold(y, p)
    r = evaluate(y, p, threshold)
    assert r["f1"] == 1 and r["tn"] == 2 and r["tp"] == 2
    assert 0 <= r["ece"] <= 1 and r["pr_baseline"] == 0.5


def test_invalid_token_accounting_becomes_missing(data):
    assert not set(ROBUST_REPO_FEATURES) & set(TOKEN_ACCOUNTING_FEATURES)
    pairs, tables, rugs, _ = data
    pair = rugs.pair.iloc[0]
    meta = pairs.set_index("pair").loc[pair]
    t0 = int(rugs.set_index("pair").loc[pair, "t0"])
    bad = tables["tok_tr"][tables["tok_tr"].token.eq(meta.token)].copy()
    if bad.empty:
        pytest.skip("synthetic token has no transfers")
    spend = bad.index[bad.frm.ne(ZERO)]
    if not len(spend):
        pytest.skip("synthetic token has no non-mint transfer")
    bad.loc[spend[0], "value"] = 1e100
    p = PoolData(pair, meta.token, int(meta.created_ts), int(meta.token_deployed_ts), meta.token_creator,
                 tables["pool_ev"][tables["pool_ev"].pair.eq(pair)],
                 tables["lp_tr"][tables["lp_tr"].pair.eq(pair)], bad,
                 tables["own_ev"][tables["own_ev"].token.eq(meta.token)])
    f = extract_features(p, t0, 1)
    assert f["token_accounting_valid"] == 0
    assert pd.isna(f["token_top1_share"]) and pd.isna(f["pair_token_share"])


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


def test_label_audit_report_handles_blinding_and_adjudication():
    master = pd.DataFrame({"audit_id": ["a", "b", "c"],
                           "audit_group": ["rug_burn", "rug_swap", "no_detected_rug"],
                           "pair": ["p1", "p2", "p3"], "token": ["t1", "t2", "t3"],
                           "adjudicated_label": ["", "rug", ""]})
    r1 = pd.DataFrame({"audit_id": ["c", "a", "b"],
                       "reviewer_label": ["not_rug", "rug", "not_rug"]})
    r2 = pd.DataFrame({"audit_id": ["b", "c", "a"],
                       "reviewer_label": ["rug", "not_rug", "rug"]})
    report, strata, disagreements = build_report(
        master, r1, r2, {"rug_burn": 10, "rug_swap": 10, "no_detected_rug": 20})
    assert report["double_reviewed_n"] == 3
    assert report["unresolved_disagreements_n"] == 0
    assert report["detected_label_precision_population_weighted"] == 1.0
    assert len(disagreements) == 1 and len(strata) == 3
