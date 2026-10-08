"""Free alternative to bq_fetch.py: crawl via the Etherscan API (V2, chainid=1).

Writes the same files as `bq_fetch export`, so the rest of the pipeline is unchanged:
    data/raw/pairs.parquet, data/raw/contracts.parquet, data/raw/events/YYYY-MM.parquet

    python -m src.etherscan_fetch pairs      # all WETH PairCreated logs in [start, end)  (~a few hundred calls)
    python -m src.etherscan_fetch sample     # stratified sample per month + token creation info
    python -m src.etherscan_fetch pools      # per-pool logs (resumable; ~4-5 calls per pool)
    python -m src.etherscan_fetch export     # cache -> events/*.parquet, contracts.parquet

Needs ETHERSCAN_API_KEY in the environment or in .env (free key from etherscan.io/myapikey).
Free tier: 5 calls/s, 100k calls/day. Every step is resumable: just rerun it.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src import constants as C
from src.artifacts import atomic_parquet, write_parquet_manifest
from src.config import ROOT, load_config, resolve

API = "https://api.etherscan.io/v2/api"
PAGE = 1000          # max records per getLogs page
MAX_PAGES = 10       # Etherscan refuses page * offset > 10000 -> split the block range instead
TOPIC_NAMES = {C.TOPIC_SYNC: "sync", C.TOPIC_MINT: "mint", C.TOPIC_BURN: "burn", C.TOPIC_SWAP: "swap",
               C.TOPIC_TRANSFER: "transfer", C.TOPIC_OWNERSHIP: "ownership"}
BLOCK_S = 12.0       # seconds per block (post-merge exact, pre-merge ~13s -> windows are filtered by ts anyway)


def _load_key() -> str:
    key = os.environ.get("ETHERSCAN_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("ETHERSCAN_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit("Put ETHERSCAN_API_KEY=... in .env (free key: https://etherscan.io/myapikey)")
    return key


class Etherscan:
    def __init__(self, calls_per_sec: float = 4.5):
        self.key = _load_key()
        self.gap = 1.0 / calls_per_sec
        self.last = 0.0
        self.calls = 0
        self.lock = threading.Lock()
        self.local = threading.local()  # one HTTP session per worker thread

    def _throttle(self):
        """Global rate limit shared by all threads: requests start >= self.gap seconds apart."""
        with self.lock:
            now = time.time()
            start = max(now, self.last + self.gap)
            self.last = start
        if start > now:
            time.sleep(start - now)

    def get(self, **params):
        params.update(chainid=1, apikey=self.key)
        attempt = 0
        while attempt < 8:
            self._throttle()
            if not hasattr(self.local, "s"):
                self.local.s = requests.Session()
            try:
                r = self.local.s.get(API, params=params, timeout=60).json()
            except (requests.RequestException, ValueError):
                attempt += 1
                time.sleep(2 ** attempt)
                continue
            self.calls += 1
            res = r.get("result")
            msg = str(r.get("message", "")) + " " + (res if isinstance(res, str) else "")
            if "rate limit" in msg.lower() or "max calls" in msg.lower():
                if "daily" in msg.lower() or "per day" in msg.lower():
                    raise SystemExit(f"Daily Etherscan limit reached ({msg.strip()}). Rerun tomorrow - progress is cached.")
                time.sleep(1.5)  # per-second limit: wait and retry without using up an attempt
                continue
            if params.get("module") == "proxy":
                return res
            if r.get("status") == "1":
                return res
            if "No records found" in msg or res == [] or res is None:
                return []
            if "Invalid API Key" in msg:
                raise SystemExit("Invalid ETHERSCAN_API_KEY")
            attempt += 1
            time.sleep(1 + attempt)
        raise RuntimeError(f"Etherscan failed: {params.get('module')}/{params.get('action')}")

    def logs(self, from_block: int, to_block: int, address: str | None = None, topic0: str | None = None) -> list[dict]:
        """All logs in [from_block, to_block], splitting the range when it exceeds 10k results."""
        out, page = [], 1
        while True:
            q = dict(module="logs", action="getLogs", fromBlock=from_block, toBlock=to_block, page=page, offset=PAGE)
            if address:
                q["address"] = address
            if topic0:
                q["topic0"] = topic0
            res = self.get(**q)
            out += res
            if len(res) < PAGE:
                return out
            if page == MAX_PAGES:
                last = int(res[-1]["blockNumber"], 16)
                if last <= from_block:
                    raise RuntimeError(
                        f"more than {PAGE * MAX_PAGES} logs in block {from_block}; "
                        "refusing to silently export an incomplete event history")
                out = [l for l in out if int(l["blockNumber"], 16) < last]
                return out + self.logs(last, to_block, address, topic0)
            page += 1

    def block_at(self, ts: int) -> int:
        r = self.get(module="block", action="getblocknobytime", timestamp=int(ts), closest="before")
        return int(r if isinstance(r, str) else r[0])


def _int(x):
    """Etherscan mixes decimal and 0x-hex strings across endpoints."""
    if x in (None, "", "0x"):
        return None
    return int(x, 16) if str(x).startswith("0x") else int(x)


def _word(data: str, i: int) -> str:
    return data[2 + 64 * i: 2 + 64 * (i + 1)]


def _num(w: str) -> float:
    return float(int(w, 16)) if w else np.nan


def _topic_addr(t: list, i: int):
    return ("0x" + t[i][-40:]).lower() if len(t) > i else None


def log_row(l: dict, role: str) -> dict | None:
    t = l["topics"]
    ev = TOPIC_NAMES.get(t[0].lower()) if t else None
    if ev is None or (ev == "transfer" and len(t) != 3):
        return None
    d = l["data"]
    return dict(address=l["address"].lower(), role=role, ts=int(l["timeStamp"], 16),
                block_number=int(l["blockNumber"], 16), transaction_hash=l["transactionHash"],
                log_index=int(l["logIndex"], 16) if l["logIndex"] not in ("0x", "") else 0, event=ev,
                addr1=_topic_addr(t, 1), addr2=_topic_addr(t, 2),
                v0=_num(_word(d, 0)), v1=_num(_word(d, 1)), v2=_num(_word(d, 2)), v3=_num(_word(d, 3)))


def selectors_from_bytecode(code: str) -> list[str]:
    """PUSH4 operands in runtime bytecode ~ the function selectors of the dispatcher (heuristic)."""
    b = bytes.fromhex(code[2:]) if code and code.startswith("0x") else b""
    out, i = set(), 0
    while i < len(b):
        op = b[i]
        if op == 0x63 and i + 4 < len(b):  # PUSH4
            out.add("0x" + b[i + 1:i + 5].hex())
        i += 1 + (op - 0x5f if 0x60 <= op <= 0x7f else 0)
    return sorted(out)


# ------------------------------------------------------------------ steps
def step_pairs(es: Etherscan, cfg, raw: Path):
    s = cfg["study"]
    ts = lambda d: int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp())
    b0, b1 = es.block_at(ts(s["start"])), es.block_at(ts(s["end"]))
    cache = raw / "etherscan" / "pair_created.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    start = b0
    done_f = cache.with_suffix(".progress")
    if done_f.exists():  # resume after the last fully crawled block range
        start = int(done_f.read_text()) + 1
    step = 50_000
    with cache.open("a", encoding="utf-8") as f:
        for lo in range(start, b1, step):
            hi = min(lo + step - 1, b1)
            for l in es.logs(lo, hi, C.UNISWAP_V2_FACTORY, C.TOPIC_PAIR_CREATED):
                t0_, t1_ = _topic_addr(l["topics"], 1), _topic_addr(l["topics"], 2)
                if C.WETH not in (t0_, t1_):
                    continue
                f.write(json.dumps(dict(block=int(l["blockNumber"], 16), ts=int(l["timeStamp"], 16),
                                        tx=l["transactionHash"], token0=t0_, token1=t1_,
                                        pair=("0x" + l["data"][26:66]).lower())) + "\n")
            f.flush()
            done_f.write_text(str(hi))
            print(f"blocks {lo:,}-{hi:,} done ({es.calls} calls)")
    df = pd.read_json(cache, lines=True).drop_duplicates("pair")
    print(f"{len(df):,} WETH pairs")


def step_sample(es: Etherscan, cfg, raw: Path):
    E, s = cfg["etherscan"], cfg["study"]
    df = pd.read_json(raw / "etherscan" / "pair_created.jsonl", lines=True).drop_duplicates("pair")
    df["token"] = np.where(df.token0 == C.WETH, df.token1, df.token0)
    df["month"] = pd.to_datetime(df.ts, unit="s").dt.to_period("M")
    rng = np.random.default_rng(E["seed"])
    # oversample: some tokens will be dropped by the token-age filter below
    n = int(E["sample_per_month"] * E["oversample"])
    pick = df.iloc[rng.permutation(len(df))].groupby("month").head(n).reset_index(drop=True)

    cpath = raw / "etherscan" / "creation.json"
    created = json.loads(cpath.read_text()) if cpath.exists() else {}
    todo = [t for t in pick.token.unique() if t not in created]
    for i in range(0, len(todo), 5):
        batch = todo[i:i + 5]
        res = es.get(module="contract", action="getcontractcreation", contractaddresses=",".join(batch))
        got = {r["contractAddress"].lower(): r for r in res}
        for t in batch:
            r = got.get(t)
            created[t] = None if r is None else dict(creator=r.get("contractCreator", "").lower(),
                                                     tx=r.get("txHash"), block=_int(r.get("blockNumber")))
        if i % 500 == 0:
            cpath.write_text(json.dumps(created))
            print(f"creation info {i:,}/{len(todo):,} ({es.calls} calls)")
    # older API responses lack block/timestamp -> look the tx up
    for t, r in created.items():
        if r and not r.get("block") and r.get("tx"):
            tx = es.get(module="proxy", action="eth_getTransactionByHash", txhash=r["tx"])
            r["block"] = _int(tx.get("blockNumber")) if isinstance(tx, dict) else None
    cpath.write_text(json.dumps(created))

    pick["token_deployed_block"] = pick.token.map(
        lambda t: created[t]["block"] if created.get(t) and created[t].get("block") else np.nan).astype("float64")
    pick["token_creator"] = pick.token.map(lambda t: created[t]["creator"] if created.get(t) else None)
    age_blocks = pick.block - pick.token_deployed_block
    max_age = s["max_token_age_days"] * 86400 / BLOCK_S
    pick = pick[(age_blocks >= 0) & (age_blocks <= max_age)]
    pick = pick.groupby("month", group_keys=False).head(E["sample_per_month"])
    pick["token_deployed_ts"] = pick.ts - (pick.block - pick.token_deployed_block) * BLOCK_S
    pairs = pd.DataFrame({
        "pair": pick.pair, "token": pick.token, "token_is_token0": pick.token0 != C.WETH,
        "token0": pick.token0, "token1": pick.token1,
        "created_ts": pd.to_datetime(pick.ts, unit="s", utc=True), "created_block": pick.block,
        "created_tx": pick.tx, "token_deployed_ts": pd.to_datetime(pick.token_deployed_ts, unit="s", utc=True),
        "token_deployed_block": pick.token_deployed_block.astype("int64"), "token_creator": pick.token_creator,
        "cohort_date": pd.to_datetime(pick.ts, unit="s").dt.date,
    })
    atomic_parquet(pairs, raw / "pairs.parquet")
    print(f"sampled {len(pairs):,} pools; per month:\n{pick.month.value_counts().sort_index().to_string()}")


RPC_URL = "https://ethereum-rpc.publicnode.com"  # free public node: takes eth_getCode load off Etherscan
_rpc = threading.local()


def rpc_get_code(address: str, block: int | str = "latest") -> str | None:
    """Bytecode from a public RPC; None on any failure (caller falls back to Etherscan)."""
    if not hasattr(_rpc, "s"):
        _rpc.s = requests.Session()
    for attempt in range(3):
        try:
            tag = hex(block) if isinstance(block, int) else block
            r = _rpc.s.post(RPC_URL, json={"jsonrpc": "2.0", "id": 1, "method": "eth_getCode",
                                           "params": [address, tag]}, timeout=30).json()
            if isinstance(r.get("result"), str):
                return r["result"]
        except (requests.RequestException, ValueError):
            pass
        time.sleep(1 + attempt)
    return None


def _crawl_pool(es: Etherscan, p, s, cache: Path):
    out = cache / f"{p.pair}.parquet"
    if out.exists():
        return
    per_day = 86400 / BLOCK_S * 1.15  # over-cover the block range, then filter by timestamp
    cts = int(p.created_ts.timestamp())
    pool_end_ts = cts + s["pool_horizon_days"] * 86400
    feat_end_ts = cts + s["feature_horizon_days"] * 86400
    rows = []
    # pair address: Sync/Mint/Burn/Swap (pool) + Transfer (LP token), one query
    for l in es.logs(int(p.created_block), int(p.created_block + s["pool_horizon_days"] * per_day), p.pair):
        r = log_row(l, "pool")
        if r is None:
            continue
        if r["event"] == "transfer":
            if r["ts"] <= feat_end_ts:
                rows.append({**r, "role": "lp"})
        elif r["ts"] <= pool_end_ts:
            rows.append(r)
    # token address: Transfer + OwnershipTransferred from deployment
    for l in es.logs(int(p.token_deployed_block), int(p.created_block + s["feature_horizon_days"] * per_day), p.token):
        r = log_row(l, "token")
        if r is not None and r["event"] in ("transfer", "ownership") and r["ts"] <= feat_end_ts:
            rows.append(r)
    block_tag = hex(int(p.created_block))
    code = rpc_get_code(p.token, int(p.created_block))
    historical = bool(code and code != "0x")
    if not historical:
        code = rpc_get_code(p.token) or es.get(module="proxy", action="eth_getCode",
                                                    address=p.token, tag="latest")
    (cache / f"{p.token}.code.json").write_text(json.dumps(
        {"code": code or "0x", "requested_block": block_tag,
         "code_at_prediction_time": historical}))
    df = pd.DataFrame(rows, columns=["address", "role", "ts", "block_number", "transaction_hash", "log_index",
                                     "event", "addr1", "addr2", "v0", "v1", "v2", "v3"])
    tmp = out.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.replace(out)  # the pool counts as done only once its file is complete


def step_pools(es: Etherscan, cfg, raw: Path):
    workers = cfg["etherscan"].get("workers", 16)
    s = cfg["study"]
    pairs = pd.read_parquet(raw / "pairs.parquet")
    cache = raw / "etherscan" / "pools"
    cache.mkdir(parents=True, exist_ok=True)
    todo = [p for p in pairs.itertuples() if not (cache / f"{p.pair}.parquet").exists()]
    print(f"{len(pairs) - len(todo):,} pools already cached, {len(todo):,} to go")
    t0, done, failed = time.time(), 0, 0

    def safe(p):
        try:
            _crawl_pool(es, p, s, cache)
            return True
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001 - skip this pool, a rerun retries it
            print(f"[skip] {p.pair}: {e}")
            return False

    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(safe, p) for p in todo]
        for f in as_completed(futs):
            done += 1
            failed += not f.result()
            if done % 100 == 0:
                rate = done / (time.time() - t0)
                print(f"pool {done:,}/{len(todo):,}  {es.calls:,} calls  {failed} failed  "
                      f"~{(len(todo) - done) / rate / 3600:.1f} h left")
    print(f"finished: {done - failed:,} ok, {failed} failed (rerun to retry them)")


def step_export(cfg, raw: Path):
    import hashlib
    pairs = pd.read_parquet(raw / "pairs.parquet")
    cache = raw / "etherscan" / "pools"
    ev_dir = raw / "events"
    ev_dir.mkdir(exist_ok=True)
    month = dict(zip(pairs.pair, pd.to_datetime(pairs.created_ts).dt.strftime("%Y-%m")))
    frames = {}
    for f in cache.glob("0x*.parquet"):
        frames.setdefault(month.get(f.stem, "unknown"), []).append(pd.read_parquet(f))
    for m, fs in frames.items():
        df = pd.concat(fs, ignore_index=True).drop_duplicates(["transaction_hash", "log_index", "role"])
        atomic_parquet(df, ev_dir / f"{m}.parquet")
        print(f"{m}: {len(df):,} events")
    rows = []
    for t in pairs.token.unique():
        f = cache / f"{t}.code.json"
        if not f.exists():
            continue
        payload = json.loads(f.read_text())
        code = payload["code"]
        rows.append(dict(token=t, function_sighashes=selectors_from_bytecode(code), is_erc20=True,
                         bytecode_size=max(len(code) - 2, 0) // 2,
                         bytecode_hash=hashlib.sha256(code.encode()).hexdigest() if code != "0x" else None,
                         code_at_prediction_time=bool(payload.get("code_at_prediction_time", False))))
    atomic_parquet(pd.DataFrame(rows), raw / "contracts.parquet")
    write_parquet_manifest([raw / "pairs.parquet", raw / "contracts.parquet", *ev_dir.glob("*.parquet")],
                           raw / "manifest.json", root=resolve("."))
    print(f"contracts: {len(rows):,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pairs", "sample", "pools", "export"])
    ap.add_argument("--config", default="experiments/common.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    raw = resolve(cfg["paths"]["raw"])
    raw.mkdir(parents=True, exist_ok=True)
    if a.step == "export":
        return step_export(cfg, raw)
    es = Etherscan(cfg["etherscan"]["calls_per_sec"])
    {"pairs": step_pairs, "sample": step_sample, "pools": step_pools}[a.step](es, cfg, raw)
    print(f"done, {es.calls:,} API calls")


if __name__ == "__main__":
    main()
