"""Crawl Uniswap V2 data from BigQuery `crypto_ethereum` into data/raw/*.parquet.

Steps (run in order, each is idempotent):
    python -m src.bq_fetch pairs      # -> {dataset}.pairs
    python -m src.bq_fetch events     # -> {dataset}.events   (one big logs scan)
    python -m src.bq_fetch contracts  # -> {dataset}.contracts
    python -m src.bq_fetch export     # -> data/raw/pairs.parquet, contracts.parquet, events/*.parquet

Every query is dry-run first; if it would scan more than bigquery.max_gb_per_query
the script stops unless you pass --yes.
"""
from __future__ import annotations

import argparse
import os
import re
from datetime import datetime, timezone

from src import constants as C
from src.artifacts import write_parquet_manifest
from src.config import ROOT, load_config, resolve

SQL_DIR = ROOT / "data" / "sql"


def hexf(word: str) -> str:
    """SQL expression: 64-hex-char uint256 word -> FLOAT64 (5 chunks, each fits INT64)."""
    parts = [(1, 4, 240), (5, 15, 180), (20, 15, 120), (35, 15, 60), (50, 15, 0)]
    terms = [f"SAFE_CAST(CONCAT('0x', SUBSTR({word}, {s}, {n})) AS INT64) * POW(2.0, {e})"
             for s, n, e in parts]
    return "(" + " + ".join(terms) + ")"


def render(name: str, dataset: str) -> str:
    sql = (SQL_DIR / name).read_text(encoding="utf-8")
    sql = re.sub(r"HEXF\((\w+)\)", lambda m: hexf(m.group(1)), sql)
    subs = {
        "FACTORY": C.UNISWAP_V2_FACTORY, "WETH": C.WETH, "DATASET": dataset,
        "TOPIC_PAIR_CREATED": C.TOPIC_PAIR_CREATED, "TOPIC_SYNC": C.TOPIC_SYNC,
        "TOPIC_MINT": C.TOPIC_MINT, "TOPIC_BURN": C.TOPIC_BURN, "TOPIC_SWAP": C.TOPIC_SWAP,
        "TOPIC_TRANSFER": C.TOPIC_TRANSFER, "TOPIC_OWNERSHIP": C.TOPIC_OWNERSHIP,
    }
    for k, v in subs.items():
        sql = sql.replace("{" + k + "}", v)
    return sql


def _params(cfg):
    from google.cloud import bigquery
    s = cfg["study"]
    ts = lambda d: datetime.fromisoformat(d).replace(tzinfo=timezone.utc)
    return [
        bigquery.ScalarQueryParameter("start_ts", "TIMESTAMP", ts(s["start"])),
        bigquery.ScalarQueryParameter("end_ts", "TIMESTAMP", ts(s["end"])),
        bigquery.ScalarQueryParameter("max_token_age_days", "INT64", s["max_token_age_days"]),
        bigquery.ScalarQueryParameter("pool_horizon_days", "INT64", s["pool_horizon_days"]),
        bigquery.ScalarQueryParameter("feature_horizon_days", "INT64", s["feature_horizon_days"]),
    ]


class BQ:
    def __init__(self, cfg, assume_yes: bool, dry_only: bool = False):
        from google.cloud import bigquery
        self.bq = bigquery
        b = cfg["bigquery"]
        project = b.get("project") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise SystemExit("Set bigquery.project in experiments/common.yaml or env GOOGLE_CLOUD_PROJECT")
        self.client = bigquery.Client(project=project, location=b["location"])
        self.dataset = f"{project}.{b['dataset']}"
        self.client.create_dataset(self.dataset, exists_ok=True)
        self.max_gb = b["max_gb_per_query"]
        self.assume_yes = assume_yes
        self.dry_only = dry_only
        self.params = _params(cfg)

    def run(self, sql: str, dest: str | None = None, partition_field: str | None = None,
            cluster: list[str] | None = None):
        bq = self.bq
        params = [p for p in self.params if re.search(rf"@{p.name}\b", sql)]
        dry = self.client.query(sql, job_config=bq.QueryJobConfig(
            dry_run=True, use_query_cache=False, query_parameters=params))
        gb = dry.total_bytes_processed / 1e9
        print(f"[dry-run] {gb:,.1f} GB will be scanned" + (f" -> {dest}" if dest else ""))
        if self.dry_only:
            return None
        if gb > self.max_gb and not self.assume_yes:
            raise SystemExit(f"Above max_gb_per_query={self.max_gb}. Narrow study dates or rerun with --yes.")
        jc = bq.QueryJobConfig(query_parameters=params)
        if dest:
            jc.destination = f"{self.dataset}.{dest}"
            jc.write_disposition = "WRITE_TRUNCATE"
            if partition_field:
                jc.time_partitioning = bq.TimePartitioning(type_=bq.TimePartitioningType.MONTH,
                                                           field=partition_field)
            if cluster:
                jc.clustering_fields = cluster
        job = self.client.query(sql, job_config=jc)
        res = job.result()
        print(f"[done] billed {(job.total_bytes_billed or 0) / 1e9:,.1f} GB")
        return res


def export(bqc: BQ, raw_dir):
    import pyarrow.parquet as pq
    raw_dir.mkdir(parents=True, exist_ok=True)
    ds = bqc.dataset
    for t in ["pairs", "contracts"]:
        tbl = bqc.client.query(f"SELECT * FROM `{ds}.{t}`").result().to_arrow()
        pq.write_table(tbl, raw_dir / f"{t}.parquet")
        print(f"{t}: {tbl.num_rows:,} rows")
    ev_dir = raw_dir / "events"
    ev_dir.mkdir(exist_ok=True)
    months = [r.m for r in bqc.client.query(
        f"SELECT DISTINCT DATE_TRUNC(cohort_date, MONTH) AS m FROM `{ds}.events` ORDER BY m").result()]
    for m in months:
        out = ev_dir / f"{m:%Y-%m}.parquet"
        if out.exists():
            print(f"skip {out.name}")
            continue
        tbl = bqc.client.query(
            f"SELECT * EXCEPT(cohort_date) FROM `{ds}.events` "
            f"WHERE DATE_TRUNC(cohort_date, MONTH) = DATE '{m:%Y-%m-%d}'").result().to_arrow()
        pq.write_table(tbl, out)
        print(f"{out.name}: {tbl.num_rows:,} rows")
    write_parquet_manifest([raw_dir / "pairs.parquet", raw_dir / "contracts.parquet", *ev_dir.glob("*.parquet")],
                           raw_dir / "manifest.json", root=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pairs", "events", "contracts", "export", "sql"])
    ap.add_argument("--config", default="experiments/common.yaml")
    ap.add_argument("--yes", action="store_true", help="allow queries above max_gb_per_query")
    ap.add_argument("--dry-run", action="store_true", help="only report how many GB the query would scan")
    a = ap.parse_args()
    cfg = load_config(a.config)

    if a.step == "sql":  # print rendered SQL, e.g. to paste into the BigQuery console
        for f in ["01_pairs.sql", "02_events.sql", "03_contracts.sql"]:
            print(f"-- ===== {f} =====\n{render(f, '<project>.' + cfg['bigquery']['dataset'])}")
        return

    bqc = BQ(cfg, a.yes, a.dry_run)
    if a.step == "pairs":
        bqc.run(render("01_pairs.sql", bqc.dataset), "pairs")
    elif a.step == "events":
        bqc.run(render("02_events.sql", bqc.dataset), "events",
                partition_field="cohort_date", cluster=["address", "role"])
    elif a.step == "contracts":
        bqc.run(render("03_contracts.sql", bqc.dataset), "contracts")
    elif a.step == "export":
        export(bqc, resolve(cfg["paths"]["raw"]))


if __name__ == "__main__":
    main()
