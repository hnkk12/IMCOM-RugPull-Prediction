-- One scan of the logs table for every event we need, restricted per address
-- to the time window that labeling/feature extraction will ever look at:
--   pool  (pair address): Sync/Mint/Burn/Swap   in [created, created + @pool_horizon_days]
--   lp    (pair address): Transfer (LP token)   in [created, created + @feature_horizon_days]
--   token (token address): Transfer, OwnershipTransferred
--                                               in [deployed, created + @feature_horizon_days]
-- The HEXF macro is expanded by src/bq_fetch.py into an exact-enough uint256 -> FLOAT64 conversion.
-- Output -> {dataset}.events (partitioned by cohort month, clustered by address)
WITH pool_t AS (
  SELECT pair AS address, created_ts AS win_start,
         TIMESTAMP_ADD(created_ts, INTERVAL @pool_horizon_days DAY) AS pool_end,
         TIMESTAMP_ADD(created_ts, INTERVAL @feature_horizon_days DAY) AS feat_end,
         cohort_date
  FROM `{DATASET}.pairs`
),
token_t AS (
  SELECT token AS address,
         MIN(token_deployed_ts) AS win_start,
         MAX(TIMESTAMP_ADD(created_ts, INTERVAL @feature_horizon_days DAY)) AS feat_end,
         MIN(cohort_date) AS cohort_date
  FROM `{DATASET}.pairs`
  GROUP BY 1
),
logs AS (
  SELECT
    address, block_timestamp, block_number, transaction_hash, log_index,
    topics[SAFE_OFFSET(0)] AS topic0,
    CONCAT('0x', SUBSTR(topics[SAFE_OFFSET(1)], 27, 40)) AS addr1,
    CONCAT('0x', SUBSTR(topics[SAFE_OFFSET(2)], 27, 40)) AS addr2,
    ARRAY_LENGTH(topics) AS n_topics,
    SUBSTR(data, 3, 64)   AS w0,
    SUBSTR(data, 67, 64)  AS w1,
    SUBSTR(data, 131, 64) AS w2,
    SUBSTR(data, 195, 64) AS w3
  FROM `bigquery-public-data.crypto_ethereum.logs`
  WHERE block_timestamp >= TIMESTAMP_SUB(@start_ts, INTERVAL @max_token_age_days DAY)
    AND block_timestamp < TIMESTAMP_ADD(@end_ts, INTERVAL (@pool_horizon_days + 1) DAY)
    AND topics[SAFE_OFFSET(0)] IN ('{TOPIC_SYNC}', '{TOPIC_MINT}', '{TOPIC_BURN}',
                                   '{TOPIC_SWAP}', '{TOPIC_TRANSFER}', '{TOPIC_OWNERSHIP}')
),
tagged AS (
  SELECT l.*, 'pool' AS role, t.cohort_date
  FROM logs l JOIN pool_t t ON l.address = t.address
  WHERE l.topic0 IN ('{TOPIC_SYNC}', '{TOPIC_MINT}', '{TOPIC_BURN}', '{TOPIC_SWAP}')
    AND l.block_timestamp BETWEEN t.win_start AND t.pool_end
  UNION ALL
  SELECT l.*, 'lp' AS role, t.cohort_date
  FROM logs l JOIN pool_t t ON l.address = t.address
  WHERE l.topic0 = '{TOPIC_TRANSFER}' AND l.n_topics = 3
    AND l.block_timestamp BETWEEN t.win_start AND t.feat_end
  UNION ALL
  SELECT l.*, 'token' AS role, t.cohort_date
  FROM logs l JOIN token_t t ON l.address = t.address
  WHERE ((l.topic0 = '{TOPIC_TRANSFER}' AND l.n_topics = 3) OR l.topic0 = '{TOPIC_OWNERSHIP}')
    AND l.block_timestamp BETWEEN t.win_start AND t.feat_end
)
SELECT
  address, role, cohort_date,
  UNIX_SECONDS(block_timestamp) AS ts, block_number, transaction_hash, log_index,
  CASE topic0
    WHEN '{TOPIC_SYNC}' THEN 'sync' WHEN '{TOPIC_MINT}' THEN 'mint'
    WHEN '{TOPIC_BURN}' THEN 'burn' WHEN '{TOPIC_SWAP}' THEN 'swap'
    WHEN '{TOPIC_TRANSFER}' THEN 'transfer' ELSE 'ownership' END AS event,
  addr1, addr2,
  HEXF(w0) AS v0, HEXF(w1) AS v1, HEXF(w2) AS v2, HEXF(w3) AS v3
FROM tagged
