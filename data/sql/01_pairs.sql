-- Uniswap V2 WETH pairs created in [@start_ts, @end_ts) whose non-WETH token was
-- deployed at most @max_token_age_days before the pair (= "new token" scope).
-- Output -> {dataset}.pairs
WITH pc AS (
  SELECT
    block_timestamp AS created_ts,
    block_number    AS created_block,
    transaction_hash AS created_tx,
    CONCAT('0x', SUBSTR(topics[SAFE_OFFSET(1)], 27, 40)) AS token0,
    CONCAT('0x', SUBSTR(topics[SAFE_OFFSET(2)], 27, 40)) AS token1,
    CONCAT('0x', SUBSTR(data, 27, 40))                   AS pair
  FROM `bigquery-public-data.crypto_ethereum.logs`
  WHERE block_timestamp >= @start_ts AND block_timestamp < @end_ts
    AND address = '{FACTORY}'
    AND topics[SAFE_OFFSET(0)] = '{TOPIC_PAIR_CREATED}'
),
weth_pairs AS (
  SELECT
    *,
    IF(token0 = '{WETH}', token1, token0) AS token,
    token0 != '{WETH}' AS token_is_token0
  FROM pc
  WHERE '{WETH}' IN (token0, token1)
),
deploy AS (
  SELECT address, block_timestamp AS token_deployed_ts, block_number AS token_deployed_block
  FROM `bigquery-public-data.crypto_ethereum.contracts`
  WHERE block_timestamp >= TIMESTAMP_SUB(@start_ts, INTERVAL @max_token_age_days DAY)
    AND block_timestamp < @end_ts
),
creator AS (
  -- EOA that sent the deployment tx (NULL when the token was deployed by a factory contract)
  SELECT receipt_contract_address AS address, ANY_VALUE(from_address) AS token_creator
  FROM `bigquery-public-data.crypto_ethereum.transactions`
  WHERE block_timestamp >= TIMESTAMP_SUB(@start_ts, INTERVAL @max_token_age_days DAY)
    AND block_timestamp < @end_ts
    AND receipt_contract_address IS NOT NULL
  GROUP BY 1
)
SELECT
  p.pair, p.token, p.token_is_token0, p.token0, p.token1,
  p.created_ts, p.created_block, p.created_tx,
  d.token_deployed_ts, d.token_deployed_block,
  c.token_creator,
  DATE(p.created_ts) AS cohort_date
FROM weth_pairs p
JOIN deploy d ON d.address = p.token
LEFT JOIN creator c ON c.address = p.token
WHERE TIMESTAMP_DIFF(p.created_ts, d.token_deployed_ts, SECOND) <= @max_token_age_days * 86400
