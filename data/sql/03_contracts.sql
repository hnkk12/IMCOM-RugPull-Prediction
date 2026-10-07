-- Static contract info for every token in {dataset}.pairs (setup D).
-- Output -> {dataset}.contracts
SELECT
  c.address AS token,
  c.function_sighashes,
  c.is_erc20,
  DIV(LENGTH(c.bytecode) - 2, 2) AS bytecode_size,
  TO_HEX(SHA256(c.bytecode)) AS bytecode_hash
FROM `bigquery-public-data.crypto_ethereum.contracts` c
JOIN (SELECT DISTINCT token FROM `{DATASET}.pairs`) p ON p.token = c.address
WHERE c.block_timestamp >= TIMESTAMP_SUB(@start_ts, INTERVAL @max_token_age_days DAY)
  AND c.block_timestamp < @end_ts
