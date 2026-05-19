INSERT INTO bronze.raw_events (
  event_time_raw,
  event_type_raw,
  product_id_raw,
  category_id_raw,
  category_code_raw,
  brand_raw,
  price_raw,
  user_id_raw,
  user_session_raw,
  _source_uri,
  _ingested_at,
  _load_id,
  _row_number
)
SELECT
  event_time,
  event_type,
  CAST(product_id AS STRING),
  CAST(category_id AS STRING),
  category_code,
  brand,
  CAST(price AS STRING),
  CAST(user_id AS STRING),
  user_session,
  'local://events.csv' AS _source_uri,
  CURRENT_TIMESTAMP() AS _ingested_at,
  'manual_local_load' AS _load_id,
  ROW_NUMBER() OVER (ORDER BY event_time, product_id, user_id, user_session) AS _row_number
FROM
  bronze.raw_events_external;