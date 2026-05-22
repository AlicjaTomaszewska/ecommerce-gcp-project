CREATE OR REPLACE TABLE bronze.raw_events (
  event_time_raw STRING,
  event_type_raw STRING,
  product_id_raw STRING,
  category_id_raw STRING,
  category_code_raw STRING,
  brand_raw STRING,
  price_raw STRING,
  user_id_raw STRING,
  user_session_raw STRING,
  _source_uri STRING,
  _ingested_at TIMESTAMP,
  _load_id STRING,
  _load_month DATE
)
PARTITION BY _load_month
OPTIONS (
  description = "Raw e-commerce events. One partition per calendar month (_load_month). Reload month = DELETE + INSERT for that partition."
);
