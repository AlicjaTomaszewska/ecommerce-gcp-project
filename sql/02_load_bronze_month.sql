-- Monthly bronze load (idempotent): set load_month and source URI, then run.
-- load_month: calendar month in the file (YYYY-MM).
-- snapshot_date for Dataform after this load: first day of the next month
-- (e.g. load_month 2020-10 -> snapshot_date 2020-11-01).

DECLARE load_month STRING DEFAULT '2020-10';
DECLARE gcs_uri STRING DEFAULT 'gs://ecommerce-bucket-csv-files/events/month=2020-10/events_2020-10.csv';
DECLARE load_month_date DATE DEFAULT PARSE_DATE('%Y-%m', load_month);
DECLARE load_id STRING DEFAULT CONCAT('monthly_', load_month);
DECLARE external_table STRING DEFAULT CONCAT('raw_events_external_', REPLACE(load_month, '-', '_'));

EXECUTE IMMEDIATE FORMAT("""
CREATE OR REPLACE EXTERNAL TABLE bronze.%s (
  event_time STRING,
  event_type STRING,
  product_id STRING,
  category_id STRING,
  category_code STRING,
  brand STRING,
  price STRING,
  user_id STRING,
  user_session STRING
)
OPTIONS (
  format = 'CSV',
  uris = ['%s'],
  skip_leading_rows = 1
)
""", external_table, gcs_uri);

DELETE FROM bronze.raw_events
WHERE _load_month = load_month_date;

EXECUTE IMMEDIATE FORMAT("""
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
  _load_month
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
  '%s' AS _source_uri,
  CURRENT_TIMESTAMP() AS _ingested_at,
  '%s' AS _load_id,
  DATE('%s') AS _load_month
FROM
  bronze.%s
""",
  gcs_uri,
  load_id,
  CAST(load_month_date AS STRING),
  external_table);

-- Optional cleanup of the temporary external table:
-- EXECUTE IMMEDIATE FORMAT('DROP TABLE IF EXISTS bronze.%s', external_table);
