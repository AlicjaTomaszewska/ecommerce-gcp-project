-- Replace @LOAD_MONTH (YYYY-MM) and @GCS_URI before running.
-- Example URI: gs://ecommerce-bucket-csv-files/events/month=2020-10/events_2020-10.csv

DECLARE load_month STRING DEFAULT '2020-11';
DECLARE gcs_uri STRING DEFAULT 'gs://ecommerce-bucket-csv-files/events_2020-11.csv';

EXECUTE IMMEDIATE FORMAT("""
CREATE OR REPLACE EXTERNAL TABLE bronze.raw_events_external_%s (
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
""", REPLACE(load_month, '-', '_'), gcs_uri);
