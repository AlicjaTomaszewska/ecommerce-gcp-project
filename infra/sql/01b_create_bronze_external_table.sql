CREATE OR REPLACE EXTERNAL TABLE bronze.raw_events_external (
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
  format = "CSV",
  uris = ["gs://ecommerce-bucket-csv-files"],
  skip_leading_rows = 1
);
