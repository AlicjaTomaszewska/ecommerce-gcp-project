CREATE SCHEMA IF NOT EXISTS bronze
OPTIONS (
  location = "europe-central2",
  description = "Bronze layer for raw e-commerce events."
);

CREATE SCHEMA IF NOT EXISTS silver
OPTIONS (
  location = "europe-central2",
  description = "Silver layer for cleaned Dataform models."
);

CREATE SCHEMA IF NOT EXISTS gold
OPTIONS (
  location = "europe-central2",
  description = "Gold layer with ML feature martes."
);

CREATE SCHEMA IF NOT EXISTS dataform_assertions
OPTIONS (
  location = "europe-central2",
  description = "Dataform quality assertion results."
);

CREATE SCHEMA IF NOT EXISTS mlops_monitoring
OPTIONS (
  location = "europe-central2",
  description = "MLOps monitoring logs for model predictions."
);