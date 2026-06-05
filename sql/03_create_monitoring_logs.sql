-- Create the api_logs table for ML serving predictions log monitoring
CREATE TABLE IF NOT EXISTS `ecommerce-project-496110.mlops_monitoring.api_logs` (
  timestamp TIMESTAMP OPTIONS(description="Time of prediction request"),
  user_id STRING OPTIONS(description="Identifier for the entity (product_id mapped to user_id to fit schema)"),
  session_id STRING OPTIONS(description="Identifier for the session/batch"),
  request_payload STRING OPTIONS(description="Full input payload in JSON format"),
  prediction_score FLOAT64 OPTIONS(description="Predicted target demand value"),
  model_version STRING OPTIONS(description="Version of the model that served the prediction")
)
PARTITION BY DATE(timestamp)
OPTIONS (
  description = "Logs of predictions served by the API for drift monitoring and feedback loop."
);
