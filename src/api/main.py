from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
from google.cloud import bigquery, storage
import random
import datetime
import json
import os
import joblib
import pandas as pd
import numpy as np

app = FastAPI(
    title="Product Demand Forecasting API",
    description="API serving the machine learning model predicting the quantity of products sold in the next month.",
    version="1.0.0",
)

# Initialize BigQuery client
bq_client = bigquery.Client()
GCP_PROJECT = os.environ.get("GCP_PROJECT") or bq_client.project
BQ_LOG_TABLE = f"{GCP_PROJECT}.mlops_monitoring.api_logs"

class DemandPredictionRequest(BaseModel):
    product_id: str = Field(..., description="Unique product identifier")
    brand: str = Field(..., description="Product brand name")
    category: str = Field(..., description="Product category")
    price: float = Field(..., description="Current price of the product")
    prev_month_sales: int = Field(..., description="Number of items sold in the previous month")

class DemandPredictionResponse(BaseModel):
    product_id: str
    predicted_quantity: float
    model_version: str

# Global variable to hold the model
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        bucket_name = f"{GCP_PROJECT}-models"
        model_blob_name = "demand_forecast_model.joblib"
        local_model_path = "/tmp/demand_forecast_model.joblib"
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(model_blob_name)
        blob.download_to_filename(local_model_path)
        
        model = joblib.load(local_model_path)
        print("Model loaded successfully from GCS.")
    except Exception as e:
        print(f"Failed to load model from GCS (using fallback mock): {e}")

def fetch_product_features_from_bq(product_id: str) -> dict:
    """
    Fetch the latest engineered features for a product from BigQuery.
    """
    query = """
    SELECT
        p.product_id,
        p.snapshot_date,
        c.month,
        c.week_of_year,
        c.days_to_end_of_month,
        p.product_category_l1,
        p.product_price,
        p.product_view_count_last_month,
        p.product_cart_count_last_month,
        p.product_purchase_count_last_month,
        p.product_avg_price_last_month,
        p.product_conversion_rate_last_month,
        p.product_unique_buyers_last_month
    FROM `{GCP_PROJECT}.gold.product_features_monthly` p
    LEFT JOIN `{GCP_PROJECT}.gold.dim_calendar` c
        ON p.snapshot_date = c.calendar_date
    WHERE p.product_id = CAST(@product_id AS INT64)
    ORDER BY p.snapshot_date DESC
    LIMIT 1
    """.format(GCP_PROJECT=GCP_PROJECT)
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("product_id", "STRING", product_id)
        ]
    )
    
    query_job = bq_client.query(query, job_config=job_config)
    results = list(query_job.result())
    
    if not results:
        return None
        
    return dict(results[0].items())

def prepare_serving_features(row_dict: dict, expected_features: list) -> pd.DataFrame:
    """
    Align raw BQ features to the model's expected training schema dynamically.
    """
    df = pd.DataFrame([row_dict])
    
    numeric_cols = [
        "product_price",
        "product_view_count_last_month",
        "product_cart_count_last_month",
        "product_purchase_count_last_month",
        "product_avg_price_last_month",
        "product_conversion_rate_last_month",
        "product_unique_buyers_last_month",
        "month",
        "week_of_year",
        "days_to_end_of_month",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    df["price_delta"] = df["product_price"] - df["product_avg_price_last_month"]
    df["view_to_cart_rate"] = df["product_cart_count_last_month"] / (df["product_view_count_last_month"] + 1)
    df["cart_to_purchase_rate"] = df["product_purchase_count_last_month"] / (df["product_cart_count_last_month"] + 1)
    df["had_sales_last_month"] = (df["product_purchase_count_last_month"] > 0).astype(int)
    df["had_views_last_month"] = (df["product_view_count_last_month"] > 0).astype(int)
    df["had_cart_last_month"] = (df["product_cart_count_last_month"] > 0).astype(int)
    
    df["log_sales_last_month"] = np.log1p(df["product_purchase_count_last_month"])
    df["price_delta_pct"] = (df["product_price"] - df["product_avg_price_last_month"]) / (
        df["product_avg_price_last_month"] + 1e-6
    )
    df["buyers_per_purchase"] = df["product_unique_buyers_last_month"] / (
        df["product_purchase_count_last_month"] + 1
    )
    df["sales_x_conversion"] = (
        df["product_purchase_count_last_month"] * df["product_conversion_rate_last_month"]
    )

    cat_l1 = str(row_dict.get("product_category_l1") or "UNKNOWN")
    active_cat_col = f"cat_{cat_l1}"
    
    for col in expected_features:
        if col.startswith("cat_"):
            df[col] = 1.0 if col == active_cat_col else 0.0
        elif col not in df.columns:
            df[col] = 0.0
            
    return df[expected_features]

def log_prediction_to_bq(request_data: dict, prediction_score: float, model_version: str):
    """
    Background task to insert the prediction and input features into BigQuery for data drift monitoring.
    """
    try:
        rows_to_insert = [
            {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                # Map product_id to user_id to avoid schema change in BigQuery
                "user_id": request_data.get("product_id"),
                "session_id": "monthly_forecast_run",
                "request_payload": json.dumps(request_data),
                "prediction_score": prediction_score,
                "model_version": model_version
            }
        ]
        
        # Insert data into BigQuery
        errors = bq_client.insert_rows_json(BQ_LOG_TABLE, rows_to_insert)
        
        if errors:
            print(f"Encountered errors while inserting rows: {errors}")
        else:
            print("Successfully logged prediction to BigQuery.")
    except Exception as e:
        print(f"Failed to log to BigQuery: {e}")

@app.post("/predict/demand", response_model=DemandPredictionResponse)
async def predict_demand(request: DemandPredictionRequest, background_tasks: BackgroundTasks):
    
    bq_features = None
    try:
        bq_features = fetch_product_features_from_bq(request.product_id)
    except Exception as e:
        print(f"Failed to fetch features from BigQuery: {e}")
        
    # Execute prediction
    if model is not None and hasattr(model, "feature_names") and bq_features is not None:
        try:
            features_df = prepare_serving_features(bq_features, model.feature_names)
            predicted_qty = float(model.predict(features_df)[0])
            model_version = "hurdle-xgb-v1"
        except Exception as e:
            print(f"Error during model prediction: {e}. Falling back to mock.")
            predicted_qty = float(round(request.prev_month_sales * random.uniform(0.9, 1.1), 2))
            model_version = "mock-fallback"
    else:
        # Mock prediction based on previous sales + some random variation (e.g. +/- 10%)
        predicted_qty = float(round(request.prev_month_sales * random.uniform(0.9, 1.1), 2))
        model_version = "mock-v1"
    
    # Trigger the background task for BigQuery logging
    background_tasks.add_task(
        log_prediction_to_bq, 
        request_data=request.model_dump(), 
        prediction_score=predicted_qty, 
        model_version=model_version
    )
    
    return DemandPredictionResponse(
        product_id=request.product_id,
        predicted_quantity=predicted_qty,
        model_version=model_version
    )

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Product Demand Forecasting"}
