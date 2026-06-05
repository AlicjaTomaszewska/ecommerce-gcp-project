import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)

GCP_PROJECT = os.getenv("GCP_PROJECT", "ecommerce-project-496110")
VAL_LAST_N_SNAPSHOTS = int(os.getenv("VAL_LAST_N_SNAPSHOTS", "1"))
NONZERO_THRESHOLD = float(os.getenv("NONZERO_THRESHOLD", "0.5"))
FEATURE_SET = os.getenv("FEATURE_SET", "extended").lower()

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", PROJECT_ROOT / "artifacts"))
PLOTS_DIR = ARTIFACT_DIR / "plots"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:{PROJECT_ROOT / 'mlruns'}")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "hurdle_product_demand")

BQ_QUERY = """
WITH base AS (
    SELECT
        p.product_id,
        p.snapshot_date,
        c.month,
        c.week_of_year,
        c.days_to_end_of_month,
        p.product_category_l1,
        SAFE_CAST(p.product_price AS FLOAT64)                        AS product_price,
        SAFE_CAST(p.product_view_count_last_month AS FLOAT64)        AS product_view_count_last_month,
        SAFE_CAST(p.product_cart_count_last_month AS FLOAT64)        AS product_cart_count_last_month,
        SAFE_CAST(p.product_purchase_count_last_month AS FLOAT64)    AS product_purchase_count_last_month,
        SAFE_CAST(p.product_avg_price_last_month AS FLOAT64)         AS product_avg_price_last_month,
        SAFE_CAST(p.product_conversion_rate_last_month AS FLOAT64)   AS product_conversion_rate_last_month,
        SAFE_CAST(p.product_unique_buyers_last_month AS FLOAT64)     AS product_unique_buyers_last_month,
        LEAD(product_purchase_count_last_month, 1) OVER (
            PARTITION BY p.product_id ORDER BY p.snapshot_date
        ) AS target_next_period
    FROM `ecommerce-project-496110.gold.product_features_monthly` p
    LEFT JOIN `ecommerce-project-496110.gold.dim_calendar` c
        ON p.snapshot_date = c.calendar_date
)
SELECT * FROM base WHERE target_next_period IS NOT NULL
"""

def ensure_directories() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "preprocessing").mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "predictions").mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "ensemble").mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "final_model").mkdir(parents=True, exist_ok=True)

def setup_mlflow() -> None:
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
