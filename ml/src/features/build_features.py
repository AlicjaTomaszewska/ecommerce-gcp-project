import numpy as np
import pandas as pd

NUMERIC_COLS = [
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
DROP_COLS = ["product_id", "snapshot_date", "target_next_period"]
FEATURE_SETS = {"base", "funnel", "extended"}


def prepare_features(df: pd.DataFrame, feature_set: str = "extended") -> pd.DataFrame:
    feature_set = feature_set.lower()
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set={feature_set!r}. Expected one of: {sorted(FEATURE_SETS)}")

    df = df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df["target_next_period"] = pd.to_numeric(df["target_next_period"], errors="coerce")
    df = df.dropna(subset=["target_next_period", "snapshot_date"])

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if feature_set in {"funnel", "extended"}:
        df["price_delta"] = df["product_price"] - df["product_avg_price_last_month"]
        df["view_to_cart_rate"] = df["product_cart_count_last_month"] / (df["product_view_count_last_month"] + 1)
        df["cart_to_purchase_rate"] = df["product_purchase_count_last_month"] / (df["product_cart_count_last_month"] + 1)
        df["had_sales_last_month"] = (df["product_purchase_count_last_month"] > 0).astype(int)
        df["had_views_last_month"] = (df["product_view_count_last_month"] > 0).astype(int)
        df["had_cart_last_month"] = (df["product_cart_count_last_month"] > 0).astype(int)

    if feature_set == "extended":
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

    df["product_category_l1"] = df["product_category_l1"].fillna("UNKNOWN").astype(str)
    return pd.get_dummies(df, columns=["product_category_l1"], prefix="cat", dummy_na=False)

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DROP_COLS]
