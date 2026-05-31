import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def rmsle(y_true, y_pred):
    y_true = np.clip(np.asarray(y_true, dtype=float), 0, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))

def wape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else np.nan

def regression_metrics_dict(y_true, y_pred, prefix=""):
    return {
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}rmse": rmse(y_true, y_pred),
        f"{prefix}rmsle": rmsle(y_true, y_pred),
        f"{prefix}wape": wape(y_true, y_pred),
    }

def classification_metrics_dict(y_true, y_prob, threshold=0.5, prefix=""):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        f"{prefix}auc_roc": float(roc_auc_score(y_true, y_prob)),
        f"{prefix}pr_auc": float(average_precision_score(y_true, y_prob)),
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
