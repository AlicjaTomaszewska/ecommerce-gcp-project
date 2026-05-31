from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_train_val_bars(metrics_train: dict, metrics_val: dict, title: str, ylabel: str, fname: Path):
    keys = list(metrics_train.keys())
    train_vals = [metrics_train[k] for k in keys]
    val_vals = [metrics_val[k] for k in keys]
    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, train_vals, width, label="Training set", color="#4C72B0")
    ax.bar(x + width / 2, val_vals, width, label="Validation set", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=20)
    ax.set_xlabel("Metric")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)

def plot_models_comparison(results_df: pd.DataFrame, metric_col: str, title: str, ylabel: str, fname: Path):
    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(results_df)), 5))
    x = np.arange(len(results_df))
    width = 0.4
    train_col = f"train_{metric_col}"
    val_col = f"val_{metric_col}"
    ax.bar(x - width / 2, results_df[train_col], width, label="Train", color="#4C72B0")
    ax.bar(x + width / 2, results_df[val_col], width, label="Val", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["run_name"], rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("Model configuration (algorithm + hyperparameters)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)

def plot_pred_vs_true(y_true, y_pred, title: str, fname: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10)
    lim = max(float(np.max(y_true)), float(np.max(y_pred))) * 1.05 if len(y_true) else 1
    ax.plot([0, lim], [0, lim], "r--", label="y = x (perfect fit)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual target_next_period (next-period purchase count)")
    ax.set_ylabel("Model prediction (expected purchase count)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)

def plot_feature_importance(features, importances, title: str, fname: Path, top_n=20):
    df_imp = pd.DataFrame({"feature": features, "importance": importances}).sort_values(
        "importance", ascending=True
    ).tail(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(df_imp))))
    ax.barh(df_imp["feature"], df_imp["importance"], color="#55A868")
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Feature name")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
