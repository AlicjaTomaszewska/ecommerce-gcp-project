import itertools

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ml.src.config import ARTIFACT_DIR, NONZERO_THRESHOLD, PLOTS_DIR
from ml.src.metrics.metrics import classification_metrics_dict
from ml.src.visualization.plots import plot_train_val_bars

def build_classifier_configs():
    configs = []
    for C in [0.1, 1.0, 10.0]:
        configs.append({
            "algo": "logreg",
            "params": {"C": C, "max_iter": 1000, "solver": "lbfgs", "random_state": 42},
        })
    for n_est, max_depth, lr in itertools.product([100, 200, 300], [5, 6, 7], [0.05, 0.07, 0.02]):
        configs.append({
            "algo": "xgb",
            "params": {
                "n_estimators": n_est,
                "max_depth": max_depth,
                "learning_rate": lr,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 2.0,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": -1,
            },
        })
    return configs

def fit_classifier(cfg, X_train, y_train, X_val, y_val):
    if cfg["algo"] == "logreg":
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_va = scaler.transform(X_val)
        model = LogisticRegression(**cfg["params"])
        model.fit(X_tr, y_train)
        return model, scaler, model.predict_proba(X_tr)[:, 1], model.predict_proba(X_va)[:, 1]

    model = XGBClassifier(**cfg["params"])
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, None, model.predict_proba(X_train)[:, 1], model.predict_proba(X_val)[:, 1]

def run_stage1_search(X_train, y_train_bin, X_val, y_val_bin, feature_cols, parent_run_id=None):
    results = []
    configs = build_classifier_configs()
    print(f"\n>>> Stage 1: testing {len(configs)} classifier configs")

    for i, cfg in enumerate(configs, 1):
        run_name = f"clf_{cfg['algo']}_{i:02d}"
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.set_tag("stage", "1_classifier")
            mlflow.set_tag("algo", cfg["algo"])
            mlflow.log_params({f"clf_{k}": v for k, v in cfg["params"].items()})

            model, scaler, p_train, p_val = fit_classifier(cfg, X_train, y_train_bin, X_val, y_val_bin)
            train_m = classification_metrics_dict(y_train_bin, p_train, NONZERO_THRESHOLD, prefix="train_")
            val_m = classification_metrics_dict(y_val_bin, p_val, NONZERO_THRESHOLD, prefix="val_")
            mlflow.log_metrics({**train_m, **val_m})

            plot_path = PLOTS_DIR / f"{run_name}_train_vs_val.png"
            plot_train_val_bars(
                {k.replace("train_", ""): v for k, v in train_m.items()},
                {k.replace("val_", ""): v for k, v in val_m.items()},
                title=f"Stage 1 ({cfg['algo']}) - Train vs Val",
                ylabel="Metric value",
                fname=plot_path,
            )
            mlflow.log_artifact(str(plot_path), artifact_path="plots")

            if cfg["algo"] == "xgb":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")
                if scaler is not None:
                    scaler_path = ARTIFACT_DIR / f"{run_name}_scaler.joblib"
                    joblib.dump(scaler, scaler_path)
                    mlflow.log_artifact(str(scaler_path), artifact_path="preprocessing")

            results.append({
                "run_name": run_name,
                "algo": cfg["algo"],
                "params": cfg["params"],
                "model": model,
                "scaler": scaler,
                **train_m,
                **val_m,
            })
            print(f"  [{run_name}] val_auc={val_m['val_auc_roc']:.4f}  val_pr_auc={val_m['val_pr_auc']:.4f}")

    return pd.DataFrame(results), results
