import itertools

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from ml.src.config import ARTIFACT_DIR, PLOTS_DIR
from ml.src.metrics.metrics import regression_metrics_dict
from ml.src.visualization.plots import plot_pred_vs_true, plot_train_val_bars

def build_regressor_configs():
    configs = []
    for alpha, max_iter in itertools.product([0.1, 1.0, 5.0], [500, 800]):
        configs.append({"algo": "poisson", "params": {"alpha": alpha, "max_iter": max_iter}})
    for n_est, max_depth, lr in itertools.product([300, 600, 80], [4, 6, 8], [0.03, 0.05, 0.001]):
        configs.append({
            "algo": "xgb",
            "params": {
                "n_estimators": n_est,
                "max_depth": max_depth,
                "learning_rate": lr,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 2.0,
                "objective": "count:poisson",
                "eval_metric": "poisson-nloglik",
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": -1,
            },
        })
    return configs

def fit_regressor(cfg, X_train_pos, y_train_pos, X_val_pos, y_val_pos):
    weights_train = 1.0 + (y_train_pos / np.mean(y_train_pos))
    if cfg["algo"] == "poisson":
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_pos)
        X_va = scaler.transform(X_val_pos)
        model = PoissonRegressor(**cfg["params"])
        model.fit(X_tr, y_train_pos, sample_weight=weights_train)
        return model, scaler, model.predict(X_tr), model.predict(X_va)

    model = XGBRegressor(**cfg["params"])
    model.fit(
        X_train_pos,
        y_train_pos,
        sample_weight=weights_train,
        eval_set=[(X_val_pos, y_val_pos)],
        verbose=False,
    )
    return model, None, model.predict(X_train_pos), model.predict(X_val_pos)

def run_stage2_search(X_train_pos, y_train_pos, X_val_pos, y_val_pos, feature_cols, parent_run_id=None):
    results = []
    configs = build_regressor_configs()
    print(f"\n>>> Stage 2: testing {len(configs)} regressor configs")

    for i, cfg in enumerate(configs, 1):
        run_name = f"reg_{cfg['algo']}_{i:02d}"
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.set_tag("stage", "2_regressor")
            mlflow.set_tag("algo", cfg["algo"])
            mlflow.log_params({f"reg_{k}": v for k, v in cfg["params"].items()})

            model, scaler, pred_train, pred_val = fit_regressor(cfg, X_train_pos, y_train_pos, X_val_pos, y_val_pos)
            pred_train = np.clip(pred_train, 0, None)
            pred_val = np.clip(pred_val, 0, None)
            train_m = regression_metrics_dict(y_train_pos, pred_train, prefix="train_")
            val_m = regression_metrics_dict(y_val_pos, pred_val, prefix="val_")
            mlflow.log_metrics({**train_m, **val_m})

            plot_path = PLOTS_DIR / f"{run_name}_train_vs_val.png"
            plot_train_val_bars(
                {k.replace("train_", ""): v for k, v in train_m.items()},
                {k.replace("val_", ""): v for k, v in val_m.items()},
                title=f"Stage 2 ({cfg['algo']}) - Train vs Val (regression on positive samples)",
                ylabel="Metric value (lower is better)",
                fname=plot_path,
            )
            mlflow.log_artifact(str(plot_path), artifact_path="plots")

            scatter_path = PLOTS_DIR / f"{run_name}_pred_vs_true_val.png"
            plot_pred_vs_true(y_val_pos, pred_val, title=f"Stage 2 ({cfg['algo']}) - Prediction vs actual (val)", fname=scatter_path)
            mlflow.log_artifact(str(scatter_path), artifact_path="plots")

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
            print(f"  [{run_name}] val_rmse={val_m['val_rmse']:.4f}  val_wape={val_m['val_wape']:.4f}")

    return pd.DataFrame(results), results
