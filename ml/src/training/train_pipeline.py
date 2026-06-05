import warnings

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from ml.src.config import (
    ARTIFACT_DIR,
    FEATURE_SET,
    NONZERO_THRESHOLD,
    PLOTS_DIR,
    VAL_LAST_N_SNAPSHOTS,
    ensure_directories,
    setup_mlflow,
)
from ml.src.data.bigquery_loader import load_data
from ml.src.features.build_features import get_feature_cols, prepare_features
from ml.src.metrics.metrics import regression_metrics_dict
from ml.src.models.classifier import run_stage1_search
from ml.src.models.hurdle import HurdleModel
from ml.src.models.regressor import run_stage2_search
from ml.src.split.snapshot_split import split_by_snapshots
from ml.src.visualization.plots import plot_feature_importance, plot_models_comparison, plot_pred_vs_true

warnings.filterwarnings("ignore")

def main():
    ensure_directories()
    setup_mlflow()

    with mlflow.start_run(run_name="hurdle_pipeline") as parent_run:
        mlflow.set_tag("pipeline", "hurdle")
        mlflow.log_param("val_last_n_snapshots", VAL_LAST_N_SNAPSHOTS)
        mlflow.log_param("nonzero_threshold", NONZERO_THRESHOLD)
        mlflow.log_param("feature_set", FEATURE_SET)
        mlflow.set_tag("feature_set", FEATURE_SET)

        df = prepare_features(load_data(), feature_set=FEATURE_SET)
        feature_cols = get_feature_cols(df)
        train_df, val_df, train_dates, val_dates = split_by_snapshots(
            df, "snapshot_date", VAL_LAST_N_SNAPSHOTS
        )

        X_train = train_df[feature_cols].copy()
        y_train = train_df["target_next_period"].copy()
        X_val = val_df[feature_cols].copy()
        y_val = val_df["target_next_period"].copy()

        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val", len(X_val))
        mlflow.log_param("train_dates", [str(pd.Timestamp(d).date()) for d in train_dates])
        mlflow.log_param("val_dates", [str(pd.Timestamp(d).date()) for d in val_dates])

        print(f"\nTrain size: {len(X_train):,} | Val size: {len(X_val):,}")
        print(f"Train zeros: {(y_train == 0).mean():.1%}, nonzeros: {(y_train > 0).mean():.1%}")

        y_train_bin = (y_train > 0).astype(int)
        y_val_bin = (y_val > 0).astype(int)
        clf_results_df, clf_results = run_stage1_search(
            X_train, y_train_bin, X_val, y_val_bin, feature_cols, parent_run.info.run_id
        )
        clf_results_df = clf_results_df.sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
        clf_results_df.drop(columns=["model", "scaler", "params"]).to_csv(
            ARTIFACT_DIR / "stage1_results.csv", index=False
        )
        mlflow.log_artifact(str(ARTIFACT_DIR / "stage1_results.csv"))

        cmp1_path = PLOTS_DIR / "stage1_models_comparison_pr_auc.png"
        plot_models_comparison(
            clf_results_df,
            metric_col="pr_auc",
            title="Stage 1 - Model comparison (PR-AUC)",
            ylabel="PR-AUC (higher is better)",
            fname=cmp1_path,
        )
        mlflow.log_artifact(str(cmp1_path), artifact_path="plots")

        cmp1b_path = PLOTS_DIR / "stage1_models_comparison_auc.png"
        plot_models_comparison(
            clf_results_df,
            metric_col="auc_roc",
            title="Stage 1 - Model comparison (ROC-AUC)",
            ylabel="ROC-AUC (higher is better)",
            fname=cmp1b_path,
        )
        mlflow.log_artifact(str(cmp1b_path), artifact_path="plots")

        best_clf_row = clf_results_df.iloc[0]
        best_clf_entry = next(r for r in clf_results if r["run_name"] == best_clf_row["run_name"])
        print(f"\n>>> Best classifier: {best_clf_row['run_name']} | val_pr_auc={best_clf_row['val_pr_auc']:.4f}")
        mlflow.log_param("best_classifier_run", best_clf_row["run_name"])
        mlflow.log_metric("best_clf_val_pr_auc", best_clf_row["val_pr_auc"])
        mlflow.log_metric("best_clf_val_auc_roc", best_clf_row["val_auc_roc"])

        train_pos = y_train > 0
        val_pos = y_val > 0
        X_train_pos = X_train[train_pos]
        y_train_pos = y_train[train_pos].values
        X_val_pos = X_val[val_pos]
        y_val_pos = y_val[val_pos].values
        print(f"\nTrain positives: {train_pos.sum():,} | Val positives: {val_pos.sum():,}")

        reg_results_df, reg_results = run_stage2_search(
            X_train_pos, y_train_pos, X_val_pos, y_val_pos, feature_cols, parent_run.info.run_id
        )
        reg_results_df = reg_results_df.sort_values("val_rmse", ascending=True).reset_index(drop=True)
        reg_results_df.drop(columns=["model", "scaler", "params"]).to_csv(
            ARTIFACT_DIR / "stage2_results.csv", index=False
        )
        mlflow.log_artifact(str(ARTIFACT_DIR / "stage2_results.csv"))

        cmp2_path = PLOTS_DIR / "stage2_models_comparison_rmse.png"
        plot_models_comparison(
            reg_results_df,
            metric_col="rmse",
            title="Stage 2 - Model comparison (RMSE)",
            ylabel="RMSE (lower is better)",
            fname=cmp2_path,
        )
        mlflow.log_artifact(str(cmp2_path), artifact_path="plots")

        cmp2b_path = PLOTS_DIR / "stage2_models_comparison_wape.png"
        plot_models_comparison(
            reg_results_df,
            metric_col="wape",
            title="Stage 2 - Model comparison (WAPE)",
            ylabel="WAPE (lower is better)",
            fname=cmp2b_path,
        )
        mlflow.log_artifact(str(cmp2b_path), artifact_path="plots")

        best_reg_row = reg_results_df.iloc[0]
        best_reg_entry = next(r for r in reg_results if r["run_name"] == best_reg_row["run_name"])
        print(f"\n>>> Best regressor: {best_reg_row['run_name']} | val_rmse={best_reg_row['val_rmse']:.4f}")
        mlflow.log_param("best_regressor_run", best_reg_row["run_name"])
        mlflow.log_metric("best_reg_val_rmse", best_reg_row["val_rmse"])
        mlflow.log_metric("best_reg_val_wape", best_reg_row["val_wape"])

        hurdle = HurdleModel(
            classifier=best_clf_entry["model"],
            classifier_scaler=best_clf_entry["scaler"],
            regressor=best_reg_entry["model"],
            regressor_scaler=best_reg_entry["scaler"],
            feature_names=feature_cols,
        )

        val_p_nonzero = hurdle.predict_proba_nonzero(X_val)
        val_pred_soft = hurdle.predict(X_val)
        val_pred_hard = hurdle.predict(X_val, threshold=NONZERO_THRESHOLD)

        soft_m = regression_metrics_dict(y_val, val_pred_soft, prefix="hurdle_soft_val_")
        hard_m = regression_metrics_dict(y_val, val_pred_hard, prefix="hurdle_hard_val_")
        baseline_pred = val_df["product_purchase_count_last_month"].values
        base_m = regression_metrics_dict(y_val, baseline_pred, prefix="baseline_val_")
        mlflow.log_metrics({**soft_m, **hard_m, **base_m})

        print("\n=== Hurdle soft ===", soft_m)
        print("=== Hurdle hard ===", hard_m)
        print("=== Baseline    ===", base_m)

        blend_weights = [round(i / 20, 2) for i in range(0, 21)]
        blend_rows = []
        best_blend = {"w": None, "rmse": float("inf")}
        print("\n>>> Ensemble blend search (hurdle_hard vs baseline):")
        for w_hurdle in blend_weights:
            blend = np.clip(w_hurdle * val_pred_hard + (1.0 - w_hurdle) * baseline_pred, 0, None)
            bm = regression_metrics_dict(y_val, blend, prefix="")
            blend_rows.append({"w_hurdle": w_hurdle, **bm})
            if bm["rmse"] < best_blend["rmse"]:
                best_blend = {"w": w_hurdle, **bm}

        blend_df = pd.DataFrame(blend_rows)
        blend_csv = ARTIFACT_DIR / "ensemble_blend_search.csv"
        blend_df.to_csv(blend_csv, index=False)
        mlflow.log_artifact(str(blend_csv), artifact_path="ensemble")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax_b, metric, color, ylabel in zip(
            axes,
            ["rmse", "wape"],
            ["#4C72B0", "#DD8452"],
            ["RMSE (lower is better)", "WAPE (lower is better)"],
        ):
            ax_b.plot(blend_df["w_hurdle"], blend_df[metric], marker="o", color=color)
            ax_b.axvline(best_blend["w"], color="red", linestyle="--", label=f"best w={best_blend['w']}")
            ax_b.set_xlabel("Hurdle weight (w) | Baseline weight = 1 - w")
            ax_b.set_ylabel(ylabel)
            ax_b.set_title(f"Ensemble blend - {metric.upper()} vs Hurdle weight")
            ax_b.legend()
            ax_b.grid(alpha=0.3)
        fig.tight_layout()
        blend_plot = PLOTS_DIR / "ensemble_blend_search.png"
        fig.savefig(blend_plot, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(str(blend_plot), artifact_path="plots")

        val_pred_blend = np.clip(best_blend["w"] * val_pred_hard + (1.0 - best_blend["w"]) * baseline_pred, 0, None)
        blend_m = regression_metrics_dict(y_val, val_pred_blend, prefix="ensemble_val_")
        mlflow.log_metrics(blend_m)
        mlflow.log_param("best_blend_w_hurdle", best_blend["w"])
        print(f"\n>>> Best blend: w_hurdle={best_blend['w']} | rmse={best_blend['rmse']:.4f} | wape={best_blend.get('wape', float('nan')):.4f}")
        print("=== Ensemble (best blend) ===", blend_m)

        cmp_final_path = PLOTS_DIR / "hurdle_vs_baseline.png"
        labels = ["MAE", "RMSE", "RMSLE", "WAPE"]
        soft_vals = [soft_m[f"hurdle_soft_val_{m.lower()}"] for m in labels]
        hard_vals = [hard_m[f"hurdle_hard_val_{m.lower()}"] for m in labels]
        base_vals = [base_m[f"baseline_val_{m.lower()}"] for m in labels]
        blend_vals = [blend_m[f"ensemble_val_{m.lower()}"] for m in labels]
        x = np.arange(len(labels))
        bw = 0.2
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - 1.5 * bw, soft_vals, bw, label="Hurdle (soft)", color="#4C72B0")
        ax.bar(x - 0.5 * bw, hard_vals, bw, label=f"Hurdle (hard, thr={NONZERO_THRESHOLD})", color="#DD8452")
        ax.bar(x + 0.5 * bw, base_vals, bw, label="Baseline (last month)", color="#8C8C8C")
        ax.bar(x + 1.5 * bw, blend_vals, bw, label=f"Ensemble (w={best_blend['w']})", color="#55A868")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Metric regresji")
        ax.set_ylabel("Metric value (lower is better)")
        ax.set_title("Final Hurdle model vs Baseline vs Ensemble on validation set")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(cmp_final_path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(str(cmp_final_path), artifact_path="plots")

        scatter_final = PLOTS_DIR / "hurdle_pred_vs_true_val.png"
        plot_pred_vs_true(y_val.values, val_pred_soft, title="Hurdle (soft) - Prediction vs actual (val)", fname=scatter_final)
        mlflow.log_artifact(str(scatter_final), artifact_path="plots")

        if hasattr(best_clf_entry["model"], "feature_importances_"):
            fi_path = PLOTS_DIR / "feature_importance_classifier.png"
            plot_feature_importance(feature_cols, best_clf_entry["model"].feature_importances_, "Feature importance - classifier (Stage 1)", fi_path)
            mlflow.log_artifact(str(fi_path), artifact_path="plots")

        if hasattr(best_reg_entry["model"], "feature_importances_"):
            fi_path = PLOTS_DIR / "feature_importance_regressor.png"
            plot_feature_importance(feature_cols, best_reg_entry["model"].feature_importances_, "Feature importance - regressor (Stage 2)", fi_path)
            mlflow.log_artifact(str(fi_path), artifact_path="plots")

        model_path = ARTIFACT_DIR / "hurdle_model.joblib"
        joblib.dump(hurdle, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="final_model")

        val_out = val_df[["product_id", "snapshot_date", "target_next_period"]].copy()
        val_out["p_nonzero"] = val_p_nonzero
        val_out["pred_soft"] = val_pred_soft
        val_out["pred_hard"] = val_pred_hard
        preds_path = ARTIFACT_DIR / "hurdle_val_predictions.csv"
        val_out.to_csv(preds_path, index=False)
        mlflow.log_artifact(str(preds_path), artifact_path="predictions")

        print("\n>>> Pipeline finished. MLflow UI: `mlflow ui --backend-store-uri file:./ml/mlruns`")

if __name__ == "__main__":
    main()
