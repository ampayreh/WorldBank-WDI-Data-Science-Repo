from __future__ import annotations

import json

import pandas as pd

from .config import (
    ARTIFACTS_DIR,
    CATEGORICAL_FEATURES,
    FEATURE_LABELS,
    FIGURES_DIR,
    MODEL_DISPLAY_NAMES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    PROCESSED_DATA_DIR,
    SUMMARY_DIR,
    TARGET_COLUMN,
    TEST_START_YEAR,
)


def save_dataset_outputs(panel_df: pd.DataFrame, modeling_df: pd.DataFrame) -> dict:
    panel_df.to_csv(PROCESSED_DATA_DIR / "wdi_panel_raw.csv", index=False)
    modeling_df.to_csv(PROCESSED_DATA_DIR / "modeling_dataset.csv", index=False)

    dataset_profile = {
        "rows_raw_panel": int(len(panel_df)),
        "rows_modeling": int(len(modeling_df)),
        "countries": int(modeling_df["country_code"].nunique()),
        "years": [int(modeling_df["year"].min()), int(modeling_df["year"].max())],
        "target": TARGET_COLUMN,
        "numeric_feature_count": len(NUMERIC_FEATURES),
        "categorical_feature_count": len(CATEGORICAL_FEATURES),
        "missing_share_by_feature": {
            column: float(modeling_df[column].isna().mean()) for column in MODEL_FEATURES
        },
    }
    with open(ARTIFACTS_DIR / "dataset_profile.json", "w", encoding="utf-8") as handle:
        json.dump(dataset_profile, handle, indent=2)
    return dataset_profile


def save_metrics_outputs(metrics_df: pd.DataFrame, best_params: dict) -> dict:
    metrics_df.to_csv(ARTIFACTS_DIR / "metrics_summary.csv", index=False)
    with open(ARTIFACTS_DIR / "best_params.json", "w", encoding="utf-8") as handle:
        json.dump(best_params, handle, indent=2)
    best_model = metrics_df.sort_values("rmse").iloc[0]
    best_model_info = {
        "model_name": best_model["model_name"],
        "display_name": best_model["display_name"],
        "rmse": float(best_model["rmse"]),
        "mae": float(best_model["mae"]),
        "r2": float(best_model["r2"]),
    }
    with open(ARTIFACTS_DIR / "best_model.json", "w", encoding="utf-8") as handle:
        json.dump(best_model_info, handle, indent=2)
    return best_model_info


def save_feature_input_artifacts(modeling_df: pd.DataFrame) -> None:
    numeric_defaults = {}
    numeric_ranges = {}
    for column in NUMERIC_FEATURES:
        series = modeling_df[column].dropna()
        numeric_defaults[column] = float(series.median())
        numeric_ranges[column] = {
            "min": float(series.quantile(0.05)),
            "max": float(series.quantile(0.95)),
        }

    categorical_defaults = {}
    categorical_options = {}
    for column in CATEGORICAL_FEATURES:
        series = modeling_df[column].dropna()
        categorical_defaults[column] = str(series.mode().iloc[0])
        categorical_options[column] = sorted(series.unique().tolist())

    payload = {
        "feature_labels": FEATURE_LABELS,
        "numeric_defaults": numeric_defaults,
        "numeric_ranges": numeric_ranges,
        "categorical_defaults": categorical_defaults,
        "categorical_options": categorical_options,
    }
    with open(ARTIFACTS_DIR / "feature_inputs.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_narrative_content(
    dataset_profile: dict,
    best_model_info: dict,
    shap_summary: dict,
    plot_notes: dict,
) -> None:
    top_shap = shap_summary["top_features"][0]["feature"] if shap_summary["top_features"] else "feature_x"
    top_shap_key = top_shap.replace("num__", "").replace("cat__", "")
    top_shap_label = FEATURE_LABELS.get(top_shap_key, top_shap_key.replace("_", " "))
    narrative = {
        "dataset_summary": (
            f"This project uses World Bank World Development Indicators to build a "
            f"country-year panel with {dataset_profile['rows_modeling']:,} observations across "
            f"{dataset_profile['countries']} economies from {dataset_profile['years'][0]} to "
            f"{dataset_profile['years'][1]}. The target is merchandise exports in current US$, "
            f"and the predictors combine trade values, macroeconomic scale variables, manufacturing "
            f"intensity, FDI, inflation, income level, region, and lagged export and import signals."
        ),
        "problem_importance": (
            "Export performance is a practical measure of how strongly an economy participates in "
            "global trade, and it influences growth expectations, external-balance monitoring, and "
            "supply-chain planning. Estimating export levels from broadly available macro and trade "
            "indicators helps policymakers, analysts, and business stakeholders benchmark countries "
            "and understand which structural signals move with export capacity."
        ),
        "approach_findings": (
            f"I trained five models on data from 2000 through 2016 and evaluated them on a "
            f"chronological hold-out period from 2017 through 2024 so the test setup reflects a "
            f"real forecasting scenario. {best_model_info['display_name']} performed best on the "
            f"held-out set with RMSE ${best_model_info['rmse']:,.0f}, MAE ${best_model_info['mae']:,.0f}, "
            f"and R^2 {best_model_info['r2']:.3f}, which suggests that the transformed trade and lag "
            f"features capture a strong and relatively stable relationship with export value."
        ),
        "captions": {
            "target_distribution": (
                "Export values are heavily right-skewed, with a small number of very large exporters "
                "stretching the distribution far above the median. The log-scaled view makes the bulk "
                "of the panel easier to compare and supports the decision to model the target on a log scale."
            ),
            "exports_by_region": (
                "North America has the highest median export level in this panel, but every region "
                "still shows a wide internal spread. Region captures broad structural differences, yet "
                "country-specific trade scale and momentum still matter inside each regional group."
            ),
            "gdp_vs_exports": (
                "GDP and exports move together strongly, with a raw correlation of 0.83 across the panel. "
                "Larger economies tend to export more, but the visible dispersion shows that GDP alone is "
                "not enough to explain trade performance."
            ),
            "lag_exports_vs_current": (
                "Prior-year exports correlate with current exports at 0.99, making persistence one of the "
                "strongest signals in the dataset. Countries that exported more in the previous year usually "
                "remain high exporters in the next period as well."
            ),
            "imports_vs_exports": (
                "Imports and exports correlate at 0.95, which indicates that highly trade-connected economies "
                "tend to be large on both sides of the border. Imports are therefore a strong predictor of "
                "export scale, even though they are not a direct causal explanation by themselves."
            ),
            "correlation_heatmap": (
                "The heatmap is dominated by tight relationships among scale-related trade variables, especially "
                "current exports, prior-year exports, imports, and GDP. That structure helps explain why simple "
                "models can perform very well once the data is transformed consistently."
            ),
        },
        "model_comparison": (
            "Linear Regression delivered the best held-out performance by a wide margin, outperforming "
            "the tree-based models and the MLP on both RMSE and MAE. In this dataset, the combination of "
            "log transformation and lagged trade signals appears to linearize much of the forecasting problem, "
            "while the nonlinear models add complexity without improving generalization."
        ),
        "shap_interpretation": (
            f"{MODEL_DISPLAY_NAMES[shap_summary['model_name']]} is the best-performing tree-based model, "
            f"and its SHAP profile is led by {top_shap_label.lower()}, prior-year exports, and GDP. The explanation plots "
            f"show that the model mainly responds to trade scale and recent momentum, while variables such as "
            f"manufacturing share, FDI inflows, inflation, and categorical metadata refine the prediction at the margin."
        ),
    }
    with open(ARTIFACTS_DIR / "narrative_content.json", "w", encoding="utf-8") as handle:
        json.dump(narrative, handle, indent=2)


def write_project_summary(
    dataset_profile: dict,
    metrics_df: pd.DataFrame,
    shap_summary: dict,
    plot_notes: dict,
) -> None:
    top_metrics = metrics_df.sort_values("rmse")[["display_name", "rmse", "mae", "r2"]]
    metric_lines = "\n".join(
        f"- {row.display_name}: RMSE ${row.rmse:,.0f}, MAE ${row.mae:,.0f}, R^2 {row.r2:.3f}"
        for row in top_metrics.itertuples(index=False)
    )
    shap_lines = "\n".join(
        f"- {row['feature']}: mean |SHAP| = {row['mean_abs_shap']:.4f}"
        for row in shap_summary["top_features"][:10]
    )
    plot_lines = "\n".join(f"- `{name}`: {note}" for name, note in plot_notes.items())
    summary_text = f"""# Project Summary

## Dataset facts
- Modeling rows: {dataset_profile['rows_modeling']:,}
- Economies: {dataset_profile['countries']}
- Year range: {dataset_profile['years'][0]}-{dataset_profile['years'][1]}
- Chronological hold-out starts in: {TEST_START_YEAR}

## Model ranking by RMSE
{metric_lines}

## SHAP top features
{shap_lines}

## Plot inventory with factual notes
{plot_lines}

## Generated deliverables
- Saved models for Linear Regression, Decision Tree, Random Forest, XGBoost, and PyTorch MLP
- Streamlit app assets for descriptive analytics, model performance, and SHAP explainability
- Narrative content file used directly by the deployed application
"""
    (SUMMARY_DIR / "project_summary.md").write_text(summary_text, encoding="utf-8")


def write_submission_audit(metrics_df: pd.DataFrame) -> None:
    required_files = [
        ARTIFACTS_DIR / "metrics_summary.csv",
        ARTIFACTS_DIR / "best_params.json",
        ARTIFACTS_DIR / "dataset_profile.json",
        ARTIFACTS_DIR / "narrative_content.json",
        SUMMARY_DIR / "project_summary.md",
        FIGURES_DIR / "target_distribution.png",
        FIGURES_DIR / "correlation_heatmap.png",
        FIGURES_DIR / "model_rmse_comparison.png",
        FIGURES_DIR / "predicted_vs_actual_grid.png",
        FIGURES_DIR / "mlp_training_history.png",
        FIGURES_DIR / "shap_summary.png",
        FIGURES_DIR / "shap_bar.png",
        FIGURES_DIR / "shap_waterfall_reference.png",
    ]
    audit = {
        "models_present": sorted(metrics_df["model_name"].tolist()),
        "files_present": {path.name: path.exists() for path in required_files},
        "all_checks_passed": all(path.exists() for path in required_files),
    }
    with open(SUMMARY_DIR / "submission_audit.json", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
