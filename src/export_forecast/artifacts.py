from __future__ import annotations

import json

import pandas as pd

from .config import (
    ARTIFACTS_DIR,
    CATEGORICAL_FEATURES,
    FEATURE_LABELS,
    FIGURES_DIR,
    MANUAL_REVIEW_SECTIONS,
    MODEL_DISPLAY_NAMES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    PROCESSED_DATA_DIR,
    REVIEW_DIR,
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


def write_narrative_template(
    dataset_profile: dict,
    best_model_info: dict,
    shap_summary: dict,
    plot_notes: dict,
) -> None:
    top_shap = shap_summary["top_features"][0]["feature"] if shap_summary["top_features"] else "feature_x"
    narrative = {
        "manual_review_required": True,
        "dataset_summary": (
            "[MANUAL REVIEW REQUIRED] Replace this with your own paragraph. Safe facts: "
            f"the modeling panel contains {dataset_profile['rows_modeling']:,} country-year rows "
            f"across {dataset_profile['countries']} economies from {dataset_profile['years'][0]} "
            f"to {dataset_profile['years'][1]}, and the target is merchandise exports in current US$."
        ),
        "problem_importance": (
            "[MANUAL REVIEW REQUIRED] Explain why forecasting export value matters in your own words. "
            "Tie it to trade planning, macro monitoring, or policy prioritization."
        ),
        "approach_findings": (
            "[MANUAL REVIEW REQUIRED] Replace this with your own 1-2 paragraphs. Safe facts: "
            f"the best held-out model is {best_model_info['display_name']} with RMSE "
            f"${best_model_info['rmse']:,.0f}, MAE ${best_model_info['mae']:,.0f}, "
            f"and R^2 {best_model_info['r2']:.3f}."
        ),
        "captions": {
            "target_distribution": (
                "[MANUAL REVIEW REQUIRED] " + plot_notes["target_distribution"]
            ),
            "exports_by_region": (
                "[MANUAL REVIEW REQUIRED] " + plot_notes["exports_by_region"]
            ),
            "gdp_vs_exports": "[MANUAL REVIEW REQUIRED] " + plot_notes["gdp_vs_exports"],
            "lag_exports_vs_current": (
                "[MANUAL REVIEW REQUIRED] " + plot_notes["lag_exports_vs_current"]
            ),
            "imports_vs_exports": (
                "[MANUAL REVIEW REQUIRED] " + plot_notes["imports_vs_exports"]
            ),
            "correlation_heatmap": (
                "[MANUAL REVIEW REQUIRED] " + plot_notes["correlation_heatmap"]
            ),
        },
        "model_comparison": (
            "[MANUAL REVIEW REQUIRED] Compare the best model against the baseline and the MLP "
            "in your own words. Use the metrics table to support every claim."
        ),
        "shap_interpretation": (
            "[MANUAL REVIEW REQUIRED] Start from this fact scaffold and rewrite it in your own words: "
            f"the best tree-based model is {MODEL_DISPLAY_NAMES[shap_summary['model_name']]} and its "
            f"top SHAP feature is {top_shap}."
        ),
    }
    with open(ARTIFACTS_DIR / "narrative_content.json", "w", encoding="utf-8") as handle:
        json.dump(narrative, handle, indent=2)


def write_review_pack(
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
    review_text = f"""# Review Pack

## Manual sign-off checklist
""" + "\n".join(f"- [ ] {item}" for item in MANUAL_REVIEW_SECTIONS) + f"""

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

## Final reminders
- Edit `artifacts/narrative_content.json` before submission.
- Confirm every interpretation sentence matches the chart or metric it references.
- Test the deployed app in an incognito window before submitting the link.
"""
    (REVIEW_DIR / "review_pack.md").write_text(review_text, encoding="utf-8")


def write_rubric_audit(metrics_df: pd.DataFrame) -> None:
    required_files = [
        ARTIFACTS_DIR / "metrics_summary.csv",
        ARTIFACTS_DIR / "best_params.json",
        ARTIFACTS_DIR / "dataset_profile.json",
        ARTIFACTS_DIR / "narrative_content.json",
        REVIEW_DIR / "review_pack.md",
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
    with open(REVIEW_DIR / "rubric_audit.json", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
