from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

from export_forecast.app_utils import (  # noqa: E402
    figure_path,
    format_currency,
    load_best_params,
    load_best_tree_explainer,
    load_dataset,
    load_feature_inputs,
    load_metrics,
    load_narrative,
    make_custom_shap_figure,
    predict_single_row,
)
from export_forecast.config import MODEL_DISPLAY_NAMES, MODEL_FEATURES  # noqa: E402


st.set_page_config(
    page_title="Global Export Forecast Studio",
    page_icon=":bar_chart:",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(160,63,45,0.16), transparent 28%),
            radial-gradient(circle at top right, rgba(47,93,98,0.12), transparent 24%),
            linear-gradient(180deg, #f7f1e3 0%, #f3ebd8 52%, #efe5d0 100%);
    }
    h1, h2, h3 {
        letter-spacing: 0.02em;
    }
    .hero {
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(28,25,23,0.08);
        border-radius: 20px;
        background: rgba(255,255,255,0.55);
        box-shadow: 0 20px 40px rgba(28,25,23,0.08);
        margin-bottom: 1rem;
    }
    .metric-card {
        padding: 0.9rem 1rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.72);
        border: 1px solid rgba(28,25,23,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

metrics_df = load_metrics()
dataset_df = load_dataset()
narrative = load_narrative()
feature_inputs = load_feature_inputs()
best_params = load_best_params()
best_model_row = metrics_df.sort_values("rmse").iloc[0]
best_tree_name = load_best_tree_explainer()[0]

st.markdown(
    f"""
    <div class="hero">
        <h1>Global Export Forecast Studio</h1>
        <p>End-to-end analysis of World Bank trade indicators, with descriptive analytics,
        model benchmarking, SHAP explainability, and interactive export prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
metric_cols[0].markdown(
    f"<div class='metric-card'><strong>Rows</strong><br>{len(dataset_df):,}</div>",
    unsafe_allow_html=True,
)
metric_cols[1].markdown(
    f"<div class='metric-card'><strong>Economies</strong><br>{dataset_df['country_code'].nunique()}</div>",
    unsafe_allow_html=True,
)
metric_cols[2].markdown(
    f"<div class='metric-card'><strong>Best Model</strong><br>{best_model_row['display_name']}</div>",
    unsafe_allow_html=True,
)
metric_cols[3].markdown(
    f"<div class='metric-card'><strong>Best RMSE</strong><br>{format_currency(best_model_row['rmse'])}</div>",
    unsafe_allow_html=True,
)

if narrative.get("manual_review_required", False):
    st.warning(
        "Manual review is still required for the executive summary, plot captions, "
        "model comparison text, and SHAP interpretation before submission."
    )

tab_summary, tab_descriptive, tab_models, tab_explain = st.tabs(
    [
        "Executive Summary",
        "Descriptive Analytics",
        "Model Performance",
        "Explainability & Interactive Prediction",
    ]
)

with tab_summary:
    st.subheader("Dataset and Prediction Task")
    st.write(narrative["dataset_summary"])
    st.subheader("Why This Problem Matters")
    st.write(narrative["problem_importance"])
    st.subheader("Approach and Key Findings")
    st.write(narrative["approach_findings"])

with tab_descriptive:
    descriptive_figures = [
        ("target_distribution.png", "target_distribution", "Target Distribution"),
        ("exports_by_region.png", "exports_by_region", "Exports by Region"),
        ("gdp_vs_exports.png", "gdp_vs_exports", "GDP vs. Exports"),
        ("lag_exports_vs_current.png", "lag_exports_vs_current", "Lagged Exports vs. Current Exports"),
        ("imports_vs_exports.png", "imports_vs_exports", "Imports vs. Exports"),
        ("correlation_heatmap.png", "correlation_heatmap", "Correlation Heatmap"),
    ]
    for filename, caption_key, title in descriptive_figures:
        st.markdown(f"### {title}")
        st.image(str(figure_path(filename)), use_container_width=True)
        st.caption(narrative["captions"][caption_key])

with tab_models:
    st.subheader("Held-Out Test Metrics")
    metrics_display = metrics_df.copy()
    metrics_display["mae"] = metrics_display["mae"].map(format_currency)
    metrics_display["rmse"] = metrics_display["rmse"].map(format_currency)
    metrics_display["r2"] = metrics_df["r2"].map(lambda value: f"{value:.3f}")
    st.dataframe(metrics_display[["display_name", "mae", "rmse", "r2"]], use_container_width=True)

    st.subheader("Best Hyperparameters")
    best_param_rows = []
    for model_name, params in best_params.items():
        best_param_rows.append(
            {
                "Model": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                "Best Hyperparameters": str(params),
            }
        )
    st.dataframe(pd.DataFrame(best_param_rows), use_container_width=True)

    st.image(str(figure_path("model_rmse_comparison.png")), use_container_width=True)
    st.caption(narrative["model_comparison"])
    st.image(str(figure_path("predicted_vs_actual_grid.png")), use_container_width=True)
    st.image(str(figure_path("mlp_training_history.png")), use_container_width=True)
    st.image(str(figure_path("best_decision_tree.png")), use_container_width=True)

with tab_explain:
    st.subheader("Tree-Based Explainability")
    st.write(narrative["shap_interpretation"])
    shap_cols = st.columns(2)
    shap_cols[0].image(str(figure_path("shap_summary.png")), use_container_width=True)
    shap_cols[1].image(str(figure_path("shap_bar.png")), use_container_width=True)

    st.subheader("Interactive Prediction")
    selected_model_display = st.selectbox(
        "Prediction model",
        options=[MODEL_DISPLAY_NAMES[name] for name in metrics_df["model_name"].tolist()],
        index=0,
    )
    selected_model_name = next(
        key for key, value in MODEL_DISPLAY_NAMES.items() if value == selected_model_display
    )

    numeric_cols = st.columns(3)
    user_inputs: dict[str, object] = {}
    numeric_defaults = feature_inputs["numeric_defaults"]
    numeric_ranges = feature_inputs["numeric_ranges"]

    for idx, feature in enumerate([name for name in MODEL_FEATURES if name in numeric_defaults]):
        with numeric_cols[idx % 3]:
            feature_range = numeric_ranges[feature]
            user_inputs[feature] = st.number_input(
                feature_inputs["feature_labels"][feature],
                value=float(numeric_defaults[feature]),
                min_value=float(feature_range["min"]),
                max_value=float(feature_range["max"]),
                step=max((feature_range["max"] - feature_range["min"]) / 100.0, 1.0),
            )

    cat_cols = st.columns(2)
    for idx, feature in enumerate(feature_inputs["categorical_options"]):
        with cat_cols[idx % 2]:
            options = feature_inputs["categorical_options"][feature]
            default_value = feature_inputs["categorical_defaults"][feature]
            user_inputs[feature] = st.selectbox(
                feature_inputs["feature_labels"][feature],
                options=options,
                index=options.index(default_value),
            )

    user_row = pd.DataFrame([user_inputs])
    predicted_value = predict_single_row(selected_model_name, user_row)
    st.metric("Predicted merchandise exports", format_currency(predicted_value))
    st.caption(
        f"Prediction uses {selected_model_display}. The SHAP waterfall below is always "
        f"generated with the best tree-based model, {MODEL_DISPLAY_NAMES[best_tree_name]}."
    )

    best_tree_used, shap_figure = make_custom_shap_figure(user_row)
    st.pyplot(shap_figure, use_container_width=True)
    st.caption(f"Custom SHAP explanation generated with {MODEL_DISPLAY_NAMES[best_tree_used]}.")
