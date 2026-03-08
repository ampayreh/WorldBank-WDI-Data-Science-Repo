from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import torch

from .config import ARTIFACTS_DIR, FIGURES_DIR, MODEL_DISPLAY_NAMES, MODEL_FEATURES, MODELS_DIR
from .models import ExportMLP


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_metrics() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS_DIR / "metrics_summary.csv")


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(Path("data/processed/modeling_dataset.csv"))


@st.cache_data
def load_feature_inputs() -> dict:
    return load_json(ARTIFACTS_DIR / "feature_inputs.json")


@st.cache_data
def load_narrative() -> dict:
    return load_json(ARTIFACTS_DIR / "narrative_content.json")


@st.cache_data
def load_best_params() -> dict:
    return load_json(ARTIFACTS_DIR / "best_params.json")


@st.cache_resource
def load_sklearn_model(model_name: str):
    return joblib.load(MODELS_DIR / f"{model_name}.joblib")


@st.cache_resource
def load_mlp_artifacts():
    preprocessor = joblib.load(MODELS_DIR / "mlp_preprocessor.joblib")
    bundle = torch.load(MODELS_DIR / "mlp_bundle.pt", map_location="cpu")
    model = ExportMLP(bundle["input_dim"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, preprocessor


def predict_single_row(model_name: str, row_df: pd.DataFrame) -> float:
    if model_name == "mlp":
        model, preprocessor = load_mlp_artifacts()
        transformed = preprocessor.transform(row_df[MODEL_FEATURES]).astype(np.float32)
        with torch.no_grad():
            prediction_log = model(torch.tensor(transformed)).item()
        return float(np.expm1(prediction_log))
    model = load_sklearn_model(model_name)
    return float(model.predict(row_df[MODEL_FEATURES])[0])


def format_currency(value: float) -> str:
    if abs(value) >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:,.2f}T"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    return f"${value:,.0f}"


@st.cache_resource
def load_best_tree_explainer():
    best_tree = load_json(ARTIFACTS_DIR / "best_tree_model.json")["model_name"]
    model = load_sklearn_model(best_tree)
    pipeline = model.regressor_
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    background_raw = pd.read_csv(ARTIFACTS_DIR / "shap_background_raw.csv")
    background_transformed = preprocessor.transform(background_raw[MODEL_FEATURES])
    explainer = shap.Explainer(
        estimator,
        background_transformed,
        feature_names=list(preprocessor.get_feature_names_out()),
    )
    return best_tree, model, preprocessor, explainer


def make_custom_shap_figure(row_df: pd.DataFrame):
    best_tree, _, preprocessor, explainer = load_best_tree_explainer()
    transformed = preprocessor.transform(row_df[MODEL_FEATURES])
    explanation = explainer(transformed)
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation[0], max_display=15, show=False)
    plt.tight_layout()
    return best_tree, fig


def figure_path(name: str) -> Path:
    return FIGURES_DIR / name
