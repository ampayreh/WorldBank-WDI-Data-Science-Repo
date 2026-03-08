from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from .config import (
    ARTIFACTS_DIR,
    CATEGORICAL_FEATURES,
    FIGURES_DIR,
    MODEL_DISPLAY_NAMES,
    MODEL_FEATURES,
    MODELS_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
)

matplotlib.use("Agg")


@dataclass
class SklearnTrainingResult:
    models: dict[str, TransformedTargetRegressor]
    metrics: pd.DataFrame
    predictions: dict[str, np.ndarray]
    best_params: dict[str, dict]


@dataclass
class MLPTrainingResult:
    metrics_row: dict
    predictions: np.ndarray
    history: pd.DataFrame
    feature_names: list[str]
    model_path: Path
    preprocessor_path: Path


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def signed_log1p(values):
    return np.sign(values) * np.log1p(np.abs(values))


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        (
            "signed_log",
            FunctionTransformer(
                signed_log1p,
                validate=False,
                feature_names_out="one-to-one",
            ),
        ),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("cat", Pipeline(categorical_steps), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_wrapped_regressor(model, scale_numeric: bool) -> TransformedTargetRegressor:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", model),
        ]
    )
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _strip_best_params(best_params: dict) -> dict:
    cleaned = {}
    for key, value in best_params.items():
        cleaned[key.replace("regressor__model__", "")] = value
    return cleaned


def train_sklearn_models(train_df: pd.DataFrame, test_df: pd.DataFrame) -> SklearnTrainingResult:
    X_train = train_df[MODEL_FEATURES]
    y_train = train_df[TARGET_COLUMN].to_numpy()
    X_test = test_df[MODEL_FEATURES]
    y_test = test_df[TARGET_COLUMN].to_numpy()
    groups = train_df["year"].to_numpy()

    model_specs = {
        "linear_regression": {
            "estimator": LinearRegression(),
            "scale_numeric": True,
            "param_grid": None,
        },
        "decision_tree": {
            "estimator": DecisionTreeRegressor(random_state=RANDOM_STATE),
            "scale_numeric": False,
            "param_grid": {
                "regressor__model__max_depth": [3, 5, 7, 10],
                "regressor__model__min_samples_leaf": [5, 10, 20, 50],
            },
        },
        "random_forest": {
            "estimator": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            "scale_numeric": False,
            "param_grid": {
                "regressor__model__n_estimators": [50, 100, 200],
                "regressor__model__max_depth": [3, 5, 8],
            },
        },
        "xgboost": {
            "estimator": XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=4,
                tree_method="hist",
                subsample=0.9,
                colsample_bytree=0.9,
            ),
            "scale_numeric": False,
            "param_grid": {
                "regressor__model__n_estimators": [50, 100, 200],
                "regressor__model__max_depth": [3, 4, 5, 6],
                "regressor__model__learning_rate": [0.01, 0.05, 0.1],
            },
        },
    }

    metrics_rows = []
    best_params: dict[str, dict] = {}
    fitted_models: dict[str, TransformedTargetRegressor] = {}
    predictions: dict[str, np.ndarray] = {}
    cv = GroupKFold(n_splits=5)

    for model_name, spec in model_specs.items():
        wrapped = build_wrapped_regressor(
            spec["estimator"], scale_numeric=spec["scale_numeric"]
        )
        started = time.time()
        if spec["param_grid"]:
            search = GridSearchCV(
                estimator=wrapped,
                param_grid=spec["param_grid"],
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train, groups=groups)
            fitted = search.best_estimator_
            best_params[model_name] = _strip_best_params(search.best_params_)
        else:
            fitted = wrapped.fit(X_train, y_train)
            best_params[model_name] = {}
        duration_seconds = time.time() - started
        y_pred = fitted.predict(X_test)
        fitted_models[model_name] = fitted
        predictions[model_name] = y_pred
        metric_row = compute_metrics(y_test, y_pred)
        metric_row.update(
            {
                "model_name": model_name,
                "display_name": MODEL_DISPLAY_NAMES[model_name],
                "train_seconds": float(duration_seconds),
            }
        )
        metrics_rows.append(metric_row)
        joblib.dump(fitted, MODELS_DIR / f"{model_name}.joblib")

    metrics = pd.DataFrame(metrics_rows).sort_values("rmse").reset_index(drop=True)
    return SklearnTrainingResult(
        models=fitted_models,
        metrics=metrics,
        predictions=predictions,
        best_params=best_params,
    )


class ExportMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def train_mlp_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> MLPTrainingResult:
    set_global_seed(RANDOM_STATE)
    X_train = train_df[MODEL_FEATURES]
    y_train = train_df[TARGET_COLUMN].to_numpy()
    X_test = test_df[MODEL_FEATURES]
    y_test = test_df[TARGET_COLUMN].to_numpy()

    preprocessor = build_preprocessor(scale_numeric=True)
    X_train_transformed = preprocessor.fit_transform(X_train).astype(np.float32)
    X_test_transformed = preprocessor.transform(X_test).astype(np.float32)
    feature_names = list(preprocessor.get_feature_names_out())

    y_train_log = np.log1p(y_train).astype(np.float32)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_transformed,
        y_train_log,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train_split), torch.tensor(y_train_split).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    X_val_tensor = torch.tensor(X_val_split)
    y_val_tensor = torch.tensor(y_val_split).unsqueeze(1)
    y_val_raw = np.expm1(y_val_split)

    model = ExportMLP(input_dim=X_train_transformed.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()
    started = time.time()

    history_rows = []
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience = 20
    epochs_without_improvement = 0

    for epoch in range(1, 151):
        model.train()
        train_losses = []
        for batch_features, batch_target in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_predictions_raw = np.expm1(np.clip(val_outputs.squeeze(1).numpy(), -20, 35))
            val_rmse = float(np.sqrt(mean_squared_error(y_val_raw, val_predictions_raw)))

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": float(val_loss),
                "val_rmse_raw": val_rmse,
            }
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_predictions_log = model(torch.tensor(X_test_transformed)).squeeze(1).numpy()
    test_predictions = np.expm1(np.clip(test_predictions_log, -20, 35))
    metrics_row = compute_metrics(y_test, test_predictions)
    metrics_row.update(
        {
            "model_name": "mlp",
            "display_name": MODEL_DISPLAY_NAMES["mlp"],
            "train_seconds": float(time.time() - started),
        }
    )

    preprocessor_path = MODELS_DIR / "mlp_preprocessor.joblib"
    model_path = MODELS_DIR / "mlp_bundle.pt"
    joblib.dump(preprocessor, preprocessor_path)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": X_train_transformed.shape[1],
        },
        model_path,
    )

    return MLPTrainingResult(
        metrics_row=metrics_row,
        predictions=test_predictions,
        history=pd.DataFrame(history_rows),
        feature_names=feature_names,
        model_path=model_path,
        preprocessor_path=preprocessor_path,
    )


def append_mlp_results(
    sklearn_result: SklearnTrainingResult, mlp_result: MLPTrainingResult
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    metrics = pd.concat(
        [sklearn_result.metrics, pd.DataFrame([mlp_result.metrics_row])],
        ignore_index=True,
    ).sort_values("rmse")
    predictions = {**sklearn_result.predictions, "mlp": mlp_result.predictions}
    return metrics.reset_index(drop=True), predictions


def select_best_tree_model(metrics_df: pd.DataFrame) -> str:
    tree_metrics = metrics_df[metrics_df["model_name"].isin(["random_forest", "xgboost"])]
    if tree_metrics.empty:
        raise RuntimeError("No tree-based model metrics available for SHAP.")
    return tree_metrics.sort_values("rmse").iloc[0]["model_name"]


def generate_shap_artifacts(
    model_name: str,
    fitted_model: TransformedTargetRegressor,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    X_train = train_df[MODEL_FEATURES]
    X_test = test_df[MODEL_FEATURES]
    y_test = test_df[TARGET_COLUMN].to_numpy()

    pipeline = fitted_model.regressor_
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]

    background_raw = X_train.sample(
        n=min(256, len(X_train)), random_state=RANDOM_STATE
    ).reset_index(drop=True)
    evaluation_raw = X_test.sample(
        n=min(300, len(X_test)), random_state=RANDOM_STATE
    ).reset_index(drop=True)
    background_transformed = preprocessor.transform(background_raw)
    evaluation_transformed = preprocessor.transform(evaluation_raw)
    feature_names = list(preprocessor.get_feature_names_out())

    explainer = shap.Explainer(estimator, background_transformed, feature_names=feature_names)
    shap_values = explainer(evaluation_transformed)
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top_features = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    top_features.to_csv(ARTIFACTS_DIR / "shap_top_features.csv", index=False)
    background_raw.to_csv(ARTIFACTS_DIR / "shap_background_raw.csv", index=False)

    plt.figure(figsize=(12, 7))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    reference_idx = int(np.argmax(y_test))
    reference_row = X_test.iloc[[reference_idx]]
    reference_transformed = preprocessor.transform(reference_row)
    reference_explanation = explainer(reference_transformed)
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(reference_explanation[0], max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_waterfall_reference.png", dpi=200, bbox_inches="tight")
    plt.close()

    with open(ARTIFACTS_DIR / "shap_reference_case.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_name": model_name,
                "country": test_df.iloc[reference_idx]["country_name"],
                "year": int(test_df.iloc[reference_idx]["year"]),
                "actual_exports_usd": float(y_test[reference_idx]),
            },
            handle,
            indent=2,
        )

    with open(ARTIFACTS_DIR / "best_tree_model.json", "w", encoding="utf-8") as handle:
        json.dump({"model_name": model_name}, handle, indent=2)

    return {
        "model_name": model_name,
        "top_features": top_features.head(10).to_dict(orient="records"),
    }


def save_mlp_history_plot(history_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Validation loss")
    axes[0].set_title("MLP Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE on log target")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["val_rmse_raw"], color="#a03f2d")
    axes[1].set_title("MLP Validation RMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE (US$)")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mlp_training_history.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
