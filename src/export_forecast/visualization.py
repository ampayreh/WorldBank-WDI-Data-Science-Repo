from __future__ import annotations

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import plot_tree

from .config import ARTIFACTS_DIR, FIGURES_DIR, MODEL_DISPLAY_NAMES, TARGET_COLUMN

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


def _safe_log(series: pd.Series) -> pd.Series:
    clipped = series.clip(lower=0)
    return np.log1p(clipped)


def create_eda_figures(modeling_df: pd.DataFrame) -> dict[str, str]:
    notes: dict[str, str] = {}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(modeling_df[TARGET_COLUMN], bins=40, ax=axes[0], color="#a03f2d")
    axes[0].set_title("Raw Export Distribution")
    axes[0].set_xlabel("Exports (current US$)")
    sns.histplot(_safe_log(modeling_df[TARGET_COLUMN]), bins=40, ax=axes[1], color="#2f5d62")
    axes[1].set_title("Log-Scaled Export Distribution")
    axes[1].set_xlabel("log(1 + exports)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "target_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    notes["target_distribution"] = (
        f"Exports are strongly right-skewed. Median exports are "
        f"${modeling_df[TARGET_COLUMN].median():,.0f} and the maximum reaches "
        f"${modeling_df[TARGET_COLUMN].max():,.0f}."
    )

    region_order = (
        modeling_df.groupby("region")[TARGET_COLUMN].median().sort_values(ascending=False).index
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=modeling_df,
        x="region",
        y=_safe_log(modeling_df[TARGET_COLUMN]),
        order=region_order,
        ax=ax,
        hue="region",
        dodge=False,
        palette="crest",
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_title("Log Export Distribution by Region")
    ax.set_xlabel("")
    ax.set_ylabel("log(1 + exports)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exports_by_region.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    top_region = modeling_df.groupby("region")[TARGET_COLUMN].median().idxmax()
    notes["exports_by_region"] = (
        f"{top_region} has the highest median export level in the panel. "
        "Regional spread is wide, which suggests structural differences matter."
    )

    scatter_df = modeling_df.dropna(subset=["gdp_usd"]).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=scatter_df,
        x=_safe_log(scatter_df["gdp_usd"]),
        y=_safe_log(scatter_df[TARGET_COLUMN]),
        hue="income_level",
        alpha=0.7,
        s=45,
        ax=ax,
    )
    ax.set_title("GDP vs. Exports")
    ax.set_xlabel("log(1 + GDP)")
    ax.set_ylabel("log(1 + exports)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "gdp_vs_exports.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    correlation = scatter_df[["gdp_usd", TARGET_COLUMN]].corr().iloc[0, 1]
    notes["gdp_vs_exports"] = (
        f"GDP and exports have a correlation of {correlation:.2f} in the raw panel. "
        "The positive slope supports including macro scale variables in the models."
    )

    lag_df = modeling_df.dropna(subset=["lag_1_exports_usd"]).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=lag_df,
        x=_safe_log(lag_df["lag_1_exports_usd"]),
        y=_safe_log(lag_df[TARGET_COLUMN]),
        hue="income_level",
        alpha=0.7,
        s=45,
        ax=ax,
    )
    ax.set_title("Prior-Year Exports vs. Current Exports")
    ax.set_xlabel("log(1 + prior-year exports)")
    ax.set_ylabel("log(1 + current exports)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "lag_exports_vs_current.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    lag_corr = lag_df[["lag_1_exports_usd", TARGET_COLUMN]].corr().iloc[0, 1]
    notes["lag_exports_vs_current"] = (
        f"Prior-year exports correlate with current exports at {lag_corr:.2f}. "
        "That persistence should help both tree models and the MLP."
    )

    imports_df = modeling_df.dropna(subset=["imports_usd"]).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=imports_df,
        x=_safe_log(imports_df["imports_usd"]),
        y=_safe_log(imports_df[TARGET_COLUMN]),
        hue="region",
        alpha=0.6,
        s=40,
        ax=ax,
        legend=False,
    )
    ax.set_title("Imports vs. Exports")
    ax.set_xlabel("log(1 + imports)")
    ax.set_ylabel("log(1 + exports)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "imports_vs_exports.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    import_corr = imports_df[["imports_usd", TARGET_COLUMN]].corr().iloc[0, 1]
    notes["imports_vs_exports"] = (
        f"Imports and exports correlate at {import_corr:.2f}, suggesting that "
        "more trade-connected economies tend to move both measures together."
    )

    numeric_columns = [
        "exports_usd",
        "imports_usd",
        "gdp_usd",
        "gdp_per_capita_usd",
        "population",
        "inflation_pct",
        "fdi_inflows_usd",
        "manufacturing_share_pct",
        "trade_share_pct",
        "lag_1_exports_usd",
        "lag_1_imports_usd",
        "lag_1_export_growth_pct",
    ]
    corr = modeling_df[numeric_columns].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    strongest_pair = corr.where(~np.eye(len(corr), dtype=bool)).abs().stack().idxmax()
    strongest_value = corr.loc[strongest_pair[0], strongest_pair[1]]
    notes["correlation_heatmap"] = (
        f"The strongest off-diagonal correlation is between {strongest_pair[0]} and "
        f"{strongest_pair[1]} at {strongest_value:.2f}."
    )

    with open(ARTIFACTS_DIR / "plot_notes.json", "w", encoding="utf-8") as handle:
        json.dump(notes, handle, indent=2)
    return notes


def create_model_performance_figures(
    metrics_df: pd.DataFrame,
    y_test: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    sorted_metrics = metrics_df.sort_values("rmse")
    sns.barplot(
        data=sorted_metrics,
        x="display_name",
        y="rmse",
        hue="display_name",
        dodge=False,
        palette="flare",
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_title("Model Comparison by RMSE")
    ax.set_xlabel("")
    ax.set_ylabel("RMSE (US$)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_rmse_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    axes = axes.flatten()
    for idx, model_name in enumerate(metrics_df["model_name"].tolist()):
        ax = axes[idx]
        y_pred = predictions[model_name]
        ax.scatter(y_test, y_pred, alpha=0.45, color="#a03f2d", s=22)
        lower = min(float(np.min(y_test)), float(np.min(y_pred)))
        upper = max(float(np.max(y_test)), float(np.max(y_pred)))
        ax.plot([lower, upper], [lower, upper], linestyle="--", color="#2f5d62")
        ax.set_title(MODEL_DISPLAY_NAMES[model_name])
        ax.set_xlabel("Actual exports")
        ax.set_ylabel("Predicted exports")
    for ax in axes[len(metrics_df) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "predicted_vs_actual_grid.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_decision_tree_figure(decision_tree_model, train_df: pd.DataFrame) -> None:
    pipeline = decision_tree_model.regressor_
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(
        estimator,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Best Decision Tree (truncated to depth 3)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "best_decision_tree.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
