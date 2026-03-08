from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

from export_forecast.artifacts import (  # noqa: E402
    save_dataset_outputs,
    save_feature_input_artifacts,
    save_metrics_outputs,
    write_narrative_template,
    write_review_pack,
    write_rubric_audit,
)
from export_forecast.config import ARTIFACTS_DIR, RANDOM_STATE, TARGET_COLUMN  # noqa: E402
from export_forecast.data import build_modeling_dataset, split_train_test  # noqa: E402
from export_forecast.models import (  # noqa: E402
    append_mlp_results,
    generate_shap_artifacts,
    save_mlp_history_plot,
    select_best_tree_model,
    set_global_seed,
    train_mlp_model,
    train_sklearn_models,
)
from export_forecast.visualization import (  # noqa: E402
    create_decision_tree_figure,
    create_eda_figures,
    create_model_performance_figures,
)


def main() -> None:
    set_global_seed(RANDOM_STATE)
    panel_df, modeling_df = build_modeling_dataset()
    dataset_profile = save_dataset_outputs(panel_df, modeling_df)
    save_feature_input_artifacts(modeling_df)

    train_df, test_df = split_train_test(modeling_df)
    sklearn_result = train_sklearn_models(train_df, test_df)
    mlp_result = train_mlp_model(train_df, test_df)
    save_mlp_history_plot(mlp_result.history)

    metrics_df, predictions = append_mlp_results(sklearn_result, mlp_result)
    best_params = {**sklearn_result.best_params, "mlp": {}}
    best_model_info = save_metrics_outputs(metrics_df, best_params)

    plot_notes = create_eda_figures(modeling_df)
    create_model_performance_figures(
        metrics_df, test_df[TARGET_COLUMN].to_numpy(), predictions
    )
    create_decision_tree_figure(sklearn_result.models["decision_tree"], train_df)

    best_tree_model = select_best_tree_model(metrics_df)
    shap_summary = generate_shap_artifacts(
        best_tree_model,
        sklearn_result.models[best_tree_model],
        train_df,
        test_df,
    )

    with open(ARTIFACTS_DIR / "train_test_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "train_years": sorted(train_df["year"].unique().tolist()),
                "test_years": sorted(test_df["year"].unique().tolist()),
            },
            handle,
            indent=2,
        )

    write_narrative_template(dataset_profile, best_model_info, shap_summary, plot_notes)
    write_review_pack(dataset_profile, metrics_df, shap_summary, plot_notes)
    write_rubric_audit(metrics_df)

    print("Training pipeline completed.")
    print(f"Modeling rows: {len(modeling_df):,}")
    print(metrics_df[["display_name", "rmse", "mae", "r2"]].to_string(index=False))


if __name__ == "__main__":
    main()
