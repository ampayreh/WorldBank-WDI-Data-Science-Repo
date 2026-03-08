# Project Summary

## Dataset facts
- Modeling rows: 5,025
- Economies: 207
- Year range: 2000-2024
- Chronological hold-out starts in: 2017

## Model ranking by RMSE
- Linear Regression: RMSE $28,826,503,003, MAE $9,062,467,080, R^2 0.991
- Random Forest: RMSE $69,133,887,503, MAE $12,389,911,888, R^2 0.946
- Decision Tree: RMSE $77,971,038,142, MAE $15,804,809,846, R^2 0.932
- XGBoost: RMSE $90,343,145,857, MAE $14,604,743,810, R^2 0.909
- PyTorch MLP: RMSE $90,991,339,865, MAE $27,894,550,756, R^2 0.907

## SHAP top features
- `num__imports_usd`: mean |SHAP| = 1.1198
- `num__lag_1_exports_usd`: mean |SHAP| = 1.0452
- `num__gdp_usd`: mean |SHAP| = 0.5229
- `num__lag_1_imports_usd`: mean |SHAP| = 0.0742
- `num__population`: mean |SHAP| = 0.0569
- `num__year`: mean |SHAP| = 0.0305
- `num__trade_share_pct`: mean |SHAP| = 0.0108
- `num__manufacturing_share_pct`: mean |SHAP| = 0.0090
- `num__fdi_inflows_usd`: mean |SHAP| = 0.0078
- `num__gdp_per_capita_usd`: mean |SHAP| = 0.0028

## Plot inventory with factual notes
- `target_distribution`: Exports are strongly right-skewed. Median exports are $5,126,000,000 and the maximum reaches $3,576,653,000,000.
- `exports_by_region`: North America has the highest median export level in the panel. Regional spread is wide, which suggests structural differences matter.
- `gdp_vs_exports`: GDP and exports have a correlation of 0.83 in the raw panel. The positive slope supports including macro scale variables in the models.
- `lag_exports_vs_current`: Prior-year exports correlate with current exports at 0.99. That persistence should help both tree models and the MLP.
- `imports_vs_exports`: Imports and exports correlate at 0.95, suggesting that more trade-connected economies tend to move both measures together.
- `correlation_heatmap`: The strongest off-diagonal correlation is between exports_usd and lag_1_exports_usd at 0.99.

## Generated deliverables
- Saved models for Linear Regression, Decision Tree, Random Forest, XGBoost, and PyTorch MLP
- Streamlit app assets for descriptive analytics, model performance, and SHAP explainability
- Narrative content file used directly by the deployed application
