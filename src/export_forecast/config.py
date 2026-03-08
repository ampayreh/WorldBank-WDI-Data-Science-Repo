from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SUMMARY_DIR = ARTIFACTS_DIR / "submission_summary"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

API_BASE_URL = "https://api.worldbank.org/v2"
START_YEAR = 2000
END_YEAR = 2024
TEST_START_YEAR = 2017
RANDOM_STATE = 42

TARGET_COLUMN = "exports_usd"
TARGET_INDICATOR = "TX.VAL.MRCH.CD.WT"

INDICATORS = {
    "exports_usd": "TX.VAL.MRCH.CD.WT",
    "imports_usd": "TM.VAL.MRCH.CD.WT",
    "gdp_usd": "NY.GDP.MKTP.CD",
    "gdp_per_capita_usd": "NY.GDP.PCAP.CD",
    "population": "SP.POP.TOTL",
    "inflation_pct": "FP.CPI.TOTL.ZG",
    "fdi_inflows_usd": "BX.KLT.DINV.CD.WD",
    "manufacturing_share_pct": "NV.IND.MANF.ZS",
    "trade_share_pct": "NE.TRD.GNFS.ZS",
}

FEATURE_LABELS = {
    "year": "Year",
    "imports_usd": "Merchandise imports (current US$)",
    "gdp_usd": "GDP (current US$)",
    "gdp_per_capita_usd": "GDP per capita (current US$)",
    "population": "Population",
    "inflation_pct": "Inflation, consumer prices (annual %)",
    "fdi_inflows_usd": "FDI net inflows (current US$)",
    "manufacturing_share_pct": "Manufacturing value added (% of GDP)",
    "trade_share_pct": "Trade (% of GDP)",
    "lag_1_exports_usd": "Prior-year exports (current US$)",
    "lag_1_imports_usd": "Prior-year imports (current US$)",
    "lag_1_export_growth_pct": "Prior-year export growth (%)",
    "region": "Region",
    "income_level": "Income level",
}

NUMERIC_FEATURES = [
    "year",
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

CATEGORICAL_FEATURES = ["region", "income_level"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODEL_DISPLAY_NAMES = {
    "linear_regression": "Linear Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "mlp": "PyTorch MLP",
}
