# MSIS 522 Homework 1: Global Export Forecast Studio

This project implements the full data science workflow for a World Bank trade forecasting assignment. The target is `TX.VAL.MRCH.CD.WT`, merchandise exports in current US dollars, modeled as a regression problem over a 2000-2024 country-year panel.

The repository automates:

- World Bank WDI data ingestion
- feature engineering and preprocessing
- model training and tuning
- SHAP explainability
- artifact generation for review
- a four-tab Streamlit app that loads only saved artifacts

## Project layout

- `scripts/train_pipeline.py`: end-to-end training and artifact generation
- `streamlit_app.py`: deployed Streamlit app entrypoint
- `src/export_forecast/`: reusable data, modeling, plotting, and app utilities
- `models/`: saved sklearn and PyTorch models
- `figures/`: exported figures for the app
- `artifacts/`: metrics, params, narrative template, SHAP summaries, and review pack

## Local setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_pipeline.py
streamlit run streamlit_app.py
```

## Deployment notes

Use `streamlit_app.py` as the app entrypoint. The app expects the saved artifacts in this repository and does not retrain models on startup.
