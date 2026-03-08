from __future__ import annotations

import time
from functools import reduce

import pandas as pd
import requests

from .config import API_BASE_URL, END_YEAR, INDICATORS, START_YEAR, TARGET_COLUMN


def _get_json(session: requests.Session, url: str, params: dict) -> list:
    last_error = None
    for attempt in range(3):
        try:
            response = session.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"Failed to fetch {url}") from last_error


def fetch_paginated(session: requests.Session, endpoint: str, params: dict) -> list[dict]:
    url = f"{API_BASE_URL}/{endpoint}"
    page = 1
    rows: list[dict] = []
    while True:
        payload = _get_json(
            session,
            url,
            {
                **params,
                "format": "json",
                "per_page": 20000,
                "page": page,
            },
        )
        if not isinstance(payload, list) or len(payload) < 2:
            raise RuntimeError(f"Unexpected payload from {endpoint}: {payload}")
        meta = payload[0]
        page_rows = payload[1] or []
        rows.extend(page_rows)
        total_pages = int(meta.get("pages", 0) or 0)
        if page >= total_pages:
            break
        page += 1
    return rows


def fetch_country_metadata(session: requests.Session) -> pd.DataFrame:
    rows = fetch_paginated(session, "country", {})
    records = []
    for row in rows:
        region = (row.get("region") or {}).get("value")
        if not region or region == "Aggregates":
            continue
        income_level = (row.get("incomeLevel") or {}).get("value") or "Unknown"
        records.append(
            {
                "country_code": row.get("id"),
                "country_name": row.get("name"),
                "region": region,
                "income_level": income_level,
            }
        )
    metadata = pd.DataFrame(records).drop_duplicates(subset=["country_code"])
    return metadata.sort_values(["region", "country_name"]).reset_index(drop=True)


def fetch_indicator_panel(
    session: requests.Session,
    indicator_code: str,
    column_name: str,
) -> pd.DataFrame:
    rows = fetch_paginated(session, f"country/all/indicator/{indicator_code}", {})
    records = []
    for row in rows:
        year_text = row.get("date")
        if not year_text or not year_text.isdigit():
            continue
        year = int(year_text)
        if year < START_YEAR or year > END_YEAR:
            continue
        records.append(
            {
                "country_code": row.get("countryiso3code"),
                "country_name": (row.get("country") or {}).get("value"),
                "year": year,
                column_name: row.get("value"),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError(f"No rows returned for indicator {indicator_code}")
    return frame


def build_modeling_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    session = requests.Session()
    session.headers.update({"User-Agent": "codex-msis522-homework"})
    metadata = fetch_country_metadata(session)

    indicator_frames = [
        fetch_indicator_panel(session, indicator_code, alias)
        for alias, indicator_code in INDICATORS.items()
    ]
    panel = reduce(
        lambda left, right: left.merge(
            right, on=["country_code", "country_name", "year"], how="outer"
        ),
        indicator_frames,
    )
    panel = panel.merge(metadata, on=["country_code", "country_name"], how="inner")
    panel = panel.sort_values(["country_code", "year"]).reset_index(drop=True)

    for column in INDICATORS:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")

    grouped = panel.groupby("country_code", group_keys=False)
    panel["export_growth_pct"] = grouped[TARGET_COLUMN].pct_change(fill_method=None) * 100
    panel["lag_1_exports_usd"] = grouped[TARGET_COLUMN].shift(1)
    panel["lag_1_imports_usd"] = grouped["imports_usd"].shift(1)
    panel["lag_1_export_growth_pct"] = grouped["export_growth_pct"].shift(1)

    modeling_df = panel.dropna(subset=[TARGET_COLUMN, "region", "income_level"]).copy()
    modeling_df = modeling_df.sort_values(["country_code", "year"]).reset_index(drop=True)
    return panel, modeling_df


def split_train_test(modeling_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = modeling_df[modeling_df["year"] < 2017].copy()
    test_df = modeling_df[modeling_df["year"] >= 2017].copy()
    if train_df.empty or test_df.empty:
        raise RuntimeError("Chronological split produced an empty train or test set.")
    return train_df, test_df
