"""Модуль загрузки данных."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from price_prediction.config import get_settings


def load_and_clean_data(file_path: str | Path | None = None) -> pd.DataFrame:
    """Загрузить и очистить данные."""
    settings = get_settings()

    if file_path is None:
        file_path = settings.data.raw_path

    file_path = Path(file_path)

    if not file_path.exists():
        logger.warning(f"Файл {file_path} не найден. Генерирую синтетические данные.")
        df = _generate_synthetic_data()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(file_path, index=False)
    else:
        logger.info(f"Загружен файл: {file_path}")
        df = pd.read_excel(file_path, decimal=settings.data.decimal_separator)

    # Парсинг даты
    df[settings.data.date_column] = pd.to_datetime(df[settings.data.date_column].astype(str), format="%Y%m%d")
    df = df.sort_values(settings.data.date_column).reset_index(drop=True)

    n_before = len(df)
    logger.info(f"Загружено записей: {n_before}")
    logger.info(f"Период: {df[settings.data.date_column].min().date()} -> {df[settings.data.date_column].max().date()}")

    return df


def _generate_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    """Сгенерировать синтетические данные."""
    np.random.seed(42)

    dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")

    df = pd.DataFrame(
        {
            "date": dates,
            "open": np.random.normal(100, 10, n_samples),
            "high": np.random.normal(105, 10, n_samples),
            "low": np.random.normal(95, 10, n_samples),
            "close": np.random.normal(100, 10, n_samples),
            "vol": np.random.randint(100000, 1000000, n_samples),
            "target": np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        }
    )

    return df
