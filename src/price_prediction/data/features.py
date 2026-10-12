"""Feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from price_prediction.config import get_settings


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Создать признаки."""
    settings = get_settings()
    df = df.copy()

    # Лаги
    df["close_lag1"] = df["close"].shift(1)
    df["close_lag2"] = df["close"].shift(2)
    df["close_lag3"] = df["close"].shift(3)
    df["vol_lag1"] = df["vol"].shift(1)

    # Доходности
    df["return_1d"] = df["close"].pct_change().shift(1)
    df["return_lag_2"] = df["close"].pct_change().shift(2)
    df["return_lag_3"] = df["close"].pct_change().shift(3)
    df["return_intraday"] = ((df["close"] - df["open"]) / df["open"]).shift(1)

    # Rolling признаки
    for w in settings.features.sma_windows:
        df[f"sma{w}_lag1"] = df["close_lag1"].rolling(w).mean()

    for w in settings.features.vol_windows:
        df[f"volatility_{w}"] = df["return_1d"].rolling(w).std()

    df["vol_sma5"] = df["vol_lag1"].rolling(5).mean()
    df["vol_ratio"] = df["vol"] / df["vol_sma5"]
    df["hl_range"] = ((df["high"] - df["low"]) / df["close"]).shift(1)
    df["price_to_sma5"] = df["close_lag1"] / df["sma5_lag1"]
    df["price_to_sma20"] = df["close_lag1"] / df["sma20_lag1"]

    # RSI
    df["rsi_14"] = _calc_rsi(df["close_lag1"], settings.features.rsi_period)

    # Временные признаки
    df["weekday"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    # Циклическое кодирование
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Обработка выбросов
    q_lo, q_hi = settings.features.clip_quantiles
    numeric_cols = ["return_1d", "return_lag_2", "return_lag_3", "return_intraday", "hl_range"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].clip(df[col].quantile(q_lo), df[col].quantile(q_hi))

    # Удаление NaN
    df = df.dropna().reset_index(drop=True)

    logger.info(f"Создано признаков: {len([c for c in df.columns if c not in ['date', 'target']])}")

    return df


def _calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Рассчитать RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
