"""Модуль проверки дрейфа данных."""

import pandas as pd
from loguru import logger
from scipy import stats


def check_data_drift(X_train: pd.DataFrame, X_new: pd.DataFrame, threshold: float = 0.05) -> bool:
    """Проверяет дрейф данных с помощью KS-теста."""
    numeric_cols = X_train.select_dtypes(include=["number"]).columns
    drift_detected = False

    logger.info(f"Проверка дрейфа данных (KS-тест, порог p={threshold}):")
    for col in numeric_cols:
        train_vals = X_train[col].dropna().values
        new_vals = X_new[col].dropna().values

        if len(train_vals) < 2 or len(new_vals) < 2:
            continue

        ks_stat, p_value = stats.ks_2samp(train_vals, new_vals)
        if p_value < threshold:
            drift_detected = True
            logger.warning(f"  [!] ДРЕЙФ | {col:<20} | p-value={p_value:.4f}")
        else:
            logger.info(f"  [OK]  | {col:<20} | p-value={p_value:.4f}")

    if drift_detected:
        logger.warning("Обнаружен дрейф данных!")
    else:
        logger.info("Дрейф не обнаружен.")

    return drift_detected
