"""Модуль для инференса (предсказания) модели."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from loguru import logger


def load_model(model_dir: str | Path = "models") -> tuple[Any, list[str]]:
    """Загрузить обученную модель и список признаков."""
    model_dir = Path(model_dir)

    model_path = model_dir / "best_model.pkl"
    features_path = model_dir / "feature_cols.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}. Сначала запустите обучение.")
    if not features_path.exists():
        raise FileNotFoundError(f"Список признаков не найден: {features_path}. Сначала запустите обучение.")

    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)

    logger.info(f"Модель загружена из {model_path}")
    return model, feature_cols


def predict(df: pd.DataFrame, model: Any, feature_cols: list[str], confidence_threshold: float = 0.5) -> pd.DataFrame:
    """
    Сделать предсказания для новых данных.

    Args:
        df: DataFrame с исходными данными (должен содержать все необходимые колонки)
        model: Обученная модель (pipeline)
        feature_cols: Список признаков для модели
        confidence_threshold: Порог уверенности для классификации

    Returns:
        DataFrame с предсказаниями
    """
    # Проверяем наличие необходимых колонок
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Отсутствуют необходимые колонки: {missing_cols}")

    # Выбираем только нужные признаки
    X = df[feature_cols]

    # Делаем предсказания
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Создаем результат
    result = pd.DataFrame(
        {
            "date": df["date"],
            "predicted_class": predictions,
            "probability_up": probabilities[:, 1],
            "probability_down": probabilities[:, 0],
        }
    )

    # Добавляем уверенность
    result["confidence"] = result.apply(lambda row: max(row["probability_up"], row["probability_down"]), axis=1)

    # Добавляем маршрут (auto/manual) на основе уверенности
    result["route"] = result["confidence"].apply(lambda x: "auto" if x >= confidence_threshold else "manual")

    logger.info(
        f"[PREDICT] Всего: {len(result)}, Авто: {(result['route'] == 'auto').sum()}, Ручная проверка: {(result['route'] == 'manual').sum()}"
    )

    return result
