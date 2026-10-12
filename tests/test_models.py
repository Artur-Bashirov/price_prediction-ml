"""Тесты для модулей модели."""

import pandas as pd
from price_prediction.models.pipeline import create_pipeline


def test_create_pipeline():
    """Тест создания пайплайна."""
    feature_cols = ["close_lag1", "volatility_5", "weekday_sin", "weekday_cos"]
    pipeline = create_pipeline(feature_cols)

    # Проверяем, что пайплайн создан и имеет нужные шаги
    assert pipeline is not None
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_pipeline_predict():
    """Тест того, что пайплайн может делать предсказания на синтетических данных."""
    feature_cols = ["close_lag1", "volatility_5", "weekday_sin", "weekday_cos"]
    pipeline = create_pipeline(feature_cols)

    # Создаем синтетические данные для теста
    X_dummy = pd.DataFrame(
        {
            "close_lag1": [100.0, 101.0, 102.0],
            "volatility_5": [0.01, 0.02, 0.015],
            "weekday_sin": [0.0, 0.5, 1.0],
            "weekday_cos": [1.0, 0.5, 0.0],
        }
    )
    y_dummy = pd.Series([0, 1, 0])

    # Обучаем на синтетике
    pipeline.fit(X_dummy, y_dummy)

    # Делаем предсказание
    predictions = pipeline.predict(X_dummy)
    probas = pipeline.predict_proba(X_dummy)

    assert len(predictions) == 3
    assert len(probas) == 3
    assert all(p in [0, 1] for p in predictions)
