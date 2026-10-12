"""Конфигурация проекта."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DataConfig(BaseModel):
    """Конфигурация данных."""

    raw_path: str
    date_column: str = "date"
    target_column: str = "target"
    decimal_separator: str = ","


class FeaturesConfig(BaseModel):
    """Конфигурация признаков."""

    sma_windows: list[int] = Field(default_factory=lambda: [5, 20, 50])
    vol_windows: list[int] = Field(default_factory=lambda: [5, 20])
    rsi_period: int = 14
    clip_quantiles: tuple[float, float] = (0.01, 0.99)


class XGBoostConfig(BaseModel):
    """Конфигурация XGBoost."""

    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    random_state: int = 42
    eval_metric: str = "logloss"
    verbosity: int = 0


class DummyConfig(BaseModel):
    """Конфигурация Dummy модели."""

    strategy: str = "stratified"
    random_state: int = 42


class ModelsConfig(BaseModel):
    """Конфигурация моделей."""

    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    dummy: DummyConfig = Field(default_factory=DummyConfig)


class TrainingConfig(BaseModel):
    """Конфигурация обучения."""

    train_test_split: float = 0.8
    cv_folds: int = 5
    random_state: int = 42


class DriftConfig(BaseModel):
    """Конфигурация дрейфа."""

    threshold: float = 0.05


class Settings(BaseSettings):  # type: ignore[misc]
    """Основные настройки проекта."""
    data: DataConfig
    features: FeaturesConfig
    models: ModelsConfig
    training: TrainingConfig
    drift: DriftConfig
    # business: BusinessConfig  # ← закомментируйте или удалите эту строку

    class Config:
        env_file = ".env"


def load_config(config_path: str | Path = "configs/config.yaml") -> Settings:
    """Загрузить конфигурацию из YAML файла."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return Settings(**config_dict)


# Глобальный экземпляр настроек
settings: Settings | None = None


def get_settings() -> Settings:
    """Получить настройки (singleton)."""
    global settings
    if settings is None:
        settings = load_config()
    return settings
