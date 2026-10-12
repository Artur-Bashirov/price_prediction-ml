"""Создание ML-пайплайна."""

from price_prediction.config import get_settings
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def create_pipeline(feature_cols: list[str]) -> Pipeline:
    """Создает и возвращает настроенный пайплайн XGBoost."""
    settings = get_settings()

    # Циклические признаки не нужно масштабировать, они уже в диапазоне [-1, 1]
    cyclic_cols = ["weekday_sin", "weekday_cos", "month_sin", "month_cos"]
    numeric_cols = [c for c in feature_cols if c not in cyclic_cols]

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_cols), ("cyc", "passthrough", cyclic_cols)],
        remainder="drop",  # Игнорируем колонки 'date' и 'target', если они попадут в X
    )

    model = XGBClassifier(
        n_estimators=settings.models.xgboost.n_estimators,
        learning_rate=settings.models.xgboost.learning_rate,
        max_depth=settings.models.xgboost.max_depth,
        subsample=settings.models.xgboost.subsample,
        colsample_bytree=settings.models.xgboost.colsample_bytree,
        reg_alpha=settings.models.xgboost.reg_alpha,
        reg_lambda=settings.models.xgboost.reg_lambda,
        random_state=settings.models.xgboost.random_state,
        eval_metric=settings.models.xgboost.eval_metric,
        verbosity=settings.models.xgboost.verbosity,
        n_jobs=-1,
    )

    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])
