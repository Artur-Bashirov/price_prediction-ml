"""Скрипт обучения модели."""

from pathlib import Path

import joblib
from loguru import logger
from price_prediction.config import get_settings
from price_prediction.data.features import engineer_features
from price_prediction.data.loading import load_and_clean_data
from price_prediction.models.pipeline import create_pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


def main() -> None:
    logger.info(" Запуск процесса обучения модели...")
    settings = get_settings()

    # 1. Загрузка и подготовка данных
    logger.info("Загрузка и инженерия признаков...")
    df = load_and_clean_data()
    df = engineer_features(df)

    # 2. Разделение по дате (чтобы избежать утечки данных из будущего)
    split_date = df["date"].quantile(settings.training.train_test_split)
    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()

    target_col = settings.data.target_column
    exclude_cols = ["date", target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    logger.info(f"Размер Train: {len(X_train)}, Test: {len(X_test)}")
    logger.info(f"Дата разделения: {split_date.date()}")

    # 3. Создание и обучение пайплайна
    logger.info("Инициализация и обучение пайплайна...")
    pipeline = create_pipeline(feature_cols)
    pipeline.fit(X_train, y_train)

    # 4. Оценка на тестовой выборке
    logger.info("Оценка модели на тестовых данных...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    logger.info(f" Test Accuracy: {acc:.4f}")
    logger.info(f" Test ROC-AUC:  {auc:.4f}")
    logger.info(
        f" Classification Report:\n{classification_report(y_test, y_pred, target_names=['Down (0)', 'Up (1)'])}"
    )

    # 5. Сохранение артефактов
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "best_model.pkl"
    features_path = models_dir / "feature_cols.pkl"

    joblib.dump(pipeline, model_path)
    joblib.dump(feature_cols, features_path)

    logger.info(f" Модель сохранена: {model_path}")
    logger.info(f" Список признаков сохранен: {features_path}")
    logger.info(" Обучение успешно завершено!")


if __name__ == "__main__":
    main()
