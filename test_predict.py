"""Скрипт для тестирования инференса модели."""

from loguru import logger
from price_prediction.data.features import engineer_features
from price_prediction.data.loading import load_and_clean_data
from price_prediction.models.predict import load_model, predict


def main():
    logger.info("🧪 Тестирование инференса модели...")

    # 1. Загружаем обученную модель
    model, feature_cols = load_model()
    logger.info(f" Модель загружена. Количество признаков: {len(feature_cols)}")

    # 2. Загружаем данные и создаем признаки
    df = load_and_clean_data()
    df = engineer_features(df)

    # 3. Берем последние 5 дней как "новые" данные
    new_data = df.tail(5).copy()
    logger.info(
        f" Тестовые данные: последние 5 дней ({new_data['date'].min().date()} -> {new_data['date'].max().date()})"
    )

    # 4. Делаем предсказания
    predictions = predict(new_data, model, feature_cols)

    # 5. Выводим результаты
    logger.info("\n Результаты предсказаний:")
    print(predictions[["date", "predicted_class", "probability_up", "confidence"]].to_string(index=False))

    # 6. Статистика
    up_count = (predictions["predicted_class"] == 1).sum()
    down_count = (predictions["predicted_class"] == 0).sum()
    logger.info(f"\n Статистика: Up={up_count}, Down={down_count}")
    logger.info(" Тестирование инференса завершено!")


if __name__ == "__main__":
    main()
