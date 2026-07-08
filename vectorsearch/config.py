"""
Конфигурация модуля векторного поиска.

Все значения читаются из переменных окружения, чтобы не хардкодить
адрес контейнера Qdrant (его поднимает Даня-Г в docker-compose) и
имя модели эмбеддингов.
"""

import os
from dataclasses import dataclass


@dataclass
class Settings:
    # Адрес Qdrant. Когда Даня-Г поднимет контейнер, сюда прилетит
    # его host:port (например "qdrant" в docker-сети или "localhost").
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6343"))

    # Имя коллекции для чанков текста.
    collection_name: str = os.getenv("QDRANT_COLLECTION", "chunks_collection")

    # Модель эмбеддингов. multilingual-e5-base хорошо работает с русским
    # текстом и не слишком тяжёлая. Можно заменить на любую модель
    # sentence-transformers.
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL", "intfloat/multilingual-e5-base"
    )

    # Размерность вектора для выбранной модели.
    # multilingual-e5-base -> 768. Если меняете модель, поменяйте и это.
    vector_size: int = int(os.getenv("EMBEDDING_DIM", "768"))

    # Сколько чанков грузить в Qdrant за один batch upsert.
    batch_size: int = int(os.getenv("INDEX_BATCH_SIZE", "64"))


settings = Settings()
