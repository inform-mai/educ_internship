"""
Генерация эмбеддингов для чанков текста и поисковых запросов.

Используется sentence-transformers с multilingual-моделью (по умолчанию
intfloat/multilingual-e5-base — хорошо работает с русским).

Модель E5 требует префиксов "query: " и "passage: " перед текстом —
это часть её обучающего протокола, без них качество поиска заметно
падает. Поэтому два отдельных метода: embed_passages / embed_query.
"""

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Модель грузится один раз и кешируется на весь процесс."""
    return SentenceTransformer(settings.embedding_model_name)


def embed_passages(texts: List[str]) -> List[List[float]]:
    """
    Эмбеддинг для чанков текста, которые кладутся в индекс.

    Args:
        texts: список текстов чанков (chunk["text"] от Дани-П)

    Returns:
        список векторов (list[float]) в том же порядке
    """
    if not texts:
        return []
    model = _get_model()
    prefixed = [f"passage: {t}" for t in texts]
    vectors = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    return _to_list(vectors)


def embed_query(query: str) -> List[float]:
    """Эмбеддинг для одного поискового запроса пользователя."""
    model = _get_model()
    vector = model.encode(
        [f"query: {query}"], normalize_embeddings=True, show_progress_bar=False
    )[0]
    return _to_list(vector)


def _to_list(vectors: np.ndarray) -> List[List[float]]:
    if vectors.ndim == 1:
        return vectors.tolist()
    return [v.tolist() for v in vectors]
