"""
Обёртка над qdrant-client: подключение, создание коллекции, upsert, поиск.

Держим все прямые обращения к Qdrant в одном месте, чтобы верхний
уровень (indexer.py, search.py) не знал деталей клиента.
"""

from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .config import settings


def get_client() -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def ensure_collection(client: Optional[QdrantClient] = None) -> None:
    """
    Создаёт коллекцию, если её ещё нет. Безопасно вызывать несколько раз —
    повторный вызов ничего не сломает.
    """
    client = client or get_client()
    existing = [c.name for c in client.get_collections().collections]
    if settings.collection_name in existing:
        return

    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config=qmodels.VectorParams(
            size=settings.vector_size,
            distance=qmodels.Distance.COSINE,
        ),
    )


def upsert_points(
    ids: List[int],
    vectors: List[List[float]],
    payloads: List[Dict[str, Any]],
    client: Optional[QdrantClient] = None,
) -> None:
    client = client or get_client()
    points = [
        qmodels.PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
        for i in range(len(ids))
    ]
    client.upsert(collection_name=settings.collection_name, points=points)


def search_points(
    query_vector: List[float],
    top_k: int = 5,
    query_filter: Optional[qmodels.Filter] = None,
    client: Optional[QdrantClient] = None,
) -> List[qmodels.ScoredPoint]:
    client = client or get_client()
    response = client.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
    )
    return response.points
