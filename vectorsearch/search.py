"""
Поиск по проиндексированным чанкам.

vector_search   — чистый семантический поиск по Qdrant.
hybrid_search   — вектор + результаты обхода графа (Memgraph), если
                  передан graph_client. Если его нет — просто откатывается
                  на vector_search, ничего не ломая.

Формат graph_client — намеренно абстрактный (Protocol), потому что
конкретную реализацию похода в Memgraph будет делать либо Катя,
либо Дима при сборке агента. От него нужен только один метод:
    graph_client.search_entities(query: str, top_k: int) -> list[dict]
где каждый dict — что-то вроде
    {"entity": "...", "related_text": "...", "score": 0.8}
"""

from typing import Any, Dict, List, Optional, Protocol

from . import store
from .embeddings import embed_query


class GraphClient(Protocol):
    def search_entities(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        ...


def vector_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Семантический поиск по чанкам.

    Returns:
        список {"text", "source", "index", "score"}, отсортированный
        по убыванию score (косинусное сходство, т.к. коллекция
        создана с distance=COSINE).
    """
    query_vector = embed_query(query)
    hits = store.search_points(query_vector=query_vector, top_k=top_k)

    results = []
    for hit in hits:
        payload = hit.payload or {}
        results.append(
            {
                "text": payload.get("text"),
                "source": payload.get("source"),
                "index": payload.get("index"),
                "score": hit.score,
                "origin": "vector",
            }
        )
    return results


def hybrid_search(
    query: str,
    top_k: int = 5,
    graph_client: Optional[GraphClient] = None,
    graph_top_k: int = 5,
    vector_weight: float = 0.6,
    graph_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Объединяет векторный поиск по чанкам с поиском по графу знаний.

    Если graph_client не передан, ведёт себя как vector_search —
    это позволяет подключить граф позже, не переписывая вызывающий код.

    Слияние — простое взвешенное объединение по score (нормализованному
    в рамках каждого источника), а не RRF: с двумя разнородными
    источниками (текст vs сущности) это проще объяснить и настроить
    вручную. Если после тестов результаты будут плохо ранжироваться —
    имеет смысл заменить на RRF (Reciprocal Rank Fusion).
    """
    vector_results = vector_search(query, top_k=top_k)

    if graph_client is None:
        return vector_results

    graph_hits = graph_client.search_entities(query, top_k=graph_top_k)
    graph_results = [
        {
            "text": hit.get("related_text") or hit.get("entity"),
            "source": "graph",
            "entity": hit.get("entity"),
            "score": hit.get("score", 0.0) * graph_weight,
            "origin": "graph",
        }
        for hit in graph_hits
    ]

    for r in vector_results:
        r["score"] = r["score"] * vector_weight

    combined = vector_results + graph_results
    combined.sort(key=lambda r: r["score"], reverse=True)
    return combined[:top_k]
