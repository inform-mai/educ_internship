"""
Индексация чанков в Qdrant.

Вход — список чанков в формате, который отдаёт main.process() у Дани-П:

    {
        "text": "кусок текста",
        "index": 0,
        "source": "test.txt",
        "start": 0,
        "end": 200,
    }

Если у него на практике поля называются иначе (например "content" вместо
"text", или нет "start"/"end") — поправить нужно только в _chunk_id и
_build_payload ниже, остальной код не трогаем.
"""

import hashlib
from typing import Any, Dict, List

from . import store
from .config import settings
from .embeddings import embed_passages


def index_chunks(chunks: List[Dict[str, Any]]) -> int:
    """
    Эмбеддит и загружает чанки в Qdrant батчами.

    Args:
        chunks: список чанков от Дани-П (process("file.txt"))

    Returns:
        количество проиндексированных чанков
    """
    if not chunks:
        return 0

    store.ensure_collection()

    total = 0
    batch_size = settings.batch_size
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [_get_text(c) for c in batch]
        vectors = embed_passages(texts)
        ids = [_chunk_id(c) for c in batch]
        payloads = [_build_payload(c) for c in batch]

        store.upsert_points(ids=ids, vectors=vectors, payloads=payloads)
        total += len(batch)

    return total


def _get_text(chunk: Dict[str, Any]) -> str:
    # На случай, если поле называется иначе (content/body и т.п.)
    return chunk.get("text") or chunk.get("content") or chunk.get("body") or ""


def _chunk_id(chunk: Dict[str, Any]) -> int:
    """
    Qdrant требует id: целое число или UUID. Строим стабильный целочисленный
    id из source+index, чтобы повторная индексация того же файла
    перезаписывала те же точки, а не плодила дубликаты.
    """
    source = str(chunk.get("source", ""))
    index = str(chunk.get("index", ""))
    key = f"{source}:{index}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    # Берём первые 15 hex-символов, чтобы влезть в 63-битный int.
    return int(digest[:15], 16)


def _build_payload(chunk: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "text": _get_text(chunk),
        "source": chunk.get("source"),
        "index": chunk.get("index"),
        "start": chunk.get("start"),
        "end": chunk.get("end"),
    }
