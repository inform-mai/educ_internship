import os
import asyncio
import aiohttp
import json
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)
# URL вашего Qdrant-инстанса
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")  # без завершающего слэша


class QdrantError(RuntimeError):
    pass


class QdrantAsync:
    """
    Минимальный асинхронный клиент для Qdrant через REST API.
    Реализует базовые функции:
      - Проверка/создание коллекции
      - Добавление (upsert) одной или нескольких точек (текста + эмбеддинга)
    """

    def __init__(self, session: aiohttp.ClientSession, base_url: str = QDRANT_URL, timeout_s: float = 30.0) -> None:
        self.session = session
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _url(self, path: str) -> str:
        """Формирует полный URL запроса."""
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"

    def _headers(self) -> Dict[str, str]:
        """Возвращает заголовки для HTTP-запросов."""
        return {"Content-Type": "application/json"}

    async def _read_json(self, resp: aiohttp.ClientResponse) -> Any:
        """Считывает и возвращает JSON-ответ, выбрасывает ошибку при неуспехе."""
        text = await resp.text()
        if not (200 <= resp.status < 300):
            try:
                data = json.loads(text)
            except Exception:
                data = text
            raise QdrantError(f"HTTP {resp.status}: {data}")
        try:
            return json.loads(text)
        except Exception:
            return text

    async def collection_exists(self, name: str) -> bool:
        """Проверяет, существует ли коллекция с заданным именем."""
        url = self._url(f"/collections/{name}")
        async with self.session.get(url, headers=self._headers(), timeout=self.timeout_s) as r:
            if r.status == 200:
                return True
            if r.status == 404:
                return False
            await self._read_json(r)
            return False

    async def scroll_points(
        self,
        collection: str,
        limit: int = 100,
        *,
        filter: Optional[Dict[str, Any]] = None,
        with_payload: bool = True,
        with_vector: bool = False,
        offset: Optional[Union[int, str]] = None,
    ) -> Dict[str, Any]:
        """
        Возвращает страницу точек из коллекции с возможностью постраничной прокрутки.

        Формат ответа Qdrant:
          {
            "result": {
              "points": [...],
              "next_page_offset": <id или None>
            },
            ...
          }

        :param collection: Имя коллекции
        :param limit: Количество точек за запрос
        :param filter: Опциональный Qdrant-фильтр
        :param with_payload: Возвращать payload
        :param with_vector: Возвращать вектора
        :param offset: Смещение (id последней точки с предыдущей страницы)
        """
        url = self._url(f"/collections/{collection}/points/scroll")
        body: Dict[str, Any] = {
            "limit": int(limit),
            "with_payload": with_payload,
            "with_vector": with_vector,
        }
        if filter is not None:
            body["filter"] = filter
        if offset is not None:
            body["offset"] = offset

        async with self.session.post(
            url,
            headers=self._headers(),
            json=body,
            timeout=self.timeout_s,
        ) as r:
            data = await self._read_json(r)
        return data

    # === ЧАСТИЧНОЕ ОБНОВЛЕНИЕ PAYLOAD ===

    async def set_payload(
        self,
        collection: str,
        point_id: Union[int, str],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Частично обновляет payload точки (merge).
        Существующие поля сохраняются, переданные ключи перезаписываются.

        :param collection: Имя коллекции
        :param point_id: ID точки
        :param payload: Словарь с обновляемыми полями payload
        """
        url = self._url(f"/collections/{collection}/points/payload")
        body = {
            "points": [point_id],
            "payload": payload,
        }
        async with self.session.post(
            url,
            headers=self._headers(),
            json=body,
            timeout=self.timeout_s,
        ) as r:
            data = await self._read_json(r)
        return data

    # === ОБОГАЩЕНИЕ ЧАНКОВ СЕМАНТИЧЕСКИМИ СУЩНОСТЯМИ ===

    async def enrich_chunks_with_semantic_entities(
        self,
        chunks_collection: str,
        entities_collection: str,
        *,
        text_field: str = "text",
        similarity_threshold: float = 0.75,
        search_limit: int = 10,
        scroll_page_size: int = 128,
        sem_entities_field: str = "sem_entyties",
        verbose: bool = True,
    ) -> None:
        """
        Берёт ВСЕ чанки из коллекции `chunks_collection`,
        для каждого чанка использует его вектор как запрос к коллекции сущностей
        `entities_collection` и пишет в payload поле `sem_entyties`
        (строка с именами сущностей через запятую).
        """

        offset = None
        total_points = 0
        updated_points = 0
        page_idx = 0

        while True:
            raw_page = await self.scroll_points(
                collection=chunks_collection,
                limit=scroll_page_size,
                with_payload=True,
                with_vector=True,
                offset=offset,
            )

            # 🔧 Критичный момент:
            #  - если scroll_points вернул полный ответ Qdrant -> берем raw_page["result"]
            #  - если scroll_points уже вернул только result -> используем его как есть
            result = raw_page.get("result") if isinstance(raw_page, dict) else None
            if result is None:
                result = raw_page  # значит scroll_points возвращает уже `result`

            points = (result or {}).get("points") or []
            offset = (result or {}).get("next_page_offset")
            page_idx += 1

            if verbose:
                print(f"[Qdrant] page {page_idx}: {len(points)} points, next_offset={offset}")

            if not points:
                break

            for point in points:
                total_points += 1
                point_id = point.get("id")
                payload = point.get("payload") or {}

                # Берём вектор чанка
                vector_raw = point.get("vector")
                if vector_raw is None:
                    continue

                if isinstance(vector_raw, dict):
                    first_key = next(iter(vector_raw))
                    chunk_vector = vector_raw[first_key]
                else:
                    chunk_vector = vector_raw

                if not isinstance(chunk_vector, list):
                    continue

                # Семантический поиск по коллекции сущностей
                search_res = await self.search(
                    collection=entities_collection,
                    vector=chunk_vector,
                    limit=search_limit,
                    with_payload=True,
                    with_vector=False,
                    score_threshold=similarity_threshold,
                )

                hits = (search_res or {}).get("result") or []

                entity_names: List[str] = []
                for hit in hits:
                    h_payload = hit.get("payload") or {}
                    name = (
                        h_payload.get("canonical")
                        or h_payload.get("text")
                        or h_payload.get("name")
                    )
                    if isinstance(name, str):
                        name = name.strip()
                        if name:
                            entity_names.append(name)

                # дедуп
                seen = set()
                unique_entities: List[str] = []
                for name in entity_names:
                    if name not in seen:
                        seen.add(name)
                        unique_entities.append(name)

                if not unique_entities:
                    continue

                sem_entities_value = ", ".join(unique_entities)

                await self.set_payload(
                    collection=chunks_collection,
                    point_id=point_id,
                    payload={sem_entities_field: sem_entities_value},
                )
                updated_points += 1

            if offset is None:
                break

        if verbose:
            print(
                f"[Qdrant] enrich_chunks_with_semantic_entities: "
                f"processed={total_points}, updated={updated_points}"
            )

    async def propagate_entities_to_chunks(
        self,
        *,
        entities_collection: str,
        chunks_collection: str,
        entity_name_field: str = "canonical",
        chunk_text_field: str = "text",
        sem_entities_field: str = "sem_ent_2",
        similarity_threshold: float = 0.6,
        search_limit: int = 200,
        scroll_page_size: int = 128,
        use_hybrid: bool = False,
        sparse_indices_field: str = "sparse_indices",
        sparse_values_field: str = "sparse_values",
        require_text_match: bool = True,
        verbose: bool = True,
        debug_samples: int = 5,
    ) -> None:
        """
        Идём по КАЖДОЙ сущности в коллекции `entities_collection` и ищем релевантные чанки
        в коллекции `chunks_collection`. Если сущность считается найденной в чанке,
        добавляем её в поле `sem_ent_2` в payload чанка.

        Логика "сущность действительно находится в чанке":
          - по умолчанию: нормализованное имя сущности должно присутствовать в
            нормализованном тексте чанка (подстрока или по словам).

        :param entities_collection: Коллекция с сущностями
        :param chunks_collection: Коллекция с текстовыми чанками
        :param entity_name_field: Поле в payload сущности с её каноническим именем
        :param chunk_text_field: Поле в payload чанка с текстом чанка
        :param sem_entities_field: Поле в payload чанка для записи связанных сущностей
        :param similarity_threshold: Порог score для отбора кандидатов
        :param search_limit: Максимум кандидатов-чанков на сущность
        :param scroll_page_size: Размер страницы при обходе коллекции сущностей
        :param use_hybrid: Если True — использовать hybrid_search вместо search
        :param sparse_indices_field: Имя поля с индексами sparse-вектора
        :param sparse_values_field: Имя поля со значениями sparse-вектора
        :param require_text_match: Обязательно ли проверять фактическое вхождение
                                   имени сущности в текст чанка
        :param verbose: Печатать базовые логи
        :param debug_samples: Сколько первых неудачных совпадений показывать в лог
                              (чтобы понять, почему они отфильтровались)
        """
        import re
        from typing import List, Any

        def normalize(s: str) -> str:
            s = s.lower()
            # убираем лишнюю пунктуацию, приводим к "словам"
            s = re.sub(r"[^\w]+", " ", s, flags=re.UNICODE)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        def entity_matches_text(entity_name: str, text: str) -> bool:
            """
            Базовая эвристика:
              - нормализуем текст и имя
              - сначала пробуем подстроку,
              - затем — по словам: все "достаточно длинные" слова имени
                должны встретиться в тексте
            """
            norm_text = normalize(text)
            norm_name = normalize(entity_name)

            if not norm_text or not norm_name:
                return False

            # прямое подстроковое вхождение
            if norm_name in norm_text:
                return True

            # по словам: чтобы не было совпадения по одному короткому слову
            tokens = [t for t in norm_name.split() if len(t) > 3]
            if not tokens:
                return False

            return all(t in norm_text for t in tokens)

        offset = None
        total_entities = 0
        total_hits = 0
        updated_chunks = 0
        page_idx = 0
        debug_shown = 0

        while True:
            raw_page = await self.scroll_points(
                collection=entities_collection,
                limit=scroll_page_size,
                with_payload=True,
                with_vector=True,
                offset=offset,
            )

            # поддержка обоих вариантов scroll_points (полный ответ или только result)
            result = raw_page.get("result") if isinstance(raw_page, dict) else None
            if result is None:
                result = raw_page

            points = (result or {}).get("points") or []
            offset = (result or {}).get("next_page_offset")
            page_idx += 1

            if verbose:
                print(f"[Qdrant] entities page {page_idx}: {len(points)} entities, next_offset={offset}")

            if not points:
                break

            for ent_point in points:
                total_entities += 1

                ent_payload = ent_point.get("payload") or {}
                vector_raw = ent_point.get("vector")

                # --- имя сущности ---
                name = (
                    ent_payload.get(entity_name_field)
                    or ent_payload.get("canonical")
                    or ent_payload.get("text")
                    or ent_payload.get("name")
                )
                if not isinstance(name, str) or not name.strip():
                    continue
                name = name.strip()

                # --- вектор сущности ---
                if vector_raw is None:
                    continue

                if isinstance(vector_raw, dict):
                    first_key = next(iter(vector_raw))
                    entity_vector = vector_raw[first_key]
                else:
                    entity_vector = vector_raw

                if not isinstance(entity_vector, list):
                    continue

                # --- Поиск чанков ---
                if use_hybrid:
                    sparse_indices = ent_payload.get(sparse_indices_field) or []
                    sparse_values = ent_payload.get(sparse_values_field) or []

                    search_res = await self.hybrid_search(
                        collection=chunks_collection,
                        dense_vector=entity_vector,
                        sparse_indices=sparse_indices,
                        sparse_values=sparse_values,
                        limit=search_limit,
                        filter=None,
                        with_payload=True,
                        with_vector=False,
                    )
                else:
                    search_res = await self.search(
                        collection=chunks_collection,
                        vector=entity_vector,
                        limit=search_limit,
                        with_payload=True,
                        with_vector=False,
                        score_threshold=similarity_threshold,
                    )

                hits = (search_res or {}).get("result") or []
                if not hits:
                    continue

                for hit in hits:
                    total_hits += 1
                    chunk_id = hit.get("id")
                    chunk_payload = hit.get("payload") or {}

                    # --- достаём текст чанка ---
                    text = chunk_payload.get(chunk_text_field)
                    # fallback: часто поле может называться иначе
                    if not isinstance(text, str):
                        text = chunk_payload.get("text") or chunk_payload.get("chunk") or chunk_payload.get("chunk_text")

                    if not isinstance(text, str):
                        # если нет текста вообще, то проверить вхождение невозможно
                        if require_text_match:
                            if debug_shown < debug_samples and verbose:
                                print(
                                    f"[DEBUG] chunk {chunk_id}: no text field "
                                    f"('{chunk_text_field}') in payload, skip"
                                )
                                debug_shown += 1
                            continue
                        else:
                            text = ""  # чтобы ниже не упасть

                    if require_text_match and not entity_matches_text(name, text):
                        if debug_shown < debug_samples and verbose:
                            print(
                                f"[DEBUG] no text match for entity '{name}' "
                                f"in chunk {chunk_id} (similarity ok, но текст не содержит имя)"
                            )
                            debug_shown += 1
                        continue

                    # --- обновляем список сущностей в чанке ---
                    existing = chunk_payload.get(sem_entities_field)
                    items: List[str] = []

                    if isinstance(existing, str):
                        parts = [p.strip() for p in existing.split(",")]
                        items = [p for p in parts if p]
                    elif isinstance(existing, list):
                        items = [str(p).strip() for p in existing if str(p).strip()]

                    seen = set(items)
                    if name not in seen:
                        items.append(name)
                        seen.add(name)
                    else:
                        # уже есть, можно не писать payload
                        continue

                    new_value: Any = ", ".join(items)

                    await self.set_payload(
                        collection=chunks_collection,
                        point_id=chunk_id,
                        payload={sem_entities_field: new_value},
                    )
                    updated_chunks += 1

            if offset is None:
                break

        if verbose:
            print(
                f"[Qdrant] propagate_entities_to_chunks: "
                f"entities={total_entities}, hits={total_hits}, updated_chunks={updated_chunks}"
            )

    async def ensure_collection(self, name: str, vector_size: int, distance: str = "Cosine", on_disk: Optional[bool] = None) -> Dict[str, Any]:
        """Создаёт коллекцию, если она отсутствует.

        :param name: Имя коллекции
        :param vector_size: Размерность вектора (len(embedding))
        :param distance: Метрика расстояния (Cosine | Euclid | Dot)
        :param on_disk: Если True — хранить вектора на диске (опционально)
        """
        if await self.collection_exists(name):
            return {"status": "already_exists", "collection": name}

        url = self._url(f"/collections/{name}")
        body: Dict[str, Any] = {"vectors": {"size": vector_size, "distance": distance}}
        if on_disk is not None:
            body["vectors"]["on_disk"] = bool(on_disk)

        async with self.session.put(url, headers=self._headers(), json=body, timeout=self.timeout_s) as r:
            data = await self._read_json(r)
        return data

    async def upsert_point(self, collection: str, vector: Sequence[float], payload: Optional[Dict[str, Any]] = None, point_id = "") -> Dict[str, Any]:
        """Добавляет или обновляет одну точку (текстовый чанк + эмбеддинг).

        :param collection: Имя коллекции в Qdrant
        :param vector: Эмбеддинг (список чисел)
        :param payload: Метаданные (например, текст, источник, сущности)
        :param point_id: Необязательный ID точки
        """
        url = self._url(f"/collections/{collection}/points")
        point: Dict[str, Any] = {"vector": vector, "payload": payload or {}}
        logger.info(f"Adding point_id={id}, {payload}")
        point_id = str(uuid4())
        if point_id is not None:
            point["id"] = point_id

        body = {"points": [point]}
        async with self.session.put(url, headers=self._headers(), json=body, timeout=self.timeout_s) as r:
            data = await self._read_json(r)
        return data

    async def upsert_points_batch(self, collection: str, vectors: Sequence[Sequence[float]], payloads: Optional[Sequence[Optional[Dict[str, Any]]]] = None, point_ids: Optional[Sequence[Union[int, str]]] = None) -> Dict[str, Any]:
        """Массовое добавление нескольких точек в коллекцию.

        :param collection: Имя коллекции
        :param vectors: Список векторов
        :param payloads: Список payload-ов (метаданных)
        :param point_ids: Необязательные ID для каждой точки
        """
        if payloads is None:
            payloads = [None] * len(vectors)
        if point_ids is None:
            points = [{"vector": list(map(float, v)), "payload": (p or {})} for v, p in zip(vectors, payloads)]
        else:
            if not (len(point_ids) == len(vectors) == len(payloads)):
                raise ValueError("point_ids, vectors и payloads должны быть одинаковой длины")
            points = [{"id": pid, "vector": list(map(float, v)), "payload": (p or {})} for pid, v, p in zip(point_ids, vectors, payloads)]

        url = self._url(f"/collections/{collection}/points")
        body = {"points": points}
        async with self.session.put(url, headers=self._headers(), json=body, timeout=self.timeout_s) as r:
            data = await self._read_json(r)
        return data


    async def list_collections(self) -> List[str]:
        """Возвращает список имён всех коллекций."""
        url = self._url("/collections")
        async with self.session.get(url, headers=self._headers(), timeout=self.timeout_s) as r:
            data = await self._read_json(r)
        # Ожидаемый формат: {"result": {"collections": [{"name": "..."}, ...]}}
        cols = data.get("result", {}).get("collections", [])
        return [c.get("name") for c in cols]

    async def delete_collection(self, name: str) -> Dict[str, Any]:
        """Удаляет коллекцию по имени (необратимо)."""
        url = self._url(f"/collections/{name}")
        async with self.session.delete(url, headers=self._headers(), timeout=self.timeout_s) as r:
            data = await self._read_json(r)
        return data

    async def search(
        self,
        collection: str,
        vector: Sequence[float],
        limit: int = 5,
        *,
        filter: Optional[Dict[str, Any]] = None,
        with_payload: bool = True,
        with_vector: bool = False,
        score_threshold: Optional[float] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Классический векторный поиск по dense-вектору.

        :param collection: Имя коллекции
        :param vector: Эмбеддинг запроса
        :param limit: Количество результатов
        :param filter: Qdrant-фильтр по payload (must/should/must_not)
        :param with_payload: Возвращать payload документов
        :param with_vector: Возвращать исходные вектора
        :param score_threshold: Порог схожести (опционально)
        :param offset: Смещение (пагинация)
        """
        url = self._url(f"/collections/{collection}/points/search")
        body: Dict[str, Any] = {
            "vector": list(map(float, vector)),
            "limit": int(limit),
            "with_payload": with_payload,
            "with_vector": with_vector,
        }
        if filter is not None:
            body["filter"] = filter
        if score_threshold is not None:
            body["score_threshold"] = float(score_threshold)
        if offset is not None:
            body["offset"] = int(offset)

        async with self.session.post(url, headers=self._headers(), json=body, timeout=self.timeout_s) as r:
            data = await self._read_json(r)
        return data

    async def hybrid_search(
        self,
        collection: str,
        dense_vector: Sequence[float],
        sparse_indices: Sequence[int],
        sparse_values: Sequence[float],
        limit: int = 5,
        *,
        filter: Optional[Dict[str, Any]] = None,
        fusion: str = "rrf",
        with_payload: bool = True,
        with_vector: bool = False,
    ) -> Dict[str, Any]:
        """Гибридный поиск (dense + sparse) через /points/query c fusion (по умолчанию RRF).

        :param collection: Имя коллекции
        :param dense_vector: Плотный эмбеддинг запроса
        :param sparse_indices: Индексы токенов для sparse-вектора
        :param sparse_values: Значения TF-IDF/BM25/около того для соответствующих индексов
        :param limit: Количество результатов
        :param filter: Qdrant-фильтр по payload (опционально)
        :param fusion: Стратегия объединения результатов (rrf, reciprocalsum, ... в зависимости от версии Qdrant)
        :param with_payload: Возвращать payload
        :param with_vector: Возвращать вектор
        """
        url = self._url(f"/collections/{collection}/points/query")
        query: Dict[str, Any] = {
            "hybrid": {
                "queries": [
                    {"vector": list(map(float, dense_vector))},
                    {"sparse": {"indices": list(map(int, sparse_indices)), "values": list(map(float, sparse_values))}},
                ],
                "fusion": fusion,
            }
        }
        body: Dict[str, Any] = {
            "query": query,
            "limit": int(limit),
            "with_payload": with_payload,
            "with_vector": with_vector,
        }
        if filter is not None:
            body["filter"] = filter

        async with self.session.post(url, headers=self._headers(), json=body, timeout=self.timeout_s) as r:
            data = await self._read_json(r)
        return data

