import os
import re
import asyncio
from typing import List, Tuple, Dict, Optional, Any
from collections import Counter
from uuid import uuid4
from pathlib import Path
import json
import aiohttp
import itertools
import stanza
from knowledge_graph import KnowledgeGraphBuilder
from VLLMwrapperClass import VLLMWrapper
from qdrant_wrapper import QdrantAsync

# ---------------------- NER по чанкам ----------------------
RELATION_CODE_MAP = {
    0: None,                   # нет связи
    1: "USES_TECHNOLOGY",
    2: "PART_OF",
    3: "TYPE_OF",
    4: "DEVELOPED_BY",
    5: "BELONGS_TO_DOMAIN",
    6: "LOCATED_IN",
    7: "REQUIRES",
    8: "MENTIONS",
}
Triplet = Tuple[str, str, str]
def extract_chunk_entities(text: str, chunk_size: int) -> List[Tuple[str, str]]:
    """
    Разбивает текст на чанки по предложениям и извлекает сущности для каждого чанка.

    :param text: исходный текст
    :param chunk_size: максимальный размер чанка в символах
    :return: список кортежей (chunk_text, chunk_entities_string),
             где chunk_entities_string — строки вида "ent1:TYPE, ent2:TYPE, ..."
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    nlp_ru = stanza.Pipeline(lang='ru', processors='tokenize,pos,lemma,depparse,ner')
    doc = nlp_ru(text)
    allowed_types = {"PER", "ORG", "LOC", "GPE", "LAW", "PRODUCT", "MISC"}

    chunks: List[Tuple[str, str]] = []

    current_sentences: List[str] = []
    current_entities: List[str] = []
    current_len = 0

    for sent in doc.sentences:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        sent_len = len(sent_text)

        # Проверяем, помещается ли предложение в текущий чанк
        if current_sentences and current_len + 1 + sent_len > chunk_size:
            chunk_text = " ".join(current_sentences)
            entities_str = ", ".join(current_entities) if current_entities else ""
            chunks.append((chunk_text, entities_str))

            current_sentences = []
            current_entities = []
            current_len = 0

        # Добавляем предложение
        if not current_sentences:
            current_len = sent_len
        else:
            current_len += 1 + sent_len
        current_sentences.append(sent_text)

        # Извлекаем сущности
        for ent in sent.ents:
            if ent.type in allowed_types:
                current_entities.append(f"{ent.text}:{ent.type}")

    # Добавляем последний чанк
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        entities_str = ", ".join(current_entities) if current_entities else ""
        chunks.append((chunk_text, entities_str))

    return chunks


# ---------------------- разбор строк "СУЩНОСТЬ:ТИП" ----------------------

ALLOWED_TAGS = {"ORG", "TEC", "PER", "LOC", "DOM", "PRD", "MISC", "CMP"}


def parse_entities_line(entities_str: str) -> List[Tuple[str, str]]:
    """
    Парсит строку/многострочный текст с сущностями в формате
    'Текст сущности:ТИП' и возвращает список (entity, tag).
    """
    result: List[Tuple[str, str]] = []
    if not entities_str:
        return result

    text = entities_str.strip()
    tags_pattern = "|".join(sorted(ALLOWED_TAGS))
    pattern = rf"(.+?):\s*({tags_pattern})\b"

    for match in re.finditer(pattern, text):
        ent = match.group(1).strip().strip('"').strip("'")
        tag = match.group(2).strip().upper()

        if ent or tag in ALLOWED_TAGS:
            new_ent = ent.replace(",", "").strip()
            result.append((new_ent, tag))

    return result


# ---------------------- основной пайплайн извлечения сущностей ----------------------

async def process_entities_pipeline(
    text: str,
    *,
    seeker: VLLMWrapper,
    checker: VLLMWrapper,
    chunk_size: int = 3000,
    concurrency: int = 5,
) -> Tuple[List[Tuple[str, str, int]], List[Dict[str, str]]]:
    """
    Полный пайплайн по всему тексту.

    1) extract_chunk_entities(text, chunk_size) ->
       список (chunk_text, nlp_entities_str).
    2) Для всех чанков:
       - Seeker (LLM) извлекает сущности по чанку.
       - Checker (LLM) нормализует объединённый список (НЛП + LLM) для каждого чанка.
    3) Все ответы чекера объединяются, сущности парсятся и считаются.

    !!! Важно:
    - seeker и checker создаются СНАРУЖИ и передаются сюда как аргументы.
    - Функция возвращает:
        * entities_stats: список (entity, tag, count)
        * chunk_results: список словарей с данными по каждому чанку:
            {
              "chunk_text": str,
              "nlp_entities": str,
              "llm_entities": str,
              "final_entities": str,
            }
    """
    # 1. Получаем чанки и НЛП-сущности по ним
    chunks: List[Tuple[str, str]] = extract_chunk_entities(text, chunk_size)

    # 2а. Формируем батч-запросы к Seeker
    seeker_inputs: List[str] = []
    for chunk_text, nlp_entities_str in chunks:
        seeker_inputs.append(
            "Фрагмент текста:\n"
            f"{chunk_text}\n\n"
            "Сущности, извлечённые НЛП (для справки, ты можешь их использовать или игнорировать):\n"
            f"{nlp_entities_str or ''}\n"
        )

    # 2б. Запуск Seeker в батче
    seeker_outputs: List[Optional[str]] = await seeker.chat_completion_batch(
        seeker_inputs,
        concurrency=concurrency,
        return_exceptions=False,
    )
    # seeker_outputs[i] — строка "ent1:TYPE, ent2:TYPE, ..."

    # 3а. Формируем батч-запросы к Checker
    checker_inputs: List[str] = []
    for (chunk_text, nlp_entities_str), llm_entities_str in zip(chunks, seeker_outputs):
        checker_inputs.append(
            "Исходный текст фрагмента:\n"
            f"{chunk_text}\n\n"
            "Сущности, извлечённые НЛП:\n"
            f"{nlp_entities_str or ''}\n\n"
            "Сущности, извлечённые ЛЛМ:\n"
            f"{llm_entities_str or ''}\n"
        )

    # 3б. Запуск Checker в батче
    checker_outputs: List[Optional[str]] = await checker.chat_completion_batch(
        checker_inputs,
        concurrency=concurrency,
        return_exceptions=False,
    )
    # checker_outputs[i] — финальная строка "СУЩНОСТЬ:ТИП, ..."

    # 4. Собираем всё по чанкам + считаем частоты сущностей
    counter: Counter[Tuple[str, str]] = Counter()
    chunk_results: List[Dict[str, str]] = []

    for idx, ((chunk_text, nlp_entities_str), llm_entities_str, final_str) in enumerate(
        zip(chunks, seeker_outputs, checker_outputs)
    ):
        final_str = final_str or ""
        nlp_entities_str = nlp_entities_str or ""
        llm_entities_str = llm_entities_str or ""

        # сохраняем информацию по чанку для последующей записи в Qdrant
        chunk_results.append(
            {
                "chunk_text": chunk_text,
                "nlp_entities": nlp_entities_str,
                "llm_entities": llm_entities_str,
                "final_entities": final_str,
            }
        )

        # обновляем счётчики сущностей
        entities = parse_entities_line(final_str)
        for ent, tag in entities:
            counter[(ent, tag)] += 1

    # 5. Преобразуем в список (entity, tag, count)
    entities_stats: List[Tuple[str, str, int]] = [
        (ent, tag, count) for (ent, tag), count in counter.items()
    ]
    entities_stats.sort(key=lambda x: x[2], reverse=True)

    return entities_stats, chunk_results




async def upsert_entities_with_llm_embeddings(
    qdrant: QdrantAsync,
    collection: str,
    entities: List[Tuple[str, str, int]],  # (text, tag, count)
    embedder: VLLMWrapper,
    similarity_threshold: float = 0.95,
) -> None:
    """
    Сохраняет сущности в Qdrant с учётом семантических дубликатов.

    - Для КАЖДОЙ сущности считаем эмбеддинг по её тексту
    - Если нашлась похожая точка (score >= similarity_threshold и tag совпадает) —
      увеличиваем её count.
    - Иначе создаём новую точку.
    """

    if not entities:
        return

    # 1. Получаем эмбеддинг первой сущности, чтобы узнать размер вектора
    first_text, first_tag, first_count = entities[0]
    first_vec_list = await embedder.embeddings_root(first_text)
    first_vector = first_vec_list[0] if isinstance(first_vec_list, list) else first_vec_list
    vector_size = len(first_vector)

    # Гарантируем наличие коллекции
    await qdrant.ensure_collection(name=collection, vector_size=vector_size)

    async def get_vector(text: str):
        vec_list = await embedder.embeddings_root(text)
        return vec_list[0] if isinstance(vec_list, list) else vec_list

    async def process_one(text: str, tag: str, count: int):
        # 2. Эмбеддинг сущности
        vector = await get_vector(text)

        # 3. Поиск похожей сущности
        search_res = await qdrant.search(
            collection=collection,
            vector=vector,
            limit=1,
            score_threshold=similarity_threshold,
            with_payload=True,
            with_vector=False,
        )

        hits = []
        if isinstance(search_res, dict):
            hits = search_res.get("result") or search_res.get("points") or []

        best_point = hits[0] if hits else None

        # 4. Если похожей точки нет — создаём новую
        if best_point is None:
            payload = {
                "text": text,
                "tag": tag,
                "count": int(count),
            }
            await qdrant.upsert_point(
                collection=collection,
                vector=vector,
                payload=payload,
                point_id=str(uuid4()),
            )
            return

        # 5. Нашли похожую — проверяем tag
        point_id = best_point.get("id")
        payload = best_point.get("payload", {}) or {}
        existing_tag = payload.get("tag")

        # Если тип другой — считаем другой сущностью, создаём новую точку
        if existing_tag and existing_tag != tag:
            new_payload = {
                "text": text,
                "tag": tag,
                "count": int(count),
            }
            await qdrant.upsert_point(
                collection=collection,
                vector=vector,
                payload=new_payload,
                point_id=str(uuid4()),
            )
            return

        # 6. Тип совпадает — увеличиваем счётчик
        old_count = int(payload.get("count", 0))
        new_count = old_count + int(count)

        updated_payload = {
            "text": payload.get("text", text),
            "tag": existing_tag or tag,
            "count": new_count,
        }

        await qdrant.upsert_point(
            collection=collection,
            vector=vector,
            payload=updated_payload,
            point_id=point_id,
        )

    for text, tag, count in entities:
        await process_one(text, tag, count)


# ---------------------- новая запись ЧАНКОВ в Qdrant ----------------------

async def upsert_chunks_with_llm_embeddings(
    qdrant: QdrantAsync,
    collection: str,
    chunks: List[Dict[str, str]],
    embedder: VLLMWrapper,
    source_path: Optional[str] = None,
) -> None:
    """
    Сохраняет ЧАНКИ текста в отдельную коллекцию Qdrant.

    Для каждого чанка считаем эмбеддинг его текста и пишем payload вида:
        {
          "chunk_text": ... (исходный текст из документа),
          "entities_final": ... (строка СУЩНОСТЬ:ТИП),
          "entities_nlp": ...,
          "entities_llm": ...,
          "source_file": ... (опционально),
        }

    Название коллекции рекомендуется делать таким же, как имя файла
    без расширения.
    """
    if not chunks:
        return

    # 1. Размерность вектора по первому чанку
    first_vec_list = await embedder.embeddings_root(chunks[0]["chunk_text"])
    first_vector = first_vec_list[0] if isinstance(first_vec_list, list) else first_vec_list
    vector_size = len(first_vector)

    await qdrant.ensure_collection(name=collection, vector_size=vector_size)

    for idx, ch in enumerate(chunks):
        vec_list = await embedder.embeddings_root(ch["chunk_text"])
        vector = vec_list[0] if isinstance(vec_list, list) else vec_list

        payload = {
            "chunk_text": ch["chunk_text"],
            "entities_final": ch.get("final_entities", ""),
            "entities_nlp": ch.get("nlp_entities", ""),
            "entities_llm": ch.get("llm_entities", ""),
        }
        if source_path:
            payload["source_file"] = os.path.basename(source_path)
            payload["source_path"] = source_path
        payload["chunk_index"] = idx

        await qdrant.upsert_point(
            collection=collection,
            vector=vector,
            payload=payload,
            point_id=str(uuid4()),
        )


async def main(doc_path: str):
    text = VLLMWrapper.read_text(doc_path)

    async with aiohttp.ClientSession() as session:
        seeker = VLLMWrapper(
            prompt=(
                "Ты эксперт по извлечению именованных сущностей.\n"
                "Для каждой сущности выбери один тип из ограниченного списка и верни результат строго в виде строки.\n"
                "Допустимые типы (значения поля tag):\n"
                "CMP — компания\n"
                "TEC — технология\n"
                "PER — персона\n"
                "LOC — локация\n"
                "DOM — область применения технологии\n"
                "PRD — продукт\n"
                "Игнорируй сущности, не подходящие ни под один тип.\n"
                "Приведи сущности к канонической форме.\n"
                "Не давай никаких дополнительных комментариев, только сущность и ее тип.\n"
                "Ответ дай в формате строки через запятую с указанием текста сущности и её типа,\n"
                "например: Белевцев А. М.:PER\n"
            ),
            session=session,
            model=os.getenv("MODEL", "cpatonn/Qwen3-VL-8B-Thinking-AWQ-8bit"),
            base_url=os.getenv("BASE_URL", "http://localhost:8000/v1"),
            extra_params={
                "temperature": 0.2,
                "top_p": 0.2,
                "max_tokens": 512,
            },
        )

        checker = VLLMWrapper(
            prompt=(
                """Ты эксперт по проверке типов сущностей.
Тебе передаются сущности, извлечённые из текста.
Твоя задача — скорректировать тип каждой сущности, не изменяя их количество, порядок и текст.
Нельзя удалять, объединять или добавлять сущности.
Исправляй только тип.

Допустимые типы:
ORG — организации, компании, учреждения (Apple, Росатом)
TEC — технологии, методы, алгоритмы, стандарты (Kubernetes, GPT-4)
PER — люди, имена, инициалы, персонажи (Илон Маск, А. С. Пушкин)
LOC — страны, города, геообъекты, здания (Москва, Европа, Байкал)
DOM — области применения, сферы деятельности (финтех, здравоохранение)
PRD — продукты, сервисы, устройства, модели (iPhone 15, Windows 11)
MISC — неопределённые сущности, которые ты обязан переклассифицировать

Верни сущности в формате:
текст:ТИП, текст:ТИП, ...
Только список, без пояснений."""
            ),
            session=session,
            model=os.getenv("MODEL", "ai-sage/GigaChat3-10B-A1.8B"),
            base_url=os.getenv("BASE_URL", "http://localhost:8000/v1"),
            extra_params={
                "temperature": 0.3,
                "top_p": 0.3,
                "max_tokens": 512,
            },
        )

        embedder = VLLMWrapper(
            prompt="You are an embedding model. Do your job",
            session=session,
            model="google/embeddinggemma-300m",
            base_url="http://localhost:8001/v1",
        )

        qdr = QdrantAsync(session)

        # 2. Запускаем пайплайн
        entities_stats, chunk_results = await process_entities_pipeline(
            text=text,
            seeker=seeker,
            checker=checker,
            chunk_size=3000,
            concurrency=24,
        )

        # --- пример: пишем агрегированные сущности в отдельную коллекцию ---
        await upsert_entities_with_llm_embeddings(
            qdrant=qdr,
            collection="entities",   # глобальная коллекция сущностей
            entities=entities_stats,
            embedder=embedder,
            similarity_threshold=0.9,
        )

        # --- пример: пишем ЧАНКИ в коллекцию, названную как файл ---
        collection_name = Path(doc_path).stem  # имя коллекции = имя файла без расширения
        await upsert_chunks_with_llm_embeddings(
            qdrant=qdr,
            collection=collection_name,
            chunks=chunk_results,
            embedder=embedder,
            source_path=doc_path,
        )


        # Для отладки можно вывести статистику
        print("Всего сущностей:", len(entities_stats))
        for e in entities_stats[:50]:
            print(e)

def parse_llm_relation_lines(
    responses: List[Optional[str]],
    output_path: str,
    verbose: bool = True,
) -> List[Tuple[str, str, str]]:
    """
    Принимает список ответов LLM (строки вида 'subject,relation,object'),
    парсит их, удаляет дубликаты по (subject, object),
    записывает результат в txt и возвращает список кортежей.
    """
    raw_relations: List[Tuple[str, str, str]] = []

    # --- парсинг строк ---
    for resp in responses:
        if not resp:
            continue

        lines = resp.strip().splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                if verbose:
                    print(f"[parse_rel] skip line (not enough parts): {line!r}")
                continue

            subject = parts[0]
            relation = parts[1]
            obj = ",".join(parts[2:]).strip()

            if not subject or not relation or not obj:
                if verbose:
                    print(f"[parse_rel] skip line (empty field): {line!r}")
                continue

            raw_relations.append((subject, relation, obj))

    # --- дедуп по (subject, object) ---
    # если несколько связей для одной пары сущностей,
    # оставляем первую встреченную
    pair_to_relation = {}
    order: List[Tuple[str, str]] = []

    for subj, rel, obj in raw_relations:
        key = (subj, obj)
        if key not in pair_to_relation:
            pair_to_relation[key] = rel
            order.append(key)

    dedup_relations: List[Tuple[str, str, str]] = [
        (subj, pair_to_relation[(subj, obj)], obj)
        for (subj, obj) in order
    ]

    # --- запись в файл ---
    with open(output_path, "w", encoding="utf-8") as f:
        for subj, rel, obj in dedup_relations:
            f.write(f"{subj},{rel},{obj}\n")

    if verbose:
        print(
            f"[parse_rel] raw={len(raw_relations)}, "
            f"dedup={len(dedup_relations)}, saved to {output_path}"
        )

    return dedup_relations


async def filter_relations_with_llm(
    connector: VLLMWrapper,
    input_path: str,
    output_path: str,
    *,
    batch_size: int = 32,
    verbose: bool = True,
) -> List[Tuple[str, str, str]]:
    """
    Читает связи из txt-файла (каждая строка: subject,relation,object),
    по одной прогоняет через ЛЛМ-валидатор (батчами под капотом),
    оставляет только те, для которых модель вернула 1.
    ЛЛМ НЕ переписывает триплет: мы всегда используем исходную строку.

    В файл output_path записываются только релевантные связи (у которых ответ == 1).
    Возвращает список кортежей (subject, relation, object) для прошедших связей.
    """
    # --- читаем входной файл ---
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    # удаляем пустые
    raw_triplets = [line for line in lines if line]
    if verbose:
        print(f"[filter_rel] loaded {len(raw_triplets)} relations from {input_path}")

    accepted_lines: List[str] = []

    # --- батчами отправляем в ЛЛМ ---
    for i in range(0, len(raw_triplets), batch_size):
        batch = raw_triplets[i : i + batch_size]

        # ЛЛМ ожидает список строк user_content
        answers = await connector.chat_completion_batch(batch)

        for triplet, answer in zip(batch, answers):
            if answer is None:
                continue

            text = str(answer).strip()

            # Ищем первый символ 0/1 в ответе
            verdict: Optional[str] = None
            for ch in text:
                if ch in ("0", "1"):
                    verdict = ch
                    break

            if verdict is None:
                if verbose:
                    print(f"[filter_rel] no 0/1 in answer for {triplet!r}: {text!r}")
                continue

            if verdict == "1":
                accepted_lines.append(triplet)

    # --- парсим прошедшие связи в кортежи ---
    accepted_triplets: List[Tuple[str, str, str]] = []
    for line in accepted_lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        subj = parts[0]
        rel = parts[1]
        obj = ",".join(parts[2:]).strip()
        if subj and rel and obj:
            accepted_triplets.append((subj, rel, obj))

    # --- записываем очищенный список в файл ---
    with open(output_path, "w", encoding="utf-8") as f:
        for line in accepted_lines:
            f.write(line + "\n")

    if verbose:
        print(
            f"[filter_rel] accepted {len(accepted_triplets)} / {len(raw_triplets)} "
            f"relations, saved to {output_path}"
        )

    return accepted_triplets

async def extract_entity_relations(
        qdrant: "QdrantAsync",
        connector: "VLLMWrapper",
        *,
        entities_collection: str,
        chunks_collection: str,
        entity_name_field: str = "canonical",
        chunk_text_field: str = "text",
        chunk_sem_entities_field: str = "sem_ent_2",
        per_entity_chunks: int = 5,
        similarity_threshold: float = 0.7,
        scroll_page_size: int = 128,
        max_entities: Optional[int] = None,
        verbose: bool = True,
) -> List[Tuple[str, str, str]]:
    """
    Находим связи: БАЗОВАЯ СУЩНОСТЬ -> сущности внутри релевантных чанков.

    Для каждой сущности:
      - ищем top-N чанков (per_entity_chunks)
      - из чанков берём text + sem_ent_2
      - составляем JSON-пакет для LLM:
        { "base": <entity>, "text": <chunk_text>, "entities": <entities_in_chunk> }
      - LLM возвращает строки "subject,relation,object"
    """

    all_entities: List[Dict[str, Any]] = []

    # === 1) Загружаем сущности ===
    offset = None
    while True:
        raw_page = await qdrant.scroll_points(
            collection=entities_collection,
            limit=scroll_page_size,
            with_payload=True,
            with_vector=True,
            offset=offset,
        )
        result = raw_page.get("result") if isinstance(raw_page, dict) else raw_page
        points = (result or {}).get("points") or []
        offset = (result or {}).get("next_page_offset")

        if not points:
            break

        all_entities.extend(points)

        if max_entities is not None and len(all_entities) >= max_entities:
            all_entities = all_entities[:max_entities]
            break

        if offset is None:
            break

    if verbose:
        print(f"[relation-e2c] entities loaded: {len(all_entities)}")

    user_contents: List[str] = []

    # === 2) Для каждой сущности ищем чанки и формируем JSON для LLM ===
    for ent_point in all_entities:
        payload = ent_point.get("payload") or {}
        vec = ent_point.get("vector")

        # извлекаем имя
        base_name = (
                payload.get(entity_name_field)
                or payload.get("canonical")
                or payload.get("text")
                or payload.get("name")
        )
        if not isinstance(base_name, str) or not base_name.strip():
            continue
        base_name = base_name.strip()

        # вектор
        if isinstance(vec, dict):
            vec = vec[next(iter(vec))]
        if not isinstance(vec, list):
            continue

        # --- поиск чанков ---
        res = await qdrant.search(
            collection=chunks_collection,
            vector=vec,
            limit=per_entity_chunks,
            with_payload=True,
            with_vector=False,
            score_threshold=similarity_threshold,
        )
        hits = (res or {}).get("result") or []
        if not hits:
            continue

        # --- формируем упаковку для каждого чанка ---
        for hit in hits:
            ch_payload = hit.get("payload") or {}

            # текст чанка
            text = ch_payload.get(chunk_text_field)
            if not isinstance(text, str):
                text = (
                        ch_payload.get("text")
                        or ch_payload.get("chunk")
                        or ch_payload.get("chunk_text")
                )
            if not isinstance(text, str):
                continue

            # сущности в чанке
            sem_raw = ch_payload.get(chunk_sem_entities_field)
            if isinstance(sem_raw, str):
                entities = [e.strip() for e in sem_raw.split(",") if e.strip()]
            elif isinstance(sem_raw, list):
                entities = [str(e).strip() for e in sem_raw if str(e).strip()]
            else:
                entities = []

            # base_entity должна быть в чанке? Нет — это не обязательное условие
            # главное: пары формируются base -> each(entities)

            # убираем base entity из списка, чтобы избежать пар (A,A)
            filtered_ents = [e for e in entities if e != base_name]
            if not filtered_ents:
                continue

            user_json = json.dumps(
                {
                    "base": base_name,
                    "text": text,
                    "entities": filtered_ents,
                },
                ensure_ascii=False,
            )
            user_contents.append(user_json)

    if verbose:
        print(f"[relation-e2c] total LLM jobs: {len(user_contents)}")

    if not user_contents:
        return []

    # === 3) Batch LLM ===
    llm_answers = await connector.chat_completion_batch(user_contents)

    # === 4) Парсим строки subject,relation,object ===
    relations = parse_llm_relation_lines(
        responses=llm_answers,
        output_path="relations.txt",
        verbose=verbose,
    )

    return relations

async def extract_relations_intra_chunk_coded(
    qdrant: "QdrantAsync",
    connector: "VLLMWrapper",
    *,
    chunks_collection: str,
    chunk_text_field: str = "text",
    chunk_sem_entities_field: str = "entities_final",
    scroll_page_size: int = 128,
    max_chunks: Optional[int] = None,
    max_pairs_per_chunk: Optional[int] = None,
    verbose: bool = True,
) -> List[Triplet]:
    """
    Ищет связи ТОЛЬКО внутри одного чанка.
    ЛЛМ возвращает только коды (0–8) по порядку пар.

    Для каждого чанка:
      - берём текст + список сущностей из метаданных (sem_ent_2)
      - строим все пары (subject, object) внутри чанка
      - отправляем JSON: {"text": ..., "pairs": [[e1,e2], [e1,e3], ...]}
      - ЛЛМ отвечает:
            <code_1>
            <code_2>
            ...
        где code_i соответствует pairs[i].
      - код 0 = нет связи, остальные — типы из RELATION_CODE_MAP.

    Возвращает список триплетов (subject, relation, object).
    """

    user_contents: List[str] = []
    all_pairs_per_chunk: List[List[Tuple[str, str]]] = []

    offset = None
    processed_chunks = 0
    page_idx = 0

    # --- 1. Собираем чанки и пары ---
    while True:
        raw_page = await qdrant.scroll_points(
            collection=chunks_collection,
            limit=scroll_page_size,
            with_payload=True,
            with_vector=False,
            offset=offset,
        )
        result = raw_page.get("result") if isinstance(raw_page, dict) else raw_page
        points = (result or {}).get("points") or []
        offset = (result or {}).get("next_page_offset")
        page_idx += 1

        if verbose:
            print(f"[intra_coded] page {page_idx}: {len(points)} chunks, next_offset={offset}")

        if not points:
            break

        for point in points:
            if max_chunks is not None and processed_chunks >= max_chunks:
                offset = None
                break

            processed_chunks += 1
            payload = point.get("payload") or {}

            # текст чанка
            text = payload.get(chunk_text_field)
            if not isinstance(text, str):
                text = (
                    payload.get("text")
                    or payload.get("chunk")
                    or payload.get("chunk_text")
                )
            if not isinstance(text, str):
                continue

            # сущности внутри чанка
            sem_raw = payload.get(chunk_sem_entities_field)
            if isinstance(sem_raw, str):
                entities = [e.strip() for e in sem_raw.split(",") if e.strip()]
            elif isinstance(sem_raw, list):
                entities = [str(e).strip() for e in sem_raw if str(e).strip()]
            else:
                entities = []

            if len(entities) < 2:
                continue

            # все пары (направление зафиксируем: first -> second)
            pairs: List[Tuple[str, str]] = list(itertools.combinations(entities, 2))
            if max_pairs_per_chunk is not None and len(pairs) > max_pairs_per_chunk:
                pairs = pairs[:max_pairs_per_chunk]

            if not pairs:
                continue

            # сериализуем pairs для LLM
            pairs_serialized = [[a, b] for (a, b) in pairs]

            user_obj = {
                "text": text,
                "pairs": pairs_serialized,
            }
            user_content = json.dumps(user_obj, ensure_ascii=False)

            user_contents.append(user_content)
            all_pairs_per_chunk.append(pairs)

        if offset is None:
            break

    if verbose:
        print(f"[intra_coded] total chunks prepared: {len(user_contents)}")

    if not user_contents:
        return []

    # --- 2. Батч в ЛЛМ ---
    # connector.chat_completion_batch ожидает список строк (user_content)
    llm_answers = await connector.chat_completion_batch(user_contents)

    # --- 3. Парсим коды и строим триплеты ---
    relations: List[Triplet] = []

    for chunk_idx, (pairs, answer) in enumerate(zip(all_pairs_per_chunk, llm_answers)):
        if answer is None:
            continue

        text = str(answer).strip()
        if not text:
            continue

        # каждая строка — одна цифра 0–8
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        if len(lines) != len(pairs):
            if verbose:
                print(
                    f"[intra_coded] WARN: codes count != pairs count "
                    f"(chunk {chunk_idx}: codes={len(lines)}, pairs={len(pairs)})"
                )
            # подгоним по минимальному количеству, чтобы не падать
            limit = min(len(lines), len(pairs))
            lines = lines[:limit]
            pairs = pairs[:limit]

        for (subj, obj), code_str in zip(pairs, lines):
            # берём ПЕРВУЮ цифру в строке
            code_digit = None
            for ch in code_str:
                if ch.isdigit():
                    code_digit = int(ch)
                    break

            if code_digit is None:
                continue

            relation = RELATION_CODE_MAP.get(code_digit)
            if not relation:
                # 0 или неизвестный код = нет связи
                continue

            relations.append((subj, relation, obj))

    # дедуп по (subject, object), как раньше
    pair2rel: Dict[Tuple[str, str], str] = {}
    order: List[Tuple[str, str]] = []
    for subj, rel, obj in relations:
        key = (subj, obj)
        if key not in pair2rel:
            pair2rel[key] = rel
            order.append(key)

    dedup_relations: List[Triplet] = [
        (subj, pair2rel[(subj, obj)], obj)
        for (subj, obj) in order
    ]

    if verbose:
        print(
            f"[intra_coded] raw_relations={len(relations)}, "
            f"dedup={len(dedup_relations)}"
        )

    return dedup_relations

async def semantyc():
    async with aiohttp.ClientSession() as session:
        qdrant = QdrantAsync(session)
        embedder = VLLMWrapper(
            prompt="You are an embedding model. Do your job",
            session=session,
            model="google/embeddinggemma-300m",
            base_url="http://localhost:8001/v1",
        )
        connector = VLLMWrapper(
            prompt="""Ты модуль извлечения связей между сущностями в одном текстовом чанке.

Вход (user message) — JSON:
{
  "base": "<центральная сущность>",
  "text": "<текст чанка>",
  "entities": ["e1", "e2", ...]   // сущности, найденные рядом
}

Для каждой пары (base, eX) определи, есть ли явная связь по тексту.

Игнорируй пары, где связь неочевидна или не подтверждается текстом.
Новых сущностей не создавай.


Допустимые типы relation (строгие строки):
USES_TECHNOLOGY
PART_OF
TYPE_OF
DEVELOPED_BY
BELONGS_TO_DOMAIN
LOCATED_IN
REQUIRES
MENTIONS   (только если нет более точного типа)

Ответ: только строки
subject,relation,object
Если нет связей — верни пустой ответ.
Без кодовых блоков и без JSON.""",
            session=session,
            model=os.getenv("MODEL", "cpatonn/Qwen3-VL-8B-Thinking-AWQ-8bit"),
            base_url=os.getenv("BASE_URL", "http://localhost:8000/v1"),
            extra_params={
                "temperature": 0.2,
                "top_p": 0.2,
                "max_tokens": 1024,
            },
        )
        # await qdrant.propagate_entities_to_chunks(
        #     entities_collection="entities",
        #     chunks_collection="манин курсач",
        #     similarity_threshold=0.35,  # подрегулируй по вкусу
        #     search_limit=300,  # можно поднять, если чанков немного
        #     use_hybrid=False,  # поставь True, если есть sparse-вектора
        #     require_text_match=False,
        # )
        relations = await extract_entity_relations(
            qdrant=qdrant,
            connector=connector,
            entities_collection="entities",
            chunks_collection="манин курсач",
            entity_name_field="text",
            chunk_text_field="chunk_text",
            chunk_sem_entities_field="sem_ent_2",
            per_entity_chunks=3,
            similarity_threshold=0.35,
            max_entities=None,
        )
        graph = KnowledgeGraphBuilder()
        graph.add_relations(relations)
        graph.visualize_interactive()
        # await qdrant.enrich_chunks_with_semantic_entities(chunks_collection="манин курсач",
        # entities_collection="entities",
        # similarity_threshold=0.4,)

async def triplets():
    async with aiohttp.ClientSession() as session:
        qdrant = QdrantAsync(session=session)
        embedder = VLLMWrapper(
            prompt="You are an embedding model. Do your job",
            session=session,
            model="google/embeddinggemma-300m",
            base_url="http://localhost:8001/v1",
        )
        connector = VLLMWrapper(
            session=session,
            base_url="http://localhost:8000/v1",
            model=os.getenv("MODEL", "cpatonn/Qwen3-VL-8B-Thinking-AWQ-8bit"),        # твоя модель
            prompt="""Ты классифицируешь связи между парами сущностей в одном текстовом чанке.

Вход (user) — JSON:
{
  "text": "<текст>",
  "pairs": [["A","B"], ["C","D"], ...]
}

Каждая пара ["X","Y"] задаёт направленную связь X → Y.

Для КАЖДОЙ пары по порядку выведи ОДНУ цифру (код связи):

0 — нет связи или недостаточно данных
1 — USES_TECHNOLOGY
2 — PART_OF
3 — TYPE_OF
4 — DEVELOPED_BY
5 — BELONGS_TO_DOMAIN
6 — LOCATED_IN
7 — REQUIRES

Требования:
- одна пара = одна строка с одной цифрой;
- количество строк = количество пар;
- без объяснений, без текста, без кодовых блоков.""",
            extra_params={
                "temperature": 0.4,
                "top_p": 0.5,
                "max_tokens": 512,
            },
        )

        relations = await extract_relations_intra_chunk_coded(
            qdrant=qdrant,
            connector=connector,
            chunks_collection="манин курсач",
            chunk_text_field="chunk_text",
            chunk_sem_entities_field="entities_final",
            max_chunks=None,           # или 100 для теста
            max_pairs_per_chunk=150,    # чтобы не раздувать запрос
        )

        print("Total relations:", len(relations))
        for triplet in relations:
            print(triplet)
        graph = KnowledgeGraphBuilder()
        graph.add_relations(relations)
        await graph.enrich_node_types_from_qdrant(
            qdrant=qdrant,
            embedder=embedder,
            collection="entities",
            similarity_threshold=0.9,
        )
        graph.visualize_interactive()


async def clean_graph():
    async with aiohttp.ClientSession() as session:
        validator = VLLMWrapper(
            prompt="""Ты бинарный фильтр связей между сущностями.

На вход (user) ты получаешь ОДНУ строку вида:
subject,relation,object

relation — один из:
USES_TECHNOLOGY, PART_OF, TYPE_OF, DEVELOPED_BY,
BELONGS_TO_DOMAIN, LOCATED_IN, REQUIRES, MENTIONS

Твоя задача — оценить, выглядит ли такая связь реалистичной
и технически правдоподобной в общем (с учётом мировых знаний).

Если связь правдоподобна — верни "1".
Если связь бессмысленна, очень сомнительна или нелогична — верни "0".

Ответ: только символ 1 или 0, без пояснений и без других символов.
Не переписывай триплет.""",
            session=session,
            model=os.getenv("MODEL", "cpatonn/Qwen3-VL-8B-Thinking-AWQ-8bit"),
            base_url=os.getenv("BASE_URL", "http://localhost:8000/v1"),
            extra_params={
                "temperature": 0.4,
                "top_p": 0.5,
                "max_tokens": 128,
            },
        )
        qdrant = QdrantAsync(session=session)
        embedder = VLLMWrapper(
            prompt="You are an embedding model. Do your job",
            session=session,
            model="google/embeddinggemma-300m",
            base_url="http://localhost:8001/v1",
        )
        results = await filter_relations_with_llm(validator,"relations.txt","clean_relations.txt", batch_size=48)
        graph = KnowledgeGraphBuilder()
        graph.add_relations(results)
        await graph.enrich_node_types_from_qdrant(
            qdrant=qdrant,
            embedder=embedder,
            collection="entities",
            similarity_threshold=0.9,
        )
        graph.visualize_interactive()


async def rag():
    async with aiohttp.ClientSession() as session:
        embedder = VLLMWrapper(
            prompt="You are an embedding model. Do your job",
            session=session,
            model="google/embeddinggemma-300m",
            base_url="http://localhost:8001/v1",
        )
        seeker = VLLMWrapper(
            prompt=(
                "Ты эксперт по извлечению именованных сущностей.\n"
                "Для каждой сущности выбери один тип из ограниченного списка и верни результат строго в виде строки.\n"
                "Допустимые типы (значения поля tag):\n"
                "CMP — компания\n"
                "TEC — технология\n"
                "PER — персона\n"
                "LOC — локация\n"
                "DOM — область применения технологии\n"
                "PRD — продукт\n"
                "Игнорируй сущности, не подходящие ни под один тип.\n"
                "Приведи сущности к канонической форме.\n"
                "Не давай никаких дополнительных комментариев, только сущность и ее тип.\n"
                "Ответ дай в формате строки через запятую с указанием текста сущности и её типа,\n"
                "например: Белевцев А. М.:PER\n"
            ),
            session=session,
            model=os.getenv("MODEL", "cpatonn/Qwen3-VL-8B-Thinking-AWQ-8bit"),
            base_url=os.getenv("BASE_URL", "http://localhost:8000/v1"),
            extra_params={
                "temperature": 0.2,
                "top_p": 0.2,
                "max_tokens": 512,
            },
        )
        res = await embedder.embeddings_root("srgefumjgrsmjsefrgaam")
        checker = VLLMWrapper(
            prompt=(
                """Ты эксперт по проверке типов сущностей.
Тебе передаются сущности, извлечённые из текста.
Твоя задача — скорректировать тип каждой сущности, не изменяя их количество, порядок и текст.
Нельзя удалять, объединять или добавлять сущности.
Исправляй только тип.

Допустимые типы:
ORG — организации, компании, учреждения (Apple, Росатом)
TEC — технологии, методы, алгоритмы, стандарты (Kubernetes, GPT-4)
PER — люди, имена, инициалы, персонажи (Илон Маск, А. С. Пушкин)
LOC — страны, города, геообъекты, здания (Москва, Европа, Байкал)
DOM — области применения, сферы деятельности (финтех, здравоохранение)
PRD — продукты, сервисы, устройства, модели (iPhone 15, Windows 11)
MISC — неопределённые сущности, которые ты обязан переклассифицировать

Верни сущности в формате:
текст:ТИП, текст:ТИП, ...
Только список, без пояснений."""
            ),
            session=session,
            model=os.getenv("MODEL", "ai-sage/GigaChat3-10B-A1.8B"),
            base_url=os.getenv("BASE_URL", "http://localhost:8000/v1"),
            extra_params={
                "temperature": 0.3,
                "top_p": 0.3,
                "max_tokens": 512,
            },
        )
        text = embedder.read_text("vizualnoe_proektirovanie_korrektiruyuschih_i_soglasuyuschih_tsepey.pdf") + embedder.read_text("sistema_upravleniya_avtonomnym_kolesnym_robotom_skif_3_dlya_apriori.pdf") + embedder.read_text("deepseek.pdf") + embedder.read_text("ant.pdf")
        qdrant = QdrantAsync(session=session)
        entities_stats, chunk_results = await process_entities_pipeline(text, seeker=seeker, checker=checker, chunk_size=3500, concurrency=10)

        await upsert_chunks_with_llm_embeddings(
            qdrant=qdrant,
            collection="test",
            chunks=chunk_results,
            embedder=embedder,
            source_path="",
        )
        await upsert_entities_with_llm_embeddings(
            qdrant=qdrant,
            collection="test_entities",  # глобальная коллекция сущностей
            entities=entities_stats,
            embedder=embedder,
            similarity_threshold=0.9,
        )

if __name__ == "__main__":
    asyncio.run(rag())
