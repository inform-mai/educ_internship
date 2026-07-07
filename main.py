import os
import re
import asyncio
from typing import Optional

import aiohttp
from VLLMwrapperClass import VLLMWrapper
from qdrant_wrapper import QdrantAsync
from uuid import uuid4
import save_chunks




async def entity_extraction():
    async with aiohttp.ClientSession() as session:
        embedder = VLLMWrapper(
            prompt="You are an embedding model. Do your job",
            session=session,
            model="google/embeddinggemma-300m",
            base_url="http://localhost:8001/v1",
        )
        async def get_vector(text: str):
            vec_list = await embedder.embeddings_root(text)
            return vec_list[0] if isinstance(vec_list, list) else vec_list
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
            model=os.getenv("MODEL", "google/gemma-4-E4B-it"),
            base_url=os.getenv("BASE_URL", "http://localhost:8000/v1"),
            extra_params={
                "temperature": 0.5,
                "top_p": 0.5,
                "max_tokens": 512,
            },
        )

        chunks = save_chunks.process("манин курсач.docx")
        chunk_text=[]
        for i in chunks:
            temp = i["text"]
            chunk_text.append(temp)
        print(chunk_text)
        results = await seeker.chat_completion_batch(chunk_text, concurrency=10)
        print(results)

        # объединяем весь вывод сущностей в одну строку и парсим по запятой
        full = ', '.join(str(r) for r in results)
        entit_list = [e.strip() for e in full.split(',') if e.strip()]

        # убираем дубликаты
        s = set()
        unique_entit = []
        for e in entit_list:
            if e not in s:
                s.add(e)
                unique_entit.append(e)

        # чанки с привязанными сущностями
        chunks_with_entit = []
        for chunk, entities in zip(chunks, results):
            chunks_with_entit.append({**chunk,
                                      'entities': entities})

        # загрузка в квадрант с двумя коллекциями
        qdrant = QdrantAsync(session=session)
        CHUNKS_COLLECTION = 'chunks_collection'
        ENTITIES_COLLECTION = 'entities_collection'
        await qdrant.ensure_collection(CHUNKS_COLLECTION,768,on_disk=True)
        await qdrant.ensure_collection(ENTITIES_COLLECTION,768,on_disk=True)

        # векторизируем текст чанка, кладем в payload текст + сущности
        for chunk in chunks_with_entit:
            vector = await get_vector(chunk["text"])
            payload = {'text': chunk["text"],
                       'source': chunk.get("source"),
                       'entities': chunk['entities']}
            await qdrant.upsert_point(CHUNKS_COLLECTION,
                                      vector=vector,
                                      payload=payload,
                                      point_id=str(uuid4()))

        # векторизируем сущности, кладем в payload имя и тип
        for entity in unique_entit:
            if ':' in entity:
                name, tag = entity.rsplit(':', 1)
            else:
                name, tag = entity, None

            vector = await get_vector(name)
            payload = {'name': name,
                       'tag': tag}
            await qdrant.upsert_point(ENTITIES_COLLECTION, vector=vector, payload=payload, point_id=str(uuid4()))

        final_result = {'chunk_with_entit': chunks_with_entit,
                        'all_entities': unique_entit}

if __name__ == "__main__":
    asyncio.run(entity_extraction())
