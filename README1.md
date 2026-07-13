# Извлечение сущностей из документов

Читает файлы (txt, pdf, docx), режет текст на чанки, извлекает именованные сущности с помощью LLM и сохраняет всё в Qdrant для дальнейшего поиска.

## Установка

```bash
pip install -r requirements.txt
```
## Использование
```bash
import asyncio
from entity_extraction import entity_extraction

asyncio.run(entity_extraction())
```
# Основные функции
## Парсинг и чанкинг
```bash
import save_chunks
chunks = save_chunks.process("file.pdf")
```
Формат чанка:
```bash
{
    "text": "текст",
    "index": 0,
    "source": "file.pdf",
    "start": 0,
    "end": 500
}
```
## Извлечение сущностей
Типы: CMP (компания), TEC (технология), PER (персона), LOC (локация), DOM (область применения), PRD (продукт)
```bash
results = await seeker.chat_completion_batch(chunk_text, concurrency=10)
```
# Результат: "Белевцев А. М.:PER, ИИ:TEC"

## Сохранение в Qdrant
Две коллекции:
```bash
chunks_collection — чанки с текстом и сущностями
entities_collection — отдельные сущности с типами
```
# Запуск
```bash
python entity_extraction.py
```
