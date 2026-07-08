# vectorsearch — векторный поиск (Санджи)

Модуль векторизации чанков и поиска (Qdrant), стыкуется с чанками
Дани-П (`main.process(file_path)`).

## Установка

```bash
pip install -r requirements.txt
```

## Быстрый старт

Нужен запущенный Qdrant (локально или в докере от Дани-Г):

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Затем:

```bash
python example_usage.py test.txt "о чём этот текст"
```

Скрипт: парсит файл → режет на чанки (код Дани-П) → эмбеддит и
загружает в Qdrant → делает векторный и гибридный поиск по запросу.

## Структура

```
vectorsearch/
├── config.py       # адрес Qdrant, имя модели эмбеддингов (через env)
├── embeddings.py   # генерация эмбеддингов (multilingual-e5-base)
├── store.py        # низкоуровневая обёртка над qdrant-client
├── indexer.py       # index_chunks(chunks) — эмбеддинг + запись в Qdrant
└── search.py       # vector_search(query), hybrid_search(query, graph_client)
```

## Как подключить граф (Катя/Дима)

`hybrid_search` принимает опциональный `graph_client` — любой объект
с методом:

```python
def search_entities(query: str, top_k: int) -> list[dict]:
    # вернуть [{"entity": "...", "related_text": "...", "score": 0.8}, ...]
    ...
```

Без graph_client `hybrid_search` ведёт себя как `vector_search` —
можно подключать граф (Memgraph) позже, ничего не переписывая
на вызывающей стороне (у Димы в агенте).

## Переменные окружения

| Переменная           | По умолчанию                    | Назначение                     |
|---------------------|--------------------|--------------------------------------|
| `QDRANT_HOST`        | `localhost`        | адрес Qdrant                         |
| `QDRANT_PORT`        | `6333`             | порт Qdrant                          |
| `QDRANT_COLLECTION`  | `chunks_collection`| имя коллекции                        |
| `EMBEDDING_MODEL`    | `intfloat/multilingual-e5-base` | модель эмбеддингов      |
| `EMBEDDING_DIM`      | `768`              | размерность вектора под модель       |
| `INDEX_BATCH_SIZE`   | `64`               | батч при индексации                  |

## Допущения, которые стоит перепроверить

- Формат чанка — `{"text", "index", "source", "start", "end"}` (взято
  из `src/chunker.py` Дани-П, актуально на момент интеграции).
- Слияние вектор+граф в `hybrid_search` — взвешенное суммирование
  score, не RRF. Если ранжирование окажется плохим на реальных
  запросах — заменить на Reciprocal Rank Fusion (обсудить с Димой,
  как он комбинирует источники в агенте).
- id точки в Qdrant — sha1(source:index), детерминированный, так что
  повторная индексация того же файла перезаписывает, а не дублирует.
