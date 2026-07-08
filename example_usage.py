"""
Демонстрация полного цикла: от файла до поискового ответа.

Запуск (после того как Qdrant поднят, например через docker-compose от Дани-Г):

    QDRANT_HOST=localhost QDRANT_PORT=6333 python example_usage.py test.txt "мой запрос"

Если main.process сейчас называется/лежит иначе — поправьте только
строку импорта ниже, остальной код не зависит от деталей парсинга.
"""

import sys

from vectorsearch import hybrid_search, index_chunks, vector_search


def main():
    if len(sys.argv) < 3:
        print("Использование: python example_usage.py <файл> <запрос>")
        sys.exit(1)

    file_path, query = sys.argv[1], sys.argv[2]

    # Импорт функции Дани-П. Если у него модуль называется иначе —
    # поменять нужно только эту строку.
    from main import process

    chunks = process(file_path)
    print(f"Получено чанков от парсера: {len(chunks)}")

    indexed = index_chunks(chunks)
    print(f"Проиндексировано в Qdrant: {indexed}")

    print(f"\nЗапрос: {query!r}\n")

    print("--- Векторный поиск ---")
    for r in vector_search(query, top_k=5):
        print(f"[{r['score']:.3f}] ({r['source']} #{r['index']}) {r['text'][:120]}")

    print("\n--- Гибридный поиск (без графа, откат на векторный) ---")
    for r in hybrid_search(query, top_k=5):
        print(f"[{r['score']:.3f}] ({r['origin']}) {r['text'][:120]}")


if __name__ == "__main__":
    main()
