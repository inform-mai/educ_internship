# Парсинг и чанкинг

Читает файлы (txt, pdf, docx) и режет текст на чанки для дальнейшей обработки.
Таблицы из pdf и docx достаются отдельно и переводятся в markdown.

## Установка

```
pip install -r requirements.txt
```

## Использование

```python
from main import process

chunks = process("test.txt")
```

Возвращает список чанков, каждый чанк это словарь:
```python
{
    "text": "кусок текста",
    "index": 0,
    "source": "test.txt",
    "start": 0,
    "end": 500
}
```

## Пример результата

Скрипт `save_chunks.py` прогоняет файл через `process()` и сохраняет результат в `chunks_result.json`, чтобы можно было посмотреть что получается на выходе.

```
python save_chunks.py test.txt
```

или для любого другого файла:
```
python save_chunks.py kursach.docx
```

## Тесты

**test.txt** - обычный текст про Пушкина.

Парсинг:
```python
from parser import parse_file
text = parse_file("test.txt")
```
На выходе - строка с текстом, 593 символа.

Чанкинг:
```python
from chunker import split_text
chunks = split_text(text, chunk_size=200, overlap=50, source="test.txt")
```
На выходе - несколько чанков по 200 символов, с перекрытием в 50 символов между соседними.

---

**test_table.pdf** - pdf с текстом и таблицей (поэты, годы рождения, города).

Парсинг:
```python
text = parse_file("test_table.pdf")
```
На выходе - текст со страницы плюс отдельно та же таблица, но в формате markdown:
```
| Имя | Год рождения | Город |
| --- | --- | --- |
| Пушкин | 1799 | Москва |
```

---

**docx файлы** - поддерживаются так же, как txt и pdf. Картинки в docx не обрабатываются, берётся только текст и таблицы (тоже переводятся в markdown).