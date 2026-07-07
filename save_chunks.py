import sys
import json
from parser import parse_file
from chunker import split_text


def process(file_path):
    # парсим файл
    text = parse_file(file_path)

    if text is None:
        return []

    # режем на чанки, передаём имя файла как источник
    chunks = split_text(text, source=file_path)

    return chunks
# по умолчанию берём test.txt, но можно передать свой файл
# пример: python save_chunks.py kursach.docx
if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    file_path = "test.txt"

chunks = process("манин курсач.docx")
print(chunks)
with open("chunks_result.json", "w", encoding="utf-8") as f:
     json.dump(chunks, f, ensure_ascii=False, indent=2)



print(f"готово, сохранили {len(chunks)} чанков в chunks_result.json")