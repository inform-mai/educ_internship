import sys
import json
from main import process

# по умолчанию берём test.txt, но можно передать свой файл
# пример: python save_chunks.py kursach.docx
if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    file_path = "test.txt"

chunks = process(file_path)

with open("chunks_result.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"готово, сохранили {len(chunks)} чанков в chunks_result.json")