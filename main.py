from src.parser import parse_file
from src.chunker import split_text


def process(file_path):
    # парсим файл
    text = parse_file(file_path)

    if text is None:
        return []

    # режем на чанки, передаём имя файла как источник
    chunks = split_text(text, source=file_path)

    return chunks