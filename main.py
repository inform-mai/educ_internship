from src.parser import parse_file
from src.chunker import split_text


def process(file_path):
    # парсим файл
    text = parse_file(file_path)

    if text is None:
        return []

    # режем на чанки
    chunks = split_text(text)

    return chunks