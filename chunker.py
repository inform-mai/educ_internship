def clean_text(text):
    # убираем лишние пробелы и переносы строк
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    # если несколько пробелов подряд - оставляем один
    while "  " in text:
        text = text.replace("  ", " ")

    text = text.strip()
    return text


def split_text(text, chunk_size=3500, overlap=150, source="unknown"):
    # сначала чистим, потом режем
    text = clean_text(text)

    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # каждый чанк это словарь с текстом и инфой о нём
        chunks.append({
            "text": chunk,
            "index": index,       # номер чанка
            "source": source,     # из какого файла
            "start": start,       # с какого символа
            "end": end            # по какой символ
        })

        start = end - overlap
        index += 1

    return chunks