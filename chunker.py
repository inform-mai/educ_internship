def split_text(text, chunk_size=500):
    # режет текст на куски по chunk_size символов
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end

    return chunks
