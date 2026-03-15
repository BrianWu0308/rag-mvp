def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> list[dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)
    step = chunk_size - overlap
    chunk_index = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = text[start:end]

        chunks.append(
            {
                "chunk_index": chunk_index,
                "start_char": start,
                "end_char": end,
                "char_count": len(chunk_text),
                "content": chunk_text,
            }
        )

        if end == text_length:
            break

        start += step
        chunk_index += 1

    return chunks