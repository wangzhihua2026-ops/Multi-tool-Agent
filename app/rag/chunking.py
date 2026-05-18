def chunk_text(content: str, chunk_size: int = 500, chunk_overlap: int = 80) -> list[str]:
    normalized = content.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    paragraphs = [paragraph.strip() for paragraph in normalized.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(paragraph) <= chunk_size:
            current = paragraph
            continue

        chunks.extend(_split_large_block(paragraph, chunk_size=chunk_size, chunk_overlap=chunk_overlap))

    if current:
        chunks.append(current)

    return chunks


def _split_large_block(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    end = len(text)

    while start < end:
        stop = min(end, start + chunk_size)
        chunk = text[start:stop].strip()
        if chunk:
            chunks.append(chunk)
        if stop >= end:
            break
        start = max(stop - chunk_overlap, start + 1)

    return chunks
