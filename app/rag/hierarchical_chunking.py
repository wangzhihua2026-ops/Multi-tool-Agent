from dataclasses import dataclass
import re


@dataclass(frozen=True)
class TextBlock:
    content: str
    index: int
    start_offset: int | None = None
    end_offset: int | None = None


def split_parent_blocks(
    content: str,
    parent_size: int = 1600,
    parent_overlap: int = 160,
) -> list[TextBlock]:
    return _split_blocks(
        content=content,
        target_size=parent_size,
        overlap=parent_overlap,
        base_offset=0,
    )


def split_child_chunks(
    parent_block: TextBlock,
    child_size: int = 450,
    child_overlap: int = 80,
) -> list[TextBlock]:
    base_offset = parent_block.start_offset or 0
    return _split_blocks(
        content=parent_block.content,
        target_size=child_size,
        overlap=child_overlap,
        base_offset=base_offset,
    )


def _split_blocks(
    content: str,
    target_size: int,
    overlap: int,
    base_offset: int,
) -> list[TextBlock]:
    normalized_size = max(target_size, 1)
    normalized_overlap = max(min(overlap, normalized_size - 1), 0)
    units = _paragraph_units(content)
    if not units:
        return []

    blocks: list[TextBlock] = []
    current_parts: list[str] = []
    current_start: int | None = None
    current_end: int | None = None

    def flush_current() -> None:
        nonlocal current_parts, current_start, current_end
        if current_parts and current_start is not None and current_end is not None:
            block_content = "\n\n".join(current_parts).strip()
            if block_content:
                blocks.append(
                    TextBlock(
                        content=block_content,
                        index=len(blocks),
                        start_offset=base_offset + current_start,
                        end_offset=base_offset + current_end,
                    )
                )
        current_parts = []
        current_start = None
        current_end = None

    for unit in units:
        unit_content = unit.content.strip()
        if not unit_content:
            continue

        if len(unit_content) > normalized_size:
            flush_current()
            blocks.extend(
                TextBlock(
                    content=block.content,
                    index=len(blocks) + offset,
                    start_offset=base_offset + (block.start_offset or 0),
                    end_offset=base_offset + (block.end_offset or 0),
                )
                for offset, block in enumerate(
                    _split_oversized_unit(
                        unit_content,
                        unit.start_offset or 0,
                        normalized_size,
                        normalized_overlap,
                    )
                )
            )
            continue

        proposed_length = _joined_length(current_parts, unit_content)
        if current_parts and proposed_length > normalized_size:
            flush_current()

        if not current_parts:
            current_start = unit.start_offset
        current_parts.append(unit_content)
        current_end = unit.end_offset

    flush_current()
    return [
        TextBlock(
            content=block.content,
            index=index,
            start_offset=block.start_offset,
            end_offset=block.end_offset,
        )
        for index, block in enumerate(blocks)
    ]


def _paragraph_units(content: str) -> list[TextBlock]:
    paragraphs: list[TextBlock] = []
    for match in re.finditer(r"\S.*?(?=(?:\r?\n\s*\r?\n)|\Z)", content, flags=re.DOTALL):
        raw = match.group(0)
        leading_trimmed = len(raw) - len(raw.lstrip())
        stripped = raw.strip()
        if not stripped:
            continue
        start = match.start() + leading_trimmed
        end = start + len(stripped)
        paragraphs.append(
            TextBlock(
                content=stripped,
                index=len(paragraphs),
                start_offset=start,
                end_offset=end,
            )
        )
    return _merge_headings_with_following_paragraphs(paragraphs)


def _merge_headings_with_following_paragraphs(paragraphs: list[TextBlock]) -> list[TextBlock]:
    merged: list[TextBlock] = []
    index = 0
    while index < len(paragraphs):
        paragraph = paragraphs[index]
        first_line = paragraph.content.splitlines()[0].strip()
        if first_line.startswith("#") and index + 1 < len(paragraphs):
            following = paragraphs[index + 1]
            merged.append(
                TextBlock(
                    content=f"{paragraph.content}\n\n{following.content}",
                    index=len(merged),
                    start_offset=paragraph.start_offset,
                    end_offset=following.end_offset,
                )
            )
            index += 2
            continue
        merged.append(
            TextBlock(
                content=paragraph.content,
                index=len(merged),
                start_offset=paragraph.start_offset,
                end_offset=paragraph.end_offset,
            )
        )
        index += 1
    return merged


def _split_oversized_unit(
    content: str,
    start_offset: int,
    target_size: int,
    overlap: int,
) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    cursor = 0
    while cursor < len(content):
        end = min(cursor + target_size, len(content))
        if end < len(content):
            boundary = content.rfind(" ", cursor + max(target_size // 2, 1), end)
            if boundary > cursor:
                end = boundary
        segment = content[cursor:end].strip()
        leading_trimmed = len(content[cursor:end]) - len(content[cursor:end].lstrip())
        trailing_trimmed = len(content[cursor:end]) - len(content[cursor:end].rstrip())
        if segment:
            absolute_start = start_offset + cursor + leading_trimmed
            absolute_end = start_offset + end - trailing_trimmed
            blocks.append(
                TextBlock(
                    content=segment,
                    index=len(blocks),
                    start_offset=absolute_start,
                    end_offset=absolute_end,
                )
            )
        if end >= len(content):
            break
        cursor = max(end - overlap, cursor + 1)
    return blocks


def _joined_length(parts: list[str], next_part: str) -> int:
    if not parts:
        return len(next_part)
    return sum(len(part) for part in parts) + len("\n\n") * len(parts) + len(next_part)
