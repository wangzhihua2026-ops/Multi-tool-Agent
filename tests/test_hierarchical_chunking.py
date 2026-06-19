from app.rag.hierarchical_chunking import TextBlock, split_child_chunks, split_parent_blocks


def test_parent_splitter_prefers_paragraph_boundaries() -> None:
    content = "\n\n".join(
        [
            "# Guide\nIntro paragraph about deployment.",
            "Second paragraph covers health checks.",
            "Third paragraph covers rollback steps.",
        ]
    )

    blocks = split_parent_blocks(content, parent_size=70, parent_overlap=10)

    assert len(blocks) >= 2
    assert all(block.content.strip() for block in blocks)
    assert blocks[0].content.startswith("# Guide")
    assert blocks[0].start_offset == 0
    assert blocks[0].end_offset is not None


def test_parent_splitter_splits_oversized_paragraphs_with_overlap() -> None:
    content = " ".join(f"token{index}" for index in range(40))

    blocks = split_parent_blocks(content, parent_size=80, parent_overlap=12)

    assert len(blocks) > 1
    assert all(len(block.content) <= 92 for block in blocks)
    assert blocks[1].start_offset is not None
    assert blocks[0].end_offset is not None
    assert blocks[1].start_offset < blocks[0].end_offset


def test_child_splitter_uses_parent_offsets() -> None:
    parent = TextBlock(
        content="alpha beta gamma delta epsilon zeta eta theta",
        index=3,
        start_offset=25,
        end_offset=68,
    )

    children = split_child_chunks(parent, child_size=18, child_overlap=5)

    assert len(children) > 1
    assert children[0].start_offset == 25
    assert children[0].index == 0
    assert all(child.start_offset is not None and child.end_offset is not None for child in children)
    assert children[1].start_offset < children[0].end_offset
