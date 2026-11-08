"""Heading-aware text chunking with complete, separate table chunks."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from config import Chunking
from ingestion.parse import clean_text, markdown_tables
from schemas import ChunkMetadata, ParsedPage, SearchChunk


def page_blocks(page: ParsedPage) -> Iterator[tuple[str, str]]:
    """Yield prose/table blocks in order, including tables outside page markdown."""
    tables = list(dict.fromkeys([*markdown_tables(page.text), *page.tables]))
    position = 0
    spans: list[tuple[int, int, str]] = []
    extra: list[str] = []
    for table in tables:
        normalized = clean_text(table)
        matches = list(re.finditer(re.escape(normalized), page.text))
        if not matches:
            extra.append(normalized)
        else:
            spans.extend((m.start(), m.end(), normalized) for m in matches)
    seen: set[str] = set()
    for start, end, table in sorted(spans):
        if start < position:
            continue
        if page.text[position:start].strip():
            yield 'text', page.text[position:start]
        if table not in seen:
            yield 'table', table
            seen.add(table)
        position = end
    if page.text[position:].strip():
        yield 'text', page.text[position:]
    for table in extra:
        if table and table not in seen:
            yield 'table', table
            seen.add(table)


def chunk_pages(pages: list[ParsedPage], options: Chunking) -> list[SearchChunk]:
    """Keep physical page boundaries, section headings, and tables intact."""
    headers = MarkdownHeaderTextSplitter(
        headers_to_split_on=[('#', 'h1'), ('##', 'h2'), ('###', 'h3'), ('####', 'h4')],
        strip_headers=False,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=options.size, chunk_overlap=options.overlap,
        separators=['\n\n', '\n', '. ', ' ', ''],
    )
    chunks: list[SearchChunk] = []
    seen_ids: set[str] = set()
    for page in pages:
        section = ''
        for kind, block in page_blocks(page):
            if kind == 'table':
                pieces = [(block, section)]
            else:
                pieces = []
                for document in headers.split_text(block):
                    if document.metadata:
                        section = ' > '.join(document.metadata.values())
                    pieces.extend((text, section) for text in splitter.split_text(document.page_content))
            for text, heading in pieces:
                text = text.strip()
                if not text or (kind == 'text' and re.sub(r'[*`#_\s]', '', text) == str(page.metadata.page_number)):
                    continue
                identity = '\0'.join([page.metadata.model_dump_json(), kind, heading, text])
                chunk_id = hashlib.sha256(identity.encode()).hexdigest()
                if chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)
                chunks.append(SearchChunk(text=text, metadata=ChunkMetadata(
                    **page.metadata.model_dump(), chunk_id=chunk_id, kind=kind, section=heading)))
    return chunks
