"""Readable terminal previews; full evidence remains available via JSON/full-text."""
from __future__ import annotations

import re
from schemas import SearchResponse
from search.deadlines import LABEL, current_deadline_intent
from search.retrieve import tokenize

DATE = re.compile(r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}-[A-Za-z]{3}-\d{4}|'
                  r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
                  r'\s+\d{1,2},?\s+\d{4})\b', re.I)


def passage_preview(text: str, query: str, limit: int = 650) -> str:
    """Show a relevant source excerpt without generating new facts."""
    readable = re.sub(r'\*{1,2}|`', '', text.replace('<br>', '\n'))
    lines = [line.strip() for line in readable.splitlines() if line.strip()]
    if not lines:
        return ''
    terms = set(tokenize(query)) - {'what', 'is', 'the', 'for', 'this', 'a', 'of', 'and'}
    deadline = bool(re.search(r'\b(deadline|due date|closing date)\b', query, re.I))
    scores = [len(terms & set(tokenize(line))) * 2 +
              (7 if deadline and DATE.search(line) else 0) for line in lines]
    if deadline:
        for i, line in enumerate(lines):
            if LABEL.search(line) and not re.search(r'questions?|clarification', line, re.I) and DATE.search(' '.join(lines[i:i+2])):
                scores[i] += 20
    best = max(range(len(lines)), key=lambda i: scores[i]) if max(scores) else 0
    start = best if deadline and LABEL.search(lines[best]) else max(0, best - 1)
    end = min(len(lines), max(best + 2, start + 4))
    if deadline and LABEL.search(lines[best]):
        end = min(len(lines), best + (1 if DATE.search(lines[best]) else 2))
    excerpt = '\n'.join(lines[start:end])
    clipped = len(excerpt) > limit
    excerpt = excerpt[:limit].rstrip()
    return ('…\n' if start else '') + excerpt + ('\n…' if end < len(lines) or clipped else '')


def format_search(response: SearchResponse, full_text: bool = False, details: bool = False) -> str:
    """Separate readable source evidence from optional retrieval diagnostics."""
    output = [f'Search results for: {response.query}',
              f'{len(response.hits)} source passage(s). These are document excerpts, not a generated answer.']
    if not full_text and response.hits:
        output.append('Showing previews; add --full-text for complete passages or --json for structured results.')
    if current_deadline_intent(response.query) and response.mode == 'hybrid_rerank':
        output.append('Dated deadline amendments are prioritized; base documents may show earlier dates.')
    if details:
        output.append(f'Mode: {response.mode} | Time: {response.latency_ms / 1000:.1f}s')
        if response.rerank_query:
            output.append(f'Reranking query: {response.rerank_query}')
    for hit in response.hits:
        kind = (f'Addendum {hit.metadata.addendum_number}' if hit.metadata.addendum_number
                else 'RFP' if hit.metadata.doc_type == 'rfp'
                else hit.metadata.doc_type.replace('_', ' ').title())
        output.extend(['', f'[{hit.rank}] {hit.metadata.bid_id} · {kind} · page {hit.source.page}',
                       f'Source: {hit.source.file}', '',
                       hit.text if full_text else passage_preview(hit.text, response.query)])
        if details:
            if hit.rerank_score is not None:
                output.append(f'Relevance score: {hit.rerank_score:.6g} (not factual confidence)')
            if hit.notes:
                output.append(f'Note: {hit.notes}')
    output.extend(f'Note: {warning}' for warning in response.warnings)
    return '\n'.join(output)
