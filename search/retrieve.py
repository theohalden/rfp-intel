"""Hybrid retrieval, query expansion, metadata filtering, and local reranking."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from time import perf_counter
from typing import Literal, Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from config import Settings
from schemas import (Evidence, SearchChunk, SearchFilters, SearchHit, SearchResponse)
from search.deadlines import prioritize_deadline_evidence, dated_deadline_priority, current_deadline_intent
from search.index import check_manifest, index_lock, read_chunks, vector_store

LOGGER = logging.getLogger(__name__)
SearchMode = Literal['vector', 'hybrid', 'hybrid_rerank']


def adjacent_context(hits: list[SearchHit], corpus: list[SearchChunk], limit: int) -> list[SearchHit]:
    """Keep ranked seeds and at most one canonical neighbor on each side, on the same page.

    Chunks are persisted in document order. Neighbors recover split qualifiers and
    addresses without reading whole documents or crossing a bid/amendment boundary.
    """
    groups: dict[tuple, list[SearchChunk]] = {}
    for chunk in corpus:
        m = chunk.metadata
        groups.setdefault((m.bid_id, m.file_name, m.page_number, m.doc_type, m.addendum_number), []).append(chunk)
    selected = {h.metadata.chunk_id: h for h in hits}
    for hit in hits:
        m = hit.metadata
        page = groups.get((m.bid_id, m.file_name, m.page_number, m.doc_type, m.addendum_number), [])
        position = next((i for i, c in enumerate(page) if c.metadata.chunk_id == m.chunk_id), None)
        if position is None:
            continue
        for i in (position - 1, position + 1):
            if len(selected) >= max(limit, len(hits)):
                break
            if 0 <= i < len(page):
                chunk = page[i]
                selected.setdefault(chunk.metadata.chunk_id, SearchHit(rank=len(selected) + 1,
                    text=chunk.text, metadata=chunk.metadata,
                    source=Evidence(file=chunk.metadata.file_name, page=chunk.metadata.page_number, quote=chunk.text),
                    notes='Adjacent same-page context; not independently reranked.'))
    return [h.model_copy(update={'rank': i + 1}) for i, h in enumerate(selected.values())]


class RankedListRetriever(BaseRetriever):
    """Expose precomputed ranked lists to LangChain's reciprocal rank fusion."""
    documents: list[Document]

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        """Return rankings already computed for a query variant."""
        return self.documents


def tokenize(text: str) -> list[str]:
    """Keep exact hyphenated identifiers alongside their individual components."""
    terms = re.findall(r'[\w]+(?:[-./][\w]+)*', text.casefold())
    return terms + [part for term in terms if re.search(r'[-./]', term)
                    for part in re.split(r'[-./]', term) if part]


def rewrite_query(query: str) -> str:
    """Clarify short intents without inventing document values or dropping long context."""
    deadline = re.search(r'\b(deadline|due date|closing date)\b', query, re.I)
    other_event = re.search(r'\b(questions?|clarifications?|meeting|delivery|payment|registration)\b', query, re.I)
    historical = re.search(r'\b(original|initial|previous|before|historical)\b', query, re.I)
    if deadline and not other_event and not historical:
        clarified = 'What is the final submission due date and time for this bid, including deadline extensions?'
        return clarified if len(query.split()) <= 4 else query + ' ' + clarified
    if re.fullmatch(r'what (?:has )?changed[?]?', query.strip(), re.I):
        return 'What changes or revisions does this addendum make to the original solicitation?'
    return query


def expand_query(query: str) -> list[str]:
    """Add a deterministic synonym query without injecting bid-specific values."""
    groups = [
        (r'\b(deadline|due|closing)\b', 'due date submission deadline closing date'),
        (r'\b(affidavit|forms?|documentation)\b', 'required affidavit certification documentation forms'),
        (r'\b(warranty|support)\b', 'warranty service support coverage repair replacement'),
        (r'\b(bond|security)\b', 'bid bond proposal security surety amount percentage'),
        (r'\b(changed?|addend(?:um|a)|amendment)\b', 'addendum amendment revised changed supersedes'),
        (r'\b(specifications?|specs|processor|ram|storage)\b', 'technical specifications processor memory storage'),
    ]
    additions = [words for pattern, words in groups if re.search(pattern, query, re.I)]
    rewritten = rewrite_query(query)
    if rewritten != query:
        return [query, rewritten]
    return [query, query + ' ' + ' '.join(additions)] if additions else [query]


def matches(chunk: SearchChunk, filters: SearchFilters) -> bool:
    """Apply the same exact scope to lexical results and the final fused ranking."""
    metadata = chunk.metadata
    for key in ('bid_id', 'doc_type', 'addendum_number', 'file_name', 'page_number'):
        value = getattr(filters, key)
        if value is not None and getattr(metadata, key) != value:
            return False
    return not (filters.exclude_addenda and metadata.doc_type == 'addendum')


def chroma_filter(filters: SearchFilters) -> dict[str, Any] | None:
    """Build Chroma's single-clause or explicit conjunction filter syntax."""
    clauses = [{key: {'$eq': value}} for key in
               ('bid_id', 'doc_type', 'addendum_number', 'file_name', 'page_number')
               if (value := getattr(filters, key)) is not None]
    if filters.exclude_addenda:
        clauses.append({'doc_type': {'$ne': 'addendum'}})
    return {'$and': clauses} if len(clauses) > 1 else clauses[0] if clauses else None


def fuse(rankings: list[list[Document]]) -> list[Document]:
    """Use the required LangChain EnsembleRetriever for weighted RRF, by chunk ID."""
    nonempty = [ranking for ranking in rankings if ranking]
    if not nonempty:
        return []
    ensemble = EnsembleRetriever(
        retrievers=[RankedListRetriever(documents=ranking) for ranking in nonempty],
        weights=[1 / len(nonempty)] * len(nonempty), id_key='chunk_id', c=60,
    )
    return ensemble.invoke('rank fusion')


def select_candidates(rankings: list[list[Document]], limit: int) -> list[Document]:
    """Fuse rankings while retaining a few leading hits from every retrieval branch."""
    selected = fuse(rankings)[:limit]
    nonempty = [r for r in rankings if r]
    if not nonempty:
        return selected
    quota = max(1, min(3, limit // len(nonempty)))
    reserved = {d.metadata['chunk_id']: d for r in nonempty for d in r[:quota]}
    for key, document in reserved.items():
        if any(d.metadata['chunk_id'] == key for d in selected):
            continue
        replace = next((i for i in reversed(range(len(selected)))
                        if selected[i].metadata['chunk_id'] not in reserved), None)
        if replace is not None:
            selected[replace] = document
    return selected


def deduplicate_passages(ranked: list[tuple[SearchChunk, float | None]]) -> list[tuple[SearchChunk, float | None]]:
    """Collapse near-contained passages on the same physical source page only."""
    def content_words(text: str) -> set[str]:
        text = re.sub(r'(?im)^\s*page\s+\d+\s*(?:\||of|/)\s*\d+\s*$', '', text)
        return set(tokenize(text))
    result: list[tuple[SearchChunk, float | None]] = []
    for chunk, score in ranked:
        words = content_words(chunk.text)
        duplicate = False
        for i, (existing, _) in enumerate(result):
            same_page = all(getattr(chunk.metadata, key) == getattr(existing.metadata, key)
                            for key in ('bid_id', 'file_name', 'page_number'))
            if not same_page:
                continue
            other = content_words(existing.text)
            smaller = min(len(words), len(other))
            if smaller and len(words & other) / smaller >= .9:
                if len(chunk.text) > len(existing.text):
                    result[i] = (chunk, score)
                duplicate = True
                break
        if not duplicate:
            result.append((chunk, score))
    return result


def rerank_candidates(query: str, chunks: list[SearchChunk], model: Any) -> list[tuple[SearchChunk, float]]:
    """Score complete chunks via overlapping model windows, retaining full citations.

    Model windows are internal scoring inputs, never new index chunks. Taking the
    best window avoids always ignoring the end of a long, indivisible table.
    """
    pairs: list[list[str]] = []
    owners: list[int] = []
    query_ids = model.tokenizer.encode(query, add_special_tokens=False)
    if len(query_ids) > 128:
        query_ids = query_ids[:128]
    scoring_query = model.tokenizer.decode(query_ids, skip_special_tokens=True)
    window = 512 - len(query_ids) - 8
    for owner, chunk in enumerate(chunks):
        ids = model.tokenizer.encode(chunk.text, add_special_tokens=False, truncation=False, verbose=False)
        for start in range(0, max(1, len(ids)), window - 64):
            passage = model.tokenizer.decode(ids[start:start + window], skip_special_tokens=True)
            pairs.append([scoring_query, passage])
            owners.append(owner)
            if start + window >= len(ids):
                break
    if not pairs:
        return []
    scores = model.predict(pairs, batch_size=8, show_progress_bar=False)
    best = [float('-inf')] * len(chunks)
    for owner, score in zip(owners, scores):
        best[owner] = max(best[owner], float(score))
    return sorted(zip(chunks, best), key=lambda item: item[1], reverse=True)


class SearchEngine:
    """Reusable search service with lazy models and process-safe index snapshots."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._store = None

    def search(self, query: str, filters: SearchFilters | None = None, top_k: int | None = None,
               mode: SearchMode = 'hybrid_rerank', expand: bool = True,
               include_neighbors: bool = False) -> SearchResponse:
        """Return source passages; scores are ranking signals, not confidence values."""
        query = query.strip()
        if not query or len(query) > 2000:
            raise ValueError('Query must contain 1–2000 characters')
        if mode not in ('vector', 'hybrid', 'hybrid_rerank'):
            raise ValueError('Unknown search mode')
        top_k = self.settings.search.top_k if top_k is None else top_k
        if not 1 <= top_k <= 50:
            raise ValueError('top_k must be between 1 and 50')
        filters = filters or SearchFilters()
        started = perf_counter()
        variants = expand_query(query) if expand else [query]
        warnings: list[str] = []
        with index_lock(self.settings.paths.index):
            if (self.settings.paths.index / '.updating').exists():
                raise RuntimeError('An index update was interrupted; rerun the index command to repair it')
            manifest_path = self.settings.paths.index / 'manifest.json'
            if not manifest_path.exists():
                raise FileNotFoundError('No index found. Run python main.py index --bid <folder> first')
            check_manifest(self.settings, json.loads(manifest_path.read_text()))
            corpus = [c for c in read_chunks(self.settings.paths.index) if matches(c, filters)]
            mapping = {c.metadata.chunk_id: c for c in corpus}
            if corpus:
                if self._store is None:
                    self._store = vector_store(self.settings)
                candidate_k = min(len(corpus), max(top_k, self.settings.search.candidate_k))
                where = chroma_filter(filters)
                lexical = None
                if mode != 'vector':
                    lexical = BM25Retriever.from_documents(
                        [Document(page_content=c.text, metadata=c.metadata.model_dump()) for c in corpus],
                        preprocess_func=tokenize, k=candidate_k,
                    )
                rankings = []
                for variant in variants:
                    # BGE's recommended query instruction is applied only to dense queries.
                    dense_query = 'Represent this sentence for searching relevant passages: ' + variant
                    rankings.append(self._store.similarity_search(dense_query, k=candidate_k, filter=where))
                    if lexical is not None:
                        terms = set(tokenize(variant))
                        rankings.append([d for d in lexical.invoke(variant)
                                         if terms.intersection(tokenize(d.page_content))])
                ranked = select_candidates(rankings, candidate_k)
                candidates = [mapping[d.metadata['chunk_id']] for d in ranked
                              if d.metadata.get('chunk_id') in mapping][:candidate_k]
                if mode == 'hybrid_rerank':
                    from search.models import reranker
                    ranked_chunks = rerank_candidates(rewrite_query(query) if expand else query, candidates, reranker(
                        self.settings.search.reranker_model, str(self.settings.paths.index)))
                else:
                    ranked_chunks = [(c, None) for c in candidates]
            else:
                ranked_chunks = []
                warnings.append('No indexed passages match these filters')
            if mode == 'hybrid_rerank':
                ranked_chunks = prioritize_deadline_evidence(query, ranked_chunks)
            ranked_chunks = deduplicate_passages(ranked_chunks)
            hits = [SearchHit(rank=i + 1, text=c.text, metadata=c.metadata,
                    source=Evidence(file=c.metadata.file_name, page=c.metadata.page_number, quote=c.text),
                    rerank_score=score,
                    notes=('Dated deadline evidence prioritized; verify the cited wording and applicable amendments. '
                           if mode == 'hybrid_rerank' and current_deadline_intent(query)
                           and dated_deadline_priority(c)[0] else '') +
                          ('Long table retained intact; dense embedding may truncate it' if
                           c.metadata.kind == 'table' and len(c.text) > self.settings.chunking.size else ''))
                    for i, (c, score) in enumerate(ranked_chunks[:top_k]) if matches(c, filters)]
            if include_neighbors:
                hits = adjacent_context(hits, corpus, limit=min(15, top_k * 3))
        response = SearchResponse(query=query, expanded_queries=variants, mode=mode,
            filters=filters, hits=hits, rerank_query=(rewrite_query(query) if expand else query)
            if mode == 'hybrid_rerank' else None, latency_ms=round((perf_counter() - started) * 1000, 2), warnings=warnings)
        self._log(response, include_neighbors=include_neighbors)
        return response

    def _log(self, response: SearchResponse, include_neighbors: bool = False) -> None:
        """Append a structured retrieval trace without model weights or secrets."""
        destination = self.settings.paths.outputs / 'search_trace.jsonl'
        destination.parent.mkdir(parents=True, exist_ok=True)
        event = {'timestamp': datetime.now(timezone.utc).isoformat(), 'agent': 'retrieval',
                 'input': {'query': response.query, 'filters': response.filters.model_dump(),
                           'include_neighbors': include_neighbors},
                 'tool_calls': ['vector'] + (['bm25', 'rrf'] if response.mode != 'vector' else [])
                               + (['cross_encoder'] if response.mode == 'hybrid_rerank' else []),
                 'output': response.model_dump(), 'tokens': 0,
                 'latency_ms': response.latency_ms}
        with destination.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event) + '\n')
