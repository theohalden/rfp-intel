"""Verify chunking, real Chroma persistence, fusion, filters, and HTTP contracts."""
from __future__ import annotations

import hashlib
from pathlib import Path
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from fastapi.testclient import TestClient

from config import Chunking, Settings
from schemas import ChunkMetadata, PageMetadata, ParsedPage, SearchChunk, SearchFilters
from search.chunk import chunk_pages
from search.index import add_bid, read_chunks
from search.retrieve import SearchEngine, fuse, rerank_candidates, tokenize


def page(text: str, tables: list[str] | None = None, number: int = 1) -> ParsedPage:
    return ParsedPage(metadata=PageMetadata(bid_id='sample', file_name='input.pdf',
        doc_type='specs', page_number=number), text=text, tables=tables or [])


def test_tables_are_never_split_and_ids_are_stable() -> None:
    table = '| Part | Description |\n| --- | --- |\n' + '\n'.join(
        f'| SKU-{i} | Component {i} |' for i in range(100))
    pages = [page('# Devices\n\n' + 'Prose sentence. ' * 100 + '\n\n' + table, [table])]
    chunks = chunk_pages(pages, Chunking())
    tables = [c for c in chunks if c.metadata.kind == 'table']
    assert len(tables) == 1 and tables[0].text == table
    assert len(tables[0].text) > 800
    assert all(len(c.text) <= 800 for c in chunks if c.metadata.kind == 'text')
    assert [c.metadata.chunk_id for c in chunks] == [c.metadata.chunk_id for c in chunk_pages(pages, Chunking())]
    assert all(c.metadata.page_number == 1 for c in chunks)
    assert 'Devices' in tables[0].metadata.section


def test_identical_passages_on_different_pages_keep_distinct_sources() -> None:
    chunks = chunk_pages([page('Same text', number=1), page('Same text', number=2)], Chunking())
    assert len(chunks) == 2
    assert chunks[0].metadata.chunk_id != chunks[1].metadata.chunk_id


def test_rrf_rewards_agreement_and_deduplicates_ids() -> None:
    def doc(key):
        return Document(page_content=key, metadata={'chunk_id': key})
    result = fuse([[doc('A'), doc('B')], [doc('C'), doc('B')]])
    assert result[0].metadata['chunk_id'] == 'B'
    assert len(result) == 3
    assert '210-blyz' in tokenize('210-BLYZ') and '210' in tokenize('210-BLYZ')


class ToyEmbeddings(Embeddings):
    """Deterministic tiny vectors for storage tests, not retrieval quality claims."""
    calls: int = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += len(texts)
        return [[(v + 1) / 256 for v in hashlib.sha256(text.encode()).digest()[:8]] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@pytest.fixture()
def indexed(tmp_path: Path, monkeypatch):
    import search.models
    embedding = ToyEmbeddings()
    monkeypatch.setattr(search.models, 'embeddings', lambda *a: embedding)
    settings = Settings()
    settings.paths.index = tmp_path / 'index'
    settings.paths.outputs = tmp_path / 'outputs'
    settings.parsing.ocr_enabled = False
    for name, phrase in [('alpha', 'Submission deadline October 12. Warranty three years.'),
                         ('beta', 'Required mercury affidavit. Payment Net 30.')]:
        folder = tmp_path / name
        folder.mkdir()
        (folder / 'portal.html').write_text(f'<main><h1>Bid</h1><p>{phrase}</p></main>')
        add_bid(folder, settings)
    return settings, embedding, tmp_path


def test_incremental_index_noop_updates_and_other_bid_survives(indexed) -> None:
    settings, embedding, root = indexed
    before = embedding.calls
    second_bid_ids = {c.metadata.chunk_id for c in read_chunks(settings.paths.index) if c.metadata.bid_id == 'beta'}
    result = add_bid(root / 'alpha', settings)
    assert result.added == result.removed == 0
    assert result.unchanged > 0 and not result.parsed
    assert embedding.calls == before
    (root / 'alpha' / 'portal.html').write_text('<main>Updated deadline November 20.</main>')
    result = add_bid(root / 'alpha', settings)
    assert result.added > 0 and result.removed > 0 and result.parsed
    assert second_bid_ids == {c.metadata.chunk_id for c in read_chunks(settings.paths.index)
                              if c.metadata.bid_id == 'beta'}
    # Reopening a service exercises persistent storage, not an in-memory-only index.
    response = SearchEngine(settings).search('deadline', SearchFilters(bid_id='alpha'), mode='hybrid')
    assert response.hits
    assert all(h.metadata.bid_id == 'alpha' and 'November' in h.text for h in response.hits)


def test_filter_isolation_empty_scope_and_citations(indexed) -> None:
    settings, _, _ = indexed
    engine = SearchEngine(settings)
    response = engine.search('affidavit', SearchFilters(bid_id='beta', doc_type='bid_page'), mode='hybrid')
    assert response.hits
    for hit in response.hits:
        assert hit.metadata.bid_id == 'beta'
        assert hit.source.file == 'portal.html' and hit.source.page == 1
        assert hit.source.quote == hit.text
    assert not engine.search('deadline', SearchFilters(bid_id='missing'), mode='hybrid').hits
    assert not engine.search('deadline', SearchFilters(addendum_number=2), mode='hybrid').hits
    with pytest.raises(ValueError):
        SearchFilters(exclude_addenda=True, addendum_number=2)


def test_http_validation_and_guarded_qa(indexed, monkeypatch) -> None:
    from search.api import create_app
    settings, _, _ = indexed
    from schemas import QAResponse
    class FakeQA:
        def ask(self, question, bid_id=None, doc_type=None):
            return QAResponse(question=question, answer='Not found in documents', claims=[], sources=[],
                not_found=True, notes='', latency_ms=0, trace_file='local-test')
    monkeypatch.setattr('agents.qa.QAEngine', lambda settings: FakeQA())
    with TestClient(create_app(settings)) as client:
        assert client.get('/health').status_code == 200
        response = client.get('/search', params={'q': 'deadline', 'bid_id': 'alpha', 'mode': 'hybrid'})
        assert response.status_code == 200
        assert response.json()['hits'][0]['source']['page'] == 1
        assert client.get('/search', params={'q': 'x', 'top_k': 0}).status_code == 422
        assert client.get('/search', params={'q': 'x', 'doc_type': 'invalid'}).status_code == 422
        assert client.post('/index', json={'bid_dir': '/does-not-exist'}).status_code == 400
        assert client.post('/ask', json={'question': 'deadline?'}).status_code == 200
        assert client.post('/ask', json={'question': 'deadline?', 'doc_type': 'invalid'}).status_code == 422


def test_interrupted_index_blocks_search_and_repairs(indexed) -> None:
    settings, _, root = indexed
    (settings.paths.index / '.updating').write_text('alpha')
    with pytest.raises(RuntimeError, match='interrupted'):
        SearchEngine(settings).search('deadline', mode='hybrid')
    with pytest.raises(ValueError, match='First repair'):
        add_bid(root / 'beta', settings)
    add_bid(root / 'alpha', settings)
    assert not (settings.paths.index / '.updating').exists()


def test_reranker_sees_end_of_long_table_without_splitting_result() -> None:
    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text.encode())
        def decode(self, ids, **kwargs):
            return bytes(ids).decode()
    class Model:
        tokenizer = Tokenizer()
        def predict(self, pairs, **kwargs):
            return [1.0 if 'TARGET' in passage else 0.0 for _, passage in pairs]
    chunk = SearchChunk(text='A' * 1600 + ' TARGET', metadata=ChunkMetadata(
        bid_id='sample', file_name='specs.pdf', doc_type='specs', page_number=1,
        chunk_id='one', kind='table'))
    results = rerank_candidates('TARGET', [chunk], Model())
    assert results == [(chunk, 1.0)]
