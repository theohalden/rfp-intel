"""Run real-model examples and incremental/unseen-folder checks for checkpoint 2."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings
from schemas import SearchFilters
from search.index import add_bid, read_chunks
from search.retrieve import SearchEngine


def main() -> None:
    """Save complete sample passages and measured verification results."""
    settings = load_settings()
    settings.paths.outputs.mkdir(parents=True, exist_ok=True)
    before = {c.metadata.chunk_id for c in read_chunks(settings.paths.index) if c.metadata.bid_id == 'Bid1'}
    noop = add_bid(settings.paths.data / 'Bid1', settings)
    assert noop.added == noop.removed == 0 and not noop.parsed
    assert before == {c.metadata.chunk_id for c in read_chunks(settings.paths.index) if c.metadata.bid_id == 'Bid1'}
    print('PASS: existing Bid1 index unchanged on repeat indexing', flush=True)
    engine = SearchEngine(settings)
    cases = [
        ('deadline_addendum_2', 'submission deadline extension', SearchFilters(bid_id='Bid1', doc_type='addendum', addendum_number=2)),
        ('processor_sku', '379-BFNZ processor', SearchFilters(bid_id='Bid2', doc_type='specs')),
        ('affidavits', 'required affidavits', SearchFilters(bid_id='Bid2')),
        ('warranty_across_bids', 'warranty requirements support coverage', SearchFilters()),
        ('bid_bond', 'bid bond amount or percentage', SearchFilters(bid_id='Bid1')),
    ]
    examples = []
    for label, query, filters in cases:
        started = perf_counter()
        response = engine.search(query, filters, top_k=3)
        assert response.hits
        assert all(hit.source.quote == hit.text for hit in response.hits)
        if filters.bid_id:
            assert all(hit.metadata.bid_id == filters.bid_id for hit in response.hits)
        if filters.addendum_number:
            assert all(hit.metadata.addendum_number == filters.addendum_number for hit in response.hits)
        if label == 'processor_sku':
            assert '379-BFNZ' in response.hits[0].text
            assert response.hits[0].metadata.page_number == 1
        examples.append({'label': label, **response.model_dump()})
        print(f'PASS: {label} ({perf_counter()-started:.1f}s): '
              f'{response.hits[0].source.file}, page {response.hits[0].source.page}', flush=True)
        (settings.paths.outputs / 'search_examples.json').write_text(json.dumps(examples, indent=2))
    for mode in ('vector', 'hybrid'):
        result = engine.search('submission deadline', SearchFilters(bid_id='Bid1'), top_k=3, mode=mode)
        assert result.hits
        examples.append({'label': f'mode_{mode}', **result.model_dump()})
    assert not engine.search('deadline', SearchFilters(bid_id='unknown_bid')).hits
    (settings.paths.outputs / 'search_examples.json').write_text(json.dumps(examples, indent=2))
    # Keep renamed-copy verification isolated so cross-bid searches do not contain duplicate bids.
    unseen = settings.model_copy(deep=True)
    unseen.paths.index = settings.paths.index / 'unseen_test'
    cache = unseen.paths.index / 'models'
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copy2(settings.paths.index / 'models' / 'registry.json', cache / 'registry.json')
    result = add_bid(settings.paths.data / 'Bid3_test', unseen)
    response = SearchEngine(unseen).search('379-BFNZ processor', SearchFilters(bid_id='Bid3_test', doc_type='specs'), top_k=1)
    assert response.hits and '379-BFNZ' in response.hits[0].text
    print('PASS: renamed Bid3_test indexes and searches unchanged code', flush=True)
    report = ['# Checkpoint 2 verification', '', 'Real local embedding and cross-encoder models used on CPU.', '',
              f'- Main index: {len(read_chunks(settings.paths.index))} chunks across Bid1 and Bid2.',
              f'- Repeat indexing: {noop.added} added, {noop.removed} removed, {noop.unchanged} unchanged; parsed={noop.parsed}.',
              f'- Renamed Bid3_test: {result.chunks} chunks in isolated index; processor/SKU search passed.',
              '- Modes vector, hybrid, and hybrid_rerank exercised.',
              '- Unknown-bid filter returned no results.',
              '- Every example citation quote matches its returned passage.', '', '## Sample top results', '']
    for item in examples:
        hit = item['hits'][0]
        report += [f"### {item['label']}", '', f"Query: {item['query']}", '',
                   f"Top source: {hit['metadata']['bid_id']} / {hit['source']['file']}, physical page {hit['source']['page']}.",
                   '', f"Mode: {item['mode']}; latency: {item['latency_ms']:.0f} ms.", '']
    report += ['## Limits', '', 'These smoke examples are not the formal Recall@k/MRR evaluation. Scores do not prove factual correctness or completeness. Automatic addendum reconciliation and generated answers remain pending. Full passages are in search_examples.json; structured retrieval traces are in search_trace.jsonl.']
    (settings.paths.outputs / 'checkpoint2.md').write_text('\n'.join(report) + '\n')


if __name__ == '__main__':
    main()
