"""Source-anchored Recall@1/3/5 and MRR@5 for the three retrieval modes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Settings, load_settings
from schemas import EvalQuestion, EvalSet, GoldSource, SearchHit
from search.retrieve import SearchEngine
from search.index import read_chunks

MODES = ('vector', 'hybrid', 'hybrid_rerank')


def normalize_evidence(text: str) -> str:
    """Ignore markup/whitespace variations, retaining actual words and numbers."""
    text = unicodedata.normalize('NFKC', text).casefold()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[*`#|]', ' ', text)
    return ' '.join(text.split())


def source_matches(source: GoldSource, hit: SearchHit) -> bool:
    """Require correct bid, filename, physical page, and the supporting anchor."""
    return (source.bid_id == hit.metadata.bid_id and source.file == hit.source.file
            and source.page == hit.source.page
            and normalize_evidence(source.quote) in normalize_evidence(hit.text))


def score_question(question: EvalQuestion, hits: list[SearchHit]) -> dict[str, float]:
    """Macro-ready target recall; one hit may cover multiple required facts."""
    ranks = [next((rank for rank, hit in enumerate(hits[:5], 1)
                   if any(source_matches(source, hit) for source in target.sources)), None)
             for target in question.targets]
    first = min((rank for rank in ranks if rank is not None), default=None)
    result = {f'recall_at_{k}': sum(rank is not None and rank <= k for rank in ranks) / len(ranks)
              for k in (1, 3, 5)}
    result['mrr_at_5'] = 1 / first if first else 0.0
    return result


def validate_gold(dataset: EvalSet, settings: Settings) -> None:
    """Check every hand-labelled anchor against saved source pages before scoring."""
    pages: dict[tuple[str, str, int], str] = {}
    bids = {source.bid_id for q in dataset.questions for t in q.targets for source in t.sources}
    for bid in bids:
        path = settings.paths.outputs / 'ingestion' / bid / 'pages.jsonl'
        for line in path.read_text().splitlines():
            page = json.loads(line)
            metadata = page['metadata']
            pages[(bid, metadata['file_name'], metadata['page_number'])] = page['text']
    for question in dataset.questions:
        for target in question.targets:
            for source in target.sources:
                key = (source.bid_id, source.file, source.page)
                if key not in pages or normalize_evidence(source.quote) not in normalize_evidence(pages[key]):
                    raise ValueError(f'{question.id}: expected quote not found on source page {key}')
                filters = question.filters
                if filters.bid_id is not None and filters.bid_id != source.bid_id:
                    raise ValueError(f'{question.id}: gold source contradicts bid filter')


def gold_index_coverage(dataset: EvalSet, settings: Settings) -> list[dict]:
    """Disclose gold anchors spanning chunk boundaries, without dropping their scores."""
    chunks = read_chunks(settings.paths.index)
    missing = []
    for question in dataset.questions:
        for index, target in enumerate(question.targets, 1):
            if not any(source.bid_id == chunk.metadata.bid_id
                       and source.file == chunk.metadata.file_name
                       and source.page == chunk.metadata.page_number
                       and normalize_evidence(source.quote) in normalize_evidence(chunk.text)
                       for source in target.sources for chunk in chunks):
                missing.append({'id': question.id, 'target': index,
                                'reason': 'No single indexed chunk contains the complete gold anchor'})
    return missing


def render_results(records: list[dict], metadata: dict) -> str:
    """Report measured scores and disclose the development-set and policy limits."""
    lines = ['# Retrieval evaluation', '',
             f"Evaluated {metadata['questions']} manually labelled questions over both bids.", '',
             '| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 | Errors |',
             '|---|---:|---:|---:|---:|---:|']
    for mode in MODES:
        rows = [row for row in records if row['mode'] == mode]
        metrics = [mean(row['metrics'][key] for row in rows) for key in
                   ('recall_at_1', 'recall_at_3', 'recall_at_5', 'mrr_at_5')]
        lines.append(f"| {mode} | " + ' | '.join(f'{value:.3f}' for value in metrics)
                     + f" | {sum(row['status'] == 'error' for row in rows)} |")
    uncovered = metadata.get('gold_targets_without_complete_single_chunk', [])
    if uncovered:
        lines.extend(['', 'Index coverage note: ' + ', '.join(item['id'] for item in uncovered)
                      + ' has a gold anchor that is not wholly contained in one indexed chunk. These targets remain in the denominator and count as misses under this strict passage metric.'])
    lines.extend(['', '## Method', '',
        '- Same indexed corpus, query expansion enabled, metadata filters, and top-5 budget for all modes.',
        '- A relevant hit must contain the annotated quote anchor on the correct bid/file/physical page. A filename/page match alone does not count.',
        '- Recall@k is the fraction of required evidence targets covered in the first k hits, averaged equally across questions. Alternative sources for one target count once.',
        '- MRR@5 is the mean reciprocal rank of the first relevant hit within five results. It does not measure whether every target in a multi-source question was found.',
        '- The cross-bid warranty question requires evidence from both bids. Multiple chunks from one source do not inflate recall.',
        '- hybrid_rerank includes the documented dated-deadline evidence ordering and same-page passage deduplication after cross-encoder scoring.',
        '- Failed requests receive zero metrics and are retained in the denominator, with errors saved in results.json.', '',
        '## Limitations', '',
        '- This is a small development set, not a held-out benchmark. The previously debugged deadline and SKU cases are explicitly tagged regression cases.',
        '- The source judgments are manually selected and not exhaustive. An unannotated but valid source can be counted as a miss.',
        '- These scores measure retrieval of supporting passages, not generated-answer correctness, complete extraction, or reliable conflict resolution.',
        '- OCR anchors are checked against saved OCR text; this is not a measurement of character recognition accuracy.',
        '- Results depend on the frozen corpus, local model revisions, chunking, expansion, and retrieval policies recorded in results.json.', '',
        '## Misses at five results', ''])
    misses = [row for row in records if row['metrics']['recall_at_5'] < 1]
    if not misses:
        lines.append('No annotated targets missed within the first five results.')
    else:
        lines += [f"- {row['mode']} / {row['id']}: Recall@5={row['metrics']['recall_at_5']:.2f}; {row['query']}"
                  for row in misses]
    return '\n'.join(lines) + '\n'


def run(questions_path: Path, output_path: Path, settings: Settings) -> dict:
    """Validate gold, execute real retrieval, and save full per-question evidence."""
    dataset = EvalSet.model_validate(json.loads(questions_path.read_text()))
    validate_gold(dataset, settings)
    engine = SearchEngine(settings)
    records: list[dict] = []
    manifest = settings.paths.index / 'manifest.json'
    registry = settings.paths.index / 'models' / 'registry.json'
    metadata = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'questions': len(dataset.questions), 'gold_targets_without_complete_single_chunk': gold_index_coverage(dataset, settings), 'question_set_sha256': hashlib.sha256(questions_path.read_bytes()).hexdigest(),
                'config': settings.model_dump(mode='json'), 'index_manifest': json.loads(manifest.read_text()),
                'model_snapshots': json.loads(registry.read_text()), 'top_k': 5, 'expand': True,
                'retrieval_code_sha256': hashlib.sha256((ROOT / 'search/retrieve.py').read_bytes()).hexdigest(),
                'deadline_policy_sha256': hashlib.sha256((ROOT / 'search/deadlines.py').read_bytes()).hexdigest()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        for question in dataset.questions:
            record = {'id': question.id, 'query': question.query, 'mode': mode, 'status': 'ok',
                      'filters': question.filters.model_dump(), 'targets': [t.model_dump() for t in question.targets]}
            try:
                response = engine.search(question.query, question.filters, top_k=5, mode=mode)
                record['metrics'] = score_question(question, response.hits)
                record['response'] = response.model_dump()
            except Exception as exc:
                import logging
                logging.exception('Evaluation failed: %s / %s', mode, question.id)
                record.update(status='error', error=f'{type(exc).__name__}: {exc}',
                              metrics={key: 0.0 for key in ('recall_at_1', 'recall_at_3', 'recall_at_5', 'mrr_at_5')})
            records.append(record)
            output_path.with_suffix('.json').write_text(json.dumps({'metadata': metadata, 'records': records}, indent=2))
            print(f"{mode}: {question.id} Recall@5={record['metrics']['recall_at_5']:.2f} [{record['status']}]", flush=True)
    output_path.write_text(render_results(records, metadata))
    print(f'Saved {output_path}', flush=True)
    return {'metadata': metadata, 'records': records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=Path, default=ROOT / 'eval/questions.yaml')
    parser.add_argument('--output', type=Path, default=ROOT / 'eval/results.md')
    parser.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    args = parser.parse_args()
    result = run(args.questions, args.output, load_settings(args.config))
    if any(row['status'] == 'error' for row in result['records']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
