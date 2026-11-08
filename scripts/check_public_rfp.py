"""Independent public-RFP smoke evaluation; test data never enters runtime prompts."""
from pathlib import Path
import hashlib
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import load_settings, ROOT
from search.retrieve import SearchEngine
from schemas import SearchFilters
from agents.qa import QAEngine

CASES = [
    ('quantities', 'What laptop categories and quantities are being purchased?', [2], ['High-End Laptops', 'Quantity']),
    ('warranty', 'What warranty period and service type are required?', [6, 7], ['3 years carry-in']),
    ('delivery', 'What is the maximum delivery time after receipt of the purchase order?', [5], ['6 weeks', 'purchase order']),
    ('submission', 'Can bids be submitted electronically, or must they be physically delivered?', [3], ['Electronic submissions shall not']),
    ('deadline', 'What is the tender submission deadline, and does the source explicitly give a time zone?', [4], ['July 2025', '11hr00']),
    ('authorization', 'What manufacturer or distributor authorization documentation is required?', [5], ['Authorization', 'Distributor']),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'outputs/public_lca_test/config.json')
    args = parser.parse_args()
    settings = load_settings(args.config)
    out = settings.paths.outputs
    manifest = json.loads((out / 'source_manifest.json').read_text())
    case_file = out / 'cases.json'
    specification = json.loads(case_file.read_text()) if case_file.exists() else {}
    cases = specification.get('cases', CASES)
    search = SearchEngine(settings)
    retrieval = []
    for label, question, pages, anchors in cases:
        print('Search:', label, flush=True)
        result = search.search(question, SearchFilters(bid_id=manifest['bid_id']), top_k=5)
        relevant = [hit.rank for hit in result.hits if hit.source.page in pages
                    and all(anchor.casefold() in ' '.join(hit.text.split()).casefold() for anchor in anchors)]
        retrieval.append({'case': label, 'gold_pages': pages, 'gold_anchors': anchors,
                          'first_relevant_rank': min(relevant) if relevant else None,
                          **result.model_dump(mode='json')})
        (out / 'search_examples.json').write_text(json.dumps(retrieval, indent=2))
    engine = QAEngine(settings, search=search)
    answers = []
    for label, question, _, _ in cases:
        print('Q&A:', label, flush=True)
        response = engine.ask(question, bid_id=manifest['bid_id'])
        answers.append({'case': label, **response.model_dump(mode='json')})
        (out / 'qa_examples.json').write_text(json.dumps(answers, indent=2))
    extra_questions = specification.get('extra_questions', [['absent_models',
        'Which exact commercial laptop model numbers and manufacturer part numbers are specified?']])
    for label, question in extra_questions:
        print('Q&A:', label, flush=True)
        answers.append({'case': label, **engine.ask(question, bid_id=manifest['bid_id']).model_dump(mode='json')})
    (out / 'qa_examples.json').write_text(json.dumps(answers, indent=2))
    lines = ['# Independent public-RFP Q&A', '', 'Actual pipeline outputs; see review.md for independent assessment.', '']
    for item in answers:
        lines.extend(['## ' + item['question'], '', item['answer'], ''])
        if item['notes']:
            lines.extend(['Evidence limits: ' + item['notes'], ''])
        for source in item['sources']:
            lines.extend([f"Source: {source['file']}, physical page {source['page']}", '',
                          '> ' + source['quote'].replace('\n', '\n> '), ''])
        lines.extend(['Trace: ' + Path(item['trace_file']).name, ''])
    (out / 'qa_examples.md').write_text('\n'.join(lines))
    changes = [name for name, digest in manifest['runtime_before'].items()
               if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    original_changes = [name for name, digest in manifest['main_index_before'].items()
                        if hashlib.sha256((ROOT / 'index' / name).read_bytes()).hexdigest() != digest]
    (out / 'integrity.json').write_text(json.dumps({'runtime_changes': changes, 'original_bid_index_changes': original_changes,
        'note': 'Gold data and this evaluation script are separate from runtime code and prompts.'}, indent=2))
    assert not changes and not original_changes, 'Test isolation or unchanged-runtime check failed'
    print('Saved retrieval, Q&A, and unchanged-runtime/main-index integrity results.', flush=True)


if __name__ == '__main__':
    main()
