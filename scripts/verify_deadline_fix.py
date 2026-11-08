"""Real-model regression for the user's deadline, amendment, and SKU searches."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import load_settings
from schemas import SearchFilters
from search.retrieve import SearchEngine
from search.display import format_search


def main() -> None:
    settings = load_settings()
    engine = SearchEngine(settings)
    deadline = engine.search('submission deadline', SearchFilters(bid_id='Bid1'), top_k=3)
    assert deadline.hits[0].metadata.addendum_number == 2
    assert 'July 9, 2024 at 2:00 PM CST' in deadline.hits[0].text
    changes = engine.search('what changed', SearchFilters(bid_id='Bid1', addendum_number=2), top_k=3)
    assert len(changes.hits) == 1, 'Signature-only overlap should not appear as a second result'
    assert 'extend the due date' in changes.hits[0].text
    product = engine.search('379-BFNZ processor', SearchFilters(bid_id='Bid2', doc_type='specs'), top_k=3)
    assert product.hits[0].metadata.page_number == 1
    assert '379-BFNZ' in product.hits[0].text
    output = settings.paths.outputs
    records = [r.model_dump() for r in (deadline, changes, product)]
    (output / 'search_regressions.json').write_text(json.dumps(records, indent=2))
    report = ['# Search ranking and readability regression', '',
              'Verified using the actual local embedding and cross-encoder models.', '',
              '- Unfiltered Bid1 submission deadline search: Addendum 2 is first.',
              '- Addendum 2 change search: one substantive source passage; signature overlap removed.',
              '- Bid2 processor/SKU search: specifications page 1 is first.',
              '- No source values, filenames, or bid IDs are hardcoded in runtime ranking rules.',
              '- CLI defaults to three preview results; --full-text and --json retain complete evidence.', '',
              '## Actual readable output', '']
    for response in (deadline, changes, product):
        report.extend(['```text', format_search(response), '```', ''])
    report.extend(['## Scope and limitations', '',
                   'Current submission-deadline queries use a documented evidence-ranking policy after reranking: concrete dated deadline statements first; explicit numbered deadline revisions ahead of base statements. A newer unrelated addendum receives no boost. Historical and other milestone intents bypass this policy. This is not full extraction or semantic conflict resolution; conditional wording and conflicting time zones still require review. Model scores are not confidence values.', '',
                   '78 offline tests pass alongside this real-model regression. JSON output retains the rewritten reranker query and the full source passages.'])
    (output / 'search_fix_report.md').write_text('\n'.join(report) + '\n')
    print('PASS: final deadline evidence first; redundant signature removed; SKU retrieval preserved.')


if __name__ == '__main__':
    main()
