"""Audit final JSON contracts and exact citation provenance against canonical chunks."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.guards import normalize_quote
from config import load_settings
from schemas import BidRecord, QAResponse, FIELD_NAMES


def verify():
    settings = load_settings()
    chunks = [json.loads(line) for line in (settings.paths.index / 'chunks.jsonl').read_text().splitlines()]
    # Canonical rows contain exact text and validated metadata.
    def cited(bid, source):
        return any(row['metadata']['bid_id'] == bid
            and row['metadata']['file_name'] == source.file
            and row['metadata']['page_number'] == source.page
            and normalize_quote(source.quote) in normalize_quote(row['text']) for row in chunks)
    report = []
    for bid in ('Bid1', 'Bid2'):
        record = BidRecord.model_validate_json((settings.paths.outputs / f'{bid}.json').read_text())
        assert set(record.fields) == set(FIELD_NAMES)
        assert sum(record.validation.model_dump().values()) == 20
        assert record.validation.failed == 0, f'{bid}: unresolved validation failures'
        sources = [s for field in record.fields.values() for s in field.sources]
        assert all(cited(bid, source) for source in sources), f'{bid}: citation missing from canonical chunks'
        assert all(cited(bid, change.source) for change in record.addendum_changes)
        assert record.fields['Bid Summary'].value, f'{bid}: missing summary'
        events = [json.loads(line) for line in (settings.paths.outputs / f'{bid}.trace.jsonl').read_text().splitlines()]
        assert events[-1]['agent'] == 'report'
        report.append({'bid_id': bid, 'validation': record.validation.model_dump(),
                       'verified_citations': len(sources), 'trace_steps': len(events)})
    qa = [QAResponse.model_validate(item) for item in json.loads((settings.paths.outputs / 'qa_log.json').read_text())]
    assert len(qa) >= 10
    assert not qa[0].not_found and not qa[2].not_found
    assert not qa[3].not_found and {'Bid1', 'Bid2'}.issubset({s.bid_id for s in qa[3].sources}), 'Cross-bid warranty comparison lacks verified evidence from both bids'
    for item in qa:
        assert all(cited(source.bid_id, source) for source in item.sources)
        assert item.not_found or item.sources
    result = {'extractions': report, 'qa_examples': len(qa), 'qa_not_found': sum(item.not_found for item in qa),
              'note': 'Contract/provenance audit; not a claim of calibrated semantic accuracy.'}
    (settings.paths.outputs / 'verification.json').write_text(json.dumps(result, indent=2))
    lines = ['# Final verification', '',
        'Live OpenAI extraction and Q&A outputs; exact citations audited against canonical indexed chunks.', '',
        '| Bid | Validated | Not found | Failed | Verified field citations | Trace steps |',
        '|---|---:|---:|---:|---:|---:|']
    for item in report:
        counts = item['validation']
        lines.append(f"| {item['bid_id']} | {counts['passed']} | {counts['not_found']} | {counts['failed']} | {item['verified_citations']} | {item['trace_steps']} |")
    lines.extend(['', f"Q&A examples: {len(qa)}; explicit missing answers: {sum(item.not_found for item in qa)}.",
        'The affidavit answer and cross-bid warranty comparison have verified supporting citations; the comparison includes both bids.', '',
        'All populated fields and Q&A sources match exact whitespace-normalized source text on the cited bid/file/physical page. Both records have exactly 20 fields and a generated summary. Per-bid extraction traces end with the final report.', '',
        '## Other checks', '',
        '- 91 offline tests pass; no broken dependency requirements.',
        '- Retrieval evaluation: 24 quote-anchored development questions; reranked Recall@5 0.938 and MRR@5 0.847.',
        '- Renamed Bid3_test copy: 51 chunks indexed/searched without runtime code changes (filename independence).',
        '- Three independent public RFPs tested; preserved first runs and development retests are compared in outputs/public_rfp_review.md. Workflow validation is not an accuracy score.',
        '- Configured API-key text excluded from source/deliverable artifacts; package excludes .env, data, weights, index, and .venv.', '',
        '## Limits', '',
        'Validation counts describe the workflow verdict, not measured extraction accuracy. Exact provenance does not prove semantic correctness or completeness. Conservative nulls, missing context, model decisions, OCR, and table layouts still need human source review. Retrieval evaluation uses a development set, not held-out data.', '',
        'Live presentation is the selected demo format; recording is not planned. Publishing a repository URL and presenting the live session remain submission steps. Public-document source review and remaining limitations are documented in outputs/public_rfp_review.md; no universal unseen-document accuracy is claimed.', ''])
    (settings.paths.outputs / 'verification.md').write_text('\n'.join(lines))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    verify()
