"""Coverage repairs must see split clauses and reject false missing values."""
import json
from types import SimpleNamespace
import pytest

from agents.graph import ExtractionWorkflow
from agents.guards import guard_draft
from config import Settings
from schemas import CitationSpan, ExtractedField, ExtractionBatch, SearchChunk, SupportReview
from search.retrieve import adjacent_context
from tests.test_agents import BASE, FakeLLM, Tool, evidence


@pytest.mark.parametrize('placeholder', ['null', 'None', 'N/A'])
def test_uncited_missing_placeholders_are_json_null_but_cited_none_is_a_fact(placeholder):
    field = ExtractedField(field='Part_no', value=placeholder, sources=[], confidence=0, notes='')
    result, issues = guard_draft(field, [], 'sample')
    assert result.value is None and not issues
    hit = evidence('Pre-proposal conference: None.')
    cited = ExtractedField(field='Pre Bid Meeting', value='None', sources=[CitationSpan(
        chunk_id=hit.metadata.chunk_id, start_line=1, end_line=1)], confidence=.9, notes='')
    result, issues = guard_draft(cited, [hit], 'sample')
    assert result.value == 'None' and result.sources and not issues


@pytest.mark.parametrize(('text', 'value', 'allowed'), [
    ('Aster Notebook Pro N420 or equivalent; protective case for N420.', 'N420', False),
    ('|Brand|Class|Quantity|\n|Aster|Business laptops|7|', 'Aster', False),
    ('Manufacturer part number: AT-900', 'AT-900', True),
    ('|Category|Model|Description|\n|DEVICE|AA123#X|Aster Notebook Pro N420|', 'AA123#X', True),
])
def test_part_evidence_distinguishes_model_prose_from_part_labels_and_code_cells(text, value, allowed):
    hit = evidence(text)
    draft = ExtractedField(field='Part_no', value=value, sources=[CitationSpan(
        chunk_id=hit.metadata.chunk_id, start_line=1, end_line=len(text.splitlines()))], confidence=.9, notes='')
    result, issues = guard_draft(draft, [hit], 'sample')
    assert (result.value == value and not issues) if allowed else (result.value is None and issues)


def test_adjacent_context_keeps_exact_text_within_page_file_bid_and_amendment():
    before = evidence('If non-federal funds were used for lobbying:')
    seed = evidence('submit the disclosure form.')
    after = evidence('Otherwise the form is not required.')
    for i, hit in enumerate([before, seed, after]):
        hit.metadata.chunk_id = f'clause-{i}'
    other_bid = evidence('Unrelated private bid.', bid='other')
    other_page = evidence('Do not attach this page.')
    other_page.metadata.page_number = 2
    amendment = evidence('Later amendment.', scope=2)
    corpus = [SearchChunk(text=h.text, metadata=h.metadata) for h in
              [before, other_bid, seed, other_page, amendment, after]]
    hits = adjacent_context([seed], corpus, limit=3)
    assert [h.text for h in hits] == [seed.text, before.text, after.text]
    assert all(h.source.quote == h.text and h.source.page == 1 for h in hits)
    assert all(h.metadata.bid_id == 'sample' and h.metadata.addendum_number is None for h in hits)
    assert len(adjacent_context([seed], corpus, limit=2)) == 2


def test_retry_extraction_retains_old_context_without_leaking_amendments(tmp_path):
    settings = Settings(); settings.paths.outputs = tmp_path
    run = ExtractionWorkflow(settings, 'sample', llm=FakeLLM(), tool=Tool())
    old_base = evidence('Original duration clause.')
    old_base.metadata.chunk_id = 'original-duration'
    old_amendment = evidence('Revised duration clause.', scope=2)
    old_amendment.metadata.chunk_id = 'revised-duration'
    scopes, _, _ = run.collect({'evidence': {'Term of Bid': [old_base, old_amendment]}}, ['Term of Bid'], 1)
    assert 'original-duration' in {h.metadata.chunk_id for h in scopes[None]['Term of Bid']}
    assert 'revised-duration' not in {h.metadata.chunk_id for h in scopes[None]['Term of Bid']}
    assert 'revised-duration' in {h.metadata.chunk_id for h in scopes[2]['Term of Bid']}
    assert 'original-duration' not in {h.metadata.chunk_id for h in scopes[2]['Term of Bid']}


def test_false_missing_field_is_reviewed_against_uncited_context_and_repaired(tmp_path):
    clause = 'Initial term: 18 months, with one optional 6-month renewal.'
    class DurationTool(Tool):
        addenda = []
        def search(self, query, field, amendment=None, top_k=3):
            hit = evidence(clause if field == 'Term of Bid' else BASE)
            if field == 'Term of Bid':
                hit.metadata.chunk_id = 'duration-clause'
            return [hit], []
    class Model(FakeLLM):
        attempts = 0
        def with_structured_output(self, schema, **kwargs):
            original = super().with_structured_output(schema, **kwargs)
            def invoke(messages):
                payload = json.loads(messages[-1][1])
                if schema is ExtractionBatch and payload['requested_fields'][0]['name'] == 'Term of Bid':
                    self.attempts += 1
                    found = self.attempts > 1
                    if found:
                        assert payload['previous_validation_feedback']['Term of Bid']
                    passage = payload['evidence'][0]
                    field = ExtractedField(field='Term of Bid', value=clause if found else None,
                        sources=[CitationSpan(chunk_id=passage['chunk_id'], start_line=1, end_line=1)] if found else [],
                        confidence=.9 if found else 0, notes='')
                    return {'parsed': ExtractionBatch(fields=[field]), 'raw': None, 'parsing_error': None}
                if schema is SupportReview and payload.get('field_name') == 'Term of Bid':
                    assert clause in str(payload.get('retrieved_context', payload['sources']))
                    return {'parsed': SupportReview(supported=payload['value'] is not None,
                        reason='Missing initial duration and renewal explicitly present in the retrieved clause.'),
                        'raw': None, 'parsing_error': None}
                return original.invoke(messages)
            return SimpleNamespace(invoke=invoke)
    settings = Settings(); settings.paths.outputs = tmp_path
    model = Model()
    run = ExtractionWorkflow(settings, 'sample', llm=model, tool=DurationTool())
    record = run.run()
    assert record.fields['Term of Bid'].value == clause
    assert model.attempts == 2 and not record.validation.failed
