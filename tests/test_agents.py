"""Run the actual LangGraph offline to verify evidence guards and bounded feedback."""
from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

from config import Settings
from agents.graph import ExtractionWorkflow
from agents.guards import guard_draft, quote_verified, format_issues
from agents.trace import TraceWriter
from schemas import (AmendmentDecision, AmendmentReview, SupportReview, Evidence,
    ExtractedField, ExtractionBatch, ExtractionPlan, FieldResult, SearchHit, CitationSpan,
    ChunkMetadata, SummaryDraft, SummarySentence)

BASE = 'Bid RFP-100. Title: Device purchase. Issuing agency: Example Schools. Due date July 10, 2028 at 14:00 UTC.'
EARLY = 'Due date remains July 10, 2028 at 14:00 UTC.'
LATE = 'The new due date is July 17, 2028 at 14:00 UTC.'


def evidence(text=BASE, bid='sample', scope=None):
    file = 'base.pdf' if scope is None else f'amendment_{scope}.pdf'
    return SearchHit(rank=1, text=text, metadata=ChunkMetadata(bid_id=bid, file_name=file,
        page_number=1, doc_type='rfp' if scope is None else 'addendum', addendum_number=scope,
        chunk_id=file, kind='text'), source=Evidence(file=file, page=1, quote=text))


class Tool:
    addenda = [1, 2]
    def search(self, query, field, amendment=None, top_k=3):
        return [evidence(BASE if amendment is None else EARLY if amendment == 1 else LATE, scope=amendment)], []


class FakeLLM:
    """Provide deterministic typed messages, without replacing the actual graph."""
    def __init__(self, bad_quote_once=False, fail=False, bad_summary=False):
        self.title_attempts = 0
        self.bad_quote_once = bad_quote_once
        self.fail = fail
        self.bad_summary = bad_summary

    def with_structured_output(self, schema, **kwargs):
        def invoke(messages):
            payload = json.loads(messages[-1][1])
            if schema is ExtractionPlan:
                parsed = ExtractionPlan(requests=[])
            elif schema is ExtractionBatch:
                if self.fail:
                    raise TimeoutError('private-unit-secret')
                scope = payload['scope']
                base = scope == 'base'
                passage = payload['evidence'][0]
                values = {'Title': 'Device purchase', 'Bid Number': 'RFP-100',
                          'company_name': 'Example Schools', 'Due Date': '2028-07-10 14:00 UTC'}
                fields = []
                for request in payload['requested_fields']:
                    name = request['name']
                    value = values.get(name) if base else ('2028-07-17 14:00 UTC' if scope.endswith('2') else '2028-07-10 14:00 UTC') if name == 'Due Date' else None
                    sources = [CitationSpan(chunk_id=passage['chunk_id'], start_line=1, end_line=len(passage['numbered_lines']))] if value else []
                    if name == 'Title' and base:
                        self.title_attempts += 1
                        if self.bad_quote_once and self.title_attempts == 1:
                            sources = [CitationSpan(chunk_id='invented-chunk', start_line=1, end_line=1)]
                    fields.append(ExtractedField(field=name, value=value, sources=sources,
                                                 confidence=.9 if value else 0, notes=''))
                parsed = ExtractionBatch(fields=fields)
            elif schema is AmendmentReview:
                parsed = AmendmentReview(decisions=[AmendmentDecision(field=name,
                    applies=payload['addendum_number'] == 2, operation='replace' if payload['addendum_number'] == 2 else 'unchanged',
                    reason='Explicit new due date' if payload['addendum_number'] == 2 else 'Unchanged')
                    for name in payload['amendment_candidates']])
            elif schema is SupportReview:
                parsed = SupportReview(supported=True, reason='Assertions supported by quotes')
            elif schema is SummaryDraft:
                parsed = SummaryDraft(sentences=[SummarySentence(text=text, supporting_fields=[field])
                    for text, field in [('This bid purchases devices.', 'Title'), ('The identifier is RFP-100.', 'Bid Number'),
                                        ('The issuer is Example Schools.', 'company_name' if not self.bad_summary else 'invented')]])
            else:
                raise AssertionError(schema)
            return {'parsed': parsed, 'raw': SimpleNamespace(usage_metadata={'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2}), 'parsing_error': None}
        return SimpleNamespace(invoke=invoke)


def workflow(tmp_path, llm):
    settings = Settings()
    settings.paths.outputs = tmp_path
    return ExtractionWorkflow(settings, 'sample', llm=llm, tool=Tool())


def test_full_graph_parallel_join_addenda_and_final_record(tmp_path):
    run = workflow(tmp_path, FakeLLM())
    record = run.run()
    assert len(record.fields) == 20
    assert record.fields['Due Date'].value == '2028-07-17 14:00 UTC'
    change = next(c for c in record.addendum_changes if c.field == 'Due Date')
    assert change.old == '2028-07-10 14:00 UTC' and change.source.file == 'amendment_2.pdf'
    assert record.fields['Bid Summary'].value is not None
    assert record.validation.passed + record.validation.failed + record.validation.not_found == 20
    events = [json.loads(line) for line in (tmp_path / 'sample.trace.jsonl').read_text().splitlines()]
    assert events[-1]['agent'] == 'report'
    assert {'identity', 'dates_logistics', 'commercial_legal', 'product_specs'} <= {e['agent'] for e in events}
    assert not record.validation.failed


def test_citation_rejection_triggers_real_graph_feedback(tmp_path):
    model = FakeLLM(bad_quote_once=True)
    record = workflow(tmp_path, model).run()
    assert record.fields['Title'].value == 'Device purchase'
    assert model.title_attempts == 2
    events = [json.loads(line) for line in (tmp_path / 'sample.trace.jsonl').read_text().splitlines()]
    assert events[-1]['input']['retries']['Title'] == 1


def test_model_timeouts_finish_without_inventing_fields_and_cap_retries(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'private-unit-secret')
    record = workflow(tmp_path, FakeLLM(fail=True)).run()
    assert all(f.value is None for f in record.fields.values())
    assert record.validation.failed == 19
    trace = (tmp_path / 'sample.trace.jsonl').read_text()
    assert 'private-unit-secret' not in trace
    events = [json.loads(line) for line in trace.splitlines()]
    assert set(events[-1]['input']['retries'].values()) == {2}


def test_quote_guard_requires_correct_bid_file_page_and_verbatim_text():
    hit = evidence('**Value** is stated.\nNext line.')
    source = Evidence(file='base.pdf', page=1, quote='**Value** is stated. Next line.')
    assert quote_verified(source, [hit], 'sample')
    assert not quote_verified(source, [hit], 'other')
    assert not quote_verified(source.model_copy(update={'page': 2}), [hit], 'sample')
    assert not quote_verified(source.model_copy(update={'quote': 'Value is stated.'}), [hit], 'sample')
    draft = ExtractedField(field='Title', value='Invented', sources=[], confidence=1, notes='')
    result, issues = guard_draft(draft, [hit], 'sample')
    assert result.value is None and issues


def test_clock_am_pm_is_not_a_timezone():
    result = FieldResult(value='July 10, 2028 at 2:00 PM', sources=[Evidence(file='x.pdf',page=1,quote='date')])
    assert 'Submission deadline timezone is missing' in format_issues('Due Date', result)


def test_summary_cannot_reference_an_unvalidated_field(tmp_path):
    record = workflow(tmp_path, FakeLLM(bad_summary=True)).run()
    assert record.fields['Bid Summary'].value is None
    assert record.validation.failed == 1


def test_trace_redacts_known_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','private-unit-secret')
    trace = TraceWriter(tmp_path,'sample')
    trace.event('test', {'accidental':'private-unit-secret'}, {})
    text = trace.path.read_text()
    assert 'private-unit-secret' not in text and '[REDACTED]' in text


def test_retry_retains_previous_evidence_used_by_other_group_fields(tmp_path):
    run = workflow(tmp_path, FakeLLM())
    old = evidence('Previously retrieved title supporting a separate field.')
    old.metadata.chunk_id = 'older-title-chunk'
    result = run.extract_fields({'evidence': {'Title': [old]}}, ['Title'], 're_extract', 1)
    assert 'older-title-chunk' in {h.metadata.chunk_id for h in result['evidence']['Title']}


def test_clarification_preserves_previously_supported_submission_methods(tmp_path):
    from schemas import FieldCandidates, AmendmentCandidate
    run = workflow(tmp_path, FakeLLM())
    run.tool.addenda = [1]
    run.invoke_llm = lambda *a, **kw: AmendmentReview(decisions=[AmendmentDecision(
        field='Bid Submission Type', applies=True, operation='clarify', reason='Electronic clarification')])
    source = Evidence(file='base.pdf', page=1, quote='Electronic or sealed paper.')
    base = FieldResult(value='Electronic or sealed paper.', sources=[source], confidence=.9)
    addition = FieldResult(value='Electronic responses need no PDF.', sources=[source], confidence=.9)
    state = {'drafts': {'Bid Submission Type': FieldCandidates(base=base, amendments=[
        AmendmentCandidate(number=1, result=addition)])}}
    result = run.reconcile_fields(state, ['Bid Submission Type'])
    assert 'sealed paper' in result['fields']['Bid Submission Type'].value
    assert 'need no PDF' in result['fields']['Bid Submission Type'].value


def test_original_deadline_retrieval_includes_main_rfp_scope(tmp_path):
    from agents.tools import EvidenceTool
    from threading import RLock
    tool = EvidenceTool.__new__(EvidenceTool)
    tool.bid_id = 'sample'
    tool.lock = RLock()
    tool.trace = TraceWriter(tmp_path, 'sample')
    calls = []
    # Use a typed response so trace serialization exercises the real contract.
    from schemas import SearchResponse
    def typed_search(query, filters, top_k, include_neighbors=False):
        calls.append(filters)
        return SearchResponse(query=query, expanded_queries=[query], hits=[evidence()],
            mode='hybrid_rerank', filters=filters, latency_ms=1, warnings=[])
    tool.engine = SimpleNamespace(search=typed_search)
    hits, errors = tool.search('deadline', 'Due Date')
    assert hits and not errors
    assert calls[0].exclude_addenda and calls[1].doc_type == 'rfp'
    calls.clear()
    tool.search('deadline', 'Due Date', amendment=2)
    assert len(calls) == 1 and calls[0].addendum_number == 2


def test_unchanged_fields_reuse_semantic_verdict_but_changed_values_are_reviewed(tmp_path):
    from schemas import FieldCandidates
    run = workflow(tmp_path, FakeLLM())
    calls = []
    def review(*args, **kwargs):
        calls.append(args[0])
        return SupportReview(supported=True, reason='Supported')
    run.invoke_llm = review
    hit = evidence()
    field = FieldResult(value='Device purchase', sources=[hit.source], confidence=.9)
    state = {'fields': {'Title': field}, 'drafts': {'Title': FieldCandidates(base=field)},
             'evidence': {'Title': [hit]}}
    run.validate(state)
    run.validate(state)
    assert len(calls) == 1
    state['fields']['Title'] = field.model_copy(update={'value': 'Changed title'})
    run.validate(state)
    assert len(calls) == 2


def test_each_field_gets_only_its_own_retrieved_scope(tmp_path):
    class ScopedTool(Tool):
        def search(self, query, field, amendment=None, top_k=3):
            hit = evidence(text=field, scope=amendment)
            hit.metadata.chunk_id = field
            return [hit], []
    class InspectModel(FakeLLM):
        def with_structured_output(self, schema, **kwargs):
            underlying = super().with_structured_output(schema, **kwargs)
            def invoke(messages):
                if schema is ExtractionBatch:
                    payload = json.loads(messages[-1][1])
                    assert len(payload['requested_fields']) == 1
                    name = payload['requested_fields'][0]['name']
                    assert all(p['chunk_id'] == name for p in payload['evidence'])
                return underlying.invoke(messages)
            return SimpleNamespace(invoke=invoke)
    run = workflow(tmp_path, InspectModel())
    run.tool = ScopedTool()
    result = run.extract_fields({}, ['Title', 'Bid Number'], 'identity')
    assert set(result['drafts']) == {'Title', 'Bid Number'}


def test_unsupported_part_is_narrowed_not_nulled_and_rechecked(tmp_path):
    """A partly unsupported value keeps its supported subset, and only after a fresh support check."""
    from schemas import NarrowedField
    settings = Settings(); settings.paths.outputs = tmp_path
    llm = FakeLLM()
    inner = llm.with_structured_output
    def wrapped(schema, **kwargs):
        if schema is NarrowedField:
            return SimpleNamespace(invoke=lambda m: {'parsed': NarrowedField(value='Device purchase', notes=''), 'raw': SimpleNamespace(usage_metadata={'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2}), 'parsing_error': None})
        return inner(schema, **kwargs)
    llm.with_structured_output = wrapped
    flow = ExtractionWorkflow(settings, 'sample', llm=llm, tool=Tool())
    hit = evidence()
    source = Evidence(file=hit.source.file, page=1, quote=BASE)
    field = FieldResult(value='Device purchase and unsupported delivery claim', sources=[source], confidence=.9)
    narrowed = flow.narrow('Title', field, 'delivery claim unsupported', [hit])
    assert narrowed.value == 'Device purchase' and narrowed.confidence <= .7
    assert 'Narrowed after validation' in narrowed.notes and narrowed.sources == [source]
    assert flow.narrow('Title', field.model_copy(update={'sources': [Evidence(file='x.pdf', page=1, quote='invented')]}), 'r', [hit]) is None


def test_evidence_sent_to_model_respects_budget_but_keeps_first_passage(tmp_path):
    flow = workflow(tmp_path, FakeLLM())
    flow.settings.agents.max_evidence_chars = 150
    hits = [evidence('a' * 100), evidence('b' * 100), evidence('c' * 40)]
    kept = flow.within_budget(hits)
    assert [h.text[0] for h in kept] == ['a', 'c']


def test_narrowing_that_removes_nothing_is_not_accepted(tmp_path):
    """An omission failure cannot be 'fixed' by returning the same incomplete value."""
    from schemas import NarrowedField
    llm = FakeLLM(); inner = llm.with_structured_output
    llm.with_structured_output = lambda schema, **k: SimpleNamespace(invoke=lambda m: {'parsed': NarrowedField(value='same', notes=''),
        'raw': SimpleNamespace(usage_metadata={'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2}), 'parsing_error': None}) if schema is NarrowedField else inner(schema, **k)
    settings = Settings(); settings.paths.outputs = tmp_path
    flow = ExtractionWorkflow(settings, 'sample', llm=llm, tool=Tool())
    hit = evidence()
    field = FieldResult(value='same', sources=[Evidence(file='base.pdf', page=1, quote=BASE)], confidence=.9)
    assert flow.narrow('Title', field, 'omits a fact', [hit]) is None
