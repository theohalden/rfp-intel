"""Regression checks for field coverage and international partial deadlines."""
from threading import RLock
from types import SimpleNamespace
import pytest
from agents.guards import guard_draft, format_issues, UNKNOWN_ZONE_NOTE
from agents.tools import EvidenceTool
from agents.trace import TraceWriter
from schemas import CitationSpan, ExtractedField, SearchResponse
from tests.test_agents import evidence


@pytest.mark.parametrize('value', [
    '18 August 2028 at 09hr30', '18th August 2028 at 09h30',
    '18 Aug 2028 at 09:30', 'August 18, 2028 at 9:30 AM',
    '18 August 2028 at 9 p.m.',
])
def test_partial_deadline_preserves_supported_date_and_warns_unknown_zone(value):
    hit = evidence(value)
    draft = ExtractedField(field='Due Date', value=value, sources=[CitationSpan(
        chunk_id=hit.metadata.chunk_id, start_line=1, end_line=1)], confidence=.95, notes='')
    result, errors = guard_draft(draft, [hit], 'sample')
    assert not errors and result.value == value
    assert UNKNOWN_ZONE_NOTE in result.notes and result.confidence <= .7
    assert not format_issues('Due Date', result)


def test_unknown_zone_never_licenses_inferred_zone_from_location():
    hit = evidence('The closing deadline is 18 August 2028 at 09hr30. Office in Pretoria.')
    draft = ExtractedField(field='Due Date', value='18 August 2028 at 09:30 SAST',
        sources=[CitationSpan(chunk_id=hit.metadata.chunk_id, start_line=1, end_line=1)], confidence=.9, notes='')
    result, errors = guard_draft(draft, [hit], 'sample')
    assert errors and result.value is None


def test_explicit_zone_written_in_words_supports_normalized_abbreviation():
    hit = evidence('Closing: August 18, 2028 at 9:30 AM Eastern Daylight Time.')
    draft = ExtractedField(field='Due Date', value='August 18, 2028 at 9:30 AM EDT',
        sources=[CitationSpan(chunk_id=hit.metadata.chunk_id, start_line=1, end_line=1)], confidence=.9, notes='')
    result, errors = guard_draft(draft, [hit], 'sample')
    assert not errors and not format_issues('Due Date', result)
    assert UNKNOWN_ZONE_NOTE not in result.notes


@pytest.mark.parametrize(('field', 'anchor'), [('company_name', 'BACKGROUND'), ('Product', 'Quantity'), ('Model_no', 'Model'), ('Part_no', 'Model'), ('Product Specification', 'Model'), ('Due Date', 'submission deadline')])
def test_coverage_queries_recover_issuer_or_quantity_when_planner_misses(tmp_path, field, anchor):
    calls = []
    hit = evidence('River Supply Authority; product quantity 12.')
    def search(query, filters, top_k, include_neighbors=False):
        calls.append((query, filters))
        return SearchResponse(query=query, expanded_queries=[query], filters=filters,
            mode='hybrid_rerank', hits=[hit] if query == anchor else [], latency_ms=1)
    tool = EvidenceTool.__new__(EvidenceTool)
    tool.engine = SimpleNamespace(search=search)
    tool.bid_id, tool.lock, tool.trace = 'sample', RLock(), TraceWriter(tmp_path, 'sample')
    hits, errors = tool.search('a poor planner query', field)
    assert hits == [hit] and not errors
    if field == 'Due Date':
        assert next(scope for query, scope in calls if query == anchor).page_number == 1
    assert all(scope.exclude_addenda and scope.bid_id == 'sample' for _, scope in calls)
    assert next(scope for query, scope in calls if query == anchor).doc_type == ('rfp' if field in {'company_name', 'Product', 'Due Date'} else None)


def test_failed_supplement_keeps_successful_evidence_and_never_retries_in_except(tmp_path):
    calls = []
    hit = evidence()
    def search(query, filters, top_k, include_neighbors=False):
        calls.append(query)
        if query != 'working primary':
            raise TimeoutError('unavailable')
        return SearchResponse(query=query, expanded_queries=[query], filters=filters,
            mode='hybrid_rerank', hits=[hit], latency_ms=1)
    tool = EvidenceTool.__new__(EvidenceTool)
    tool.engine = SimpleNamespace(search=search)
    tool.bid_id, tool.lock, tool.trace = 'sample', RLock(), TraceWriter(tmp_path, 'sample')
    hits, errors = tool.search('working primary', 'Contract or Cooperative to use')
    assert hits == [hit] and not errors
    assert len(calls) == 3


def test_equipment_table_evidence_is_reused_without_mixing_amendment_scopes(tmp_path):
    import json
    from agents.graph import ExtractionWorkflow
    from config import Settings
    from schemas import ExtractionBatch
    from tests.test_agents import FakeLLM, Tool
    names = ['Model_no', 'Part_no', 'Product', 'Product Specification']
    class RelatedTool(Tool):
        def search(self, query, field, amendment=None, top_k=3):
            hit = evidence(text=field, scope=amendment)
            hit.metadata.chunk_id = f'{field}:{amendment}'
            return [hit], []
    class Inspect(FakeLLM):
        def with_structured_output(self, schema, **kwargs):
            underlying = super().with_structured_output(schema, **kwargs)
            def invoke(messages):
                if schema is ExtractionBatch:
                    payload = json.loads(messages[-1][1])
                    scope = None if payload['scope'] == 'base' else int(payload['scope'].split()[-1])
                    assert len(payload['requested_fields']) == 1
                    assert {p['chunk_id'] for p in payload['evidence']} == {f'{n}:{scope}' for n in names}
                return underlying.invoke(messages)
            return SimpleNamespace(invoke=invoke)
    settings = Settings(); settings.paths.outputs = tmp_path
    run = ExtractionWorkflow(settings, 'sample', llm=Inspect(), tool=RelatedTool())
    run.extract_fields({}, names, 'product_specs')
