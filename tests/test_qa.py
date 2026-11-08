"""Q&A provenance, request scoping, and safe fallback without paid API calls."""
import json
import pytest
from types import SimpleNamespace

from agents.qa import QAEngine, qa_guard, resolved_qa_claims
from config import Settings
from schemas import CitationSpan, SupportReview, QADraft, QASelection
from tests.test_agents import evidence


def draft(chunk='base.pdf', start=1, end=1):
    return QADraft(claims=[QASelection(text='Device purchase.', sources=[
        CitationSpan(chunk_id=chunk, start_line=start, end_line=end)])], not_found=False, notes='')


def test_qa_span_copies_exact_quote_and_bid():
    hit = evidence('**Device purchase.**\nSecond line.')
    claims, errors = resolved_qa_claims(draft(), [hit])
    assert not errors
    assert claims[0].sources[0].quote == '**Device purchase.**'
    assert claims[0].sources[0].bid_id == 'sample'


def test_qa_rejects_invented_or_out_of_range_sources():
    assert qa_guard(draft('invented'), [evidence()])
    assert qa_guard(draft(end=99), [evidence()])
    assert qa_guard(QADraft(claims=[], not_found=False, notes=''), [evidence()])


def test_not_found_does_not_create_citations():
    claims, errors = resolved_qa_claims(QADraft(claims=[], not_found=True, notes='Missing'), [])
    assert claims == [] and errors == []


class Model:
    def with_structured_output(self, schema, **kwargs):
        assert schema is SupportReview
        return SimpleNamespace(invoke=lambda messages: {'parsed': SupportReview(supported=True, reason='Assertions supported by quotes'), 'raw': None})


class Search:
    def __init__(self, empty=False):
        self.filters = []
        self.empty = empty

    def search(self, query, filters, top_k):
        self.filters.append(filters)
        hits = [] if self.empty else [evidence(bid=filters.bid_id or 'sample')]
        return SimpleNamespace(hits=hits)


def engine(tmp_path, search):
    settings = Settings()
    settings.paths.index = tmp_path / 'index'
    settings.paths.index.mkdir()
    (settings.paths.index / 'manifest.json').write_text(json.dumps({'bids': {'sample': {}, 'other': {}}}))
    settings.paths.outputs = tmp_path / 'outputs'
    return QAEngine(settings, llm=Model(), search=search)


def test_qa_uses_seeded_tool_and_explicit_filter(tmp_path, monkeypatch):
    search = Search()
    run = engine(tmp_path, search)
    def agent(*args, **kwargs):
        def invoke(state, config):
            assert state['messages'][-1].type == 'tool'
            # A tool request cannot widen an explicit user filter.
            denied = args[1][0].invoke({'query': 'anything', 'search_bid_id': 'other'})
            assert 'outside the user filter' in denied
            return {'structured_response': draft()}
        return SimpleNamespace(invoke=invoke)
    monkeypatch.setattr('agents.qa.create_react_agent', agent)
    result = run.ask('What is the title?', bid_id='sample', doc_type='rfp')
    assert not result.not_found and result.sources[0].bid_id == 'sample'
    assert len(search.filters) == 1 and search.filters[0].doc_type == 'rfp'
    assert '[sample: base.pdf, p. 1]' in result.answer


def test_cross_bid_seed_searches_each_bid(tmp_path, monkeypatch):
    search = Search()
    run = engine(tmp_path, search)
    monkeypatch.setattr('agents.qa.create_react_agent', lambda *a, **kw: SimpleNamespace(
        invoke=lambda *a, **kw: {'structured_response': QADraft(claims=[], not_found=True, notes='Partial')}))
    result = run.ask('Compare both bids')
    assert {f.bid_id for f in search.filters} == {'sample', 'other'}
    assert result.not_found


def test_no_evidence_returns_safe_missing_answer(tmp_path, monkeypatch):
    run = engine(tmp_path, Search(empty=True))
    monkeypatch.setattr('agents.qa.create_react_agent', lambda *a, **k: (_ for _ in ()).throw(AssertionError('No model call expected')))
    result = run.ask('Unknown fact', bid_id='sample')
    assert result.answer == 'Not found in documents' and not result.sources


def test_qa_rejects_invented_time_zone_equivalence():
    value = draft()
    value.claims[0].text = '2 PM CST corresponds to the same moment as 3 PM EDT.'
    assert any('time-zone' in issue for issue in qa_guard(value, [evidence()]))


def test_partial_answer_keeps_supported_claims_even_if_model_sets_missing_flag():
    value = draft()
    value.not_found = True
    claims, errors = resolved_qa_claims(value, [evidence()])
    assert len(claims) == 1 and not errors


@pytest.mark.parametrize('missing_other', [False, True])
def test_partial_comparison_keeps_only_verified_claims_from_each_bid(tmp_path, monkeypatch, missing_other):
    class MultiSearch(Search):
        def search(self, query, filters, top_k):
            hit = evidence(bid=filters.bid_id)
            hit.metadata.chunk_id = filters.bid_id
            return SimpleNamespace(hits=[hit])
    run = engine(tmp_path, MultiSearch())
    run.settings.agents.max_retries = 0
    claims = [QASelection(text=text, sources=[CitationSpan(chunk_id=bid, start_line=1, end_line=1)])
        for bid, text in [('sample', 'Supported sample fact'), ('other', 'Supported other fact'),
                          ('sample', 'Unsupported extra fact')]]
    monkeypatch.setattr('agents.qa.create_react_agent', lambda *a, **k: SimpleNamespace(
        invoke=lambda *a, **k: {'structured_response': QADraft(claims=claims, not_found=False, notes='')}))
    class Critic:
        def with_structured_output(self, schema, **kwargs):
            def invoke(messages):
                payload = json.loads(messages[-1][1])
                assert 'question' not in payload  # Atomic review must not demand a whole comparison per claim.
                text = payload['claims'][0]['text']
                return {'parsed': SupportReview(supported=not text.startswith('Unsupported') and not (missing_other and text.startswith('Supported other')), reason='Source check'), 'raw': None}
            return SimpleNamespace(invoke=invoke)
    run.llm = Critic()
    result = run.ask('Compare both bids')
    if missing_other:
        assert result.not_found and not result.claims
        return
    assert not result.not_found and len(result.claims) == 2
    assert {s.bid_id for s in result.sources} == {'sample', 'other'}
    assert 'Unsupported' not in result.answer
