"""Regression checks for dated evidence, amendment applicability, and readable output."""
from schemas import ChunkMetadata, SearchChunk, SearchHit, Evidence, SearchResponse, SearchFilters
from search.deadlines import dated_deadline_priority, prioritize_deadline_evidence
from search.retrieve import rewrite_query, deduplicate_passages, select_candidates
from search.display import format_search
from langchain_core.documents import Document


def chunk(key: str, text: str, amendment: int | None = None, page: int = 1) -> SearchChunk:
    return SearchChunk(text=text, metadata=ChunkMetadata(bid_id='unseen', file_name=f'{key}.pdf',
        doc_type='addendum' if amendment else 'rfp', addendum_number=amendment,
        page_number=page, chunk_id=key, kind='text'))


def test_explicit_dated_revision_beats_old_keyword_match() -> None:
    old = chunk('arbitrary_a', 'We do not anticipate extending the submission deadline.', 1)
    updated = chunk('renamed_b', 'The new due date will be November 18, 2027 at 11:00 AM UTC.', 4)
    result = prioritize_deadline_evidence('submission deadline', [(old, .99), (updated, .02)])
    assert result[0][0] == updated
    assert dated_deadline_priority(updated) == (2, 4)


def test_newer_unrelated_addendum_does_not_override_deadline() -> None:
    relevant = chunk('first', 'The revised due date is 2027-11-18 at 11:00 UTC.', 2)
    unrelated = chunk('second', 'Warranty coverage begins December 1, 2027.', 8)
    result = prioritize_deadline_evidence('final due date', [(unrelated, .99), (relevant, .1)])
    assert result[0][0] == relevant
    assert dated_deadline_priority(unrelated) == (0, 0)


def test_latest_explicit_deadline_revision_wins() -> None:
    earlier = chunk('one', 'The new due date is May 12, 2028 at noon.', 2)
    later = chunk('two', 'The revised due date is May 19, 2028 at noon.', 3)
    assert prioritize_deadline_evidence('closing date', [(earlier, .9), (later, .1)])[0][0] == later


def test_questions_and_historical_queries_keep_their_own_meaning() -> None:
    question = chunk('q', 'Questions due date: May 12, 2028.')
    assert dated_deadline_priority(question) == (0, 0)
    original = chunk('base', 'Submission deadline: May 1, 2028.')
    later = chunk('revision', 'New due date: May 19, 2028.', 3)
    order = [(original, .9), (later, .1)]
    assert prioritize_deadline_evidence('original submission deadline', order) == order
    assert rewrite_query('deadline for questions') == 'deadline for questions'
    assert rewrite_query('original submission deadline') == 'original submission deadline'
    assert 'final submission due date and time' in rewrite_query('submission deadline')


def test_signature_tail_removed_only_within_same_source_page() -> None:
    full = chunk('same', 'New due date November 18, 2027. Company Name Signature Telephone Date END OF ADDENDUM', 2)
    tail = full.model_copy(update={'text': 'Company Name Signature Telephone Date END OF ADDENDUM\nPage 1 | 1'})
    other_page = tail.model_copy(deep=True)
    other_page.metadata.page_number = 2
    result = deduplicate_passages([(full, .9), (tail, .1), (other_page, .1)])
    assert len(result) == 2


def test_fusion_keeps_a_leading_hit_from_each_branch() -> None:
    def doc(key):
        return Document(page_content=key, metadata={'chunk_id': key})
    rankings = [[doc('shared'), doc('other')], [doc('rare'), doc('shared')]]
    ids = {d.metadata['chunk_id'] for d in select_candidates(rankings, 2)}
    assert ids == {'shared', 'rare'}


def test_default_output_shows_source_preview_without_internal_scores() -> None:
    c = chunk('source', 'The new due date is November 18, 2027 at 11:00 AM UTC.', 4)
    hit = SearchHit(rank=1, text=c.text, metadata=c.metadata, source=Evidence(
        file=c.metadata.file_name, page=1, quote=c.text), rerank_score=.0000003)
    response = SearchResponse(query='submission deadline', expanded_queries=['submission deadline'],
        mode='hybrid_rerank', filters=SearchFilters(), hits=[hit], latency_ms=100)
    text = format_search(response)
    assert 'November 18, 2027' in text and 'Source: source.pdf' in text
    assert 'Relevance score' not in text and '--full-text' in text
    assert 'not factual confidence' in format_search(response, details=True)
