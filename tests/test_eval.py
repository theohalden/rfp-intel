"""Check relevance judgments and metrics against misleading retrieval successes."""
from schemas import (ChunkMetadata, SearchHit, Evidence, GoldSource, GoldTarget, EvalQuestion)
from eval.run_eval import score_question, render_results, normalize_evidence


def hit(text: str, file: str = 'source.pdf', page: int = 1, bid: str = 'sample') -> SearchHit:
    return SearchHit(rank=1, text=text, metadata=ChunkMetadata(bid_id=bid, file_name=file,
        doc_type='rfp', page_number=page, chunk_id=text, kind='text'),
        source=Evidence(file=file, page=page, quote=text))


def target(quote: str, file: str = 'source.pdf', page: int = 1) -> GoldTarget:
    return GoldTarget(sources=[GoldSource(bid_id='sample', file=file, page=page, quote=quote)])


def test_right_page_without_supporting_text_is_not_relevant() -> None:
    q = EvalQuestion(id='deadline', query='deadline', targets=[target('New due date October 12')])
    result = score_question(q, [hit('The submission format is electronic.')])
    assert result['recall_at_5'] == result['mrr_at_5'] == 0


def test_multiple_targets_and_rank_sensitive_metrics() -> None:
    q = EvalQuestion(id='compare', query='compare', targets=[target('one year'), target('three years', 'other.pdf')])
    result = score_question(q, [hit('one year'), hit('irrelevant'), hit('three years', 'other.pdf')])
    assert result == {'recall_at_1': .5, 'recall_at_3': 1, 'recall_at_5': 1, 'mrr_at_5': 1}
    result = score_question(q, [hit('irrelevant'), hit('irrelevant too'), hit('one year')])
    assert result['mrr_at_5'] == 1 / 3 and result['recall_at_5'] == .5


def test_equivalent_sources_and_duplicate_hits_count_once() -> None:
    gold = target('one year')
    gold.sources.append(GoldSource(bid_id='sample', file='amendment.pdf', page=2, quote='one-year warranty'))
    q = EvalQuestion(id='warranty', query='warranty', targets=[gold])
    result = score_question(q, [hit('one-year warranty', 'amendment.pdf', 2), hit('one year'), hit('one year')])
    assert result['recall_at_1'] == result['recall_at_5'] == 1


def test_wrong_bid_page_and_results_after_five_do_not_count() -> None:
    q = EvalQuestion(id='scope', query='scope', targets=[target('evidence')])
    hits = [hit('evidence', page=2), hit('evidence', bid='other'),
            hit('noise'), hit('more noise'), hit('still noise'), hit('evidence')]
    assert score_question(q, hits)['recall_at_5'] == 0


def test_markup_normalization_preserves_numbers_and_words() -> None:
    assert normalize_evidence('**RAM**<br>16 GB | DDR5') == 'ram 16 gb ddr5'
    assert normalize_evidence('16 GB') != normalize_evidence('32 GB')


def test_errors_stay_in_report_denominator() -> None:
    rows = []
    for mode in ('vector', 'hybrid', 'hybrid_rerank'):
        for value, status in [(1., 'ok'), (0., 'error')]:
            rows.append({'id': status, 'query': 'test', 'mode': mode, 'status': status,
                         'metrics': {key: value for key in ('recall_at_1', 'recall_at_3', 'recall_at_5', 'mrr_at_5')}})
    text = render_results(rows, {'questions': 2})
    assert '| vector | 0.500 | 0.500 | 0.500 | 0.500 | 1 |' in text
