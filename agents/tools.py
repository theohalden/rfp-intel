"""The sole evidence access path for extraction agents: cited search."""
from __future__ import annotations

from threading import RLock
from time import perf_counter
from config import Settings
from schemas import SearchFilters, SearchHit
from search.index import index_lock, read_chunks
from search.retrieve import SearchEngine
from agents.trace import TraceWriter

# General section/field questions supplement the LLM planner's query. These contain
# no issuer names, document-specific values, or page assumptions.
COVERAGE_QUERIES = {
    'Due Date': ['Solicitation Due'],
    'company_name': ['Who is the issuing organization or authority? Background introduction.', 'BACKGROUND'],
    'Title': ['solicitation title request for proposals'],
    'Term of Bid': ['LEASE TERMS', 'Term'],
    'Pre Bid Meeting': ['Pre-Proposal Conference'],
    'Installation': ['installed prior to shipment'],
    'Payment Terms': ['LEASE TERMS'],
    'Product': ['What products and quantities are being purchased?', 'Quantity'],
    # Short table labels retrieve equipment rows that prose-heavy questions miss.
    'Model_no': ['Model'],
    'Part_no': ['Model'],
    'Product Specification': ['Model', 'RAM'],
    'Bid Submission Type': ['Can bids be submitted electronically or must they be physically delivered?', 'submission copies sealed'],
    'MFG for Registration': ['Brand Name'],
    'contact_info': ['clarification procurement contact email telephone', 'questions regarding', 'Purchasing Department'],
    'Any Additional Documentation Required': ['required supporting documents financial capability bank reference', 'Vendor Packet', 'required forms', 'insurance documentation'],
    'Contract or Cooperative to use': [
        'eligible master contractors under master contract contract number special instructions',
        'cooperative interlocal agreement'],
}


class EvidenceTool:
    """Reuse local models; serialize CPU retrieval while LLM branches run in parallel."""

    def __init__(self, settings: Settings, bid_id: str, trace: TraceWriter):
        self.engine = SearchEngine(settings)
        self.bid_id = bid_id
        self.trace = trace
        self.lock = RLock()
        with index_lock(settings.paths.index):
            self.addenda = sorted({c.metadata.addendum_number for c in read_chunks(settings.paths.index)
                if c.metadata.bid_id == bid_id and c.metadata.doc_type == 'addendum'
                and c.metadata.addendum_number is not None})

    def search(self, query: str, field: str, amendment: int | None = None,
               top_k: int = 3) -> tuple[list[SearchHit], list[str]]:
        """Combine planner evidence and bounded field-coverage searches in the same scope."""
        filters = SearchFilters(bid_id=self.bid_id, addendum_number=amendment,
                                exclude_addenda=amendment is None)
        started = perf_counter()
        hits: dict[str, SearchHit] = {}
        errors = []
        supplemental = COVERAGE_QUERIES.get(field, []) if amendment is None else []
        requests = [(query, filters), *[(q, filters.model_copy(update={
            'doc_type': 'rfp' if field in {'Due Date', 'company_name', 'Product', 'Contract or Cooperative to use'} else None
        })) for q in supplemental if q != query]]
        if amendment is None and field in {'Title', 'company_name', 'Due Date'}:
            cover_query = 'submission deadline' if field == 'Due Date' else 'issuer title solicitation'
            requests.append((cover_query, filters.model_copy(update={'doc_type': 'rfp', 'page_number': 1})))
        with self.lock:
            for index, (planned_query, scope) in enumerate(requests):
                began = perf_counter()
                try:
                    response = self.engine.search(planned_query, filters=scope, top_k=top_k,
                                                  include_neighbors=True)
                    hits.update({h.metadata.chunk_id: h for h in response.hits})
                    self.trace.event('search_tool' if index == 0 else 'search_tool.supplement',
                        {'field': field, 'query': planned_query, 'filters': scope}, response,
                        (perf_counter() - began) * 1000,
                        tool_calls=[{'tool': 'search', 'scope': 'base' if amendment is None else amendment}])
                except Exception as exc:
                    reason = f'Retrieval failed ({type(exc).__name__})'
                    errors.append(reason)
                    self.trace.event('search_tool', {'field': field, 'query': planned_query, 'filters': scope},
                        {'error': reason}, (perf_counter() - began) * 1000, status='error')
        # A failed supplemental query must not discard independently successful
        # evidence, but failure of every search remains a retriable field error.
        self.trace.event('search_tool.complete', {'field': field, 'scope': amendment},
                         {'chunk_ids': list(hits), 'search_errors': errors}, (perf_counter() - started) * 1000)
        return list(hits.values()), errors if not hits else []
