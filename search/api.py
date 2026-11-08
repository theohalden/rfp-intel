"""REST interface for incremental indexing, cited search, and guarded Q&A."""
from __future__ import annotations

from pathlib import Path
import logging
from typing import Literal
from fastapi import FastAPI, HTTPException, Query
from config import Settings, load_settings
from schemas import DocType, IndexResult, SearchFilters, SearchResponse, IndexRequest, AskRequest, QAResponse
from search.index import add_bid
from search.retrieve import SearchEngine

LOGGER = logging.getLogger(__name__)



def create_app(settings: Settings | None = None) -> FastAPI:
    """Construct an app without loading models until its first real request."""
    settings = settings or load_settings()
    engine = SearchEngine(settings)
    qa_engine = None
    application = FastAPI(title='RFP Intelligence Platform', version='0.3.0')

    @application.get('/health')
    def health() -> dict[str, str]:
        """Check that the HTTP process is alive, without claiming models are ready."""
        return {'status': 'ok', 'stage': 'search_and_agents'}

    @application.post('/index', response_model=IndexResult)
    def index_bid(request: IndexRequest) -> IndexResult:
        """Incrementally index a local bid directory."""
        path = Path(request.bid_dir).expanduser().resolve()
        if not path.is_dir():
            raise HTTPException(400, 'bid_dir must be an existing directory on this server')
        try:
            return add_bid(path, settings)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            LOGGER.exception('Index request failed')
            raise HTTPException(503, 'Indexing failed; inspect server logs and the ingestion report') from exc

    @application.get('/search', response_model=SearchResponse)
    def search(
        q: str = Query(min_length=1, max_length=2000),
        bid_id: str | None = None, doc_type: DocType | None = None,
        addendum_number: int | None = Query(default=None, ge=1),
        file_name: str | None = None, page_number: int | None = Query(default=None, ge=1),
        exclude_addenda: bool = False, top_k: int = Query(default=5, ge=1, le=50),
        mode: Literal['vector', 'hybrid', 'hybrid_rerank'] = 'hybrid_rerank', expand: bool = True,
    ) -> SearchResponse:
        """Return full passages with file/page citations and retrieval metadata."""
        try:
            filters = SearchFilters(bid_id=bid_id, doc_type=doc_type, addendum_number=addendum_number,
                                    file_name=file_name, page_number=page_number,
                                    exclude_addenda=exclude_addenda)
            return engine.search(q, filters, top_k, mode, expand)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(409, str(exc)) from exc
        except Exception as exc:
            LOGGER.exception('Search request failed')
            raise HTTPException(503, 'Search unavailable; inspect server logs and model/index setup') from exc

    @application.post('/ask', response_model=QAResponse)
    def ask(request: AskRequest) -> QAResponse:
        """Answer from retrieved source passages using the guarded ReAct agent."""
        nonlocal qa_engine
        try:
            if qa_engine is None:
                from agents.qa import QAEngine
                qa_engine = QAEngine(settings)
            return qa_engine.ask(request.question, request.bid_id, request.doc_type)
        except ValueError as exc:
            raise HTTPException(400, 'Check question, filters, and configured LLM settings') from exc
        except FileNotFoundError as exc:
            raise HTTPException(409, str(exc)) from exc
        except Exception as exc:
            LOGGER.error('Q&A request failed: %s', type(exc).__name__)
            raise HTTPException(503, 'Q&A unavailable; inspect model settings and local traces') from exc

    return application


app = create_app()
