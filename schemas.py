"""Typed contracts shared by ingestion, retrieval, and agents."""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict
import operator
from pydantic import BaseModel, ConfigDict, Field, model_validator

FIELD_NAMES = (
    'Bid Number', 'Title', 'Due Date', 'Bid Submission Type', 'Term of Bid',
    'Pre Bid Meeting', 'Installation', 'Bid Bond Requirement', 'Delivery Date',
    'Payment Terms', 'Any Additional Documentation Required', 'MFG for Registration',
    'Contract or Cooperative to use', 'Model_no', 'Part_no', 'Product', 'contact_info',
    'company_name', 'Bid Summary', 'Product Specification',
)
DocType = Literal['bid_page', 'rfp', 'addendum', 'specs', 'affidavit', 'other']


class StrictModel(BaseModel):
    """Reject accidental fields in structured messages."""
    model_config = ConfigDict(extra='forbid')


class Evidence(StrictModel):
    """A verbatim supporting passage and its physical source page."""
    file: str
    page: int = Field(ge=1)
    quote: str = Field(min_length=1)


class FieldResult(StrictModel):
    """One extracted value; quote support is checked separately against retrieval."""
    value: str | None = None
    sources: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)
    notes: str = ''

    @model_validator(mode='after')
    def require_evidence(self) -> FieldResult:
        """Require citations for populated values and a reason for missing values."""
        if self.value is not None and (not self.value.strip() or not self.sources):
            raise ValueError('A nonempty value requires at least one source')
        if self.value is None and not self.notes:
            self.notes = 'Not found in documents'
        return self


class ValidationReport(StrictModel):
    """Counts of mutually exclusive field validation outcomes."""
    passed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    not_found: int = Field(default=0, ge=0)


class AddendumChange(StrictModel):
    """An evidence-backed change to a field."""
    field: str
    old: str | None
    new: str | None
    source: Evidence


class BidRecord(StrictModel):
    """Complete extraction output for a bid."""
    bid_id: str
    fields: dict[str, FieldResult]
    addendum_changes: list[AddendumChange] = Field(default_factory=list)
    validation: ValidationReport = Field(default_factory=ValidationReport)

    @model_validator(mode='after')
    def require_twenty_fields(self) -> BidRecord:
        """Keep final records consistent with the assignment's exact field names."""
        if set(self.fields) != set(FIELD_NAMES):
            raise ValueError('BidRecord must contain exactly the 20 required fields')
        return self


class PageMetadata(StrictModel):
    """Metadata inherited by every later search chunk."""
    bid_id: str
    file_name: str
    doc_type: DocType
    addendum_number: int | None = Field(default=None, ge=1)
    page_number: int = Field(ge=1)
    doc_date: str | None = None
    text_source: Literal['markdown', 'native_fallback', 'pdfplumber', 'ocr'] = 'markdown'


class ParsedPage(StrictModel):
    """Page markdown and intact table blocks; HTML uses synthetic page 1."""
    metadata: PageMetadata
    text: str
    tables: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FileReport(StrictModel):
    """Per-file ingestion diagnostics, including partial failures."""
    file_name: str
    doc_type: DocType = 'other'
    addendum_number: int | None = None
    page_count: int = 0
    parsed_pages: int = 0
    ocr_pages: int = 0
    tables_found: int = 0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class IngestionReport(StrictModel):
    """Summary of all discovered files in a bid directory."""
    bid_id: str
    files: list[FileReport] = Field(default_factory=list)


def merge_state_maps(left: dict, right: dict) -> dict:
    """Merge disjoint parallel-node updates; serial retries replace matching keys."""
    return {**left, **right}


class State(TypedDict, total=False):
    """Shared LangGraph state with reducers for the four parallel specialists."""
    bid_id: str
    plan: dict[str, list[str]]
    evidence: Annotated[dict[str, list[SearchHit]], merge_state_maps]
    drafts: Annotated[dict[str, FieldCandidates], merge_state_maps]
    fields: Annotated[dict[str, FieldResult], merge_state_maps]
    changes: list[AddendumChange]
    validation: ValidationReport
    issues: dict[str, list[str]]
    retries: dict[str, int]
    final: BidRecord
    errors: Annotated[list[str], operator.add]


class ChunkMetadata(PageMetadata):
    """Source metadata plus stable identity and chunk structure."""
    chunk_id: str
    kind: Literal['text', 'table']
    section: str = ''


class SearchChunk(StrictModel):
    """Persisted retrieval unit; one complete detected table stays one unit."""
    text: str = Field(min_length=1)
    metadata: ChunkMetadata


class SearchFilters(StrictModel):
    """Explicit metadata constraints shared by both retrieval branches."""
    bid_id: str | None = None
    doc_type: DocType | None = None
    addendum_number: int | None = Field(default=None, ge=1)
    file_name: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    exclude_addenda: bool = False

    @model_validator(mode='after')
    def reject_conflicts(self) -> SearchFilters:
        """Do not silently accept mutually exclusive search scopes."""
        if self.exclude_addenda and (self.doc_type == 'addendum' or self.addendum_number):
            raise ValueError('Cannot request addenda and exclude them together')
        return self


class SearchHit(StrictModel):
    """Full cited passage with rank and optional non-probabilistic model score."""
    rank: int = Field(ge=1)
    text: str
    metadata: ChunkMetadata
    source: Evidence
    rerank_score: float | None = None
    notes: str = ''


class SearchResponse(StrictModel):
    """Search response with enough detail to reproduce its configuration."""
    query: str
    expanded_queries: list[str]
    rerank_query: str | None = None
    mode: Literal['vector', 'hybrid', 'hybrid_rerank']
    filters: SearchFilters
    hits: list[SearchHit]
    latency_ms: float
    warnings: list[str] = Field(default_factory=list)


class IndexResult(StrictModel):
    """Incremental indexing counts for one bid directory."""
    bid_id: str
    chunks: int
    added: int
    removed: int
    unchanged: int
    parsed: bool
    warnings: list[str] = Field(default_factory=list)


class GoldSource(Evidence):
    """A manually selected source anchor used only for retrieval evaluation."""
    bid_id: str


class GoldTarget(StrictModel):
    """One required fact; equivalent passages can be accepted as alternatives."""
    sources: list[GoldSource] = Field(min_length=1)


class EvalQuestion(StrictModel):
    """A query with source-backed relevance judgments, independent of retrieval."""
    id: str
    query: str
    filters: SearchFilters = Field(default_factory=SearchFilters)
    targets: list[GoldTarget] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


class EvalSet(StrictModel):
    """Small development benchmark; no claim of held-out generalization."""
    description: str
    questions: list[EvalQuestion] = Field(min_length=15)

    @model_validator(mode='after')
    def unique_ids(self) -> EvalSet:
        """Prevent accidental duplicated questions from weighting the averages twice."""
        ids = [q.id for q in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError('Evaluation question IDs must be unique')
        return self


class QueryRequest(StrictModel):
    """A field and the search queries planned for it."""
    field: str
    queries: list[str]


class ExtractionPlan(StrictModel):
    """LLM query expansion constrained to known extraction fields."""
    requests: list[QueryRequest]


class CitationSpan(StrictModel):
    """A source selection copied verbatim by code rather than retyped by the LLM."""
    chunk_id: str
    start_line: int
    end_line: int


class ExtractedField(StrictModel):
    """Untrusted model draft, verified by code before becoming a FieldResult."""
    field: str
    value: str | None
    sources: list[CitationSpan]
    confidence: float
    notes: str


class ExtractionBatch(StrictModel):
    """Structured specialist response for a requested field subset."""
    fields: list[ExtractedField]


class AmendmentCandidate(StrictModel):
    """A verified candidate extracted from exactly one numbered amendment."""
    number: int
    result: FieldResult


class FieldCandidates(StrictModel):
    """Base and separately retrieved amendment evidence for one field."""
    base: FieldResult
    amendments: list[AmendmentCandidate] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)


class AmendmentDecision(StrictModel):
    """Whether an amendment actually modifies a field, reviewed before application."""
    field: str
    applies: bool
    operation: Literal['replace', 'append', 'clarify', 'unchanged']
    reason: str


class AmendmentReview(StrictModel):
    """One amendment's explicit field-by-field applicability decisions."""
    decisions: list[AmendmentDecision]


class SupportReview(StrictModel):
    """Explicit semantic verdict after checking all assertions against source quotes."""
    supported: bool
    reason: str


class NarrowedField(StrictModel):
    """A candidate rewritten to keep only the assertions its selected quotes support."""
    value: str | None
    notes: str


class SummarySentence(StrictModel):
    """One generated sentence referencing its validated supporting fields."""
    text: str
    supporting_fields: list[str]


class SummaryDraft(StrictModel):
    """A 3–6 sentence summary assembled only after field validation."""
    sentences: list[SummarySentence]


class QAEvidence(Evidence):
    """A Q&A citation includes bid identity so cross-bid answers remain unambiguous."""
    bid_id: str


class QAClaim(StrictModel):
    """One answer statement and the exact retrieved evidence that supports it."""
    text: str
    sources: list[QAEvidence]


class QASelection(StrictModel):
    """Untrusted claim with source-line selections resolved to quotes by code."""
    text: str
    sources: list[CitationSpan]


class QADraft(StrictModel):
    """Untrusted structured ReAct answer, verified before presentation."""
    claims: list[QASelection]
    not_found: bool
    notes: str


class QAResponse(StrictModel):
    """Guarded answer with source quotes, diagnostic notes, and the local trace path."""
    question: str
    answer: str
    claims: list[QAClaim]
    sources: list[QAEvidence]
    not_found: bool
    notes: str
    latency_ms: float
    trace_file: str


class IndexRequest(StrictModel):
    """Local folder to incrementally index on the API host."""
    bid_dir: str = Field(min_length=1)


class AskRequest(StrictModel):
    """Question and optional user-enforced retrieval scope."""
    question: str = Field(min_length=1, max_length=2000)
    bid_id: str | None = None
    doc_type: DocType | None = None
