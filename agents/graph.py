"""LangGraph extraction: parallel specialists, amendment review, and bounded retries."""
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from pathlib import Path
from time import perf_counter
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from agents import prompts
from agents.guards import format_issues, guard_draft, missing, quote_verified, unique_sources
from agents.llm import create_llm
from agents.tools import EvidenceTool
from agents.trace import TraceWriter, jsonable
from config import Settings
from schemas import (AmendmentCandidate, AmendmentReview, AddendumChange, BidRecord,
    SupportReview, NarrowedField, ExtractionBatch, ExtractionPlan, FIELD_NAMES, FieldCandidates,
    FieldResult, SearchHit, State, SummaryDraft, ValidationReport)


class ExtractionWorkflow:
    """Orchestrate typed agent messages without placing whole documents in prompts."""

    def __init__(self, settings: Settings, bid_id: str, llm: Any | None = None,
                 tool: Any | None = None, progress: Callable[[str], None] | None = None):
        self.settings = settings
        self.bid_id = bid_id
        self.trace = TraceWriter(settings.paths.outputs, bid_id)
        self.llm = llm if llm is not None else create_llm(settings)
        self.tool = tool if tool is not None else EvidenceTool(settings, bid_id, self.trace)
        self.progress = progress or (lambda message: None)
        self._semantic_reviews: dict[str, SupportReview] = {}
        graph = StateGraph(State)
        graph.add_node('expand_queries', self.expand_queries)
        for group in prompts.GROUPS:
            graph.add_node(group, self.specialist(group))
        graph.add_node('reconcile_addenda', self.reconcile_addenda)
        graph.add_node('validate', self.validate)
        graph.add_node('re_extract', self.re_extract)
        graph.add_node('report', self.report)
        graph.add_edge(START, 'expand_queries')
        for group in prompts.GROUPS:
            graph.add_edge('expand_queries', group)
        graph.add_edge(list(prompts.GROUPS), 'reconcile_addenda')
        graph.add_edge('reconcile_addenda', 'validate')
        graph.add_conditional_edges('validate', self.route_validation,
                                    {'retry': 're_extract', 'done': 'report'})
        graph.add_edge('re_extract', 'validate')
        graph.add_edge('report', END)
        self.graph = graph.compile()

    def invoke_llm(self, agent: str, schema: type[BaseModel], instruction: str,
                   payload: dict) -> BaseModel | None:
        """Use provider schema constraints; log raw usage and safe error metadata."""
        started = perf_counter()
        try:
            result = self.llm.with_structured_output(schema, method='json_schema', strict=True,
                include_raw=True).invoke([
                    ('system', (prompts.SYSTEM if schema is ExtractionBatch else prompts.REVIEW_SYSTEM) + '\n\n' + instruction),
                    ('human', json.dumps(jsonable(payload), ensure_ascii=False)),
                ])
            parsed = result.get('parsed')
            raw = result.get('raw')
            tokens = getattr(raw, 'usage_metadata', None)
            if parsed is None or result.get('parsing_error'):
                self.trace.event(agent, payload, {'error': 'Structured output could not be parsed'},
                    (perf_counter() - started) * 1000, tokens=tokens, status='error')
                return None
            self.trace.event(agent, payload, parsed, (perf_counter() - started) * 1000, tokens=tokens)
            return parsed
        except Exception as exc:
            # Do not serialize provider exceptions: they can include credential/request details.
            self.trace.event(agent, payload, {'error_type': type(exc).__name__,
                'status_code': getattr(exc, 'status_code', None), 'code': getattr(exc, 'code', None)},
                (perf_counter() - started) * 1000, status='error')
            self.progress(f'{agent}: model call failed ({type(exc).__name__}); using guarded fallback')
            return None

    def expand_queries(self, state: State) -> dict:
        """Plan field-specific queries, retaining deterministic defaults on failure."""
        self.progress('Planning field searches')
        result = self.invoke_llm('expand_queries', ExtractionPlan,
            'Produce up to two concise search queries per requested field. Use synonyms, not answers. '
            'Do not invent bid-specific identifiers. Return each field exactly once.',
            {'bid_id': self.bid_id, 'fields': prompts.FIELD_QUERIES, 'field_notes': prompts.FIELD_NOTES})
        plan = {field: [query] for field, query in prompts.FIELD_QUERIES.items()}
        if isinstance(result, ExtractionPlan):
            for request in result.requests:
                if request.field in plan:
                    queries = [q.strip() for q in request.queries if q.strip() and len(q) <= 1000]
                    if queries:
                        plan[request.field] = list(dict.fromkeys(queries[:2] + plan[request.field]))
        return {'plan': plan, 'retries': {}, 'issues': {}, 'changes': []}

    @staticmethod
    def pool(evidence: dict[str, list[SearchHit]]) -> list[SearchHit]:
        """Deduplicate retrieved passages for a node without altering their contents."""
        return list({hit.metadata.chunk_id: hit for hits in evidence.values() for hit in hits}.values())

    @staticmethod
    def passages(hits: list[SearchHit]) -> list[dict]:
        """Expose source metadata and retrieved text, never original whole files."""
        return [{'chunk_id': h.metadata.chunk_id, 'file': h.source.file, 'page': h.source.page,
                 'doc_type': h.metadata.doc_type, 'addendum_number': h.metadata.addendum_number,
                 'text_source': h.metadata.text_source, 'numbered_lines': [
                     {'line': i + 1, 'text': line} for i, line in enumerate(h.text.splitlines())]} for h in hits]

    def within_budget(self, hits: list[SearchHit]) -> list[SearchHit]:
        """Send the model only as many retrieved passages as the per-call budget allows.

        Passages keep their retrieval order, so lower-ranked ones are dropped first; the
        first passage is always kept. Dropped passages stay in the pool used for quote checks.
        """
        kept, used = [], 0
        for hit in hits:
            if kept and used + len(hit.text) > self.settings.agents.max_evidence_chars:
                continue
            kept.append(hit)
            used += len(hit.text)
        return kept

    def collect(self, state: State, names: list[str], attempt: int = 0) -> tuple[dict, dict, dict]:
        """Retrieve independently for every field and every document scope."""
        scopes = [None, *self.tool.addenda]
        by_scope: dict = {scope: {} for scope in scopes}
        all_evidence = {}
        problems = {field: [] for field in names}
        for field in names:
            queries = state.get('plan', {}).get(field, [prompts.FIELD_QUERIES[field]])
            query = queries[min(attempt, len(queries) - 1)]
            if attempt:
                query += ' exact requirement details exceptions supporting clause'
            all_evidence[field] = []
            for scope in scopes:
                hits, errors = self.tool.search(query, field, amendment=scope, top_k=3 if not attempt else 5)
                # A repair must retain already-found facts, while keeping base and
                # numbered amendments separate for extraction/reconciliation.
                previous = [h for h in state.get('evidence', {}).get(field, [])
                    if (h.metadata.doc_type != 'addendum' if scope is None else
                        h.metadata.doc_type == 'addendum' and h.metadata.addendum_number == scope)]
                hits = self.pool({'new': hits, 'previous': previous})
                by_scope[scope][field] = hits
                all_evidence[field].extend(hits)
                problems[field].extend(errors)
        equipment = {'Model_no', 'Part_no', 'Product', 'Product Specification'} & set(names)
        for scope in scopes:
            shared = self.pool({name: by_scope[scope][name] for name in equipment})
            for name in equipment:
                by_scope[scope][name] = shared
                all_evidence[name].extend(shared)
        return by_scope, all_evidence, problems

    def extract_fields(self, state: State, names: list[str], label: str, attempt: int = 0) -> dict:
        """Extract base and each amendment in separate schema-constrained calls."""
        started = perf_counter()
        self.progress(f'{label}: searching evidence for {len(names)} fields')
        by_scope, evidence, problems = self.collect(state, names, attempt)
        drafts = {field: FieldCandidates(base=missing(), problems=problems[field]) for field in names}
        for scope, scoped_evidence in by_scope.items():
            for field in names:
                hits = scoped_evidence[field]
                if not hits:
                    continue
                instruction = prompts.EXTRACT
                if scope is None:
                    instruction += (' For Due Date, record the original main RFP deadline when available. '
                        'A portal can already reflect later amendments; preserve its discrepancy in notes '
                        'rather than erasing the original RFP date. Supplement missing time/zone only with evidence.')
                result = self.invoke_llm(f'{label}.{field}.extract.{scope if scope is not None else "base"}',
                    ExtractionBatch, instruction,
                    {'requested_fields': [{'name': field, 'query': prompts.FIELD_QUERIES[field],
                        'notes': prompts.FIELD_NOTES.get(field, '')}],
                     'scope': 'base' if scope is None else f'Addendum {scope}',
                     'evidence': self.passages(self.within_budget(hits)),
                     'previous_validation_feedback': {field: state.get('issues', {}).get(field, [])}})
                if not isinstance(result, ExtractionBatch):
                    drafts[field].problems.append(f'Structured extraction failed in scope {scope}')
                    continue
                matches = [draft for draft in result.fields if draft.field == field]
                if len(matches) != 1:
                    drafts[field].problems.append(f'Missing or duplicate field in extraction scope {scope}')
                    continue
                guarded, failures = guard_draft(matches[0], hits, self.bid_id)
                drafts[field].problems.extend(failures)
                if scope is None:
                    drafts[field].base = guarded
                else:
                    drafts[field].amendments.append(AmendmentCandidate(number=scope, result=guarded))
        self.trace.event(label, {'fields': names, 'attempt': attempt}, {'drafts': drafts},
                         (perf_counter() - started) * 1000, tool_calls=[{'tool': 'search', 'scopes': ['base', *self.tool.addenda]}])
        retained = {name: self.pool({'old': state.get('evidence', {}).get(name, []), 'new': hits})
                    for name, hits in evidence.items()}
        return {'drafts': drafts, 'evidence': retained}

    def specialist(self, group: str) -> Callable:
        """Create one independent parallel branch over a disjoint field group."""
        def node(state: State) -> dict:
            return self.extract_fields(state, prompts.GROUPS[group], group)
        return node

    def reconcile_fields(self, state: State, names: list[str]) -> dict:
        """Review applicability and apply only accepted changes in numbered order."""
        fields = {name: state['drafts'][name].base for name in names}
        changes = [change for change in state.get('changes', []) if change.field not in names]
        reconciliation_issues: dict[str, list[str]] = {}
        for number in self.tool.addenda:
            candidates = {name: next((c.result for c in state['drafts'][name].amendments
                if c.number == number and c.result.value is not None), None) for name in names}
            candidates = {name: value for name, value in candidates.items() if value is not None}
            if not candidates:
                continue
            review = self.invoke_llm(f'reconcile_addenda.{number}', AmendmentReview, prompts.RECONCILE,
                {'addendum_number': number,
                 'current_fields': {name: fields[name] for name in candidates}, 'amendment_candidates': candidates,
                 'field_meanings': {name: prompts.FIELD_NOTES.get(name, prompts.FIELD_QUERIES[name]) for name in candidates}})
            decisions = {} if review is None else {d.field: d for d in review.decisions}
            for name, candidate in candidates.items():
                decision = decisions.get(name)
                if decision is None:
                    fields[name] = missing()
                    reconciliation_issues.setdefault(name, []).append(f'Addendum {number} reconciliation unavailable')
                    continue
                if not decision.applies or decision.operation == 'unchanged':
                    continue
                old = fields[name]
                if decision.operation in {'append', 'clarify'} and old.value:
                    new = FieldResult(value=old.value + '; ' + candidate.value,
                        sources=unique_sources(old.sources + candidate.sources),
                        confidence=min(old.confidence, candidate.confidence), notes=decision.reason)
                else:
                    new = candidate.model_copy(update={'notes': decision.reason})
                fields[name] = new
                if new.value != old.value:
                    changes.append(AddendumChange(field=name, old=old.value, new=new.value,
                                                  source=candidate.sources[0]))
        updated_drafts = {name: state['drafts'][name].model_copy(deep=True) for name in names}
        for name in names:
            updated_drafts[name].problems.extend(reconciliation_issues.get(name, []))
        return {'fields': fields, 'changes': changes, 'drafts': updated_drafts}

    def reconcile_addenda(self, state: State) -> dict:
        """Join all specialist branches before reviewing the amendments."""
        self.progress('Reconciling amendment changes')
        started = perf_counter()
        result = self.reconcile_fields(state, list(prompts.FIELD_QUERIES))
        self.trace.event('reconcile_addenda', {'bid_id': self.bid_id, 'addenda': self.tool.addenda},
                         result, (perf_counter() - started) * 1000)
        return result

    def validate(self, state: State) -> dict:
        """Verify quotes/formats in code, then ask a short evidence-only consistency critic."""
        self.progress('Validating fields and source support')
        started = perf_counter()
        evidence = self.pool(state.get('evidence', {}))
        issues = {}
        for name, field in state['fields'].items():
            failures = list(state['drafts'][name].problems)
            if field.value is not None:
                if not field.sources or not all(quote_verified(source, evidence, self.bid_id) for source in field.sources):
                    failures.append('Citation guard rejected field evidence')
                failures.extend(format_issues(name, field))
            if failures:
                issues[name] = list(dict.fromkeys(failures))
        checkable = {name: field for name, field in state['fields'].items()
                     if name not in issues and (field.value is not None or
                         name in prompts.COVERAGE_FIELDS and state.get('evidence', {}).get(name))}
        if checkable:
            # Isolate each field's interpretation review: large multi-field quote pools
            # caused unrelated citations and amendment scopes to be confused in live runs.
            def review_field(item):
                name, field = item
                # Completeness is checked against retrieved context, not just the
                # model's chosen quotes. Applied amendments retain their separate
                # applicability review; do not demand superseded base facts back.
                coverage = name in prompts.COVERAGE_FIELDS and not any(
                    change.field == name for change in state.get('changes', []))
                instruction = (prompts.FIELD_REVIEW if coverage else prompts.CRITIC) + '\nField definition (instructions, not a claim): ' + prompts.FIELD_NOTES.get(name, prompts.FIELD_QUERIES[name])
                if coverage:
                    instruction += '\nCoverage checklist for THIS field only: ' + prompts.COVERAGE_CHECKS[name]
                payload = {'field_name': name, 'value': field.value, 'sources': field.sources,
                           'notes': field.notes}
                if coverage:
                    if field.value is not None:
                        support_signature = hashlib.sha256(json.dumps(jsonable({
                            'instruction': prompts.SOURCE_SUPPORT, 'payload': payload}), sort_keys=True).encode()).hexdigest()
                        support = self._semantic_reviews.get(support_signature)
                        if support is None:
                            support = self.invoke_llm('validate.source_support.' + name,
                                SupportReview, prompts.SOURCE_SUPPORT, payload)
                            if support is not None:
                                self._semantic_reviews[support_signature] = support
                        if support is None:
                            return name, ['Source-only consistency validation unavailable']
                        if not support.supported:
                            return name, [support.reason]
                    payload['retrieved_context'] = self.passages([h for h in state.get('evidence', {}).get(name, [])
                        if h.metadata.doc_type != 'addendum'])
                signature = hashlib.sha256(json.dumps(jsonable({'instruction': instruction, 'payload': payload}), sort_keys=True).encode()).hexdigest()
                critic = self._semantic_reviews.get(signature)
                if critic is None:
                    critic = self.invoke_llm('validate.consistency.' + name, SupportReview, instruction, payload)
                    if critic is not None:
                        self._semantic_reviews[signature] = critic
                if critic is None:
                    return name, ['LLM consistency validation unavailable']
                return name, [] if critic.supported else [critic.reason]
            with ThreadPoolExecutor(max_workers=4) as executor:
                for name, failures in executor.map(review_field, checkable.items()):
                    if failures:
                        issues.setdefault(name, []).extend(failures)
        counts = ValidationReport(
            passed=sum(field.value is not None and name not in issues for name, field in state['fields'].items()),
            failed=len(issues),
            not_found=sum(field.value is None and name not in issues for name, field in state['fields'].items()) + 1)
        self.trace.event('validate', {'fields': state['fields']}, {'issues': issues, 'validation': counts},
                         (perf_counter() - started) * 1000)
        return {'issues': issues, 'validation': counts}

    def route_validation(self, state: State) -> str:
        """Retry only rejected fields and never exceed the configured per-field cap."""
        return 'retry' if any(state.get('retries', {}).get(name, 0) < self.settings.agents.max_retries
                              for name in state.get('issues', {})) else 'done'

    def re_extract(self, state: State) -> dict:
        """Expand failed-field retrieval, replace drafts, and review their amendments again."""
        started = perf_counter()
        names = [name for name in state['issues']
                 if state.get('retries', {}).get(name, 0) < self.settings.agents.max_retries]
        retries = dict(state.get('retries', {}))
        for name in names:
            retries[name] = retries.get(name, 0) + 1
        attempt = max(retries[name] for name in names)
        self.progress(f'Retrieving again for {len(names)} rejected fields (attempt {attempt})')
        updates = self.extract_fields(state, names, 're_extract', attempt)
        merged = {**state, 'drafts': {**state['drafts'], **updates['drafts']},
                  'evidence': {**state['evidence'], **updates['evidence']}}
        reconciled = self.reconcile_fields(merged, names)
        self.trace.event('re_extract.complete', {'fields': names, 'retries': retries},
                         reconciled, (perf_counter() - started) * 1000)
        return {**updates, **reconciled, 'retries': retries}

    def summarize(self, fields: dict[str, FieldResult], evidence: list[SearchHit]) -> tuple[FieldResult, bool]:
        """Generate the summary last and validate it against the supporting field quotes."""
        validated = {name: field for name, field in fields.items() if field.value is not None}
        if len(validated) < 3:
            return missing(), False
        feedback = []
        for attempt in range(self.settings.agents.max_retries + 1):
            draft = self.invoke_llm('report.summary', SummaryDraft, prompts.SUMMARY,
                {'validated_fields': validated, 'previous_validation_feedback': feedback})
            if draft is None or not 3 <= len(draft.sentences) <= 6:
                continue
            supporting = [name for sentence in draft.sentences for name in sentence.supporting_fields]
            if not supporting or any(not sentence.supporting_fields for sentence in draft.sentences) or any(name not in validated for name in supporting):
                continue
            sources = unique_sources([source for name in supporting for source in validated[name].sources])
            if not all(quote_verified(source, evidence, self.bid_id) for source in sources):
                continue
            result = FieldResult(value=' '.join(sentence.text for sentence in draft.sentences), sources=sources,
                confidence=min(validated[name].confidence for name in supporting),
                notes='Generated from validated fields; confidence is an uncalibrated estimate.')
            critic = self.invoke_llm('report.summary_validation', SupportReview, prompts.CRITIC,
                                     {'fields': {'Bid Summary': result}})
            if critic is not None and critic.supported:
                return result, False
            feedback = [critic.reason] if critic else ['Review unavailable']
        return missing('Not found in documents; summary could not be validated'), True

    def narrow(self, name: str, field: FieldResult, reason: str, evidence: list[SearchHit]) -> FieldResult | None:
        """Drop only the assertions a reviewer found unsupported, then re-check the remainder.

        The citations stay the same and the shortened value must pass the same source-support
        review; nothing new can be added, so the guard is not weakened.
        """
        if field.value is None or not all(quote_verified(s, evidence, self.bid_id) for s in field.sources):
            return None
        draft = self.invoke_llm('report.narrow.' + name, NarrowedField, prompts.NARROW,
            {'field_name': name, 'value': field.value, 'notes': field.notes,
             'sources': field.sources, 'reviewer_reason': reason})
        if draft is None or not draft.value or not draft.value.strip() or draft.value.strip() == field.value.strip():
            return None  # nothing was removed, so the failure was not an unsupported claim
        narrowed = field.model_copy(update={'value': draft.value, 'notes': draft.notes,
                                            'confidence': min(field.confidence, .7)})
        if format_issues(name, narrowed):
            return None
        review = self.invoke_llm('report.narrow_check.' + name, SupportReview, prompts.SOURCE_SUPPORT,
            {'field_name': name, 'value': narrowed.value, 'sources': narrowed.sources, 'notes': narrowed.notes})
        if review is None or not review.supported:
            return None
        note = ('Narrowed after validation: assertions not supported by the cited quotes were removed '
                f'({reason}); the field may be incomplete.')
        return narrowed.model_copy(update={'notes': (narrowed.notes + ' ' + note).strip()})

    def report(self, state: State) -> dict:
        """Finalize all 20 fields, preserving unresolved failures as explicit nulls."""
        self.progress('Generating the final summary and JSON report')
        started = perf_counter()
        fields = {name: state['fields'].get(name, missing()) for name in prompts.FIELD_QUERIES}
        evidence = self.pool(state.get('evidence', {}))
        narrowed_names = set()
        for name, reasons in state.get('issues', {}).items():
            narrowed = (self.narrow(name, state['fields'][name], '; '.join(reasons), evidence)
                        if name in state['fields'] else None)
            if narrowed is not None:
                fields[name] = narrowed
                narrowed_names.add(name)
            else:
                fields[name] = missing('Not found in documents; validation failed: ' + '; '.join(reasons))
        summary, summary_failed = self.summarize(fields, self.pool(state.get('evidence', {})))
        fields['Bid Summary'] = summary
        fields = {name: fields[name] for name in FIELD_NAMES}
        failed = len(set(state.get('issues', {})) - narrowed_names) + int(summary_failed)
        passed = sum(field.value is not None for field in fields.values())
        validation = ValidationReport(passed=passed, failed=failed, not_found=len(FIELD_NAMES) - passed - failed)
        # Only changes to successfully validated final fields are reportable.
        changes = [change for change in state.get('changes', []) if fields[change.field].value is not None]
        final = BidRecord(bid_id=self.bid_id, fields=fields, addendum_changes=changes, validation=validation)
        destination = self.settings.paths.outputs / f'{self.bid_id}.json'
        destination.write_text(final.model_dump_json(indent=2), encoding='utf-8')
        self.trace.event('report', {'validated_fields': fields, 'retries': state.get('retries', {})},
                         final, (perf_counter() - started) * 1000)
        return {'final': final, 'validation': validation}

    def run(self) -> BidRecord:
        """Execute the compiled graph with a bounded recursion limit."""
        result = self.graph.invoke({'bid_id': self.bid_id, 'drafts': {}, 'evidence': {}, 'fields': {},
                                    'errors': []}, config={'recursion_limit': 20, 'max_concurrency': 4})
        return result['final']


def extract_bid(bid_dir: Path, settings: Settings, progress: Callable[[str], None] | None = None) -> BidRecord:
    """One-command incremental indexing followed by guarded graph extraction."""
    from search.index import add_bid
    if progress:
        progress('Checking the bid index')
    add_bid(bid_dir, settings)
    return ExtractionWorkflow(settings, bid_dir.resolve().name, progress=progress).run()
