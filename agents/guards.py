"""Deterministic citation and format guards applied independently of the LLM."""
from __future__ import annotations

import re
from schemas import CitationSpan, Evidence, ExtractedField, FieldResult, SearchHit


UNKNOWN_ZONE_NOTE = 'Time zone not stated in cited evidence; deadline is incomplete.'
ZONE = re.compile(r'\b(?:UTC|GMT|[ECMP][SD]T|AK[SD]T|H[AS]T|CET|CEST|IST|BST|JST|AEST|AEDT|NZST|NZDT|SAST|CAT|EAT|WAT|ET|CT|MT|PT)\b|'
                  r'\b(?:Africa|America|Asia|Atlantic|Australia|Europe|Indian|Pacific)/[A-Za-z_]+(?:/[A-Za-z_]+)?|'
                  r'(?<![\w\d])[+-](?:0\d|1[0-4]):?[0-5]\d\b')
ZONE_NAMES = {
    'Eastern Standard Time': 'EST', 'Eastern Daylight Time': 'EDT', 'Eastern Time': 'ET',
    'Central Standard Time': 'CST', 'Central Daylight Time': 'CDT', 'Central Time': 'CT',
    'Mountain Standard Time': 'MST', 'Mountain Daylight Time': 'MDT',
    'Pacific Standard Time': 'PST', 'Pacific Daylight Time': 'PDT', 'Pacific Time': 'PT',
    'South African Standard Time': 'SAST', 'Coordinated Universal Time': 'UTC',
    'Greenwich Mean Time': 'GMT',
}


def deadline_zones(text: str) -> set[str]:
    """Read literal zone labels; never infer one from an agency address or date."""
    found = {m.group().replace(':', '') for m in ZONE.finditer(text)}
    for name, abbreviation in ZONE_NAMES.items():
        if re.search(r'\b' + re.escape(name) + r'\b', text, re.I):
            found.add(abbreviation)
    return found


def prepare_deadline(result: FieldResult) -> tuple[FieldResult, list[str]]:
    """Preserve supported partial dates and reject invented source time zones."""
    value_zones = deadline_zones(result.value or '')
    source_zones = deadline_zones(' '.join(source.quote for source in result.sources))
    if value_zones - source_zones:
        return missing(), ['Submission deadline time zone is not supported by its cited evidence']
    if not value_zones and not source_zones:
        notes = result.notes
        if UNKNOWN_ZONE_NOTE not in notes:
            notes = (notes + ' ' + UNKNOWN_ZONE_NOTE).strip()
        return result.model_copy(update={'notes': notes, 'confidence': min(result.confidence, .7)}), []
    return result, []


def normalize_quote(text: str) -> str:
    """Only normalize whitespace; preserve case, punctuation, and markdown."""
    return ' '.join(text.split())


def quote_verified(source: Evidence, evidence: list[SearchHit], bid_id: str) -> bool:
    """Check the quote against a matching retrieved bid, physical page, and file."""
    quote = normalize_quote(source.quote)
    return bool(quote) and any(hit.metadata.bid_id == bid_id
        and hit.source.file == source.file and hit.source.page == source.page
        and quote in normalize_quote(hit.text) for hit in evidence)


def missing(notes: str = 'Not found in documents') -> FieldResult:
    """Return an explicit missing value without fabricated citations."""
    return FieldResult(value=None, sources=[], confidence=0, notes=notes)


def part_identifier_evidence(texts: list[str]) -> bool:
    """Require a part/order label or a separate coded table cell, not model prose.

    An opaque code in a product table can be labeled Model by the issuer. A
    commercial-model suffix in a prose sentence does not establish a second SKU.
    """
    label = re.compile(r'\b(?:part[\s_-]*(?:number|num\.?|no\.?|#)|p/n|sku|'
                       r'(?:product|order|configuration)[\s_-]*(?:code|number|id)|si\s*#)', re.I)
    for text in texts:
        if label.search(text):
            return True
        for line in text.splitlines():
            if not line.strip().startswith('|'):
                continue
            cells = [re.sub(r'<br\s*/?>', '', c, flags=re.I).strip(' *`')
                     for c in line.strip().strip('|').split('|')]
            if len([c for c in cells if c]) < 3:
                continue
            if any(re.fullmatch(r'(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9#._/\-]+', c) for c in cells):
                return True
    return False


def resolve_sources(sources: list[Evidence | CitationSpan], evidence: list[SearchHit],
                    bid_id: str) -> tuple[list[Evidence], list[str]]:
    """Copy valid line selections from retrieved chunks; reject invented/out-of-scope IDs."""
    resolved = []
    for source in sources:
        if isinstance(source, CitationSpan):
            hit = next((h for h in evidence if h.metadata.chunk_id == source.chunk_id
                        and h.metadata.bid_id == bid_id), None)
            if hit is None:
                return [], ['Citation references an unavailable source chunk']
            lines = hit.text.splitlines()
            if not 1 <= source.start_line <= source.end_line <= len(lines):
                return [], ['Citation line span is out of bounds']
            quote = '\n'.join(lines[source.start_line - 1:source.end_line])
            if not quote.strip():
                return [], ['Citation span contains no text']
            source = Evidence(file=hit.source.file, page=hit.source.page, quote=quote)
        if not quote_verified(source, evidence, bid_id):
            return [], ['Citation guard rejected missing or non-verbatim evidence']
        resolved.append(source)
    return resolved, []


def guard_draft(draft: ExtractedField, evidence: list[SearchHit], bid_id: str) -> tuple[FieldResult, list[str]]:
    """Reject the entire value if any citation is unsupported or missing."""
    if draft.value is None or (not draft.sources and draft.value.strip().casefold() in {
            'not found in documents', 'not stated', 'not specified', 'null', 'none', 'n/a'}):
        return missing(), []
    sources, failures = resolve_sources(draft.sources, evidence, bid_id)
    if not sources or failures:
        return missing(), failures or ['Citation guard rejected missing or non-verbatim evidence']
    if not draft.value.strip():
        return missing(), ['Empty extracted value']
    if draft.field == 'Part_no':
        cited_chunks = [h.text for h in evidence if any(s.file == h.source.file and s.page == h.source.page
            and normalize_quote(s.quote) in normalize_quote(h.text) for s in sources)]
        if not part_identifier_evidence(cited_chunks):
            return missing(), ['Part identifiers require an explicit part/order/configuration label or a separate coded product-table cell; commercial model/compatibility prose alone is not part-number evidence']
    confidence = max(0, min(.95, draft.confidence))
    if any(hit.metadata.text_source == 'ocr' and any(
        hit.source.file == s.file and hit.source.page == s.page for s in sources) for hit in evidence):
        confidence = min(confidence, .7)
    result = FieldResult(value=draft.value, sources=sources, confidence=confidence, notes=draft.notes)
    return prepare_deadline(result) if draft.field == 'Due Date' else (result, [])


def format_issues(field: str, result: FieldResult) -> list[str]:
    """Conservative field-format checks without inventing missing date components."""
    if result.value is None:
        return []
    issues = []
    if field == 'Due Date':
        from search.deadlines import DATE
        if not DATE.search(result.value):
            issues.append('Submission deadline needs a recognizable calendar date')
        if not re.search(r'\b(?:[01]?\d|2[0-3])(?::|hr?)(?:[0-5]\d)\b|\b(?:0?[1-9]|1[0-2])\s*[AP]\.?M\.?(?!\w)|\bnoon\b', result.value, re.I):
            issues.append('Submission deadline time is missing')
        if not deadline_zones(result.value) and UNKNOWN_ZONE_NOTE not in result.notes:
            issues.append('Submission deadline timezone is missing')
    if field == 'Bid Number' and not re.search(r'\d', result.value):
        issues.append('Bid number has no numeric component')
    return issues


def unique_sources(sources: list[Evidence]) -> list[Evidence]:
    """Deduplicate evidence without rewriting its verbatim quote."""
    seen = set()
    result = []
    for source in sources:
        key = (source.file, source.page, normalize_quote(source.quote))
        if key not in seen:
            seen.add(key)
            result.append(source)
    return result
