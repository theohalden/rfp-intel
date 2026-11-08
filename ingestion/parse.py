"""Discover bid files and parse page-level markdown with auditable diagnostics."""
from __future__ import annotations

from collections import Counter
import json
import logging
from pathlib import Path
import re
import unicodedata
from time import perf_counter
from typing import Any

from bs4 import BeautifulSoup
from markdownify import markdownify
import pdfplumber
import pymupdf
import pymupdf4llm

from config import Parsing
from ingestion.ocr import ocr_page
from schemas import DocType, FileReport, IngestionReport, PageMetadata, ParsedPage

LOGGER = logging.getLogger(__name__)
SUPPORTED = {'.pdf', '.html', '.htm'}
ADDENDUM = re.compile(r'\b(?:addendum|amendment)\s*(?:no\.?\s*|number\s*|#\s*)?(\d+)\b', re.I)
TABLE_SEPARATOR = re.compile(r'^\s*\|?\s*:?-{3,}:?\s*\|[\s|:\-]*$')


def discover_files(bid_dir: Path) -> list[Path]:
    """Find files recursively, without relying on particular bid or file names."""
    if not bid_dir.is_dir():
        raise ValueError(f'Bid directory does not exist: {bid_dir}')
    return sorted(p for p in bid_dir.rglob('*') if p.is_file() and not p.name.startswith('.'))


def classify_document(file_name: str, first_page: str, suffix: str) -> tuple[DocType, int | None]:
    """Combine filename and first-page signals using generic document terms."""
    if suffix.lower() in {'.html', '.htm'}:
        return 'bid_page', None
    name = file_name.replace('_', ' ')
    first_page = re.sub(r'[*#`]', '', first_page)
    content = f'{name}\n{first_page[:12000]}'
    # A referenced amendment inside an RFP is not itself an amendment.
    match = ADDENDUM.search(name) or ADDENDUM.search(first_page[:1500])
    if match:
        return 'addendum', int(match.group(1))
    if (re.search(r'affidavit', name, re.I)
            or re.search(r'^\s*(?:\w+\s+){0,4}affidavit\b', first_page, re.I)
            or ('I FURTHER AFFIRM THAT' in first_page and 'CERTIFICATION' in first_page)):
        return 'affidavit', None
    if (re.search(r'\bspec(?:s|ifications?)?\b', name, re.I)
            or (re.search(r'\b(description|specifications)\b', first_page, re.I)
                and re.search(r'\bSKU\b', first_page)
                and re.search(r'\b(processor|memory|storage|DDR\d)\b', first_page, re.I))):
        return 'specs', None
    if re.search(r'\b(rfp|porfp|request for (?:proposal|quotation)|solicitation|invitation (?:to|for) bid)\b', content, re.I):
        return 'rfp', None
    return 'other', None


def document_date(first_page: str) -> str | None:
    """Keep only explicitly labelled issue dates; never guess from deadlines."""
    match = re.search(r'\b(?:date issued|issue date|issued on|document date|dated)\s*[:\-]?\s*'
                      r'([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})', first_page, re.I)
    return match.group(1) if match else None


def clean_text(text: str) -> str:
    """Normalize prose conservatively, leaving markdown table rows intact."""
    text = text.replace('\u00ad', '').replace('\xa0', ' ').replace('\r\n', '\n')
    text = re.sub(r'(?<=[a-z])-\n(?=[a-z])', '', text)
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.splitlines()]
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()


def markdown_tables(text: str) -> list[str]:
    """Identify complete pipe-table blocks for later table-aware chunking."""
    tables: list[str] = []
    lines = text.splitlines()
    i = 0
    while i + 1 < len(lines):
        if '|' in lines[i] and TABLE_SEPARATOR.match(lines[i + 1]):
            start = i
            i += 2
            while i < len(lines) and '|' in lines[i] and lines[i].strip():
                i += 1
            tables.append('\n'.join(lines[start:i]))
        else:
            i += 1
    return tables


def rows_to_markdown(rows: list[list[Any]]) -> str:
    """Convert a fallback table to markdown, escaping cell delimiters."""
    width = max((len(row) for row in rows), default=0)
    if not width:
        return ''
    normalized = [[str(cell or '').replace('|', '\\|').replace('\n', '<br>') for cell in row]
                  + [''] * (width - len(row)) for row in rows]
    render = lambda row: '| ' + ' | '.join(row) + ' |'
    return '\n'.join([render(normalized[0]), render(['---'] * width),
                      *[render(row) for row in normalized[1:]]])


def native_word_coverage(native: str, parsed: str) -> float:
    """Measure native word retention, accounting for repeated words and ligatures."""
    def words(text: str) -> Counter[str]:
        return Counter(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))
    source = words(native)
    return sum((source & words(parsed)).values()) / sum(source.values()) if source else 1.0


def inventory_table(page: Any) -> tuple[str, float, float] | None:
    """Recover two-column inventory rows using aligned identifier anchors.

    Reject prose pages without a repeated right-hand code column. This deliberately
    supports a conservative layout instead of inferring arbitrary borderless tables.
    """
    words = page.extract_words()
    code = re.compile(r'(?=.*[0-9])(?=.*-)[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+$')
    candidates = [w for w in words if code.fullmatch(w['text'])
                  and w['x0'] > page.width * .4]
    if len(candidates) < 3:
        return None
    anchors = max(([w for w in candidates if abs(w[edge] - seed[edge]) < 6]
                   for seed in candidates for edge in ('x0', 'x1')), key=len)
    if len(anchors) < 3:
        return None
    anchors.sort(key=lambda w: w['top'])
    divider = min(w['x0'] for w in anchors) - 4
    start, end = max(0, anchors[0]['top'] - 2), min(page.height, anchors[-1]['bottom'] + 12)
    centers = [(w['top'] + w['bottom']) / 2 for w in anchors]
    boundaries = [start] + [(a + b) / 2 for a, b in zip(centers, centers[1:])] + [end]
    rows = [['Description', 'SKU / Part number']]
    for i, anchor in enumerate(anchors):
        selected = [w for w in words if w['x1'] < divider
                    and boundaries[i] <= (w['top'] + w['bottom']) / 2 < boundaries[i + 1]]
        selected.sort(key=lambda w: (round(w['top'] / 3), w['x0']))
        description = ' '.join(w['text'] for w in selected)
        if not description:
            return None
        rows.append([description, anchor['text']])
    return rows_to_markdown(rows), start, end


def remove_repeated_margins(pages: list[ParsedPage]) -> None:
    """Keep one copy of repeated boundary text so document titles are not erased."""
    if len(pages) < 3:
        return
    counts: Counter[str] = Counter()
    for page in pages:
        lines = [line for line in page.text.splitlines() if line.strip()]
        counts.update(set(lines[:2] + lines[-2:]))
    repeated = {line for line, count in counts.items()
                if count >= max(3, int(len(pages) * .7 + .99)) and len(line) < 160
                and '|' not in line and not re.search(r'\b(due|deadline|shall|must)\b', line, re.I)}
    retained: set[str] = set()
    for page in pages:
        lines = page.text.splitlines()
        nonempty = [i for i, line in enumerate(lines) if line.strip()]
        boundary = set(nonempty[:2] + nonempty[-2:])
        cleaned = []
        for i, line in enumerate(lines):
            if i in boundary and line in repeated:
                if line in retained:
                    continue
                retained.add(line)
            cleaned.append(line)
        page.text = '\n'.join(cleaned).strip()


def parse_html(path: Path) -> tuple[list[tuple[str, list[str], list[str]]], list[str]]:
    """Strip non-content HTML and convert text and tables to markdown."""
    soup = BeautifulSoup(path.read_bytes(), 'html.parser')
    for tag in soup(['script', 'style', 'noscript', 'nav', 'footer', 'header']):
        tag.decompose()
    body = soup.find('main') or soup.find('article') or soup.body or soup
    text = clean_text(markdownify(str(body), heading_style='ATX'))
    warnings = [] if text else ['Empty HTML content']
    return [(text, markdown_tables(text), warnings)], []


def parse_pdf(path: Path, options: Parsing) -> tuple[list[tuple[str, list[str], list[str]]], list[str]]:
    """Parse pages independently so one damaged page cannot discard a whole file."""
    result: list[tuple[str, list[str], list[str]]] = []
    errors: list[str] = []
    with pymupdf.open(path) as doc, pdfplumber.open(path) as fallback:
        headers = pymupdf4llm.IdentifyHeaders(doc)
        opening = doc[0].get_text() if len(doc) else ''
        borderless_inventory = bool(re.search(r'\bdescription\b', opening, re.I)
                                    and re.search(r'\b(SKU|part number|unit price)\b', opening, re.I))
        for index in range(len(doc)):
            warnings: list[str] = []
            try:
                chunks = pymupdf4llm.to_markdown(doc, pages=[index], page_chunks=True,
                                               table_strategy=options.table_strategy, hdr_info=headers,
                                               show_progress=False)
                text = clean_text(chunks[0]['text'] if isinstance(chunks, list) else chunks)
            except Exception as exc:
                LOGGER.exception('Markdown parsing failed: %s page %s', path.name, index + 1)
                errors.append(f'Page {index + 1}: markdown parsing failed: {exc}')
                text = clean_text(doc[index].get_text())
                warnings.append('Plain-text fallback used; layout may be lost')
            native_text = clean_text(doc[index].get_text(sort=True))
            if len(text) < .35 * len(native_text):
                text = native_text
                warnings.append('Markdown omitted most native text; plain-text coverage fallback used')
            if (options.ocr_enabled and len(re.sub(r'\s+', '', native_text))
                    < options.ocr_min_native_chars):
                try:
                    recognized = clean_text(ocr_page(doc[index], options))
                    if recognized:
                        text = recognized
                        warnings.append('Tesseract OCR used; recognition and table layout require review')
                    else:
                        warnings.append('Tesseract OCR returned no text')
                except Exception as exc:
                    LOGGER.exception('OCR failed: %s page %s', path.name, index + 1)
                    errors.append(f'Page {index + 1}: OCR failed: {exc}')
            tables = markdown_tables(text)
            if not tables:
                try:
                    page = fallback.pages[index]
                    detected = page.extract_tables()
                    if not detected and borderless_inventory:
                        inventory = inventory_table(page)
                        if inventory:
                            table, top, bottom = inventory
                            prefix = page.crop((0, 0, page.width, top)).extract_text() or ''
                            suffix = page.crop((0, bottom, page.width, page.height)).extract_text() or ''
                            text = clean_text(prefix + '\n\n' + table + '\n\n' + suffix)
                            tables = [table]
                            warnings.append('Borderless inventory recovered from aligned code column; verify rows')
                    for rows in detected:
                        table = rows_to_markdown(rows)
                        if table:
                            tables.append(table)
                            text += '\n\n' + table
                    if detected and tables:
                        warnings.append('pdfplumber table fallback used; prose may duplicate table cells')
                except Exception as exc:
                    LOGGER.exception('Table fallback failed: %s page %s', path.name, index + 1)
                    errors.append(f'Page {index + 1}: table fallback failed: {exc}')
            if native_word_coverage(native_text, text) < .97:
                text += '\n\n### Native text supplement (layout fallback)\n\n' + native_text
                warnings.append('Native text supplement retained omitted words; some content may repeat')
            if not text:
                warnings.append('No extractable text after parsing; OCR ' +
                                ('attempted without recovery' if options.ocr_enabled else 'is not enabled'))
            result.append((text, tables, warnings))
    return result, errors


def ingest_bid(bid_dir: Path, output_dir: Path, options: Parsing | None = None) -> IngestionReport:
    """Persist normalized pages, a summary, and per-file JSONL diagnostic events."""
    options = options or Parsing()
    files = discover_files(bid_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = IngestionReport(bid_id=bid_dir.resolve().name)
    all_pages: list[ParsedPage] = []
    with (output_dir / 'ingestion_trace.jsonl').open('w', encoding='utf-8') as trace:
        for path in files:
            started = perf_counter()
            relative = path.relative_to(bid_dir).as_posix()
            item = FileReport(file_name=relative)
            try:
                if path.suffix.lower() not in SUPPORTED:
                    item.warnings.append('Unsupported file type; skipped')
                    raw, errors = [], []
                elif path.suffix.lower() == '.pdf':
                    raw, errors = parse_pdf(path, options)
                else:
                    raw, errors = parse_html(path)
                item.errors.extend(errors)
                first_page = raw[0][0] if raw else ''
                item.doc_type, item.addendum_number = classify_document(path.name, first_page, path.suffix)
                date = document_date(first_page)
                pages = [ParsedPage(metadata=PageMetadata(
                    bid_id=report.bid_id, file_name=relative, doc_type=item.doc_type,
                    addendum_number=item.addendum_number, page_number=i + 1, doc_date=date,
                    text_source=('ocr' if any(w.startswith('Tesseract OCR used') for w in warnings)
                                 else 'pdfplumber' if any(w.startswith('Borderless inventory recovered') for w in warnings)
                                 else 'native_fallback' if any(w.startswith(('Plain-text fallback', 'Markdown omitted')) for w in warnings)
                                 else 'markdown')),
                    text=text, tables=tables, warnings=warnings)
                    for i, (text, tables, warnings) in enumerate(raw)]
                if options.remove_repeated_margins:
                    remove_repeated_margins(pages)
                all_pages.extend(pages)
                item.page_count = len(raw)
                item.parsed_pages = sum(bool(p.text) for p in pages)
                item.ocr_pages = sum(p.metadata.text_source == 'ocr' for p in pages)
                item.tables_found = sum(len(p.tables) for p in pages)
                item.warnings.extend(f'Page {p.metadata.page_number}: {w}' for p in pages for w in p.warnings)
            except Exception as exc:
                LOGGER.exception('File ingestion failed: %s', path)
                item.errors.append(f'{type(exc).__name__}: {exc}')
            report.files.append(item)
            trace.write(json.dumps({'agent': 'ingestion', 'input': relative,
                'tool_calls': ['parse_pdf' if path.suffix.lower() == '.pdf' else 'parse_html']
                    if path.suffix.lower() in SUPPORTED else [],
                'output': item.model_dump(), 'tokens': 0,
                'latency_ms': round((perf_counter() - started) * 1000, 2)}) + '\n')
            trace.flush()
    with (output_dir / 'pages.jsonl').open('w', encoding='utf-8') as stream:
        for page in all_pages:
            stream.write(page.model_dump_json() + '\n')
    (output_dir / 'report.json').write_text(report.model_dump_json(indent=2), encoding='utf-8')
    return report
