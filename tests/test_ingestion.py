"""Check source metadata, table preservation, and partial-failure behavior."""
from pathlib import Path
import pymupdf
from pydantic import ValidationError
import pytest
from ingestion.parse import classify_document, ingest_bid, markdown_tables
from schemas import FieldResult


def test_renamed_pdf_classified_from_content(tmp_path: Path) -> None:
    bid = tmp_path / 'unseen_bid'
    bid.mkdir()
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), 'Amendment No. 3\nSubmission deadline extended to May 20, 2027.')
        doc.save(bid / 'renamed.pdf')
    report = ingest_bid(bid, tmp_path / 'output')
    assert report.files[0].doc_type == 'addendum'
    assert report.files[0].addendum_number == 3
    assert report.files[0].parsed_pages == 1
    assert not report.files[0].errors


def test_html_table_and_bad_file_do_not_stop_bid(tmp_path: Path) -> None:
    bid = tmp_path / 'bid'
    bid.mkdir()
    (bid / 'a.pdf').write_bytes(b'not a PDF')
    (bid / 'b.html').write_text('<main><h1>Devices</h1><script>secret noise</script>'
        '<table><tr><th>Model</th><th>Qty</th></tr><tr><td>ABC-123</td><td>20</td></tr></table></main>')
    report = ingest_bid(bid, tmp_path / 'output')
    assert report.files[0].errors
    assert report.files[1].tables_found == 1
    assert report.files[1].page_count == 1
    pages = (tmp_path / 'output' / 'pages.jsonl').read_text()
    assert 'ABC-123' in pages
    assert 'secret noise' not in pages
    assert 'bid_page' in pages


def test_table_boundaries() -> None:
    text = 'Intro\n\n| CPU | RAM |\n| --- | --- |\n| i7 | 16 GB |\n\nEnd'
    assert markdown_tables(text) == ['| CPU | RAM |\n| --- | --- |\n| i7 | 16 GB |']


def test_unlabelled_date_is_not_assumed_issue_date() -> None:
    from ingestion.parse import document_date
    assert document_date('Submission deadline: 2027-05-20') is None
    assert document_date('Issue date: 2027-04-01') == '2027-04-01'


def test_non_null_field_needs_citation() -> None:
    with pytest.raises(ValidationError):
        FieldResult(value='Net 30')
    assert FieldResult().notes == 'Not found in documents'


def test_generic_classification() -> None:
    assert classify_document('renamed.pdf', 'Contract Affidavit', '.pdf')[0] == 'affidavit'
    assert classify_document('renamed.pdf', 'Request for Proposal\nDevices', '.pdf')[0] == 'rfp'


def test_renamed_specialist_documents() -> None:
    assert classify_document('x.pdf', '**MERCURY AFFIDAVIT**', '.pdf')[0] == 'affidavit'
    assert classify_document('y.pdf', 'A. AUTHORITY\nCERTIFICATION\nI FURTHER AFFIRM THAT:', '.pdf')[0] == 'affidavit'
    assert classify_document('z.pdf', '**Description**\nProcessor DDR5 memory\n**SKU**', '.pdf')[0] == 'specs'


def test_borderless_inventory_keeps_wrapped_description(tmp_path: Path) -> None:
    from ingestion.parse import inventory_table
    import pdfplumber
    path = tmp_path / 'inventory.pdf'
    with pymupdf.open() as doc:
        page = doc.new_page()
        for y, description, code in [(100, 'Processor 12 cores', '100-ABCD'),
                                      (140, 'Memory 16 GB', '200-EFGH'),
                                      (180, 'Storage SSD', '300-IJKL')]:
            page.insert_text((40, y), description)
            page.insert_text((350, y), code)
        page.insert_text((40, 112), 'up to 4 GHz')
        doc.save(path)
    with pdfplumber.open(path) as doc:
        result = inventory_table(doc.pages[0])
    assert result is not None
    assert '| Processor 12 cores up to 4 GHz | 100-ABCD |' in result[0]
    assert '| Memory 16 GB | 200-EFGH |' in result[0]


def test_borderless_prose_is_not_inventory(tmp_path: Path) -> None:
    from ingestion.parse import inventory_table
    import pdfplumber
    path = tmp_path / 'terms.pdf'
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((40, 100), 'Terms of Sale. This quote is valid for thirty days.')
        doc.save(path)
    with pdfplumber.open(path) as doc:
        assert inventory_table(doc.pages[0]) is None


def test_silent_markdown_omission_uses_native_text(tmp_path: Path, monkeypatch) -> None:
    from ingestion.parse import parse_pdf
    from config import Parsing
    import ingestion.parse as parser
    path = tmp_path / 'terms.pdf'
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((40, 100), 'Terms of Sale. This quotation is valid for thirty days. Payment terms apply.')
        doc.save(path)
    monkeypatch.setattr(parser.pymupdf4llm, 'to_markdown', lambda *a, **k: [{'text': 'Footer'}])
    pages, errors = parse_pdf(path, Parsing())
    assert not errors
    assert 'valid for thirty days' in pages[0][0]
    assert any('coverage fallback' in warning for warning in pages[0][2])


def test_ocr_recovery_and_failure_are_reported(tmp_path: Path, monkeypatch) -> None:
    from ingestion.parse import parse_pdf
    from config import Parsing
    import ingestion.parse as parser
    path = tmp_path / 'outlined.pdf'
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(path)
    monkeypatch.setattr(parser, 'ocr_page', lambda *args: 'Request for Taxpayer Identification Number')
    pages, errors = parse_pdf(path, Parsing(ocr_enabled=True))
    assert not errors
    assert 'Taxpayer Identification' in pages[0][0]
    assert any('Tesseract OCR used' in w for w in pages[0][2])
    def fail(*args):
        raise TimeoutError('OCR timed out')
    monkeypatch.setattr(parser, 'ocr_page', fail)
    pages, errors = parse_pdf(path, Parsing(ocr_enabled=True))
    assert 'OCR failed' in errors[0]
    assert pages[0][0] == ''


def test_word_coverage_detects_missing_labels_despite_long_output() -> None:
    from ingestion.parse import native_word_coverage
    assert native_word_coverage('Name Signature Date', 'Name ' * 100) == pytest.approx(1 / 3)
    assert native_word_coverage('Office affiliation', 'Oﬃce aﬃliation') == 1


def test_omitted_labels_get_native_supplement(tmp_path: Path, monkeypatch) -> None:
    from ingestion.parse import parse_pdf
    from config import Parsing
    import ingestion.parse as parser
    path = tmp_path / 'form.pdf'
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((40, 100), 'Company Name Signature Declaration of Compliance')
        doc.save(path)
    monkeypatch.setattr(parser.pymupdf4llm, 'to_markdown',
                        lambda *a, **k: [{'text': 'Company Name ' * 20}])
    pages, errors = parse_pdf(path, Parsing())
    assert not errors
    assert 'Signature Declaration of Compliance' in pages[0][0]
    assert any('supplement retained' in w for w in pages[0][2])


def test_portal_mentioning_affidavit_or_addendum_stays_bid_page() -> None:
    assert classify_document('portal.html', '# Bid\nRequired mercury affidavit. Addendum 2.', '.html') == ('bid_page', None)


def test_repeated_document_title_is_retained_once_not_erased():
    from ingestion.parse import remove_repeated_margins
    from schemas import ParsedPage, PageMetadata
    pages = [ParsedPage(metadata=PageMetadata(bid_id='new', file_name='rfp.pdf', doc_type='rfp',
        page_number=n), text=f'RFP 2030 Equipment Lease\n\nPage {n} body text\nFooter', tables=[])
        for n in range(1, 5)]
    remove_repeated_margins(pages)
    assert 'RFP 2030 Equipment Lease' in pages[0].text
    assert all('RFP 2030 Equipment Lease' not in p.text for p in pages[1:])
    assert all(f'Page {n} body text' in p.text for n,p in enumerate(pages,1))


def test_footer_only_page_is_not_a_retrieval_chunk_but_numeric_identifier_is():
    from search.chunk import chunk_pages
    from schemas import ParsedPage, PageMetadata
    from config import Chunking
    pages = [ParsedPage(metadata=PageMetadata(bid_id='new', file_name='rfp.pdf', doc_type='rfp',
        page_number=8), text='8', tables=[]), ParsedPage(metadata=PageMetadata(bid_id='new',
        file_name='rfp.pdf', doc_type='rfp', page_number=9), text='7032', tables=[])]
    chunks = chunk_pages(pages, Chunking())
    assert [c.text for c in chunks] == ['7032']
