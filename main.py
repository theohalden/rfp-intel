"""CLI entry point for ingestion, search, extraction, and cited Q&A."""
import logging
import os

# Keep machine-readable CLI output free of optional-library installation suggestions.
os.environ.setdefault('PYMUPDF_SUGGEST_LAYOUT_ANALYZER', '0')
from pathlib import Path
import typer
from config import ROOT, load_settings
from ingestion.parse import ingest_bid

app = typer.Typer(no_args_is_help=True, help='RFP Intelligence Platform')


@app.command()
def ingest(
    bid: Path = typer.Option(..., '--bid', exists=True, file_okay=False, help='One bid folder'),
    config: Path = typer.Option(ROOT / 'config.yaml', '--config', exists=True, dir_okay=False),
) -> None:
    """Parse one bid and print page, type, table, and failure diagnostics."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    settings = load_settings(config)
    destination = settings.paths.outputs / 'ingestion' / bid.resolve().name
    report = ingest_bid(bid, destination, settings.parsing)
    typer.echo(f'Bid: {report.bid_id}')
    for item in report.files:
        typer.echo(f'{item.file_name}\n  type={item.doc_type} addendum={item.addendum_number} '
                   f'pages={item.page_count} nonempty={item.parsed_pages} tables={item.tables_found} '
                   f'ocr_pages={item.ocr_pages}')
        for warning in item.warnings:
            typer.echo(f'  WARNING: {warning}')
        for error in item.errors:
            typer.echo(f'  ERROR: {error}')
    typer.echo(f'Reports and normalized pages: {destination}')
    if not report.files or any(item.errors for item in report.files):
        raise typer.Exit(code=1)


@app.command('index')
def index_command(
    bid: Path = typer.Option(..., '--bid', exists=True, file_okay=False),
    config: Path = typer.Option(ROOT / 'config.yaml', '--config', exists=True, dir_okay=False),
) -> None:
    """Incrementally parse, chunk, and index one bid folder."""
    from search.index import add_bid
    try:
        result = add_bid(bid, load_settings(config))
        typer.echo(result.model_dump_json(indent=2))
    except Exception as exc:
        logging.exception('Indexing failed')
        typer.echo(f'Indexing failed: {exc}', err=True)
        raise typer.Exit(1) from exc


@app.command('search')
def search_command(
    query: str = typer.Argument(..., help='Question or keywords to search'),
    bid_id: str | None = typer.Option(None, '--bid-id'),
    doc_type: str | None = typer.Option(None, '--doc-type'),
    addendum_number: int | None = typer.Option(None, '--addendum-number', min=1),
    file_name: str | None = typer.Option(None, '--file-name'),
    page_number: int | None = typer.Option(None, '--page-number', min=1),
    exclude_addenda: bool = typer.Option(False, '--exclude-addenda'),
    top_k: int = typer.Option(3, '--top-k', min=1, max=50),
    mode: str = typer.Option('hybrid_rerank', '--mode', help='vector, hybrid, or hybrid_rerank'),
    expand: bool = typer.Option(True, '--expand/--no-expand'),
    as_json: bool = typer.Option(False, '--json', help='Output complete structured results'),
    full_text: bool = typer.Option(False, '--full-text', help='Show complete passages instead of previews'),
    details: bool = typer.Option(False, '--details', help='Show retrieval scores, timing, and query rewriting'),
    config: Path = typer.Option(ROOT / 'config.yaml', '--config', exists=True, dir_okay=False),
) -> None:
    """Retrieve source passages with citations (not an AI-generated answer)."""
    from schemas import SearchFilters
    from search.retrieve import SearchEngine
    try:
        filters = SearchFilters(bid_id=bid_id, doc_type=doc_type, addendum_number=addendum_number,
                                file_name=file_name, page_number=page_number, exclude_addenda=exclude_addenda)
        response = SearchEngine(load_settings(config)).search(query, filters, top_k, mode, expand)
        if as_json:
            typer.echo(response.model_dump_json(indent=2))
            return
        from search.display import format_search
        typer.echo(format_search(response, full_text, details))
    except Exception as exc:
        logging.exception('Search failed')
        typer.echo(f'Search failed: {exc}', err=True)
        raise typer.Exit(1) from exc


@app.command('ask')
def ask_command(
    question: str = typer.Argument(...),
    bid_id: str | None = typer.Option(None, '--bid-id'),
    doc_type: str | None = typer.Option(None, '--doc-type'),
    as_json: bool = typer.Option(False, '--json'),
    config: Path = typer.Option(ROOT / 'config.yaml', '--config', exists=True, dir_okay=False),
) -> None:
    """Generate a guarded answer with exact file/page citations."""
    from agents.qa import QAEngine
    from schemas import SearchFilters
    try:
        filters = SearchFilters(bid_id=bid_id, doc_type=doc_type)
        response = QAEngine(load_settings(config)).ask(question, filters.bid_id, filters.doc_type)
        if as_json:
            typer.echo(response.model_dump_json(indent=2))
            return
        typer.echo(response.answer)
        if response.notes:
            typer.echo(f'Note: {response.notes}')
        typer.echo(f'\nTrace: {response.trace_file}')
    except Exception as exc:
        typer.echo(f'Q&A failed ({type(exc).__name__}); check model settings and local traces.', err=True)
        raise typer.Exit(1) from exc


@app.command('extract')
def extract_command(
    bid: Path = typer.Option(..., '--bid', exists=True, file_okay=False),
    config: Path = typer.Option(ROOT / 'config.yaml', '--config', exists=True, dir_okay=False),
) -> None:
    """Extract the 20 required fields using the validated LangGraph workflow."""
    from agents.graph import extract_bid
    try:
        settings = load_settings(config)
        record = extract_bid(bid, settings, progress=lambda message: typer.echo(message, err=True))
        typer.echo(f'Extraction complete: {record.bid_id}')
        typer.echo(f'Validated: {record.validation.passed} | Not found: {record.validation.not_found} | '
                   f'Failed: {record.validation.failed}')
        from agents.guards import UNKNOWN_ZONE_NOTE
        if UNKNOWN_ZONE_NOTE in record.fields['Due Date'].notes:
            typer.echo('WARNING: Deadline date/time retained; source time zone is not stated. Review before bidding.')
        typer.echo(f'JSON: {settings.paths.outputs / (record.bid_id + ".json")}')
        typer.echo(f'Trace: {settings.paths.outputs / (record.bid_id + ".trace.jsonl")}')
        if record.validation.failed:
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as exc:
        # Keep credential/provider payloads out of terminal error dumps.
        typer.echo(f'Extraction failed ({type(exc).__name__}). Check model settings and the trace.', err=True)
        raise typer.Exit(1) from exc


@app.command()
def serve(
    host: str = typer.Option('127.0.0.1', '--host'),
    port: int = typer.Option(8000, '--port', min=1, max=65535),
) -> None:
    """Start the local FastAPI server; see /docs for the interactive API."""
    import uvicorn
    uvicorn.run('search.api:app', host=host, port=port)


@app.callback()
def cli() -> None:
    """Expose project commands."""


if __name__ == '__main__':
    app()
