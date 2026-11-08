"""Persist page-aware chunks and incrementally update a bid's vector entries."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
from collections.abc import Iterator
from typing import Any

from config import Settings
from schemas import IndexResult, ParsedPage, SearchChunk
from search.chunk import chunk_pages

INDEX_VERSION = 1


@contextmanager
def index_lock(index_dir: Path, exclusive: bool = False) -> Iterator[None]:
    """Serialize mutations and prevent readers seeing partial multi-file updates."""
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def atomic_write(path: Path, content: str) -> None:
    """Replace a small manifest or JSONL file atomically."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(path)


def read_chunks(index_dir: Path) -> list[SearchChunk]:
    """Load the canonical chunk corpus used to reconstruct BM25."""
    path = index_dir / 'chunks.jsonl'
    if not path.exists():
        return []
    return [SearchChunk.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def source_fingerprint(bid_dir: Path, settings: Settings) -> str:
    """Hash source bytes and parser settings; renamed or edited files invalidate cache."""
    import ingestion.parse as parser
    digest = hashlib.sha256()
    digest.update(json.dumps(settings.parsing.model_dump(), sort_keys=True).encode())
    for module in ('ingestion/parse.py', 'ingestion/ocr.py'):
        digest.update((Path(__file__).resolve().parents[1] / module).read_bytes())
    for path in parser.discover_files(bid_dir):
        digest.update(path.relative_to(bid_dir).as_posix().encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def vector_store(settings: Settings):
    """Open the persistent cosine collection through langchain-chroma."""
    from langchain_chroma import Chroma
    from chromadb.config import Settings as ChromaSettings
    from search.models import embeddings
    return Chroma(
        collection_name='rfp_chunks_v1',
        embedding_function=embeddings(settings.search.embedding_model, str(settings.paths.index)),
        persist_directory=str(settings.paths.index / 'chroma'),
        collection_metadata={'hnsw:space': 'cosine'},
        client_settings=ChromaSettings(anonymized_telemetry=False),
    )


def vector_metadata(chunk: SearchChunk) -> dict[str, Any]:
    """Chroma metadata omits nulls; canonical JSONL retains the exact schema."""
    return {k: v for k, v in chunk.metadata.model_dump().items() if v is not None}


def embedding_revision(settings: Settings) -> str | None:
    """Track downloaded model revisions so updates cannot silently mix vector spaces."""
    registry = settings.paths.index / 'models' / 'registry.json'
    if not registry.exists():
        return None
    return json.loads(registry.read_text()).get(settings.search.embedding_model, {}).get('revision')


def check_manifest(settings: Settings, manifest: dict[str, Any]) -> None:
    """Refuse to mix embedding spaces or chunk settings in one index."""
    expected = {'version': INDEX_VERSION, 'embedding_model': settings.search.embedding_model,
                'chunking': settings.chunking.model_dump(), 'embedding_revision': embedding_revision(settings)}
    for key, value in expected.items():
        if key in manifest and manifest[key] != value:
            raise ValueError(f'Index {key} differs from config; use a fresh index directory')


def add_bid(bid_dir: Path, settings: Settings) -> IndexResult:
    """Add/update one bid; unchanged chunks and other bids are never re-embedded."""
    from ingestion.parse import ingest_bid
    bid_dir = bid_dir.resolve()
    if not bid_dir.is_dir():
        raise ValueError(f'Bid directory does not exist: {bid_dir}')
    bid_id = bid_dir.name
    with index_lock(settings.paths.index, exclusive=True):
        pending = settings.paths.index / '.updating'
        if pending.exists() and pending.read_text() != bid_id:
            raise ValueError(f'First repair interrupted indexing for {pending.read_text()!r}')
        manifest_path = settings.paths.index / 'manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        check_manifest(settings, manifest)
        bids = manifest.get('bids', {})
        previous = bids.get(bid_id, {})
        if previous and previous['source_dir'] != str(bid_dir):
            raise ValueError(f'Bid ID {bid_id!r} already belongs to a different directory')
        fingerprint = source_fingerprint(bid_dir, settings)
        destination = settings.paths.outputs / 'ingestion' / bid_id
        cache = destination / 'pages.jsonl'
        stamp = destination / 'fingerprint.txt'
        parsed = not (cache.exists() and stamp.exists() and stamp.read_text() == fingerprint)
        if parsed:
            report = ingest_bid(bid_dir, destination, settings.parsing)
            if not report.files or any(f.errors for f in report.files):
                raise ValueError(f'Ingestion failed; review {destination / "report.json"}')
            atomic_write(stamp, fingerprint)
        pages = [ParsedPage.model_validate_json(line) for line in cache.read_text().splitlines() if line.strip()]
        chunks = chunk_pages(pages, settings.chunking)
        if not chunks:
            raise ValueError('No searchable content found; existing bid index was retained')
        old = read_chunks(settings.paths.index)
        old_bid = {c.metadata.chunk_id: c for c in old if c.metadata.bid_id == bid_id}
        current = {c.metadata.chunk_id: c for c in chunks}
        added = [c for key, c in current.items() if key not in old_bid]
        removed = sorted(set(old_bid) - set(current))
        # A failed previous update is repaired from this bid's actual Chroma IDs.
        store = vector_store(settings)
        actual_ids = set(store.get(where={'bid_id': bid_id}, include=[])['ids'])
        to_add = [c for key, c in current.items() if key not in actual_ids]
        to_remove = sorted(actual_ids - set(current))
        atomic_write(settings.paths.index / '.updating', bid_id)
        for start in range(0, len(to_add), 64):
            batch = to_add[start:start + 64]
            store.add_texts([c.text for c in batch], metadatas=[vector_metadata(c) for c in batch],
                            ids=[c.metadata.chunk_id for c in batch])
        if to_remove:
            store.delete(ids=to_remove)
        combined = [c for c in old if c.metadata.bid_id != bid_id] + chunks
        atomic_write(settings.paths.index / 'chunks.jsonl', ''.join(c.model_dump_json() + '\n' for c in combined))
        bids[bid_id] = {'source_dir': str(bid_dir), 'fingerprint': fingerprint, 'chunks': len(chunks)}
        manifest.update(version=INDEX_VERSION, embedding_model=settings.search.embedding_model,
                        chunking=settings.chunking.model_dump(), embedding_revision=embedding_revision(settings), bids=bids)
        atomic_write(manifest_path, json.dumps(manifest, indent=2))
        (settings.paths.index / '.updating').unlink()
        warnings = [w for f in json.loads((destination / 'report.json').read_text())['files']
                    for w in f['warnings']]
        return IndexResult(bid_id=bid_id, chunks=len(chunks), added=len(added), removed=len(removed),
                           unchanged=len(current) - len(added), parsed=parsed,
                           warnings=[f'{len(warnings)} ingestion warnings; see {destination / "report.json"}'] if warnings else [])
