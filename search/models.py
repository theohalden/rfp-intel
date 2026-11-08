"""Lazy local model loading; indexing and search do not use Anthropic."""
from __future__ import annotations

from functools import lru_cache
import os
import json
import logging
from pathlib import Path


def configure_model_cache(index_dir: Path) -> Path:
    """Keep downloaded model weights inside the project, disable telemetry."""
    cache = index_dir / 'models'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('HF_HOME', str(cache))
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    os.environ.setdefault('ANONYMIZED_TELEMETRY', 'False')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    return cache


class _PlatformNoticeFilter(logging.Filter):
    """Hide one irrelevant distributed-training notice, leaving real warnings visible."""
    def filter(self, record: logging.LogRecord) -> bool:
        return 'Redirects are currently not supported in MacOs' not in record.getMessage()


def prepare_inference() -> None:
    """Disable weight progress bars and a known unrelated training-platform notice."""
    logging.getLogger('torch.distributed.elastic.multiprocessing.redirects').addFilter(_PlatformNoticeFilter())
    from transformers.utils import logging as transformer_logging
    transformer_logging.disable_progress_bar()


def cached_model_path(model_name: str, cache: Path) -> str:
    """Resolve an explicitly downloaded snapshot without network lookups."""
    registry = cache / 'registry.json'
    if registry.exists():
        entry = json.loads(registry.read_text()).get(model_name)
        if entry and Path(entry['path']).is_dir():
            return entry['path']
    raise FileNotFoundError(f'Model {model_name} is not ready. Run python scripts/download_models.py')


@lru_cache(maxsize=4)
def embeddings(model_name: str, index_dir: str):
    """Cache the embedding model in-process and normalize vectors for cosine search."""
    cache = configure_model_cache(Path(index_dir))
    prepare_inference()
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=cached_model_path(model_name, cache),
        model_kwargs={'device': 'cpu', 'local_files_only': True}, encode_kwargs={'normalize_embeddings': True},
    )


@lru_cache(maxsize=2)
def reranker(model_name: str, index_dir: str):
    """Load the cross-encoder only for searches that request reranking."""
    cache = configure_model_cache(Path(index_dir))
    prepare_inference()
    from sentence_transformers import CrossEncoder
    return CrossEncoder(cached_model_path(model_name, cache), device='cpu', max_length=512, local_files_only=True)
