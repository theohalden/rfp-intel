"""Read validated configuration without adding a YAML parser dependency."""
from pathlib import Path
import json
import os
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parent


class SettingsModel(BaseModel):
    """Reject misspelled configuration keys."""
    model_config = ConfigDict(extra='forbid')


class Paths(SettingsModel):
    data: Path = Path('data')
    index: Path = Path('index')
    outputs: Path = Path('outputs')


class Parsing(SettingsModel):
    table_strategy: str = 'lines_strict'
    remove_repeated_margins: bool = True
    ocr_enabled: bool = False
    ocr_executable: str = 'tesseract'
    ocr_language: str = 'eng'
    ocr_dpi: int = Field(default=300, ge=150, le=600)
    ocr_min_native_chars: int = Field(default=30, ge=0)
    ocr_timeout_seconds: int = Field(default=60, gt=0)


class Chunking(SettingsModel):
    size: int = Field(default=800, gt=0)
    overlap: int = Field(default=100, ge=0)

    @model_validator(mode='after')
    def check_overlap(self) -> 'Chunking':
        """Overlap must leave room for new text."""
        if self.overlap >= self.size:
            raise ValueError('Chunk overlap must be smaller than size')
        return self


class Search(SettingsModel):
    embedding_model: str = 'BAAI/bge-small-en-v1.5'
    reranker_model: str = 'BAAI/bge-reranker-base'
    candidate_k: int = Field(default=30, gt=0)
    top_k: int = Field(default=5, gt=0)


class Agents(SettingsModel):
    max_retries: int = Field(default=2, ge=0, le=2)
    timeout_seconds: int = Field(default=60, gt=0)
    max_evidence_chars: int = Field(default=32000, gt=0)  # per extraction call, ~8k tokens


class Settings(SettingsModel):
    paths: Paths = Field(default_factory=Paths)
    parsing: Parsing = Field(default_factory=Parsing)
    chunking: Chunking = Field(default_factory=Chunking)
    search: Search = Field(default_factory=Search)
    agents: Agents = Field(default_factory=Agents)


def load_settings(path: Path = ROOT / 'config.yaml') -> Settings:
    """Load JSON-compatible YAML, resolving paths relative to its location."""
    load_dotenv(ROOT / '.env', override=False)
    settings = Settings.model_validate(json.loads(path.read_text(encoding='utf-8')))
    for name in ('data', 'index', 'outputs'):
        value = getattr(settings.paths, name)
        if not value.is_absolute():
            setattr(settings.paths, name, (path.resolve().parent / value).resolve())
    return settings


def llm_environment() -> tuple[str, str]:
    """Read the selected provider's credentials only when an agent needs them."""
    load_dotenv(ROOT / '.env', override=False)
    provider = os.getenv('LLM_PROVIDER', 'openai').strip().lower()
    prefixes = {'openai': 'OPENAI', 'anthropic': 'ANTHROPIC'}
    if provider not in prefixes:
        raise ValueError('LLM_PROVIDER must be openai or anthropic; a local Ollama adapter is not implemented yet')
    prefix = prefixes[provider]
    key, model = os.getenv(f'{prefix}_API_KEY', ''), os.getenv(f'{prefix}_MODEL', '')
    if not key or not model:
        raise ValueError(f'Set {prefix}_API_KEY and {prefix}_MODEL in .env')
    return key, model
