"""Model setup errors must be explicit and local snapshots must be reproducible."""
from pathlib import Path
import json
import pytest
from search.models import cached_model_path


def test_missing_models_explain_setup(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match='download_models.py'):
        cached_model_path('example/model', tmp_path)


def test_registry_resolves_exact_local_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    (tmp_path / 'registry.json').write_text(json.dumps({'example/model': {'path': str(snapshot), 'revision': 'abc'}}))
    assert cached_model_path('example/model', tmp_path) == str(snapshot)
