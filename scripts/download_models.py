"""Download only the configured local search model files into the project cache."""
from pathlib import Path
import sys
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import load_settings
from search.models import configure_model_cache


def main() -> None:
    """Prefer safetensors weights; avoid duplicate ONNX/PyTorch downloads."""
    settings = load_settings()
    cache = configure_model_cache(settings.paths.index)
    from huggingface_hub import model_info, snapshot_download
    registry_path = cache / 'registry.json'
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    for name in (settings.search.embedding_model, settings.search.reranker_model):
        info = model_info(name)
        names = [s.rfilename for s in info.siblings]
        weights = 'model.safetensors' if 'model.safetensors' in names else 'pytorch_model.bin'
        print(f'Downloading {name}: {weights}', flush=True)
        snapshot = snapshot_download(name, revision=info.sha, cache_dir=str(cache),
                          allow_patterns=['*.json', '*.txt', weights],
                          ignore_patterns=['onnx/*', 'openvino/*', '*.onnx'], max_workers=4)
        registry[name] = {'path': snapshot, 'revision': info.sha}
        registry_path.write_text(json.dumps(registry, indent=2))
        print(f'Ready: {name}', flush=True)


if __name__ == '__main__':
    main()
