"""Create a source/deliverable ZIP without secrets, inputs, weights, or environments."""
from pathlib import Path
import hashlib
import json
import zipfile
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    root_files = ['.env.example', '.gitignore', 'README.md', 'DEMO.md', 'LICENSE',
                  'config.py', 'config.yaml', 'main.py', 'schemas.py',
                  'requirements.txt', 'requirements.lock.txt', 'data/.gitkeep', 'index/.gitkeep']
    outputs = ['Bid1.json', 'Bid2.json', 'Bid1.trace.jsonl', 'Bid2.trace.jsonl',
               'trace.jsonl', 'qa_log.md', 'qa_log.json', 'verification.json',
               'verification.md', 'extraction_graph.mmd']
    files = [ROOT / name for name in root_files] + [ROOT / 'outputs' / name for name in outputs]
    for directory in ['agents', 'search', 'ingestion', 'tests', 'scripts', 'eval']:
        files.extend(path for path in (ROOT / directory).rglob('*') if path.is_file()
                     and path.suffix in {'.py', '.md', '.json', '.yaml'} and '__pycache__' not in path.parts)
    files.extend(ROOT / 'outputs' / 'ingestion' / bid / 'report.json' for bid in ['Bid1', 'Bid2'])
    for item in json.loads((ROOT / 'outputs' / 'qa_log.json').read_text()):
        files.append(ROOT / 'outputs' / 'qa_traces' / Path(item['trace_file']).name)
    for name in ['public_lca_test', 'public_cvsu_test', 'public_beaumont_test',
                 'public_lca_reviewed', 'public_cvsu_reviewed', 'public_beaumont_reviewed',
                 'public_beaumont_final',
                 'public_lca_complete', 'public_cvsu_complete', 'public_beaumont_complete',
                 'public_lca_verified', 'public_cvsu_verified', 'public_beaumont_verified',
                 'public_lca_corrected', 'public_cvsu_corrected', 'public_beaumont_corrected',
                 'public_cvsu_narrowed', 'public_beaumont_narrowed']:
        public = ROOT / 'outputs' / name
        if not ((public / 'review.json').exists() or name.endswith('_narrowed')):
            continue
        files.extend(path for path in public.iterdir() if path.is_file()
                     and path.suffix in {'.json', '.jsonl', '.md', '.zip'})
        files.extend(public.glob('ingestion/*/report.json'))
        qa_file = public / 'qa_examples.json'
        if qa_file.exists():
            for item in json.loads(qa_file.read_text()):
                files.append(public / 'qa_traces' / Path(item['trace_file']).name)
    if (ROOT / 'outputs/public_rfp_review.md').exists():
        files.append(ROOT / 'outputs/public_rfp_review.md')
    files.extend(path for path in (ROOT / 'outputs' / 'validator_checks').iterdir()
                 if path.is_file() and path.suffix in {'.json', '.jsonl', '.md'})
    keys = [value for name, value in dotenv_values(ROOT / '.env').items() if 'KEY' in name and value]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f'Missing deliverable: {path.relative_to(ROOT)}')
        if any(key.encode() in path.read_bytes() for key in keys):
            raise ValueError(f'Credential text detected in {path.relative_to(ROOT)}')
    destination = ROOT.parent / 'rfp-intel-submission.zip'
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, 'rfp-intel/' + str(path.relative_to(ROOT)))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert all('/.env' not in name or name.endswith('/.env.example') for name in archive.namelist())
    print(f'{destination}\n{len(set(files))} files; {destination.stat().st_size:,} bytes')
    print('SHA-256: ' + hashlib.sha256(destination.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
