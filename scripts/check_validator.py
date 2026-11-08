"""Run paid, source-backed semantic validator regressions without loading search models."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents import prompts
from agents.graph import ExtractionWorkflow
from agents.guards import normalize_quote, part_identifier_evidence
from config import load_settings
from schemas import SupportReview


def main():
    out = ROOT / 'outputs/validator_checks'
    cases = json.loads((out / 'cases.json').read_text())['cases']
    settings = load_settings(); settings.paths.outputs = out
    run = ExtractionWorkflow(settings, 'validator_suite', tool=SimpleNamespace(addenda=[]))

    def check(case):
        payload = case['payload']
        field = payload['field_name']
        instruction = prompts.FIELD_REVIEW + '\nField definition: ' + prompts.FIELD_NOTES.get(field, prompts.FIELD_QUERIES[field])
        instruction += '\nCoverage checklist for THIS field only: ' + prompts.COVERAGE_CHECKS[field]
        # Match the production gate before semantic review. This is especially
        # important for model-only prose being proposed as a separate part number.
        supporting_texts = []
        for passage in payload.get('retrieved_context', []):
            text = '\n'.join(line['text'] for line in passage['numbered_lines'])
            if any(s['file'] == passage['file'] and s['page'] == passage['page']
                   and normalize_quote(s['quote']) in normalize_quote(text) for s in payload.get('sources', [])):
                supporting_texts.append(text)
        rejected_by_guard = field == 'Part_no' and payload.get('value') is not None and not part_identifier_evidence(supporting_texts)
        stage = 'identifier_guard' if rejected_by_guard else 'live_semantic_review'
        verdict = SupportReview(supported=False, reason='Identifier-evidence guard rejects prose-only model/compatibility codes.') if rejected_by_guard else None
        if not rejected_by_guard and payload.get('value') is not None:
            source_payload = {k: v for k, v in payload.items() if k != 'retrieved_context'}
            verdict = run.invoke_llm(case['case'] + '.sources', SupportReview, prompts.SOURCE_SUPPORT, source_payload)
            stage = 'source_support'
        if not rejected_by_guard and (payload.get('value') is None or verdict is not None and verdict.supported):
            verdict = run.invoke_llm(case['case'], SupportReview, instruction, payload)
            stage = 'live_semantic_review'
        passed = verdict is not None and verdict.supported == case['expected_supported']
        print(case['case'], 'PASS' if passed else 'FAIL', flush=True)
        return {'case': case['case'], 'expected_supported': case['expected_supported'],
                'actual': verdict.model_dump() if verdict else None, 'passed': passed,
                'stage': stage}

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(check, cases))
    report = {'method': 'Targeted live validator regressions, not extraction-accuracy measurement.',
              'model': run.llm.model_name, 'reasoning': run.llm.reasoning, 'cases': results}
    (out / 'results.json').write_text(json.dumps(report, indent=2))
    assert all(r['passed'] for r in results), 'Validator regression failed; inspect results.json'
    print(f'PASS: {len(results)} live validator checks.', flush=True)


if __name__ == '__main__':
    main()
