# Targeted validator regressions

The final source-support/classification/completeness checks passed all 12 prepared cases: 11 use the configured live model and one is rejected by the deterministic part-identifier guard. These are targeted regressions, not an extraction accuracy score. Expected verdicts are not sent to the model.

The source-support stage sees only selected quotations; a separate field-specific stage receives retrieved context to check completeness. The positive submission fixture initially lacked an Exhibit A quotation, so its rejection was correct. That fixture was corrected by adding the exact retrieved source span; the original fixture and results are retained in before_submission_citation_* files. No runtime or expected verdict was changed to force a pass.

See cases.json, results.json and validator_suite.trace.jsonl for inputs, verdicts and traces. Run `python scripts/check_validator.py` to reproduce (paid API calls). Model judgments can vary between runs.
