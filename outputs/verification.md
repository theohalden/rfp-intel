# Final verification

Live OpenAI extraction and Q&A outputs; exact citations audited against canonical indexed chunks.

| Bid | Validated | Not found | Failed | Verified field citations | Trace steps |
|---|---:|---:|---:|---:|---:|
| Bid1 | 15 | 5 | 0 | 47 | 166 |
| Bid2 | 16 | 4 | 0 | 40 | 70 |

Q&A examples: 10; explicit missing answers: 1.
The affidavit answer and cross-bid warranty comparison have verified supporting citations; the comparison includes both bids.

All populated fields and Q&A sources match exact whitespace-normalized source text on the cited bid/file/physical page. Both records have exactly 20 fields and a generated summary. Per-bid extraction traces end with the final report.

## Other checks

- 91 offline tests pass; no broken dependency requirements.
- Retrieval evaluation: 24 quote-anchored development questions; reranked Recall@5 0.938 and MRR@5 0.847.
- Renamed Bid3_test copy: 51 chunks indexed/searched without runtime code changes (filename independence).
- Three independent public RFPs tested; preserved first runs and development retests are compared in outputs/public_rfp_review.md. Workflow validation is not an accuracy score.
- Configured API-key text excluded from source/deliverable artifacts; package excludes .env, data, weights, index, and .venv.

## Limits

Validation counts describe the workflow verdict, not measured extraction accuracy. Exact provenance does not prove semantic correctness or completeness. Conservative nulls, missing context, model decisions, OCR, and table layouts still need human source review. Retrieval evaluation uses a development set, not held-out data.

Live presentation is the selected demo format; recording is not planned. Publishing a repository URL and presenting the live session remain submission steps. Public-document source review and remaining limitations are documented in outputs/public_rfp_review.md; no universal unseen-document accuracy is claimed.
