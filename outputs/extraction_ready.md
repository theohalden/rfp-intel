# Extraction and Q&A completed

Live OpenAI extraction and Q&A verification is complete. See verification.md and verification.json for the audited results, and Bid1.json / Bid2.json for the 20-field records.

The LangGraph workflow uses four parallel specialists, separate per-field/per-scope retrieval, structured source-line selections, code-verified exact citations, numbered amendment applicability review, isolated semantic checks, stable reviews for unchanged fields, and at most two retries per rejected field. Final summaries are generated and reviewed separately.

The ReAct Q&A tool enforces user filters, retrieves each relevant bid for comparisons, checks exact source spans, reviews claims individually, and requires evidence from both bids for cross-bid answers. Unsupported facts are omitted or reported as Not found in documents.

91 offline tests pass. Known limitations and submission steps are listed in README.md and verification.md. Confidence is uncalibrated; provenance and workflow validation do not establish perfect semantic accuracy.

## Independent public-document follow-up

Three public RFPs were tested using isolated indexes and expectations saved before their first live run. General fixes address short section/table retrieval, related equipment evidence, title preservation, deadline formats and explicit unknown zones. Preserved first runs and later development retests are compared in public_rfp_review.md. Source review remains necessary. The original supplied-bid outputs pass their audits against the preserved index; the 24-question retrieval evaluation was rerun with unchanged metrics. Live presentation is selected; no recording is planned.
