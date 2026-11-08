# Checkpoint 2 verification

Real local embedding and cross-encoder models used on CPU.

- Main index: 662 chunks across Bid1 and Bid2.
- Repeat indexing: 0 added, 0 removed, 611 unchanged; parsed=False.
- Renamed Bid3_test: 51 chunks in isolated index; processor/SKU search passed.
- Modes vector, hybrid, and hybrid_rerank exercised.
- Unknown-bid filter returned no results.
- Every example citation quote matches its returned passage.

## Sample top results

### deadline_addendum_2

Query: submission deadline extension

Top source: Bid1 / Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf, physical page 1.

Mode: hybrid_rerank; latency: 3112 ms.

### processor_sku

Query: 379-BFNZ processor

Top source: Bid2 / Dell_Laptop_Specs.pdf, physical page 1.

Mode: hybrid_rerank; latency: 1892 ms.

### affidavits

Query: required affidavits

Top source: Bid2 / PORFP_-_Dell_Laptop_Final.pdf, physical page 2.

Mode: hybrid_rerank; latency: 4639 ms.

### warranty_across_bids

Query: warranty requirements support coverage

Top source: Bid1 / JA-207652 Student and Staff Computing Devices FINAL.pdf, physical page 6.

Mode: hybrid_rerank; latency: 2589 ms.

### bid_bond

Query: bid bond amount or percentage

Top source: Bid1 / JA-207652 Student and Staff Computing Devices FINAL.pdf, physical page 21.

Mode: hybrid_rerank; latency: 3998 ms.

### mode_vector

Query: submission deadline

Top source: Bid1 / Student and Staff Computing Devices __SOURCING #168884__ - Bid Information - {3} _ BidNet Direct.html, physical page 1.

Mode: vector; latency: 30 ms.

### mode_hybrid

Query: submission deadline

Top source: Bid1 / JA-207652 Student and Staff Computing Devices FINAL.pdf, physical page 32.

Mode: hybrid; latency: 74 ms.

## Limits

These smoke examples are not the formal Recall@k/MRR evaluation. Scores do not prove factual correctness or completeness. Automatic addendum reconciliation and generated answers remain pending. Full passages are in search_examples.json; structured retrieval traces are in search_trace.jsonl.

## Automated and HTTP verification

- 24 automated tests passed; pip check reported no broken requirements.
- Live HTTP health check: 200.
- Live HTTP SKU search: 200, expected specs file/page returned.
- Live HTTP repeat indexing of Bid2: 200, 0 added/removed, 51 unchanged.
- Reserved Q&A endpoint: 501 as documented.
- CLI --json output parsed successfully as JSON and returned the cited SKU table.
- Temporary HTTP server stopped after verification.

## Subsequent ranking correction

The original broad-deadline example had inadequate ranking. See search_fix_report.md for the corrected, source-asserted regression and the documented dated-deadline ranking policy. Earlier smoke checks did not establish answer accuracy.
