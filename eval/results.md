# Retrieval evaluation

Evaluated 24 manually labelled questions over both bids.

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 | Errors |
|---|---:|---:|---:|---:|---:|
| vector | 0.646 | 0.688 | 0.771 | 0.699 | 0 |
| hybrid | 0.604 | 0.812 | 0.812 | 0.722 | 0 |
| hybrid_rerank | 0.750 | 0.938 | 0.938 | 0.847 | 0 |

Index coverage note: b1_prebid has a gold anchor that is not wholly contained in one indexed chunk. These targets remain in the denominator and count as misses under this strict passage metric.

## Method

- Same indexed corpus, query expansion enabled, metadata filters, and top-5 budget for all modes.
- A relevant hit must contain the annotated quote anchor on the correct bid/file/physical page. A filename/page match alone does not count.
- Recall@k is the fraction of required evidence targets covered in the first k hits, averaged equally across questions. Alternative sources for one target count once.
- MRR@5 is the mean reciprocal rank of the first relevant hit within five results. It does not measure whether every target in a multi-source question was found.
- The cross-bid warranty question requires evidence from both bids. Multiple chunks from one source do not inflate recall.
- hybrid_rerank includes the documented dated-deadline evidence ordering and same-page passage deduplication after cross-encoder scoring.
- Failed requests receive zero metrics and are retained in the denominator, with errors saved in results.json.

## Limitations

- This is a small development set, not a held-out benchmark. The previously debugged deadline and SKU cases are explicitly tagged regression cases.
- The source judgments are manually selected and not exhaustive. An unannotated but valid source can be counted as a miss.
- These scores measure retrieval of supporting passages, not generated-answer correctness, complete extraction, or reliable conflict resolution.
- OCR anchors are checked against saved OCR text; this is not a measurement of character recognition accuracy.
- Results depend on the frozen corpus, local model revisions, chunking, expansion, and retrieval policies recorded in results.json.

## Misses at five results

- vector / b1_deadline: Recall@5=0.00; submission deadline
- vector / b1_prebid: Recall@5=0.00; When and how is the pre-proposal meeting held?
- vector / b2_submission: Recall@5=0.00; Which system must be used to submit a proposal, and are email bids accepted?
- vector / b2_mercury_required: Recall@5=0.00; Which mercury-related affidavit must the contractor provide?
- vector / b2_warranty: Recall@5=0.00; What extended hardware warranty is required for the Dell laptops?
- vector / cross_warranty: Recall@5=0.50; Compare the warranty period for student Chromebooks and the Dell laptops.
- hybrid / b1_deadline: Recall@5=0.00; submission deadline
- hybrid / b1_prebid: Recall@5=0.00; When and how is the pre-proposal meeting held?
- hybrid / b2_submission: Recall@5=0.00; Which system must be used to submit a proposal, and are email bids accepted?
- hybrid / b2_warranty: Recall@5=0.00; What extended hardware warranty is required for the Dell laptops?
- hybrid / cross_warranty: Recall@5=0.50; Compare the warranty period for student Chromebooks and the Dell laptops.
- hybrid_rerank / b1_prebid: Recall@5=0.00; When and how is the pre-proposal meeting held?
- hybrid_rerank / cross_warranty: Recall@5=0.50; Compare the warranty period for student Chromebooks and the Dell laptops.
