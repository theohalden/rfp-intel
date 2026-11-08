# Native-text retention audit

Generated after adding word-coverage supplementation. This measures word-count retention, not semantic correctness, table alignment, or OCR accuracy. OCR pages are excluded because they have no useful native text to compare.

| Bid | Native word retention | PDF pages below 95% | OCR pages |
|---|---:|---:|---:|
| Bid1 | 99.54% | 1 | 6 |

### Bid1 notes

Pages supplemented: 8.
- Addendum 1 RFP JA-207652 Student and Staff Computing Devices.pdf, page 5: 89.0%
Remaining low-retention pages need interpretation: conservative repeated-margin cleaning can remove repeated headings. Counts are calculated after cleaning.

| Bid2 | 99.65% | 0 | 0 |

### Bid2 notes

Pages supplemented: 2.

## Interpretation

All files completed without recorded parser/OCR errors. Nonempty pages alone do not establish extraction completeness. Native supplements may repeat content; they preserve words omitted by markdown conversion. OCR retains review warnings and may misrecognize characters.

13 automated tests pass. Both 12-page Bid2 and renamed Bid3_test have identical saved text.
