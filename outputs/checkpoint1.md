# Checkpoint 1: ingestion verification (revised)

Includes source review, table alignment fixes, native-text coverage fallback, and approved optional Tesseract OCR.

| Bid | File | Type | Addendum | Pages | Nonempty | Tables | OCR pages | Errors |
|---|---|---|---|---:|---:|---:|---:|---:|
| Bid1 | Addendum 1 RFP JA-207652 Student and Staff Computing Devices.pdf | addendum | 1 | 5 | 5 | 0 | 0 | 0 |
| Bid1 | Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf | addendum | 2 | 1 | 1 | 0 | 0 | 0 |
| Bid1 | JA-207652 Student and Staff Computing Devices FINAL.pdf | rfp | — | 62 | 62 | 82 | 6 | 0 |
| Bid1 | Student and Staff Computing Devices __SOURCING #168884__ - Bid Information - {3} _ BidNet Direct.html | bid_page | — | 1 | 1 | 0 | 0 | 0 |
| Bid2 | Contract_Affidavit.pdf | affidavit | — | 3 | 3 | 1 | 0 | 0 |
| Bid2 | Dell Laptops w_Extended Warranty - Bid Information - {3} _ BidNet Direct.html | bid_page | — | 1 | 1 | 0 | 0 | 0 |
| Bid2 | Dell_Laptop_Specs.pdf | specs | — | 3 | 3 | 2 | 0 | 0 |
| Bid2 | Mercury_Affidavit.pdf | affidavit | — | 1 | 1 | 0 | 0 | 0 |
| Bid2 | PORFP_-_Dell_Laptop_Final.pdf | rfp | — | 4 | 4 | 4 | 0 | 0 |
| Bid3_test | document_01.pdf | affidavit | — | 3 | 3 | 1 | 0 | 0 |
| Bid3_test | document_02.html | bid_page | — | 1 | 1 | 0 | 0 | 0 |
| Bid3_test | document_03.pdf | specs | — | 3 | 3 | 2 | 0 | 0 |
| Bid3_test | document_04.pdf | affidavit | — | 1 | 1 | 0 | 0 | 0 |
| Bid3_test | document_05.pdf | rfp | — | 4 | 4 | 4 | 0 | 0 |

## Source review and corrections

- Visually inspected Bid1 physical pages 54–59. They contain the W-9 form and instructions rendered as vector outlines, with almost no native text. All six now have OCR output, labelled in metadata and warnings.
- Visually inspected all three Dell specs pages. Recovered the first two pages as description/code tables (29 and 9 rows); the final terms page is prose.
- Added a native-text coverage fallback: the markdown parser had silently omitted most of the Dell terms page. The terms text is now retained.
- Spot-checked Bid1 physical pages 2 and 48: the source includes bordered requirements/form regions. Fallback table counts may include small form blocks and should not be interpreted as unique logical tables.

## Verification

- 13 automated tests pass, including wrapped inventory descriptions, rejection of prose as inventory, silent markdown omissions, and OCR recovery/failure handling.
- Bid3_test retains identical text and document types across all 12 pages after filenames are changed. This checks ingestion only.
- No recorded parsing/OCR errors in either bid.

## Remaining limitations

OCR can misrecognize characters and reading order; it does not preserve W-9 table structure. Borderless recovery supports conservative two-column code layouts, not arbitrary tables. Bordered fallback tables may duplicate prose. Coverage checking is a heuristic, not proof of completeness. Search and extraction are not implemented yet.

## Additional word-retention guard

A native-word coverage check now adds a labelled supplement below 97% retention. See coverage_audit.md for the measured audit. Supplements can duplicate content; OCR accuracy is not measured by this comparison.
