# Search ranking and readability regression

Verified using the actual local embedding and cross-encoder models.

- Unfiltered Bid1 submission deadline search: Addendum 2 is first.
- Addendum 2 change search: one substantive source passage; signature overlap removed.
- Bid2 processor/SKU search: specifications page 1 is first.
- No source values, filenames, or bid IDs are hardcoded in runtime ranking rules.
- CLI defaults to three preview results; --full-text and --json retain complete evidence.

## Actual readable output

```text
Search results for: submission deadline
3 source passage(s). These are document excerpts, not a generated answer.
Showing previews; add --full-text for complete passages or --json for structured results.
Dated deadline amendments are prioritized; base documents may show earlier dates.

[1] Bid1 · Addendum 2 · page 1
Source: Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf

…
The new due date for this RFP will be July 9, 2024 at 2:00 PM CST.
…

[2] Bid1 · Bid Page · page 1
Source: Student and Staff Computing Devices __SOURCING #168884__ - Bid Information - {3} _ BidNet Direct.html

…
Closing Date
07/09/2024 03:00 PM EDT
…

[3] Bid1 · RFP · page 2
Source: JA-207652 Student and Staff Computing Devices FINAL.pdf

…
Solicitation Due 27-JUN-2024 14:00:00
…
```

```text
Search results for: what changed
1 source passage(s). These are document excerpts, not a generated answer.
Showing previews; add --full-text for complete passages or --json for structured results.

[1] Bid1 · Addendum 2 · page 1
Source: Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf

ADDENDUM No. 2
RFP JA-207652 Student and Staff Computing Devices
The Purpose of this Addendum is to extend the due date of this RFP.
The new due date for this RFP will be July 9, 2024 at 2:00 PM CST.
…
```

```text
Search results for: 379-BFNZ processor
3 source passage(s). These are document excerpts, not a generated answer.
Showing previews; add --full-text for complete passages or --json for structured results.

[1] Bid2 · Specs · page 1
Source: Dell_Laptop_Specs.pdf

…
| Dell Latitude 5550 XCTO Base | 210-BLYZ |
| Intel Core Ultra 5 125U processor (12 MB cache, 12 cores, 14 threads, up to 4.3 GHz Turbo) | 379-BFNZ |
| Windows 11 Pro, English, Brazilian Portuguese PT-BR, French, Spanish | 619-ARSB |
| No Microsoft Office License Included - 30 day Trial Offer Only | 658-BCSB |
…

[2] Bid2 · Specs · page 1
Source: Dell_Laptop_Specs.pdf

…
210-BLYZDell Latitude 5550 XCTO Base
Intel Core Ultra 5 125U processor (12 MB cache, 12 cores, 14 379-BFNZ
threads, up to 4.3 GHz Turbo)
Windows 11 Pro, English, Brazilian Portuguese PT-BR, French, 619-ARSB
…

[3] Bid2 · Specs · page 1
Source: Dell_Laptop_Specs.pdf

…
634-BRWGWindows AutoPilot
379-BDZBEPEAT 2018 Registered (Gold)
```

## Scope and limitations

Current submission-deadline queries use a documented evidence-ranking policy after reranking: concrete dated deadline statements first; explicit numbered deadline revisions ahead of base statements. A newer unrelated addendum receives no boost. Historical and other milestone intents bypass this policy. This is not full extraction or semantic conflict resolution; conditional wording and conflicting time zones still require review. Model scores are not confidence values.

31 automated tests passed before this real-model regression. JSON output retains the rewritten reranker query and the full source passages.
