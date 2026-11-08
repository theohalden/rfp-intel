# Public RFP testing: final development results

The final runs are complete. The main retrieval gaps were repaired, but **two fields remain unresolved**. This platform processes new RFP folders; these tests do not establish reliable unattended extraction for every RFP.

| Final development retest | Populated | Missing | Failed | Remaining issue |
|---|---:|---:|---:|---|
| [LCA](public_lca_corrected/review.md) | 12 | 8 | 0 | Deadline zone absent in source; payment field is a preparation-expense disclaimer, not invoice timing |
| [CVSU](public_cvsu_corrected/review.md) | 13 | 6 | 1 | Installation rejected because selected quotes did not support all delivery claims |
| [Beaumont ISD](public_beaumont_corrected/review.md) | 15 | 4 | 1 | Required-document list rejected because selected quotes did not support an insurance claim |

Counts describe workflow verdicts, not semantic accuracy. Rejected fields are null with validation-failure explanations, and the command exits nonzero. They must not be interpreted as requirements absent from the source. In particular CVSU has assembly/demonstration/delivery requirements, and Beaumont has required forms and conditional disclosures.

## Recovered facts

- LCA: physical delivery, original plus four copies, separate technical/financial packaging, outer markings/address, all four quantities and detailed category specifications, banker reference and manufacturer authorization alternatives. Exact commercial model and SKU remain absent.
- CVSU: Eastern Time deadline, original plus two copies/address, three-year lease and $1-per-unit end purchase, annual/biannual payment options, commercial model plus four distinct product codes, 1231 units with non-guaranteed-volume caveat, detailed wireless/display/warranty specifications and documentation timing.
- Beaumont: one-year term plus four optional renewals, three copies/address/no email, actual commercial model, correct missing separate SKU, unit pricing versus actual quantities, detailed specifications, Whiteglove/case services, Net 30, procurement contact and optional cooperative adoption.

Read each linked review beside its JSON and original source. Exact citation audits verified 41 LCA, 49 CVSU and the count recorded in Beaumont’s review; exact provenance does not establish interpretation or completeness. Source URLs and hashes are in each source_manifest.json. These historical RFPs do not establish current bidding terms.

## Implementation and evidence

The existing LangGraph workflow now retains evidence during repair, adds bounded canonical same-page neighboring passages, shares equipment passages within amendment scope, reviews missing fields against field-specific checklists, and separates selected-quote support from completeness. A conservative identifier guard rejects prose-only commercial/compatibility models as separate part numbers. Retry limits remain two. No new service, agent node or dependency was added.

The tested gpt-5.4-mini adapter requests medium reasoning by default. Runtime snapshots, hashes and complete traces are preserved. The original Bid1/Bid2 canonical index is unchanged. Expected source facts were written before the first live runs and never included in runtime prompts.

[First LCA run](public_lca_test/review.md), [first CVSU run](public_cvsu_test/review.md), and [first Beaumont run](public_beaumont_test/review.md) remain intact. Earlier development phases, including complete and verified folders, remain available with their reviews and runtime snapshots. Later retests followed observed failures; they are not held-out evaluations.

## Checks and remaining work

91 offline tests pass; dependencies have no broken requirements. All 12 targeted validator cases pass (11 live model cases and one deterministic identifier guard); see [validator report](validator_checks/README.md). These checks do not negate the two full-run failures above. The original supplied-bid records and ten Q&A examples pass exact-citation audits; those records were not regenerated in this follow-up. Original 24-question development retrieval metrics remain Recall@5 0.938 and MRR@5 0.847 with reranking.

The remaining engineering issue is selecting complete supporting quotes for multi-clause fields without exhausting retries. Keep these two failed fields under manual source review. Live presentation is the selected demo format; no recording is planned. Publishing the repository and presenting the session remain submission steps.

## Narrowing retest

### Narrowing step and latest public retest

Both earlier failures were the same pattern: most of a value was supported but one assertion was not, and the whole field was nulled. After the retries, the report step now asks the model to delete only the assertions the support reviewer named, then re-runs the same source-support check against the unchanged citations. Narrowed values get a note, confidence capped at 0.7, and cannot gain new facts. Completeness failures (omissions) are not narrowed.

Development retest (`outputs/public_cvsu_narrowed/`, `outputs/public_beaumont_narrowed/`; earlier folders preserved): Beaumont 16 populated / 4 missing / 0 failed; CVSU 13 / 6 / 1 failed. CVSU Installation is now populated with the supported assembly/servicing/demonstration requirements. A different CVSU field, Required documentation, was then rejected by the completeness review (it omitted the signed-proposal requirement), so CVSU still has one failure; live-model runs vary between runs. Null on a failed field means validation failure, not absence of the requirement. This is a development retest, not a held-out accuracy measure.
