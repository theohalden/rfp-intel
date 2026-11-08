# 5–10 minute live demo guide

The selected submission format is a live presentation. No recording is planned. This guide prepares the session; the live presentation itself has not happened yet.

## 0:00–1:00 — Explain the problem and architecture

RFP facts are spread across PDFs, portal HTML, specification tables, and numbered amendments. Show the README diagram. Search finds evidence; specialists extract facts; reconciliation applies explicit amendments; validation rejects unsupported claims. Explain that LangChain provides retrieval/model integrations while LangGraph controls the workflow and shared state.

## 1:00–2:00 — Parse and index

```bash
source .venv/bin/activate
python main.py ingest --bid ./data/Bid2
python main.py index --bid ./data/Bid2
```

Show per-file type/page/table counts. Open `outputs/ingestion/Bid2/pages.jsonl` and point to physical page metadata and a specification table. Explain that table chunks are intact and incremental indexing keeps other bids. Six low-native-text W-9 pages in Bid1 use optional OCR.

## 2:00–3:00 — Show standalone search

```bash
python main.py search "379-BFNZ processor" --bid-id Bid2 --doc-type specs
python main.py search "submission deadline" --bid-id Bid1 --details
```

Explain dense semantic retrieval, BM25 exact identifiers, RRF fusion, and cross-encoder reranking. A result is a source passage, not an answer. Show file/page citations and use `--full-text` if needed.

## 3:00–5:00 — Extraction and amendments

```bash
python main.py extract --bid ./data/Bid1
```

The run can take several minutes on CPU; start it before presenting and inspect saved results while it runs. Open `outputs/Bid1.json`: the deadline changes from the original RFP date to Addendum 2's date; cite the exact quote. Show a null field and explain why absence does not mean “not required.” Show the 20 fields and validation counts. Open `outputs/Bid1.trace.jsonl` to show parallel specialists, retrieved passages, reconciliation, critic, retries if present, token usage, and final summary.

## 5:00–7:00 — Cited Q&A

```bash
python main.py ask "What is the submission deadline after all addendums?" --bid-id Bid1
python main.py ask "Which affidavits are required?" --bid-id Bid2
python main.py ask "Compare the warranty requirements of Bid1 and Bid2."
python main.py ask "Is a bid bond required, and how much?" --bid-id Bid1
```

Show `outputs/qa_log.md` for other examples. Explain that ReAct calls the same search tool; models select source-line spans and code copies exact quotes. The critic checks semantic support. Missing evidence yields “Not found in documents.”

## 7:00–8:00 — Evaluation and limitations

Show the README Recall@k/MRR table and the two disclosed misses in `eval/results.md`. Explain the 24 quote-anchored development questions, not a held-out claim. Show tests:

```bash
python -m pytest -q
```

Discuss OCR/layout limits, semantic critic errors, uncalibrated confidence, and literal time-zone ambiguity. Explain the renamed-folder test demonstrates filename independence, not unseen-content accuracy.

## Optional API demonstration

```bash
python main.py serve
```

Open `http://127.0.0.1:8000/docs`; exercise `/search` or `/ask` without exposing `.env`.

## Questions to be ready to explain

- Why BM25 is useful for exact SKUs and dense search for paraphrases.
- Why RRF combines rankings instead of incompatible raw scores.
- Why tables are intact but reranker scoring uses internal windows.
- How source-line selections preserve verbatim citations.
- How a relevant later amendment overrides a field while unrelated amendments do not.
- Why retries are per failed field and capped at two.
- Why a cited answer can still be semantically wrong and needs a critic/source review.
- Why missing evidence is null rather than an inferred negative requirement.

## Before the live session

- Activate the environment, verify the API key locally, and keep `.env` off-screen.
- Complete extraction before the meeting; CPU reranking and API calls can take several minutes. Present the saved result alongside its full trace, and run search/Q&A live.
- Keep the original source PDFs, saved reports, evaluation table, and Q&A log available if the network fails.
- For public-document checks, show `outputs/public_rfp_review.md`, then the linked first-run and development-retest reports. Explain the preserved misses, general fixes, and remaining limits. Show a source quote beside its field; explain that passing citation checks does not establish completeness or correct field classification. Do not claim these few documents prove universal support.
- Use an explicit bid filter when demonstrating one bid. Show the missing-time-zone case as an evidence limitation; never guess the zone from the agency's location.
- Practice explaining the difference between retrieval quality, citation provenance, and semantic accuracy.

Public retest status: LCA completed with zero failed fields; CVSU Installation and Beaumont required documentation remain rejected. Show these explicit failures honestly when discussing limits. Final records/reviews are under `outputs/public_*_corrected/`.
