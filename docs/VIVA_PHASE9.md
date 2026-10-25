# CiteWise viva notes and live demonstration

## Short live demonstration

1. Open the local CiteWise page and point out that search is restricted to the official QASPER validation split. The application retrieves evidence paragraphs; it does not generate scientific answers.
2. Select a validation paper and one of its annotated questions. Run E5-base-v2 if its pinned weights are staged; then select MiniLM to show the measured speed-oriented alternative.
3. Match a returned evidence ID, section heading and passage text to the selected paper. Explain that the API returns ranked source paragraphs, not generated claims.
4. Show the comparison table, 95% paper-cluster intervals, paired neural comparison and quality/latency chart. Explain that all six methods share 888 questions and 276 evaluated papers.
5. Show the 375/390-pixel browser evidence and automated axe result. State that local browser verification is complete, while hosted HTTPS and manual assistive-technology testing remain pending.

## Viva questions

1. **What problem does CiteWise solve?** It ranks text passages within a selected scientific paper so a reader can inspect evidence relevant to a question. It is an evidence-retrieval tool, not an answer generator.
2. **Why QASPER?** QASPER contains questions asked about real research papers, answer annotations and supporting evidence. It supports a paper-local paragraph retrieval task with human evidence IDs.
3. **How was it cleaned?** Empty paragraphs were removed, whitespace normalized, and textual annotations matched to stable original paragraph IDs. Unanswerable questions and annotations without mapped textual evidence are counted and excluded from ranking metrics.
4. **How was leakage prevented?** The official paper-level train, validation and test splits were retained. The logistic ranker was fitted on train only; model comparison and selection used validation. The earlier four-method held-out test ran once and was not accessed or changed in Phase 9.
5. **How does BM25 differ from the neural models?** BM25 scores lexical term overlap and document frequency. Neural retrievers encode a query and each passage into vectors, then order candidates by normalized dot product; they can match related wording but can miss exact terms or long passage details.
6. **What are the four neural models?** MiniLM-L6-v2 is a compact six-layer, mean-pooled encoder with 256-token truncation. BGE-base-en-v1.5 uses a retrieval instruction on queries, CLS pooling and 512-token truncation. E5-base-v2 uses `query:` and `passage:` prefixes, masked mean pooling and 512-token truncation. GTE-base is a general text-embedding model using mean pooling and 512-token truncation. CiteWise pins each revision and uses its matching tokenizer and SentenceTransformers configuration.
7. **Why do they rank differently?** Their training objectives and representation spaces differ; the retrieval instruction and E5's asymmetric prefixes also change query/document vectors. Sequence limits can truncate different content. These comparisons show ranking differences but do not isolate a causal mechanism.
8. **How do rankings become evidence passages?** Each model scores all paragraphs in the selected paper. The API returns the top passages with their original IDs, section headings and source text. Gold relevance is based on QASPER's mapped textual evidence annotations.
9. **What do the metrics mean?** Recall@1/5 is the fraction of annotated gold passages retrieved in the first one/five positions. MRR@5 rewards an early first relevant result. nDCG@5 discounts relevant results lower in the ranking. Reported values are macro means over 888 scorable questions.
10. **Which model leads?** E5-base-v2 has the highest validation point estimate for Recall@1 (21.66%), Recall@5 (54.67%), MRR@5 (46.09%) and nDCG@5 (43.40%). Its paired Recall@5 interval versus BGE narrowly excludes zero; paired Recall@1, MRR and nDCG intervals include zero. The intervals are unadjusted for the multiple comparisons.
11. **What is the speed/quality trade-off?** Among neural encoders, MiniLM has the lowest measured warm P50/P95 (18.81/26.22 ms) and peak RSS (0.70 GiB). E5 has higher validation point estimates at 73.31/91.23 ms warm P50/P95 and 1.16 GiB peak RSS. Model load and one-time document encoding are reported separately; they are not part of warm query latency.
12. **Why did logistic underperform?** The train diagnostic sampled 3,603 positive and 12,581 negative pairs (22.3% positive), with selected hard negatives averaging BM25 5.044. Full validation non-relevant candidates averaged BM25 1.704. Training therefore emphasized hard candidates while inference ranked every paper paragraph. Feature standardization left bounded validation scores unchanged. This supports a candidate-distribution mismatch; it does not prove a single cause.
13. **How does the UI use the models?** The selection list reflects weights and ranker artifacts staged in the running environment. The API validates the selected paper and model, performs ranking, and returns evidence IDs and text. The public UI/API accept validation only; held-out test search returns an error.
14. **What has deployment and accessibility testing established?** Playwright exercised the local app, evidence parity, keyboard focus, mobile widths and model inference; axe reported zero WCAG 2.1 A/AA violations. No hosted HTTPS application, cross-platform hosting, manual screen-reader review or public traffic test has been verified.

## Metric and artifact references

- Full validation aggregate: `artifacts/phase8/full-validation/benchmark_validation.json`.
- Verified per-query predictions and paired comparisons: `artifacts/phase8/full-validation/per_query_validation.csv` and `paired_comparisons_validation.csv`.
- Phase 9 table and figures: `artifacts/phase9/model_comparison.csv` and `artifacts/phase9/figures/`.
- Four exact model revisions, preprocessing settings, memory and latency are in `src/citewise/research_summary.json`.
- Model method sources: [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [BGE-base model card](https://huggingface.co/BAAI/bge-base-en-v1.5), [E5-base-v2 model card](https://huggingface.co/intfloat/e5-base-v2), [GTE-base model card](https://huggingface.co/thenlper/gte-base).
