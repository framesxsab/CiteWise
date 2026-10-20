# CiteWise Mini Project: 10-Page Report Blueprint

**Working title:** CiteWise: Benchmarking Neural and Lexical Retrieval for Evidence Selection in Scientific Papers

These are **page allocations** and writing prompts for the final report, not claims of experiments already run. Page count includes references; insert measured values and screenshots after execution. No placeholder numbers should remain in the submitted document.

### Page 1 — Introduction
Problem: finding correct evidence inside long scientific articles is harder than displaying a generated answer. Define question/paper/relevant paragraph, use cases, measurable objectives, scope, exclusions and own contribution.

### Pages 2–3 — Literature Review (at least 5 recent papers)
Create comparison with authors, year, method, dataset, strength, limitation, relevance; discuss (not merely cite) these six:

1. **Chen et al. (2024), M3-Embedding**, ACL Findings. Multilingual dense/sparse/multivector retrieval. Strength: multi-function embedding; limitation: heavier inference vs lightweight encoders. Link: https://aclanthology.org/2024.findings-acl.137/
2. **Saad-Falcon et al. (2024), ARES**, NAACL. Retrieval relevance/answer faithfulness relevance evaluation. Strength: evaluator framework; limitation: synthetic judges require validation against human labels. Link: https://aclanthology.org/2024.naacl-long.20/
3. **Zhang et al. (2025), Qwen3 Embedding**, research preprint. Embedding and reranking family with multiple sizes. Strength: strong dense retrieval; limitation: resource requirements for local deployment. Link: https://arxiv.org/abs/2506.05176
4. **Wang et al. (2025), Astute RAG**, ACL. Studies conflicting/irrelevant retrieved context and robust generation. Strength: source-aware handling of uncertainty; limitation: heavier post-retrieval pipeline outside our retrieval-only benchmark. Link: https://aclanthology.org/2025.acl-long.1476/
5. **Liu and Zhang (2025), Exp4Fuse**, ACL Findings. Rank fusion for improved sparse retrieval. Strength: illustrates fusion approaches; limitation: relies on LLM query expansion, which can increase time/cost. Link: https://aclanthology.org/2025.findings-acl.9/
6. **OpenScholar research team (2026), Synthesizing scientific literature with retrieval-augmented language models**, Nature. Large-scale citation-backed research synthesis. Strength: scientific-use-case validation; limitation: heavy corpus/model infrastructure and multi-paper tasks beyond this mini project. Link: https://www.nature.com/articles/s41586-025-10072-4

Historical benchmark citation (separate from six recent papers): **Dasigi et al. (2021), QASPER**, https://arxiv.org/abs/2105.03011 . Compare the **task setting** carefully: QASPER is anchored to a particular paper, not full-web question answering.

### Page 4 — Methodology
System diagram: question + selected real paper → normalization → chunk IDs and section metadata → retrieval model → ranked evidence → API → UI. Describe fair experiment protocol and reproducibility.

### Page 5 — Data Cleaning, Preprocessing and Splitting
State original 888/281/416 paper splits. Document audit: paper/paragraph/question totals, skipped empty paragraphs, casefold/whitespace normalization, repeated evidence text, text/figure annotation handling, unmatched gold passages, unanswerable questions. Demonstrate **paper-level leakage check** and never random-split paragraphs from one paper into different sets.

### Page 6 — Results and Discussion
Report macro Recall@5, MRR@5, nDCG@5, evaluated query counts, scorable annotation coverage, latency hardware, and quantitative errors for the final experiments. Use `docs/PHASE2_EVALUATION.md` for the train-only ranker diagnosis and phase-separated benchmark evidence. Explain particular missed evidence and model-speed tradeoffs, not invented improvements.

### Page 7 — State-of-the-art Models Comparison
**Measured comparison:** BM25, the QASPER-trained logistic ranker, all-MiniLM-L6-v2 and BGE-base-en-v1.5. This is four retrieval methods, but only MiniLM and BGE-base are neural retrievers. E5/GTE, BGE-M3 and Qwen3 have no CiteWise results; do not list them as evaluated models. If the faculty interprets its requirement as four neural/state-of-the-art models, state that this requirement is not met rather than relabeling the classical and supervised baselines.

Comparison table columns: model, release family, parameter size verified from model cards, Recall@5, MRR@5, nDCG@5, P95 query latency, deployment footprint; fill only after trials. Add ablation if possible (cleaning or top-k or hybrid fusion).

### Page 8 — Results Figures
Figure 1: Recall@5 by model; Figure 2: nDCG@5 by model; Figure 3: warm query P95 by model. Validation plots may support model decisions; final result plots must use the frozen one-time test output only. Include model version, timing phase, and evaluation hardware in each caption.

### Page 9 — User Interface Development and Deployment
Screenshots of real paper selector, user question, top evidence passages, section names; deployment architecture FastAPI + browser, keyboard flow, 390px responsive layout, endpoints, error handling and practical limitations. Include *actual* hosted URL if deployed, otherwise be transparent.

### Page 10 — Conclusion and References
Summarize actual empirical finding, limitations (evidence annotations, table/figure exclusions, frozen model weights, scope), future integration with AutoSD. Full bibliographic details for dataset paper and six recent papers. Ensure references fit allocated space; shorten prose rather than removing required sections.

## Viva preparation

- Why paper-level splitting? Prevents same-paper leakage across train/validation/test.
- What is the difference between BM25 and dense embeddings? Exact-term-frequency weighting versus learned semantic vector similarity.
- What is truly learned in this project? The logistic ranker is fitted using real training-split evidence labels; neural encoder weights remain pretrained and are evaluated zero-shot.
- Why not report accuracy? Passage ranking is not a single-label classification problem; use Recall@k, MRR@k and nDCG@k.
- Why is QASPER useful? Its human evidence annotations allow an objective gold reference.
- Why might a large model be a bad deployed model? Latency, memory, cold start and cost can dominate minor quality advantages.
- What does P95 mean? 95% of measured query latencies are no greater than this value on the stated test setup.
- What cannot the model do? It does not validate scientific truth, reconstruct figures/tables, or reliably answer questions without evidence.
