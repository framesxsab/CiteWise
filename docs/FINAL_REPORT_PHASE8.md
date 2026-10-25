# CiteWise: Evidence Retrieval in Scientific Papers

**Phase 8 validation update | Mini-project report**  
QASPER official splits | Paragraph-level retrieval | 25 October 2026

<!-- PAGE 1 -->

# 1. Introduction

Researchers looking for evidence in a paper must find passages that support a question, often across long sections of technical text. CiteWise addresses this retrieval step: a user selects a real QASPER paper, asks a question, and receives ranked paragraph candidates with section headings and stable evidence IDs. It does not generate answers or certify scientific truth.

The study asks: *How do lexical retrieval, a QASPER-trained lexical ranker, and four pretrained neural encoders differ in evidence ranking and CPU cost?* The contribution is a controlled comparison on the official paper-level split, a documented supervised-ranker failure, a reproducible pipeline, and an application that returns passages from the source paper.

Phase 8 evaluates six methods on QASPER validation: BM25 and supervised logistic baselines, plus MiniLM-L6-v2, BGE-base-en-v1.5, E5-base-v2 and GTE-base. These are the four neural encoders specified for this project. This comparison tests the requested models on one within-paper task; it does not claim that every checkpoint is the newest model family or reproduce published leaderboards.

The held-out test evaluation was performed once before E5 and GTE were added, using the original four methods. It was not rerun or used for Phase 8 decisions, and E5/GTE have no test scores. CiteWise remains locally tested; no public deployment or manual assistive-technology evaluation is claimed.

<!-- PAGE 2 -->

# 2. Literature Review

QASPER frames information seeking around a particular research paper. Dasigi et al. collected questions from readers who saw a paper title and abstract, then gathered answers and supporting evidence from full text [1]. This provides human evidence paragraphs for retrieval evaluation. CiteWise uses the paper-local setting and mapped textual evidence, while excluding unanswerable and unmatched questions from ranking metrics.

Chen et al.'s M3-Embedding combines dense, sparse, and multi-vector retrieval and supports multilingual inputs and long documents [2]. Its multi-function training is relevant to retrieval versatility, but CiteWise neither uses nor reproduces M3-Embedding. BGE-M3 was an intended model comparison and has no measured CiteWise score.

Saad-Falcon et al.'s ARES evaluates retrieval-augmented generation along context relevance, answer faithfulness, and answer relevance, using trained language-model judges and a small human-labeled set for prediction-powered inference [3]. CiteWise instead evaluates paragraph ranking against QASPER annotations. ARES motivates keeping retrieval quality distinct from generated-answer quality; CiteWise does not claim answer-faithfulness metrics.

Zhang et al.'s Qwen3 Embedding preprint describes embedding and reranking checkpoints at multiple scales, trained with large-scale pretraining and supervised fine-tuning [4]. It motivates comparing quality with resource cost. Qwen3-Embedding-0.6B was not downloaded or evaluated here, so no published benchmark number is attributed to CiteWise.

<!-- PAGE 3 -->

# 2. Literature Review (continued)

Wang et al.'s Astute RAG studies imperfect retrieved context and conflicts between parametric and retrieved knowledge, then proposes a source-aware generation process [5]. CiteWise shares the concern that a plausible passage can still be misleading, but stops at ranking and displays source text for inspection. It does not implement Astute RAG's generation or conflict-resolution pipeline.

Liu and Zhang's Exp4Fuse applies LLM-based query expansion to sparse retrieval, forms rankings for original and expanded queries, and fuses the lists [6]. This supports studying rank fusion as a future extension. CiteWise compares the current retrievers without query expansion and makes no Exp4Fuse implementation or performance claim.

Asai et al.'s OpenScholar indexes a large scholarly corpus and synthesizes citation-backed responses to multi-paper scientific questions [7]. It demonstrates the broader research-synthesis setting. CiteWise is narrower: retrieval is constrained to one paper and evaluated against annotated paragraphs.

Together these works distinguish retrieval from answer generation, and scientific search from multi-paper synthesis. CiteWise contributes a reproducible six-method comparison of paragraph evidence ranking on QASPER. Its findings are specific to this dataset, split, preprocessing and CPU setup; they are not reproductions of the papers' published benchmarks.

<!-- PAGE 4 -->

# 3. Methodology

**Architecture.** A QASPER paper is loaded with original paragraph IDs, section names and annotations. The user selects a paper and question. A selected retriever scores every original paragraph in that paper; the API returns ranked passage IDs, text, headings, scores and latency. The UI labels them evidence candidates, not verified answers.

**Data discipline.** The logistic classifier was fitted on train papers only (`max_iter=1500`, `random_state=42`). Model comparisons and decisions use the fixed validation split. The held-out test result remains a separate earlier four-method experiment; no Phase 8 tuning or scoring used test questions.

**Neural encoders.** MiniLM uses plain text, mean pooling and 256-token truncation. BGE-base uses the query instruction `Represent this sentence for searching relevant passages: `, CLS pooling and 512-token truncation. E5 uses `query: ` and `passage: ` prefixes with mean pooling and 512-token truncation. GTE uses plain text, mean pooling and 512-token truncation. All encode normalized float32 vectors and rank by dot product. Immutable revisions are listed on page 8.

**Metrics.** For gold paragraph set G, Recall@k = |G intersect top-k| / |G|. Recall@1 was calculated from the first saved validation candidate. MRR@5 is the reciprocal rank of the first relevant passage or zero. Binary nDCG@5 discounts relevant results by log rank and divides by ideal gain. Metrics are macro-averaged over scorable questions. Percentile 95% intervals use 2,000 paper-cluster bootstrap replicates with seed 42; paired comparisons use identical questions.

**Timing and checkpoint provenance.** Warm P50/P95 includes query encoding, scoring and sorting with model/document vectors cached. Model loading and paragraph encoding are timed separately. A Windows checkpoint-write repair was applied mid-run; retrieval/evaluation code hashes, data SHA, query-plan SHA and protocol match across runs. Per-model source hashes and the checkpoint-writer migration are retained in `artifacts/phase8/full-validation/benchmark_validation.json`.

<!-- PAGE 5 -->

# 4. Data Cleaning and Preprocessing

The source is Allen Institute's QASPER dataset, revision `91bc63e221475d59a6e17e4299fbfc5a919d7c7a`, distributed under CC BY 4.0 [1]. The official paper-level partitions are retained; no random paragraph split or cross-split resampling was performed. The prepared data audit records:

| Official split | Papers | Non-empty paragraphs | Questions | Scorable | Unmatched/no text | Unanswerable |
|---|---:|---:|---:|---:|---:|---:|
| Train | 888 | 46,882 | 2,593 | 2,098 | 224 | 271 |
| Validation | 281 | 13,266 | 1,005 | 888 | 57 | 60 |
| Test | 416 | 19,817 | 1,451 | 1,309 | 63 | 79 |

Preparation strips empty paragraphs and collapses whitespace. Evidence phrases are whitespace-normalized and case-folded before mapping to stable IDs tied to original section/paragraph positions. Repeated identical paragraph text can map to multiple IDs. Figure/table-only annotations are counted as unsupported textual evidence rather than converted into fabricated paragraphs: 386 items in train, 253 in validation, and 459 in test.

A question is scorable if at least one annotated textual evidence phrase maps to a paragraph. Unanswerable questions and questions without mapped text evidence are excluded consistently for every method and reported as coverage exclusions. The final test run evaluated all 1,309 scorable questions across 408 papers; its 63 unmatched/no-text and 79 unanswerable questions were excluded. The final test questions were not used for training or model selection.

<!-- PAGE 6 -->

# 5. Results and Discussion

The complete six-method comparison evaluated all **888 scorable validation questions** across **276 papers**. Every model has the same query, source-paper paragraphs and mapped gold evidence. The intervals are 95% paper-cluster bootstrap estimates (2,000 replicates, seed 42).

| Method | Recall@1 [95% CI] | Recall@5 [95% CI] | MRR@5 [95% CI] | nDCG@5 [95% CI] |
|---|---:|---:|---:|---:|
| BM25 | 13.96% [11.95, 16.07] | 44.32% [41.04, 47.44] | 34.56% [32.01, 37.21] | 32.86% [30.33, 35.32] |
| Supervised logistic | 0.71% [0.35, 1.16] | 4.93% [3.74, 6.27] | 5.20% [4.13, 6.38] | 3.91% [3.09, 4.86] |
| MiniLM-L6-v2 | 14.04% [11.89, 16.31] | 46.11% [43.14, 48.99] | 36.05% [33.34, 38.91] | 34.57% [32.20, 36.95] |
| BGE-base-en-v1.5 | 20.11% [17.87, 22.52] | 52.10% [49.20, 55.08] | 43.87% [40.99, 46.60] | 41.42% [38.98, 43.95] |
| E5-base-v2 | 21.66% [19.10, 24.16] | 54.67% [51.54, 57.63] | 46.09% [43.27, 48.89] | 43.40% [40.83, 45.79] |
| GTE-base | 13.93% [11.83, 16.14] | 45.69% [42.65, 48.69] | 36.19% [33.27, 38.89] | 34.51% [32.05, 37.00] |

![Full validation Recall@5 with paper-cluster confidence intervals](assets/phase8/validation-recall-at-5.png)

**Figure 1.** Recall@5 means and 95% paper-cluster intervals from the saved full-validation predictions. E5-base-v2 has the highest neural point estimate on each displayed ranking metric.

<!-- PAGE 7 -->

# 5. Results and Discussion (continued)

Paired bootstrap comparisons use the same validation questions and resample papers as clusters. E5-base-v2 minus MiniLM gains 8.56 percentage points in Recall@5 [95% CI 5.68, 11.54], 10.03 in MRR@5 [7.38, 12.94] and 8.83 in nDCG@5 [6.59, 11.13]. E5 minus GTE gains 8.98 [5.90, 12.02], 9.90 [7.40, 12.47] and 8.88 [6.65, 11.10]. E5 minus BGE gains are smaller: 2.57 [0.04, 5.26], 2.21 [-0.17, 4.60] and 1.98 [-0.07, 4.06]. The paired Recall@5 interval barely excludes zero; the MRR/nDCG intervals include zero. Intervals are descriptive and are not adjusted for multiple comparisons.

![Paired full-validation metric differences](assets/phase8/validation-paired-differences.png)

**Figure 2.** Model A minus model B paired mean differences with paper-cluster 95% intervals. It is generated from the actual validation prediction file.

| Method | Warm query P50/P95 (ms) | Model load (ms) | Paragraph encoding (s) | Peak worker RSS (GiB) |
|---|---:|---:|---:|---:|
| BM25 | 2.70 / 5.26 | 0.01 | 0.0 | 0.06 |
| Supervised logistic | 4.81 / 9.31 | 4,809.5 | 0.0 | 0.18 |
| MiniLM-L6-v2 | 18.81 / 26.22 | 659.8 | 421.9 | 0.70 |
| BGE-base-en-v1.5 | 88.18 / 110.41 | 1,081.2 | 3,029.0 | 1.18 |
| E5-base-v2 | 73.31 / 91.23 | 1,042.1 | 2,982.8 | 1.16 |
| GTE-base | 70.83 / 88.80 | 999.3 | 2,958.6 | 1.17 |

These are CPU measurements from separate model workers on a Windows 11 host (16 logical CPUs, 15.82 GiB installed RAM, no CUDA). Document encoding covers the 12,869 paragraphs belonging to evaluated papers; warm query timing excludes it. The small logistic baseline remains a negative result consistent with its hard-negative sampling mismatch (Section 5 of the Phase 6 report).

The token audit deliberately tokenizes raw paragraphs without truncation to identify long inputs. It recorded 569 MiniLM paragraphs above 256 tokens, 21 BGE, 22 E5 and 21 GTE paragraphs above 512 tokens. A warning such as `273 > 256` reports the audit length, not the inference length: SentenceTransformer preprocessing truncates MiniLM inputs to 256. All six models completed all 888 queries without indexing errors.

The table below preserves the **earlier, one-time held-out test** results for the original four methods only. It is not a six-model comparison; E5/GTE were not tested, and Phase 8 decisions used validation only.

| Frozen test method | Recall@5 (95% CI) | MRR@5 (95% CI) | nDCG@5 (95% CI) |
|---|---:|---:|---:|
| BM25 | 43.77% [41.38, 46.24] | 37.92% [35.63, 40.19] | 34.57% [32.61, 36.58] |
| Supervised logistic | 5.50% [4.53, 6.60] | 6.66% [5.54, 7.91] | 4.87% [4.03, 5.85] |
| MiniLM-L6-v2 | 47.36% [44.92, 49.87] | 43.70% [41.34, 46.07] | 38.77% [36.66, 40.90] |
| BGE-base-en-v1.5 | 52.35% [49.84, 54.86] | 49.97% [47.45, 52.30] | 44.32% [42.18, 46.47] |

<!-- PAGE 8 -->

# 6. State-of-the-Art Models Comparison

The four requested neural encoders were evaluated with the same full-validation question set, paragraph universe, relevance mapping and ranking pipeline. Their quality/latency trade-off is shown below; baselines remain separate on page 6.

| Neural model | Input and pooling | Recall@5 | MRR@5 | nDCG@5 | Warm P50/P95 (ms) |
|---|---|---:|---:|---:|---:|
| MiniLM-L6-v2 | Plain text; mean; 256 tokens | 46.11% | 36.05% | 34.57% | 18.81 / 26.22 |
| BGE-base-en-v1.5 | BGE query instruction; CLS; 512 tokens | 52.10% | 43.87% | 41.42% | 88.18 / 110.41 |
| E5-base-v2 | `query:` / `passage:`; mean; 512 tokens | 54.67% | 46.09% | 43.40% | 73.31 / 91.23 |
| GTE-base | Plain text; mean; 512 tokens | 45.69% | 36.19% | 34.51% | 70.83 / 88.80 |

The neural evaluation checkpoint revisions are MiniLM `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, BGE `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, E5 `1c644c92ad3ba1efdad3f1451a637716616a20e8` and GTE `f846641ce4fb595a946b68fe18389798a8da9a64`. Full repository identifiers and model-specific prefixes are stored with the dashboard bundle and benchmark configurations.

![Measured model loading and paragraph encoding phases](assets/phase8/validation-inference-phases.png)

**Figure 3.** Model initialization and document encoding are separate from cached warm query retrieval. E5 is the validation quality leader among the four neural models; MiniLM is the fastest neural option in warm query latency and the least expensive to encode across the evaluated corpus. This does not establish performance on other papers or the held-out test set.

E5/GTE are prior-generation encoders relative to current embedding releases; they satisfy the project-specified four-model experiment but should not be described as the newest available systems. Published leaderboard scores are not CiteWise results.

<!-- PAGE 9 -->

# 7. User Interface Development and Deployment

CiteWise integrates FastAPI, real QASPER paper selection, model availability checks, the retrieval API and a validation-only results dashboard. Search returns actual ranked passages with section headings, IDs, source text and scores. Unavailable models are disabled; the held-out test split is rejected by the interactive API.

<div class="screenshot-pair">
<figure><img src="assets/phase8/comparison-desktop.png" alt="CiteWise local desktop interface with six-model validation metrics, confidence intervals and research methodology"><figcaption>Figure 4. Current desktop interface showing the full validation comparison and method information.</figcaption></figure>
<figure><img src="assets/phase8/mobile-390-evidence.png" alt="CiteWise mobile interface at 390 CSS pixels with actual retrieved evidence"><figcaption>Figure 5. Local mobile evidence retrieval at 390 CSS pixels; 375-pixel layout also passed.</figcaption></figure>
</div>

Ten Playwright browser tests passed locally, covering BM25/BGE evidence parity, MiniLM/E5/GTE real inference and evidence parity, the saved dashboard, invalid requests, keyboard focus, validation-only access and 375/390-pixel layouts. Axe reported zero WCAG 2.1 A/AA violations. These are automated checks; manual screen-reader and assistive-technology evaluations were not performed.

The Dockerfile, pinned model staging and health check remain deployment instructions. Docker Engine was unavailable, so no container build was verified. There is no public HTTPS URL or hosted search flow; CiteWise is locally demonstrated, not publicly deployed.

<!-- PAGE 10 -->

# 8. References

[1] P. Dasigi, K. Lo, I. Beltagy, A. Cohan, N. A. Smith, and M. Gardner. "A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers." *Proceedings of NAACL-HLT*, 2021, pp. 4599-4610. https://aclanthology.org/2021.naacl-main.365/

[2] J. Chen et al. "M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation." *Findings of ACL*, 2024, pp. 2318-2335. https://aclanthology.org/2024.findings-acl.137/

[3] J. Saad-Falcon, O. Khattab, C. Potts, and M. Zaharia. "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems." *NAACL-HLT*, 2024, pp. 338-354. https://aclanthology.org/2024.naacl-long.20/

[4] Y. Zhang et al. "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models." arXiv:2506.05176, 2025. https://arxiv.org/abs/2506.05176

[5] F. Wang et al. "Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models." *ACL*, 2025, pp. 30553-30571. https://aclanthology.org/2025.acl-long.1476/

[6] L. Liu and M. Zhang. "Exp4Fuse: A Rank Fusion Framework for Enhanced Sparse Retrieval using Large Language Model-based Query Expansion." *Findings of ACL*, 2025, pp. 163-173. https://aclanthology.org/2025.findings-acl.9/

[7] A. Asai et al. "Synthesizing scientific literature with retrieval-augmented language models." *Nature* 650, 857-863 (2026). https://doi.org/10.1038/s41586-025-10072-4

**Data and model cards.** QASPER dataset card: https://huggingface.co/datasets/allenai/qasper. MiniLM: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2. BGE: https://huggingface.co/BAAI/bge-base-en-v1.5. E5: https://huggingface.co/intfloat/e5-base-v2. GTE: https://huggingface.co/thenlper/gte-base. QASPER is CC BY 4.0; retain source attribution. Model licenses and project code-license status are in `docs/LICENSING.md`.
