# Literature Review Notes (source-verified starting draft)

This comparison separates **what each paper studies** from **what CiteWise actually implements**. A citation is not a claim that the published model was retrained or reproduced in this mini project. All dates and titles below were checked against the primary publication pages.

| Paper | Main contribution | Strength | Boundary relative to CiteWise |
|---|---|---|---|
| Chen et al., **M3-Embedding** (2024, ACL Findings) | A multifunction embedding system combining dense, sparse and multi-vector retrieval signals with support for different text lengths. | Embedding architecture designed for versatile retrieval tasks. | BGE-M3 dense retrieval is a planned comparison; no local measured score is available. CiteWise does not reproduce its training or full fusion method. |
| Saad-Falcon et al., **ARES** (2024, NAACL) | Evaluation of RAG systems through context relevance, answer faithfulness and answer relevance. | Distinguishes retrieval quality from answer quality. | CiteWise uses human QASPER gold passages for retrieval metrics, not ARES LM judges, and does not report answer faithfulness as a measured outcome. |
| Zhang et al., **Qwen3 Embedding** (2025, research preprint) | Trained embedding and reranker families from the Qwen3 model line with sizes from 0.6B to 8B. | Provides a strong large-encoder retrieval comparison with an explicit performance/compute trade-off. | The pretrained 0.6B checkpoint is planned; execution is blocked pending weights/compute. No leaderboard result is used as a CiteWise result. |
| Wang et al., **Astute RAG** (2025, ACL) | Handles imperfect retrieval and knowledge conflicts during RAG generation. | Highlights failure modes when retrieved evidence is inaccurate or inconsistent. | CiteWise keeps evidence provenance and avoids pretending that retrieved snippets are verified answers; conflict resolution is future work. |
| Liu & Zhang, **Exp4Fuse** (2025, ACL Findings) | Improves sparse retrieval through LLM-expanded queries and ranking fusion. | Demonstrates that the retrieval pipeline can be improved without replacing its underlying model. | CiteWise presently compares retrievers without LLM-based query expansion. A separate hybrid-fusion ablation could be added but must not be described as an Exp4Fuse implementation. |
| Asai et al., **OpenScholar** (2026, Nature) | Large-scale scientific literature synthesis with retrieval and citation-backed responses. | Validates a scientific-research use case and motivates traceable evidence. | OpenScholar spans many documents and generates answers; CiteWise focuses on within-paper annotated evidence ranking, which is smaller but allows a controlled comparison. |

## Critical synthesis

The literature exposes two distinct evaluation targets. **Retrieval systems** aim to rank relevant document passages, while **RAG question-answering systems** additionally have to synthesize an answer that is supported by those passages. The ARES and Astute RAG studies caution against equating an apparently fluent response with a correct source-supported answer. The M3-Embedding and Qwen3 Embedding approaches motivate studying the quality/latency trade-off between neural retrievers. The Exp4Fuse work motivates a possible rank-fusion extension, while OpenScholar illustrates why provenance matters in a realistic scientific workflow.

Our specific gap is therefore **an explainable, reproducible comparison of paragraph-level evidence retrievers on original-paper QASPER annotations, integrated into a usable lightweight web interface**. The original target list included four pretrained neural checkpoints, BM25 and a QASPER-trained lexical ranker; the completed CiteWise comparison measured only MiniLM and BGE-base as neural encoders, alongside BM25 and the logistic baseline. E5/GTE, BGE-M3 and Qwen3 have no CiteWise scores. CiteWise does not claim four neural evaluations, a novel neural architecture, a full QASPER QA system, or equivalence to author-reported results; it contributes an experimental engineering comparison with independently verifiable outputs.

## Phase 3 source verification and comparison limits

Primary publication records were checked again on 2026-10-18. M3-Embedding, ARES, Astute RAG and Exp4Fuse have the ACL publication records linked below; Qwen3 Embedding remains labeled a 2025 preprint. The Nature OpenScholar article was published 4 February 2026, with the listed volume/pages. None of these papers' published benchmark numbers are entered into CiteWise's result tables.

The methods suggest distinct experiments rather than interchangeable scores: M3 motivates a dense-only model ablation; Qwen motivates explicit task instructions; ARES motivates separating retrieval relevance from generated-answer quality; Astute RAG motivates examining misleading candidates; Exp4Fuse motivates future query expansion/rank fusion; OpenScholar motivates scientific provenance but evaluates a broader multi-paper synthesis task. The current within-paper QASPER metrics do not reproduce those papers' task settings. E5/GTE are older retrieval alternatives and should be discussed as resource-driven substitutions, not recent state-of-the-art replacements.

## Primary references

1. Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., & Liu, Z. (2024). M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. *Findings of ACL 2024.* https://aclanthology.org/2024.findings-acl.137/
2. Saad-Falcon, J., Khattab, O., Potts, C., & Zaharia, M. (2024). ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems. *NAACL 2024.* https://aclanthology.org/2024.naacl-long.20/
3. Zhang, Y., Li, M., Long, D., et al. (2025). Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. *arXiv:2506.05176.* https://arxiv.org/abs/2506.05176
4. Wang, F., Wan, X., Sun, R., Chen, J., & Arik, S. O. (2025). Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models. *ACL 2025.* https://aclanthology.org/2025.acl-long.1476/
5. Liu, L., & Zhang, M. (2025). Exp4Fuse: A Rank Fusion Framework for Enhanced Sparse Retrieval using Large Language Model-based Query Expansion. *Findings of ACL 2025.* https://aclanthology.org/2025.findings-acl.9/
6. Asai, A., He, J., Shao, R., et al. (2026). Synthesizing scientific literature with retrieval-augmented language models. *Nature, 650*, 857–863. https://doi.org/10.1038/s41586-025-10072-4
7. Dasigi, P., Lo, K., Beltagy, I., Cohan, A., Smith, N. A., & Gardner, M. (2021). A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers. *NAACL 2021.* https://arxiv.org/abs/2105.03011

Dataset license: QASPER is distributed under **CC BY 4.0**. Preserve attribution in software/demo and report. Check individual neural model licenses before hosting models or commercial redistribution.
