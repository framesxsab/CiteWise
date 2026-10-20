# Representative validation disagreements

These cases come from the full validation export in `artifacts/phase3/full-validation-v1/FAILURE_REVIEW.md`. They are small case studies, not causal labels or a substitute for blinded human annotation.

## Terminology and semantic matching

For *Addressing word-order Divergence in Multilingual Neural Machine Translation for extremely Low Resource Languages*, the question “Which dataset(s) do they experiment with?” had zero shared unique tokens with the mapped gold paragraph. BM25 did not put evidence in its top five; BGE-base did. The paper has a section headed “datasets used” and discusses IITB and ILCI corpora. This is a concrete case where the neural ranking recovered evidence with low lexical overlap.

In *Improved Neural Relation Detection for Knowledge Base Question Answering*, BM25 retrieved the annotated result passage for “On which benchmarks they achieve the state of the art?” BGE-base's top passage discusses model training difficulty. The question's benchmark term is a useful lexical anchor; the embedding result is topically related but misses the requested result. This illustrates semantic drift in a question whose target is a specific reported fact.

## Long passages

For *Toward Multilingual Neural Machine Translation with Universal Encoder and Decoder*, MiniLM missed the gold paragraph for “Which languages do they test on for the under-resourced scenario?” while BM25 retrieved it. The full gold paragraph exceeds MiniLM's 256-token input cap. This is a truncation risk and a plausible contributor; QASPER labels a paragraph, not the exact supporting span, so the export cannot prove that the answer occurs beyond token 256.

BGE-base has a 512-token cap and also has gold passages beyond that limit, although fewer occur in these 888 questions. The corresponding IDs and lengths are in `failure_analysis.json` and the review queue.

## Missing evidence

Validation has 60 questions annotated unanswerable and 57 with no matched textual evidence. The audit records 253 unsupported figure/table annotation items. These are excluded from retrieval scoring rather than treated as retrieved-text misses. `excluded_validation_questions.csv` retains their paper and question IDs, status and annotation counts. This defines a limitation of the text-only comparison: figure/table evidence is outside its retrieval candidates.

## Ambiguity and manual review boundary

The review queue flags short questions and low query-to-gold lexical coverage to help select cases. These are candidate strata only. An automated similarity or overlap threshold cannot establish that a natural-language question is ambiguous. Any viva claim about ambiguity should cite a specific question and compare its wording, annotated passage and ranked alternatives directly.
