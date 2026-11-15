# Legal RAG Evaluation

Status: **PARTIAL**
Run date: `2026-10-21T19:42:52.585473+00:00`

## Datasets
- **retrieval**: pramaan-synthetic-retrieval-gold v1, 5 examples, SHA-256 `a18d6f5d90379e5195f0eb6275f5f3e2dd879de74510bc227e1e2ab95c4b71db`
  - Provenance: Project fixture labels for the repository's synthetic sample corpus; relevance judgments are not externally validated and must resolve against the active indexed corpus before scoring.
- **answers**: pramaan-abstention-observations v1, 2 examples, SHA-256 `63b892c56e41f1e71f69366cee972b87f42f638acc7d22260448b6ce617bc837`
  - Provenance: Project-maintained expected-abstention observations; the dataset contains no judged reference answers and supports no semantic answer-correctness score.

## Configuration

- `llm_provider`: stub
- `llm_model`: placeholder
- `answer_execution_path`: LangGraph question-answering workflow
- `llm_api_key_configured`: False
- `llm_temperature`: 0.3
- `llm_max_tokens`: 2048
- `embedding_provider`: sentence_transformer
- `embedding_model`: intfloat/multilingual-e5-large
- `embedding_dimension`: 1024
- `reranker_provider`: None
- `reranker_model`: None
- `vector_top_k`: 20
- `keyword_top_k`: 20
- `rerank_top_k`: 10
- `final_top_k`: 5
- `retrieval_evaluation_top_k`: 10
- `rrf_k`: 60
- `vector_weight`: 0.6
- `keyword_weight`: 0.4
- `max_retrieval_attempts`: 2

## Evaluations

### Document Analysis - NOT RUN

Not measured: no answer-reference evaluation is wired for document analysis.


### Document Comparison - NOT RUN

Not measured: no answer-reference evaluation is wired for document comparison.


### Retrieval - BLOCKED

No valid, indexed relevance labels were available.

**Diagnostics**

- `unique_retrieval_requests`: 0
- `failed_requests`: 0
- `empty_results`: 0
- `latency_ms`: {"count": 0, "mean_ms": null, "p50_ms": null, "p95_ms": null}


### Answer Generation - PARTIAL

Operational observations completed; semantic answer-quality evaluation remains blocked.

**Operational metrics**

- `citation_identifier_validity`: not measured
- `citation_valid_identifier_records`: 0
- `citation_provenance_validity`: not measured
- `citation_valid_provenance_records`: 0
- `citation_record_integrity`: not measured
- `citation_integrity_valid_records`: 0
- `citation_records`: 0
- `valid_citation_records`: 0
- `inline_reference_resolution`: not measured
- `inline_references`: 0
- `unresolved_inline_references`: 0
- `malformed_inline_references`: 0
- `citation_semantic_support_evaluated`: False
- `abstention_accuracy`: 1.0
- `abstention_cases`: 1
- `correct_abstentions`: 1
- `skipped_abstention_cases`: 1
- `required_term_coverage`: not measured
- `required_terms_checked`: 0
- `required_terms_matched`: 0
- `expected_citation_document_coverage`: not measured
- `expected_citation_documents`: 0
- `matched_expected_citation_documents`: 0
- `unresolved_expected_source_cases`: 1

**Diagnostics**

- `answers_observed`: 2
- `failed_requests`: 0
- `retrieval_failures`: 0
- `fallback_responses`: 0
- `empty_retrievals`: 2
- `latency_ms`: {"count": 2, "mean_ms": 19253.211, "p50_ms": 19253.211, "p95_ms": 35900.029}

**Answer quality: BLOCKED**
Configured LLM is a deterministic stub; answer-quality evaluation is blocked.

- Citation identifier validity verifies source-ID syntax and retrieved document/chunk membership only.
- Required-term checks are literal substring checks and do not establish answer correctness.
- No semantic claim-evidence entailment evaluator is available.

## Run limitations

- No semantic claim-evidence entailment metric is implemented.
- No human answer-quality evaluator or judged answer reference set is configured.
- Document analysis and document comparison evaluations are not executed by this workflow.
- Retrieval results depend on the indexed MongoDB corpus and its current contents.
- Provider output, approximate vector search, model hardware, and corpus changes can introduce nondeterminism.
