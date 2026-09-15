# Pramaan — Indian Legal AI Research Assistant

Source-first semantic search for Indian statutes, rules, and judgments. Built on FastAPI, Next.js, MongoDB, and configurable embedding providers.

> **Disclaimer**: Pramaan is an AI research assistant and is **not a substitute for professional legal advice**. All sample documents are synthetic.

---

## Architecture

```
                    Query
                      ↓
          ┌───────────┴───────────┐
          ↓                       ↓
     Vector Search             BM25
     (embed query →           (tokenise →
      cosine similarity)       BM25 scoring)
          ↓                       ↓
          └───────────┬───────────┘
                      ↓
                 RRF Fusion
                 (reciprocal rank
                  fusion, k=60)
                      ↓
               Candidate Set
                      ↓
              Cross-Encoder
                 Reranker
                      ↓
                Top Results
                (chunk + scores
                 + provenance)
```

**Current status**: Steps 1–4 complete (ingestion, embedding, hybrid retrieval, reranking). Step 5 (LLM answer generation) is next.

---

## Why These Technology Choices

### Embeddings
Dense vector embeddings map text to a high-dimensional space where semantically similar passages are geometrically close. This enables *concept* search rather than keyword matching — a query like "valid contract requirements" retrieves passages about offer, acceptance, and consideration even if those exact words do not appear.

### intfloat/multilingual-e5-large
Chosen for three reasons:
1. **Retrieval quality** — trained specifically for passage retrieval tasks, consistently outperforming general sentence encoders on legal text benchmarks.
2. **Multilingual** — covers 100+ languages including English, Hindi, Marathi, Tamil, Bengali, and other Indian languages that will appear in future documents.
3. **Swappable** — the model name and embedding dimension are configuration values. Switching to a smaller model (`paraphrase-multilingual-MiniLM-L12-v2`, 384-dim) requires only two env var changes.

### Why Vector Search Alone Is Insufficient
Legal queries often contain exact terms that semantic similarity can miss:
- **Section numbers**: "Section 302 IPC"
- **Act names**: "Indian Contract Act"
- **Case citations**: "State v. Accused Person"
- **Legal terminology**: "habeas corpus", "certiorari"
- **Article numbers**: "Article 14"

A semantic search for "Section 302 murder" may rank a general discussion of criminal law higher than the actual Section 302 text if the embedding doesn't strongly encode the exact term match.

### Why Keyword Retrieval Matters
BM25 excels at exact term matching and TF-IDF weighting. A chunk containing "Section 302" will score highly for a query containing "Section 302", regardless of semantic similarity. Combining both approaches provides better recall.

### Why RRF (Reciprocal Rank Fusion)
Vector scores (cosine similarity, range 0–1) and BM25 scores (unbounded, corpus-dependent) have different numerical ranges and distributions. Direct score averaging is meaningless. RRF operates on *ranks* rather than raw scores, making it robust to scale differences:

```
RRF score = Σ 1 / (k + rank)
```

The constant `k=60` (standard value from literature) balances the influence of top-ranked vs. lower-ranked results.

### Why a Cross-Encoder Reranker
The bi-encoder (embedding model) used for initial retrieval encodes query and document independently — it cannot model fine-grained query-document interactions. A cross-encoder processes `(query, document)` jointly through a single transformer, producing more accurate relevance judgments. This is especially important for legal text where subtle differences in wording change meaning significantly.

### MongoDB
- MongoDB stores normalized document metadata, traceable chunks, and embedding vectors together.
- MongoDB Atlas Vector Search can be enabled with `ATLAS_VECTOR_SEARCH=true`.
- Local MongoDB uses a NumPy cosine-similarity fallback, which is useful for development and small corpora.

### How Retrieval Is Evaluated
The evaluation script compares three systems on 24 queries across 5 categories (semantic, exact section, act-specific, case-related, terminology-heavy):

| Metric | Description |
|--------|-------------|
| Recall@K | Fraction of queries where at least one relevant document appears in top-K |
| MRR | Mean Reciprocal Rank — average of 1/rank of first relevant result |
| NDCG@5 | Normalised Discounted Cumulative Gain — measures ranking quality |

---

## Quickstart (local, no Docker)

```powershell
# 1. Copy env vars
Copy-Item .env.example .env
# Edit .env and set DATABASE_URL to your MongoDB local or Atlas connection string

# 2. Start MongoDB
docker compose up db -d

# 3. Install Python dependencies
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt

# 4. Start the API (MongoDB indexes are created on first startup)
uvicorn app.main:app --reload --app-dir backend

# 5. Start the frontend (in a separate terminal)
cd frontend
npm install
npm run dev
```

API docs: http://localhost:8000/docs  
Frontend: http://localhost:3000

---

## Production configuration

Set `APP_ENV=production` and provide an explicit `CORS_ORIGINS` allowlist in
`.env`. The API exposes `/api/live` for liveness and `/api/ready` for readiness;
readiness returns HTTP 503 when MongoDB is unavailable. Debug endpoints are
disabled in production. Rate limits, upload limits, logging level, model
providers, and MongoDB settings are configurable through `.env.example`.

The portable deployment shape is:

```
User -> Next.js frontend -> FastAPI API -> MongoDB Atlas
                                      -> embedding/reranker models
                                      -> configured LLM provider
```

Do not expose MongoDB publicly in production. The Compose database port is for
local development; use MongoDB Atlas or a private MongoDB network in production.

---

## Indexing Documents

### Step 1 — Process raw documents

```powershell
python scripts/process_documents.py
```

Reads `data/raw/sample/*.txt|json|pdf|html`, cleans and chunks each file, writes `data/processed/sample.jsonl`.

### Step 2 — Generate embeddings and index into MongoDB

```powershell
python scripts/index_documents.py
```

Reads JSONL files, generates embeddings (downloads model on first run), inserts documents and chunks into the database with vector embeddings. Idempotent — re-running updates existing records rather than duplicating them.

---

## Hybrid Search

### Via API

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the requirements for a valid contract?", "top_k": 5}'
```

Response includes multi-signal scores:

```json
{
  "query": "What are the requirements for a valid contract?",
  "results": [
    {
      "chunk_id": "sample_statute:0",
      "content": "A valid contract requires the following essential elements...",
      "document_id": "sample_statute",
      "title": "Sample Statute",
      "section": "Section 1. Essential Elements of a Valid Contract.",
      "page_start": 1,
      "page_end": 1,
      "document_type": "TXT",
      "source": "raw_file",
      "vector_score": 0.94,
      "keyword_score": 7.41,
      "rrf_score": 0.031,
      "rerank_score": 0.91,
      "final_rank": 1
    }
  ]
}
```

### Debug Endpoint

```bash
curl -X POST http://localhost:8000/api/search/debug \
  -H "Content-Type: application/json" \
  -d '{"query": "Section 302 murder", "top_k": 5}'
```

Exposes all retrieval stages: vector results, keyword results, fused results, reranked results, and timing information.

### Retrieval Evaluation

```powershell
python scripts/evaluate_retrieval.py
```

Runs the chunk-level golden queries in `data/evaluation/retrieval_gold.json` and
compares four approaches. If MongoDB is unavailable, the command reports
`NOT RUN` rather than treating unavailable retrieval as zero-quality results.

```
Method                  Recall@1    Recall@3    Recall@5    Recall@10  MRR        NDCG@5
---------------------------------------------------------------------------
Vector                  ...         ...         ...         ...        ...        ...
BM25                    ...         ...         ...         ...        ...        ...
Hybrid                  ...         ...         ...         ...        ...        ...
Hybrid + Reranker       ...         ...         ...         ...        ...        ...
```

---

## Evaluation

Evaluation assets live under `data/evaluation/` and use only the synthetic
documents shipped with this repository. Reusable metric functions cover
Recall@K, MRR, NDCG@5, citation validity, citation coverage, abstention, and
document-comparison citation namespaces.

Run the complete evaluation entry point:

```powershell
python scripts/run_all_evaluations.py
```

The workflow records configuration and writes
`reports/evaluation_report.json` and `reports/evaluation_report.md`. Retrieval
and RAG benchmarks require MongoDB with indexed chunks. If MongoDB is
unavailable, experiments are reported as **NOT RUN**, never as zero-quality
results. Automated citation metrics measure provenance and formatting, not
legal correctness; substantive grounding requires human review.

The development dashboard is available at `/evaluation`.

---

## How Retrieval Works

### Stage 1: Vector Search
1. Query is prefixed with `"query: "` (required by multilingual-e5) and encoded into a 1024-dimensional vector.
2. MongoDB Atlas Vector Search or the local NumPy fallback returns `vector_top_k` (default 20) closest chunks.
3. Each result gets a `vector_score` (cosine similarity) and `vector_rank`.

### Stage 2: Keyword Search (BM25)
1. Query is tokenised using a legal-aware tokeniser that preserves section numbers, case citations, and legal terms.
2. BM25 scoring over the indexed corpus returns `keyword_top_k` (default 20) results.
3. Each result gets a `keyword_score` (BM25 relevance).

### Stage 3: RRF Fusion
1. Results from both retrievers are merged using Reciprocal Rank Fusion.
2. Duplicate chunks (appearing in both lists) get combined scores.
3. The fused list is sorted by RRF score.

### Stage 4: Cross-Encoder Reranking
1. The top `rerank_top_k` (default 10) candidates are re-scored using a cross-encoder model.
2. Each `(query, chunk)` pair is processed jointly for more accurate relevance.
3. Final `final_top_k` (default 5) results are returned.

### Fallback Behaviour
- If MongoDB is unavailable: empty results (API stays up).
- If vector search fails: falls back to keyword-only or empty results.
- If keyword search fails: falls back to vector-only.
- If reranker is unavailable: returns RRF-ranked results without reranking.

---

## API Reference

All routes are available under both `/api` and `/api/v1`:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check (shows DB, embedding, reranker config) |
| POST | `/api/search` | Hybrid search with multi-signal scores |
| POST | `/api/search/debug` | Debug endpoint exposing all retrieval stages |
| POST | `/api/search/semantic` | Legacy vector-only semantic search |
| POST | `/api/chat` | Bounded LangGraph workflow with grounded source citations |
| POST | `/api/chat/debug` | Development-only graph execution summary |
| POST | `/api/documents/upload` | Validate, process, and index a document when MongoDB is available |
| POST | `/api/documents/process` | Process a document (returns chunks + metadata) |
| GET | `/api/documents` | List indexed documents |
| GET | `/api/documents/{id}` | Retrieve document metadata and chunk count |
| GET | `/api/documents/{id}/analysis` | Retrieve grounded document analysis |
| POST | `/api/documents/{id}/ask` | Ask a question restricted to one document |
| POST | `/api/documents/compare` | Compare exactly two indexed documents |

---

## Configuration

### Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_TOP_K` | `20` | Candidates from vector search |
| `KEYWORD_TOP_K` | `20` | Candidates from BM25 search |
| `RERANK_TOP_K` | `10` | Candidates sent to reranker |
| `FINAL_TOP_K` | `5` | Final results returned |
| `VECTOR_WEIGHT` | `0.6` | Weight for vector results (weighted fusion fallback) |
| `KEYWORD_WEIGHT` | `0.4` | Weight for keyword results (weighted fusion fallback) |
| `RRF_K` | `60` | RRF constant — higher = less emphasis on top ranks |

### Reranker

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_PROVIDER` | `""` | `"cross_encoder"` or `""` to disable |
| `RERANKER_MODEL` | `""` | HuggingFace model ID (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) |

### Grounded answer generation

The chat pipeline selects and formats retrieved evidence before generation. The
LLM receives retrieved documents as untrusted data, and the backend resolves
`SOURCE N` references back to actual chunk metadata before returning citations.
Questions without sufficient evidence are answered with an explicit abstention.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `stub` | `stub` for deterministic development or `openai` |
| `LLM_MODEL` | `placeholder` | Provider-specific model name |
| `LLM_TEMPERATURE` | `0.3` | Generation temperature |
| `LLM_MAX_TOKENS` | `2048` | Maximum generated tokens |
| `MAX_EVIDENCE_CHUNKS` | `8` | Maximum evidence passages supplied to the LLM |
| `MAX_CONTEXT_TOKENS` | `6000` | Approximate context budget |
| `MIN_RERANK_SCORE` | `-10.0` | Minimum score accepted for evidence selection |

Confidence is rule-based: high confidence requires valid citations and strong
retrieval evidence; otherwise the response is medium or low. This is not a
measure of legal correctness.

### Controlled LangGraph workflow

The chat route now runs a bounded workflow: query analysis, retrieval strategy
selection, retrieval, deterministic evidence evaluation, optional one-time
query refinement, grounded generation, and citation validation. Retrieval is
limited by `MAX_RETRIEVAL_ATTEMPTS`; failed or insufficient paths abstain rather
than asking the LLM to invent an answer. This is a controlled workflow, not an
unrestricted autonomous agent.

### Embeddings

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `sentence_transformer` | `sentence_transformer` or `stub` |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | HuggingFace model name |
| `EMBEDDING_DIMENSION` | `1024` | Must match the model's output dimension |
| `EMBEDDING_BATCH_SIZE` | `32` | Chunks per embedding batch |

### Database

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `mongodb://localhost:27017/pramaan` | MongoDB connection string |
| `ATLAS_VECTOR_SEARCH` | `false` | Enable MongoDB Atlas vector search |

---

## Folder Structure

```
backend/
  app/
    api/         HTTP routes
    db/          Database layer (connection, models, repository)
    rag/
      embeddings/    Embedding service (sentence-transformer + stub)
      indexing/      DocumentIndexer — JSONL → MongoDB
      retrieval/     VectorRetriever, KeywordRetriever, HybridRetriever
      reranking/     CrossEncoderReranker
      services/      LegalRetrievalService (high-level coordinator)
      loaders/       PDF/TXT/HTML/JSON document loaders
    services/    Orchestration services (search, chat, document processing)
    schemas/     Pydantic request/response models
    config.py    Settings (env vars)
  tests/
    test_keyword_retrieval.py   BM25 tokeniser and index tests
    test_hybrid_retrieval.py    RRF fusion tests
    test_reranking.py           Reranker tests
    test_pipeline.py            Pipeline data flow tests
    test_fallbacks.py           Graceful degradation tests
    test_embeddings.py          Embedding service tests
    test_ingestion.py           Loader and chunking tests
    test_retrieval.py           MongoDB retrieval tests

frontend/
  app/page.tsx   Search UI with hybrid retrieval scores

scripts/
  process_documents.py   Raw → cleaned chunks → JSONL
  index_documents.py     JSONL → embeddings → MongoDB
  evaluate_retrieval.py  Vector vs Hybrid vs Hybrid+Reranker comparison

data/
  raw/sample/         Synthetic sample documents
  processed/          JSONL output from process_documents.py
  evaluation/         24 evaluation queries (5 categories)
```

---

## Running Tests

```powershell
# Unit tests (no DB required)
pytest backend/tests -q

# With MongoDB retrieval tests
$env:TEST_DATABASE_URL = "mongodb://localhost:27017/pramaan_test"
pytest backend/tests -v

# Specific test modules
pytest backend/tests/test_keyword_retrieval.py -v
pytest backend/tests/test_hybrid_retrieval.py -v
pytest backend/tests/test_reranking.py -v
pytest backend/tests/test_fallbacks.py -v
```

---

## Full Docker Compose

```powershell
docker compose up --build
```

Starts MongoDB (with healthcheck), FastAPI backend, and Next.js frontend.
The API container runs as a non-root user and exposes a liveness healthcheck.

---

## Current production-readiness limits

Authentication and authorization are intentionally not included yet. Place the
API behind an authenticated gateway before exposing it publicly. MongoDB and
external LLM/model services are required for complete end-to-end verification;
local tests use stubs or skip infrastructure-dependent checks.
