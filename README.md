# Pramaan — Indian Legal AI Research Assistant

Source-first semantic search for Indian statutes, rules, and judgments. Built on FastAPI, Next.js, MongoDB, and configurable embedding providers.

> **Disclaimer**: Pramaan is an AI research assistant and is **not a substitute for professional legal advice**. All sample documents are synthetic.

---

## Architecture

```
General question (home UI sends one request):
  POST /api/chat  → routes.chat → LangGraph → LegalRetrievalService
                  → vector + BM25 → RRF → optional reranker
                  → LegalRAGService → evidence/context → configured LLM
                  → citation validation → ChatResponse
  ChatResponse.search_results supplies the Evidence tab from the graph's
  final retrieval attempt. Standalone POST /api/search remains available:
    routes.search → LegalRetrievalService
                  → vector + BM25 → RRF → optional reranker → SearchResponse

Document operations:
  POST /api/documents/{id}/ask → filtered LegalRetrievalService
                               → LegalRAGService → ChatResponse
  POST /api/documents/compare  → ComparisonService → MongoDB chunks
  POST /api/documents/upload   → DocumentProcessingService
                               → DocumentIndexer → embeddings + MongoDB

Offline ingestion:
  data/raw → scripts/process_documents.py → data/processed/*.jsonl
           → scripts/index_documents.py → DocumentIndexer → MongoDB
```

The classic retrieval endpoint and the bounded LangGraph answer workflow are
separate entry points that share `LegalRetrievalService`. The home page uses
the graph's retrieved results for both the answer and evidence tab, avoiding a
second retrieval request. Document-specific questions use the same retrieval
and RAG services directly, with retrieval restricted to one document.
Comparison currently returns a metadata overview and similarities; analysis is
extractive over indexed chunks.

The upload response keeps `status: "accepted"` for compatibility and adds
`indexing_status` (`indexed`, `skipped`, or `failed`) so callers can distinguish
file acceptance from successful indexing.

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
RRF score = Σ weight(source) / (k + rank)
```

`RRF_K` defaults to `60`; `VECTOR_WEIGHT` and `KEYWORD_WEIGHT` default to `0.6` and `0.4`. The weights apply to dense and BM25 rank contributions respectively. Duplicate chunks contribute at most once per source list, and tied fused scores are ordered by document ID and chunk ID.

### Why a Cross-Encoder Reranker
The bi-encoder (embedding model) used for initial retrieval encodes query and document independently — it cannot model fine-grained query-document interactions. A cross-encoder processes `(query, document)` jointly through a single transformer, producing more accurate relevance judgments. This is especially important for legal text where subtle differences in wording change meaning significantly.

### MongoDB
- MongoDB stores normalized document metadata, traceable chunks, and embedding vectors together.
- MongoDB Atlas Vector Search can be enabled with `ATLAS_VECTOR_SEARCH=true`.
- Local MongoDB uses a NumPy cosine-similarity fallback, which is useful for development and small corpora.

### How Retrieval Is Evaluated
The evaluation command scores the configured hybrid retrieval path against the
explicit relevance labels in `data/evaluation/retrieval_gold.json`. It does not
run separate vector-only, BM25-only, or reranker ablation benchmarks.

| Metric | Description |
|--------|-------------|
| Recall@K | Fraction of labeled relevant chunks/documents found in the top-K results |
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
`.env` (copy and edit `.env.example`). Settings are validated when the backend
starts; an OpenAI-compatible provider requires `OPENAI_API_KEY`, production
requires a non-default CORS origin, and upload/request limits and retrieval
parameters must be valid. The API exposes `/api/live` for liveness and
`/api/ready` for readiness; readiness returns HTTP 503 when MongoDB is
unavailable. Debug endpoints are disabled in production.

The portable deployment shape is:

```
User -> Next.js frontend -> FastAPI API -> MongoDB Atlas
                                      -> embedding/reranker models
                                      -> configured LLM provider
```

The API currently has **no authentication or authorization**. In particular,
document upload, listing, metadata, analysis, document-specific questions,
comparison, and search routes are not protected by a user identity or tenant
boundary. Do not expose this API or its document data to untrusted networks;
put it behind an authenticated/authorized gateway and implement/test access
control before handling private legal documents. No protected multi-user
behavior is implemented or tested.

Docker Compose is a local development stack, not a production deployment
configuration. MongoDB has no authentication configured in this Compose file,
though its host port is bound to loopback. Use MongoDB Atlas or a private,
authenticated MongoDB deployment for hosted environments.

There is no GitHub Actions deployment workflow. The workflow named `CI` only
checks backend tests, frontend API contract tests, and a frontend production
build; it does not publish images, deploy services, or verify a live deployment.

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

Run both commands from the repository root. For a host-run backend, set
`DATABASE_URL` to a host-reachable MongoDB URI (the `.env.example` default is
`localhost`). For the Compose API container, set `COMPOSE_DATABASE_URL` to a
container-reachable MongoDB URI (the default is `mongodb://db:27017/pramaan`).
Indexing must complete successfully before documents can be searched; an
upload accepted while MongoDB is unavailable is stored locally and reports
`indexing_status: skipped`, not indexed.

### Optional: small Indian Supreme Court judgment sample

The public [Indian Supreme Court Judgments dataset](https://registry.opendata.aws/indian-supreme-court-judgments/)
provides individual English PDFs at
`data/pdf/year=<YEAR>/english/` and matching JSON metadata at
`metadata/json/year=<YEAR>/`. Do not download the yearly archives or the full
corpus. For example, list no more than 25 individual PDF keys:

```powershell
aws s3api list-objects-v2 --bucket indian-supreme-court-judgments `
  --prefix data/pdf/year=2025/english/ --max-items 25 --no-sign-request `
  --query "Contents[].Key" --output text
```

Download each selected PDF and its same-stem JSON metadata into:
`data/raw/supreme_court_english/pdfs/` and
`data/raw/supreme_court_english/metadata/`. Then use the shared parser and
indexer:

```powershell
python scripts/process_documents.py data/raw/supreme_court_english/pdfs `
  --metadata-dir data/raw/supreme_court_english/metadata `
  --output data/processed/supreme_court_english/supreme_court_english.jsonl
python scripts/index_documents.py --processed-dir data/processed/supreme_court_english
```

The processor keeps source filenames, page boundaries, available case metadata,
the original metadata JSON, and source object URLs. It records the dataset's
eCourts attribution and CC BY 4.0 license in document metadata. Local source
PDFs are excluded from Git; obtain them again using the documented object paths.
Atlas vector retrieval additionally requires `ATLAS_VECTOR_SEARCH=true` and a
Search index named `embedding_index` on `document_chunks.embedding`, with the
configured embedding dimension.

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

Runs the unified evaluation workflow in retrieval-only mode. It scores the
configured hybrid retrieval service, not separate vector/BM25 ablations.

---

## Evaluation

Evaluation data under `data/evaluation/` is versioned JSON. Retrieval examples
provide a query and relevant chunk IDs and/or document IDs; any supplied IDs
must resolve to the connected, indexed corpus before the example is scored.
The answer-observation dataset contains explicit abstention expectations but
does not contain judged reference answers.

Run the workflow from the repository root:

```powershell
python scripts/run_evaluation.py
```

Optional focused modes:

```powershell
python scripts/run_evaluation.py --retrieval-only
python scripts/run_evaluation.py --answers-only
```

The legacy commands `scripts/run_all_evaluations.py`,
`scripts/evaluate_retrieval.py`, `scripts/evaluate_rag.py`, and
`scripts/generate_evaluation_report.py` delegate to this same runner. None
creates a report by inserting placeholder metric values.

Retrieval needs a reachable MongoDB instance with the indexed synthetic corpus
and the project's configured embedding/retrieval dependencies. Answer
observations run the bounded LangGraph `/api/chat` workflow; the default
`LLM_PROVIDER=stub` permits deterministic plumbing observations but blocks
semantic answer-quality evaluation. To test a real provider, configure a
supported `LLM_PROVIDER` (`openai`, `openai-compatible`, or `openrouter`),
`LLM_MODEL`, and the matching provider API key. OpenRouter uses
`LLM_PROVIDER=openrouter`, `LLM_MODEL=openrouter/free`, and
`OPENROUTER_API_KEY`. Live provider output and approximate vector search remain
sources of nondeterminism.

The command records the UTC run time, dataset version/count/SHA-256, provider
and model configuration, retrieval parameters, request latencies, failures,
empty results, abstentions, inline citation-ID resolution, and citation
identifier/provenance validity. Retrieval Recall@k is the fraction of
dataset-labeled relevant chunks/documents found in the first k results; MRR and
NDCG@5 use those same relevance labels. Invalid or absent labels are skipped
or block scoring rather than becoming misses. A missing MongoDB, empty dataset,
or absent indexed labels produces `NOT RUN` or `BLOCKED`, not numeric zero
benchmark results.

Citation checks verify source-ID format, response citation mapping, and
membership of each citation's document/chunk pair in evidence returned to the
answer workflow. They do **not** verify that evidence semantically supports a
legal claim. Required-term checks, where provided by a dataset, are literal
string checks only. Answer correctness, groundedness, document analysis, and
document comparison are not scored by this runner without judged references
and an explicitly identified evaluator; they are reported as blocked or not
run. Ordinary unit tests verify implementation behavior, not legal answer
quality.

The command writes `reports/evaluation_report.json` and
`reports/evaluation_report.md`; the development dashboard at `/evaluation`
shows only values present in that report.

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
2. Each rank contributes its configured source weight divided by `RRF_K + rank`; the default weights are `0.6` dense and `0.4` BM25.
3. A chunk appearing in both lists is deduplicated and receives both contributions; duplicate appearances within one list count only once.
4. The fused list is sorted by weighted RRF score, with document ID and chunk ID as deterministic tie-breakers.

### Stage 4: Cross-Encoder Reranking
1. Reranking is enabled only when `RERANKER_PROVIDER=cross_encoder` and `RERANKER_MODEL` is set; otherwise fused order is retained.
2. The top `rerank_top_k` (default 10) candidates are re-scored using the configured cross-encoder model.
3. Each `(query, chunk)` pair is processed jointly.
4. Final results preserve source-document, chunk, section, page, and source-reference metadata.

### Context Assembly for Answers
The RAG service passes retrieved results to `EvidenceSelector`, which applies the configured minimum rerank score, maximum evidence chunk count, and approximate context-token limit. If all candidates fall below the score threshold, the current selector warns and uses the top candidates anyway. Selected chunks receive stable `[SOURCE N]` IDs; `ContextBuilder` includes their text and available document/section/page/source metadata in the prompt.

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
| `VECTOR_WEIGHT` | `0.6` | Dense/vector contribution to weighted RRF |
| `KEYWORD_WEIGHT` | `0.4` | BM25/lexical contribution to weighted RRF |
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

Citation verification is deterministic and limited to provenance: inline
`[SOURCE N]` references must be canonical, must have been supplied in the
selected evidence, and must resolve to a non-empty retrieved chunk tied to an
existing stored document. Returned document/chunk IDs and location metadata are
server-derived. This check does not prove that a passage semantically entails
or legally supports every sentence or claim; no such entailment evaluation is
implemented.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `stub` | `stub`, `openai`, or `openrouter` |
| `LLM_MODEL` | `placeholder` | Provider-specific model name |
| `OPENAI_API_KEY` | unset | Required when `LLM_PROVIDER=openai` |
| `OPENROUTER_API_KEY` | unset | Required when `LLM_PROVIDER=openrouter` |
| `LLM_BASE_URL` | unset | Optional endpoint for an OpenAI-compatible service |
| `LLM_TEMPERATURE` | `0.3` | Generation temperature |
| `LLM_MAX_TOKENS` | `2048` | Maximum generated tokens |
| `MAX_EVIDENCE_CHUNKS` | `8` | Maximum evidence passages supplied to the LLM |
| `MAX_CONTEXT_TOKENS` | `6000` | Approximate context budget |
| `MIN_RERANK_SCORE` | `-10.0` | Minimum score accepted for evidence selection |

Confidence is rule-based: high confidence requires valid citations and strong
retrieval evidence; otherwise the response is medium or low. This is not a
measure of legal correctness.

The default `stub` provider returns clearly labeled mock output; it is not real
legal-answer generation. To use OpenAI, configure `LLM_PROVIDER=openai`,
`LLM_MODEL=<your model>`, and `OPENAI_API_KEY=<your key>`. To use OpenRouter,
configure `LLM_PROVIDER=openrouter`, `LLM_MODEL=openrouter/free`, and
`OPENROUTER_API_KEY=<your key>`. OpenRouter requests use
`https://openrouter.ai/api/v1` through the existing OpenAI SDK integration;
`LLM_BASE_URL` is not required for this provider. The chat response includes
`generation_mode` so clients can distinguish generated output from the stub,
evidence-only fallback, and abstention. Provider failures do not switch to the
stub; the workflow records generation failure and abstains.

### Controlled LangGraph workflow

The chat route runs `analyze_query → select_retrieval_strategy →
retrieve_evidence → evaluate_evidence`. Sufficient evidence continues through
`generate_answer → validate_answer → finalize`; otherwise, retrieval can be
broadened once and evaluated again before `abstain`. The configured
`MAX_RETRIEVAL_ATTEMPTS` controls this count, with a hard graph ceiling of two
total attempts. Retrieval and generation errors are recorded as safe workflow
status and lead to bounded retry or abstention. Citation references are
validated in the answer and key points (not merely in the model's JSON citation
metadata) before the generated answer is accepted. If generation is unavailable,
the RAG service may return clearly labeled `evidence_only` passages; invalid
structured output or failed citation checks abstain.

The checked-in default is `LLM_PROVIDER=stub`. Stub output is deterministic mock
content and is not real legal analysis. The chat response's `generation_mode`
identifies `stub`, `openai-compatible`, `openrouter`, `evidence_only`, `failed`,
or `not_run`.
The repository tests mock the provider and do not constitute a live LLM
integration test.

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
| `COMPOSE_DATABASE_URL` | `mongodb://db:27017/pramaan` | API-container URI used by Docker Compose |
| `DATABASE_NAME` | `pramaan` | Database name for local indexing |
| `ATLAS_VECTOR_SEARCH` | `false` | Enable MongoDB Atlas vector search |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated exact browser origins |
| `MAX_UPLOAD_SIZE_MB` | `25` | Maximum file payload accepted by upload/process endpoints |
| `MAX_REQUEST_BODY_MB` | `30` | Maximum complete POST/PUT/PATCH body |
| `RATE_LIMIT_REQUESTS` | `60` | Per-client in-process request limit |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate-limit window |

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
  run_evaluation.py      Dataset-driven retrieval/answer observations

data/
  raw/sample/         Synthetic sample documents
  processed/          JSONL output from process_documents.py
  evaluation/         Versioned relevance and abstention datasets
```

---

## Running Tests

```powershell
# Unit/integration tests (MongoDB-specific tests skip unless opted in)
python -m pytest backend/tests -q

# Optional: use a dedicated disposable test database. Tests drop their test
# collections during cleanup; never point this at an application database.
$env:TEST_DATABASE_URL = "mongodb://localhost:27017/pramaan_test"
python -m pytest backend/tests/test_retrieval.py -v

# Frontend request/response contract checks and production build
Set-Location frontend
npm ci
npm run test:api
npx tsc --noEmit
npm run build
```

The backend test fixture redirects `DATABASE_URL` to `TEST_DATABASE_URL`, or
to an intentionally unavailable local test URI when the opt-in variable is
unset. It does not use a developer's configured application database by
default. CI runs backend tests without `TEST_DATABASE_URL` (MongoDB-dependent
tests are skipped), plus frontend API tests and a production build.

---

## Full Docker Compose

```powershell
docker compose up --build
```

Starts MongoDB (with healthcheck), FastAPI backend, and Next.js frontend.
The API container runs as a non-root user and exposes a liveness healthcheck.

---

## Current production-readiness limits

No authentication/authorization or per-user document isolation exists. Routes
for upload, list, metadata, analysis, document questions, comparison, and
search are public unless an external gateway protects them. Do not expose the
API to untrusted networks or process private legal documents until access
controls are implemented and tested.

GitHub Actions runs verification only; there is no automated deployment
workflow. Docker Compose is a local development setup, not a hardened
production deployment. The checked-in evaluation report records retrieval
as blocked because its source labels do not resolve in the indexed corpus,
marks semantic answer quality blocked because the provider is `stub` and
there are no judged references, and does not run document analysis/comparison
evaluation. See `reports/evaluation_report.md` for its timestamp and measured
operational observations.
