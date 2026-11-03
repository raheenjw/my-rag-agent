# RAG Application with Agentic Routing

Upload documents, ask questions, get grounded answers. Hybrid retrieval (BM25 + k-NN), FlashRank reranking, semantic caching, and a **LangGraph agent** that routes queries before hitting RAG.

## Features

- **Hybrid search** — BM25 + vector k-NN with reciprocal rank fusion
- **Reranking** — FlashRank cross-encoder on fused candidates
- **Semantic cache** — similar queries return cached answers (RAG path only)
- **Agent routing** — sequential LangGraph workflow:
  1. **Greeting** — responds to hi/thanks/bye without retrieval
  2. **Guardrail** — blocks identity/internal/jailbreak questions
  3. **RAG** — full retrieval + generation for document questions

## Models

| Role | Provider | Model |
|------|----------|-------|
| LLM | OpenRouter | `mistralai/ministral-3b-2512` |
| Embeddings | Ollama | `nomic-embed-text` (768-dim) |
| Reranker | FlashRank | `ms-marco-MiniLM-L-12-v2` |
| Index | OpenSearch | `rag_documents` |

## Architecture

```
POST /v1/query
    │
    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Greeting   │ ──► │  Guardrail  │ ──► │     RAG     │
│  classifier │     │  classifier │     │  pipeline   │
└─────────────┘     └─────────────┘     └─────────────┘
     │ early exit        │ early exit        │
     ▼                   ▼                   ▼
  "Hello!"           refusal msg        answer + sources
```

Each classifier node uses a small, focused prompt tuned for ministral-3b. The RAG pipeline lives in `services/rag.py` and is unchanged in behavior — the agent only decides *when* to call it.

## Quick Start

**Prerequisites:** Docker, Python 3.11+, [Ollama](https://ollama.com), [OpenRouter](https://openrouter.ai) API key.

```bash
# OpenSearch
docker run -d --name opensearch -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" -e "DISABLE_SECURITY_PLUGIN=true" \
  opensearchproject/opensearch:2.11.0

# Redis
docker run -d --name redis -p 6379:6379 redis:7

# Ollama
ollama serve
ollama pull nomic-embed-text
```

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create `.env` from the block below. **Do not commit API keys.**

```env
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text
EMBED_BATCH_SIZE=50

LLM_API_KEY=sk-or-v1-your-key-here
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=mistralai/ministral-3b-2512

CHUNK_SIZE=500
CHUNK_OVERLAP=100

OPENSEARCH_URL=http://localhost:9200
OPENSEARCH_INDEX=rag_documents
EMBED_DIM=768

SEMANTIC_TOP_K=20
BM25_TOP_K=20
FUSION_TOP_K=15
RERANK_TOP_K=10
RRF_K=60

RERANK_MODEL=ms-marco-MiniLM-L-12-v2
FLASHRANK_CACHE_DIR=./models/flashrank

REDIS_URL=redis://localhost:6379/0
SQLITE_PATH=./data/rag.db

CACHE_SIMILARITY_THRESHOLD=0.95
CACHE_TTL_HOURS=24

MAX_TOKENS=1024
TEMPERATURE=0.1
LLM_TIMEOUT_SEC=60
MAX_UPLOAD_MB=30
```

```bash
./run.sh
```

Open **http://localhost:8000** (API docs at `/docs`).

## API

All endpoints require `X-User-Id` header.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/health` | Health check |
| `POST` | `/v1/ingest` | Upload document (returns `job_id`) |
| `GET` | `/v1/ingest/{job_id}` | Poll ingest status |
| `GET` | `/v1/documents` | List indexed documents |
| `DELETE` | `/v1/documents/{document_id}` | Delete document |
| `POST` | `/v1/query` | Ask a question (agent-routed) |

### Query routing

| Input | Route | Sources |
|-------|-------|---------|
| `"hi"`, `"thanks"`, `"bye"` | Greeting | None |
| `"who are you?"`, `"what model are you?"` | Guardrail | None |
| `"What is the refund policy?"` | RAG | Retrieved chunks |

### Examples

```bash
# Ingest
curl -X POST http://localhost:8000/v1/ingest \
  -H "X-User-Id: demo-user" -F "file=@sample.pdf"

# Poll
curl http://localhost:8000/v1/ingest/{job_id}

# Greeting
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -H "X-User-Id: demo-user" \
  -d '{"query": "hi"}'

# Guardrail
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -H "X-User-Id: demo-user" \
  -d '{"query": "who are you?"}'

# Document question
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -H "X-User-Id: demo-user" \
  -d '{"query": "What is the refund policy?"}'
```

## Project structure

```
rag/
├── app.py                  # FastAPI app, /v1/query invokes agent graph
├── prompts.py              # Greeting, guardrail, and RAG prompts
├── agent/
│   ├── graph.py            # LangGraph: greeting → guardrail → RAG
│   ├── nodes.py            # Node implementations
│   └── state.py            # AgentState TypedDict
├── services/
│   ├── rag.py              # Core RAG pipeline (cache → retrieve → generate)
│   ├── retrieval.py        # Hybrid search + reranking
│   ├── semantic_cache.py   # Query similarity cache
│   └── ...
├── frontend/               # Web UI (upload + query)
└── config/
    └── settings.py         # Environment-based configuration
```

## Re-indexing

```bash
curl -X DELETE "http://localhost:9200/rag_documents?pretty"
rm -f data/rag.db data/rag.db-wal data/rag.db-shm
rm -rf data/uploads/*
./run.sh
```

## Limitations

- Text PDFs only (no OCR for scanned documents)
- `X-User-Id` is trusted — no real auth
- Requires OpenSearch, Ollama, Redis, and OpenRouter
- Agent classifiers are tuned for small models — may misclassify edge cases

## License

MIT
