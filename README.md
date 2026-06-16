# 🧠 DomainGPT

**Enterprise-grade RAG + LoRA AI Chatbot** — built to production standards.

> Combines Retrieval-Augmented Generation (RAG) with LoRA fine-tuning on open-weight LLMs (Llama 3 / Qwen / Mistral), served via Groq or local vLLM, backed by Pinecone vector search, FastAPI, Celery, PostgreSQL, and Redis.

---

## Architecture

```
Streamlit UI  ──►  FastAPI Gateway (Auth · Rate Limit · Logging)
                        │                    │
                   Chat Service         Upload API
                        │                    │
                  RAG Orchestrator    Document Parser
                  (LangChain/Graph)   (PDF·DOCX·TXT·OCR)
                        │                    │
                  Embedding Model ◄──────────┘
                        │
                  Pinecone Index
                        │
               LoRA Fine-tuned LLM  (Groq / local)
                        │
                Final grounded answer + citations
```

---

## Quick Start

### 1. Clone & configure

```bash
git clone https://github.com/yourname/DomainGPT.git
cd DomainGPT
cp .env.example .env
# Edit .env and fill in your API keys
```

### 2. Run with Docker Compose (recommended)

```bash
cd docker
docker-compose up --build
```

| Service     | URL                     |
|-------------|-------------------------|
| FastAPI     | http://localhost:8000   |
| Swagger UI  | http://localhost:8000/docs |
| Streamlit   | http://localhost:8501   |
| Flower      | http://localhost:5555   |
| Grafana     | http://localhost:3000   |
| Prometheus  | http://localhost:9090   |

### 3. Initialize the database & Pinecone index

```bash
python scripts/init_db.py
python scripts/create_pinecone_index.py
```

---

## Running Locally (without Docker)

### Prerequisites

- Python 3.10+
- PostgreSQL 15+
- Redis 7+
- Tesseract OCR (`brew install tesseract` / `apt install tesseract-ocr`)

```bash
pip install -e .
uvicorn apps.api.main:app --reload --port 8000
```

**Celery worker** (separate terminal):
```bash
celery -A apps.workers.celery_app worker --loglevel=info
```

**Streamlit UI** (separate terminal):
```bash
streamlit run apps/ui/streamlit_app.py
```

---

## API Endpoints

### Authentication
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/auth/register` | Register a new user |
| POST | `/api/v1/auth/login` | Login and get JWT token |

### Documents
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/documents/upload` | Upload PDF/DOCX/TXT/image |
| GET | `/api/v1/documents/{id}` | Check processing status |

### Chat
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/chat/` | Send question, get grounded answer |
| POST | `/api/v1/chat/stream` | Streaming (SSE) response |
| GET | `/api/v1/chat/conversations` | List conversations |

### System
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness probe |
| GET | `/health/ready` | Readiness probe |

---

## Keys You Need

### Required APIs

| Service | How to get | Cost |
|---------|-----------|------|
| **Groq** | [console.groq.com](https://console.groq.com) — free tier | Free (rate-limited) |
| **Pinecone** | [pinecone.io](https://pinecone.io) — free starter tier | Free (1 index) |

### Optional APIs

| Service | Use case |
|---------|---------|
| **OpenAI** | Fallback embeddings (not required — BGE runs locally) |
| **HuggingFace** | Download gated models for local LoRA training |
| **AWS S3** | Document storage (local filesystem used if not set) |

---

## LoRA Fine-tuning

### 1. Prepare your dataset

Create a `.jsonl` file:
```json
{"instruction": "Summarize this document", "input": "...", "output": "..."}
```

```bash
python -m apps.training.prepare_dataset \
  --input ./datasets/my_data.jsonl \
  --output ./datasets/processed
```

### 2. Train

```bash
python -m apps.training.lora_train \
  --dataset ./datasets/processed \
  --output ./models/lora/adapter \
  --epochs 3
```

### 3. Use the fine-tuned model locally

```bash
# In .env:
USE_LOCAL_LLM=true
LORA_ADAPTER_PATH=./models/lora/adapter
```

### 4. Evaluate

```bash
python -m apps.training.evaluate \
  --eval-file ./datasets/eval_set.jsonl \
  --tenant-id demo
```

---

## Project Structure

```
DomainGPT/
├── apps/
│   ├── api/           # FastAPI gateway, routers, auth, models
│   ├── ingestion/     # PDF, DOCX, TXT, image parsers + chunking
│   ├── rag/           # Retriever, hybrid search, reranker, pipeline
│   ├── llm/           # Inference abstraction + LoRA loader
│   ├── training/      # Dataset prep, LoRA training, evaluation
│   ├── workers/       # Celery tasks
│   └── ui/            # Streamlit frontend
├── configs/           # YAML configs
├── datasets/          # Training and eval data
├── models/            # Base and LoRA adapters
├── docker/            # Dockerfiles + Compose
├── kubernetes/        # K8s deployment + HPA
├── monitoring/        # Prometheus + Grafana
├── scripts/           # DB init, Pinecone setup
├── tests/             # pytest suite
└── .github/workflows/ # CI/CD
```

---

## Tech Stack

| Layer | Choice |
|-------|--------|
| API | FastAPI + Uvicorn |
| LLM inference | Groq (Llama 3 70B) or local vLLM |
| Fine-tuning | PEFT LoRA + TRL SFTTrainer |
| Embeddings | BGE-large-en (local, no API cost) |
| Vector DB | Pinecone |
| Orchestration | LangChain + LangGraph |
| Reranking | FlashRank (cross-encoder) |
| Hybrid search | BM25 + Dense (RRF fusion) |
| Task queue | Celery + Redis |
| Database | PostgreSQL (async SQLAlchemy) |
| Cache | Redis |
| Monitoring | Prometheus + Grafana |
| Containers | Docker + Kubernetes |
| CI/CD | GitHub Actions |
