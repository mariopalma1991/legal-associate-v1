---
title: Legal Associate — Licitaciones Públicas de Chihuahua
emoji: 🏛️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.18.0"
app_file: app/app.py
pinned: false
---

# Legal Associate — Asistente de Licitaciones Públicas de Chihuahua

A RAG-powered procurement assistant that monitors all active public tenders from the State of Chihuahua, Mexico. Users can search and explore them in natural language and get detailed answers about requirements and technical specifications by querying the official PDF documents.

---

## Why use this instead of the government portal?

The Chihuahua State procurement portal lists tenders, but finding and qualifying opportunities requires opening each record manually, downloading PDFs one by one, and reading through 50–100 page documents to extract the handful of facts that actually matter — participation cost, deadlines, technical requirements, required documents.

**This assistant compresses that research from hours to seconds:**

| Task | Government portal | This assistant |
|---|---|---|
| Find relevant tenders | Keyword search + manual scanning | Natural language: _"road paving tenders"_ |
| Know which ones are urgent | Click each, check date manually | Color-coded urgency on every result (🔴 🟡 🟢) |
| Understand requirements | Download + read a 60-page PDF | Ask: _"what documents do I need to participate?"_ |
| Get technical specs | Find and read the Anexo Técnico | Click **📦 Technical specs** — answer in seconds |
| Know participation cost | Open the Convocatoria PDF | Shown on the detail card automatically |
| Check clarification outcomes | Download the Acta PDF | Click **🗣️ Clarification Meeting** |
| Resume research next day | Start over from scratch | Conversations are saved and resumable |

The target user is a sales representative or supplier who monitors government procurement to find business opportunities. Every hour spent navigating PDFs is an hour not spent preparing a competitive proposal. This assistant handles the reading — you handle the bid.

---

## What it does

1. **Discovery** — ask "what road-paving tenders are open?" and get a ranked list with urgency indicators and deadlines.
2. **Anchor** — select a specific tender; all subsequent questions are scoped to that record.
3. **Detail retrieval** — ask "what are the technical requirements?" and the assistant retrieves relevant chunks from official PDFs and streams a synthesized answer.
4. **Quick-action buttons** — one-click queries for Bases, Convocatoria, Clarification Meeting, Requirements, Technical Specs, Required Documents, and Economic Conditions.
5. **Right-panel detail card** — structured view of key tender metadata (dates, cost, location, contracting entity) shown alongside the chat.
6. **Bilingual UI** — toggle between Spanish and English at any time; all labels, buttons, and prompts switch instantly.
7. **Conversation memory** — sessions are persisted in PostgreSQL; resume any past conversation from the sidebar.

---

## Architecture

```
ingestion/
  fetch_vigentes.py    → discover new Vigente licitaciones from XLS export
  ingest.py            → scrape full metadata from portal HTML (async, 50 workers)
  download_docs.py     → download PDFs → Supabase Storage (10 workers)
  chunk_docs.py        → parse PDFs → text chunks (LlamaParse + PyMuPDF fallback)
  embed_index.py       → embed chunks (Cohere multilingual-v3.0 / OpenAI)
  refresh_status.py    → nightly re-check of Vigente records; detect status changes
  extract_licitacion.py → HTML parser shared by ingest + refresh
  _pipeline.py         → tracking helper used by ingestion scripts

shared/
  _storage.py          → Supabase S3 helper (upload/download PDFs)

infra/
  setup_db.py          → create / reset the database schema
  cleanup_storage.py   → delete Storage files and chunks for Terminado records
  upload_existing_docs.py → one-time bulk upload of local docs/ to Storage

app/
  app.py               → Gradio chat UI, one-brain router, streaming synthesis
  query.py             → dense retrieval + Cohere reranking + metadata search

eval/
  generate_eval_set.py → generate evaluation questions with Claude Haiku
  eval_retriever.py    → benchmark five retrieval strategies
  eval_pipeline.py     → end-to-end answer quality scoring with LLM judge
  eval_set.json        → 189 retriever + 100 pipeline evaluation questions
  eval_results_*.md    → retriever benchmark results by chunk config
  eval_pipeline_summary_*.md → pipeline evaluation results by model combination

run_pipeline.py        → orchestrator: runs all ingestion steps in sequence
```

Database: PostgreSQL (Supabase) with pgvector for HNSW cosine-distance index, tsvector for BM25 full-text search, and a `chunks` table that stores one embedding column per model so all models share the same chunked text.

### Table schemas

**`licitaciones`** — one row per tender, scraped from the portal HTML

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Portal licitacion ID |
| `url` | TEXT | Source page URL |
| `numero_procedimiento` | TEXT | Procedure code |
| `tipo_procedimiento` | TEXT | Procedure type |
| `licitacion_status` | TEXT | Live status: `Vigente \| Terminado \| Cancelado \| En seguimiento` |
| `ente_contratante` | TEXT | Contracting entity |
| `ente_solicitante` | TEXT | Requesting entity |
| `materia` | TEXT | Sector (obras, servicios, adquisiciones…) |
| `concepto_contratacion` | TEXT | Short description |
| `descripcion` | TEXT | Full description |
| `fecha_convocatoria` | TEXT | Publication date |
| `fecha_junta_aclaraciones` | TEXT | Clarification meeting date |
| `fecha_apertura` | TEXT | Proposal deadline |
| `costo_participacion` | TEXT | Participation cost |
| `pipeline_status` | TEXT | Ingestion stage: `discovered → scraped → downloaded → chunked → indexed` |
| `last_checked_at` | TIMESTAMPTZ | Last time `refresh_status.py` ran for this record |

**`documents`** — one row per PDF linked from a licitacion page

| Column | Type | Description |
|---|---|---|
| `id` | UUID PK | |
| `licitacion_id` | INTEGER FK | Parent licitacion |
| `tipo` | TEXT | Document type: `Convocatoria \| Bases \| Bases - Anexo 1 \| …` |
| `url` | TEXT | Download URL |
| `local_path` | TEXT | `docs/{licitacion_id}/{tipo}.pdf` |
| `status` | TEXT | `pending → downloaded → chunked → indexed` |
| `parser` | TEXT | Parser used: `llamaparse \| pymupdf` |
| `raw_text` | JSONB | Cached `[{page, text}]` from last parse run |

**`chunks`** — text segments extracted from PDFs, with embeddings

| Column | Type | Description |
|---|---|---|
| `id` | UUID PK | |
| `licitacion_id` | INTEGER FK | |
| `document_id` | UUID FK | |
| `doc_type` | TEXT | Copied from `documents.tipo` |
| `chunk_index` | INTEGER | Position within document |
| `page_number` | INTEGER | Source page |
| `text` | TEXT | Chunk content |
| `token_count` | INTEGER | cl100k token count |
| `chunk_config` | TEXT | e.g. `1024_256` (size_overlap) |
| `fts` | TSVECTOR | Auto-generated Spanish BM25 index |
| `emb_openai` | vector(1536) | text-embedding-3-large |
| `emb_cohere` | vector(1024) | embed-multilingual-v3.0 |

**`config`** — key/value pipeline state (`scan_watermark`, `last_status_refresh`)

---

## Data sources

Two independent data sources are collected and curated by the scripts in this repository:

1. **Portal HTML pages** — scraped from the Chihuahua State government procurement portal. The HTML parser (`ingestion/extract_licitacion.py`) emits structured JSON including dates, entity names, procedure codes, and a `documentos_json` list of `{tipo, url}` objects that drives selective PDF downloading.

2. **Official PDF documents** — tender Bases, Convocatorias, Actas de Junta de Aclaraciones, and technical annexes. Downloaded for every active (Vigente) record, parsed with LlamaParse (cloud OCR) with PyMuPDF as a fallback, and stored in Supabase Storage and the `chunks` table for retrieval.

---

## Implemented optional functionalities

1. **Streaming responses** — synthesis answers stream token-by-token via `synthesize_stream`, which calls the Claude streaming API and yields deltas directly to the Gradio `Chatbot` component.

2. **RAG evaluation** — the `eval/` folder contains the full evaluation suite: `eval_set.json` (189 retriever questions + 100 end-to-end pipeline questions), `eval_retriever.py` (benchmarks five retrieval strategies), `eval_pipeline.py` (end-to-end scoring with an LLM judge), and results files for multiple model and chunk-config combinations.

3. **Domain-specific app** — focused entirely on Mexican public procurement law (licitaciones públicas del estado de Chihuahua). Not a generic Q&A or AI tutor.

4. **Two data sources with structured JSON** — HTML scraping produces `documentos_json` (structured list of PDF links per tender). The one-brain router (`_route_turn` in `app/app.py`) emits structured JSON `{intent, search, route, anchor_index}` that controls retrieval strategy, anchor switching, and response mode — all from a single LLM call per turn.

5. **PDFs parsed for RAG** — the full pipeline downloads and parses government PDFs using LlamaParse with PyMuPDF as a fallback. Parsed text is stored per-document in `raw_text` (JSONB) and re-chunked without re-parsing. Chunks are the primary source for technical specifications, requirements, and clarification content.

6. **Reranker** — `retrieve_chunks` fetches up to 200 candidates from pgvector then re-ranks them with Cohere `rerank-multilingual-v3.0`, cutting to the top-K before synthesis. This is the best-performing retrieval configuration (see evaluation results).

7. **Metadata filtering** — `search_licitaciones` filters by `materia` (sector), keyword ILIKE on `descripcion` and `concepto_contratacion`, and date (only records whose `fecha_apertura` is in the future). Filters are extracted from the user message by the router LLM and always applied as parameterized SQL — no injection risk.

8. **Query routing** — `_route_turn` classifies every message into one of five intents (`discovery`, `anchor`, `detail`, `clarify_no_context`, `clarify_which`) and selects a route (`metadata` for a structured card, `rag` for PDF retrieval). This avoids unnecessary embedding or synthesis calls for simple metadata lookups.

9. **Automated daily pipeline** — a GitHub Actions workflow (`.github/workflows/daily_pipeline.yml`) runs the full ingestion pipeline every night: refresh status → cleanup Terminado records → fetch new → ingest → download → chunk → embed.

---

## Evaluation results

### Retriever — chunk config 1024 tokens / 256 overlap (189 questions)

| Strategy | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|
| BM25 only | 0.032 | 0.048 | 0.058 | 0.039 |
| Dense Cohere | 0.233 | 0.402 | 0.503 | 0.318 |
| **Dense Cohere + Rerank** | **0.439** | **0.656** | **0.741** | **0.528** |
| Hybrid Cohere + BM25 | 0.280 | 0.429 | 0.519 | 0.355 |
| Hybrid Cohere + Rerank | 0.418 | 0.603 | 0.672 | 0.501 |

Winner: **Dense Cohere + Rerank** — +1278% Hit@5 over BM25 alone.

### End-to-end pipeline — gpt-4o generator, gpt-4o judge (100 questions)

| Dimension | Score (0–10) |
|---|---|
| Relevance | 6.31 |
| Faithfulness | 4.73 |
| Completeness | 4.90 |
| **Average** | **5.31** |

Faithfulness and completeness are limited by garbled text in PyMuPDF-parsed chunks (broken Unicode for Spanish accents in certain government PDFs). Re-parsing with a higher-quality OCR engine is tracked as a pending improvement in `PENDINGS.md`.

---

## Required API keys

| Key | Provider | Used for | Required |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Anthropic (Claude) | Answer synthesis — `claude-sonnet-4-6` | **Yes** |
| `OPENAI_API_KEY` | OpenAI | Intent routing — `gpt-4o-mini` | **Yes** |
| `COHERE_API_KEY` | Cohere | Query embedding + reranking | **Yes** |
| `LLAMA_CLOUD_API_KEY` | LlamaCloud | PDF parsing (ingestion pipeline only) | Only for ingestion |

All keys are read from environment variables (`.env` file locally, Space Secrets on Hugging Face). **Never commit keys to the repository.**

---

## Cost estimation

A typical session with 10 questions costs approximately **$0.20–$0.25**:

| Component | Model | Tokens / call | Cost per query |
|---|---|---|---|
| Router | gpt-4o-mini | ~700 in + 150 out | ~$0.0002 |
| Embed query | Cohere multilingual-v3.0 | ~50 tokens | negligible |
| Rerank | Cohere rerank-multilingual-v3.0 | 200 docs | ~$0.0004 |
| Synthesis | claude-sonnet-4-6 | ~3,000 in + 500 out | ~$0.017 |
| **Total per query** | | | **~$0.018** |

10 queries ≈ **$0.18**. A full exploratory session covering 3–4 tenders in depth stays comfortably under **$0.30**.

---

## Running locally

```bash
# 1. Clone and create virtualenv
git clone <repo-url>
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 2. Install dependencies
pip install -r requirements-app.txt   # app only (Gradio + AI)
# pip install -r requirements.txt     # full pipeline (adds parsing, scraping libs)

# 3. Set environment variables
cp .env.example .env
# Edit .env and fill in DATABASE_URL, ANTHROPIC_API_KEY, OPENAI_API_KEY, COHERE_API_KEY

# 4. Set up the database schema
python infra/setup_db.py

# 5. Run the ingestion pipeline
python run_pipeline.py                        # full run
python run_pipeline.py --from chunk_docs      # resume from a step
python run_pipeline.py --only embed_index     # run one step only

# 6. Launch the app
python app/app.py
python app/app.py --share                     # public Gradio link
```

---

## Pipeline scripts

| Script | Purpose |
|---|---|
| `run_pipeline.py` | Orchestrator — runs all ingestion steps in sequence with `--from` / `--only` / `--dry-run` flags |
| `ingestion/fetch_vigentes.py` | Discovers new Vigente licitacion IDs from XLS export |
| `ingestion/ingest.py` | Scrapes HTML metadata for discovered records (async, 50 workers) |
| `ingestion/download_docs.py` | Downloads PDFs to Supabase Storage (async, 10 workers) |
| `ingestion/chunk_docs.py` | Parses PDFs → text chunks (LlamaParse + PyMuPDF fallback) |
| `ingestion/embed_index.py` | Computes and stores embeddings (Cohere / OpenAI) |
| `ingestion/refresh_status.py` | Re-checks Vigente records for status changes; queues new documents |
| `infra/setup_db.py` | Creates the schema; `--reset` drops and recreates (prompts confirmation) |
| `infra/cleanup_storage.py` | Deletes Supabase Storage files and chunks for Terminado licitaciones |
| `eval/generate_eval_set.py` | Generates evaluation questions with Claude Haiku |
| `eval/eval_retriever.py` | Benchmarks retrieval strategies (BM25, dense, hybrid, rerank) |
| `eval/eval_pipeline.py` | End-to-end answer quality evaluation with an LLM judge |
