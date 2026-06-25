# Legal Associate — Asistente de Licitaciones Públicas de Chihuahua

A RAG-powered legal assistant for Mexican public procurement. It monitors all active tenders from the State of Chihuahua, lets users search and explore them in natural language, and answers questions about requirements and technical specifications by reading the official PDF documents.

---

## What it does

1. **Discovery** — asks "what road-paving tenders are open?" and gets a ranked list with deadlines and urgency indicators.
2. **Anchor** — selects a specific tender from the list; subsequent questions are scoped to that record.
3. **Detail retrieval** — asks "what are the technical requirements?" and the assistant retrieves relevant chunks from the official PDFs and synthesizes an answer.
4. **Conversation memory** — every session is persisted in PostgreSQL; the user can resume past conversations from the sidebar.

---

## Architecture

```
fetch_vigentes.py   → discover licitacion IDs (async HEAD checks)
ingest.py             → scrape metadata from HTML pages (async)
download_docs.py      → download PDFs (Vigente only)
chunk_docs.py         → parse PDFs → text chunks (LlamaParse + PyMuPDF fallback)
embed_index.py        → embed chunks (Cohere multilingual-v3.0 / OpenAI)
refresh_status.py     → nightly re-check of Vigente records; detect status changes
app.py                → Gradio chat UI + one-brain router + streaming synthesis
query.py              → dense retrieval + Cohere reranking + metadata search
```

Database: PostgreSQL (Supabase) with pgvector for HNSW index, tsvector for BM25, and a `chunks` table that stores one embedding column per model so all models share the same chunked text.

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
| `pipeline_status` | TEXT | Ingestion stage: `scraped → downloaded → chunked → indexed` |
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

- **Structured metadata**: HTML pages scraped from the Chihuahua State government procurement portal — licitacion IDs, dates, entities, costs, document lists.
- **PDFs**: Official tender documents (bases, convocatorias, technical annexes) downloaded for every Vigente record and parsed into searchable chunks.

Both sources are collected and curated with the scripts in this repository. The HTML parser (`extract_licitacion.py`) emits structured JSON (dates, documentos_json, estatus) that drives downstream pipeline decisions (which docs to download, when to re-queue, etc.).

---

## Implemented features

1. **Streaming responses** — synthesis answers stream token-by-token via `synthesize_stream`, which calls the OpenAI or Claude streaming API and yields deltas directly to the Gradio `Chatbot`.

2. **RAG evaluation** — `eval_set.json` contains 189 retriever questions and 100 end-to-end pipeline questions. `eval_retriever.py` benchmarks five retrieval strategies. `eval_pipeline.py` runs end-to-end scoring with an LLM judge. Results are in `eval_results_1024_256.md` and `eval_pipeline_summary_gpt_4o__judge_gpt_4o.md`.

3. **Domain-specific app** — focused entirely on Mexican public procurement law (licitaciones públicas).

4. **PDFs parsed** — the full pipeline downloads and parses official government PDFs using LlamaParse with PyMuPDF as a fallback. Parsed text is stored per-document and re-chunked without re-parsing.

5. **Structured JSON for advanced RAG** — the HTML parser emits `documentos_json` (list of `{tipo, url}` objects) that drives selective downloading. The one-brain router (`_route_turn`) returns structured JSON `{intent, search, route, anchor_index}` that determines retrieval strategy, anchor switching, and response mode — all from a single LLM call per turn.

6. **Reranker** — `retrieve_chunks` fetches up to 200 candidates from pgvector then re-ranks them with Cohere `rerank-multilingual-v3.0`, cutting to the top-K before synthesis. This is the best-performing retrieval configuration (see evaluation results).

7. **Metadata filtering** — `search_licitaciones` filters by `materia`, keyword ILIKE on `descripcion`/`concepto_contratacion`, and date (only records whose `fecha_apertura` is in the future). The router extracts these filters from the user message with one LLM call so the SQL query is always parameterized.

8. **Query routing** — `_route_turn` classifies every message into one of five intents (`discovery`, `anchor`, `detail`, `clarify_no_context`, `clarify_which`) and selects a route (`metadata` for structured card, `rag` for PDF retrieval). This avoids unnecessary embedding or synthesis calls for simple metadata lookups.

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

Faithfulness and completeness are limited by garbled text in PyMuPDF-parsed chunks (broken Unicode for Spanish accents in certain government PDFs). Re-parsing with a higher-quality OCR engine is tracked as a pending improvement.

---

## Required API keys

| Key | Used for |
|---|---|
| `OPENAI_API_KEY` | Intent routing (gpt-4o-mini) + answer synthesis (gpt-4o) |
| `COHERE_API_KEY` | Embedding queries (embed-multilingual-v3.0) + reranking (rerank-multilingual-v3.0) |
| `ANTHROPIC_API_KEY` | Optional — synthesis with Claude models (claude-sonnet-4-6, claude-opus-4-8) |

All keys are read from environment variables (`.env` file locally, Secrets on HuggingFace Spaces). **Never commit your keys to the repository.**

---

## Cost estimation

A typical session with 10 questions costs approximately **$0.30–$0.40**:

| Component | Model | Tokens (per query) | Cost per query |
|---|---|---|---|
| Router | gpt-4o-mini | ~700 in + 150 out | ~$0.0001 |
| Embed query | Cohere multilingual-v3.0 | ~50 tokens | ~$0.000005 |
| Rerank | Cohere rerank-multilingual-v3.0 | 200 docs × ~300 tokens | ~$0.0002 |
| Synthesis | gpt-4o | ~3 000 in + 1 500 out | ~$0.023 |
| **Total per query** | | | **~$0.024** |

10 queries ≈ **$0.24**. A full exploratory session exploring 3–4 licitaciones in depth stays comfortably under **$0.50**.

---

## Running locally

```bash
# 1. Clone and create virtualenv
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env
# Edit .env and fill in DATABASE_URL, OPENAI_API_KEY, COHERE_API_KEY

# 4. Set up the database schema
python setup_db.py

# 5. Run the ingestion pipeline (see CLAUDE.md for full pipeline)
python fetch_vigentes.py --start 200001 --end 271720
python ingest.py
python download_docs.py
python chunk_docs.py --chunk-size 1024 --overlap 256
python embed_index.py --model cohere

# 6. Launch the app
python app.py
```

---

## Pipeline scripts

| Script | Purpose |
|---|---|
| `fetch_vigentes.py` | Async discovery of valid licitacion IDs via HEAD checks |
| `ingest.py` | Scrapes HTML metadata for discovered records |
| `download_docs.py` | Downloads PDFs for Vigente licitaciones |
| `chunk_docs.py` | Parses and chunks documents (LlamaParse + PyMuPDF) |
| `embed_index.py` | Computes and stores embeddings (Cohere / OpenAI) |
| `refresh_status.py` | Re-checks Vigente records for status changes; queues new docs |
| `generate_eval_set.py` | Generates evaluation questions with Claude Haiku |
| `eval_retriever.py` | Benchmarks retrieval strategies (BM25, dense, hybrid, rerank) |
| `eval_pipeline.py` | End-to-end answer quality evaluation with LLM judge |
