"""
query.py — Dual-path procurement assistant for Chihuahua licitaciones.

Dual-path strategy:
  1. Structured DB search: metadata lookup for dates, costs, places, entities
  2. RAG: dense vector search + Cohere rerank of PDF chunks

Usage:
  python query.py "¿Qué licitaciones de fármacos hay vigentes?"
  python query.py "¿Cuándo es la apertura de propuestas para neumáticos?"
  python query.py --top-k 20 --no-rerank "¿Cuáles son los requisitos?"

Programmatic (e.g. from eval_pipeline.py):
  from query import run, get_conn
  conn = get_conn()
  result = run(conn, "¿Qué medicamentos solicita el gobierno?")
  print(result["answer"])
"""

import argparse
import json
import os
from datetime import date, datetime

import psycopg2
from dotenv import load_dotenv

load_dotenv()

DEFAULT_TOP_K = 40
DEFAULT_RERANK_CANDIDATES = 200
DEFAULT_N_LICITACIONES = 10
DEFAULT_CHUNK_CONFIG = "1024_256"


# ── Date helpers ──────────────────────────────────────────────────────────────

def _parse_date(s: str):
    if not s:
        return None
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _days_remaining(date_str: str):
    d = _parse_date(date_str)
    return (d - date.today()).days if d else None


def _urgency_label(days) -> str:
    if days is None:
        return ""
    if days <= 5:
        return "URGENTE"
    if days <= 14:
        return "PRÓXIMO"
    return "CON TIEMPO"


# ── DB connection ─────────────────────────────────────────────────────────────

def get_conn():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL not set in .env")
    at = url.rfind("@")
    creds, host_part = url[len("postgresql://"):at], url[at + 1:]
    user, password = creds.split(":", 1)
    host, port = host_part.split("/")[0].split(":")
    dbname = host_part.split("/")[1].split("?")[0]
    return psycopg2.connect(
        host=host, port=int(port), dbname=dbname,
        user=user, password=password, sslmode="require"
    )


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_query(question: str, model: str = "cohere") -> list[float]:
    if model == "cohere":
        import cohere
        client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
        resp = client.embed(
            texts=[question],
            model="embed-multilingual-v3.0",
            input_type="search_query",
            embedding_types=["float"],
        )
        return resp.embeddings.float_[0]
    elif model == "openai":
        from openai import OpenAI
        resp = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).embeddings.create(
            model="text-embedding-3-large",
            input=[question],
            dimensions=1536,
        )
        return resp.data[0].embedding
    else:
        raise ValueError(f"Unknown embedding model: {model}")


# ── Structured metadata search ────────────────────────────────────────────────

_METADATA_COLS = """
    id, url, numero_procedimiento, tipo_procedimiento,
    ente_contratante, ente_solicitante,
    concepto_contratacion, descripcion,
    fecha_convocatoria,
    fecha_junta_aclaraciones, hora_junta_aclaraciones, lugar_junta_aclaraciones,
    fecha_apertura, hora_apertura, lugar_apertura,
    costo_participacion, materia, tipo_contrato, modalidad
"""


def search_licitaciones(conn, keywords: list[str],
                        materia: str | None = None,
                        urgent_only: bool = False,
                        limit: int = DEFAULT_N_LICITACIONES) -> list[dict]:
    """
    Execute a parameterized search using pre-extracted filters from the LLM router.
    keywords and materia come from _route_turn() in app.py — no LLM call here.
    urgent_only=True restricts to licitaciones closing within 5 days.
    """
    cur = conn.cursor()

    date_filter = """
        AND fecha_apertura IS NOT NULL
        AND fecha_apertura != ''
        AND TO_DATE(fecha_apertura, 'DD/MM/YYYY') > CURRENT_DATE
    """
    if urgent_only:
        date_filter += "AND TO_DATE(fecha_apertura, 'DD/MM/YYYY') - CURRENT_DATE <= 5\n    "

    def _run_query(conditions: list[str], params: list, lim: int | None = limit) -> list[dict]:
        where = (" AND " + " AND ".join(conditions)) if conditions else ""
        limit_clause = "LIMIT %s" if lim is not None else ""
        limit_params = [lim]      if lim is not None else []
        cur.execute(f"""
            SELECT {_METADATA_COLS}
            FROM licitaciones
            WHERE licitacion_status = 'Vigente'
              {date_filter}
              {where}
            ORDER BY TO_DATE(fecha_apertura, 'DD/MM/YYYY') ASC
            {limit_clause}
        """, params + limit_params)
        rows = cur.fetchall()
        col_names = [d[0] for d in cur.description]
        return [dict(zip(col_names, r)) for r in rows]

    conditions: list[str] = []
    params: list = []

    for kw in keywords:
        conditions.append(
            "(descripcion ILIKE %s OR concepto_contratacion ILIKE %s"
            " OR numero_procedimiento ILIKE %s OR ente_contratante ILIKE %s)"
        )
        params += [f"%{kw}%"] * 4

    if materia:
        conditions.append("materia ILIKE %s")
        params.append(f"%{materia}%")

    had_filters = bool(conditions)
    results = _run_query(conditions, params)

    # Materia too restrictive — retry with keywords only
    if not results and keywords and materia:
        kw_conds = [
            "(descripcion ILIKE %s OR concepto_contratacion ILIKE %s"
            " OR numero_procedimiento ILIKE %s OR ente_contratante ILIKE %s)"
        ] * len(keywords)
        kw_params: list = []
        for kw in keywords:
            kw_params += [f"%{kw}%"] * 4
        results = _run_query(kw_conds, kw_params)

    # No topic match (or no filters) → show all upcoming vigentes without limit.
    # Keywords like "vigente"/"licitacion" never appear in descriptions, so a zero-match
    # on keywords with no materia should still return the full list, not an empty result.
    if not results and not materia:
        results = _run_query([], [], lim=None)

    cur.close()
    return results


# ── Direct licitacion lookup ──────────────────────────────────────────────────

def get_licitacion_by_numero(conn, numero: str):
    """Find a Vigente licitacion by partial numero_procedimiento match."""
    cur = conn.cursor()
    cur.execute(
        f"SELECT {_METADATA_COLS} FROM licitaciones "
        "WHERE licitacion_status = 'Vigente' AND numero_procedimiento ILIKE %s LIMIT 1",
        (f"%{numero}%",),
    )
    row = cur.fetchone()
    col_names = [d[0] for d in cur.description]
    cur.close()
    return dict(zip(col_names, row)) if row else None


def get_licitacion_by_id(conn, licitacion_id: int):
    cur = conn.cursor()
    cur.execute(f"SELECT {_METADATA_COLS} FROM licitaciones WHERE id = %s", (licitacion_id,))
    row = cur.fetchone()
    col_names = [d[0] for d in cur.description]
    cur.close()
    return dict(zip(col_names, row)) if row else None


# ── Dense + rerank retrieval ──────────────────────────────────────────────────

def retrieve_chunks(conn, question: str, model: str = "cohere",
                    top_k: int = DEFAULT_TOP_K, rerank: bool = True,
                    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
                    chunk_config: str = DEFAULT_CHUNK_CONFIG,
                    licitacion_ids: list | None = None,
                    doc_types: list | None = None) -> list[dict]:
    """
    Dense vector retrieval with optional Cohere reranking.
    Pass licitacion_ids to scope retrieval to a specific licitacion (anchored mode).
    Pass doc_types to further scope to specific document types (user-selected filter).
    Returns chunk dicts with text + licitacion metadata fields for context building.
    """
    col = "emb_cohere" if model == "cohere" else "emb_openai"
    fetch_k = rerank_candidates if rerank else top_k
    query_vec = embed_query(question, model)

    lic_filter = ""
    lic_params = []
    if licitacion_ids:
        placeholders = ",".join(["%s"] * len(licitacion_ids))
        lic_filter = f"AND c.licitacion_id IN ({placeholders})"
        lic_params = list(licitacion_ids)

    doc_filter = ""
    doc_params = []
    if doc_types:
        placeholders = ",".join(["%s"] * len(doc_types))
        doc_filter = f"AND c.doc_type IN ({placeholders})"
        doc_params = list(doc_types)

    cur = conn.cursor()
    cur.execute(f"""
        SELECT
            c.id::text,
            c.text,
            c.doc_type,
            c.licitacion_id,
            l.descripcion,
            l.ente_contratante,
            l.url
        FROM chunks c
        JOIN documents d    ON c.document_id   = d.id
        JOIN licitaciones l ON c.licitacion_id = l.id
        WHERE l.licitacion_status = 'Vigente'
          AND c.chunk_config = %s
          AND c.{col} IS NOT NULL
          {lic_filter}
          {doc_filter}
        ORDER BY c.{col} <=> %s::vector
        LIMIT %s
    """, [chunk_config] + lic_params + doc_params + [json.dumps(query_vec), fetch_k])

    rows = cur.fetchall()
    col_names = [d[0] for d in cur.description]
    cur.close()

    chunks = [dict(zip(col_names, row)) for row in rows]
    if not chunks:
        return []

    # Drop blank Anexo form templates: chunks with a date placeholder ("___ de")
    # and an ANEXO heading are blank signature/form pages with zero retrieval value.
    # ANEXO TÉCNICO and ANEXO ECONÓMICO with real content don't have these placeholders.
    chunks = [
        c for c in chunks
        if not ("___ de" in (c.get("text") or "") and
                "NEXO" in (c.get("text") or ""))
    ]

    # Deduplicate by normalized first 300 chars — repeated page headers with slight
    # whitespace differences (\n vs \n\n) produce near-identical chunks
    seen: set[str] = set()
    deduped = []
    for c in chunks:
        raw = (c.get("text") or "")[:300]
        key = " ".join(raw.split())  # collapse all whitespace before comparing
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    chunks = deduped

    if rerank:
        import cohere
        client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
        resp = client.rerank(
            query=question,
            documents=[c["text"] for c in chunks],
            model="rerank-multilingual-v3.0",
            top_n=min(top_k, len(chunks)),
        )
        reranked = []
        for r in resp.results:
            c = dict(chunks[r.index])
            c["score"] = round(r.relevance_score, 3)
            reranked.append(c)
        chunks = reranked

    return chunks[:top_k]


# ── Context builder ───────────────────────────────────────────────────────────

def _field(label: str, value) -> str:
    v = str(value).strip() if value else ""
    return f"  {label}: {v}" if v else ""


def build_context(licitaciones: list[dict], chunks: list[dict]) -> str:
    parts: list[str] = []

    if licitaciones:
        parts.append("=== DATOS OFICIALES (fuente: base de datos — usar SOLO estos para fechas, lugares y costos) ===\n")
        for i, l in enumerate(licitaciones, 1):
            ap_str   = l.get("fecha_apertura") or ""
            ap_hora  = l.get("hora_apertura") or ""
            ap_days  = _days_remaining(ap_str)
            ap_label = f"{ap_str} {ap_hora}".strip()
            if ap_days is not None:
                ap_label += f" ({ap_days} días — {_urgency_label(ap_days)})"

            jun_str  = l.get("fecha_junta_aclaraciones") or ""
            jun_hora = l.get("hora_junta_aclaraciones") or ""
            jun_days = _days_remaining(jun_str)
            jun_label = f"{jun_str} {jun_hora}".strip()
            if jun_days is not None:
                if jun_days >= 0:
                    jun_label += f" ({jun_days} días — asiste para resolver dudas)"
                else:
                    jun_label += " (ya ocurrió — aún puedes presentar propuesta)"

            block = "\n".join(filter(None, [
                f"[{i}] {l.get('descripcion') or l.get('concepto_contratacion', 'Sin descripción')}",
                _field("ID", l.get("id")),
                _field("Número procedimiento", l.get("numero_procedimiento")),
                _field("Tipo", l.get("tipo_procedimiento")),
                _field("Materia", l.get("materia")),
                _field("Ente contratante", l.get("ente_contratante")),
                _field("Ente solicitante", l.get("ente_solicitante")),
                _field("Fecha convocatoria", l.get("fecha_convocatoria")),
                _field("Junta de aclaraciones", jun_label or None),
                _field("Lugar junta", l.get("lugar_junta_aclaraciones")),
                _field("Deadline propuesta", ap_label or None),
                _field("Lugar apertura", l.get("lugar_apertura")),
                _field("Costo participación", l.get("costo_participacion")),
                _field("URL", l.get("url")),
            ]))
            parts.append(block)
            parts.append("")

    if chunks:
        parts.append("=== CONTENIDO DE DOCUMENTOS OFICIALES (fuente: PDFs — usar para requisitos y especificaciones técnicas) ===\n")
        char_budget = 60_000  # ~15k tokens — leaves room for metadata + system prompt
        used = sum(len(p) for p in parts)
        for i, c in enumerate(chunks, 1):
            text = c.get("text", "")
            desc = (c.get("descripcion") or "")[:60]
            label = f"{c.get('doc_type', 'Documento')} — {desc}"
            fragment = f"[Fragmento {i} | {label}]\n{text}\n"
            if used + len(fragment) > char_budget:
                break
            parts.append(f"[Fragmento {i} | {label}]")
            parts.append(text)
            parts.append("")
            used += len(fragment)

    return "\n".join(parts)


# ── Claude synthesis ──────────────────────────────────────────────────────────

_SYSTEM_ES = """\
Eres un asistente especializado en licitaciones públicas del estado de Chihuahua, México.
Ayudas a representantes de ventas a identificar oportunidades de negocio y preparar su participación.

Se te proporciona:
1. DATOS OFICIALES de la licitación (fechas, lugares, costos) — fuente: base de datos estructurada
2. CONTENIDO DE DOCUMENTOS (requisitos, especificaciones técnicas) — fuente: fragmentos de PDFs oficiales

REGLAS CRÍTICAS:
- Para fechas, plazos, horarios, lugares y costos usa ÚNICAMENTE la sección "DATOS OFICIALES".
  Nunca deduzcas ni inventes estos datos a partir de los fragmentos de documentos.
- Para requisitos técnicos y documentación requerida usa la sección "CONTENIDO DE DOCUMENTOS".
- Si la información solicitada no está en el contexto, responde exactamente:
  "No encontré esa información en los datos disponibles para esta licitación."
  NUNCA uses conocimiento propio para rellenar datos faltantes.
- Si la pregunta es ambigua o necesitas saber de qué licitación específica se trata,
  haz UNA sola pregunta de aclaración en lugar de adivinar.

Responde en español de forma clara y directa.\
"""

_SYSTEM_EN = """\
You are a specialized assistant for public procurement tenders in the state of Chihuahua, Mexico.
You help sales representatives identify business opportunities and prepare their participation.

You are provided with:
1. OFFICIAL DATA from the tender (dates, locations, costs) — source: structured database
2. DOCUMENT CONTENT (requirements, technical specifications) — source: official PDF excerpts

CRITICAL RULES:
- For dates, deadlines, schedules, locations, and costs use ONLY the "OFFICIAL DATA" section.
  Never infer or fabricate this data from document fragments.
- For technical requirements and required documentation use the "DOCUMENT CONTENT" section.
- If the requested information is not in the context, respond exactly:
  "I could not find that information in the available data for this tender."
  NEVER use your own knowledge to fill in missing data.
- If the question is ambiguous or you need to know which specific tender is being referred to,
  ask ONE clarifying question instead of guessing.

Always respond in English, clearly and directly.\
"""


def _get_system(lang: str = "es") -> str:
    return _SYSTEM_EN if lang == "en" else _SYSTEM_ES


def synthesize_stream(question: str, context: str, model: str = "gpt-4o", lang: str = "es", api_key: str = ""):
    """Generator version of synthesize — yields text deltas for streaming."""
    import time
    system   = _get_system(lang)
    user_msg = f"Contexto:\n{context}\n\nPregunta: {question}"

    if not model.startswith("claude"):
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ]
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=model, max_tokens=2048,
                    messages=messages, stream=True,
                )
                for chunk in resp:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
                return
            except Exception as e:
                if attempt < 3 and ("429" in str(e) or "rate" in str(e).lower()):
                    time.sleep(15 * (2 ** attempt))
                else:
                    raise
    else:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        for attempt in range(4):
            try:
                thinking = {"type": "adaptive"} if "opus" in model else None
                extra = {"thinking": thinking} if thinking else {}
                with client.messages.stream(
                    model=model, max_tokens=2048,
                    system=system,
                    messages=[{"role": "user", "content": user_msg}],
                    **extra,
                ) as s:
                    for text in s.text_stream:
                        yield text
                return
            except Exception as e:
                if attempt < 3 and ("529" in str(e) or "overloaded" in str(e).lower()):
                    time.sleep(15 * (2 ** attempt))
                else:
                    raise


def synthesize(question: str, context: str, stream: bool = True,
               model: str = "claude-sonnet-4-6") -> str:
    import time
    user_msg = f"Contexto:\n{context}\n\nPregunta: {question}"

    # ── OpenAI path ───────────────────────────────────────────────────────────
    if not model.startswith("claude"):
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": user_msg},
        ]
        for attempt in range(4):
            try:
                if stream:
                    full_text = ""
                    resp = client.chat.completions.create(
                        model=model, max_tokens=2048,
                        messages=messages, stream=True,
                    )
                    for chunk in resp:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            print(delta, end="", flush=True)
                            full_text += delta
                    print()
                    return full_text
                else:
                    resp = client.chat.completions.create(
                        model=model, max_tokens=2048, messages=messages,
                    )
                    return resp.choices[0].message.content or ""
            except Exception as e:
                if attempt < 3 and ("429" in str(e) or "rate" in str(e).lower()):
                    wait = 15 * (2 ** attempt)
                    print(f"[synthesize] Rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        return ""

    # ── Anthropic path ────────────────────────────────────────────────────────
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    for attempt in range(4):
        try:
            thinking = {"type": "adaptive"} if "opus" in model else None
            extra = {"thinking": thinking} if thinking else {}

            if stream:
                full_text = ""
                with client.messages.stream(
                    model=model,
                    max_tokens=2048,
                    system=_SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                    **extra,
                ) as s:
                    for text in s.text_stream:
                        print(text, end="", flush=True)
                        full_text += text
                print()
                return full_text
            else:
                resp = client.messages.create(
                    model=model,
                    max_tokens=2048,
                    system=_SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                    **extra,
                )
                for block in reversed(resp.content):
                    if block.type == "text":
                        return block.text
                return ""
        except Exception as e:
            if attempt < 3 and ("529" in str(e) or "overloaded" in str(e).lower()):
                wait = 15 * (2 ** attempt)
                print(f"[synthesize] API overloaded, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    return ""


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run(conn, question: str, model: str = "cohere",
        top_k: int = DEFAULT_TOP_K, rerank: bool = True,
        rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
        n_licitaciones: int = DEFAULT_N_LICITACIONES,
        synthesis_model: str = "claude-opus-4-8",
        chunk_config: str = DEFAULT_CHUNK_CONFIG,
        stream: bool = False,
        licitacion_ids: list = None) -> dict:
    """
    Full dual-path query pipeline.
    Pass licitacion_ids to skip FTS and anchor retrieval to specific licitaciones.
    Returns dict with answer, licitaciones, chunks, and context
    so eval_pipeline.py can inspect intermediate results.
    """
    if licitacion_ids:
        licitaciones = [l for lid in licitacion_ids
                        if (l := get_licitacion_by_id(conn, lid))]
    else:
        licitaciones = search_licitaciones(conn, [], limit=n_licitaciones)

    chunks = retrieve_chunks(
        conn, question,
        model=model, top_k=top_k,
        rerank=rerank, rerank_candidates=rerank_candidates,
        chunk_config=chunk_config,
        licitacion_ids=licitacion_ids,
    )
    context = build_context(licitaciones, chunks)
    answer  = synthesize(question, context, stream=stream, model=synthesis_model)

    return {
        "question":    question,
        "answer":      answer,
        "licitaciones": licitaciones,
        "chunks":       chunks,
        "context":      context,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ask a natural-language question about Chihuahua licitaciones"
    )
    parser.add_argument("question", nargs="?",
                        help="Question in Spanish (prompted if omitted)")
    parser.add_argument("--model", choices=["cohere", "openai"], default="cohere")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"RAG chunks to retrieve (default: {DEFAULT_TOP_K})")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Disable Cohere reranking")
    parser.add_argument("--rerank-candidates", type=int, default=DEFAULT_RERANK_CANDIDATES)
    parser.add_argument("--n-licitaciones", type=int, default=DEFAULT_N_LICITACIONES,
                        help="Structured DB results to include (default: 5)")
    args = parser.parse_args()

    question = args.question or input("Pregunta: ").strip()
    if not question:
        print("No question provided.")
        return

    print(f"\n[Pregunta] {question}\n")
    print("[Buscando en base de datos y documentos...]\n")

    conn = get_conn()
    result = run(
        conn, question,
        model=args.model,
        top_k=args.top_k,
        rerank=not args.no_rerank,
        rerank_candidates=args.rerank_candidates,
        n_licitaciones=args.n_licitaciones,
        stream=True,
    )
    conn.close()

    print(f"\n--- Fuentes usadas ---")
    print(f"  Licitaciones DB : {len(result['licitaciones'])}")
    print(f"  Fragmentos RAG  : {len(result['chunks'])}")


if __name__ == "__main__":
    main()
