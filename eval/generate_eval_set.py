"""
Generates a synthetic evaluation set for RAG retriever testing.

Four question types reflecting real vendor/sales-rep queries:

  specific  — generated from a single chunk; ground truth = chunk_id
              e.g. "¿Qué garantía pide el gobierno para este contrato?"

  topic     — generated from a batch of licitacion titles; ground truth = licitacion_ids
              e.g. "¿Cuáles son las licitaciones de fármacos disponibles?"

  metadata  — generated from structured fields (dates, costs, places, entities);
              ground truth = licitacion_id
              e.g. "¿Cuándo es la apertura de propuestas para neumáticos?"

  summary   — generated from a full document; ground truth = chunk_ids of that document
              e.g. "¿Cuáles son los requisitos para participar en la licitación de X?"

Output: eval_set.json

Usage:
  python generate_eval_set.py
  python generate_eval_set.py --samples 75 --broad-samples 40 --metadata-samples 30 --summary-samples 20
"""

import argparse
import json
import os
import time

import anthropic
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

# ── Prompts ───────────────────────────────────────────────────────────────────

SPECIFIC_PROMPT = """Eres un representante de ventas que evalúa si su empresa puede participar
en licitaciones del gobierno de Chihuahua, México.

Se te muestra un fragmento de un documento de licitación (tipo: {doc_type}).
Genera {n} preguntas en español que haría un proveedor al leer este fragmento.
Las preguntas deben ser:
- Respondibles ÚNICAMENTE con la información de este fragmento
- Concretas y orientadas a decidir si participar: requisitos, especificaciones, condiciones
- Del tipo: "¿Se requiere X?", "¿Cuál es el plazo de entrega?", "¿Qué garantía piden?"

Fragmento:
{text}

Devuelve ÚNICAMENTE un array JSON con las preguntas, sin explicación.
Ejemplo: ["¿Qué fianza se requiere para participar?", "¿Cuál es el plazo de entrega del material?"]"""


BROAD_PROMPT = """Eres un representante de ventas con cartera de clientes en distintos sectores
(salud, construcción, tecnología, transporte, etc.) en el estado de Chihuahua.

Se te presentan títulos y descripciones de licitaciones vigentes del gobierno:
{titles}

Genera {n} preguntas o búsquedas en español que escribirías para identificar oportunidades
para tus clientes. Las preguntas deben variar en tipo:
- Descubrimiento de categoría: "¿Qué licitaciones de medicamentos hay?"
- Por ente comprador: "¿Qué está comprando la Secretaría de Salud?"
- Por tipo de servicio: "¿Hay contratos de mantenimiento de vehículos?"
- Urgencia: "¿Qué licitaciones cierran pronto para construcción?"

Devuelve ÚNICAMENTE un array JSON con las preguntas, sin explicación.
Ejemplo: ["¿Cuáles son las licitaciones vigentes de material médico?", "¿Qué contratos de pavimentación están abiertos?"]"""


METADATA_PROMPT = """Eres un representante de ventas evaluando si tu empresa puede participar
en una licitación del gobierno de Chihuahua.

Tienes esta información estructurada de la licitación:
- Descripción: {descripcion}
- Concepto: {concepto}
- Comprador: {ente_contratante} para {ente_solicitante}
- Junta de aclaraciones: {fecha_junta} a las {hora_junta} en {lugar_junta}
- Apertura de propuestas: {fecha_apertura} a las {hora_apertura} en {lugar_apertura}
- Costo de participación: {costo}

Genera {n} preguntas en español que haría un vendedor sobre esta licitación.
Las preguntas deben cubrir diferentes aspectos:
- Fechas y horarios (junta, apertura, cierre)
- Dónde se entregan los documentos
- Cuánto cuesta inscribirse o participar
- Quién está comprando y para quién
- Qué tipo de producto o servicio se solicita

Devuelve ÚNICAMENTE un array JSON con las preguntas, sin explicación.
Ejemplo: ["¿Cuándo es la junta de aclaraciones para neumáticos?", "¿Cuánto cuesta participar en esta licitación?", "¿Dónde entrego mi propuesta?"]"""


SUMMARY_PROMPT = """Eres un representante de ventas que acaba de descargar un documento
de licitación del gobierno de Chihuahua y quiere entender rápidamente si es relevante.

Tipo de documento: {doc_type}
Licitación: {descripcion}
Concepto: {concepto}
Comprador: {ente_contratante}

Genera {n} preguntas en español que haría el vendedor para entender el contenido del documento.
Las preguntas deben ser sobre:
- Qué requisitos o documentos pide el gobierno para participar
- Especificaciones técnicas del producto o servicio
- Condiciones de entrega, garantía, o penalidades
- Si necesita estar registrado en algún padrón
- Resumen general de las bases

Devuelve ÚNICAMENTE un array JSON con las preguntas, sin explicación.
Ejemplo: ["¿Qué documentos necesito presentar para participar?", "¿Cuáles son las especificaciones técnicas requeridas?", "¿Hay penalidades por incumplimiento?"]"""


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


# ── Claude call ───────────────────────────────────────────────────────────────

def ask_claude(client, prompt: str) -> list[str]:
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = resp.content[0].text.strip()
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        return [q for q in json.loads(raw[start:end]) if q.strip()]
    except Exception as e:
        print(f"    [WARN] {e}")
        return []


# ── Question generators ────────────────────────────────────────────────────────

def generate_specific(conn, client, samples: int, questions_per_chunk: int,
                      min_tokens: int, chunk_config: str = "1024_256") -> list[dict]:
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT c.id::text, c.licitacion_id, c.doc_type, c.text, c.token_count,
               l.numero_procedimiento, l.ente_contratante,
               l.descripcion, l.concepto_contratacion
        FROM chunks c
        JOIN documents d    ON c.document_id   = d.id
        JOIN licitaciones l ON c.licitacion_id = l.id
        WHERE d.status IN ('chunked', 'indexed')
          AND d.parser = 'llamaparse'
          AND l.licitacion_status = 'Vigente'
          AND c.chunk_config = %s
          AND c.token_count >= %s
        ORDER BY RANDOM()
        LIMIT %s
    """, (chunk_config, min_tokens, samples))
    chunks = cur.fetchall()
    cur.close()

    if not chunks:
        print("  [specific] No chunks found.")
        return []

    print(f"\n[Specific] {len(chunks)} chunks → "
          f"~{len(chunks) * questions_per_chunk} questions ...")

    results = []
    for i, chunk in enumerate(chunks):
        questions = ask_claude(client, SPECIFIC_PROMPT.format(
            n=questions_per_chunk,
            doc_type=chunk["doc_type"] or "Documento",
            text=chunk["text"][:2000],
        ))
        for q in questions:
            results.append({
                "type":                 "specific",
                "question":             q,
                "chunk_id":             chunk["id"],
                "licitacion_id":        chunk["licitacion_id"],
                "doc_type":             chunk["doc_type"],
                "numero_procedimiento": chunk["numero_procedimiento"],
                "ente_contratante":     chunk["ente_contratante"] or "",
                "text_snippet":         chunk["text"][:300],
            })
        print(f"  [{i+1}/{len(chunks)}] {(chunk['doc_type'] or ''):<30} → {len(questions)} questions")
        time.sleep(0.2)

    return results


def generate_broad(conn, client, broad_samples: int,
                   batch_size: int = 8, questions_per_batch: int = 3) -> list[dict]:
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT id, descripcion, concepto_contratacion,
               ente_contratante, ente_solicitante
        FROM licitaciones
        WHERE licitacion_status = 'Vigente'
          AND (descripcion IS NOT NULL OR concepto_contratacion IS NOT NULL)
          AND id IN (
              SELECT DISTINCT licitacion_id FROM documents WHERE parser = 'llamaparse'
          )
        ORDER BY RANDOM()
        LIMIT %s
    """, (broad_samples,))
    licitaciones = cur.fetchall()
    cur.close()

    if not licitaciones:
        print("  [topic] No licitaciones found.")
        return []

    batches = [licitaciones[i:i + batch_size]
               for i in range(0, len(licitaciones), batch_size)]

    print(f"\n[Topic] {len(licitaciones)} licitaciones → "
          f"{len(batches)} batches → ~{len(batches) * questions_per_batch} questions ...")

    results = []
    for i, batch in enumerate(batches):
        titles = "\n".join(
            f"- {lic['descripcion'] or lic['concepto_contratacion'] or '(sin título)'}"
            f"  [{lic['ente_contratante'] or ''}]"
            for lic in batch
        )
        licitacion_ids = [lic["id"] for lic in batch]

        questions = ask_claude(client, BROAD_PROMPT.format(
            n=questions_per_batch,
            titles=titles,
        ))
        for q in questions:
            results.append({
                "type":           "topic",
                "question":       q,
                "licitacion_ids": licitacion_ids,
                "titles_snippet": titles[:400],
            })
        print(f"  [batch {i+1}/{len(batches)}] {len(batch)} licitaciones → {len(questions)} questions")
        time.sleep(0.2)

    return results


def generate_metadata(conn, client, metadata_samples: int,
                      questions_per_lic: int = 3) -> list[dict]:
    """Generate questions from structured metadata fields (dates, places, costs)."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT id, descripcion, concepto_contratacion,
               ente_contratante, ente_solicitante,
               fecha_junta_aclaraciones, hora_junta_aclaraciones, lugar_junta_aclaraciones,
               fecha_apertura, hora_apertura, lugar_apertura,
               costo_participacion
        FROM licitaciones
        WHERE licitacion_status = 'Vigente'
          AND (fecha_apertura IS NOT NULL AND fecha_apertura != ''
               OR costo_participacion IS NOT NULL AND costo_participacion != ''
               OR lugar_apertura IS NOT NULL AND lugar_apertura != '')
          AND id IN (
              SELECT DISTINCT licitacion_id FROM documents WHERE parser = 'llamaparse'
          )
        ORDER BY RANDOM()
        LIMIT %s
    """, (metadata_samples,))
    licitaciones = cur.fetchall()
    cur.close()

    if not licitaciones:
        print("  [metadata] No licitaciones with structured fields found.")
        return []

    print(f"\n[Metadata] {len(licitaciones)} licitaciones → "
          f"~{len(licitaciones) * questions_per_lic} questions ...")

    results = []
    for i, lic in enumerate(licitaciones):
        questions = ask_claude(client, METADATA_PROMPT.format(
            n=questions_per_lic,
            descripcion=lic["descripcion"] or lic["concepto_contratacion"] or "N/A",
            concepto=lic["concepto_contratacion"] or "N/A",
            ente_contratante=lic["ente_contratante"] or "N/A",
            ente_solicitante=lic["ente_solicitante"] or "N/A",
            fecha_junta=lic["fecha_junta_aclaraciones"] or "N/A",
            hora_junta=lic["hora_junta_aclaraciones"] or "N/A",
            lugar_junta=lic["lugar_junta_aclaraciones"] or "N/A",
            fecha_apertura=lic["fecha_apertura"] or "N/A",
            hora_apertura=lic["hora_apertura"] or "N/A",
            lugar_apertura=lic["lugar_apertura"] or "N/A",
            costo=lic["costo_participacion"] or "N/A",
        ))
        for q in questions:
            results.append({
                "type":              "metadata",
                "question":          q,
                "licitacion_id":     lic["id"],
                "descripcion":       lic["descripcion"] or lic["concepto_contratacion"],
                "ente_contratante":  lic["ente_contratante"],
                "fecha_apertura":    lic["fecha_apertura"],
                "costo":             lic["costo_participacion"],
            })
        desc = (lic["descripcion"] or lic["concepto_contratacion"] or "")[:40]
        print(f"  [{i+1}/{len(licitaciones)}] {desc:<40} → {len(questions)} questions")
        time.sleep(0.2)

    return results


def generate_summary(conn, client, summary_samples: int,
                     questions_per_doc: int = 2,
                     chunk_config: str = "1024_256") -> list[dict]:
    """Generate questions that require reading and summarizing a full document."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT d.id::text AS doc_id, d.tipo AS doc_type, d.licitacion_id,
               l.descripcion, l.concepto_contratacion, l.ente_contratante,
               COUNT(c.id) AS chunk_count,
               ARRAY_AGG(c.id::text ORDER BY c.chunk_index) AS chunk_ids
        FROM documents d
        JOIN licitaciones l ON d.licitacion_id = l.id
        JOIN chunks c       ON c.document_id   = d.id
        WHERE d.status IN ('chunked', 'indexed')
          AND d.parser = 'llamaparse'
          AND l.licitacion_status = 'Vigente'
          AND c.chunk_config = %s
          AND d.tipo IN ('Convocatoria', 'Bases', 'Bases - Anexo 1 - Bases editables')
        GROUP BY d.id, d.tipo, d.licitacion_id,
                 l.descripcion, l.concepto_contratacion, l.ente_contratante
        HAVING COUNT(c.id) >= 3
        ORDER BY RANDOM()
        LIMIT %s
    """, (chunk_config, summary_samples,))
    docs = cur.fetchall()
    cur.close()

    if not docs:
        print("  [summary] No documents with enough chunks found.")
        return []

    print(f"\n[Summary] {len(docs)} documents → "
          f"~{len(docs) * questions_per_doc} questions ...")

    results = []
    for i, doc in enumerate(docs):
        questions = ask_claude(client, SUMMARY_PROMPT.format(
            n=questions_per_doc,
            doc_type=doc["doc_type"],
            descripcion=doc["descripcion"] or "N/A",
            concepto=doc["concepto_contratacion"] or "N/A",
            ente_contratante=doc["ente_contratante"] or "N/A",
        ))
        for q in questions:
            results.append({
                "type":         "summary",
                "question":     q,
                "doc_id":       doc["doc_id"],
                "doc_type":     doc["doc_type"],
                "licitacion_id": doc["licitacion_id"],
                "chunk_ids":    doc["chunk_ids"],
                "descripcion":  doc["descripcion"] or doc["concepto_contratacion"],
            })
        print(f"  [{i+1}/{len(docs)}] {doc['doc_type']:<35} ({doc['chunk_count']} chunks) "
              f"→ {len(questions)} questions")
        time.sleep(0.2)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate RAG evaluation set")
    parser.add_argument("--samples",             type=int, default=50,
                        help="Chunks for specific questions (default: 50)")
    parser.add_argument("--questions-per-chunk", type=int, default=2,
                        help="Specific questions per chunk (default: 2)")
    parser.add_argument("--broad-samples",       type=int, default=40,
                        help="Licitaciones for topic questions (default: 40)")
    parser.add_argument("--metadata-samples",    type=int, default=30,
                        help="Licitaciones for metadata questions (default: 30)")
    parser.add_argument("--summary-samples",     type=int, default=20,
                        help="Documents for summary questions (default: 20)")
    parser.add_argument("--output",              default="eval_set.json")
    parser.add_argument("--min-tokens",          type=int, default=200)
    parser.add_argument("--chunk-config",        default="1024_256",
                        help="Chunk config to draw specific/summary questions from (default: 1024_256)")
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in .env")

    client = anthropic.Anthropic(api_key=api_key)
    conn   = get_conn()

    eval_set = []

    if args.samples > 0:
        eval_set += generate_specific(
            conn, client, args.samples, args.questions_per_chunk, args.min_tokens,
            chunk_config=args.chunk_config,
        )

    if args.broad_samples > 0:
        eval_set += generate_broad(conn, client, args.broad_samples)

    if args.metadata_samples > 0:
        eval_set += generate_metadata(conn, client, args.metadata_samples)

    if args.summary_samples > 0:
        eval_set += generate_summary(conn, client, args.summary_samples,
                                     chunk_config=args.chunk_config)

    conn.close()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, ensure_ascii=False, indent=2)

    counts = {t: sum(1 for e in eval_set if e["type"] == t)
              for t in ["specific", "topic", "metadata", "summary"]}

    print(f"\nDone — {len(eval_set)} eval pairs saved to {args.output}")
    print(f"  Specific (chunk facts)     : {counts['specific']}")
    print(f"  Topic (category discovery) : {counts['topic']}")
    print(f"  Metadata (dates/costs/places): {counts['metadata']}")
    print(f"  Summary (doc requirements) : {counts['summary']}")


if __name__ == "__main__":
    main()
