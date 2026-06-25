"""
Scrapes metadata from discovered licitacion pages and writes to PostgreSQL.

Reads:   licitaciones WHERE pipeline_status = 'discovered'
Updates: licitaciones SET all fields, pipeline_status = 'scraped'
Inserts: documents (one row per PDF found on the page)

Usage:
  python ingest.py              # process all discovered records
  python ingest.py --batch 50  # override batch size
"""

import asyncio
import argparse
import os
import time
import json

import aiohttp
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

from extract_licitacion import parse_page
from _pipeline import start_run, finish_run, start_stage, finish_stage

load_dotenv()

CONCURRENCY  = 50
TIMEOUT_SECS = 20
BATCH_SIZE   = 100


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


# ── Fetch ─────────────────────────────────────────────────────────────────────

async def fetch(session: aiohttp.ClientSession, sem: asyncio.Semaphore,
                licitacion_id: int, url: str) -> tuple[int, str | None]:
    """Fetch a licitacion page. Returns (id, html) or (id, None) on failure."""
    async with sem:
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECS),
                allow_redirects=True
            ) as resp:
                if resp.status == 200 and f"/licitaciones/{licitacion_id}/" in str(resp.url):
                    return licitacion_id, await resp.text(errors="replace")
                return licitacion_id, None
        except Exception as e:
            return licitacion_id, None


# ── DB writes ─────────────────────────────────────────────────────────────────

def write_batch(conn, records: list[dict]):
    """
    Write a batch of parsed records to the DB in a single transaction.
    Updates licitaciones, inserts documents.
    On deadlock, rolls back and leaves records as 'discovered' for retry.
    """
    if not records:
        return

    try:
        with conn:
            cur = conn.cursor()

            for r in records:
                cur.execute("""
                    UPDATE licitaciones SET
                        numero_procedimiento     = %(numero_procedimiento)s,
                        tipo_procedimiento       = %(tipo_procedimiento)s,
                        licitacion_status        = %(licitacion_status)s,
                        ente_contratante         = %(ente_contratante)s,
                        ente_solicitante         = %(ente_solicitante)s,
                        documento_programado     = %(documento_programado)s,
                        materia                  = %(materia)s,
                        tipo_contrato            = %(tipo_contrato)s,
                        concepto_contratacion    = %(concepto_contratacion)s,
                        descripcion              = %(descripcion)s,
                        fundamento_legal         = %(fundamento_legal)s,
                        modalidad                = %(modalidad)s,
                        fecha_convocatoria       = %(fecha_convocatoria)s,
                        fecha_junta_aclaraciones = %(fecha_junta_aclaraciones)s,
                        hora_junta_aclaraciones  = %(hora_junta_aclaraciones)s,
                        lugar_junta_aclaraciones = %(lugar_junta_aclaraciones)s,
                        fecha_apertura           = %(fecha_apertura)s,
                        hora_apertura            = %(hora_apertura)s,
                        lugar_apertura           = %(lugar_apertura)s,
                        costo_participacion      = %(costo_participacion)s,
                        pipeline_status          = 'scraped',
                        last_checked_at          = now()
                    WHERE id = %(licitacion_id)s
                """, r)

                # Only queue documents for Vigente records —
                # no point downloading PDFs for closed procedures.
                docs = json.loads(r.get("documentos_json", "[]"))
                if docs and r.get("licitacion_status") == "Vigente":
                    psycopg2.extras.execute_values(cur, """
                        INSERT INTO documents (licitacion_id, tipo, url, status)
                        VALUES %s
                        ON CONFLICT (licitacion_id, tipo) DO UPDATE
                            SET url = EXCLUDED.url
                            WHERE documents.status IN ('pending', 'error')
                    """, [(r["licitacion_id"], d["tipo"], d["url"], "pending") for d in docs])

            cur.close()
    except psycopg2.errors.DeadlockDetected:
        conn.rollback()
        ids = [r["licitacion_id"] for r in records]
        print(f"  [DEADLOCK] batch rolled back, {len(ids)} records stay 'discovered' for retry")


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def process_batch(session: aiohttp.ClientSession, sem: asyncio.Semaphore,
                        batch: list[tuple[int, str]], conn) -> tuple[int, int]:
    """
    Fetch, parse, and write one batch.
    Returns (scraped_count, docs_count).
    """
    # Fetch all pages concurrently
    results = await asyncio.gather(*[
        fetch(session, sem, lid, url) for lid, url in batch
    ])

    records  = []
    failed   = []
    doc_count = 0

    url_map = {lid: url for lid, url in batch}

    for lid, html in results:
        if html is None:
            failed.append(lid)
            print(f"  [FAIL]  id={lid}")
            continue

        raw = parse_page(html, url_map[lid])

        # Map parse_page keys → DB column names
        record = {
            "licitacion_id":           lid,
            "numero_procedimiento":    raw.get("numero_procedimiento", ""),
            "tipo_procedimiento":      raw.get("tipo_procedimiento", ""),
            "licitacion_status":       raw.get("estatus", ""),
            "ente_contratante":        raw.get("ente_contratante", ""),
            "ente_solicitante":        raw.get("ente_solicitante", ""),
            "documento_programado":    raw.get("documento_programado", ""),
            "materia":                 raw.get("materia", ""),
            "tipo_contrato":           raw.get("tipo_contrato", ""),
            "concepto_contratacion":   raw.get("concepto_contratacion", ""),
            "descripcion":             raw.get("descripcion_procedimiento", ""),
            "fundamento_legal":        raw.get("fundamento_legal", ""),
            "modalidad":               raw.get("modalidad", ""),
            "fecha_convocatoria":      raw.get("fecha_publicacion_convocatoria", ""),
            "fecha_junta_aclaraciones":raw.get("fecha_junta_aclaraciones", ""),
            "hora_junta_aclaraciones": raw.get("hora_junta_aclaraciones", ""),
            "lugar_junta_aclaraciones":raw.get("lugar_junta_aclaraciones", ""),
            "fecha_apertura":          raw.get("fecha_apertura_propuestas", ""),
            "hora_apertura":           raw.get("hora_apertura_propuestas", ""),
            "lugar_apertura":          raw.get("lugar_apertura_propuestas", ""),
            "costo_participacion":     raw.get("costo_participacion", ""),
            "documentos_json":         raw.get("documentos_json", "[]"),
        }
        records.append(record)
        ndocs = len(json.loads(record["documentos_json"]))
        doc_count += ndocs
        status = record["licitacion_status"] or "?"
        docs_label = f"  {ndocs} docs" if ndocs else ""
        print(f"  [OK]    id={lid:<8}  {status:<15}{docs_label}")

    write_batch(conn, records)
    return len(records), doc_count, failed


async def run(batch_size: int, limit: int | None = None):
    conn    = get_conn()
    run_id  = start_run(conn, notes="ingest")
    stage_id = start_stage(conn, run_id, "ingest", config={"batch_size": batch_size})
    cur     = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM licitaciones WHERE pipeline_status = 'discovered';")
    row = cur.fetchone()
    total_available = row[0] if row else 0

    if total_available == 0:
        print("No discovered records to process.")
        finish_stage(conn, stage_id, "completed", items_found=0)
        finish_run(conn, run_id, "completed")
        conn.close()
        return

    total = min(total_available, limit) if limit else total_available
    print(f"Found {total_available:,} discovered records — processing {total:,} "
          f"(batch_size={batch_size}) ...")
    cur.close()

    sem       = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ssl=False)
    headers   = {"User-Agent": "Mozilla/5.0 (compatible; licitacion-ingestor/1.0)"}

    scraped_total = 0
    docs_total    = 0
    failed_total  = []
    t0            = time.time()

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        while scraped_total < total:
            remaining  = total - scraped_total
            fetch_size = min(batch_size, remaining)

            cur = conn.cursor()
            cur.execute("""
                SELECT id, url FROM licitaciones
                WHERE pipeline_status = 'discovered'
                ORDER BY id
                LIMIT %s
            """, (fetch_size,))
            batch = cur.fetchall()
            cur.close()

            if not batch:
                break

            scraped, docs, failed = await process_batch(session, sem, batch, conn)
            scraped_total += scraped
            docs_total    += docs
            failed_total  += failed

            elapsed = time.time() - t0
            rate    = scraped_total / elapsed if elapsed > 0 else 0
            eta     = (total - scraped_total) / rate / 60 if rate > 0 else 0

            print(
                f"  {scraped_total:,}/{total:,} scraped  "
                f"{docs_total:,} docs  "
                f"{len(failed_total)} errors  "
                f"~{eta:.1f} min remaining"
            )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  Scraped  : {scraped_total:,}")
    print(f"  Documents: {docs_total:,}")
    print(f"  Failed   : {len(failed_total)}")
    if failed_total:
        print(f"  Failed IDs (stay 'discovered', retry next run): {failed_total[:20]}")

    stage_status = "completed" if not failed_total else "partial"
    finish_stage(conn, stage_id, stage_status,
                 items_found=total, items_ok=scraped_total,
                 items_error=len(failed_total),
                 error_summary=f"Failed IDs: {failed_total[:10]}" if failed_total else None)
    finish_run(conn, run_id, stage_status)
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Scrape metadata into PostgreSQL")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE,
                        help=f"Records per batch (default: {BATCH_SIZE})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N records (useful for testing)")
    args = parser.parse_args()
    asyncio.run(run(args.batch, args.limit))


if __name__ == "__main__":
    main()
