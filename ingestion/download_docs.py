"""
Downloads documents (PDF, DOCX, XLSX) for Vigente licitaciones.

Reads:   documents WHERE status = 'pending' (joined to Vigente licitaciones)
Uploads: file bytes → Supabase Storage (bucket: licitaciones-docs)
Updates: documents SET local_path (storage key), file_size, status = 'downloaded'
         documents SET error, status = 'error'  (on failure, retried next run)

Usage:
  python ingestion/download_docs.py              # download all pending
  python ingestion/download_docs.py --limit 50  # download N docs (testing)
"""

import asyncio
import argparse
import io
import os
import time
import zipfile

import aiohttp
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
import sys as _sys; _sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
from shared._storage import BUCKET, make_key, upload_bytes
from _pipeline import start_run, finish_run, start_stage, finish_stage

load_dotenv()

CONCURRENCY  = 10
TIMEOUT_SECS = 60
BATCH_SIZE   = 50


# â”€â”€ DB connection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def detect_file_type(data: bytes) -> str:
    """Return 'pdf', 'docx', 'xlsx', 'pptx', 'zip', 'rar', 'ole', or 'unknown' from magic bytes."""
    if data[:4] == b"%PDF":
        return "pdf"
    if data[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = zf.namelist()
                if any(n.startswith("word/") for n in names):
                    return "docx"
                if any(n.startswith("xl/") for n in names):
                    return "xlsx"
                if any(n.startswith("ppt/") for n in names):
                    return "pptx"
        except Exception:
            pass
        return "zip"
    if data[:6] == b"Rar!\x1a\x07":      # RAR4 and RAR5
        return "rar"
    if data[:4] == b"\xd0\xcf\x11\xe0":
        return "ole"   # old DOC/XLS â€” handled in chunk_docs
    return "unknown"



# â”€â”€ Download â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def download_doc(session: aiohttp.ClientSession, sem: asyncio.Semaphore,
                       doc_id: str, licitacion_id: int, tipo: str,
                       url: str, verbose: bool = False) -> dict:
    """
    Download a document (PDF, DOCX, XLSX).
    File type is detected from content, not from the Content-Type header
    (the portal incorrectly sends application/pdf for all file types).
    """
    if verbose:
        print(f"  --> [{licitacion_id}] {tipo}  {url}")
    async with sem:
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECS),
                allow_redirects=True
            ) as resp:
                if resp.status != 200:
                    return {"id": doc_id, "status": "error",
                            "error": f"HTTP {resp.status}"}

                data = await resp.read()
                if not data:
                    return {"id": doc_id, "status": "error", "error": "Empty response"}

                file_type = detect_file_type(data)
                if file_type == "unknown":
                    return {"id": doc_id, "status": "error",
                            "error": f"Unknown file type (first bytes: {data[:8].hex()})"}

                key = make_key(licitacion_id, tipo, file_type)
                try:
                    upload_bytes(key, data)
                except Exception as e:
                    return {"id": doc_id, "status": "error",
                            "error": f"Storage upload failed: {e}"}

                return {
                    "id":         doc_id,
                    "status":     "downloaded",
                    "local_path": key,   # storage key, e.g. "271717/Bases.pdf"
                    "file_size":  len(data),
                    "error":      None,
                }

        except asyncio.TimeoutError:
            return {"id": doc_id, "status": "error", "error": "Timeout"}
        except Exception as e:
            return {"id": doc_id, "status": "error", "error": str(e)[:200]}


# â”€â”€ DB write â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def write_results(conn, results: list[dict]):
    with conn:
        cur = conn.cursor()
        for r in results:
            if r["status"] == "downloaded":
                cur.execute("""
                    UPDATE documents SET
                        status     = 'downloaded',
                        local_path = %s,
                        file_size  = %s,
                        error      = NULL,
                        updated_at = now()
                    WHERE id = %s
                """, (r["local_path"], r["file_size"], r["id"]))
            else:
                cur.execute("""
                    UPDATE documents SET
                        status     = 'error',
                        error      = %s,
                        updated_at = now()
                    WHERE id = %s
                """, (r["error"], r["id"]))
        cur.close()


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def run(limit: int | None = None, verbose: bool = False):
    conn     = get_conn()
    run_id   = start_run(conn, notes="download")
    stage_id = start_stage(conn, run_id, "download")
    cur      = conn.cursor()

    # Count pending docs from Vigente licitaciones only
    cur.execute("""
        SELECT COUNT(*) FROM documents d
        JOIN licitaciones l ON d.licitacion_id = l.id
        WHERE d.status = 'pending'
          AND l.licitacion_status = 'Vigente'
    """)
    row = cur.fetchone()
    total_available = row[0] if row else 0

    if total_available == 0:
        print("No pending documents to download.")
        finish_stage(conn, stage_id, "completed", items_found=0)
        finish_run(conn, run_id, "completed")
        conn.close()
        return

    total = min(total_available, limit) if limit else total_available
    print(f"Found {total_available:,} pending documents â€” downloading {total:,} ...")
    cur.close()

    sem       = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ssl=False)
    headers   = {"User-Agent": "Mozilla/5.0 (compatible; licitacion-downloader/1.0)"}

    downloaded  = 0
    errors      = 0
    total_bytes = 0
    t0          = time.time()

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        while downloaded + errors < total:
            remaining  = total - downloaded - errors
            fetch_size = min(BATCH_SIZE, remaining)

            cur = conn.cursor()
            cur.execute("""
                SELECT d.id, d.licitacion_id, d.tipo, d.url
                FROM documents d
                JOIN licitaciones l ON d.licitacion_id = l.id
                WHERE d.status = 'pending'
                  AND l.licitacion_status = 'Vigente'
                ORDER BY d.licitacion_id, d.tipo
                LIMIT %s
            """, (fetch_size,))
            batch = cur.fetchall()
            cur.close()

            if not batch:
                break

            results = await asyncio.gather(*[
                download_doc(session, sem, str(doc_id), lid, tipo, url, verbose)
                for doc_id, lid, tipo, url in batch
            ])

            write_results(conn, results)

            for r in results:
                if r["status"] == "downloaded":
                    kb = r["file_size"] / 1024
                    if verbose:
                        print(f"  [OK]    {r['local_path']}  ({kb:.0f} KB)")
                else:
                    print(f"  [FAIL]  id={r['id']}  {r['error']}")

            batch_dl  = sum(1 for r in results if r["status"] == "downloaded")
            batch_err = sum(1 for r in results if r["status"] == "error")
            batch_bytes = sum(r.get("file_size", 0) for r in results)

            downloaded  += batch_dl
            errors      += batch_err
            total_bytes += batch_bytes

            elapsed = time.time() - t0
            rate    = downloaded / elapsed if elapsed > 0 else 0
            eta     = (total - downloaded - errors) / rate / 60 if rate > 0 else 0

            print(
                f"  {downloaded + errors:,}/{total:,}  "
                f"downloaded={downloaded:,}  errors={errors}  "
                f"{total_bytes/1024/1024:.1f} MB  "
                f"~{eta:.1f} min remaining"
            )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  Downloaded : {downloaded:,} files")
    print(f"  Total size : {total_bytes/1024/1024:.1f} MB")
    print(f"  Errors     : {errors}")
    print(f"  Saved to   : Supabase Storage ({BUCKET})")

    stage_status = "completed" if errors == 0 else ("partial" if downloaded > 0 else "failed")
    finish_stage(conn, stage_id, stage_status,
                 items_found=total, items_ok=downloaded, items_error=errors)
    finish_run(conn, run_id, stage_status)
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Download PDFs for Vigente licitaciones")
    parser.add_argument("--limit", type=int, default=None,
                        help="Download only N documents (useful for testing)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each URL before downloading and confirm on success")
    args = parser.parse_args()
    asyncio.run(run(args.limit, args.verbose))


if __name__ == "__main__":
    main()
