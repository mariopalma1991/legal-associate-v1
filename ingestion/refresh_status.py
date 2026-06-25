"""
refresh_status.py — Re-check licitacion_status for all Vigente records.

For each Vigente licitacion:
  1. Re-scrapes the live page to get the current status and document list
  2. Updates licitacion_status if it changed (e.g. Vigente → Terminado/Cancelado)
  3. Queues any new documents found on the page (status='pending') for download
  4. Updates last_checked_at for every successfully checked record

Usage:
  python refresh_status.py              # check all Vigente records
  python refresh_status.py --limit 20  # test with 20 records
  python refresh_status.py --dry-run   # report what would change, no writes
"""

import asyncio
import argparse
import json
import os
import time

import aiohttp
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

from extract_licitacion import parse_page

load_dotenv()

CONCURRENCY  = 30
TIMEOUT_SECS = 20
BATCH_SIZE   = 100


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
        user=user, password=password, sslmode="require",
    )


async def fetch_one(session: aiohttp.ClientSession, sem: asyncio.Semaphore,
                    licitacion_id: int, url: str) -> tuple[int, str | None]:
    async with sem:
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECS),
                allow_redirects=True,
            ) as resp:
                if resp.status == 200 and f"/licitaciones/{licitacion_id}/" in str(resp.url):
                    return licitacion_id, await resp.text(errors="replace")
                return licitacion_id, None
        except Exception:
            return licitacion_id, None


def _load_existing_doc_tipos(conn, licitacion_ids: list) -> dict:
    """Returns {licitacion_id: set(tipo)} for documents already in the DB."""
    if not licitacion_ids:
        return {}
    placeholders = ",".join(["%s"] * len(licitacion_ids))
    cur = conn.cursor()
    cur.execute(
        f"SELECT licitacion_id, tipo FROM documents WHERE licitacion_id IN ({placeholders})",
        licitacion_ids,
    )
    result: dict = {}
    for lid, tipo in cur.fetchall():
        result.setdefault(lid, set()).add(tipo)
    cur.close()
    return result


def _flush(conn, updates: list, dry_run: bool):
    """Write a batch of updates to the DB."""
    if dry_run or not updates:
        return
    with conn:
        cur = conn.cursor()
        for u in updates:
            lid        = u["licitacion_id"]
            new_status = u["new_status"]

            if u["status_changed"]:
                cur.execute(
                    "UPDATE licitaciones SET licitacion_status = %s, last_checked_at = now() WHERE id = %s",
                    (new_status, lid),
                )
            else:
                cur.execute(
                    "UPDATE licitaciones SET last_checked_at = now() WHERE id = %s",
                    (lid,),
                )

            # Only queue new docs when licitacion is still Vigente
            if u["new_docs"] and new_status == "Vigente":
                psycopg2.extras.execute_values(cur, """
                    INSERT INTO documents (licitacion_id, tipo, url, status)
                    VALUES %s
                    ON CONFLICT (licitacion_id, tipo) DO UPDATE
                        SET url = EXCLUDED.url
                        WHERE documents.status IN ('pending', 'error')
                """, [(lid, d["tipo"], d["url"], "pending") for d in u["new_docs"]])

        cur.close()


async def run(limit: int | None, dry_run: bool):
    conn = get_conn()
    cur  = conn.cursor()

    query = "SELECT id, url, licitacion_status FROM licitaciones WHERE licitacion_status = 'Vigente' ORDER BY id"
    if limit:
        query += f" LIMIT {limit}"
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()

    if not rows:
        print("No Vigente records found.")
        conn.close()
        return

    suffix = " (dry-run, no changes written)" if dry_run else ""
    print(f"Checking {len(rows):,} Vigente licitaciones{suffix} ...\n")

    licitacion_ids = [r[0] for r in rows]
    existing_docs  = _load_existing_doc_tipos(conn, licitacion_ids)
    id_to_row      = {r[0]: (r[1], r[2]) for r in rows}  # id → (url, old_status)

    # Suppress benign ProactorEventLoop pipe-close noise on Windows
    loop = asyncio.get_running_loop()
    def _suppress_10022(loop, context):
        exc = context.get("exception")
        if isinstance(exc, OSError) and getattr(exc, "winerror", None) == 10022:
            return
        loop.default_exception_handler(context)
    loop.set_exception_handler(_suppress_10022)

    sem       = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ssl=False)
    headers   = {"User-Agent": "Mozilla/5.0 (compatible; licitacion-refresh/1.0)"}

    status_changed: list[tuple] = []
    new_docs_found: list[tuple] = []
    failed:         list[int]   = []
    unchanged  = 0
    pending_writes: list[dict]  = []
    t0 = time.time()

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        tasks = [fetch_one(session, sem, r[0], r[1]) for r in rows]

        for coro in asyncio.as_completed(tasks):
            lid, html = await coro

            if html is None:
                failed.append(lid)
                print(f"  [FAIL]    id={lid}")
                continue

            url, old_status = id_to_row[lid]
            parsed     = parse_page(html, url)
            new_status = parsed.get("estatus", "")
            page_docs  = json.loads(parsed.get("documentos_json", "[]"))

            # Find docs on the page whose tipo isn't yet in the DB
            known = existing_docs.get(lid, set())
            new_docs = [d for d in page_docs if d.get("tipo") and d["tipo"] not in known]

            changed = old_status != new_status

            pending_writes.append({
                "licitacion_id": lid,
                "new_status":    new_status,
                "status_changed": changed,
                "new_docs":      new_docs,
            })

            if changed:
                status_changed.append((lid, old_status, new_status))
                print(f"  [CHANGED]  id={lid:<8}  {old_status} → {new_status}")
            elif new_docs:
                new_docs_found.append((lid, new_docs))
                tipos = ", ".join(d["tipo"] for d in new_docs)
                print(f"  [NEW DOCS] id={lid:<8}  +{len(new_docs)}: {tipos[:60]}")
            else:
                unchanged += 1

            if len(pending_writes) >= BATCH_SIZE:
                _flush(conn, pending_writes, dry_run)
                pending_writes = []

    _flush(conn, pending_writes, dry_run)

    elapsed = time.time() - t0
    total_new_docs = sum(len(docs) for _, docs in new_docs_found)

    print(f"\n{'-' * 55}")
    print(f"Done in {elapsed:.1f}s  ({len(rows)} checked)")
    print(f"  Status changed  : {len(status_changed)}")
    for lid, old, new in status_changed:
        print(f"    id={lid:<8}  {old} → {new}")
    print(f"  New docs queued : {total_new_docs} across {len(new_docs_found)} licitaciones")
    print(f"  Unchanged       : {unchanged}")
    print(f"  Failed (HTTP)   : {len(failed)}")

    if dry_run:
        print("\n[dry-run] No changes were written to the DB.")
    elif total_new_docs:
        print("\nNext step: run download_docs.py to fetch the new documents.")

    conn.close()


def main():
    ap = argparse.ArgumentParser(description="Re-check status of all Vigente licitaciones")
    ap.add_argument("--limit",   type=int, default=None,
                    help="Check at most N records (useful for testing)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report changes without writing to the DB")
    args = ap.parse_args()

    asyncio.run(run(args.limit, args.dry_run))


if __name__ == "__main__":
    main()
