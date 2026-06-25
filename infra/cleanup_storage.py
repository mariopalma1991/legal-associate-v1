"""
Delete Supabase Storage files and DB chunks for Terminado licitaciones.

Marks documents.status = 'deleted' (keeps the row for audit).
Deletes chunks to reclaim pgvector space.

Usage:
  python cleanup_storage.py            # delete all Terminado docs
  python cleanup_storage.py --dry-run  # preview only
"""

import argparse
import os

import psycopg2
from dotenv import load_dotenv
import sys as _sys; _sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
from shared._storage import delete_doc

load_dotenv()


def get_conn():
    url = os.getenv("DATABASE_URL")
    at = url.rfind("@")
    creds, host_part = url[len("postgresql://"):at], url[at + 1:]
    user, password = creds.split(":", 1)
    host, port = host_part.split("/")[0].split(":")
    dbname = host_part.split("/")[1].split("?")[0]
    return psycopg2.connect(
        host=host, port=int(port), dbname=dbname,
        user=user, password=password, sslmode="require"
    )


def main():
    ap = argparse.ArgumentParser(description="Clean up Storage + chunks for Terminado licitaciones")
    ap.add_argument("--dry-run", action="store_true", help="Preview only â€” no deletes")
    args = ap.parse_args()

    conn = get_conn()
    cur  = conn.cursor()

    cur.execute("""
        SELECT d.id, d.licitacion_id, d.tipo, d.local_path, d.status
        FROM documents d
        JOIN licitaciones l ON d.licitacion_id = l.id
        WHERE l.licitacion_status = 'Terminado'
          AND d.status != 'deleted'
          AND d.local_path IS NOT NULL
        ORDER BY d.licitacion_id, d.tipo
    """)
    docs = cur.fetchall()

    if not docs:
        print("No Terminado documents to clean up.")
        conn.close()
        return

    # Count unique licitaciones
    lics = {row[1] for row in docs}
    print(f"Found {len(docs):,} documents across {len(lics):,} Terminado licitaciones")
    if args.dry_run:
        for doc_id, lid, tipo, path, status in docs[:20]:
            print(f"  lid={lid}  {tipo[:50]}  [{status}]  {path}")
        if len(docs) > 20:
            print(f"  ... and {len(docs) - 20} more")
        cur.close()
        conn.close()
        return

    storage_deleted = storage_errors = chunks_deleted = docs_marked = 0

    for doc_id, lid, tipo, key, status in docs:
        # Delete from Storage
        if key:
            ok = delete_doc(key)
            if ok:
                storage_deleted += 1
            else:
                storage_errors += 1
                print(f"  [WARN] Storage delete failed: {key}")

        # Delete chunks for this document
        cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
        chunks_deleted += cur.rowcount

        # Mark document as deleted
        cur.execute("""
            UPDATE documents SET status = 'deleted', local_path = NULL, updated_at = now()
            WHERE id = %s
        """, (doc_id,))
        docs_marked += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"\nDone.")
    print(f"  Storage files deleted : {storage_deleted:,}  (errors: {storage_errors})")
    print(f"  Chunks deleted        : {chunks_deleted:,}")
    print(f"  Documents marked      : {docs_marked:,}")


if __name__ == "__main__":
    main()
