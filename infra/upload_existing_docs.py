"""
One-time bulk upload: push all files in docs/ to Supabase Storage and
update documents.local_path to the storage key.

Usage:
  python upload_existing_docs.py           # upload everything
  python upload_existing_docs.py --dry-run # preview only, no uploads
"""

import argparse
import os
import time

import psycopg2
from dotenv import load_dotenv
import sys as _sys; _sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
from shared._storage import BUCKET, DOCS_DIR, local_to_key, upload_bytes

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


def collect_files() -> list[tuple[str, str]]:
    """Return [(local_path, storage_key)] for every file under docs/."""
    files = []
    for root, _, filenames in os.walk(DOCS_DIR):
        for filename in filenames:
            local = os.path.join(root, filename)
            key   = local_to_key(local)
            files.append((local, key))
    return files


def main():
    ap = argparse.ArgumentParser(description="Upload local docs/ to Supabase Storage")
    ap.add_argument("--dry-run", action="store_true", help="Preview only â€” no uploads")
    args = ap.parse_args()

    files = collect_files()
    if not files:
        print(f"No files found under {DOCS_DIR}/")
        return

    print(f"Found {len(files):,} files to upload â†’ bucket: {BUCKET}")
    if args.dry_run:
        for local, key in files[:20]:
            print(f"  {local}  â†’  {key}")
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more")
        return

    conn = get_conn()
    cur  = conn.cursor()

    uploaded = errors = skipped = 0
    total_bytes = 0
    t0 = time.time()

    for i, (local, key) in enumerate(files, 1):
        # Skip if already uploaded (key already in DB)
        cur.execute("SELECT 1 FROM documents WHERE local_path = %s", (key,))
        if cur.fetchone():
            skipped += 1
            continue

        size = os.path.getsize(local)
        try:
            with open(local, "rb") as f:
                data = f.read()
            upload_bytes(key, data)
            cur.execute(
                "UPDATE documents SET local_path = %s WHERE local_path = %s",
                (key, local),
            )
            conn.commit()
            uploaded += 1
            total_bytes += size
            print(f"  [OK]  ({i}/{len(files)})  {key}  ({size/1024:.0f} KB)")
        except Exception as e:
            errors += 1
            print(f"  [ERR] ({i}/{len(files)})  {key}  {e}")

    elapsed = time.time() - t0
    cur.close()
    conn.close()

    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  Uploaded : {uploaded:,} files  ({total_bytes/1024/1024:.1f} MB)")
    print(f"  Skipped  : {skipped} (already uploaded)")
    print(f"  Errors   : {errors}")


if __name__ == "__main__":
    main()
