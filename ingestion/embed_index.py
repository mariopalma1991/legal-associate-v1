"""
Compute and store text embeddings for chunked documents.

Reads:   chunks WHERE emb_{model} IS NULL (documents with status 'chunked')
Updates: chunks SET emb_{model} = <vector>
         documents SET status = 'indexed'  (once all chunks have the embedding)

Models:
  openai  → text-embedding-3-small  (1536-d)  needs OPENAI_API_KEY
  cohere  → embed-multilingual-v3.0 (1024-d)  needs COHERE_API_KEY
  me5     → intfloat/multilingual-e5-large     runs locally (CPU/GPU, ~2 GB)

Usage:
  python embed_index.py --model openai
  python embed_index.py --model cohere
  python embed_index.py --model me5
  python embed_index.py --model all
  python embed_index.py --model openai --limit 500
  python embed_index.py --model openai --batch-size 128
"""

import argparse
import os
import time

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from _pipeline import start_run, finish_run, start_stage, finish_stage

load_dotenv()

DEFAULT_BATCH = {"openai": 256, "cohere": 96, "me5": 32}

MODEL_COL = {"openai": "emb_openai", "cohere": "emb_cohere", "me5": "emb_me5"}


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


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _vec_str(v: list[float]) -> str:
    return "[" + ",".join(map(str, v)) + "]"


def embed_openai(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in .env")
    resp = OpenAI(api_key=api_key).embeddings.create(
        model="text-embedding-3-large",
        input=texts,
        dimensions=1536,
    )
    return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]


def embed_cohere(texts: list[str]) -> list[list[float]]:
    import cohere
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY not set in .env")
    # Support cohere v4 (Client) and v5+ (ClientV2)
    try:
        client = cohere.ClientV2(api_key=api_key)
        resp = client.embed(
            texts=texts,
            model="embed-multilingual-v3.0",
            input_type="search_document",
            embedding_types=["float"],
        )
        return resp.embeddings.float_
    except AttributeError:
        client = cohere.Client(api_key=api_key)
        resp = client.embed(
            texts=texts,
            model="embed-multilingual-v3.0",
            input_type="search_document",
        )
        return resp.embeddings


def load_me5():
    from sentence_transformers import SentenceTransformer
    print("  Loading intfloat/multilingual-e5-large (first run downloads ~2 GB) ...")
    return SentenceTransformer("intfloat/multilingual-e5-large")


def embed_me5(texts: list[str], model) -> list[list[float]]:
    # e5 models need "passage: " prefix for document chunks
    prefixed = [f"passage: {t}" for t in texts]
    return model.encode(prefixed, normalize_embeddings=True).tolist()


# ── Core loop ─────────────────────────────────────────────────────────────────

def run_model(conn, model_name: str, limit: int | None, batch_size: int,
              llamaparse_only: bool = False, chunk_config: str = "1024_256",
              run_id: str | None = None):
    col = MODEL_COL[model_name]

    me5_model = load_me5() if model_name == "me5" else None

    llamaparse_filter = "AND d.parser = 'llamaparse'" if llamaparse_only else ""

    stage_id = start_stage(conn, run_id, "embed", config={
        "model": model_name, "chunk_config": chunk_config,
    }) if run_id else None

    cur = conn.cursor()
    cur.execute(f"""
        SELECT COUNT(*) FROM chunks c
        JOIN documents d   ON c.document_id   = d.id
        JOIN licitaciones l ON c.licitacion_id = l.id
        WHERE c.{col} IS NULL
          AND c.chunk_config = %s
          AND d.status IN ('chunked', 'indexed')
          AND l.licitacion_status = 'Vigente'
          {llamaparse_filter}
    """, (chunk_config,))
    total_available = cur.fetchone()[0]
    cur.close()

    if total_available == 0:
        print(f"  [{model_name}] No chunks to embed for config={chunk_config}.")
        if stage_id:
            finish_stage(conn, stage_id, "completed", items_found=0)
        return

    total = min(total_available, limit) if limit else total_available
    label = " (LlamaParse only)" if llamaparse_only else ""
    print(f"  [{model_name}]{label} config={chunk_config}  "
          f"{total_available:,} chunks pending → embedding {total:,} "
          f"(batch_size={batch_size})")

    embedded     = 0
    errors       = 0
    docs_indexed = set()
    t0           = time.time()

    while embedded + errors < total:
        remaining  = total - embedded - errors
        fetch_size = min(batch_size, remaining)

        cur = conn.cursor()
        cur.execute(f"""
            SELECT c.id, c.text, c.document_id
            FROM chunks c
            JOIN documents d   ON c.document_id   = d.id
            JOIN licitaciones l ON c.licitacion_id = l.id
            WHERE c.{col} IS NULL
              AND c.chunk_config = %s
              AND d.status IN ('chunked', 'indexed')
              AND l.licitacion_status = 'Vigente'
              {llamaparse_filter}
            ORDER BY c.document_id, c.chunk_index
            LIMIT %s
        """, (chunk_config, fetch_size))
        batch = cur.fetchall()
        cur.close()

        if not batch:
            break

        chunk_ids = [row[0] for row in batch]
        texts     = [row[1] for row in batch]
        doc_ids   = [str(row[2]) for row in batch]

        try:
            if model_name == "openai":
                vectors = embed_openai(texts)
            elif model_name == "cohere":
                vectors = embed_cohere(texts)
            else:
                vectors = embed_me5(texts, me5_model)
        except Exception as e:
            print(f"  [{model_name}] ERROR: {e}")
            errors += len(batch)
            continue

        with conn:
            cur = conn.cursor()
            psycopg2.extras.execute_values(cur, f"""
                UPDATE chunks AS c
                SET {col} = v.vec::vector
                FROM (VALUES %s) AS v(id, vec)
                WHERE c.id = v.id::uuid
            """, [(str(cid), _vec_str(vec)) for cid, vec in zip(chunk_ids, vectors)])
            cur.close()

        embedded += len(batch)

        # Mark documents as indexed once all their chunks (for this config) have the embedding
        for doc_id in set(doc_ids) - docs_indexed:
            cur = conn.cursor()
            cur.execute(
                f"SELECT COUNT(*) FROM chunks WHERE document_id = %s AND chunk_config = %s AND {col} IS NULL",
                (doc_id, chunk_config),
            )
            remaining_chunks = cur.fetchone()[0]
            cur.close()

            if remaining_chunks == 0:
                with conn:
                    cur = conn.cursor()
                    cur.execute("""
                        UPDATE documents SET status = 'indexed', updated_at = now()
                        WHERE id = %s
                    """, (doc_id,))
                    cur.close()
                docs_indexed.add(doc_id)

        elapsed = time.time() - t0
        rate    = embedded / elapsed if elapsed > 0 else 0
        eta     = (total - embedded - errors) / rate / 60 if rate > 0 else 0
        print(f"  [{model_name}] {embedded:,}/{total:,}  "
              f"docs_indexed={len(docs_indexed)}  ~{eta:.1f} min remaining")

    elapsed = time.time() - t0
    print(f"\n  [{model_name}] Done in {elapsed / 60:.1f} min")
    print(f"    Embedded : {embedded:,} chunks")
    print(f"    Indexed  : {len(docs_indexed)} documents")
    print(f"    Errors   : {errors}")

    if stage_id:
        stage_status = "completed" if errors == 0 else ("partial" if embedded > 0 else "failed")
        finish_stage(conn, stage_id, stage_status,
                     items_found=total, items_ok=embedded, items_error=errors)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Embed chunks and store in pgvector")
    parser.add_argument("--model",      choices=["openai", "cohere", "me5", "all"],
                        default="openai",
                        help="Embedding model (default: openai)")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Max chunks to process (testing)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override default API batch size")
    parser.add_argument("--llamaparse-only", action="store_true",
                        help="Only embed chunks from LlamaParse-parsed documents")
    parser.add_argument("--chunk-config", default="1024_256",
                        help="Which chunk config to embed (default: 1024_256)")
    args = parser.parse_args()

    conn    = get_conn()
    run_id  = start_run(conn, notes=f"embed:{args.model}:{args.chunk_config}")
    models  = ["openai", "cohere", "me5"] if args.model == "all" else [args.model]

    for m in models:
        run_model(conn, m, args.limit, args.batch_size or DEFAULT_BATCH[m],
                  llamaparse_only=args.llamaparse_only,
                  chunk_config=args.chunk_config,
                  run_id=run_id)

    finish_run(conn, run_id, "completed")
    conn.close()
    print("\nAll done.")


if __name__ == "__main__":
    main()
