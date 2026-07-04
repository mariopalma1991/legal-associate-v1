"""
Evaluates retriever configurations against the generated eval set.
Computes Hit@1, Hit@5, Hit@10 and MRR for each config.
Outputs a comparison table to eval_results.md.

Configs tested (skips any model whose embedding column is not yet populated):
  - BM25 only
  - Dense - Cohere
  - Dense - OpenAI
  - Dense - mE5
  - Hybrid Cohere + BM25 (RRF)
  - Hybrid OpenAI + BM25 (RRF)

Usage:
  python eval_retriever.py
  python eval_retriever.py --eval-set eval_set.json --output eval_results.md --top-k 10
"""

import argparse
import json
import os
import time
from datetime import datetime

import psycopg2
from dotenv import load_dotenv

load_dotenv()

K_VALUES = [1, 5, 10, 30]


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


# ── Query embedders ───────────────────────────────────────────────────────────

def embed_query_cohere(question: str) -> list[float]:
    import cohere
    client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    resp = client.embed(
        texts=[question],
        model="embed-multilingual-v3.0",
        input_type="search_query",
        embedding_types=["float"],
    )
    return resp.embeddings.float_[0]


def embed_query_openai(question: str) -> list[float]:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.embeddings.create(input=[question], model="text-embedding-3-small")
    return resp.data[0].embedding


def embed_query_me5(question: str, model) -> list[float]:
    return model.encode([f"query: {question}"], normalize_embeddings=True)[0].tolist()


# ── Reranker ─────────────────────────────────────────────────────────────────

def rerank_cohere(conn, question: str, chunk_ids: list[str], top_n: int) -> list[str]:
    """Rerank a list of chunk_ids using Cohere rerank-multilingual-v3.0."""
    if not chunk_ids:
        return []
    import cohere
    cur = conn.cursor()
    cur.execute(
        "SELECT id::text, text FROM chunks WHERE id = ANY(%s::uuid[])",
        (chunk_ids,)
    )
    id_to_text = {row[0]: row[1] for row in cur.fetchall()}
    cur.close()

    docs = [id_to_text.get(cid, "") for cid in chunk_ids]
    client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    resp = client.rerank(
        query=question,
        documents=docs,
        model="rerank-multilingual-v3.0",
        top_n=min(top_n, len(docs)),
    )
    return [chunk_ids[r.index] for r in resp.results]


# ── Retrievers ────────────────────────────────────────────────────────────────

def retrieve_bm25(conn, question: str, k: int, chunk_config: str) -> list[str]:
    cur = conn.cursor()
    cur.execute("""
        SELECT c.id::text
        FROM chunks c
        JOIN documents d    ON c.document_id   = d.id
        JOIN licitaciones l ON c.licitacion_id = l.id
        WHERE l.licitacion_status = 'Vigente'
          AND d.parser = 'llamaparse'
          AND c.chunk_config = %s
          AND c.fts @@ plainto_tsquery('spanish', %s)
        ORDER BY ts_rank(c.fts, plainto_tsquery('spanish', %s)) DESC
        LIMIT %s
    """, (chunk_config, question, question, k))
    results = [r[0] for r in cur.fetchall()]
    cur.close()
    return results


def retrieve_dense(conn, query_vec: list[float], col: str, k: int,
                   chunk_config: str) -> list[str]:
    cur = conn.cursor()
    cur.execute(f"""
        SELECT c.id::text
        FROM chunks c
        JOIN documents d    ON c.document_id   = d.id
        JOIN licitaciones l ON c.licitacion_id = l.id
        WHERE l.licitacion_status = 'Vigente'
          AND d.parser = 'llamaparse'
          AND c.chunk_config = %s
          AND c.{col} IS NOT NULL
        ORDER BY c.{col} <=> %s::vector
        LIMIT %s
    """, (chunk_config, json.dumps(query_vec), k))
    results = [r[0] for r in cur.fetchall()]
    cur.close()
    return results


def retrieve_hybrid_rrf(conn, question: str, query_vec: list[float],
                         col: str, k: int, chunk_config: str) -> list[str]:
    cur = conn.cursor()
    cur.execute(f"""
        WITH dense AS (
            SELECT c.id,
                   ROW_NUMBER() OVER (ORDER BY c.{col} <=> %s::vector) AS rk
            FROM chunks c
            JOIN documents d    ON c.document_id   = d.id
            JOIN licitaciones l ON c.licitacion_id = l.id
            WHERE l.licitacion_status = 'Vigente'
              AND d.parser = 'llamaparse'
              AND c.chunk_config = %s
              AND c.{col} IS NOT NULL
            LIMIT 60
        ),
        bm25 AS (
            SELECT c.id,
                   ROW_NUMBER() OVER (ORDER BY ts_rank(c.fts, query) DESC) AS rk
            FROM chunks c
            JOIN documents d    ON c.document_id   = d.id
            JOIN licitaciones l ON c.licitacion_id = l.id,
                 plainto_tsquery('spanish', %s) query
            WHERE l.licitacion_status = 'Vigente'
              AND d.parser = 'llamaparse'
              AND c.chunk_config = %s
              AND c.fts @@ plainto_tsquery('spanish', %s)
            LIMIT 60
        )
        SELECT COALESCE(d.id, b.id)::text,
               1.0 / (60 + COALESCE(d.rk, 1000)) +
               1.0 / (60 + COALESCE(b.rk, 1000)) AS rrf_score
        FROM dense d
        FULL OUTER JOIN bm25 b ON d.id = b.id
        ORDER BY rrf_score DESC
        LIMIT %s
    """, (json.dumps(query_vec), chunk_config, question, chunk_config, question, k))
    results = [r[0] for r in cur.fetchall()]
    cur.close()
    return results


# ── Snippet matching ──────────────────────────────────────────────────────────

def chunks_matching_snippet(conn, chunk_ids: list[str], snippet: str) -> set[str]:
    """
    Return the subset of chunk_ids whose text contains the answer snippet.
    Uses the first 80 chars of the snippet as the search needle — short enough
    to fit inside any chunk size we might test, long enough to be distinctive.
    """
    if not chunk_ids or not snippet:
        return set()
    needle = snippet[:80].strip()
    cur = conn.cursor()
    cur.execute(
        "SELECT id::text FROM chunks WHERE id = ANY(%s::uuid[]) AND text ILIKE %s",
        (chunk_ids, f"%{needle}%"),
    )
    result = {row[0] for row in cur.fetchall()}
    cur.close()
    return result


# ── Metrics ───────────────────────────────────────────────────────────────────

def hit_at_k_set(retrieved: list[str], valid_ids: set, k: int) -> float:
    """Hit@k: did any retrieved item appear in the valid set?"""
    return 1.0 if set(retrieved[:k]) & valid_ids else 0.0


def reciprocal_rank_set(retrieved: list[str], valid_ids: set) -> float:
    for i, r in enumerate(retrieved):
        if r in valid_ids:
            return 1.0 / (i + 1)
    return 0.0


def compute_metrics(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    metrics = {}
    for k in K_VALUES:
        metrics[f"hit@{k}"] = sum(r[f"hit@{k}"] for r in results) / n
    metrics["mrr"]        = sum(r["rr"] for r in results) / n
    metrics["latency_ms"] = sum(r["latency_ms"] for r in results) / n
    return metrics


def get_chunks_for_licitaciones(conn, licitacion_ids: list[int],
                                chunk_config: str) -> set[str]:
    """Return chunk_ids for the given licitaciones restricted to chunk_config."""
    if not licitacion_ids:
        return set()
    cur = conn.cursor()
    cur.execute(
        "SELECT id::text FROM chunks WHERE licitacion_id = ANY(%s) AND chunk_config = %s",
        (licitacion_ids, chunk_config),
    )
    result = {row[0] for row in cur.fetchall()}
    cur.close()
    return result


# ── Column availability check ─────────────────────────────────────────────────

def check_columns(conn, chunk_config: str) -> dict[str, bool]:
    cur = conn.cursor()
    available = {}
    for col in ["emb_cohere", "emb_openai", "emb_me5"]:
        cur.execute(
            f"SELECT COUNT(*) FROM chunks WHERE chunk_config = %s AND {col} IS NOT NULL",
            (chunk_config,),
        )
        available[col] = cur.fetchone()[0] > 0
    cur.close()
    return available


# ── Eval loop ─────────────────────────────────────────────────────────────────

def eval_config(conn, name: str, retriever_fn, eval_set: list[dict],
                top_k: int, chunk_config: str) -> dict:
    print(f"  Running: {name} ...")
    rows_by_type: dict[str, list] = {
        "specific": [], "topic": [], "metadata": [], "summary": []
    }

    skipped = 0
    for item in eval_set:
        question      = item["question"]
        question_type = item.get("type", "specific")

        t0 = time.perf_counter()
        try:
            retrieved = retriever_fn(question)
        except Exception as e:
            print(f"    [WARN] skipping question due to error: {e}")
            skipped += 1
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Resolve ground truth to a set of valid chunk_ids.
        if question_type == "specific":
            snippet = item.get("text_snippet", "")
            if snippet:
                # Text-based: any retrieved chunk containing the answer snippet counts.
                # Config-agnostic — works across all chunk sizes.
                valid_chunks = chunks_matching_snippet(conn, retrieved, snippet)
            else:
                # Fallback for old eval set entries without text_snippet
                valid_chunks = {item["chunk_id"]}

        elif question_type == "topic":
            valid_chunks = get_chunks_for_licitaciones(
                conn, item["licitacion_ids"], chunk_config)

        elif question_type == "metadata":
            valid_chunks = get_chunks_for_licitaciones(
                conn, [item["licitacion_id"]], chunk_config)

        elif question_type == "summary":
            # chunk_ids are tied to the config used when the eval set was generated;
            # on other configs these UUIDs won't exist and hits will be 0.
            if item.get("licitacion_id"):
                valid_chunks = get_chunks_for_licitaciones(
                    conn, [item["licitacion_id"]], chunk_config)
            else:
                valid_chunks = set(item.get("chunk_ids", []))

        else:
            valid_chunks = set()

        row = {
            "latency_ms": elapsed_ms,
            "rr": reciprocal_rank_set(retrieved, valid_chunks),
        }
        for k in K_VALUES:
            row[f"hit@{k}"] = hit_at_k_set(retrieved, valid_chunks, k)

        rows_by_type[question_type].append(row)

    all_rows = [r for rows in rows_by_type.values() for r in rows]
    metrics  = compute_metrics(all_rows)
    metrics["config"]  = name
    metrics["skipped"] = skipped
    for qtype, rows in rows_by_type.items():
        metrics[f"{qtype}_hit@5"] = compute_metrics(rows).get("hit@5") if rows else None
    if skipped:
        print(f"    [WARN] {skipped} questions skipped due to errors")
    print(f"  ✓ {name:<28} Hit@5={metrics.get('hit@5', 0):.3f}  "
          f"MRR={metrics.get('mrr', 0):.3f}  "
          f"specific={fmt(metrics.get('specific_hit@5'))}  "
          f"metadata={fmt(metrics.get('metadata_hit@5'))}  "
          f"summary={fmt(metrics.get('summary_hit@5'))}  "
          f"{metrics.get('latency_ms', 0):.0f}ms/q")
    return metrics


# ── Markdown output ───────────────────────────────────────────────────────────

def fmt(val) -> str:
    return f"{val:.3f}" if val is not None else "—"


def write_markdown(results: list[dict], eval_set: list[dict],
                   chunk_size: int, overlap: int, output: str):
    counts = {t: sum(1 for e in eval_set if e.get("type", "specific") == t)
              for t in ["specific", "topic", "metadata", "summary"]}
    best_hit5 = max(r.get("hit@5", 0) for r in results)

    lines = [
        "# RAG Retriever Evaluation Results",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "\n## Configuration",
        f"- Chunk size : {chunk_size} tokens",
        f"- Overlap    : {overlap} tokens",
        f"- Eval set   : {len(eval_set)} questions  "
        f"(specific={counts['specific']}, topic={counts['topic']}, "
        f"metadata={counts['metadata']}, summary={counts['summary']})",
        "\n## Overall Results\n",
        "| Config | Hit@1 | Hit@5 | Hit@10 | MRR | Latency (ms) |",
        "|--------|-------|-------|--------|-----|-------------|",
    ]
    for r in results:
        bold = r.get("hit@5", 0) == best_hit5
        lines.append(
            f"| {'**' if bold else ''}{r['config']}{'**' if bold else ''} "
            f"| {fmt(r.get('hit@1'))} "
            f"| {fmt(r.get('hit@5'))} "
            f"| {fmt(r.get('hit@10'))} "
            f"| {fmt(r.get('mrr'))} "
            f"| {r.get('latency_ms', 0):.0f} ms |"
        )

    lines += [
        "\n## Hit@5 by Question Type\n",
        "| Config | Specific | Topic | Metadata | Summary |",
        "|--------|----------|-------|----------|---------|",
    ]
    for r in results:
        lines.append(
            f"| {r['config']} "
            f"| {fmt(r.get('specific_hit@5'))} "
            f"| {fmt(r.get('topic_hit@5'))} "
            f"| {fmt(r.get('metadata_hit@5'))} "
            f"| {fmt(r.get('summary_hit@5'))} |"
        )

    winner    = max(results, key=lambda r: r.get("hit@5", 0))
    bm25_hit5 = next((r.get("hit@5", 0) for r in results if r["config"] == "BM25 only"), 0)
    improvement = ((winner.get("hit@5", 0) - bm25_hit5) / bm25_hit5 * 100) if bm25_hit5 > 0 else 0

    lines += [f"\n## Winner: {winner['config']}", f"Best Hit@5: {winner.get('hit@5', 0):.3f}"]
    if improvement > 0:
        lines.append(f"Improvement over BM25: +{improvement:.1f}%")

    lines += ["\n## Sample Questions by Type", ""]
    for qtype in ["specific", "topic", "metadata", "summary"]:
        sample = next((e for e in eval_set if e.get("type") == qtype), None)
        if sample:
            lines.append(f"**{qtype.capitalize()}:** *{sample['question']}*")

    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved to {output}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retriever configurations")
    parser.add_argument("--eval-set",   default="eval_set.json")
    parser.add_argument("--output",     default=None,
                        help="Output file (default: eval_results_{chunk_config}.md)")
    parser.add_argument("--top-k",            type=int, default=30)
    parser.add_argument("--rerank-candidates", type=int, default=100,
                        help="Candidates fetched before reranking (default: 100)")
    parser.add_argument("--chunk-config", default="1024_256",
                        help="Which chunk config to evaluate (default: 1024_256)")
    args = parser.parse_args()

    # Derive chunk_size / overlap from chunk_config for reporting
    try:
        cs, ov = args.chunk_config.split("_")
        chunk_size, overlap = int(cs), int(ov)
    except ValueError:
        chunk_size, overlap = 1024, 256

    output = args.output or f"eval_results_{args.chunk_config}.md"

    if not os.path.exists(args.eval_set):
        print(f"Eval set not found: {args.eval_set}")
        print("Run generate_eval_set.py first.")
        return

    with open(args.eval_set, encoding="utf-8") as f:
        eval_set = json.load(f)

    print(f"Loaded {len(eval_set)} eval questions from {args.eval_set}")
    print(f"Chunk config: {args.chunk_config}")

    cfg       = args.chunk_config
    conn      = get_conn()
    available = check_columns(conn, cfg)
    results   = []

    rc = args.rerank_candidates  # shorthand

    # ── 1. BM25 ──────────────────────────────────────────────────────────────
    results.append(eval_config(
        conn, "BM25 only",
        lambda q: retrieve_bm25(conn, q, args.top_k, cfg),
        eval_set, args.top_k, cfg
    ))

    # ── 2. Dense - Cohere ────────────────────────────────────────────────────
    if available.get("emb_cohere"):
        results.append(eval_config(
            conn, "Dense - Cohere",
            lambda q: retrieve_dense(conn, embed_query_cohere(q), "emb_cohere", args.top_k, cfg),
            eval_set, args.top_k, cfg
        ))

    # ── 3. Dense - Cohere + Rerank ───────────────────────────────────────────
    if available.get("emb_cohere"):
        results.append(eval_config(
            conn, "Dense - Cohere + Rerank",
            lambda q: rerank_cohere(
                conn, q,
                retrieve_dense(conn, embed_query_cohere(q), "emb_cohere", rc, cfg),
                args.top_k
            ),
            eval_set, args.top_k, cfg
        ))

    # ── 4. Hybrid Cohere + BM25 ──────────────────────────────────────────────
    if available.get("emb_cohere"):
        results.append(eval_config(
            conn, "Hybrid Cohere + BM25",
            lambda q: retrieve_hybrid_rrf(conn, q, embed_query_cohere(q), "emb_cohere", args.top_k, cfg),
            eval_set, args.top_k, cfg
        ))

    # ── 5. Hybrid Cohere + BM25 + Rerank ────────────────────────────────────
    if available.get("emb_cohere"):
        results.append(eval_config(
            conn, "Hybrid Cohere + Rerank",
            lambda q: rerank_cohere(
                conn, q,
                retrieve_hybrid_rrf(conn, q, embed_query_cohere(q), "emb_cohere", rc, cfg),
                args.top_k
            ),
            eval_set, args.top_k, cfg
        ))

    # ── 6. Dense - OpenAI ────────────────────────────────────────────────────
    if available.get("emb_openai"):
        results.append(eval_config(
            conn, "Dense - OpenAI",
            lambda q: retrieve_dense(conn, embed_query_openai(q), "emb_openai", args.top_k, cfg),
            eval_set, args.top_k, cfg
        ))

    conn.close()

    write_markdown(results, eval_set, chunk_size, overlap, output)

    # Final summary table
    print(f"\n{'═' * 75}")
    print(f"{'FINAL RESULTS':^75}")
    print(f"{'═' * 75}")
    print(f"{'Config':<30} {'Hit@5':>6} {'MRR':>6} {'Specific':>9} {'Topic':>7} {'Metadata':>9} {'Summary':>8} {'ms':>6}")
    print(f"{'─' * 83}")
    for r in results:
        print(f"{r['config']:<30} "
              f"{r.get('hit@5', 0):>6.3f} "
              f"{r.get('mrr', 0):>6.3f} "
              f"{fmt(r.get('specific_hit@5')):>9} "
              f"{fmt(r.get('topic_hit@5')):>7} "
              f"{fmt(r.get('metadata_hit@5')):>9} "
              f"{fmt(r.get('summary_hit@5')):>8} "
              f"{r.get('latency_ms', 0):>5.0f}ms")


if __name__ == "__main__":
    main()
