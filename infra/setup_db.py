"""
Creates the PostgreSQL + pgvector schema for the Chihuahua licitaciones RAG system.

Usage:
  Normal create:  python setup_db.py
  Reset (drop all tables and recreate):  python setup_db.py --reset

Connection (in order of priority):
  1. DATABASE_URL in .env file  (postgresql://user:pass@host:port/db)
  2. Individual env vars: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
  3. Defaults: localhost:5432/licitaciones postgres/postgres
"""

import argparse
import os
import sys
import psycopg2
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────
# SQL: DROP
# ──────────────────────────────────────────────
DROP_SQL = """
DROP TABLE IF EXISTS pipeline_stages CASCADE;
DROP TABLE IF EXISTS pipeline_runs   CASCADE;
DROP TABLE IF EXISTS chunks          CASCADE;
DROP TABLE IF EXISTS documents       CASCADE;
DROP TABLE IF EXISTS licitaciones    CASCADE;
DROP TABLE IF EXISTS config          CASCADE;
"""

# ──────────────────────────────────────────────
# SQL: CREATE
# ──────────────────────────────────────────────
CREATE_SQL = """
-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Config ──────────────────────────────────────────────────────────────────
-- Key/value store for pipeline state (watermarks, timestamps, etc.)
CREATE TABLE config (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- ── Licitaciones ─────────────────────────────────────────────────────────────
CREATE TABLE licitaciones (
    id                       INTEGER PRIMARY KEY,
    url                      TEXT    NOT NULL,

    -- Procedure identity
    numero_procedimiento     TEXT,
    tipo_procedimiento       TEXT,

    -- Live status from the page (refreshed nightly for Vigente records)
    licitacion_status        TEXT,    -- Vigente | Terminado | Cancelado | En seguimiento

    -- General info
    ente_contratante         TEXT,
    ente_solicitante         TEXT,
    documento_programado     TEXT,
    materia                  TEXT,
    tipo_contrato            TEXT,
    concepto_contratacion    TEXT,
    descripcion              TEXT,
    fundamento_legal         TEXT,
    modalidad                TEXT,

    -- Key dates
    fecha_convocatoria       TEXT,
    fecha_junta_aclaraciones TEXT,
    hora_junta_aclaraciones  TEXT,
    lugar_junta_aclaraciones TEXT,
    fecha_apertura           TEXT,
    hora_apertura            TEXT,
    lugar_apertura           TEXT,
    costo_participacion      TEXT,

    -- Contract details (populated if available)
    nombre_proveedor         TEXT,
    razon_social             TEXT,
    monto_contrato           TEXT,
    fecha_firma_contrato     TEXT,

    -- Pipeline tracking
    pipeline_status          TEXT        DEFAULT 'scraped',
                             -- scraped → downloaded → chunked → indexed
    discovered_at            TIMESTAMPTZ DEFAULT now(),
    last_checked_at          TIMESTAMPTZ,   -- last time status was refreshed
    indexed_at               TIMESTAMPTZ
);

-- ── Documents ────────────────────────────────────────────────────────────────
-- One row per PDF linked from a licitacion page
CREATE TABLE documents (
    id             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    licitacion_id  INTEGER     NOT NULL REFERENCES licitaciones(id) ON DELETE CASCADE,
    tipo           TEXT,                   -- Convocatoria | Bases | Bases - Anexo 1 | ...
    url            TEXT        NOT NULL,
    local_path     TEXT,                   -- docs/{licitacion_id}/{tipo}.pdf
    file_size      INTEGER,
    status         TEXT        DEFAULT 'pending',
                   -- pending → downloaded → chunked → indexed
    error          TEXT,                   -- last error message if failed
    created_at     TIMESTAMPTZ DEFAULT now(),
    updated_at     TIMESTAMPTZ DEFAULT now()
);

-- ── Chunks ───────────────────────────────────────────────────────────────────
-- Text chunks extracted from PDFs, with embeddings per model
CREATE TABLE chunks (
    id             UUID     PRIMARY KEY DEFAULT gen_random_uuid(),
    licitacion_id  INTEGER  NOT NULL REFERENCES licitaciones(id) ON DELETE CASCADE,
    document_id    UUID     NOT NULL REFERENCES documents(id)    ON DELETE CASCADE,
    doc_type       TEXT,
    chunk_index    INTEGER,
    page_number    INTEGER,
    text           TEXT     NOT NULL,
    token_count    INTEGER,

    -- Full-text search vector (auto-updated from text)
    fts            TSVECTOR GENERATED ALWAYS AS (
                       to_tsvector('spanish', coalesce(text, ''))
                   ) STORED,

    -- Embedding columns — one per model
    emb_openai     vector(1536),   -- text-embedding-3-small
    emb_cohere     vector(1024),   -- embed-multilingual-v3.0
    emb_me5        vector(1024),   -- multilingual-e5-large

    created_at     TIMESTAMPTZ DEFAULT now()
);

-- ── Indexes ──────────────────────────────────────────────────────────────────

-- Licitaciones
CREATE INDEX idx_lic_status      ON licitaciones (licitacion_status);
CREATE INDEX idx_lic_pipeline    ON licitaciones (pipeline_status);
CREATE INDEX idx_lic_fecha       ON licitaciones (fecha_convocatoria);

-- Documents
CREATE INDEX idx_doc_licitacion  ON documents (licitacion_id);
CREATE INDEX idx_doc_status      ON documents (status);

-- Chunks — relational
CREATE INDEX idx_chunk_licitacion ON chunks (licitacion_id);
CREATE INDEX idx_chunk_document   ON chunks (document_id);

-- Chunks — full-text search
CREATE INDEX idx_chunk_fts        ON chunks USING GIN (fts);

-- Chunks — vector (HNSW per model, cosine distance)
CREATE INDEX idx_chunk_emb_openai ON chunks USING hnsw (emb_openai vector_cosine_ops);
CREATE INDEX idx_chunk_emb_cohere ON chunks USING hnsw (emb_cohere vector_cosine_ops);
CREATE INDEX idx_chunk_emb_me5    ON chunks USING hnsw (emb_me5    vector_cosine_ops);

-- ── Pipeline tracking ────────────────────────────────────────────────────────
CREATE TABLE pipeline_runs (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    triggered_by  TEXT        DEFAULT 'manual',   -- 'manual' | 'scheduled'
    started_at    TIMESTAMPTZ DEFAULT now(),
    finished_at   TIMESTAMPTZ,
    status        TEXT        DEFAULT 'running',  -- 'running' | 'completed' | 'partial' | 'failed'
    notes         TEXT
);

CREATE TABLE pipeline_stages (
    id             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id         UUID        REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    stage          TEXT        NOT NULL,  -- 'discover'|'ingest'|'download'|'chunk'|'embed'
    started_at     TIMESTAMPTZ DEFAULT now(),
    finished_at    TIMESTAMPTZ,
    duration_sec   INTEGER,
    status         TEXT        DEFAULT 'running',  -- 'running'|'completed'|'failed'
    items_found    INTEGER     DEFAULT 0,
    items_ok       INTEGER     DEFAULT 0,
    items_skipped  INTEGER     DEFAULT 0,
    items_error    INTEGER     DEFAULT 0,
    config         JSONB,
    error_summary  TEXT
);

CREATE INDEX idx_pipeline_runs_started  ON pipeline_runs  (started_at DESC);
CREATE INDEX idx_pipeline_stages_run    ON pipeline_stages (run_id);
CREATE INDEX idx_pipeline_stages_stage  ON pipeline_stages (stage, started_at DESC);

-- ── Seed config ──────────────────────────────────────────────────────────────
INSERT INTO config (key, value) VALUES
    ('last_status_refresh',  NULL);
"""


def parse_database_url(url: str) -> dict:
    """Parse DATABASE_URL handling @ signs inside passwords."""
    url = url.replace("postgresql://", "").replace("postgres://", "")
    at_idx = url.rfind("@")
    creds, host_part = url[:at_idx], url[at_idx + 1:]
    user, password = creds.split(":", 1)
    host_db = host_part.split("/")
    host, port = host_db[0].split(":")
    dbname = host_db[1].split("?")[0]
    return dict(host=host, port=int(port), dbname=dbname, user=user, password=password)


def get_conn(host, port, dbname, user, password):
    return psycopg2.connect(
        host=host, port=port, dbname=dbname,
        user=user, password=password,
        sslmode="require"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Set up the licitaciones PostgreSQL schema"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop all existing tables before recreating (destructive)",
    )
    args = parser.parse_args()

    # Connection: DATABASE_URL takes priority, then individual env vars, then defaults
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        params = parse_database_url(database_url)
    else:
        params = dict(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            dbname=os.getenv("DB_NAME", "licitaciones"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
        )

    print(f"Connecting to {params['user']}@{params['host']}:{params['port']}/{params['dbname']} ...")
    try:
        conn = get_conn(**params)
    except psycopg2.OperationalError as e:
        print(f"\nERROR: Could not connect to PostgreSQL.\n{e}")
        sys.exit(1)

    conn.autocommit = True
    cur = conn.cursor()

    if args.reset:
        confirm = input(
            "\nWARNING: --reset will delete ALL data. Type 'yes' to confirm: "
        )
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(0)
        print("Dropping tables ...")
        cur.execute(DROP_SQL)
        print("  Done.")

    print("Creating schema ...")
    try:
        cur.execute(CREATE_SQL)
    except psycopg2.errors.DuplicateTable as e:
        print(f"\nERROR: Tables already exist.\nRun with --reset to drop and recreate.\n{e}")
        sys.exit(1)

    # Verify
    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    tables = [r[0] for r in cur.fetchall()]

    cur.execute("SELECT key, value FROM config ORDER BY key;")
    config_rows = cur.fetchall()

    conn.close()

    print("\n  Tables created:")
    for t in tables:
        print(f"    [ok] {t}")

    print("\n  Config seeded:")
    for key, value in config_rows:
        print(f"    {key} = {value}")

    print("\nDatabase ready.\n")


if __name__ == "__main__":
    main()
