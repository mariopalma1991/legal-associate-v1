"""Lightweight pipeline run/stage tracking helpers."""

import json
import uuid


def start_run(conn, triggered_by: str = "manual", notes: str = None) -> str:
    run_id = str(uuid.uuid4())
    with conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO pipeline_runs (id, triggered_by, started_at, status, notes)
            VALUES (%s, %s, now(), 'running', %s)
        """, (run_id, triggered_by, notes))
        cur.close()
    return run_id


def finish_run(conn, run_id: str, status: str = "completed"):
    with conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE pipeline_runs
            SET status = %s, finished_at = now()
            WHERE id = %s
        """, (status, run_id))
        cur.close()


def start_stage(conn, run_id: str, stage: str, config: dict = None) -> str:
    stage_id = str(uuid.uuid4())
    with conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO pipeline_stages
                (id, run_id, stage, started_at, status, config)
            VALUES (%s, %s, %s, now(), 'running', %s)
        """, (stage_id, run_id, stage, json.dumps(config or {})))
        cur.close()
    return stage_id


def finish_stage(conn, stage_id: str, status: str,
                 items_found: int = 0, items_ok: int = 0,
                 items_skipped: int = 0, items_error: int = 0,
                 error_summary: str = None):
    with conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE pipeline_stages SET
                status        = %s,
                finished_at   = now(),
                duration_sec  = EXTRACT(EPOCH FROM (now() - started_at))::int,
                items_found   = %s,
                items_ok      = %s,
                items_skipped = %s,
                items_error   = %s,
                error_summary = %s
            WHERE id = %s
        """, (status, items_found, items_ok, items_skipped, items_error,
              error_summary, stage_id))
        cur.close()
