"""
Fetches Vigente licitaciones via the portal export (XLS) for a given date range.

The date range filters on 'Fecha de publicación de la convocatoria'.
The XLS export (/exportar/) returns 42 columns with full metadata — dates,
hours, locations, costs, contract details — in a single request.

Usage:
    python fetch_vigentes.py --start-date 01/01/2026 --end-date 15/06/2026
    python fetch_vigentes.py                        # watermark → today (auto)
    python fetch_vigentes.py --dry-run              # print records, no DB write
"""

import argparse
import os
import re
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

import psycopg2
import requests
import xlrd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from _pipeline import start_run, finish_run, start_stage, finish_stage

load_dotenv()

BASE_URL = "https://contrataciones.chihuahua.gob.mx"
TIMEOUT  = 600   # government site is very slow


# ── Helpers ───────────────────────────────────────────────────────────────────

def _xls_date(val) -> str:
    """Convert Excel serial date float → 'DD/MM/YYYY', or '' if empty/N/A."""
    if not val or val == "No aplica":
        return ""
    if isinstance(val, float) and val > 0:
        return xlrd.xldate_as_datetime(val, 0).strftime("%d/%m/%Y")
    return str(val).strip()


def _xls_str(val) -> str:
    """Clean an XLS cell value to a plain string."""
    if val is None or val == "No aplica":
        return ""
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    return str(val).strip()


# ── Portal session ────────────────────────────────────────────────────────────

def get_csrf_and_session():
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (compatible; licitacion-fetcher/1.0)"
    resp = s.get(BASE_URL + "/", timeout=TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    csrf_input = soup.find("input", {"name": "csrfmiddlewaretoken"})
    if not csrf_input:
        raise RuntimeError("CSRF token not found on main page")
    return s, csrf_input["value"]


# ── XLS export ────────────────────────────────────────────────────────────────

def fetch_export(session, csrf, start_date: str, end_date: str) -> list[dict]:
    """
    POST to /exportar/ and return one dict per licitacion with all available fields.
    Dates are converted from Excel serial numbers to DD/MM/YYYY strings.
    """
    form_data = {
        "csrfmiddlewaretoken": csrf,
        "Estatus":               "0",    # Vigente
        "Tipo_de_Licitaci_n":    "-1",
        "TipoProc":              "-1",
        "Unidades_Responsables": "",
        "num_pricedimineto":     "",
        "num_contrato":          "",
        "rdFechas":              "2",    # fecha de publicación de la convocatoria
        "fechainicio":           start_date,
        "fechafin":              end_date,
        "nom_proveedor":         "",
        "concepto_contratacion": "",
        "desc_procedimiento":    "",
        "proyecto_esp":          "",
    }
    headers = {"Referer": BASE_URL + "/busqueda/", "X-CSRFToken": csrf}
    resp = session.post(BASE_URL + "/exportar/", data=form_data,
                        headers=headers, timeout=TIMEOUT)
    resp.raise_for_status()

    wb = xlrd.open_workbook(file_contents=resp.content)
    ws = wb.sheet_by_index(0)

    results = []
    for row_idx in range(1, ws.nrows):       # skip header row
        row = ws.row_values(row_idx)

        url = _xls_str(row[38])
        m = re.search(r"/licitaciones/(\d+)/", url)
        if not m:
            continue

        results.append({
            "id":                     int(m.group(1)),
            "url":                    url,
            # Procedure identity
            "numero_procedimiento":   _xls_str(row[0]),
            "ente_contratante":       _xls_str(row[1]),
            "ente_solicitante":       _xls_str(row[2]),
            "documento_programado":   _xls_str(row[3]),
            "tipo_procedimiento":     _xls_str(row[4]),
            "materia":                _xls_str(row[5]),
            "tipo_contrato":          _xls_str(row[6]),
            "concepto_contratacion":  _xls_str(row[7]),
            "descripcion":            _xls_str(row[41]),
            # Dates + locations
            "fecha_convocatoria":         _xls_date(row[12]),
            "fecha_junta_aclaraciones":   _xls_date(row[13]),
            "hora_junta_aclaraciones":    _xls_str(row[14]),
            "lugar_junta_aclaraciones":   _xls_str(row[15]),
            "fecha_apertura":             _xls_date(row[16]),
            "hora_apertura":              _xls_str(row[17]),
            "lugar_apertura":             _xls_str(row[18]),
            "costo_participacion":        _xls_str(row[22]),
            # Live status
            "licitacion_status":          _xls_str(row[26]),
            # Contract info (may be empty for open procedures)
            "nombre_proveedor":           _xls_str(row[34]),
            "razon_social":               _xls_str(row[37]),
            "monto_contrato":             _xls_str(row[32]),
            "fecha_firma_contrato":       _xls_date(row[29]) if isinstance(row[29], float) else _xls_str(row[29]),
        })

    return results


# ── DB ────────────────────────────────────────────────────────────────────────

def db_connect():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set in .env")
    at = url.rfind("@")
    creds, host_part = url[len("postgresql://"):at], url[at + 1:]
    user, password = creds.split(":", 1)
    host, port = host_part.split("/")[0].split(":")
    dbname = host_part.split("/")[1].split("?")[0]
    return psycopg2.connect(
        host=host, port=int(port), dbname=dbname,
        user=user, password=password, sslmode="require"
    )


def get_watermark(conn) -> str | None:
    cur = conn.cursor()
    cur.execute("SELECT value FROM config WHERE key = 'fetch_watermark'")
    row = cur.fetchone()
    cur.close()
    return row[0] if row else None


def set_watermark(conn, date_str: str):
    with conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO config (key, value) VALUES ('fetch_watermark', %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """, (date_str,))
        cur.close()


def upsert_records(conn, records: list[dict]) -> tuple[int, int]:
    """
    Upsert all XLS fields into licitaciones.
    New rows → pipeline_status = 'discovered'.
    Existing rows → metadata updated, pipeline_status preserved.
    Returns (inserted, updated).
    """
    inserted = updated = 0
    with conn:
        cur = conn.cursor()
        for r in records:
            cur.execute("""
                INSERT INTO licitaciones (
                    id, url, pipeline_status,
                    numero_procedimiento, tipo_procedimiento, licitacion_status,
                    ente_contratante, ente_solicitante, documento_programado,
                    materia, tipo_contrato, concepto_contratacion, descripcion,
                    fecha_convocatoria,
                    fecha_junta_aclaraciones, hora_junta_aclaraciones, lugar_junta_aclaraciones,
                    fecha_apertura, hora_apertura, lugar_apertura,
                    costo_participacion,
                    nombre_proveedor, razon_social, monto_contrato, fecha_firma_contrato
                ) VALUES (
                    %(id)s, %(url)s, 'discovered',
                    %(numero_procedimiento)s, %(tipo_procedimiento)s, %(licitacion_status)s,
                    %(ente_contratante)s, %(ente_solicitante)s, %(documento_programado)s,
                    %(materia)s, %(tipo_contrato)s, %(concepto_contratacion)s, %(descripcion)s,
                    %(fecha_convocatoria)s,
                    %(fecha_junta_aclaraciones)s, %(hora_junta_aclaraciones)s, %(lugar_junta_aclaraciones)s,
                    %(fecha_apertura)s, %(hora_apertura)s, %(lugar_apertura)s,
                    %(costo_participacion)s,
                    %(nombre_proveedor)s, %(razon_social)s, %(monto_contrato)s, %(fecha_firma_contrato)s
                )
                ON CONFLICT (id) DO UPDATE SET
                    numero_procedimiento     = EXCLUDED.numero_procedimiento,
                    tipo_procedimiento       = EXCLUDED.tipo_procedimiento,
                    licitacion_status        = EXCLUDED.licitacion_status,
                    ente_contratante         = EXCLUDED.ente_contratante,
                    ente_solicitante         = EXCLUDED.ente_solicitante,
                    documento_programado     = EXCLUDED.documento_programado,
                    materia                  = EXCLUDED.materia,
                    tipo_contrato            = EXCLUDED.tipo_contrato,
                    concepto_contratacion    = EXCLUDED.concepto_contratacion,
                    descripcion              = EXCLUDED.descripcion,
                    fecha_convocatoria       = EXCLUDED.fecha_convocatoria,
                    fecha_junta_aclaraciones = EXCLUDED.fecha_junta_aclaraciones,
                    hora_junta_aclaraciones  = EXCLUDED.hora_junta_aclaraciones,
                    lugar_junta_aclaraciones = EXCLUDED.lugar_junta_aclaraciones,
                    fecha_apertura           = EXCLUDED.fecha_apertura,
                    hora_apertura            = EXCLUDED.hora_apertura,
                    lugar_apertura           = EXCLUDED.lugar_apertura,
                    costo_participacion      = EXCLUDED.costo_participacion,
                    nombre_proveedor         = EXCLUDED.nombre_proveedor,
                    razon_social             = EXCLUDED.razon_social,
                    monto_contrato           = EXCLUDED.monto_contrato,
                    fecha_firma_contrato     = EXCLUDED.fecha_firma_contrato,
                    last_checked_at          = now()
            """, r)

            # xmax=0 → freshly inserted; xmax>0 → updated existing row
            cur.execute("SELECT xmax::text::int FROM licitaciones WHERE id = %s", (r["id"],))
            xmax = cur.fetchone()[0]
            if xmax == 0:
                inserted += 1
            else:
                updated += 1

        cur.close()
    return inserted, updated


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch Vigente licitaciones via portal XLS export"
    )
    parser.add_argument("--start-date", help="DD/MM/YYYY (default: DB watermark or yesterday)")
    parser.add_argument("--end-date",   help="DD/MM/YYYY (default: 6 months from today)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print records without writing to DB")
    return parser.parse_args()


def main():
    args = parse_args()

    today         = date.today()
    yesterday_str = (today - timedelta(days=1)).strftime("%d/%m/%Y")
    six_months_str = (today + relativedelta(months=6)).strftime("%d/%m/%Y")

    end_date = args.end_date or six_months_str

    if args.start_date:
        start_date = args.start_date
    else:
        conn = db_connect()
        watermark = get_watermark(conn)
        conn.close()
        if watermark:
            start_date = watermark
            print(f"  Auto start from DB watermark: {start_date}")
        else:
            start_date = yesterday_str
            print(f"  No watermark — defaulting to yesterday: {start_date}")

    print(f"Fetching Vigente licitaciones: {start_date} → {end_date}")
    print("  (date range = fecha de publicación de la convocatoria)")
    print()

    print("  Getting CSRF token...")
    session, csrf = get_csrf_and_session()

    print("  Downloading XLS export from /exportar/ ...")
    records = fetch_export(session, csrf, start_date, end_date)
    print(f"  Found {len(records)} records\n")

    if not records:
        print("  Nothing to do.")
        if not args.dry_run:
            conn = db_connect()
            set_watermark(conn, yesterday_str)
            conn.close()
            print(f"  Watermark updated to {yesterday_str}")
        return

    # Print what we extracted
    print(f"  {'ID':<8}  {'Procedimiento':<30}  {'Ente contratante':<35}  {'Convocatoria':<14}  {'Apertura':<14}  {'Costo'}")
    print(f"  {'-'*8}  {'-'*30}  {'-'*35}  {'-'*14}  {'-'*14}  {'-'*10}")
    for r in records:
        print(
            f"  {r['id']:<8}  {r['numero_procedimiento'][:30]:<30}  "
            f"{r['ente_contratante'][:35]:<35}  "
            f"{r['fecha_convocatoria'] or 'N/A':<14}  "
            f"{r['fecha_apertura'] or 'N/A':<14}  "
            f"{r['costo_participacion'] or '0'}"
        )

    if args.dry_run:
        print("\nDry run — no DB writes.")
        return

    print("\n  Upserting into DB...")
    conn = db_connect()
    run_id   = start_run(conn, notes="discover")
    stage_id = start_stage(conn, run_id, "discover", config={
        "start_date": start_date, "end_date": end_date,
    })
    inserted, updated = upsert_records(conn, records)
    set_watermark(conn, yesterday_str)
    finish_stage(conn, stage_id, "completed",
                 items_found=len(records), items_ok=inserted + updated,
                 items_skipped=0)
    finish_run(conn, run_id, "completed")
    conn.close()
    print(f"  Done — {inserted} new, {updated} updated.")
    print(f"  Watermark updated to {yesterday_str}")


if __name__ == "__main__":
    main()
