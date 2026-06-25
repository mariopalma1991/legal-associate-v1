"""
Extracts structured data from Chihuahua licitacion pages.

Usage:
  Single URL:   python extract_licitacion.py --url https://contrataciones.chihuahua.gob.mx/licitaciones/271576/
  Single ID:    python extract_licitacion.py --id 271576
  Batch file:   python extract_licitacion.py --file valid_pages.txt --output licitaciones.csv
"""

import asyncio
import csv
import json
import argparse
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

import aiohttp
from bs4 import BeautifulSoup

BASE_URL = "https://contrataciones.chihuahua.gob.mx/licitaciones/{}/"
CONCURRENCY = 30
TIMEOUT_SECS = 20

FIELD_MAP = {
    "descripción ente público contratante": "ente_contratante",
    "descripción ente público solicitante": "ente_solicitante",
    "documento programado":                 "documento_programado",
    "materia":                              "materia",
    "tipo de contrato":                     "tipo_contrato",
    "concepto de contratación":             "concepto_contratacion",
    "descripción del procedimiento":        "descripcion_procedimiento",
    "fecha de publicación de la convocatoria": "fecha_publicacion_convocatoria",
    "fecha de junta de aclaraciones":       "fecha_junta_aclaraciones",
    "hora junta de aclaraciones":           "hora_junta_aclaraciones",
    "lugar de junta de aclaraciones":       "lugar_junta_aclaraciones",
    "fecha de apertura de propuestas":      "fecha_apertura_propuestas",
    "hora apertura de propuestas":          "hora_apertura_propuestas",
    "lugar de apertura de propuestas":      "lugar_apertura_propuestas",
    "costo de participación":               "costo_participacion",
    # less common fields seen on other pages
    "fundamento legal":                     "fundamento_legal",
    "modalidad":                            "modalidad",
    "número de procedimiento":              "numero_procedimiento_alt",
}

CSV_COLUMNS = [
    "licitacion_id",
    "url",
    "numero_procedimiento",
    "tipo_procedimiento",
    "estatus",
    "ente_contratante",
    "ente_solicitante",
    "documento_programado",
    "materia",
    "tipo_contrato",
    "concepto_contratacion",
    "descripcion_procedimiento",
    "fecha_publicacion_convocatoria",
    "fecha_junta_aclaraciones",
    "hora_junta_aclaraciones",
    "lugar_junta_aclaraciones",
    "fecha_apertura_propuestas",
    "hora_apertura_propuestas",
    "lugar_apertura_propuestas",
    "costo_participacion",
    "fundamento_legal",
    "modalidad",
    "documentos_json",
]


def parse_page(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    record: dict = {"url": url}

    # ID from URL
    m = re.search(r"/licitaciones/(\d+)/", url)
    record["licitacion_id"] = m.group(1) if m else ""

    # Procedure number — last breadcrumb item
    breadcrumb = soup.select("ol.breadcrumb li")
    record["numero_procedimiento"] = breadcrumb[-1].get_text(strip=True) if breadcrumb else ""

    # Procedure type — h2 in the title block (h3 is malformed-nested inside h2,
    # so take only the direct text nodes to avoid including the procedure number)
    h2 = soup.select_one("h2.bold.text-gris")
    if h2:
        from bs4 import NavigableString
        record["tipo_procedimiento"] = "".join(
            str(c).strip() for c in h2.children if isinstance(c, NavigableString)
        ).strip()
    else:
        record["tipo_procedimiento"] = ""

    # Status — find "Estatus del Procedimiento" h4 then take the next h4.
    # Container class varies by status (bg-cyan=Vigente, bg-grisFuerte2=Terminado, etc.)
    # so we match on text content instead of CSS class.
    record["estatus"] = ""
    for h4 in soup.find_all("h4"):
        if "Estatus del Procedimiento" in h4.get_text():
            sibling = h4.find_next_sibling("h4")
            if sibling:
                record["estatus"] = sibling.get_text(strip=True)
            break

    # General info — bold label / plain value pairs inside .container rows
    # Strategy: walk every div.row, collect all child divs in order,
    # pair consecutive bold-p (label) → next-p (value).
    for row in soup.select("div.row"):
        divs = row.find_all("div", recursive=False)
        i = 0
        while i < len(divs):
            bold = divs[i].find("p", class_="bold")
            if bold and i + 1 < len(divs):
                label = bold.get_text(strip=True).lower()
                value_p = divs[i + 1].find("p")
                value = value_p.get_text(strip=True) if value_p else ""
                field_name = FIELD_MAP.get(label)
                if field_name:
                    record[field_name] = value
                i += 2
            else:
                i += 1

    # Documents table
    docs = []
    for table in soup.select("table"):
        for tr in table.select("tbody tr"):
            tds = tr.find_all("td")
            if len(tds) >= 3:
                tipo = tds[0].get_text(strip=True)
                fecha = tds[1].get_text(strip=True)
                link_tag = tds[2].find("a")
                link = link_tag["href"] if link_tag and link_tag.get("href") else ""
                docs.append({"tipo": tipo, "fecha_actualizacion": fecha, "url": link})
    record["documentos_json"] = json.dumps(docs, ensure_ascii=False)

    # Fill missing fields with empty string
    for col in CSV_COLUMNS:
        record.setdefault(col, "")

    return record


async def fetch_and_parse(session: aiohttp.ClientSession, sem: asyncio.Semaphore, url: str) -> Optional[dict]:
    async with sem:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECS), allow_redirects=True) as resp:
                if resp.status != 200 or f"/licitaciones/" not in str(resp.url):
                    return None
                html = await resp.text(errors="replace")
                return parse_page(html, url)
        except Exception as e:
            print(f"  ERROR {url}: {e}", file=sys.stderr)
            return None


async def process_batch(urls: list[str], output_csv: str):
    sem = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ssl=False)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; licitacion-extractor/1.0)"}

    total = len(urls)
    done = 0
    written = 0

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            tasks = [fetch_and_parse(session, sem, url) for url in urls]
            for coro in asyncio.as_completed(tasks):
                result = await coro
                done += 1
                if result:
                    writer.writerow(result)
                    written += 1
                if done % 100 == 0:
                    print(f"  {done}/{total} processed, {written} records written")

    print(f"\nDone — {written} records saved to {output_csv}")


def process_single(url: str):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req).read().decode("utf-8", errors="replace")
    record = parse_page(html, url)
    # Pretty-print without documentos_json clutter
    docs = json.loads(record.pop("documentos_json", "[]"))
    for k, v in record.items():
        if v:
            print(f"  {k:40s}: {v}")
    if docs:
        print(f"\n  {'documentos':40s}:")
        for d in docs:
            print(f"    [{d['tipo']}] {d['fecha_actualizacion']}  {d['url']}")


def main():
    parser = argparse.ArgumentParser(description="Extract data from Chihuahua licitacion pages")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url",  help="Single licitacion URL")
    group.add_argument("--id",   help="Single licitacion ID (number)")
    group.add_argument("--file", help="Text file with one URL per line (e.g. valid_pages.txt)")
    parser.add_argument("--output", default="licitaciones.csv", help="Output CSV file (batch mode)")
    args = parser.parse_args()

    if args.url:
        process_single(args.url)
    elif args.id:
        process_single(BASE_URL.format(args.id))
    else:
        with open(args.file) as f:
            urls = [line.strip() for line in f if line.strip()]
        print(f"Processing {len(urls):,} URLs → {args.output}")
        asyncio.run(process_batch(urls, args.output))


if __name__ == "__main__":
    main()
