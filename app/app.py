"""
app.py — Gradio chat interface for the Chihuahua licitaciones RAG assistant.

Conversations are persisted to the PostgreSQL DB (conversations table).
Each conversation stores its full message history and RAG session state so
the user can resume from exactly where they left off.

Usage:
  python app.py
  python app.py --synthesis-model gpt-4o --share
"""

import argparse
import json
import uuid
import re
from datetime import date, datetime

import gradio as gr
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

from query import (
    build_context,
    get_conn,
    get_licitacion_by_id,
    get_licitacion_by_numero,
    retrieve_chunks,
    search_licitaciones,
    synthesize,
    synthesize_stream,
    _parse_date,
    _days_remaining,
    _urgency_label,
)

DEFAULT_EMB_MODEL    = "cohere"
DEFAULT_SYNTH_MODEL  = "claude-sonnet-4-6"
DEFAULT_TOP_K        = 30
DEFAULT_CHUNK_CONFIG = "1024_256"

# The only doc types that are chunked, indexed, and shown to users.
INDEXED_DOC_TYPES = ("Bases", "Convocatoria", "Acta de la junta de aclaraciones")


# ── UI translations ───────────────────────────────────────────────────────────

_T = {
    "es": {
        "lang_toggle":       "🇺🇸 English",
        "btn_new_convo":     "＋ Nueva conversación",
        "history_label":     "---\n**Historial**",
        "history_info":      "Selecciona una conversación anterior para continuar.",
        "chat_title":        "### Asistente de Licitaciones Públicas\n_Solo muestra licitaciones cuyo plazo de participación no ha vencido._",
        "placeholder":       "Ej: ¿qué licitaciones de pavimentación hay vigentes?",
        "send_btn":          "Enviar",
        "btn_vigentes":      "📋 Ver Licitaciones Vigentes",
        "btn_bases":         "📄 Bases",
        "btn_convocatoria":  "📢 Convocatoria",
        "btn_junta":         "🗣️ Junta de Aclaraciones",
        "btn_req":           "📋 Ver requisitos",
        "btn_delete_convo":  "🗑️ Eliminar esta conversación",
        "right_panel_title": "### 📋 Detalle",
        "right_panel_empty": "_Selecciona una licitación para ver sus detalles aquí._",
        "selector_label":    lambda n: f"{n} licitaciones vigentes — selecciona una",
        # quick-action trigger messages
        "msg_vigentes":      "dame las licitaciones vigentes",
        "msg_bases":         "dame un resumen de las Bases de esta licitación",
        "msg_convocatoria":  "dame un resumen de la Convocatoria de esta licitación",
        "msg_junta":         "qué se dijo en la Junta de Aclaraciones",
        "msg_req":           "cuáles son los requisitos para participar",
        "btn_specs":         "📦 Especificaciones técnicas",
        "btn_docs":          "📑 Documentos requeridos",
        "btn_economic":      "💰 Condiciones económicas",
        "msg_specs":         "¿cuáles son las especificaciones técnicas, cantidades y requisitos del anexo técnico?",
        "msg_docs":          "¿qué documentos debo presentar para participar?",
        "msg_economic":      "¿cuál es el anticipo, la garantía de cumplimiento y las condiciones de pago?",
        # discovery
        "found_n":           lambda n: f"Encontré **{n} licitación{'es' if n > 1 else ''}** vigente{'s' if n > 1 else ''} con tiempo para participar:",
        "showing_first_10":  lambda n: f"_Mostrando las primeras 10 de {n}. Usa el selector debajo del chat para ver todas._",
        "discovery_hint":    "¿Quieres más detalles de alguna? Indícame el número o pregúntame, por ejemplo: _'¿cuáles son los requisitos del [1]?'_",
        "no_results_sectors":lambda s: f"No encontré licitaciones vigentes que coincidan con tu búsqueda. Los sectores disponibles actualmente son:\n\n{s}\n\n¿Te interesa alguno de estos sectores?",
        "no_results_generic":"No encontré licitaciones vigentes con tiempo para participar en esa búsqueda. ¿Puedes darme más detalles sobre el tipo de producto, servicio u obra que buscas?",
        # clarify
        "clarify_no_ctx":    "¿Sobre qué tipo de licitación tienes la pregunta? Puedo buscar las vigentes disponibles si me dices el sector, tipo de obra o servicio.",
        "clarify_which":     lambda n, o: f"Tengo {n} licitaciones activas ({o}). ¿A cuál te refieres?",
        "clarify_anchor":    lambda o: f"¿A cuál licitación te refieres? ({o})",
        "focused_on":        lambda name: f"📌 **Enfocado en:** {name}\n\n",
        # format_discovery_list
        "no_desc":           "Sin descripción",
        "code":              "Código",
        "contracting_ent":   "Ente contratante",
        "pub_date":          "Fecha de publicación de la convocatoria",
        "clarif_meeting":    "Junta de aclaraciones",
        "already_occurred":  "ya ocurrió",
        "days":              "días",
        "deadline":          "Fecha límite para presentar propuesta",
        "urgent":            "🔴 URGENTE",
        "upcoming":          "🟡 PRÓXIMO",
        "on_track":          "🟢 CON TIEMPO",
        "location":          "Lugar de presentación de propuestas",
        "cost":              "Costo de participación",
        "view_portal":       "Ver licitación en portal",
        "docs_consultable":  "Documentos consultables",
        "summary_lbl":       "Resumen",
        # anchor card
        "procedure_type":    "Tipo de procedimiento",
        "requesting_ent":    "Ente solicitante",
        "subject":           "Materia",
        "contract_type":     "Tipo de contrato",
        "pub_date_card":     "Publicación de convocatoria",
        "at_time":           "a las",
        "deadline_card":     "Fecha límite propuesta",
        "location_card":     "Lugar de presentación",
        "cost_card":         "Costo de participación",
        "view_portal_card":  "Ver licitación en portal",
        "docs_available":    "Documentos disponibles",
        "docs_hint":         "_Puedes preguntarme sobre el contenido de cualquiera de estos documentos._",
        # batch summarize prompt
        "summarize_prompt":  (
            "Lee cada bloque de documentos licitatorios (identificados por ID:). "
            "Para cada ID, escribe un resumen de 2 oraciones sobre lo que pide la licitación, "
            "incluyendo el objeto principal y los aspectos más relevantes. "
            "Responde ÚNICAMENTE con un objeto JSON: {\"<id>\": \"<resumen>\", ...}. "
            "Ejemplo: {\"12345\": \"El municipio solicita...\"}.\n\n"
        ),
    },
    "en": {
        "lang_toggle":       "🇲🇽 Español",
        "btn_new_convo":     "＋ New conversation",
        "history_label":     "---\n**History**",
        "history_info":      "Select a previous conversation to continue.",
        "chat_title":        "### Public Procurement Assistant\n_Only shows tenders whose participation deadline has not passed._",
        "placeholder":       "E.g.: what paving tenders are currently open?",
        "send_btn":          "Send",
        "btn_vigentes":      "📋 View Active Tenders",
        "btn_bases":         "📄 Bidding Docs",
        "btn_convocatoria":  "📢 Announcement",
        "btn_junta":         "🗣️ Clarification Meeting",
        "btn_req":           "📋 View requirements",
        "btn_delete_convo":  "🗑️ Delete this conversation",
        "right_panel_title": "### 📋 Detail",
        "right_panel_empty": "_Select a tender to see its details here._",
        "selector_label":    lambda n: f"{n} active tenders — select one",
        # quick-action trigger messages
        "msg_vigentes":      "show me the active tenders",
        "msg_bases":         "give me a summary of the Bases for this tender",
        "msg_convocatoria":  "give me a summary of the Convocatoria for this tender",
        "msg_junta":         "what was discussed in the Clarification Meeting",
        "msg_req":           "what are the requirements to participate",
        "btn_specs":         "📦 Technical specs",
        "btn_docs":          "📑 Required documents",
        "btn_economic":      "💰 Economic conditions",
        "msg_specs":         "what are the technical specifications, quantities and requirements in the technical annex?",
        "msg_docs":          "what documents do I need to submit to participate?",
        "msg_economic":      "what are the advance payment percentage, performance bond and payment conditions?",
        # discovery
        "found_n":           lambda n: f"Found **{n} active tender{'s' if n > 1 else ''}** open for participation:",
        "showing_first_10":  lambda n: f"_Showing the first 10 of {n}. Use the selector below the chat to see all._",
        "discovery_hint":    "Want more details on any of them? Tell me the number or ask, e.g.: _'What are the requirements for [1]?'_",
        "no_results_sectors":lambda s: f"No active tenders matched your search. Currently available sectors are:\n\n{s}\n\nAre you interested in any of these?",
        "no_results_generic":"No active tenders with remaining time were found for that search. Can you give me more details about the type of product, service, or work you're looking for?",
        # clarify
        "clarify_no_ctx":    "What type of tender are you asking about? I can search the active ones if you tell me the sector, type of work, or service.",
        "clarify_which":     lambda n, o: f"I have {n} active tenders ({o}). Which one are you referring to?",
        "clarify_anchor":    lambda o: f"Which tender are you referring to? ({o})",
        "focused_on":        lambda name: f"📌 **Focused on:** {name}\n\n",
        # format_discovery_list
        "no_desc":           "No description",
        "code":              "Code",
        "contracting_ent":   "Contracting entity",
        "pub_date":          "Publication date",
        "clarif_meeting":    "Clarification meeting",
        "already_occurred":  "already occurred",
        "days":              "days",
        "deadline":          "Proposal deadline",
        "urgent":            "🔴 URGENT",
        "upcoming":          "🟡 UPCOMING",
        "on_track":          "🟢 ON TRACK",
        "location":          "Submission location",
        "cost":              "Participation cost",
        "view_portal":       "View tender on portal",
        "docs_consultable":  "Available documents",
        "summary_lbl":       "Summary",
        # anchor card
        "procedure_type":    "Procedure type",
        "requesting_ent":    "Requesting entity",
        "subject":           "Subject",
        "contract_type":     "Contract type",
        "pub_date_card":     "Publication date",
        "at_time":           "at",
        "deadline_card":     "Proposal deadline",
        "location_card":     "Submission location",
        "cost_card":         "Participation cost",
        "view_portal_card":  "View tender on portal",
        "docs_available":    "Available documents",
        "docs_hint":         "_You can ask me about the content of any of these documents._",
        # batch summarize prompt
        "summarize_prompt":  (
            "Read each procurement document block (identified by ID:). "
            "For each ID, write a 2-sentence summary of what the tender requests, "
            "including the main object and most relevant aspects. "
            "Reply ONLY with a JSON object: {\"<id>\": \"<summary>\", ...}. "
            "Example: {\"12345\": \"The municipality requests...\"}.\n\n"
        ),
    },
}


# ── Conversation DB helpers ───────────────────────────────────────────────────

def _db_load_all(conn) -> list:
    """Return all conversations ordered by most recently updated."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT id, title, created_at, updated_at, messages, chat_state
        FROM conversations
        ORDER BY updated_at DESC
    """)
    rows = cur.fetchall()
    cur.close()
    return [dict(r) for r in rows]


def _db_load_one(conn, convo_id: str) -> dict | None:
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM conversations WHERE id = %s", (convo_id,))
    row = cur.fetchone()
    cur.close()
    return dict(row) if row else None


def _db_upsert(conn, convo_id: str, title: str, messages: list, chat_state: dict):
    with conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO conversations (id, title, messages, chat_state, updated_at)
            VALUES (%s, %s, %s::jsonb, %s::jsonb, now())
            ON CONFLICT (id) DO UPDATE
                SET title      = EXCLUDED.title,
                    messages   = EXCLUDED.messages,
                    chat_state = EXCLUDED.chat_state,
                    updated_at = now()
        """, (
            convo_id,
            title,
            json.dumps(messages, ensure_ascii=False, default=str),
            json.dumps(chat_state, ensure_ascii=False, default=str),
        ))


def _sidebar_choices(convos: list) -> list:
    """(label, id) tuples with recency tag and last-message preview, newest first."""
    from datetime import timedelta
    today     = date.today()
    yesterday = today - timedelta(days=1)
    week_ago  = today - timedelta(days=7)

    choices = []
    for c in convos:
        upd = c.get("updated_at")
        upd_date = upd.date() if upd and hasattr(upd, "date") else today

        if upd_date >= today:
            tag = "hoy"
        elif upd_date >= yesterday:
            tag = "ayer"
        elif upd_date >= week_ago:
            tag = upd_date.strftime("%a").lower()
        else:
            tag = upd_date.strftime("%d/%m")

        title   = c.get("title") or "Sin título"
        choices.append((f"[{tag}]  {title}", c["id"]))
    return choices


def _format_anchor_banner(chat_state: dict, lang: str = "es") -> str:
    """One-line banner shown above the chat when a licitación is anchored."""
    anchor_id = chat_state.get("anchor_id")
    if not anchor_id:
        return ""
    lic = next((l for l in chat_state.get("active_licitaciones", [])
                if l.get("id") == anchor_id), None)
    if not lic:
        return ""
    t    = _T[lang]
    desc = (lic.get("descripcion") or lic.get("concepto_contratacion") or f"#{anchor_id}")[:80]
    ap_str  = lic.get("fecha_apertura") or ""
    ap_date = _parse_date(ap_str)
    if ap_date:
        days   = (ap_date - date.today()).days
        badge  = t["urgent"] if days <= 5 else (t["upcoming"] if days <= 14 else t["on_track"])
        timing = f"⏰ {ap_str} · **{days}d** · {badge}"
    else:
        timing = f"⏰ {ap_str}" if ap_str else ""
    line = f"📌 **{desc}**"
    if timing:
        line += f"&nbsp;&nbsp;|&nbsp;&nbsp;{timing}"
    return line


def _format_right_panel(chat_state: dict, indexed_types: list, lang: str = "es") -> str:
    """Markdown content for the right info panel."""
    anchor_id = chat_state.get("anchor_id")
    if not anchor_id:
        return (
            "_Selecciona una licitación para ver sus detalles aquí._"
            if lang == "es" else
            "_Select a tender to see its details here._"
        )
    lic = next((l for l in chat_state.get("active_licitaciones", [])
                if l.get("id") == anchor_id), None)
    if not lic:
        return ""
    t     = _T[lang]
    today = date.today()
    lines = []

    desc = lic.get("descripcion") or lic.get("concepto_contratacion") or ""
    if desc:
        lines.append(f"**{desc[:100]}**\n")
    if ente := lic.get("ente_contratante"):
        lines.append(f"🏛️ {ente}")
    if codigo := lic.get("numero_procedimiento"):
        lines.append(f"📄 `{codigo}`")
    if materia := lic.get("materia"):
        lines.append(f"📌 {materia}")
    lines.append("")

    ap_str  = lic.get("fecha_apertura") or ""
    ap_hora = lic.get("hora_apertura") or ""
    if ap_str:
        ap_d  = _parse_date(ap_str)
        days  = (ap_d - today).days if ap_d else None
        label = f"{ap_str} {t['at_time']} {ap_hora}" if ap_hora else ap_str
        if days is not None:
            badge = t["urgent"] if days <= 5 else (t["upcoming"] if days <= 14 else t["on_track"])
            lines.append(f"⏰ **{label}**\n_{days} {t['days']}_ {badge}")
        else:
            lines.append(f"⏰ {label}")

    jun_str  = lic.get("fecha_junta_aclaraciones") or ""
    jun_hora = lic.get("hora_junta_aclaraciones") or ""
    if jun_str:
        label = f"{jun_str} {t['at_time']} {jun_hora}" if jun_hora else jun_str
        lines.append(f"📋 {label}")

    if costo := lic.get("costo_participacion"):
        lines.append(f"💰 {costo}")
    if url := lic.get("url"):
        view_lbl = "Ver en portal" if lang == "es" else "View on portal"
        lines.append(f"\n🔗 [{view_lbl}]({url})")

    if indexed_types:
        lbl = "Docs indexados" if lang == "es" else "Indexed docs"
        lines.append(f"\n**📂 {lbl}:**")
        for dt in indexed_types:
            lines.append(f"- {dt}")

    return "\n".join(lines)



def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _auto_title(message: str) -> str:
    clean = message.strip()
    return (clean[:50] + "…") if len(clean) > 50 else clean


# ── One-brain router ──────────────────────────────────────────────────────────

_VALID_INTENTS = {"discovery", "anchor", "detail", "clarify_no_context", "clarify_which"}
_VALID_ROUTES  = {"metadata", "rag"}


def _route_turn(message: str, state: dict, openai_key: str = "") -> dict:
    """
    Single LLM call that decides intent, search filters, display route, and
    which licitacion the user is referring to — all at once.

    Returns:
      {
        "intent":        "discovery" | "anchor" | "detail" | "clarify_no_context" | "clarify_which",
        "search":        {"keywords": [...], "materia": null | "..."},  # only used when intent=discovery
        "route":         "metadata" | "rag",  # anchor/detail only; "metadata" → card, "rag" → LLM
        "anchor_index":  null | int (1-based index into active_licitaciones)
      }
    """
    import os as _os
    import json as _json
    from openai import OpenAI

    anchor_id = state.get("anchor_id")
    active    = state.get("active_licitaciones", [])

    if anchor_id:
        anchored = next((l for l in active if l.get("id") == anchor_id), None)
        name = (
            (anchored.get("descripcion") or anchored.get("concepto_contratacion") or str(anchor_id))
            if anchored else str(anchor_id)
        )
        state_desc = f'Licitación actualmente seleccionada: "{name[:80]}"'
    elif active:
        previews = "; ".join(
            f"[{i+1}] {(l.get('descripcion') or l.get('concepto_contratacion') or '')[:50]}"
            for i, l in enumerate(active[:9])
        )
        state_desc = f"Lista activa ({len(active)} licitaciones): {previews}"
    else:
        state_desc = "Sin contexto previo — primera interacción"

    prompt = (
        "Eres el enrutador de un asistente de licitaciones publicas mexicanas.\n"
        "Analiza el mensaje del usuario y el estado de la conversacion, "
        "y toma UNA decision de enrutamiento en formato JSON.\n\n"
        f"Estado: {state_desc}\n"
        f'Mensaje: "{message}"\n\n'
        "Devuelve EXACTAMENTE este JSON (sin texto adicional):\n"
        '{"intent": "...", "search": {"keywords": [], "materia": null}, '
        '"route": null, "anchor_index": null}\n\n'
        "Reglas de intent:\n"
        "- discovery: el usuario quiere buscar, ver o listar licitaciones (nuevas o distintas)\n"
        "- anchor: el usuario MENCIONA UN NUMERO, CORCHETE [N], ORDINAL o NOMBRE de la lista activa "
        "(aunque tambien haga una pregunta sobre esa licitacion). Si el mensaje contiene [1], [2], "
        "'la primera', 'la segunda', o el nombre de alguna licitacion de la lista → SIEMPRE anchor.\n"
        "- detail: el usuario pregunta sobre la licitacion YA SELECCIONADA (anchor activo) "
        "sin referenciar un numero o indice de la lista\n"
        "- clarify_no_context: pregunta de detalle pero no hay licitacion ni lista activa\n"
        "- clarify_which: pregunta de detalle pero hay multiples activas y NINGUNA seleccionada\n\n"
        "Reglas de search (solo si intent=discovery):\n"
        "- keywords: lista de 2-5 terminos que cubran el concepto pedido. "
        "Incluye raiz singular, sinonimos y variantes. "
        "Ejemplo: 'seguros de vida' → ['seguro', 'vida', 'poliza']; "
        "'TI' → ['tecnologia', 'informatica', 'software', 'computo'].\n"
        "- materia: sector SOLO si el usuario lo menciona explicitamente, de lo contrario null\n\n"
        "Reglas de route (solo si intent=anchor o detail):\n"
        "- metadata: el usuario pide datos generales ('saber mas', 'dame informacion', 'ver detalles'). "
        "Tambien cuando selecciona una licitacion sin pregunta concreta.\n"
        "- rag: el usuario menciona EXPLICITAMENTE un documento ('convocatoria', 'bases', 'anexo') "
        "O hace una pregunta sobre CONTENIDO ('que piden', 'que requisitos', 'como participar'). "
        "Si el mensaje menciona 'convocatoria' o 'bases' → SIEMPRE rag.\n\n"
        "anchor_index: numero (1-based) si intent=anchor y el usuario menciona un numero u ordinal, "
        "de lo contrario null."
    )

    default = {
        "intent": "detail" if anchor_id else ("discovery" if not active else "clarify_which"),
        "search": {"keywords": [], "materia": None},
        "route": "rag",
        "anchor_index": None,
    }

    try:
        client = OpenAI(api_key=openai_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=200,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content or "{}"
        data = _json.loads(raw)

        intent = data.get("intent", "")
        if not isinstance(intent, str):
            intent = ""
        if intent not in _VALID_INTENTS:
            print(f"[_route_turn] invalid intent {intent!r}, using default")
            return default

        search = data.get("search") or {}
        keywords = search.get("keywords") or []
        materia  = search.get("materia") or None

        route = data.get("route") or "rag"
        if route not in _VALID_ROUTES:
            route = "rag"

        anchor_index = data.get("anchor_index")
        if anchor_index is not None:
            try:
                anchor_index = int(anchor_index)
            except (TypeError, ValueError):
                anchor_index = None

        return {
            "intent":       intent,
            "search":       {"keywords": keywords, "materia": materia},
            "route":        route,
            "anchor_index": anchor_index,
        }
    except Exception as e:
        print(f"[_route_turn] LLM failed: {e}, using default")
        return default


# ── Index resolver ────────────────────────────────────────────────────────────

def resolve_index(message: str, active: list):
    msg = message.lower()
    m = re.search(r'\[([1-9][0-9]*)\]|\b([1-9])\b', message)
    if m:
        idx = int(m.group(1) or m.group(2)) - 1
        if 0 <= idx < len(active):
            return idx
    ordinals = {
        "primer": 0, "primera": 0,
        "segund": 1, "segunda": 1,
        "tercer": 2, "tercera": 2,
        "cuart":  3, "cuarta":  3,
        "quint":  4, "quinta":  4,
    }
    for word, idx in ordinals.items():
        if word in msg and idx < len(active):
            return idx
    return None


# ── Document chunk helpers ────────────────────────────────────────────────────



def _discover_via_chunks(conn, query: str, emb_model: str, top_k: int,
                         cohere_key: str = "", openai_key: str = "") -> list[dict]:
    """
    Semantic fallback for discovery: run vector search across all Vigente chunks,
    deduplicate by licitacion_id (preserving relevance order), fetch their metadata.
    Used when ILIKE keyword search returns no results.
    """
    chunks = retrieve_chunks(
        conn, query,
        model=emb_model, top_k=top_k,
        chunk_config=DEFAULT_CHUNK_CONFIG,
        cohere_key=cohere_key, openai_key=openai_key,
    )
    seen = set()
    licitaciones = []
    for chunk in chunks:
        lid = chunk.get("licitacion_id")
        if lid and lid not in seen:
            seen.add(lid)
            lic = get_licitacion_by_id(conn, lid)
            if lic:
                licitaciones.append(lic)
    return licitaciones


def _get_available_materias(conn) -> list[str]:
    """Return distinct non-empty materia values for Vigente licitaciones with future deadlines."""
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT materia
        FROM licitaciones
        WHERE licitacion_status = 'Vigente'
          AND materia IS NOT NULL
          AND materia != ''
          AND fecha_apertura IS NOT NULL
          AND fecha_apertura != ''
          AND TO_DATE(fecha_apertura, 'DD/MM/YYYY') > CURRENT_DATE
        ORDER BY materia
    """)
    rows = cur.fetchall()
    cur.close()
    return [r[0] for r in rows]


def _get_available_doc_types(conn, licitacion_ids: list) -> dict:
    """
    Returns {licitacion_id: [doc_type, ...]} for each licitacion.
    Only doc_types that actually have indexed chunks are included
    (i.e., the user can ask questions about them).
    """
    if not licitacion_ids:
        return {}
    placeholders = ",".join(["%s"] * len(licitacion_ids))
    type_placeholders = ",".join(["%s"] * len(INDEXED_DOC_TYPES))
    cur = conn.cursor()
    cur.execute(f"""
        SELECT licitacion_id, doc_type
        FROM chunks
        WHERE licitacion_id IN ({placeholders})
          AND chunk_config = %s
          AND doc_type IN ({type_placeholders})
        GROUP BY licitacion_id, doc_type
        ORDER BY licitacion_id, doc_type
    """, licitacion_ids + [DEFAULT_CHUNK_CONFIG] + list(INDEXED_DOC_TYPES))
    rows = cur.fetchall()
    cur.close()

    result: dict = {}
    for lid, doc_type in rows:
        result.setdefault(lid, []).append(doc_type)
    return result


def _get_docs_table(conn, licitacion_id: int) -> list[dict]:
    """Returns Bases/Convocatoria/Junta document rows for a licitación from the documents table."""
    placeholders = ",".join(["%s"] * len(INDEXED_DOC_TYPES))
    cur = conn.cursor()
    cur.execute(f"""
        SELECT tipo, status
        FROM documents
        WHERE licitacion_id = %s
          AND tipo IN ({placeholders})
        ORDER BY tipo
    """, (licitacion_id, *INDEXED_DOC_TYPES))
    rows = cur.fetchall()
    cur.close()
    return [{"tipo": r[0], "status": r[1]} for r in rows]


def _format_docs_list(docs: list[dict], indexed_types: list[str], lang: str = "es") -> str:
    """Markdown listing of documents with indexing status."""
    if not docs:
        return (
            "Esta licitación no tiene documentos asociados aún."
            if lang == "es" else
            "This tender has no associated documents yet."
        )
    header = "📂 **Documentos disponibles:**\n" if lang == "es" else "📂 **Available documents:**\n"
    lines  = [header]
    for d in docs:
        tipo   = d["tipo"]
        status = d["status"]
        if tipo in indexed_types:
            icon = "✅"
            note = " _(indexado)_" if lang == "es" else " _(indexed)_"
        elif status in ("downloaded", "chunked"):
            icon = "📄"
            note = " _(descargado, pendiente de indexar)_" if lang == "es" else " _(downloaded, pending indexing)_"
        elif status == "pending":
            icon = "⏳"
            note = " _(pendiente de descarga)_" if lang == "es" else " _(pending download)_"
        elif status == "error":
            icon = "❌"
            note = " _(error al descargar)_" if lang == "es" else " _(download error)_"
        else:
            icon = "📄"
            note = ""
        lines.append(f"- {icon} {tipo}{note}")
    hint = (
        "\n_Puedes preguntarme sobre el contenido de los documentos indexados (✅)._"
        if lang == "es" else
        "\n_You can ask me about the content of indexed documents (✅)._"
    )
    lines.append(hint)
    return "\n".join(lines)


def _get_doc_chunks_for_summaries(conn, licitacion_ids: list) -> dict:
    """
    Fetch up to 2 text chunks per licitacion for doc_type matching
    Convocatoria or Bases (ILIKE). Returns {licitacion_id: combined_text}.
    """
    if not licitacion_ids:
        return {}
    placeholders = ",".join(["%s"] * len(licitacion_ids))
    cur = conn.cursor()
    cur.execute(f"""
        SELECT licitacion_id, doc_type, text
        FROM (
            SELECT licitacion_id, doc_type, text,
                   ROW_NUMBER() OVER (PARTITION BY licitacion_id, doc_type ORDER BY chunk_index) AS rn
            FROM chunks
            WHERE licitacion_id IN ({placeholders})
              AND (doc_type ILIKE '%%convocatoria%%' OR doc_type ILIKE '%%base%%')
              AND chunk_config = %s
        ) ranked
        WHERE rn <= 2
        ORDER BY licitacion_id, doc_type, rn
    """, licitacion_ids + [DEFAULT_CHUNK_CONFIG])
    rows = cur.fetchall()
    cur.close()

    result: dict = {}
    for lid, doc_type, text in rows:
        existing = result.get(lid, "")
        header = f"[{doc_type}]\n"
        result[lid] = existing + header + (text or "") + "\n\n"
    return result


def _batch_summarize(chunks_by_id: dict, model: str = "gpt-4o-mini", lang: str = "es", openai_key: str = "") -> dict:
    """
    One LLM call to produce a 2-sentence summary per licitacion.
    chunks_by_id: {licitacion_id: combined_text}
    Returns {licitacion_id: summary_str} or {} on failure.
    """
    if not chunks_by_id:
        return {}

    import json as _json
    from openai import OpenAI
    import os as _os

    blocks = []
    for lid, text in chunks_by_id.items():
        snippet = text[:1200]  # cap per licitacion to control token cost
        blocks.append(f"=== ID:{lid} ===\n{snippet}")

    prompt = _T[lang]["summarize_prompt"] + "\n\n".join(blocks)

    try:
        client = OpenAI(api_key=openai_key)
        resp = client.chat.completions.create(
            model=model,
            max_tokens=800,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content or "{}"
        data = _json.loads(raw)
        # Normalise keys: the model may return them as strings or ints
        return {int(k): v for k, v in data.items() if v}
    except Exception as e:
        print(f"[_batch_summarize] warning: {e}")
        return {}


def _quick_action_updates(state: dict, indexed_types: list = None) -> list:
    """
    Return gr.update() for the 8 quick-action buttons based on current state.
    Order: btn_vigentes, btn_bases, btn_convocatoria, btn_junta, btn_req,
           btn_specs, btn_docs, btn_economic
    """
    anchor_id = state.get("anchor_id")
    if not anchor_id:
        return [
            gr.update(visible=True),   # btn_vigentes
            gr.update(visible=False),  # btn_bases
            gr.update(visible=False),  # btn_convocatoria
            gr.update(visible=False),  # btn_junta
            gr.update(visible=False),  # btn_req
            gr.update(visible=False),  # btn_specs
            gr.update(visible=False),  # btn_docs
            gr.update(visible=False),  # btn_economic
        ]
    indexed = set(indexed_types or [])
    has_bases = "Bases" in indexed
    has_conv  = "Convocatoria" in indexed
    has_junta = any("junta" in dt.lower() or "aclaraciones" in dt.lower() for dt in indexed)
    return [
        gr.update(visible=False),          # btn_vigentes
        gr.update(visible=has_bases),      # btn_bases
        gr.update(visible=has_conv),       # btn_convocatoria
        gr.update(visible=has_junta),      # btn_junta
        gr.update(visible=True),           # btn_req
        gr.update(visible=has_bases or has_conv),  # btn_specs
        gr.update(visible=has_bases or has_conv),  # btn_docs
        gr.update(visible=has_conv),               # btn_economic
    ]


def _lic_selector_update(active: list, lang: str = "es", anchor_id=None) -> gr.update:
    """Return gr.update() for the licitacion selector dropdown."""
    if not active:
        return gr.update(choices=[], visible=False, value=None)
    today = date.today()
    choices = []
    selected_val = ""
    for i, l in enumerate(active, 1):
        desc = (l.get("descripcion") or l.get("concepto_contratacion") or f"Licitación {i}")
        ap_str  = l.get("fecha_apertura") or ""
        ap_date = _parse_date(ap_str)
        days    = (ap_date - today).days if ap_date else None
        urgency = " 🔴" if days is not None and days <= 5 else (" 🟡" if days is not None and days <= 14 else " 🟢")
        date_info = f"  ·  {ap_str} ({days}d){urgency}" if days is not None else ""
        choices.append((f"[{i}] {desc}{date_info}", str(i)))
        if anchor_id and l.get("id") == anchor_id:
            selected_val = str(i)
    placeholder = "— Select a tender —" if lang == "es" else "— Select a tender —"
    return gr.update(choices=[(placeholder, "")] + choices, visible=True, value=selected_val,
                     label=_T[lang]["selector_label"](len(active)))



# ── Discovery list formatter ──────────────────────────────────────────────────

def format_discovery_list(licitaciones: list,
                          summaries: dict | None = None,
                          doc_types: dict | None = None,
                          lang: str = "es") -> str:
    t      = _T[lang]
    today  = date.today()
    blocks = []

    for i, l in enumerate(licitaciones, 1):
        desc   = (l.get("descripcion") or l.get("concepto_contratacion") or t["no_desc"])
        if len(desc) > 120:
            desc = desc[:117] + "…"
        codigo = l.get("numero_procedimiento") or ""
        ente   = l.get("ente_contratante") or ""
        conv_str = l.get("fecha_convocatoria") or ""

        jun_str  = l.get("fecha_junta_aclaraciones") or ""
        jun_hora = l.get("hora_junta_aclaraciones") or ""
        jun_date = _parse_date(jun_str)
        if jun_date:
            jun_days  = (jun_date - today).days
            jun_label = f"{jun_str} {t['at_time']} {jun_hora}" if jun_hora else jun_str
            if jun_days >= 0:
                jun_line = f"📋 {t['clarif_meeting']}: **{jun_label}** · _{jun_days} {t['days']}_"
            else:
                jun_line = f"📋 {t['clarif_meeting']}: {jun_label} _({t['already_occurred']})_"
        else:
            jun_line = ""

        ap_str  = l.get("fecha_apertura") or ""
        ap_hora = l.get("hora_apertura") or ""
        ap_date = _parse_date(ap_str)
        ap_days = (ap_date - today).days if ap_date else None
        ap_label = f"{ap_str} {t['at_time']} {ap_hora}" if ap_hora else ap_str

        if ap_days is not None:
            urgency = t["urgent"] if ap_days <= 5 else (t["upcoming"] if ap_days <= 14 else t["on_track"])
            ap_line = f"⏰ {t['deadline']}: **{ap_label}** · _{ap_days} {t['days']}_  {urgency}"
        else:
            ap_line = f"⏰ {t['deadline']}: {ap_label}"

        costo = l.get("costo_participacion") or ""
        url   = l.get("url") or ""

        lines = [f"**[{i}] {desc}**"]
        if codigo:
            lines.append(f"📄 {t['code']}: `{codigo}`")
        lines.append(f"🏛️ {t['contracting_ent']}: {ente}")
        if conv_str:
            lines.append(f"📅 {t['pub_date']}: {conv_str}")
        if jun_line:
            lines.append(jun_line)
        lines.append(ap_line)
        lugar = l.get("lugar_apertura") or ""
        if lugar:
            lines.append(f"📍 {t['location']}: {lugar}")
        if costo:
            lines.append(f"💰 {t['cost']}: {costo}")
        if url:
            lines.append(f"🔗 [{t['view_portal']}]({url})")

        lid = l.get("id")
        if doc_types and lid and lid in doc_types:
            types_str = " · ".join(doc_types[lid])
            lines.append(f"📂 {t['docs_consultable']}: {types_str}")
        if summaries and lid and lid in summaries:
            lines.append(f"📝 **{t['summary_lbl']}:** {summaries[lid]}")

        blocks.append("\n".join(lines))

    return "\n\n---\n\n".join(blocks)


# ── Anchor detail card ────────────────────────────────────────────────────────

def _format_anchor_card(lic: dict, doc_types_list: list, lang: str = "es") -> str:
    """Full metadata card shown when the user selects a licitacion without a specific question."""
    t     = _T[lang]
    today = date.today()
    lines = []

    desc = lic.get("descripcion") or lic.get("concepto_contratacion") or t["no_desc"]
    lines.append(f"**{desc}**\n")

    if codigo := lic.get("numero_procedimiento"):
        lines.append(f"📄 **{t['code']}:** `{codigo}`")
    if tipo := lic.get("tipo_procedimiento"):
        lines.append(f"🔖 **{t['procedure_type']}:** {tipo}")
    if ente := lic.get("ente_contratante"):
        lines.append(f"🏛️ **{t['contracting_ent']}:** {ente}")
    if ente_sol := lic.get("ente_solicitante"):
        if ente_sol != lic.get("ente_contratante"):
            lines.append(f"🏢 **{t['requesting_ent']}:** {ente_sol}")
    if materia := lic.get("materia"):
        lines.append(f"📌 **{t['subject']}:** {materia}")
    if tipo_c := lic.get("tipo_contrato"):
        lines.append(f"📝 **{t['contract_type']}:** {tipo_c}")

    lines.append("")

    if conv := lic.get("fecha_convocatoria"):
        lines.append(f"📅 **{t['pub_date_card']}:** {conv}")

    jun_str  = lic.get("fecha_junta_aclaraciones") or ""
    jun_hora = lic.get("hora_junta_aclaraciones") or ""
    if jun_str:
        jun_d  = _parse_date(jun_str)
        days   = (jun_d - today).days if jun_d else None
        label  = f"{jun_str} {t['at_time']} {jun_hora}" if jun_hora else jun_str
        if days is not None:
            suffix = f" · _{days} {t['days']}_" if days >= 0 else f" _({t['already_occurred']})_"
        else:
            suffix = ""
        lines.append(f"📋 **{t['clarif_meeting']}:** {label}{suffix}")
    if lugar_jun := lic.get("lugar_junta_aclaraciones"):
        lines.append(f"   📍 {lugar_jun}")

    ap_str  = lic.get("fecha_apertura") or ""
    ap_hora = lic.get("hora_apertura") or ""
    if ap_str:
        ap_d  = _parse_date(ap_str)
        days  = (ap_d - today).days if ap_d else None
        label = f"{ap_str} {t['at_time']} {ap_hora}" if ap_hora else ap_str
        if days is not None:
            urgency = t["urgent"] if days <= 5 else (t["upcoming"] if days <= 14 else t["on_track"])
            lines.append(f"⏰ **{t['deadline_card']}:** **{label}** · _{days} {t['days']}_ {urgency}")
        else:
            lines.append(f"⏰ **{t['deadline_card']}:** {label}")
    if lugar_ap := lic.get("lugar_apertura"):
        lines.append(f"   📍 **{t['location_card']}:** {lugar_ap}")

    if costo := lic.get("costo_participacion"):
        lines.append(f"💰 **{t['cost_card']}:** {costo}")
    if url := lic.get("url"):
        lines.append(f"\n🔗 [{t['view_portal_card']}]({url})")

    if doc_types_list:
        lines.append(f"\n📂 **{t['docs_available']}:** {' · '.join(doc_types_list)}")
        lines.append(t["docs_hint"])

    return "\n".join(lines)


# ── Core chat logic ───────────────────────────────────────────────────────────

def _prepare_turn(message: str, state: dict, emb_model: str, top_k: int, conn, lang: str = "es",
                  openai_key: str = "", cohere_key: str = ""):
    """
    Runs all non-LLM work for one chat turn.

    Returns (prefix, new_state, llm_context, llm_question) where:
      - prefix       : text to show immediately (complete response for non-LLM paths,
                       or anchor confirmation line for LLM paths)
      - new_state    : updated chat state
      - llm_context  : str if the LLM should be called, None otherwise
      - llm_question : the cleaned question to send to the LLM (or None)
    """
    tr = _T[lang]
    decision = _route_turn(message, state, openai_key=openai_key)
    intent   = decision["intent"]

    # Hard override: explicit index/ordinal reference always means anchor selection,
    # regardless of how the LLM classified it (LLM sometimes returns "detail" for
    # messages like "quiero saber mas sobre [3]")
    #
    # When an anchor is already set, ONLY [N] bracket notation can switch it —
    # bare digits in text like "Anexo económico 1" must not falsely trigger an anchor switch.
    active         = state.get("active_licitaciones", [])
    current_anchor = state.get("anchor_id")
    if active and intent not in ("discovery", "clarify_no_context"):
        if current_anchor:
            # Already focused: require explicit [N] brackets to switch
            bracket = re.search(r'\[([1-9][0-9]*)\]', message)
            forced_idx = (int(bracket.group(1)) - 1) if bracket else None
        else:
            # No anchor yet: use full resolver (brackets, ordinals, bare digits)
            forced_idx = resolve_index(message, active)
        if (
            forced_idx is not None
            and 0 <= forced_idx < len(active)
            and active[forced_idx]["id"] != current_anchor
        ):
            decision = {**decision, "intent": "anchor", "anchor_index": forced_idx + 1}
            intent = "anchor"

    # Hard override: bare [N] or ordinal with no real question → always show metadata card.
    # Prevents the LLM receiving "[31]" as a question and returning the fallback phrase.
    if intent == "anchor" and re.fullmatch(r'\s*\[?[1-9][0-9]*\]?\s*', message.strip()):
        decision = {**decision, "route": "metadata"}

    # Hard override: message clearly asks about document content → force RAG
    _DOC_KWS = ("convocatoria", "bases", "base", "anexo", "formato",
                 "requisitos", "especificaciones", "contrato", "documento")
    if intent in ("anchor", "detail") and decision.get("route") == "metadata":
        if any(kw in message.lower() for kw in _DOC_KWS):
            decision = {**decision, "route": "rag"}

    # Procedure code override: detect "ICHIFE-LP-004-2026" style codes and anchor directly.
    # Runs after routing so it can short-circuit clarify_which / anchor-by-index failures.
    if intent not in ("discovery",):
        proc_match = re.search(r'\b([A-Z]{2,}(?:-[A-Z0-9]+){2,})\b', message)
        if proc_match:
            proc_code = proc_match.group(1)
            # Check in current active list first (no DB round-trip)
            found_idx = next(
                (i for i, l in enumerate(active)
                 if proc_code in (l.get("numero_procedimiento") or "")),
                None,
            )
            if found_idx is not None:
                decision = {**decision, "intent": "anchor", "anchor_index": found_idx + 1}
                intent = "anchor"
            else:
                lic = get_licitacion_by_numero(conn, proc_code)
                if lic:
                    lid = lic["id"]
                    avail = _get_available_doc_types(conn, [lid]).get(lid, [])
                    new_active = [lic] + [l for l in active if l["id"] != lid]
                    state = {**state, "active_licitaciones": new_active, "anchor_id": lid, "selected_doc_types": []}
                    card = _format_anchor_card(lic, avail, lang=lang)
                    lic_name = lic.get("descripcion") or lic.get("concepto_contratacion") or proc_code
                    return tr["focused_on"](lic_name) + card, state, None, None, []

    # ── Discovery ──────────────────────────────────────────────────────────────
    if intent == "discovery":
        search      = decision.get("search") or {}
        keywords     = search.get("keywords") or []
        materia      = search.get("materia") or None
        licitaciones = search_licitaciones(conn, keywords, materia, limit=None)
        if not licitaciones:
            licitaciones = _discover_via_chunks(conn, message, emb_model, top_k,
                                                cohere_key=cohere_key, openai_key=openai_key)
        if not licitaciones:
            materias = _get_available_materias(conn)
            if materias:
                materias_str = " · ".join(f"**{m}**" for m in materias)
                msg = tr["no_results_sectors"](materias_str)
            else:
                msg = tr["no_results_generic"]
            return msg, state, None, None, []
        state = {**state, "active_licitaciones": licitaciones, "anchor_id": None}
        n            = len(licitaciones)
        display_lics = licitaciones[:10]
        lid_list     = [l["id"] for l in display_lics if l.get("id")]
        doc_types    = _get_available_doc_types(conn, lid_list)
        chunks_by_id = _get_doc_chunks_for_summaries(conn, lid_list)
        summaries    = _batch_summarize(chunks_by_id, lang=lang, openai_key=openai_key) if chunks_by_id else {}
        suffix = f"\n\n{tr['showing_first_10'](n)}" if n > 10 else ""
        text = (
            f"{tr['found_n'](n)}\n\n"
            f"{format_discovery_list(display_lics, summaries, doc_types, lang=lang)}{suffix}\n\n"
            f"{tr['discovery_hint']}"
        )
        return text, state, None, None, []

    # ── Clarify ────────────────────────────────────────────────────────────────
    if intent == "clarify_no_context":
        return tr["clarify_no_ctx"], state, None, None, []

    if intent == "clarify_which":
        active  = state.get("active_licitaciones", [])
        options = "  ".join(f"[{i+1}]" for i in range(min(len(active), 9)))
        return tr["clarify_which"](len(active), options), state, None, None, []

    # ── Anchor / Detail ────────────────────────────────────────────────────────
    active    = state.get("active_licitaciones", [])
    anchor_id = state.get("anchor_id")

    anchor_confirmed_msg = ""
    if intent == "anchor":
        ai = decision.get("anchor_index")
        idx = (int(ai) - 1) if ai is not None else None
        if idx is None or not (0 <= idx < len(active)):
            idx = resolve_index(message, active)
        if idx is not None:
            prev_anchor = anchor_id
            anchor_id   = active[idx]["id"]
            if anchor_id != prev_anchor:
                lic_name = (
                    active[idx].get("descripcion")
                    or active[idx].get("concepto_contratacion")
                    or f"licitación #{anchor_id}"
                )
                anchor_confirmed_msg = tr["focused_on"](lic_name)
                state = {**state, "anchor_id": anchor_id, "selected_doc_types": []}
            else:
                state = {**state, "anchor_id": anchor_id}
        elif not anchor_id:
            options = "  ".join(f"[{i+1}]" for i in range(min(len(active), 9)))
            return tr["clarify_anchor"](options), state, None, None, []

    if anchor_id:
        lic = get_licitacion_by_id(conn, anchor_id)
        licitaciones   = [lic] if lic else []
        licitacion_ids = [anchor_id]
    elif len(active) == 1:
        licitaciones   = active
        licitacion_ids = [active[0]["id"]]
    else:
        licitaciones   = active[:5]
        licitacion_ids = [l["id"] for l in licitaciones]

    # Generic "more info" anchor → return full metadata card, no LLM needed
    if decision.get("route", "rag") == "metadata":
        avail_types = _get_available_doc_types(conn, licitacion_ids).get(licitacion_ids[0], []) \
                      if len(licitacion_ids) == 1 else []
        card = _format_anchor_card(licitaciones[0], avail_types, lang=lang) if licitaciones else ""
        return anchor_confirmed_msg + card, state, None, None, []

    # Document listing
    _DOC_LIST_PHRASES = (
        "qué documentos", "que documentos",
        "cuáles son los documentos", "cuales son los documentos",
        "what documents", "documentos disponibles", "documentos tiene", "documentos hay",
    )
    msg_lower = message.lower()
    if any(p in msg_lower for p in _DOC_LIST_PHRASES) and len(licitacion_ids) == 1:
        docs    = _get_docs_table(conn, licitacion_ids[0])
        indexed = _get_available_doc_types(conn, licitacion_ids).get(licitacion_ids[0], [])
        return anchor_confirmed_msg + _format_docs_list(docs, indexed, lang=lang), state, None, None, []

    # Specific question → vector search
    selected_doc_types = state.get("selected_doc_types", []) if len(licitacion_ids) == 1 else []
    rag_query = message
    if len(licitacion_ids) == 1 and licitaciones:
        desc = (licitaciones[0].get("descripcion")
                or licitaciones[0].get("concepto_contratacion") or "")
        if desc:
            rag_query = f"{desc}: {message}"
    rag_chunks = retrieve_chunks(
        conn, rag_query,
        model=emb_model, top_k=top_k,
        chunk_config=DEFAULT_CHUNK_CONFIG,
        licitacion_ids=licitacion_ids or [],
        doc_types=selected_doc_types or None,
        cohere_key=cohere_key, openai_key=openai_key,
    )

    context = build_context(licitaciones, rag_chunks)
    return anchor_confirmed_msg, state, context, message, rag_chunks


# ── Gradio event handlers ─────────────────────────────────────────────────────

_NO_BTN = [gr.update()] * 11  # no-op: lic_selector + 8 btns + banner + right_panel


def send_message(message: str, history: list, convo_id: str, chat_state: dict,
                 emb_model: str, synth_model: str, top_k: int, lang: str = "es",
                 anthropic_key: str = "", openai_key: str = "", cohere_key: str = ""):
    """Streaming generator — yields 16 outputs:
    history, txt, convo_id, state, dropdown, lic_selector,
    btn_vigentes, btn_bases, btn_convocatoria, btn_junta, btn_req,
    btn_specs, btn_docs, btn_economic,
    anchor_banner, right_panel_md
    """
    if not message.strip():
        yield history, "", convo_id, chat_state, gr.update(), *_NO_BTN
        return

    conn = get_conn()
    if not convo_id:
        convo_id = _new_id()
    title = _auto_title(message) if not history else None

    history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": "⏳ _Pensando..._" if lang == "es" else "⏳ _Thinking..._"},
    ]
    yield history, "", convo_id, chat_state, gr.update(), *_NO_BTN

    prefix, chat_state, llm_context, llm_question, _ = _prepare_turn(
        message, chat_state, emb_model, top_k, conn, lang=lang,
        openai_key=openai_key, cohere_key=cohere_key,
    )
    anchor_id     = chat_state.get("anchor_id")
    indexed_types = _get_available_doc_types(conn, [anchor_id]).get(anchor_id, []) if anchor_id else []
    btns       = _quick_action_updates(chat_state, indexed_types)
    sel_upd    = _lic_selector_update(chat_state.get("active_licitaciones", []), lang=lang,
                                      anchor_id=anchor_id)
    banner_upd = _format_anchor_banner(chat_state, lang=lang)
    right_upd  = _format_right_panel(chat_state, indexed_types, lang=lang)
    if llm_context is None:
        history[-1]["content"] = prefix
        yield history, "", convo_id, chat_state, gr.update(), sel_upd, *btns, banner_upd, right_upd
    else:
        history[-1]["content"] = prefix
        if prefix:
            yield history, "", convo_id, chat_state, gr.update(), sel_upd, *btns, banner_upd, right_upd

        full_llm = ""
        for delta in synthesize_stream(llm_question or "", llm_context, model=synth_model, lang=lang, api_key=anthropic_key):
            full_llm += delta
            history[-1]["content"] = prefix + full_llm
            yield history, "", convo_id, chat_state, gr.update(), sel_upd, *btns, banner_upd, right_upd

    existing    = _db_load_one(conn, convo_id)
    final_title = title if title else (existing["title"] if existing else _auto_title(message))
    _db_upsert(conn, convo_id, final_title, history, chat_state)

    convos  = _db_load_all(conn)
    choices = _sidebar_choices(convos)
    conn.close()

    yield history, "", convo_id, chat_state, gr.update(choices=choices, value=convo_id), sel_upd, *btns, banner_upd, right_upd


def new_conversation():
    conn    = get_conn()
    convos  = _db_load_all(conn)
    choices = _sidebar_choices(convos)
    conn.close()
    init_state = {"active_licitaciones": [], "anchor_id": None, "selected_doc_types": []}
    return ([], "", None, init_state, gr.update(choices=choices, value=None),
            _lic_selector_update([]),
            *_quick_action_updates(init_state),
            "",
            _format_right_panel(init_state, []))


def delete_conversation(convo_id_val: str, lang: str = "es"):
    """Delete selected conversation and reset the UI to a blank state."""
    init_state = {"active_licitaciones": [], "anchor_id": None, "selected_doc_types": []}
    conn   = get_conn()
    if convo_id_val:
        with conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM conversations WHERE id = %s", (convo_id_val,))
            cur.close()
    convos  = _db_load_all(conn)
    choices = _sidebar_choices(convos)
    conn.close()
    return (
        [], "", None, init_state,
        gr.update(choices=choices, value=None),
        _lic_selector_update([]),
        *_quick_action_updates(init_state),
        "",
        _format_right_panel(init_state, [], lang),
    )


def load_conversation(selected_id: str):
    empty_state = {"active_licitaciones": [], "anchor_id": None, "selected_doc_types": []}
    if not selected_id:
        return ([], empty_state, None, _lic_selector_update([]),
                *_quick_action_updates(empty_state),
                "", _format_right_panel(empty_state, []))
    conn  = get_conn()
    convo = _db_load_one(conn, selected_id)
    if not convo:
        conn.close()
        return ([], empty_state, None, _lic_selector_update([]),
                *_quick_action_updates(empty_state),
                "", _format_right_panel(empty_state, []))
    messages   = convo.get("messages") or []
    chat_state = convo.get("chat_state") or empty_state
    active     = chat_state.get("active_licitaciones", [])
    anchor_id  = chat_state.get("anchor_id")
    indexed    = _get_available_doc_types(conn, [anchor_id]).get(anchor_id, []) if anchor_id else []
    conn.close()
    return (messages, chat_state, selected_id,
            _lic_selector_update(active, anchor_id=anchor_id),
            *_quick_action_updates(chat_state, indexed),
            _format_anchor_banner(chat_state),
            _format_right_panel(chat_state, indexed))


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui(emb_model: str, synth_model: str, top_k: int):
    conn         = get_conn()
    convos       = _db_load_all(conn)
    init_choices = _sidebar_choices(convos)
    conn.close()

    with gr.Blocks(title="Licitaciones Chihuahua") as demo:

        convo_id   = gr.State(None)
        chat_state = gr.State({"active_licitaciones": [], "anchor_id": None, "selected_doc_types": []})
        lang_state = gr.State("es")

        with gr.Row():

            # ── Sidebar ───────────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("## 🏛️ Licitaciones\n_Estado de Chihuahua_")
                with gr.Row():
                    new_btn  = gr.Button("＋ Nueva conversación", variant="primary", size="sm", scale=3)
                    lang_btn = gr.Button("🇺🇸 English", size="sm", scale=1)
                hist_md = gr.Markdown("---\n**Historial**")
                convo_dropdown = gr.Dropdown(
                    choices=init_choices,
                    label="",
                    interactive=True,
                    allow_custom_value=False,
                    show_label=False,
                    info="Selecciona una conversación anterior para continuar.",
                )
                delete_btn = gr.Button(
                    "🗑️ Eliminar esta conversación",
                    size="sm", variant="stop", visible=False,
                )
                with gr.Accordion("🔑 API Keys", open=False):
                    anthropic_key_input = gr.Textbox(
                        label="Anthropic API Key",
                        placeholder="sk-ant-...",
                        type="password",
                        show_label=True,
                        info="Used for answer synthesis (Claude). Never stored.",
                    )
                    openai_key_input = gr.Textbox(
                        label="OpenAI API Key",
                        placeholder="sk-...",
                        type="password",
                        show_label=True,
                        info="Used for intent routing and licitacion summaries (gpt-4o-mini). Never stored.",
                    )
                    cohere_key_input = gr.Textbox(
                        label="Cohere API Key",
                        placeholder="...",
                        type="password",
                        show_label=True,
                        info="Used for query embedding and reranking. Never stored.",
                    )

            # ── Chat area ─────────────────────────────────────────────────────
            with gr.Column(scale=5):
                chat_title_md = gr.Markdown(
                    "### Asistente de Licitaciones Públicas\n"
                    "_Solo muestra licitaciones cuyo plazo de participación no ha vencido._"
                )
                anchor_banner = gr.Markdown("")
                chatbot = gr.Chatbot(height=560, min_height=480, show_label=False, autoscroll=True)
                lic_selector = gr.Dropdown(
                    choices=[], visible=False, value=None,
                    label="", interactive=True, show_label=False,
                )
                with gr.Row():
                    btn_vigentes    = gr.Button("📋 Ver Licitaciones Vigentes", size="sm", visible=True)
                    btn_bases       = gr.Button("📄 Bases",                     size="sm", visible=False)
                    btn_convocatoria = gr.Button("📢 Convocatoria",             size="sm", visible=False)
                    btn_junta       = gr.Button("🗣️ Junta de Aclaraciones",     size="sm", visible=False)
                    btn_req         = gr.Button("📋 Ver requisitos",            size="sm", visible=False)
                with gr.Row():
                    btn_specs    = gr.Button("📦 Especificaciones técnicas", size="sm", visible=False)
                    btn_docs     = gr.Button("📑 Documentos requeridos",     size="sm", visible=False)
                    btn_economic = gr.Button("💰 Condiciones económicas",    size="sm", visible=False)
                with gr.Row():
                    txt = gr.Textbox(
                        placeholder="Ej: ¿qué licitaciones de pavimentación hay vigentes?",
                        scale=8,
                        show_label=False,
                        autofocus=True,
                    )
                    send_btn = gr.Button("Enviar", scale=1, variant="primary")
                    stop_btn = gr.Button("⏹", scale=0, variant="stop", min_width=48)

            # ── Right info panel ──────────────────────────────────────────────
            with gr.Column(scale=2, min_width=240):
                right_panel_title = gr.Markdown("### 📋 Detalle")
                right_panel_md = gr.Markdown(
                    "_Selecciona una licitación para ver sus detalles aquí._"
                )

        # ── Wiring ────────────────────────────────────────────────────────────

        _btn_outs = [btn_vigentes, btn_bases, btn_convocatoria, btn_junta, btn_req,
                     btn_specs, btn_docs, btn_economic]

        send_in  = [txt, chatbot, convo_id, chat_state,
                    gr.State(emb_model), gr.State(synth_model), gr.State(top_k), lang_state,
                    anthropic_key_input, openai_key_input, cohere_key_input]
        send_out = ([chatbot, txt, convo_id, chat_state, convo_dropdown,
                     lic_selector] + _btn_outs +
                    [anchor_banner, right_panel_md])

        submit_ev   = txt.submit(send_message,     inputs=send_in, outputs=send_out)
        send_btn_ev = send_btn.click(send_message, inputs=send_in, outputs=send_out)
        stop_btn.click(fn=None, cancels=[submit_ev, send_btn_ev])

        # Language toggle: flip state and update all UI labels
        def _toggle_lang(current_lang, state):
            new_lang = "en" if current_lang == "es" else "es"
            t = _T[new_lang]
            active    = state.get("active_licitaciones", [])
            anchor_id = state.get("anchor_id")
            return (
                new_lang,
                gr.update(value=t["lang_toggle"]),
                gr.update(value=t["btn_new_convo"]),
                gr.update(value=t["chat_title"]),
                gr.update(value=t["history_label"]),
                gr.update(placeholder=t["placeholder"]),
                gr.update(value=t["send_btn"]),
                gr.update(value=t["btn_vigentes"]),
                gr.update(value=t["btn_bases"]),
                gr.update(value=t["btn_convocatoria"]),
                gr.update(value=t["btn_junta"]),
                gr.update(value=t["btn_req"]),
                gr.update(value=t["btn_specs"]),
                gr.update(value=t["btn_docs"]),
                gr.update(value=t["btn_economic"]),
                gr.update(value=t["btn_delete_convo"]),
                _lic_selector_update(active, lang=new_lang, anchor_id=anchor_id),
                gr.update(value=t["right_panel_title"]),
                gr.update(value=t["right_panel_empty"]) if not anchor_id else gr.update(),
            )

        lang_btn.click(
            _toggle_lang,
            inputs=[lang_state, chat_state],
            outputs=[lang_state, lang_btn, new_btn, chat_title_md, hist_md,
                     txt, send_btn] + _btn_outs +
                    [delete_btn, lic_selector,
                     right_panel_title, right_panel_md],
        )

        # Quick-action buttons: inject language-appropriate message then submit
        def _make_quick_click(msg_key):
            def handler(lang):
                return _T[lang][msg_key]
            return handler

        for btn, msg_key in [
            (btn_vigentes,     "msg_vigentes"),
            (btn_bases,        "msg_bases"),
            (btn_convocatoria, "msg_convocatoria"),
            (btn_junta,        "msg_junta"),
            (btn_req,          "msg_req"),
            (btn_specs,        "msg_specs"),
            (btn_docs,         "msg_docs"),
            (btn_economic,     "msg_economic"),
        ]:
            btn.click(_make_quick_click(msg_key), inputs=[lang_state], outputs=[txt]).then(
                send_message, inputs=send_in, outputs=send_out
            )

        # Selector dropdown: inject "[N]" and auto-submit
        lic_selector.change(
            lambda v: f"[{v}]" if v else "",
            inputs=[lic_selector],
            outputs=[txt],
        ).then(send_message, inputs=send_in, outputs=send_out)

        # New conversation — reset everything, hide delete button
        new_btn_ev = new_btn.click(
            new_conversation,
            outputs=([chatbot, txt, convo_id, chat_state, convo_dropdown,
                      lic_selector] + _btn_outs +
                     [anchor_banner, right_panel_md]),
        )
        new_btn_ev.then(lambda: gr.update(visible=False), outputs=[delete_btn])

        # Load conversation — restore state and show delete button
        convo_dropdown.change(
            load_conversation,
            inputs=[convo_dropdown],
            outputs=([chatbot, chat_state, convo_id,
                      lic_selector] + _btn_outs +
                     [anchor_banner, right_panel_md]),
        )
        convo_dropdown.change(
            lambda v: gr.update(visible=bool(v)),
            inputs=[convo_dropdown],
            outputs=[delete_btn],
        )

        # Delete conversation — wipe state, hide delete button
        delete_btn_ev = delete_btn.click(
            delete_conversation,
            inputs=[convo_id, lang_state],
            outputs=([chatbot, txt, convo_id, chat_state, convo_dropdown,
                      lic_selector] + _btn_outs +
                     [anchor_banner, right_panel_md]),
        )
        delete_btn_ev.then(lambda: gr.update(visible=False), outputs=[delete_btn])

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Licitaciones Chihuahua — chat UI")
    parser.add_argument("--model",           default=DEFAULT_EMB_MODEL,
                        choices=["cohere", "openai"])
    parser.add_argument("--synthesis-model", default=DEFAULT_SYNTH_MODEL,
                        help="LLM for answer synthesis (default: gpt-4o)")
    parser.add_argument("--top-k",           type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--share",           action="store_true")
    parser.add_argument("--port",            type=int, default=7860)
    args = parser.parse_args()

    print(f"Starting  emb={args.model}  synth={args.synthesis_model}  top_k={args.top_k}")
    demo = build_ui(args.model, args.synthesis_model, args.top_k)
    theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
    demo.launch(share=args.share, server_name="0.0.0.0", server_port=args.port, theme=theme)


if __name__ == "__main__":
    main()
