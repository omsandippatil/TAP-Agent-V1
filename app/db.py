import logging
from datetime import datetime, timezone
from typing import Any

from supabase import create_client, Client
from postgrest.exceptions import APIError

from app.config import settings

logger = logging.getLogger("tap.db")

_client: Client | None = None

SCREENING_FILES_BUCKET = "screening-files"

HISTORY_LIST_FIELDS = (
    "id, company, company_slug, mode, state, fit_score, tier_label, "
    "tier_color, strategic_insight, logo_url, created_at, user_id"
)

FULL_RESULT_FIELDS = (
    "id, company, company_slug, mode, state, fit_score, tier_label, "
    "tier_key, tier_color, analysis, sources, source_bank, decision_makers, "
    "important_links, score_breakdown, strategic_insight, logo_url, created_at"
)


def supabase_configured() -> bool:
    return bool(settings.supabase_url.strip() and settings.supabase_key.strip())


def get_client() -> Client | None:
    global _client
    if not supabase_configured():
        return None
    if _client is None:
        _client = create_client(settings.supabase_url, settings.supabase_key)
    return _client


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(company: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in company.strip().lower()).strip("_") or "company"


def save_screening_result(company: str, mode: str, result: dict, cfg: dict | None = None,
                           user_id: str | None = None) -> dict | None:
    client = get_client()
    if client is None:
        logger.warning("save_screening_result skipped: supabase not configured company=%r", company)
        return None

    tier = result.get("scoring_tier") or {}
    row = {
        "company": company,
        "company_slug": _slugify(company),
        "mode": mode,
        "user_id": user_id,
        "state": result.get("state"),
        "fit_score": result.get("fit_score"),
        "tier_label": tier.get("label"),
        "tier_key": tier.get("key"),
        "tier_color": tier.get("color"),
        "analysis": result.get("analysis"),
        "sources": result.get("sources"),
        "source_bank": result.get("source_bank"),
        "decision_makers": result.get("decision_makers"),
        "important_links": result.get("important_links"),
        "score_breakdown": result.get("score_breakdown"),
        "strategic_insight": result.get("strategic_insight"),
        "logo_url": result.get("logo_url"),
        "cfg_snapshot": cfg,
        "completed_at": _now_iso(),
    }
    try:
        response = client.table("screenings").insert(row).execute()
    except APIError as exc:
        logger.error("save_screening_result failed company=%r error=%s", company, exc)
        return None

    data = response.data or []
    if not data:
        logger.error("save_screening_result returned no row company=%r", company)
        return None

    screening_id = data[0]["id"]
    log_job_event(screening_id, "scored",
                  f"Scored company={company!r} fit_score={result.get('fit_score')} state={result.get('state')}",
                  meta={"fit_score": result.get("fit_score"), "state": result.get("state")})
    return data[0]


def get_screening(screening_id: str) -> dict | None:
    client = get_client()
    if client is None:
        return None
    try:
        response = client.table("screenings").select(FULL_RESULT_FIELDS).eq("id", screening_id).single().execute()
    except APIError as exc:
        logger.info("get_screening not found id=%s error=%s", screening_id, exc)
        return None
    return response.data


def delete_screening(screening_id: str) -> None:
    client = get_client()
    if client is None:
        return
    try:
        client.table("screenings").delete().eq("id", screening_id).execute()
    except APIError as exc:
        logger.error("delete_screening failed id=%s error=%s", screening_id, exc)


def list_screenings(limit: int = 100, offset: int = 0, company_slug: str | None = None) -> list[dict]:
    client = get_client()
    if client is None:
        return []
    query = client.table("screenings").select(HISTORY_LIST_FIELDS).order(
        "created_at", desc=True
    ).range(offset, offset + limit - 1)

    if company_slug is not None:
        query = query.eq("company_slug", company_slug)

    try:
        response = query.execute()
    except APIError as exc:
        logger.error("list_screenings failed error=%s", exc)
        return []
    return response.data or []


def company_history(company: str, limit: int = 20) -> list[dict]:
    return list_screenings(limit=limit, company_slug=_slugify(company))


def search_screenings(query_text: str, limit: int = 8) -> list[dict]:
    client = get_client()
    if client is None:
        return []
    cleaned = query_text.strip()
    if not cleaned:
        return []

    query = client.table("screenings").select(HISTORY_LIST_FIELDS).ilike(
        "company", f"%{cleaned}%"
    ).order("created_at", desc=True).limit(limit * 4)

    try:
        response = query.execute()
    except APIError as exc:
        logger.error("search_screenings failed query=%r error=%s", query_text, exc)
        return []

    rows = response.data or []
    seen_slugs: set[str] = set()
    deduped: list[dict] = []
    for row in rows:
        slug = row.get("company_slug")
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        deduped.append(row)
        if len(deduped) >= limit:
            break
    return deduped


def get_company_logo(company: str) -> str | None:
    client = get_client()
    if client is None:
        return None
    query = (
        client.table("screenings")
        .select("logo_url, created_at")
        .eq("company_slug", _slugify(company))
        .not_.is_("logo_url", "null")
        .order("created_at", desc=True)
        .limit(1)
    )
    try:
        response = query.execute()
    except APIError as exc:
        logger.error("get_company_logo failed company=%r error=%s", company, exc)
        return None
    rows = response.data or []
    return rows[0]["logo_url"] if rows else None


def log_job_event(screening_id: str, event: str, message: str = "", meta: dict | None = None) -> None:
    client = get_client()
    if client is None:
        return
    try:
        client.table("job_events").insert({
            "screening_id": screening_id,
            "event": event,
            "message": message[:2000] if message else "",
            "meta": meta or {},
        }).execute()
    except APIError as exc:
        logger.warning("log_job_event failed id=%s event=%s error=%s", screening_id, event, exc)


def get_job_events(screening_id: str) -> list[dict]:
    client = get_client()
    if client is None:
        return []
    try:
        response = (
            client.table("job_events")
            .select("*")
            .eq("screening_id", screening_id)
            .order("created_at")
            .execute()
        )
    except APIError as exc:
        logger.error("get_job_events failed id=%s error=%s", screening_id, exc)
        return []
    return response.data or []


def log_job_run(company: str, mode: str, event: str, message: str = "", meta: dict | None = None) -> None:
    client = get_client()
    if client is None:
        return
    try:
        client.table("job_runs").insert({
            "company": company,
            "company_slug": _slugify(company),
            "mode": mode,
            "event": event,
            "message": message[:2000] if message else "",
            "meta": meta or {},
        }).execute()
    except APIError as exc:
        logger.warning("log_job_run failed company=%r event=%s error=%s", company, event, exc)


def save_screening_file(screening_id: str, label: str, filename: str, mime_type: str,
                         content_bytes: bytes) -> dict | None:
    client = get_client()
    if client is None:
        logger.warning("save_screening_file skipped: supabase not configured filename=%r", filename)
        return None

    storage_path = f"{screening_id}/{filename}"
    try:
        client.storage.from_(SCREENING_FILES_BUCKET).upload(
            path=storage_path,
            file=content_bytes,
            file_options={"content-type": mime_type, "upsert": "true"},
        )
    except Exception as exc:
        logger.error("save_screening_file upload failed id=%s filename=%r error=%s", screening_id, filename, exc)
        return None

    row = {
        "screening_id": screening_id,
        "label": label,
        "filename": filename,
        "mime_type": mime_type,
        "storage_bucket": SCREENING_FILES_BUCKET,
        "storage_path": storage_path,
        "size_bytes": len(content_bytes),
    }
    try:
        response = client.table("screening_files").insert(row).execute()
    except APIError as exc:
        logger.error("save_screening_file db insert failed id=%s filename=%r error=%s", screening_id, filename, exc)
        return None
    return (response.data or [None])[0]


def get_screening_files(screening_id: str) -> list[dict]:
    client = get_client()
    if client is None:
        return []
    try:
        response = (
            client.table("screening_files")
            .select("*")
            .eq("screening_id", screening_id)
            .execute()
        )
    except APIError as exc:
        logger.error("get_screening_files failed id=%s error=%s", screening_id, exc)
        return []
    return response.data or []


def get_signed_file_url(storage_path: str, expires_in_seconds: int = 3600) -> str | None:
    client = get_client()
    if client is None:
        return None
    try:
        response = client.storage.from_(SCREENING_FILES_BUCKET).create_signed_url(
            storage_path, expires_in_seconds
        )
    except Exception as exc:
        logger.error("get_signed_file_url failed path=%r error=%s", storage_path, exc)
        return None
    return response.get("signedURL") or response.get("signed_url")


def add_company_note(company: str, note: str, user_id: str | None = None) -> dict | None:
    client = get_client()
    if client is None:
        return None
    row = {
        "company": company,
        "company_slug": _slugify(company),
        "note": note,
        "user_id": user_id,
    }
    try:
        response = client.table("company_notes").insert(row).execute()
    except APIError as exc:
        logger.error("add_company_note failed company=%r error=%s", company, exc)
        return None
    return (response.data or [None])[0]


def get_company_notes(company: str) -> list[dict]:
    client = get_client()
    if client is None:
        return []
    query = (
        client.table("company_notes")
        .select("*")
        .eq("company_slug", _slugify(company))
        .order("created_at", desc=True)
    )
    try:
        response = query.execute()
    except APIError as exc:
        logger.error("get_company_notes failed company=%r error=%s", company, exc)
        return []
    return response.data or []


def find_screening_by_company(company: str) -> dict | None:
    client = get_client()
    if client is None:
        return None
    query = (
        client.table("screenings")
        .select(HISTORY_LIST_FIELDS)
        .eq("company_slug", _slugify(company))
        .order("created_at", desc=True)
        .limit(1)
    )
    try:
        response = query.execute()
    except APIError as exc:
        logger.error("find_screening_by_company failed company=%r error=%s", company, exc)
        return None
    rows = response.data or []
    return rows[0] if rows else None