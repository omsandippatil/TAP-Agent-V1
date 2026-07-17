import logging

import httpx

from app.config import settings

logger = logging.getLogger("tap.google_search")

GOOGLE_SEARCH_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
GOOGLE_SEARCH_DAILY_CAP_DEFAULT = 90

_startup_logged = False


def _log_startup_status_once():
    global _startup_logged
    if _startup_logged:
        return
    _startup_logged = True
    key_present = bool(settings.google_search_api_key.strip())
    cx_present = bool(settings.google_search_engine_id.strip())
    logger.info(
        "google search config check key_present=%s engine_id_present=%s configured=%s",
        key_present, cx_present, settings.google_search_configured,
    )


def google_search_configured_and_available(quota_guard=None) -> bool:
    _log_startup_status_once()
    if not settings.google_search_configured:
        return False
    if quota_guard is None:
        return True
    return bool(quota_guard.has_quota())


async def _register_quota_usage(quota_guard) -> None:
    if quota_guard is not None:
        await quota_guard.record_usage()


async def call_google_custom_search(query: str, num: int = 8, quota_guard=None) -> list[dict]:
    if not google_search_configured_and_available(quota_guard):
        return []
    params = {
        "key": settings.google_search_api_key,
        "cx": settings.google_search_engine_id,
        "q": query,
        "num": min(max(num, 1), 10),
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(GOOGLE_SEARCH_ENDPOINT, params=params)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "google custom search http error status=%s body=%s query=%r",
            exc.response.status_code, exc.response.text[:300], query,
        )
        return []
    except httpx.HTTPError as exc:
        logger.warning("google custom search request failed error=%s query=%r", exc, query)
        return []
    await _register_quota_usage(quota_guard)
    items = payload.get("items", []) or []
    if not items:
        logger.info("google custom search zero results query=%r", query)
    return items


async def google_search_web(query: str, max_results: int = 5, quota_guard=None) -> list[dict]:
    items = await call_google_custom_search(query, num=max_results, quota_guard=quota_guard)
    return [
        {"href": item.get("link", ""), "title": item.get("title", ""), "body": item.get("snippet", "")}
        for item in items
        if item.get("link")
    ]


def is_linkedin_profile_url(url: str) -> bool:
    return bool(url) and "linkedin.com/in/" in url.lower()


async def google_search_linkedin_profiles(company: str, role_hint: str = "", max_results: int = 8, quota_guard=None) -> list[dict]:
    query = f'site:linkedin.com/in "{company}" {role_hint}'.strip()
    items = await call_google_custom_search(query, num=max_results, quota_guard=quota_guard)
    return [
        {"href": item.get("link", ""), "title": item.get("title", ""), "body": item.get("snippet", "")}
        for item in items
        if is_linkedin_profile_url(item.get("link", ""))
    ]