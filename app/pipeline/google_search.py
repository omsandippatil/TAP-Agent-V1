import asyncio
import logging

import httpx

from app.config import settings

logger = logging.getLogger("tap.google_search")

GOOGLE_SEARCH_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
GOOGLE_SEARCH_DAILY_CAP_DEFAULT = 90
CANARY_QUERY = "site:wikipedia.org test"

_startup_logged = False
_canary_checked = False
_canary_ok: bool | None = None
_canary_lock = asyncio.Lock()


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


async def run_startup_canary_check() -> bool:
    """Fire one known-good query at boot. A zero-result response for this query is a
    strong signal the CSE is misconfigured (e.g. 'search entire web' is off, or the
    engine is scoped to an empty/wrong site list) rather than normal search variance.
    Safe to call repeatedly; only runs once.
    """
    global _canary_checked, _canary_ok
    async with _canary_lock:
        if _canary_checked:
            return bool(_canary_ok)
        _canary_checked = True
        if not settings.google_search_configured:
            _canary_ok = None
            return False
        items = await call_google_custom_search(CANARY_QUERY, num=1, quota_guard=None, _is_canary=True)
        _canary_ok = bool(items)
        if not _canary_ok:
            logger.error(
                "GOOGLE CSE STARTUP CANARY FAILED — query=%r returned zero results. This almost "
                "always means the Programmable Search Engine is misconfigured: check that "
                "'Search the entire web' is enabled and the site restriction list is empty, at "
                "https://programmablesearchengine.google.com/. All searches will silently "
                "degrade until this is fixed.",
                CANARY_QUERY,
            )
        else:
            logger.info("google cse startup canary OK query=%r", CANARY_QUERY)
        return _canary_ok


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


async def call_google_custom_search(query: str, num: int = 8, quota_guard=None, _is_canary: bool = False) -> list[dict]:
    if not google_search_configured_and_available(quota_guard):
        return []
    if not _is_canary and not _canary_checked:
        # Fire-and-forget: don't block real queries on the canary, but make sure it runs.
        asyncio.ensure_future(run_startup_canary_check())

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

    if not _is_canary:
        await _register_quota_usage(quota_guard)

    items = payload.get("items", []) or []
    if not items:
        search_info = payload.get("searchInformation", {})
        logger.info(
            "google custom search zero results query=%r total_results=%s status_ok=%s",
            query, search_info.get("totalResults", "?"), "items" in payload or response.status_code == 200,
        )
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