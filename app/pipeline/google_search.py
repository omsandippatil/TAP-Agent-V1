import asyncio
import logging
import random

import httpx

from app.config import settings

logger = logging.getLogger("tap.google_search")

GOOGLE_SEARCH_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
CANARY_QUERY = "site:wikipedia.org test"

_startup_logged = False
_canary_checked = False
_canary_ok: bool | None = None
_canary_lock = asyncio.Lock()

_KEY_FAILURE_RETRY_STATUSES = {400, 403, 429}


class GoogleCseInvalidArgumentError(Exception):
    pass


def _log_startup_status_once():
    global _startup_logged
    if _startup_logged:
        return
    _startup_logged = True
    keys = settings.google_search_api_keys
    cx_present = bool(settings.google_search_engine_id.strip())
    logger.info(
        "google search config check key_count=%d engine_id_present=%s configured=%s",
        len(keys), cx_present, settings.google_search_configured,
    )


def _shuffled_key_pool() -> list[str]:
    keys = list(settings.google_search_api_keys)
    random.shuffle(keys)
    return keys


async def run_startup_canary_check() -> bool:
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
                "GOOGLE CSE STARTUP CANARY FAILED — query=%r returned zero results. Check that "
                "'Search the entire web' is enabled and the site restriction list is empty at "
                "https://programmablesearchengine.google.com/.",
                CANARY_QUERY,
            )
        else:
            logger.info("google cse startup canary OK")
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


async def _call_with_key(query: str, num: int, api_key: str) -> tuple[list[dict] | None, int | None]:
    params = {
        "key": api_key,
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
        status_code = exc.response.status_code
        body_text = exc.response.text[:300]
        logger.warning(
            "google custom search http error status=%s query=%r key_suffix=%s",
            status_code, query, api_key[-6:],
        )
        if status_code == 400:
            raise GoogleCseInvalidArgumentError(body_text) from exc
        return None, status_code
    except httpx.HTTPError as exc:
        logger.warning("google custom search request failed error=%s query=%r", exc, query)
        return None, None

    return payload.get("items", []) or [], None


async def call_google_custom_search(query: str, num: int = 8, quota_guard=None, _is_canary: bool = False) -> list[dict]:
    if not google_search_configured_and_available(quota_guard):
        return []
    if not _is_canary and not _canary_checked:
        asyncio.ensure_future(run_startup_canary_check())

    key_pool = _shuffled_key_pool()
    if not key_pool:
        return []

    payload = None
    last_status = None
    for api_key in key_pool:
        items, status_code = await _call_with_key(query, num, api_key)
        if items is not None:
            payload = items
            break
        last_status = status_code
        if status_code in _KEY_FAILURE_RETRY_STATUSES and len(key_pool) > 1:
            continue
        break

    if payload is None:
        if last_status is not None:
            logger.warning(
                "google custom search all keys exhausted status=%s query=%r keys_tried=%d",
                last_status, query, len(key_pool),
            )
        return []

    if not _is_canary:
        await _register_quota_usage(quota_guard)

    return payload


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