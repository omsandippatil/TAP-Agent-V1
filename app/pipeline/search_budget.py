import asyncio
import logging
import time

logger = logging.getLogger("tap.search_budget")

DEFAULT_MAX_GOOGLE_QUERIES = 16
DEFAULT_MAX_DDGS_QUERIES = 8

# Queries tagged with one of these categories can never be blocked once the reserve
# floor is in effect — this stops early, lower-value steps (domain/CIN/legal-name
# resolution) from consuming the entire budget before partners/people/education ever run.
RESERVED_CATEGORIES = frozenset({"partner_search", "education_programme_search", "people_search"})
RESERVED_GOOGLE_FLOOR = 4

DDGS_MIN_INTERVAL_SECONDS = 1.5

_DDGS_GLOBAL_LOCK = asyncio.Lock()
_ddgs_last_call_monotonic = 0.0


class SearchBudget:

    def __init__(self, company: str, max_google_queries: int = DEFAULT_MAX_GOOGLE_QUERIES,
                 max_ddgs_queries: int = DEFAULT_MAX_DDGS_QUERIES,
                 reserved_google_floor: int = RESERVED_GOOGLE_FLOOR):
        self.company = company
        self.max_google_queries = max_google_queries
        self.max_ddgs_queries = max_ddgs_queries
        self.reserved_google_floor = min(reserved_google_floor, max_google_queries)
        self.google_queries_used = 0
        self.ddgs_queries_used = 0
        self.legal_entity_name_cache = None
        self.legal_entity_name_resolved = False

    def google_has_budget(self, category: str = "") -> bool:
        if self.google_queries_used >= self.max_google_queries:
            return False
        if category in RESERVED_CATEGORIES:
            return True
        # Non-reserved callers must leave the reserved floor untouched for the
        # high-value downstream categories (partners/people/education).
        effective_ceiling = self.max_google_queries - self.reserved_google_floor
        return self.google_queries_used < effective_ceiling

    def ddgs_has_budget(self) -> bool:
        return self.ddgs_queries_used < self.max_ddgs_queries

    def record_google_query(self):
        self.google_queries_used += 1
        if self.google_queries_used == self.max_google_queries:
            logger.info("google query budget exhausted company=%r used=%d", self.company, self.google_queries_used)

    def record_ddgs_query(self):
        self.ddgs_queries_used += 1
        if self.ddgs_queries_used == self.max_ddgs_queries:
            logger.info("ddgs query budget exhausted company=%r used=%d", self.company, self.ddgs_queries_used)

    def summary(self) -> dict:
        return {
            "google_queries_used": self.google_queries_used,
            "google_budget": self.max_google_queries,
            "ddgs_queries_used": self.ddgs_queries_used,
            "ddgs_budget": self.max_ddgs_queries,
        }


def ddgs_global_lock() -> asyncio.Lock:
    return _DDGS_GLOBAL_LOCK


async def ddgs_pace() -> None:
    """Enforce a minimum spacing between DDGS calls across the whole process, since
    DDGS soft-blocks/rate-limits aggressively per source IP. Call this while holding
    ddgs_global_lock(), right before issuing a request.
    """
    global _ddgs_last_call_monotonic
    now = time.monotonic()
    elapsed = now - _ddgs_last_call_monotonic
    if elapsed < DDGS_MIN_INTERVAL_SECONDS:
        await asyncio.sleep(DDGS_MIN_INTERVAL_SECONDS - elapsed)
    _ddgs_last_call_monotonic = time.monotonic()