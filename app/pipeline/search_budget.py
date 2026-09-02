import asyncio
import logging
import time

logger = logging.getLogger("tap.search_budget")

DEFAULT_MAX_GOOGLE_QUERIES = 24
DEFAULT_MAX_DDGS_QUERIES = 8

CATEGORY_FLOORS_DEFAULT = {
    "partner_search": 6,
    "education_programme_search": 5,
    "people_search": 4,
}

DDGS_MIN_INTERVAL_SECONDS = 1.5

_DDGS_GLOBAL_LOCK = asyncio.Lock()
_ddgs_last_call_monotonic = 0.0


class SearchBudget:

    def __init__(self, company: str, max_google_queries: int = DEFAULT_MAX_GOOGLE_QUERIES,
                 max_ddgs_queries: int = DEFAULT_MAX_DDGS_QUERIES,
                 category_floors: dict[str, int] | None = None):
        self.company = company
        self.max_google_queries = max_google_queries
        self.max_ddgs_queries = max_ddgs_queries
        self.category_floors = dict(category_floors) if category_floors is not None else dict(CATEGORY_FLOORS_DEFAULT)
        self.google_queries_used = 0
        self.ddgs_queries_used = 0
        self.category_used: dict[str, int] = {}
        self.legal_entity_name_cache = None
        self.legal_entity_name_resolved = False

    def google_has_budget(self, category: str = "") -> bool:
        if self.google_queries_used >= self.max_google_queries:
            return False
        floor = self.category_floors.get(category, 0)
        if floor and self.category_used.get(category, 0) < floor:
            return True
        reserved_remaining = sum(
            max(0, floor - self.category_used.get(cat, 0))
            for cat, floor in self.category_floors.items()
            if cat != category
        )
        effective_ceiling = self.max_google_queries - reserved_remaining
        return self.google_queries_used < effective_ceiling

    def ddgs_has_budget(self) -> bool:
        return self.ddgs_queries_used < self.max_ddgs_queries

    def record_google_query(self, category: str = ""):
        self.google_queries_used += 1
        if category:
            self.category_used[category] = self.category_used.get(category, 0) + 1
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
            "category_used": dict(self.category_used),
            "category_floors": dict(self.category_floors),
        }


def ddgs_global_lock() -> asyncio.Lock:
    return _DDGS_GLOBAL_LOCK


async def ddgs_pace() -> None:
    global _ddgs_last_call_monotonic
    now = time.monotonic()
    elapsed = now - _ddgs_last_call_monotonic
    if elapsed < DDGS_MIN_INTERVAL_SECONDS:
        await asyncio.sleep(DDGS_MIN_INTERVAL_SECONDS - elapsed)
    _ddgs_last_call_monotonic = time.monotonic()