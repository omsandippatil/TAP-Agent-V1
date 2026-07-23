import asyncio
import logging

logger = logging.getLogger("tap.search_budget")

DEFAULT_MAX_GOOGLE_QUERIES = 12
DEFAULT_MAX_DDGS_QUERIES = 6

_DDGS_GLOBAL_LOCK = asyncio.Lock()


class SearchBudget:

    def __init__(self, company: str, max_google_queries: int = DEFAULT_MAX_GOOGLE_QUERIES,
                 max_ddgs_queries: int = DEFAULT_MAX_DDGS_QUERIES):
        self.company = company
        self.max_google_queries = max_google_queries
        self.max_ddgs_queries = max_ddgs_queries
        self.google_queries_used = 0
        self.ddgs_queries_used = 0
        self.legal_entity_name_cache = None
        self.legal_entity_name_resolved = False

    def google_has_budget(self) -> bool:
        return self.google_queries_used < self.max_google_queries

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