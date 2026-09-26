import logging

logger = logging.getLogger("tap.search_budget")

DEFAULT_MAX_GOOGLE_QUERIES = 30

CATEGORY_FLOORS_DEFAULT = {
    "csr_page": 4,
    "annual_report": 4,
    "partner_search": 4,
    "education_programme_search": 5,
    "people_search": 3,
    "mca_filing": 2,
    "cin": 1,
    "legal_entity": 1,
    "second_pass": 4,
}


class SearchBudget:

    def __init__(self, company: str, max_google_queries: int = DEFAULT_MAX_GOOGLE_QUERIES,
                 category_floors: dict[str, int] | None = None):
        self.company = company
        self.max_google_queries = max_google_queries
        self.category_floors = dict(category_floors) if category_floors is not None else dict(CATEGORY_FLOORS_DEFAULT)
        self.google_queries_used = 0
        self.category_used: dict[str, int] = {}
        self.legal_entity_name_cache = None
        self.legal_entity_name_resolved = False
        self.related_entities_cache = None
        self.related_entities_resolved = False
        self.dead_domains: set[str] = set()
        self.dead_paths: set[str] = set()

    def google_has_budget(self, category: str = "") -> bool:
        if self.google_queries_used >= self.max_google_queries:
            return False
        used_in_category = self.category_used.get(category, 0)
        floor = self.category_floors.get(category, 0)
        if floor and used_in_category < floor:
            return True
        other_reserved_remaining = sum(
            max(0, cat_floor - self.category_used.get(cat, 0))
            for cat, cat_floor in self.category_floors.items()
            if cat != category
        )
        effective_ceiling = self.max_google_queries - other_reserved_remaining
        return self.google_queries_used < effective_ceiling

    def record_google_query(self, category: str = ""):
        self.google_queries_used += 1
        if category:
            self.category_used[category] = self.category_used.get(category, 0) + 1
        if self.google_queries_used >= self.max_google_queries:
            logger.info(
                "google query budget exhausted company=%r used=%d category_breakdown=%s",
                self.company, self.google_queries_used, self.category_used,
            )

    def ddgs_has_budget(self) -> bool:
        return False

    def record_ddgs_query(self):
        return

    def mark_domain_dead(self, domain: str, reason: str = ""):
        if not domain:
            return
        domain = domain.lower()
        if domain not in self.dead_domains:
            self.dead_domains.add(domain)
            logger.info(
                "domain marked dead for this run company=%r domain=%s reason=%s",
                self.company, domain, reason,
            )

    def is_domain_dead(self, domain: str) -> bool:
        return bool(domain) and domain.lower() in self.dead_domains

    def mark_path_dead(self, url: str):
        if url:
            self.dead_paths.add(url)

    def is_path_dead(self, url: str) -> bool:
        return url in self.dead_paths

    def summary(self) -> dict:
        return {
            "google_queries_used": self.google_queries_used,
            "google_budget": self.max_google_queries,
            "category_used": dict(self.category_used),
            "category_floors": dict(self.category_floors),
            "dead_domains": len(self.dead_domains),
            "dead_paths": len(self.dead_paths),
        }