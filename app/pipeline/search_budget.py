import logging

logger = logging.getLogger("tap.search_budget")

DEFAULT_MAX_GOOGLE_QUERIES = 34
SCREEN_MAX_GOOGLE_QUERIES = 30
DEFAULT_MAX_EMPTY_STREAK = 2

CATEGORY_FLOORS_DEFAULT = {
    "csr_page": 4,
    "annual_report": 3,
    "partner_search": 2,
    "education_programme_search": 3,
    "people_search": 2,
    "mca_filing": 1,
    "cin": 1,
    "legal_entity": 1,
    "second_pass_unreadable_doc": 2,
}

CATEGORY_SUCCESS_TARGET_DEFAULT = {
    "csr_page": 1,
    "annual_report": 1,
    "partner_search": 2,
    "education_programme_search": 2,
    "people_search": 2,
    "mca_filing": 1,
    "cin": 1,
    "legal_entity": 1,
    "national_csr_portal": 1,
    "plans_search": 1,
    "sector_eligibility_search": 1,
    "multi_year_financials": 1,
    "entity_resolution": 2,
    "second_pass_unreadable_doc": 2,
}


def default_category_floors_for_mode(mode: str) -> dict[str, int]:
    return dict(CATEGORY_FLOORS_DEFAULT)


def default_category_success_targets_for_mode(mode: str) -> dict[str, int]:
    targets = dict(CATEGORY_SUCCESS_TARGET_DEFAULT)
    if mode == "screen":
        targets["education_programme_search"] = 1
    return targets


class SearchBudget:
    """Per-run Google query budget. Pass category_floors={} for ad-hoc budgets
    (second pass, directed search) so no categories are reserved."""

    def __init__(self, company: str, max_google_queries: int | None = None,
                 category_floors: dict[str, int] | None = None,
                 category_success_target: dict[str, int] | None = None,
                 max_empty_streak: int = DEFAULT_MAX_EMPTY_STREAK,
                 mode: str = "deep"):
        self.company = company
        self.mode = mode
        self.max_google_queries = max_google_queries if max_google_queries is not None else (
            SCREEN_MAX_GOOGLE_QUERIES if mode == "screen" else DEFAULT_MAX_GOOGLE_QUERIES
        )
        self.category_floors = (
            dict(category_floors) if category_floors is not None
            else default_category_floors_for_mode(mode)
        )
        self.category_success_target = (
            dict(category_success_target) if category_success_target is not None
            else default_category_success_targets_for_mode(mode)
        )
        self.max_empty_streak = max_empty_streak
        self.google_queries_used = 0
        self.category_used: dict[str, int] = {}
        self.category_hits: dict[str, int] = {}
        self.category_empty_streak: dict[str, int] = {}
        self.legal_entity_name_cache = None
        self.legal_entity_name_resolved = False
        self.related_entities_cache = None
        self.related_entities_resolved = False
        self.resolved_domains: list[str] = []
        self.dead_domains: set[str] = set()
        self.dead_paths: set[str] = set()
        self.guessed_path_miss_count: dict[str, int] = {}
        self.quota_exhausted_globally = False

    def set_resolved_domains(self, domains: list[str]):
        self.resolved_domains = list(dict.fromkeys([*self.resolved_domains, *(d for d in (domains or []) if d)]))

    def category_is_satisfied(self, category: str) -> bool:
        if not category:
            return False
        target = self.category_success_target.get(category)
        if target and self.category_hits.get(category, 0) >= target:
            return True
        return self.category_empty_streak.get(category, 0) >= self.max_empty_streak

    def _other_reserved_remaining(self, exclude_category: str = "") -> int:
        return sum(
            max(0, floor - self.category_used.get(cat, 0))
            for cat, floor in self.category_floors.items()
            if cat != exclude_category and not self.category_is_satisfied(cat)
        )

    def google_has_budget(self, category: str = "") -> bool:
        if self.quota_exhausted_globally or self.google_queries_used >= self.max_google_queries:
            return False
        if category and self.category_is_satisfied(category):
            return False
        floor = self.category_floors.get(category, 0)
        if floor and self.category_used.get(category, 0) < floor:
            return True
        ceiling = self.max_google_queries - self._other_reserved_remaining(exclude_category=category)
        return self.google_queries_used < ceiling

    def has_room_for_directed_search(self) -> bool:
        return not any(
            floor and self.category_used.get(cat, 0) < floor and not self.category_is_satisfied(cat)
            for cat, floor in self.category_floors.items()
        )

    def record_google_query(self, category: str = ""):
        self.google_queries_used += 1
        if category:
            self.category_used[category] = self.category_used.get(category, 0) + 1
        if self.google_queries_used >= self.max_google_queries:
            logger.info("google query budget exhausted company=%r used=%d breakdown=%s",
                        self.company, self.google_queries_used, self.category_used)

    def record_query_results(self, category: str, result_count: int):
        if category:
            self.category_empty_streak[category] = (
                self.category_empty_streak.get(category, 0) + 1 if result_count == 0 else 0
            )

    def mark_category_hit(self, category: str, count: int = 1):
        if category:
            self.category_hits[category] = self.category_hits.get(category, 0) + count
            self.category_empty_streak[category] = 0

    def mark_quota_exhausted_globally(self):
        if not self.quota_exhausted_globally:
            self.quota_exhausted_globally = True
            logger.warning("search budget quota_exhausted_globally company=%r used=%d",
                           self.company, self.google_queries_used)

    def mark_domain_dead(self, domain: str, reason: str = ""):
        domain = (domain or "").lower()
        if domain and domain not in self.dead_domains:
            self.dead_domains.add(domain)
            logger.info("domain marked dead company=%r domain=%s reason=%s", self.company, domain, reason)

    def is_domain_dead(self, domain: str) -> bool:
        return bool(domain) and domain.lower() in self.dead_domains

    def mark_path_dead(self, url: str):
        if url:
            self.dead_paths.add(url)

    def is_path_dead(self, url: str) -> bool:
        return url in self.dead_paths

    def record_guessed_path_miss(self, host: str) -> int:
        if not host:
            return 0
        self.guessed_path_miss_count[host] = self.guessed_path_miss_count.get(host, 0) + 1
        return self.guessed_path_miss_count[host]

    def summary(self) -> dict:
        return {
            "google_queries_used": self.google_queries_used,
            "google_budget": self.max_google_queries,
            "category_used": dict(self.category_used),
            "category_hits": dict(self.category_hits),
            "quota_exhausted_globally": self.quota_exhausted_globally,
        }