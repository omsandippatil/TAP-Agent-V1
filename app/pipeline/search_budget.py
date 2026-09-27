import logging

logger = logging.getLogger("tap.search_budget")

DEFAULT_MAX_GOOGLE_QUERIES = 34
SCREEN_MAX_GOOGLE_QUERIES = 30

SECOND_PASS_CATEGORIES = (
    "second_pass_short_extract",
    "second_pass_unreadable_doc",
    "second_pass_education",
    "second_pass_named_entity",
    "second_pass_broad_fallback",
    "second_pass_chained_followup",
)

DIRECTED_SEARCH_CATEGORY_PREFIX = "directed_search_field"

CATEGORY_FLOORS_DEFAULT = {
    "csr_page": 4,
    "annual_report": 3,
    "partner_search": 2,
    "education_programme_search": 3,
    "people_search": 2,
    "mca_filing": 1,
    "cin": 1,
    "legal_entity": 1,
    "second_pass_short_extract": 2,
    "second_pass_unreadable_doc": 2,
    "second_pass_education": 3,
    "second_pass_named_entity": 2,
    "second_pass_broad_fallback": 1,
    "second_pass_chained_followup": 3,
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
    "second_pass_short_extract": 2,
    "second_pass_unreadable_doc": 2,
    "second_pass_education": 1,
    "second_pass_named_entity": 2,
    "second_pass_broad_fallback": 1,
    "second_pass_chained_followup": 3,
}

DEFAULT_MAX_EMPTY_STREAK = 2


def default_category_floors_for_mode(mode: str) -> dict[str, int]:
    floors = dict(CATEGORY_FLOORS_DEFAULT)
    if mode == "screen":
        floors["second_pass_education"] = 2
        floors["second_pass_chained_followup"] = 2
    return floors


def default_category_success_targets_for_mode(mode: str) -> dict[str, int]:
    targets = dict(CATEGORY_SUCCESS_TARGET_DEFAULT)
    if mode == "screen":
        targets["education_programme_search"] = 1
    return targets


def directed_search_category_for_field(target_field: str, fallback_index: int) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in (target_field or "").strip().lower())
    normalized = normalized.strip("_") or f"unlabelled_{fallback_index}"
    return f"{DIRECTED_SEARCH_CATEGORY_PREFIX}_{normalized}"


class SearchBudget:

    def __init__(self, company: str, max_google_queries: int | None = None,
                 category_floors: dict[str, int] | None = None,
                 category_success_target: dict[str, int] | None = None,
                 max_empty_streak: int = DEFAULT_MAX_EMPTY_STREAK,
                 mode: str = "deep"):
        self.company = company
        self.mode = mode
        if max_google_queries is not None:
            self.max_google_queries = max_google_queries
        else:
            self.max_google_queries = SCREEN_MAX_GOOGLE_QUERIES if mode == "screen" else DEFAULT_MAX_GOOGLE_QUERIES
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
        self.category_closed: set[str] = set()
        self.legal_entity_name_cache = None
        self.legal_entity_name_resolved = False
        self.related_entities_cache = None
        self.related_entities_resolved = False
        self.resolved_domains: list[str] = []
        self.dead_domains: set[str] = set()
        self.dead_paths: set[str] = set()
        self.guessed_path_miss_count: dict[str, int] = {}
        self.quota_exhausted_globally = False

    def register_dynamic_category(self, category: str, floor: int = 1, success_target: int = 1) -> None:
        if not category:
            return
        self.category_floors.setdefault(category, floor)
        self.category_success_target.setdefault(category, success_target)

    def set_resolved_domains(self, domains: list[str]):
        merged = list(dict.fromkeys([*self.resolved_domains, *(d for d in (domains or []) if d)]))
        self.resolved_domains = merged

    def category_is_satisfied(self, category: str) -> bool:
        if not category:
            return False
        if category in self.category_closed:
            return True
        target = self.category_success_target.get(category)
        if target and self.category_hits.get(category, 0) >= target:
            return True
        if self.category_empty_streak.get(category, 0) >= self.max_empty_streak:
            return True
        return False

    def _other_reserved_remaining(self, exclude_category: str = "") -> int:
        return sum(
            max(0, cat_floor - self.category_used.get(cat, 0))
            for cat, cat_floor in self.category_floors.items()
            if cat != exclude_category and not self.category_is_satisfied(cat)
        )

    def google_has_budget(self, category: str = "") -> bool:
        if self.quota_exhausted_globally:
            return False
        if self.google_queries_used >= self.max_google_queries:
            return False
        if category and self.category_is_satisfied(category):
            return False
        used_in_category = self.category_used.get(category, 0)
        floor = self.category_floors.get(category, 0)
        if floor and used_in_category < floor:
            return True
        other_reserved_remaining = self._other_reserved_remaining(exclude_category=category)
        effective_ceiling = self.max_google_queries - other_reserved_remaining
        return self.google_queries_used < effective_ceiling

    def has_room_for_directed_search(self) -> bool:
        for category, floor in self.category_floors.items():
            if floor and self.category_used.get(category, 0) < floor and not self.category_is_satisfied(category):
                return False
        return True

    def record_google_query(self, category: str = ""):
        self.google_queries_used += 1
        if category:
            self.category_used[category] = self.category_used.get(category, 0) + 1
        if self.google_queries_used >= self.max_google_queries:
            logger.info(
                "google query budget exhausted company=%r used=%d category_breakdown=%s",
                self.company, self.google_queries_used, self.category_used,
            )

    def record_query_results(self, category: str, result_count: int):
        if not category:
            return
        if result_count == 0:
            self.category_empty_streak[category] = self.category_empty_streak.get(category, 0) + 1
        else:
            self.category_empty_streak[category] = 0

    def mark_category_hit(self, category: str, count: int = 1):
        if not category:
            return
        self.category_hits[category] = self.category_hits.get(category, 0) + count
        self.category_empty_streak[category] = 0

    def close_category(self, category: str):
        if category:
            self.category_closed.add(category)

    def mark_quota_exhausted_globally(self):
        if not self.quota_exhausted_globally:
            self.quota_exhausted_globally = True
            logger.warning(
                "search budget marking quota_exhausted_globally company=%r queries_used=%d",
                self.company, self.google_queries_used,
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
            "category_floors": dict(self.category_floors),
            "resolved_domains": list(self.resolved_domains),
            "dead_domains": len(self.dead_domains),
            "dead_paths": len(self.dead_paths),
            "quota_exhausted_globally": self.quota_exhausted_globally,
        }