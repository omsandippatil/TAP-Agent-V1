import logging
import time

from app.pipeline import scraper
from app.pipeline.search_budget import SearchBudget
from app.pipeline.source_registry import SourceRegistry
from app.pipeline.utils import make_source

logger = logging.getLogger("tap.directed_search")

DEFAULT_DEADLINE_SECONDS = 40.0
MAX_CANDIDATES_PER_DIRECTIVE = 2
MIN_TEXT_LENGTH = 200

PRIORITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def _normalize_priority(value: str) -> str:
    normalized = (value or "").strip().upper()
    return normalized if normalized in PRIORITY_ORDER else "MEDIUM"


def _filter_directives(directives: list[dict], mode: str, budget: SearchBudget,
                        medium_priority_enabled: bool, max_queries: int) -> list[dict]:
    filtered = []
    for directive in directives or []:
        if not isinstance(directive, dict):
            continue
        query = (directive.get("search_query") or "").strip()
        if not query:
            continue
        priority = _normalize_priority(directive.get("priority"))
        if priority == "HIGH":
            filtered.append(directive)
        elif priority == "MEDIUM":
            if medium_priority_enabled and budget.has_room_for_directed_search():
                filtered.append(directive)
        elif priority == "LOW":
            if mode == "deep":
                filtered.append(directive)

    filtered.sort(key=lambda d: PRIORITY_ORDER.get(_normalize_priority(d.get("priority")), 1))
    return filtered[:max_queries]


async def _fetch_best_candidates(company: str, query: str, budget: SearchBudget, quota_guard,
                                  deadline: float, category: str) -> list[tuple[str, str]]:
    results = await scraper.search_web(
        query, budget, max_results=6, quota_guard=quota_guard, category=category,
    )
    accepted: list[tuple[str, str]] = []
    for result in results:
        if len(accepted) >= MAX_CANDIDATES_PER_DIRECTIVE or time.monotonic() >= deadline:
            break
        url = result.get("href", "")
        title = result.get("title", "")
        body = result.get("body", "")
        if not url or any(domain in url for domain in scraper.AGGREGATOR_DOMAINS):
            continue
        if not scraper.mentions_company(company, f"{title} {body}"):
            continue
        is_pdf = url.lower().endswith(".pdf")
        text = await (scraper.fetch_pdf_text(url) if is_pdf else scraper.fetch_page_text(url)) or body
        if not text or len(text) < MIN_TEXT_LENGTH:
            continue
        if not scraper.mentions_company(company, text):
            continue
        if not scraper.is_csr_relevant(text):
            continue
        accepted.append((url, text))
    return accepted


async def run_directed_search(company: str, mode: str, directives: list[dict], budget: SearchBudget,
                               quota_guard=None, registry: SourceRegistry | None = None,
                               deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
                               max_queries: int = 4,
                               medium_priority_enabled: bool = True) -> tuple[list[dict], dict]:
    deadline = time.monotonic() + deadline_seconds
    filtered_directives = _filter_directives(directives, mode, budget, medium_priority_enabled, max_queries)

    new_sources: list[dict] = []
    fields_addressed: list[str] = []
    fields_still_empty: list[str] = []
    directives_attempted = 0

    for directive in filtered_directives:
        if time.monotonic() >= deadline:
            break
        query = (directive.get("search_query") or "").strip()
        target_field = (directive.get("target_field") or "").strip()
        if not query:
            continue

        directives_attempted += 1
        candidates = await _fetch_best_candidates(
            company, query, budget, quota_guard, deadline, category="second_pass",
        )

        if not candidates:
            if target_field:
                fields_still_empty.append(target_field)
            continue

        for url, text in candidates:
            source = make_source("directed_search", 11, url, text, "FOUND", "directed_search")
            source["target_field"] = target_field
            source["directive_question"] = directive.get("question", "")
            source["directive_priority"] = _normalize_priority(directive.get("priority"))
            if registry is not None:
                registry.register_core_source(source)
            new_sources.append(source)
            budget.mark_category_hit("second_pass")

        if target_field and target_field not in fields_addressed:
            fields_addressed.append(target_field)

    summary = {
        "directives_attempted": directives_attempted,
        "sources_recovered": len(new_sources),
        "fields_addressed": fields_addressed,
        "fields_still_empty": [f for f in fields_still_empty if f not in fields_addressed],
    }

    logger.info(
        "directed_search DONE company=%r mode=%s directives_attempted=%d sources_recovered=%d fields_addressed=%s",
        company, mode, directives_attempted, len(new_sources), fields_addressed,
    )

    return new_sources, summary