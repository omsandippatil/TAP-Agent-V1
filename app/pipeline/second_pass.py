import logging
import re
import time
from urllib.parse import urlparse

from app.pipeline.search_budget import SearchBudget
from app.pipeline.scraper import (
    AGGREGATOR_DOMAINS,
    fetch_page_text,
    fetch_pdf_text,
    search_web,
    mentions_company,
    is_csr_relevant,
    has_india_or_education_signal,
    find_india_location_mentions,
    is_plausible_entity_name,
    _extract_entities_from_lines,
    NAMED_INITIATIVE_PATTERN,
    NAMED_NGO_PATTERN,
    RELATED_ENTITY_NAME_PATTERN,
)
from app.pipeline.utils import make_source, normalize_block_text

logger = logging.getLogger("tap.second_pass")

SECOND_PASS_TEXT_LENGTH_FLOOR = 500
SECOND_PASS_THIN_EVIDENCE_FLOOR = 900
SECOND_PASS_MAX_GOOGLE_QUERIES = 10
SECOND_PASS_DEADLINE_SECONDS = 55.0
SECOND_PASS_MAX_NAMED_FOLLOWUPS = 4
SECOND_PASS_MAX_STATE_FOLLOWUPS = 3

_ENTITY_PATTERNS = (NAMED_INITIATIVE_PATTERN, NAMED_NGO_PATTERN, RELATED_ENTITY_NAME_PATTERN)

EDUCATION_RECOVERY_QUERIES = [
    '"{company}" CSR education OR STEM OR "government school" students beneficiaries India',
    '"{company}" CSR NGO partner education skilling program named India',
    '"{company}" CSR "students" "schools" impact program press release',
]

UNREADABLE_DOC_FOLLOWUP_QUERIES = [
    '"{company}" CSR programme name education beneficiaries press release',
    '"{company}" CSR partner NGO announcement India',
]

GENERIC_BROAD_FALLBACK_QUERY = '"{company}" CSR school STEM partnership'

_DOMAIN_LABEL_PATTERN = re.compile(r"^[a-z0-9.-]+\.[a-z]{2,}$", re.IGNORECASE)


def _discover_named_entities(sources):
    names = set()
    for source in sources or []:
        text = source.get("text", "")
        if not text:
            continue
        for name in _extract_entities_from_lines(text, _ENTITY_PATTERNS, is_plausible_entity_name):
            names.add(name)
    return names


def _discover_states(sources):
    states = set()
    for source in sources or []:
        for hit in find_india_location_mentions(source.get("text", "")):
            if hit["kind"] in ("state", "city"):
                states.add(hit["text"])
    return states


def _has_unreadable_document(sources):
    return any(
        s.get("fetch_method", "").endswith("unreadable") or "unreadable" in s.get("fetch_method", "")
        for s in (sources or [])
    )


def _is_usable_search_label(label):
    if not label or len(label) > 80:
        return False
    if label.startswith(("http://", "https://")):
        return False
    if bool(_DOMAIN_LABEL_PATTERN.match(label.strip())):
        return True
    return not re.search(r"[/?#]", label)


def _label_for_short_extract_query(source):
    domain = source.get("domain", "")
    if _is_usable_search_label(domain):
        return domain
    title = source.get("title", "")
    if _is_usable_search_label(title):
        return title
    return ""


def _has_substantive_education_source(sources):
    for source in sources or []:
        if source.get("status") != "FOUND":
            continue
        text = source.get("text", "")
        if not text or not has_india_or_education_signal(text):
            continue
        if len(text) >= SECOND_PASS_THIN_EVIDENCE_FLOOR:
            return True
    return False


def _has_education_evidence(sources):
    return _has_substantive_education_source(sources)


def should_run_second_pass(sources, coverage_pct=None, coverage_floor=45):
    short_extract = any(
        s.get("status") == "FOUND" and s.get("url")
        and len(s.get("text", "")) < SECOND_PASS_TEXT_LENGTH_FLOOR
        for s in (sources or [])
    )
    coverage_low = coverage_pct is not None and coverage_pct < coverage_floor
    missing_education = not _has_education_evidence(sources)
    unreadable_doc = _has_unreadable_document(sources)
    return short_extract or coverage_low or missing_education or unreadable_doc


async def _fetch_and_accept(candidate_url, relevance_check):
    text = await (
        fetch_pdf_text(candidate_url) if candidate_url.lower().endswith(".pdf")
        else fetch_page_text(candidate_url)
    )
    if not text or len(text) <= SECOND_PASS_TEXT_LENGTH_FLOOR:
        return None, text
    if not relevance_check(text):
        return None, text
    return text, text


async def run_second_pass_recovery(company, search_cfg, sources, registry=None, quota_guard=None,
                                    max_google_queries=SECOND_PASS_MAX_GOOGLE_QUERIES,
                                    deadline_seconds=SECOND_PASS_DEADLINE_SECONDS,
                                    budget: SearchBudget | None = None):
    budget = budget or SearchBudget(company, max_google_queries=max_google_queries)
    deadline = time.monotonic() + deadline_seconds
    recovered = []
    queries_run = []
    queries_skipped_no_label = 0

    def is_dead(url: str) -> bool:
        host = urlparse(url).netloc.lower()
        return budget.is_domain_dead(host) or budget.is_path_dead(url)

    short_extract_sources = [
        s for s in sources
        if s.get("status") == "FOUND" and s.get("url")
        and len(s.get("text", "")) < SECOND_PASS_TEXT_LENGTH_FLOOR
    ]
    for source in short_extract_sources:
        if time.monotonic() >= deadline:
            break
        label = _label_for_short_extract_query(source)
        if not label:
            queries_skipped_no_label += 1
            continue
        query = f'"{company}" "{label}" CSR'
        queries_run.append(query)
        results = await search_web(query, budget, max_results=5, quota_guard=quota_guard, category="second_pass")
        for result in results:
            candidate_url = result.get("href", "")
            if not candidate_url or candidate_url == source.get("url") or any(
                domain in candidate_url for domain in AGGREGATOR_DOMAINS
            ):
                continue
            if is_dead(candidate_url):
                continue
            text, raw_text = await _fetch_and_accept(
                candidate_url, lambda t: mentions_company(company, t),
            )
            if text is None:
                if not raw_text:
                    budget.mark_path_dead(candidate_url)
                continue
            new_source = make_source(
                "second_pass_recovery", 11, candidate_url, text, "FOUND", "second_pass_short_extract"
            )
            if registry is not None:
                registry.register_core_source(new_source)
            recovered.append(new_source)
            budget.mark_category_hit("second_pass")
            break

    unreadable_present = _has_unreadable_document(sources)
    if unreadable_present and _budget_has_room(budget, deadline):
        for query_template in UNREADABLE_DOC_FOLLOWUP_QUERIES:
            if time.monotonic() >= deadline or not budget.google_has_budget("second_pass"):
                break
            query = query_template.format(company=company)
            queries_run.append(query)
            results = await search_web(query, budget, max_results=6, quota_guard=quota_guard, category="second_pass")
            for result in results:
                url = result.get("href", "")
                title = result.get("title", "")
                body = result.get("body", "")
                if not url or any(domain in url for domain in AGGREGATOR_DOMAINS) or is_dead(url):
                    continue
                if not mentions_company(company, f"{title} {body}"):
                    continue
                text = await (fetch_pdf_text(url) if url.lower().endswith(".pdf") else fetch_page_text(url)) or body
                if not text:
                    budget.mark_path_dead(url)
                    continue
                if len(text) > SECOND_PASS_TEXT_LENGTH_FLOOR and is_csr_relevant(text):
                    new_source = make_source(
                        "second_pass_recovery", 11, url, text, "FOUND", "second_pass_unreadable_followup"
                    )
                    if registry is not None:
                        registry.register_core_source(new_source)
                    recovered.append(new_source)
                    budget.mark_category_hit("second_pass")

    if not _has_education_evidence(sources + recovered) and _budget_has_room(budget, deadline):
        for query_template in EDUCATION_RECOVERY_QUERIES:
            if time.monotonic() >= deadline or not budget.google_has_budget("second_pass"):
                break
            query = query_template.format(company=company)
            queries_run.append(query)
            results = await search_web(query, budget, max_results=6, quota_guard=quota_guard, category="second_pass")
            for result in results:
                url = result.get("href", "")
                title = result.get("title", "")
                body = result.get("body", "")
                if not url or any(domain in url for domain in AGGREGATOR_DOMAINS) or is_dead(url):
                    continue
                if not mentions_company(company, f"{title} {body}"):
                    continue
                text = await (fetch_pdf_text(url) if url.lower().endswith(".pdf") else fetch_page_text(url)) or body
                if not text:
                    budget.mark_path_dead(url)
                    continue
                if len(text) > SECOND_PASS_TEXT_LENGTH_FLOOR and has_india_or_education_signal(text):
                    new_source = make_source(
                        "second_pass_recovery", 11, url, text, "FOUND", "second_pass_education_recovery"
                    )
                    if registry is not None:
                        registry.register_core_source(new_source)
                    recovered.append(new_source)
                    budget.mark_category_hit("second_pass")
            if _has_education_evidence(sources + recovered):
                break

    named_entities = list(_discover_named_entities(sources))[:SECOND_PASS_MAX_NAMED_FOLLOWUPS]
    states = list(_discover_states(sources))[:SECOND_PASS_MAX_STATE_FOLLOWUPS]

    followup_queries = [f'"{name}" "{company}" (partnership OR funded OR implementing)' for name in named_entities]
    if states:
        followup_queries.append(f'"{company}" CSR ({" OR ".join(states)})')

    for query in followup_queries:
        if time.monotonic() >= deadline:
            break
        if not budget.google_has_budget("second_pass"):
            break
        queries_run.append(query)
        results = await search_web(query, budget, max_results=5, quota_guard=quota_guard, category="second_pass")
        for result in results:
            url = result.get("href", "")
            title = result.get("title", "")
            body = result.get("body", "")
            if not url or any(domain in url for domain in AGGREGATOR_DOMAINS):
                continue
            if is_dead(url):
                continue
            if not mentions_company(company, f"{title} {body}"):
                continue
            text = await (fetch_pdf_text(url) if url.lower().endswith(".pdf") else fetch_page_text(url)) or body
            if not text:
                budget.mark_path_dead(url)
                continue
            if len(text) > SECOND_PASS_TEXT_LENGTH_FLOOR and is_csr_relevant(text):
                new_source = make_source(
                    "second_pass_recovery", 11, url, text, "FOUND", "second_pass_followup"
                )
                if registry is not None:
                    registry.register_core_source(new_source)
                recovered.append(new_source)
                budget.mark_category_hit("second_pass")

    if not recovered and _budget_has_room(budget, deadline):
        query = GENERIC_BROAD_FALLBACK_QUERY.format(company=company)
        queries_run.append(query)
        results = await search_web(query, budget, max_results=5, quota_guard=quota_guard, category="second_pass")
        for result in results:
            url = result.get("href", "")
            title = result.get("title", "")
            body = result.get("body", "")
            if not url or any(domain in url for domain in AGGREGATOR_DOMAINS) or is_dead(url):
                continue
            if not mentions_company(company, f"{title} {body}"):
                continue
            text = await (fetch_pdf_text(url) if url.lower().endswith(".pdf") else fetch_page_text(url)) or body
            if text and len(text) > SECOND_PASS_TEXT_LENGTH_FLOOR and is_csr_relevant(text):
                new_source = make_source(
                    "second_pass_recovery", 11, url, text, "FOUND", "second_pass_broad_fallback"
                )
                if registry is not None:
                    registry.register_core_source(new_source)
                recovered.append(new_source)
                budget.mark_category_hit("second_pass")

    logger.info(
        "second_pass_recovery DONE company=%r short_extract_candidates=%d short_extract_skipped_no_label=%d "
        "unreadable_present=%s queries_run=%d recovered=%d",
        company, len(short_extract_sources), queries_skipped_no_label, unreadable_present,
        len(queries_run), len(recovered),
    )
    return recovered, queries_run


def _budget_has_room(budget, deadline):
    return time.monotonic() < deadline and budget.google_has_budget("second_pass")