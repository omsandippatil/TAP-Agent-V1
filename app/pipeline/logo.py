from urllib.parse import urlparse

from app.pipeline import scraper
from app.pipeline.search_budget import SearchBudget
from app.pipeline.utils import get_session

FAVICON_ENDPOINT = "https://www.google.com/s2/favicons"
FAVICON_SIZE = 128


def favicon_url_for_domain(domain: str) -> str:
    if not domain:
        return ""
    return f"{FAVICON_ENDPOINT}?domain={domain}&sz={FAVICON_SIZE}"


def _homepage_confirms_company(domain: str, company: str) -> bool:
    try:
        response = get_session().get(f"https://{domain}", timeout=8)
        if not response.ok:
            return False
        return scraper.mentions_company(company, response.text[:20000])
    except Exception:
        return False


def _domain_from_source_url(url: str) -> str:
    if not url:
        return ""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


async def resolve_company_logo_url(company: str, search_cfg: dict, quota_guard=None, sources: list | None = None,
                                    budget: SearchBudget | None = None) -> str:
    for source in sources or []:
        candidate = source.get("domain", "") or _domain_from_source_url(source.get("url", ""))
        if candidate and _homepage_confirms_company(candidate, company):
            return favicon_url_for_domain(candidate)

    # discover_company_domains requires a real SearchBudget as its third positional arg.
    # Build a small dedicated budget for logo resolution rather than reusing quota_guard,
    # which is a different type (an API quota tracker, not a SearchBudget).
    resolved_budget = budget or SearchBudget(company, max_google_queries=2, max_ddgs_queries=2)
    domains = await scraper.discover_company_domains(company, search_cfg, resolved_budget, quota_guard)
    for candidate in domains:
        if _homepage_confirms_company(candidate, company):
            return favicon_url_for_domain(candidate)

    return ""