import asyncio
import logging
import re
import time
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from app.pipeline import google_search
from app.pipeline.people_parser import parse_linkedin_hit
from app.pipeline.search_budget import SearchBudget, ddgs_global_lock
from app.pipeline.source_registry import SourceRegistry
from app.pipeline.utils import clean_text, extract_main_text, get_session, make_source

logger = logging.getLogger("tap.scraper")

GENERIC_COMPANY_TOKENS = {
    "india", "limited", "ltd", "private", "pvt", "the", "and", "of",
    "company", "corp", "corporation", "inc", "group", "technologies",
    "solutions", "services", "international", "holdings", "enterprises",
    "industries", "systems", "global",
}

AGGREGATOR_DOMAINS = (
    "youtube.", "twitter.", "x.com", "facebook.", "instagram.", "linkedin.",
    "wikipedia.", "glassdoor.", "indeed.", "crunchbase.", "bloomberg.",
    "zaubacorp", "tofler.", "justdial.", "indiamart.", "ambitionbox.",
    "moneycontrol.", "economictimes.", "livemint.", "reuters.",
    "apkpure.", "h1bgrader.", "quora.", "reddit.", "pinterest.",
    "medium.com", "slideshare.", "scribd.", "vimeo.", "tiktok.",
    "naukri.", "shine.com", "timesjobs.", "monsterindia.", "tracxn.",
)

BLOCKED_403_DOMAINS = (
    "zaubacorp.", "tracxn.",
)

OFFICIAL_GOV_DOMAINS = (
    "mca.gov.in", "csr.gov.in", "nic.in", "india.gov.in", "meity.gov.in",
    "pib.gov.in", "sebi.gov.in", "rbi.org.in",
)

CSR_LINK_PATTERN = re.compile(
    r"(csr|corporate[\s_-]?social|social[\s_-]?responsib|sustainab|esg|"
    r"social[\s_-]?impact|citizenship|community[\s_-]?(initiativ|develop|engag|invest)|"
    r"responsible[\s_-]?business|foundation|giving[\s_-]?back|annual[\s_-]?report|"
    r"investor[\s_-]?relation|philanthrop|impact[\s_-]?report|esg[\s_-]?report|"
    r"corporate[\s_-]?responsibilit)",
    re.IGNORECASE,
)

NEGATIVE_LINK_PATTERN = re.compile(
    r"(career|job|vacanc|recruit|login|sign-?in|privacy|cookie|terms|disclaimer|"
    r"sitemap|contact-?us|unsubscribe|logout|register|password)",
    re.IGNORECASE,
)

CSR_PAGE_PATHS = [
    "/csr", "/corporate-social-responsibility", "/sustainability",
    "/social-responsibility", "/esg", "/corporate-responsibility",
    "/about/csr", "/about-us/csr", "/company/csr", "/in/en/about/csr",
    "/india/csr", "/en/about/sustainability", "/about/sustainability",
    "/social-impact", "/corporate-citizenship", "/en/sustainability",
    "/sustainability/csr", "/about/corporate-responsibility",
    "/investors/annual-reports", "/investor-relations/annual-reports",
    "/csr-initiatives", "/csr-activities", "/csr-policy",
    "/about-us/corporate-social-responsibility", "/community",
    "/impact", "/responsibility", "/our-impact", "/csr-in-india",
    "/india/about/csr", "/en-in/csr", "/en-in/sustainability",
]

CSR_KEYWORDS = [
    "csr", "corporate social", "philanthrop", "social responsibility",
    "schedule vii", "csr spend", "csr expenditure", "csr budget",
    "csr obligation", "csr fund", "community investment", "esg report",
    "sustainability report", "impact report", "csr committee",
]

EDUCATION_KEYWORDS = [
    "education", "school", "skilling", "skill development", "stem",
    "digital literacy", "coding", "21st century skills", "21st-century skills",
    "learning", "curriculum", "classroom", "student", "literacy",
]

CURRENCY_FIGURE_PATTERN = re.compile(
    r"(?:(?:rs\.?|inr|₹)\s?[\d,]+(?:\.\d+)?\s?(?:crore|cr\.?|lakh|lac|million|mn|billion|bn|thousand)?"
    r"|[\d,]+(?:\.\d+)?\s?(?:crore|cr\.?|lakh|lac)\b)",
    re.IGNORECASE,
)

INDIA_STATE_PATTERN = re.compile(
    r"\b(maharashtra|karnataka|tamil\s*nadu|gujarat|rajasthan|uttar\s*pradesh|"
    r"west\s*bengal|telangana|kerala|punjab|haryana|bihar|odisha|orissa|assam|goa|"
    r"jharkhand|chhattisgarh|uttarakhand|himachal\s*pradesh|andhra\s*pradesh|"
    r"madhya\s*pradesh|manipur|meghalaya|mizoram|nagaland|sikkim|tripura|"
    r"arunachal\s*pradesh|jammu\s*(?:and|&)\s*kashmir|ladakh|delhi|puducherry|"
    r"chandigarh|andaman|lakshadweep|dadra|daman|diu)\b",
    re.IGNORECASE,
)

INDIA_CITY_PATTERN = re.compile(
    r"\b(mumbai|bombay|delhi|new\s*delhi|bengaluru|bangalore|chennai|madras|"
    r"kolkata|calcutta|hyderabad|pune|ahmedabad|surat|jaipur|lucknow|kanpur|"
    r"nagpur|indore|thane|bhopal|visakhapatnam|patna|vadodara|ghaziabad|"
    r"ludhiana|agra|nashik|faridabad|meerut|rajkot|kalyan|vasai|varanasi|"
    r"srinagar|aurangabad|dhanbad|amritsar|navi\s*mumbai|allahabad|prayagraj|"
    r"ranchi|howrah|coimbatore|jabalpur|gwalior|vijayawada|jodhpur|madurai|"
    r"raipur|kota|guwahati|chandigarh|solapur|hubli|mysore|mysuru|"
    r"tiruchirappalli|trichy|bareilly|aligarh|gurgaon|gurugram|noida|"
    r"moradabad|jalandhar|bhubaneswar|salem|warangal|thiruvananthapuram|"
    r"trivandrum|kochi|cochin|dehradun|shimla|panaji|panjim|imphal|shillong|"
    r"gangtok|itanagar|agartala|kohima|aizawl)\b",
    re.IGNORECASE,
)

INDIA_COUNTRY_PATTERN = re.compile(r"\bindia\b|\bbharat\b", re.IGNORECASE)

NEGATIVE_LOCATION_PATTERN = re.compile(
    r"\b(united\s*states|u\.?s\.?a?\.?|united\s*kingdom|u\.?k\.?|australia|canada|"
    r"singapore|germany|france|netherlands|u\.?a\.?e\.?|dubai|abu\s*dhabi|"
    r"hong\s*kong|(?<!south )china|japan|south\s*africa|new\s*zealand|ireland|spain|italy|"
    r"switzerland|sweden|norway|denmark|brazil|mexico)\b",
    re.IGNORECASE,
)

LINKEDIN_PROFILE_PATTERN = re.compile(
    r"^https?://([a-z]{2,3}\.)?linkedin\.com/in/[^/?#]+/?(?:[?#].*)?$", re.IGNORECASE
)

CIN_PATTERN = re.compile(r"\b[LUlu]\d{5}[A-Za-z]{2}\d{4}[A-Za-z]{3}\d{6}\b")

INDIA_LEGAL_ENTITY_PATTERN = re.compile(
    r"\b([A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*){0,6}\s+India\s+"
    r"(?:Private\s+Limited|Pvt\.?\s+Ltd\.?|Limited|Ltd\.?)|"
    r"[A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*){0,6}\s+(?:Technology|Technologies|Services|"
    r"Solutions|Software|Systems)\s+India\s+(?:Private\s+Limited|Pvt\.?\s+Ltd\.?|Limited|Ltd\.?)|"
    r"[A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*){0,6}\s+India\s+"
    r"(?:Foundation|Trust|Chapter))\b"
)

CURRENCY_NEAR_INDIA_WINDOW_CHARS = 200

CURRENT_FY_LABEL = "FY2025-26"
PRIOR_FY_LABELS = ["FY2024-25", "FY2023-24", "2024-25", "2023-24", "FY2022-23"]

CSR_PAGE_QUERIES = [
    'site:{d} CSR OR "corporate social responsibility" policy',
    '"{c}" "corporate social responsibility" report annual site:{d}',
]

CSR_PAGE_FALLBACK_QUERIES = [
    '"{c}" "CSR policy" filetype:pdf India',
    '"{c}" sustainability report India filetype:pdf',
]

MCA_CIN_QUERIES = [
    '"{c}" CIN site:mca.gov.in',
    '"{c}" "corporate identification number" India',
]

MCA_ENTITY_CIN_QUERIES = [
    '"{legal_name}" CIN site:mca.gov.in',
]

MCA_FILING_QUERIES = [
    '"{c}" "Form CSR-2" India filetype:pdf',
    '"{c}" MCA annual filing CSR India',
]

MCA_ENTITY_FILING_QUERIES = [
    '"{legal_name}" "Form CSR-2" filetype:pdf',
]

NATIONAL_CSR_PORTAL_QUERIES = [
    'site:csr.gov.in "{c}"',
]

LEGAL_ENTITY_RESOLUTION_QUERIES = [
    '"{c}" India "Private Limited" CIN MCA',
    '"{c}" India Pvt Ltd registered company name MCA',
]

ANNUAL_REPORT_QUERIES = [
    '"{c}" "annual report" {fy} India CSR crore filetype:pdf',
    '"{c}" "business responsibility and sustainability report" India filetype:pdf',
]

ANNUAL_REPORT_ENTITY_QUERIES = [
    '"{legal_name}" "annual report" {fy} CSR crore filetype:pdf',
]

CSR_SPEND_QUERIES = [
    '"{c}" "CSR expenditure" crore India {fy}',
]

CSR_SPEND_ENTITY_QUERIES = [
    '"{legal_name}" "CSR expenditure" crore {fy}',
]

PARTNER_QUERIES = [
    '"{c}" CSR "implementation partner" OR "implementing partner" India NGO named',
    '"{c}" foundation "grant recipients" OR "funded organisations" India CSR named',
]

PLAN_QUERIES = [
    '"{c}" CSR "partnered with" OR "partnership with" education India announced',
]

RFP_QUERIES = [
    '"{c}" CSR "request for proposal" OR "call for proposals" India',
]

LINKEDIN_PEOPLE_QUERIES = [
    'site:linkedin.com/in "{c}" "head of CSR" OR "CSR head"',
    'site:linkedin.com/in "{c}" sustainability head OR director India',
]

LINKEDIN_PEOPLE_GLOBAL_FALLBACK_QUERIES = [
    'site:linkedin.com/in "{c}" "head of sustainability" OR "chief sustainability officer"',
]

EDUCATION_PROGRAMME_QUERIES = [
    '"{c}" CSR "digital literacy" OR STEM OR coding OR skilling India students',
    '"{c}" "21st century skills" OR "21st-century skills" India CSR',
]

SECTOR_QUERIES = [
    '"{c}" India sector industry business overview annual report',
]

GROUP_FOUNDATION_QUERIES = [
    '"{c}" "group foundation" CSR India',
]

FOLLOWUP_QUERY_TEMPLATES = {
    "education_programme": [
        '"{c}" education OR skilling OR STEM programme India named',
    ],
    "csr_budget": [
        '"{c}" "CSR expenditure" OR "amount spent" crore India annual report',
    ],
    "decision_maker": [
        'site:linkedin.com/in "{c}" CSR OR sustainability OR ESG head India',
    ],
    "ngo_partner": [
        '"{c}" CSR "implementation partner" OR "implementing partner" education named',
    ],
    "csr_policy": [
        '"{c}" "CSR policy" OR "CSR annual report" India filetype:pdf',
    ],
}

MAX_PAGE_TEXT_CHARS = 6000
MAX_PDF_TEXT_CHARS = 10000
MAX_PDF_PAGES = 15
FINANCIAL_PDF_SCAN_PAGES = 25
CANDIDATE_EVAL_LIMIT = 2
MIN_ACCEPT_SCORE = 6
STRONG_ACCEPT_SCORE = 10

PAGE_FETCH_TIMEOUT_SECONDS = 6
PDF_FETCH_TIMEOUT_SECONDS = 8
HOMEPAGE_FETCH_TIMEOUT_SECONDS = 5
DDGS_TOTAL_BUDGET_SECONDS = 4
DDGS_BACKENDS = ("duckduckgo",)
SEARCH_TASK_TIMEOUT_SECONDS = 6
FETCH_TASK_TIMEOUT_SECONDS = 8
SOURCE_DEADLINE_SECONDS = 12
FOLLOWUP_DEADLINE_SECONDS = 10
CONCURRENT_FETCH_LIMIT = 4

_FETCH_SEMAPHORE = asyncio.Semaphore(CONCURRENT_FETCH_LIMIT)


def pdf_is_csr_relevant(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    csr_hits = sum(1 for kw in CSR_KEYWORDS if kw in lowered)
    return csr_hits >= 1 or is_csr_relevant(text)


def is_literal_linkedin_profile_url(url: str) -> bool:
    return bool(url) and bool(LINKEDIN_PROFILE_PATTERN.match(url.strip()))


def mentions_csr_context(snippet: str) -> bool:
    lowered = snippet.lower()
    return any(kw in lowered for kw in CSR_KEYWORDS)


def is_csr_relevant(text: str) -> bool:
    lowered = text.lower()
    relevance_keywords = [
        "csr", "corporate social", "sustainability", "philanthrop",
        "community", "crore", "education", "skill", "digital",
        "social responsibility", "esg", "impact report",
    ]
    return sum(1 for kw in relevance_keywords if kw in lowered) >= 2


def has_financial_figures(text: str) -> bool:
    return bool(CURRENCY_FIGURE_PATTERN.search(text))


def count_financial_figures(text: str) -> int:
    return len(CURRENCY_FIGURE_PATTERN.findall(text))


def find_india_location_mentions(text: str) -> list[dict]:
    if not text:
        return []
    hits = []
    for pattern, kind in (
        (INDIA_COUNTRY_PATTERN, "country"),
        (INDIA_STATE_PATTERN, "state"),
        (INDIA_CITY_PATTERN, "city"),
    ):
        for match in pattern.finditer(text):
            hits.append({
                "text": re.sub(r"\s+", " ", match.group(0)).strip(),
                "kind": kind,
                "start": match.start(),
                "end": match.end(),
            })
    hits.sort(key=lambda h: h["start"])
    return hits


def has_india_location_signal(text: str) -> bool:
    return bool(find_india_location_mentions(text))


def has_india_specific_financial_figure(text: str, window_chars: int = CURRENCY_NEAR_INDIA_WINDOW_CHARS) -> bool:
    if not text:
        return False
    location_hits = find_india_location_mentions(text)
    if not location_hits:
        return False
    location_positions = [hit["start"] for hit in location_hits]
    for match in CURRENCY_FIGURE_PATTERN.finditer(text):
        if any(abs(match.start() - pos) <= window_chars for pos in location_positions):
            return True
    return False


def company_name_tokens(company: str) -> list[str]:
    return [
        token for token in re.sub(r"[^a-z0-9 ]", " ", company.lower()).split()
        if len(token) > 2 and token not in GENERIC_COMPANY_TOKENS
    ]


def mentions_company(company: str, text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    tokens = company_name_tokens(company)
    if not tokens:
        return company.lower() in lowered
    return any(token in lowered for token in tokens)


def candidate_domains(company: str) -> list[str]:
    tokens = company_name_tokens(company) or [re.sub(r"[^a-z0-9]", "", company.lower())]
    slugs = list(dict.fromkeys(["".join(tokens), tokens[0]]))
    return [f"www.{slug}{tld}" for slug in slugs if slug for tld in (".com", ".in", ".co.in", ".org")]


def url_belongs_to_company(company: str, url: str, known_domains: list[str] | None = None) -> bool:
    if not url:
        return False
    host = urlparse(url).netloc.lower()
    if not host:
        return False
    if any(gov in host for gov in OFFICIAL_GOV_DOMAINS):
        return True
    if known_domains and any(host == d or host.endswith("." + d) for d in known_domains):
        return True
    tokens = company_name_tokens(company)
    if not tokens:
        return False
    host_base = host.replace("www.", "")
    return any(token in host_base for token in tokens)


def is_known_blocked_domain(url: str) -> bool:
    if not url:
        return False
    host = urlparse(url).netloc.lower()
    return any(domain in host for domain in BLOCKED_403_DOMAINS)


def accept_fetched_text(company: str, text: str, min_len: int = 400) -> bool:
    return bool(text) and len(text) > min_len and is_csr_relevant(text) and mentions_company(company, text)


def score_candidate_text(company: str, text: str, url: str = "") -> float:
    if not text:
        return -1.0
    if not mentions_company(company, text):
        return -1.0
    csr_hits = sum(1 for kw in CSR_KEYWORDS if kw in text.lower())
    if csr_hits == 0 and not is_csr_relevant(text):
        return -1.0
    figure_hits = count_financial_figures(text)
    india_figure_bonus = 4.0 if has_india_specific_financial_figure(text) else 0.0
    india_location_bonus = 2.0 if has_india_location_signal(text) else 0.0
    length_bonus = min(len(text) / 2000.0, 4.0)
    domain_bonus = 6.0 if url and any(gov in url.lower() for gov in OFFICIAL_GOV_DOMAINS) else 0.0
    pdf_bonus = 1.5 if url.lower().endswith(".pdf") else 0.0
    return (
        csr_hits * 2.0 + figure_hits * 5.0 + india_figure_bonus + india_location_bonus
        + length_bonus + domain_bonus + pdf_bonus
    )


def is_plausible_legal_entity_name(company: str, candidate: str) -> bool:
    if not candidate:
        return False
    if len(candidate) > 120:
        return False
    if candidate.count(".") > 3:
        return False
    lowered = candidate.lower()
    if " ahmedabad" in lowered or " mumbai" in lowered or " bangalore" in lowered:
        return False
    if not re.search(r"(private\s+limited|pvt\.?\s*ltd\.?|limited|ltd\.?|foundation|trust)$", lowered.strip(), re.IGNORECASE):
        return False
    word_count = len(candidate.split())
    if word_count > 9:
        return False
    tokens = company_name_tokens(company)
    if tokens and not any(token in lowered for token in tokens):
        return False
    return True


async def ddgs_search_web(query: str, budget: SearchBudget | None, max_results: int = 5,
                           total_budget_seconds: float = DDGS_TOTAL_BUDGET_SECONDS) -> list[dict]:
    if budget is not None and not budget.ddgs_has_budget():
        logger.info("ddgs skipped, budget exhausted query=%r", query)
        return []

    def _run_sync() -> list[dict]:
        try:
            from ddgs import DDGS
        except Exception as exc:
            logger.warning("ddgs import failed query=%r error=%s", query, exc)
            return []
        deadline = time.monotonic() + total_budget_seconds
        for backend in DDGS_BACKENDS:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.info("ddgs time budget exhausted query=%r", query)
                break
            try:
                with DDGS(timeout=max(1, min(remaining, 5))) as ddgs:
                    results = list(ddgs.text(query, max_results=max_results, backend=backend))
                if results:
                    return results
            except Exception as exc:
                logger.info("ddgs backend failed backend=%s query=%r error=%s", backend, query, exc)
                continue
        return []

    lock = ddgs_global_lock()
    async with lock:
        if budget is not None:
            budget.record_ddgs_query()
        try:
            return await asyncio.wait_for(asyncio.to_thread(_run_sync), timeout=SEARCH_TASK_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.warning("ddgs_search_web timed out query=%r", query)
            return []


async def search_web(query: str, budget: SearchBudget, max_results: int = 6,
                      prefer_google: bool = True, quota_guard=None) -> list[dict]:
    google_available = prefer_google and google_search.google_search_configured_and_available(quota_guard)

    if google_available and budget.google_has_budget():
        budget.record_google_query()
        try:
            results = await asyncio.wait_for(
                google_search.google_search_web(query, max_results=max_results, quota_guard=quota_guard),
                timeout=SEARCH_TASK_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning("google search timed out query=%r", query)
            results = []
        if results:
            return results
        logger.info("google search returned empty query=%r", query)
        if not budget.ddgs_has_budget():
            return []
        return await ddgs_search_web(query, budget, max_results=max_results)

    if google_available and not budget.google_has_budget():
        logger.info("google search skipped, company budget exhausted query=%r", query)

    if not budget.ddgs_has_budget():
        return []
    return await ddgs_search_web(query, budget, max_results=max_results)


def _fetch_page_text_sync(url: str, max_chars: int, verify_ssl: bool) -> str:
    if is_known_blocked_domain(url):
        return ""
    try:
        response = get_session().get(url, timeout=PAGE_FETCH_TIMEOUT_SECONDS, verify=verify_ssl)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        return extract_main_text(soup, max_chars)
    except Exception as exc:
        if "404" in str(exc):
            logger.debug("fetch_page_text 404 url=%s", url)
        elif "403" in str(exc):
            logger.debug("fetch_page_text 403 url=%s", url)
        else:
            logger.info("fetch_page_text failed url=%s error=%s", url, exc)
        return ""


async def fetch_page_text(url: str, max_chars: int = MAX_PAGE_TEXT_CHARS, verify_ssl: bool = True) -> str:
    async with _FETCH_SEMAPHORE:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_fetch_page_text_sync, url, max_chars, verify_ssl),
                timeout=FETCH_TASK_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.info("fetch_page_text timed out url=%s", url)
            return ""


def _select_financial_pdf_pages(pdf, max_pages: int, max_scan: int) -> list:
    total_pages = len(pdf.pages)
    scan_upper = min(total_pages, max_scan)
    if total_pages <= max_pages:
        return list(pdf.pages[:total_pages])

    scored_indices = []
    for idx in range(scan_upper):
        try:
            snippet = pdf.pages[idx].extract_text() or ""
        except Exception:
            snippet = ""
        if not snippet:
            continue
        lowered = snippet.lower()
        score = count_financial_figures(snippet) * 3
        if has_india_location_signal(snippet):
            score += 2
        if "csr" in lowered:
            score += 2
        if any(term in lowered for term in ("schedule vii", "annexure", "amount spent", "csr expenditure", "csr committee")):
            score += 3
        if score > 0:
            scored_indices.append((score, idx))

    if not scored_indices:
        return list(pdf.pages[:max_pages])

    scored_indices.sort(key=lambda pair: pair[0], reverse=True)
    top_indices = sorted(idx for _, idx in scored_indices[:max_pages])
    return [pdf.pages[idx] for idx in top_indices]


def _fetch_pdf_text_sync(url: str, max_chars: int, max_pages: int) -> str:
    if is_known_blocked_domain(url):
        return ""
    try:
        import io
        import pdfplumber

        response = get_session().get(url, timeout=PDF_FETCH_TIMEOUT_SECONDS)
        response.raise_for_status()
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            pages = _select_financial_pdf_pages(pdf, max_pages, FINANCIAL_PDF_SCAN_PAGES)
            pages_text = []
            total_len = 0
            for page in pages:
                page_text = page.extract_text() or ""
                if page_text:
                    pages_text.append(page_text)
                    total_len += len(page_text)
                if total_len >= max_chars:
                    break
        return clean_text(" ".join(pages_text), max_chars)
    except Exception as exc:
        logger.info("fetch_pdf_text failed url=%s error=%s", url, exc)
        return ""


async def fetch_pdf_text(url: str, max_chars: int = MAX_PDF_TEXT_CHARS, max_pages: int = MAX_PDF_PAGES) -> str:
    async with _FETCH_SEMAPHORE:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_fetch_pdf_text_sync, url, max_chars, max_pages),
                timeout=FETCH_TASK_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.info("fetch_pdf_text timed out url=%s", url)
            return ""


def csr_links_from_html(base_url: str, html: str, limit: int = 10) -> list[str]:
    scored_links = []
    try:
        soup = BeautifulSoup(html, "lxml")
        for anchor_tag in soup.find_all("a", href=True):
            href = anchor_tag["href"]
            if href.startswith(("#", "mailto:", "javascript:", "tel:")):
                continue
            anchor_text = anchor_tag.get_text(" ", strip=True)[:80]
            if NEGATIVE_LINK_PATTERN.search(href) or NEGATIVE_LINK_PATTERN.search(anchor_text):
                continue
            score = (2 if CSR_LINK_PATTERN.search(href) else 0) + (1 if CSR_LINK_PATTERN.search(anchor_text) else 0)
            if href.lower().endswith(".pdf") and CSR_LINK_PATTERN.search(anchor_text + href):
                score += 2
            if score:
                scored_links.append((score, urljoin(base_url, href)))
    except Exception as exc:
        logger.info("csr_links_from_html parse failed base_url=%s error=%s", base_url, exc)
    scored_links.sort(key=lambda item: -item[0])
    seen_urls, ordered_urls = set(), []
    for _, url in scored_links:
        if url not in seen_urls:
            seen_urls.add(url)
            ordered_urls.append(url)
        if len(ordered_urls) >= limit:
            break
    return ordered_urls


def _sitemap_csr_urls_sync(domain: str, limit: int) -> list[str]:
    try:
        response = get_session().get(f"https://{domain}/sitemap.xml", timeout=PAGE_FETCH_TIMEOUT_SECONDS)
        response.raise_for_status()
        urls = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", response.text)[:1200]
        matched = [url for url in urls if CSR_LINK_PATTERN.search(url)]
        if matched:
            return matched[:limit]
        nested_sitemaps = [url for url in urls if url.endswith(".xml")][:4]
        for nested_url in nested_sitemaps:
            try:
                nested_response = get_session().get(nested_url, timeout=PAGE_FETCH_TIMEOUT_SECONDS)
                nested_response.raise_for_status()
                nested_urls = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", nested_response.text)[:1200]
                nested_matched = [url for url in nested_urls if CSR_LINK_PATTERN.search(url)]
                if nested_matched:
                    return nested_matched[:limit]
            except Exception as exc:
                logger.info("sitemap nested fetch failed url=%s error=%s", nested_url, exc)
                continue
    except Exception as exc:
        logger.info("sitemap fetch failed domain=%s error=%s", domain, exc)
    return []


async def sitemap_csr_urls(domain: str, limit: int = 8) -> list[str]:
    async with _FETCH_SEMAPHORE:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_sitemap_csr_urls_sync, domain, limit),
                timeout=FETCH_TASK_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.info("sitemap_csr_urls timed out domain=%s", domain)
            return []


async def discover_company_domain(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None) -> str:
    domains = await discover_company_domains(company, search_cfg, budget, quota_guard)
    return domains[0] if domains else ""


async def discover_company_domains(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None) -> list[str]:
    tokens = company_name_tokens(company)
    acronym = "".join(token[0] for token in tokens) if len(tokens) >= 2 else ""
    results = await search_web(
        f'"{company}" official website India',
        budget,
        max_results=6,
        prefer_google=search_cfg.get("csr_pages", True),
        quota_guard=quota_guard,
    )
    matched_domains = []
    for result in results:
        host = urlparse(result.get("href", "")).netloc.lower()
        if not host or any(domain in host for domain in AGGREGATOR_DOMAINS):
            continue
        if NEGATIVE_LINK_PATTERN.search(host):
            continue
        host_base = host.replace("www.", "").split(".")[0]
        if any(token in host for token in tokens) or (acronym and host_base == acronym):
            if host not in matched_domains:
                matched_domains.append(host)
    return matched_domains[:3]


async def resolve_india_legal_entity_name(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None) -> str:
    if budget.legal_entity_name_resolved:
        return budget.legal_entity_name_cache or ""

    resolved_name = ""
    for query_template in LEGAL_ENTITY_RESOLUTION_QUERIES:
        query = query_template.format(c=company)
        results = await search_web(
            query, budget, max_results=6, prefer_google=search_cfg.get("mca", True), quota_guard=quota_guard,
        )
        for result in results:
            haystack = f"{result.get('title', '')} {result.get('body', '')}"
            match = INDIA_LEGAL_ENTITY_PATTERN.search(haystack)
            if match:
                candidate = re.sub(r"\s+", " ", match.group(1)).strip()
                if is_plausible_legal_entity_name(company, candidate):
                    resolved_name = candidate
                    break
        if resolved_name:
            break

    budget.legal_entity_name_resolved = True
    budget.legal_entity_name_cache = resolved_name
    logger.info("resolve_india_legal_entity_name DONE company=%r resolved=%r", company, resolved_name or None)
    return resolved_name


async def _within_deadline(deadline: float) -> bool:
    return time.monotonic() < deadline


async def fetch_india_csr_page(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None,
                                max_fetches: int = 20, registry: SourceRegistry | None = None) -> dict:
    deadline = time.monotonic() + SOURCE_DEADLINE_SECONDS
    tried_urls = set()
    remaining_budget = [max_fetches]
    resolved_domain = [""]
    best_candidate = [None]

    def consider(url: str, method: str, text: str):
        if not accept_fetched_text(company, text, 250):
            return
        score = score_candidate_text(company, text, url)
        if best_candidate[0] is None or score > best_candidate[0][0]:
            source = make_source("india_csr_page", 1, url, text, "FOUND", method)
            source["domain"] = urlparse(url).netloc.lower()
            source["india_location_hits"] = find_india_location_mentions(text)[:10]
            best_candidate[0] = (score, source)

    async def try_fetch(url: str, method: str, is_pdf: bool = False):
        if not url or url in tried_urls or remaining_budget[0] <= 0 or not await _within_deadline(deadline):
            return
        tried_urls.add(url)
        remaining_budget[0] -= 1
        text = await (fetch_pdf_text(url) if is_pdf else fetch_page_text(url))
        consider(url, method, text)

    discovered_domains = await discover_company_domains(company, search_cfg, budget, quota_guard)
    domains = list(dict.fromkeys(discovered_domains + candidate_domains(company)))
    logger.info("india_csr_page discovered_domains company=%r domains=%s", company, domains)

    async def check_homepage(domain: str):
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(get_session().get, f"https://{domain}", timeout=HOMEPAGE_FETCH_TIMEOUT_SECONDS),
                timeout=FETCH_TASK_TIMEOUT_SECONDS,
            )
            if response.ok and mentions_company(company, response.text):
                return domain, response.text
        except Exception as exc:
            logger.info("homepage fetch failed domain=%s error=%s", domain, exc)
        return None

    live_homepages = []
    for domain in domains[:5]:
        result = await check_homepage(domain)
        if result:
            live_homepages.append(result)
        if len(live_homepages) >= 3:
            break
    logger.info("india_csr_page live_homepages company=%r count=%d", company, len(live_homepages))

    if live_homepages:
        resolved_domain[0] = live_homepages[0][0]

    candidates_checked = 0
    for domain, homepage_html in live_homepages:
        if not await _within_deadline(deadline):
            break
        links = csr_links_from_html(f"https://{domain}", homepage_html)
        for link in links:
            if best_candidate[0] and best_candidate[0][0] >= MIN_ACCEPT_SCORE:
                break
            if candidates_checked >= CANDIDATE_EVAL_LIMIT and best_candidate[0]:
                break
            await try_fetch(link, "homepage_link", is_pdf=link.lower().endswith(".pdf"))
            candidates_checked += 1
        for path in CSR_PAGE_PATHS[:12]:
            if best_candidate[0] and best_candidate[0][0] >= MIN_ACCEPT_SCORE:
                break
            if candidates_checked >= CANDIDATE_EVAL_LIMIT and best_candidate[0]:
                break
            await try_fetch(f"https://{domain}{path}", "direct")
            candidates_checked += 1
        if best_candidate[0] and best_candidate[0][0] >= MIN_ACCEPT_SCORE:
            break
        for sitemap_url in await sitemap_csr_urls(domain):
            if best_candidate[0] and best_candidate[0][0] >= MIN_ACCEPT_SCORE:
                break
            await try_fetch(sitemap_url, "sitemap", is_pdf=sitemap_url.lower().endswith(".pdf"))
            candidates_checked += 1
        if best_candidate[0] and best_candidate[0][0] >= MIN_ACCEPT_SCORE:
            break

    remaining_budget[0] = max(remaining_budget[0], 8)
    if (not best_candidate[0] or best_candidate[0][0] < MIN_ACCEPT_SCORE) and await _within_deadline(deadline):
        query_pool = []
        for domain, _ in live_homepages[:1]:
            for template in CSR_PAGE_QUERIES:
                query_pool.append(template.format(c=company, d=domain))
        query_pool.extend(template.format(c=company) for template in CSR_PAGE_FALLBACK_QUERIES)

        for query in query_pool:
            if not await _within_deadline(deadline):
                break
            if best_candidate[0] and best_candidate[0][0] >= MIN_ACCEPT_SCORE:
                break
            results = await search_web(
                query, budget, max_results=6, prefer_google=search_cfg.get("csr_pages", True), quota_guard=quota_guard
            )
            for result in results:
                url = result.get("href", "")
                title = result.get("title", "")
                snippet_body = result.get("body", "")
                if not url or any(domain in url for domain in AGGREGATOR_DOMAINS):
                    continue
                if not mentions_company(company, f"{title} {snippet_body}"):
                    continue
                if not url_belongs_to_company(company, url, list(dict.fromkeys(discovered_domains))):
                    continue
                await try_fetch(url, "search", is_pdf=url.lower().endswith(".pdf"))
                consider(url, "snippet", snippet_body)
                if best_candidate[0] and best_candidate[0][0] >= MIN_ACCEPT_SCORE:
                    break

    if best_candidate[0]:
        logger.info(
            "india_csr_page DONE company=%r found=True score=%.1f url=%s",
            company, best_candidate[0][0], best_candidate[0][1].get("url", ""),
        )
        result_source = best_candidate[0][1]
        if registry is not None:
            registry.register_core_source(result_source)
        return result_source

    logger.info("india_csr_page DONE company=%r found=False", company)
    fallback = make_source("india_csr_page", 1, status="NOT_FOUND")
    fallback["domain"] = resolved_domain[0]
    return fallback


async def find_company_cin(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None,
                            legal_name: str = "") -> str:
    templates = list(MCA_CIN_QUERIES)
    if legal_name:
        templates = [t.format(legal_name=legal_name, c="{c}") for t in MCA_ENTITY_CIN_QUERIES] + templates
    for query_template in templates:
        query = query_template if legal_name and "{legal_name}" not in query_template else query_template.format(c=company)
        results = await search_web(
            query, budget, max_results=5, prefer_google=search_cfg.get("mca", True), quota_guard=quota_guard,
        )
        for result in results:
            body = result.get("body", "") + " " + result.get("title", "") + " " + result.get("href", "")
            match = CIN_PATTERN.search(body)
            if match:
                logger.info("find_company_cin DONE company=%r cin=%s query=%r", company, match.group(0).upper(), query)
                return match.group(0).upper()
    return ""


async def fetch_mca_company_data_gov_page(cin: str) -> str:
    if not cin:
        return ""
    candidate_urls = [
        f"https://www.mca.gov.in/mcafoportal/viewCompanyMasterData.do?cid={cin}",
        f"https://www.mca.gov.in/content/mca/global/en/mca/master-data/MDS.html?cin={cin}",
    ]
    for url in candidate_urls:
        text = await fetch_page_text(url)
        if text and len(text) > 150:
            return text
    return ""


async def fetch_mca_portal(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None,
                            registry: SourceRegistry | None = None) -> dict:
    deadline = time.monotonic() + SOURCE_DEADLINE_SECONDS
    legal_name = await resolve_india_legal_entity_name(company, search_cfg, budget, quota_guard)
    cin = await find_company_cin(company, search_cfg, budget, quota_guard, legal_name=legal_name)

    if cin:
        mca_text = await fetch_mca_company_data_gov_page(cin)
        if mca_text and mentions_company(company, mca_text):
            source = make_source(
                "mca_portal", 2,
                f"https://www.mca.gov.in/mcafoportal/viewCompanyMasterData.do?cid={cin}",
                mca_text, "FOUND", "direct",
            )
            source["cin"] = cin
            if legal_name:
                source["legal_entity_name"] = legal_name
            if registry is not None:
                registry.register_core_source(source)
            return source

    best_candidate = None
    filing_templates = list(MCA_FILING_QUERIES)
    if legal_name:
        filing_templates = [t.format(legal_name=legal_name) for t in MCA_ENTITY_FILING_QUERIES] + filing_templates
    for query_template in filing_templates:
        if not await _within_deadline(deadline):
            break
        if best_candidate and best_candidate[0] >= MIN_ACCEPT_SCORE:
            break
        query = query_template if "{c}" not in query_template else query_template.format(c=company)
        results = await search_web(
            query, budget, max_results=6,
            prefer_google=search_cfg.get("mca", True), quota_guard=quota_guard,
        )
        for result in results:
            url = result.get("href", "")
            body = result.get("body", "")
            if not url:
                continue
            text = await (fetch_pdf_text(url) if url.lower().endswith(".pdf") else fetch_page_text(url))
            if not text:
                text = body
            if not (text and is_csr_relevant(text) and mentions_company(company, text)):
                continue
            score = score_candidate_text(company, text, url)
            if best_candidate is None or score > best_candidate[0]:
                found_cin = cin or ""
                if not found_cin:
                    cin_match = CIN_PATTERN.search(text) or CIN_PATTERN.search(body)
                    if cin_match:
                        found_cin = cin_match.group(0).upper()
                source = make_source("mca_via_search", 2, url, text, "FOUND", "search_proxy")
                if found_cin:
                    source["cin"] = found_cin
                if legal_name:
                    source["legal_entity_name"] = legal_name
                best_candidate = (score, source)
            if best_candidate and best_candidate[0] >= MIN_ACCEPT_SCORE:
                break

    if best_candidate:
        logger.info(
            "mca_portal DONE company=%r found=True cin=%s legal_name=%r",
            company, cin or best_candidate[1].get("cin", ""), legal_name,
        )
        if registry is not None:
            registry.register_core_source(best_candidate[1])
        return best_candidate[1]

    logger.info("mca_portal DONE company=%r found=False cin=%s legal_name=%r", company, cin, legal_name)
    return make_source("mca_portal", 2, status="NOT_FOUND")


async def fetch_national_csr_portal(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None,
                                     registry: SourceRegistry | None = None) -> dict:
    deadline = time.monotonic() + SOURCE_DEADLINE_SECONDS
    company_query = company.replace(" ", "+")
    direct_urls = [
        f"https://www.csr.gov.in/content/csr/global/master/home/companydetail.html?companyName={company_query}",
    ]
    for url in direct_urls:
        text = await fetch_page_text(url)
        if text and len(text) > 250 and mentions_company(company, text):
            source = make_source("national_csr_portal", 3, url, text, "FOUND", "direct")
            if registry is not None:
                registry.register_core_source(source)
            return source

    best_candidate = None
    for query_template in NATIONAL_CSR_PORTAL_QUERIES:
        if not await _within_deadline(deadline):
            break
        results = await search_web(
            query_template.format(c=company), budget, max_results=6,
            prefer_google=search_cfg.get("mca", True), quota_guard=quota_guard,
        )
        for result in results[:6]:
            url = result.get("href", "")
            body = result.get("body", "")
            if not url:
                continue
            text = await fetch_page_text(url) or body
            if text and mentions_company(company, text) and ("csr.gov.in" in url.lower() or is_csr_relevant(text)):
                score = score_candidate_text(company, text, url)
                if best_candidate is None or score > best_candidate[0]:
                    best_candidate = (score, make_source("national_csr_portal", 3, url, text, "FOUND", "search"))
        if best_candidate and best_candidate[0] >= MIN_ACCEPT_SCORE:
            break

    if best_candidate:
        logger.info("national_csr_portal DONE company=%r found=True", company)
        if registry is not None:
            registry.register_core_source(best_candidate[1])
        return best_candidate[1]

    logger.info("national_csr_portal DONE company=%r found=False", company)
    return make_source("national_csr_portal", 3, status="NOT_FOUND")


async def fetch_annual_report(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None,
                               registry: SourceRegistry | None = None) -> dict:
    deadline = time.monotonic() + SOURCE_DEADLINE_SECONDS
    legal_name = await resolve_india_legal_entity_name(company, search_cfg, budget, quota_guard)
    best_candidate = None
    urls_tried = 0
    rejected_non_india_specific = 0

    query_templates = [(t, False) for t in ANNUAL_REPORT_QUERIES]
    if legal_name:
        query_templates = [(t, True) for t in ANNUAL_REPORT_ENTITY_QUERIES] + query_templates

    for template, is_entity_query in query_templates:
        if not await _within_deadline(deadline):
            break
        if best_candidate and best_candidate[0] >= STRONG_ACCEPT_SCORE:
            break
        query = template.format(legal_name=legal_name, fy=CURRENT_FY_LABEL) if is_entity_query \
            else template.format(c=company, fy=CURRENT_FY_LABEL)
        results = await search_web(
            query,
            budget,
            max_results=8,
            prefer_google=search_cfg.get("annual_reports", True),
            quota_guard=quota_guard,
        )
        for result in results:
            url = result.get("href", "")
            title = result.get("title", "")
            body = result.get("body", "")
            if not url:
                continue
            if not mentions_company(company, f"{title} {body}"):
                continue
            if not url_belongs_to_company(company, url):
                continue
            urls_tried += 1
            if url.lower().endswith(".pdf"):
                text = await fetch_pdf_text(url)
                if not text and body and len(body) > 100 and mentions_company(company, body):
                    text = body
                if text and not pdf_is_csr_relevant(text):
                    logger.info("annual_report rejected non-csr pdf company=%r url=%s", company, url)
                    continue
                if text and count_financial_figures(text) > 0 and not has_india_specific_financial_figure(text):
                    rejected_non_india_specific += 1
                    continue
            else:
                text = await fetch_page_text(url) or body
            if not (text and len(text) > 250 and mentions_company(company, text)):
                continue
            score = score_candidate_text(company, text, url)
            if best_candidate is None or score > best_candidate[0]:
                fetch_method = "pdf" if url.lower().endswith(".pdf") else "search"
                best_candidate = (score, make_source("annual_report", 4, url, text, "FOUND", fetch_method))
            if best_candidate and best_candidate[0] >= STRONG_ACCEPT_SCORE:
                break

    if (not best_candidate or count_financial_figures(best_candidate[1].get("text", "")) == 0) and await _within_deadline(deadline):
        for fy in PRIOR_FY_LABELS[:1]:
            if not await _within_deadline(deadline):
                break
            query = f'"{company}" "annual report" {fy} CSR filetype:pdf'
            results = await search_web(
                query, budget, max_results=6, prefer_google=search_cfg.get("annual_reports", True), quota_guard=quota_guard
            )
            for result in results:
                url = result.get("href", "")
                title = result.get("title", "")
                body = result.get("body", "")
                if not url or not url.lower().endswith(".pdf"):
                    continue
                if not mentions_company(company, f"{title} {body}") or not url_belongs_to_company(company, url):
                    continue
                text = await fetch_pdf_text(url)
                if text and not pdf_is_csr_relevant(text):
                    continue
                if text and has_financial_figures(text) and mentions_company(company, text):
                    score = score_candidate_text(company, text, url)
                    if best_candidate is None or score > best_candidate[0]:
                        best_candidate = (score, make_source("annual_report", 4, url, text, "FOUND", "pdf_prior_fy"))
            if best_candidate and count_financial_figures(best_candidate[1].get("text", "")) > 0:
                break

    logger.info(
        "annual_report DONE company=%r urls_tried=%d rejected_non_india_specific=%d found=%s legal_name=%r",
        company, urls_tried, rejected_non_india_specific, bool(best_candidate), legal_name,
    )

    if best_candidate:
        if registry is not None:
            registry.register_core_source(best_candidate[1])
        return best_candidate[1]

    return make_source("annual_report", 4, status="NOT_FOUND")


async def fetch_partner_source(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None,
                                registry: SourceRegistry | None = None) -> dict:
    deadline = time.monotonic() + SOURCE_DEADLINE_SECONDS
    best_candidate = None
    urls_tried = 0
    for template in PARTNER_QUERIES:
        if not await _within_deadline(deadline):
            break
        if best_candidate and best_candidate[0] >= MIN_ACCEPT_SCORE:
            break
        query = template.format(c=company)
        results = await search_web(query, budget, max_results=6, prefer_google=search_cfg.get("partners", True), quota_guard=quota_guard)
        for result in results:
            url = result.get("href", "")
            title = result.get("title", "")
            body = result.get("body", "")
            if not url or any(domain in url for domain in AGGREGATOR_DOMAINS):
                continue
            if not mentions_company(company, f"{title} {body}"):
                continue
            urls_tried += 1
            is_pdf = url.lower().endswith(".pdf")
            text = await (fetch_pdf_text(url) if is_pdf else fetch_page_text(url)) or body
            if is_pdf and text and not pdf_is_csr_relevant(text):
                continue
            if not (text and len(text) > 250 and is_csr_relevant(text) and mentions_company(company, text)):
                continue
            score = score_candidate_text(company, text, url)
            if best_candidate is None or score > best_candidate[0]:
                best_candidate = (score, make_source("partner_search", 5, url, text, "FOUND", "search"))
            if best_candidate and best_candidate[0] >= MIN_ACCEPT_SCORE:
                break

    logger.info("partner_search DONE company=%r urls_tried=%d found=%s", company, urls_tried, bool(best_candidate))

    if best_candidate:
        if registry is not None:
            registry.register_core_source(best_candidate[1])
        return best_candidate[1]

    return make_source("partner_search", 5, status="NOT_FOUND")


async def fetch_education_programme_source(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None,
                                            registry: SourceRegistry | None = None) -> dict:
    deadline = time.monotonic() + SOURCE_DEADLINE_SECONDS
    best_candidate = None
    urls_tried = 0
    for template in EDUCATION_PROGRAMME_QUERIES:
        if not await _within_deadline(deadline):
            break
        if best_candidate and best_candidate[0] >= MIN_ACCEPT_SCORE:
            break
        query = template.format(c=company)
        results = await search_web(
            query, budget, max_results=6, prefer_google=search_cfg.get("partners", True), quota_guard=quota_guard
        )
        for result in results:
            url = result.get("href", "")
            title = result.get("title", "")
            body = result.get("body", "")
            if not url or any(domain in url for domain in AGGREGATOR_DOMAINS):
                continue
            if not mentions_company(company, f"{title} {body}"):
                continue
            urls_tried += 1
            is_pdf = url.lower().endswith(".pdf")
            text = await (fetch_pdf_text(url) if is_pdf else fetch_page_text(url)) or body
            if not text or len(text) < 200 or not mentions_company(company, text):
                continue
            if "education" not in text.lower() and not any(kw in text.lower() for kw in EDUCATION_KEYWORDS):
                continue
            score = score_candidate_text(company, text, url) + 5.0
            if best_candidate is None or score > best_candidate[0]:
                best_candidate = (score, make_source("education_programme_search", 9, url, text, "FOUND", "search"))
            if best_candidate and best_candidate[0] >= MIN_ACCEPT_SCORE:
                break

    logger.info(
        "education_programme_search DONE company=%r urls_tried=%d found=%s",
        company, urls_tried, bool(best_candidate),
    )

    if best_candidate:
        if registry is not None:
            registry.register_core_source(best_candidate[1])
        return best_candidate[1]

    return make_source("education_programme_search", 9, status="NOT_FOUND")


async def _run_linkedin_query_batch(company: str, queries: list[str], search_cfg: dict, budget: SearchBudget,
                                     quota_guard, deadline: float, add_hit_fn, max_hits: int) -> int:
    collected = 0
    for query_template in queries:
        if not await _within_deadline(deadline) or collected >= max_hits:
            break
        if not budget.google_has_budget():
            break
        role_hint = ""
        if '"' in query_template:
            parts = query_template.split('"')
            if len(parts) >= 4:
                role_hint = parts[3]
        budget.record_google_query()
        profiles = await google_search.google_search_linkedin_profiles(
            company, role_hint=role_hint, max_results=8, quota_guard=quota_guard,
        )
        for profile in profiles:
            url = profile.get("href", "")
            if not is_literal_linkedin_profile_url(url):
                continue
            if add_hit_fn(profile.get("title", ""), profile.get("body", ""), url):
                collected += 1
    return collected


async def fetch_linkedin_people(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None,
                                 registry: SourceRegistry | None = None) -> dict:
    deadline = time.monotonic() + SOURCE_DEADLINE_SECONDS
    hits: list[dict] = []
    seen_urls: set[str] = set()

    def _add_hit(raw_title: str, snippet: str, url: str) -> bool:
        if url in seen_urls:
            return False
        parsed = parse_linkedin_hit(raw_title, snippet, url, company)
        if not parsed["name"] or not parsed["has_csr_signal"]:
            return False
        seen_urls.add(url)
        if registry is not None:
            parsed["source_number"] = registry.register_child_hit(
                source_name="people_search",
                url=url,
                label=f"LinkedIn — {parsed['name']}",
                excerpt=f"{parsed['title']} — {parsed['snippet']}"[:280],
            )
        hits.append(parsed)
        return True

    if search_cfg.get("linkedin_people", True):
        await _run_linkedin_query_batch(company, LINKEDIN_PEOPLE_QUERIES, search_cfg, budget, quota_guard, deadline, _add_hit, 15)

    india_signal_hits = [h for h in hits if h.get("india_location_signal")]
    if not india_signal_hits and await _within_deadline(deadline):
        logger.info(
            "fetch_linkedin_people no india-scoped hits, trying global CSR contact fallback company=%r",
            company,
        )
        await _run_linkedin_query_batch(
            company, LINKEDIN_PEOPLE_GLOBAL_FALLBACK_QUERIES, search_cfg, budget, quota_guard, deadline, _add_hit, 10
        )

    if not hits:
        return make_source("people_search", 6, status="NOT_FOUND")

    hits.sort(key=lambda h: (h.get("confidence") != "HIGH", h.get("confidence") != "MEDIUM"))
    high_confidence_hits = [h for h in hits if h.get("confidence") == "HIGH"]
    medium_confidence_hits = [h for h in hits if h.get("confidence") == "MEDIUM"]
    final_hits = (high_confidence_hits + medium_confidence_hits)[:10] or hits[:6]

    combined_text = " || ".join(
        f"{hit['name']} — {hit['title']} — {hit['snippet']}" for hit in final_hits
    )
    source = make_source("people_search", 6, final_hits[0]["url"], clean_text(combined_text, 4000), "FOUND", "search_snippets")
    source["people_hits"] = final_hits
    source["used_global_fallback"] = not bool(india_signal_hits) and bool(final_hits)
    if registry is not None:
        registry.register_core_source(source)
    logger.info(
        "people_search DONE company=%r hits_total=%d high_confidence=%d medium_confidence=%d used_global_fallback=%s",
        company, len(hits), len(high_confidence_hits), len(medium_confidence_hits), source["used_global_fallback"],
    )
    return source


async def fetch_plans_source(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None,
                              max_pages: int = 3, registry: SourceRegistry | None = None) -> dict:
    deadline = time.monotonic() + SOURCE_DEADLINE_SECONDS
    hits, fetched_texts, first_url = [], [], ""
    all_queries = PLAN_QUERIES + RFP_QUERIES
    for query_template in all_queries:
        if not await _within_deadline(deadline):
            break
        if len(fetched_texts) >= max_pages:
            break
        query = query_template.format(c=company)
        results = await search_web(
            query, budget, max_results=5, prefer_google=search_cfg.get("partners", True), quota_guard=quota_guard
        )
        for result in results:
            url = result.get("href", "")
            title = result.get("title", "")
            body = result.get("body", "")
            if not url or any(domain in url for domain in AGGREGATOR_DOMAINS):
                continue
            if not mentions_company(company, f"{title} {body}"):
                continue
            source_number = None
            if registry is not None:
                source_number = registry.register_child_hit(
                    source_name="plans_search", url=url, label=title or url, excerpt=body,
                )
            hits.append({"title": title, "snippet": body, "url": url, "source_number": source_number})
            if len(fetched_texts) < max_pages:
                text = await (fetch_pdf_text(url) if url.lower().endswith(".pdf") else fetch_page_text(url)) or body
                if text and len(text) > 250 and is_csr_relevant(text) and mentions_company(company, text):
                    fetched_texts.append(text)
                    first_url = first_url or url

    if not hits and not fetched_texts:
        return make_source("plans_search", 7, status="NOT_FOUND")

    combined_text = " || ".join(fetched_texts) if fetched_texts else " || ".join(f"{hit['title']} — {hit['snippet']}" for hit in hits)
    source = make_source(
        "plans_search", 7, first_url or hits[0]["url"], clean_text(combined_text, 7000), "FOUND",
        "search" if fetched_texts else "search_snippets",
    )
    source["plan_hits"] = hits[:10]
    if registry is not None:
        registry.register_core_source(source)
    return source


async def fetch_sector_eligibility_source(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None,
                                           registry: SourceRegistry | None = None) -> dict:
    deadline = time.monotonic() + SOURCE_DEADLINE_SECONDS
    hits, fetched_texts, first_url = [], [], ""
    queries = [t.format(c=company, fy=CURRENT_FY_LABEL) for t in SECTOR_QUERIES] + \
              [t.format(c=company) for t in GROUP_FOUNDATION_QUERIES]
    for query in queries:
        if not await _within_deadline(deadline):
            break
        if len(fetched_texts) >= 3:
            break
        results = await search_web(
            query, budget, max_results=5,
            prefer_google=search_cfg.get("mca", True), quota_guard=quota_guard,
        )
        for result in results:
            url = result.get("href", "")
            title = result.get("title", "")
            body = result.get("body", "")
            if not url or any(domain in url for domain in AGGREGATOR_DOMAINS):
                continue
            if not mentions_company(company, f"{title} {body}"):
                continue
            hits.append({"title": title, "snippet": body, "url": url})
            if len(fetched_texts) < 3:
                text = await fetch_page_text(url) or body
                if text and len(text) > 200 and mentions_company(company, text):
                    fetched_texts.append(text)
                    first_url = first_url or url

    if not hits and not fetched_texts:
        return make_source("sector_eligibility_search", 8, status="NOT_FOUND")

    combined_text = " || ".join(fetched_texts) if fetched_texts else " || ".join(f"{hit['title']} — {hit['snippet']}" for hit in hits)
    source = make_source(
        "sector_eligibility_search", 8, first_url or hits[0]["url"], clean_text(combined_text, 6000), "FOUND",
        "search" if fetched_texts else "search_snippets",
    )
    if registry is not None:
        registry.register_core_source(source)
    return source


async def run_targeted_queries(company: str, question_category: str, search_cfg: dict, budget: SearchBudget,
                                quota_guard=None, registry: SourceRegistry | None = None, max_fetches: int = 4,
                                deadline_seconds: float = FOLLOWUP_DEADLINE_SECONDS) -> dict:
    templates = FOLLOWUP_QUERY_TEMPLATES.get(question_category, [])
    if not templates:
        return make_source(f"followup_{question_category}", 10, status="NOT_TRIED")

    deadline = time.monotonic() + deadline_seconds
    best_candidate = None
    fetches_done = 0

    for template in templates:
        if not await _within_deadline(deadline) or fetches_done >= max_fetches:
            break
        query = template.format(c=company, fy=CURRENT_FY_LABEL)
        results = await search_web(query, budget, max_results=5, prefer_google=True, quota_guard=quota_guard)
        for result in results:
            url = result.get("href", "")
            title = result.get("title", "")
            body = result.get("body", "")
            if not url or any(domain in url for domain in AGGREGATOR_DOMAINS):
                continue
            if not mentions_company(company, f"{title} {body}"):
                continue
            if fetches_done >= max_fetches:
                break
            fetches_done += 1
            is_pdf = url.lower().endswith(".pdf")
            text = await (fetch_pdf_text(url) if is_pdf else fetch_page_text(url)) or body
            if not text or len(text) < 150 or not mentions_company(company, text):
                continue
            score = score_candidate_text(company, text, url)
            if best_candidate is None or score > best_candidate[0]:
                best_candidate = (score, make_source(f"followup_{question_category}", 10, url, text, "FOUND", "followup_search"))
        if best_candidate and best_candidate[0] >= MIN_ACCEPT_SCORE:
            break

    if best_candidate:
        if registry is not None:
            registry.register_core_source(best_candidate[1])
        return best_candidate[1]

    return make_source(f"followup_{question_category}", 10, status="NOT_FOUND")


async def fetch_screen_sources(company: str, search_cfg: dict, quota_guard=None,
                                registry: SourceRegistry | None = None) -> list[dict]:
    registry = registry or SourceRegistry(company)
    budget = SearchBudget(company)

    source_1 = await fetch_india_csr_page(company, search_cfg, budget, quota_guard, registry=registry)
    source_4 = await fetch_annual_report(company, search_cfg, budget, quota_guard, registry=registry)
    source_2 = await fetch_mca_portal(company, search_cfg, budget, quota_guard, registry=registry)
    source_5 = await fetch_partner_source(company, search_cfg, budget, quota_guard, registry=registry)

    source_3 = make_source("national_csr_portal", 3, status="NOT_TRIED")
    source_6 = make_source("people_search", 6, status="NOT_TRIED")
    source_7 = make_source("plans_search", 7, status="NOT_TRIED")
    source_8 = make_source("sector_eligibility_search", 8, status="NOT_TRIED")
    source_9 = make_source("education_programme_search", 9, status="NOT_TRIED")

    sources = [source_1, source_2, source_3, source_4, source_5, source_6, source_7, source_8, source_9]
    found_count = sum(1 for s in sources if s.get("status") == "FOUND")
    logger.info(
        "fetch_screen_sources DONE company=%r found=%d/9 google_used=%d ddgs_used=%d",
        company, found_count, budget.google_queries_used, budget.ddgs_queries_used,
    )
    return sources


async def fetch_deep_sources(company: str, search_cfg: dict, quota_guard=None, progress_cb=None,
                              registry: SourceRegistry | None = None) -> list[dict]:
    registry = registry or SourceRegistry(company)
    budget = SearchBudget(company)

    async def advance_step(message: str):
        if progress_cb:
            await progress_cb(message)

    await advance_step("Sources 1-4/9 — CSR page, MCA, National CSR Portal, annual report...")
    source_1 = await fetch_india_csr_page(company, search_cfg, budget, quota_guard, registry=registry)
    source_2 = await fetch_mca_portal(company, search_cfg, budget, quota_guard, registry=registry)
    source_3 = await fetch_national_csr_portal(company, search_cfg, budget, quota_guard, registry=registry)
    source_4 = await fetch_annual_report(company, search_cfg, budget, quota_guard, registry=registry)
    logger.info(
        "deep sources 1-4 company=%r csr_page=%s mca=%s national=%s annual=%s google_used=%d ddgs_used=%d",
        company, source_1.get("status"), source_2.get("status"), source_3.get("status"), source_4.get("status"),
        budget.google_queries_used, budget.ddgs_queries_used,
    )

    await advance_step("Sources 5-9/9 — partners, decision-makers, plans, sector, education programmes...")
    source_5 = await fetch_partner_source(company, search_cfg, budget, quota_guard, registry=registry)
    source_6 = await fetch_linkedin_people(company, search_cfg, budget, quota_guard, registry=registry)
    source_7 = await fetch_plans_source(company, search_cfg, budget, quota_guard, registry=registry)
    source_8 = await fetch_sector_eligibility_source(company, search_cfg, budget, quota_guard, registry=registry)
    source_9 = await fetch_education_programme_source(company, search_cfg, budget, quota_guard, registry=registry)

    sources = [source_1, source_2, source_3, source_4, source_5, source_6, source_7, source_8, source_9]
    total_figures = sum(count_financial_figures(s.get("text", "")) for s in sources)
    if total_figures == 0 and budget.google_has_budget():
        legal_name = await resolve_india_legal_entity_name(company, search_cfg, budget, quota_guard)
        spend_templates = [(t, True) for t in CSR_SPEND_ENTITY_QUERIES] if legal_name else []
        spend_templates += [(t, False) for t in CSR_SPEND_QUERIES]
        spend_deadline = time.monotonic() + SOURCE_DEADLINE_SECONDS
        for template, is_entity_query in spend_templates:
            if time.monotonic() >= spend_deadline or not budget.google_has_budget():
                break
            query = template.format(legal_name=legal_name, fy=CURRENT_FY_LABEL) if is_entity_query \
                else template.format(c=company, fy=CURRENT_FY_LABEL)
            results = await search_web(
                query, budget, max_results=6,
                prefer_google=search_cfg.get("csr_pages", True), quota_guard=quota_guard,
            )
            for result in results:
                url = result.get("href", "")
                body = result.get("body", "")
                if not url or not mentions_company(company, body):
                    continue
                text = await (fetch_pdf_text(url) if url.lower().endswith(".pdf") else fetch_page_text(url)) or body
                if text and has_financial_figures(text) and mentions_company(company, text):
                    source_4 = make_source("annual_report", 4, url, text, "FOUND", "spend_fallback")
                    registry.register_core_source(source_4)
                    sources[3] = source_4
                    logger.info(
                        "deep fallback spend search recovered figures company=%r url=%s legal_name=%r",
                        company, url, legal_name,
                    )
                    break
            if count_financial_figures(sources[3].get("text", "")) > 0:
                break

    found_count = sum(1 for s in sources if s.get("status") == "FOUND")
    logger.info(
        "fetch_deep_sources DONE company=%r found=%d/9 total_financial_figures=%d source_bank_entries=%d "
        "google_used=%d ddgs_used=%d",
        company, found_count, sum(count_financial_figures(s.get("text", "")) for s in sources),
        len(registry.entries()), budget.google_queries_used, budget.ddgs_queries_used,
    )
    return sources