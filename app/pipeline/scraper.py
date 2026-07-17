import logging
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from app.pipeline import google_search
from app.pipeline.utils import clean_text, extract_main_text, get_session, make_source

logger = logging.getLogger("tap.scraper")

GENERIC_COMPANY_TOKENS = {
    "india", "limited", "ltd", "private", "pvt", "the", "and", "of",
    "company", "corp", "corporation", "inc", "group", "technologies",
    "solutions", "services", "international",
}

AGGREGATOR_DOMAINS = (
    "youtube.", "twitter.", "x.com", "facebook.", "instagram.", "linkedin.",
    "wikipedia.", "glassdoor.", "indeed.", "crunchbase.", "bloomberg.",
    "zaubacorp", "tofler.", "justdial.", "indiamart.", "ambitionbox.",
    "moneycontrol.", "economictimes.", "livemint.", "reuters.",
    "apkpure.", "h1bgrader.", "quora.", "reddit.", "pinterest.",
)

OFFICIAL_GOV_DOMAINS = ("mca.gov.in", "csr.gov.in", "nic.in")

CSR_LINK_PATTERN = re.compile(
    r"(csr|corporate[\s_-]?social|social[\s_-]?responsib|sustainab|esg|"
    r"social[\s_-]?impact|citizenship|community[\s_-]?(initiativ|develop|engag)|"
    r"responsible[\s_-]?business|foundation|giving[\s_-]?back|annual[\s_-]?report|"
    r"investor[\s_-]?relation)",
    re.IGNORECASE,
)

NEGATIVE_LINK_PATTERN = re.compile(
    r"(career|job|vacanc|recruit|login|sign-?in|privacy|cookie|terms|disclaimer|sitemap|contact-?us)",
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
    "/csr-initiatives", "/csr-activities",
]

CSR_KEYWORDS = [
    "csr", "corporate social", "philanthrop", "social responsibility",
    "schedule vii", "csr spend", "csr expenditure", "csr budget",
    "csr obligation", "csr fund", "community investment",
]

CURRENCY_FIGURE_PATTERN = re.compile(
    r"(?:(?:rs\.?|inr|₹)\s?[\d,]+(?:\.\d+)?\s?(?:crore|cr|lakh|lac|million|mn|billion|bn)?"
    r"|[\d,]+(?:\.\d+)?\s?(?:crore|cr|lakh|lac)\b)",
    re.IGNORECASE,
)

LINKEDIN_PROFILE_PATTERN = re.compile(r"^https?://([a-z]{2,3}\.)?linkedin\.com/in/[^/?#]+/?$", re.IGNORECASE)

CSR_ROLE_TITLE_PATTERN = re.compile(
    r"(chief\s+csr\s+officer|head[\s,]*(?:of\s+)?csr|csr\s+head|csr\s+director|"
    r"chief\s+sustainability\s+officer|sustainability\s+head|head\s+of\s+sustainability|"
    r"vp[\s,\-]*csr|csr\s+manager|csr\s+lead|foundation\s+(?:ceo|director|head)|"
    r"esg\s+head|head\s+of\s+esg|social\s+impact\s+(?:head|lead|director))",
    re.IGNORECASE,
)

FORMER_ROLE_PATTERN = re.compile(
    r"\b(former|ex-|previously|until\s+\d{4}|retired|alumnus|alumni)\b", re.IGNORECASE
)

CIN_PATTERN = re.compile(r"\b[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}\b")

CURRENT_FY_LABEL = "FY2025-26"
PRIOR_FY_LABELS = ["FY2024-25", "FY2023-24", "2024-25", "2023-24"]

DOMAIN_QUERIES = [
    'site:{d} "{c}" official site',
]

CSR_PAGE_QUERIES = [
    'site:{d} CSR OR "corporate social responsibility" policy',
    '"{c}" CSR policy filetype:pdf site:{d}',
    '"{c}" "corporate social responsibility" report annual site:{d}',
]

CSR_PAGE_FALLBACK_QUERIES = [
    '"{c}" "CSR policy" filetype:pdf India',
    '"{c}" "corporate social responsibility policy" India official',
    '"{c}" CSR annual report India filetype:pdf',
]

MCA_CIN_QUERIES = [
    '"{c}" CIN site:mca.gov.in',
    '"{c}" "corporate identification number" India',
    '"{c}" CIN "roc" India company master data',
]

MCA_FILING_QUERIES = [
    '"{c}" "Form CSR-2" India filetype:pdf',
    '"{c}" "CSR-2" filing MCA India',
    '"{c}" CIN site:mca.gov.in "master data"',
]

NATIONAL_CSR_PORTAL_QUERIES = [
    'site:csr.gov.in "{c}"',
    '"{c}" site:csr.gov.in company profile',
    '"{c}" "csr.gov.in" India CSR company details',
]

ANNUAL_REPORT_QUERIES = [
    '"{c}" "annual report" {fy} India filetype:pdf',
    '"{c}" "integrated annual report" India filetype:pdf',
    '"{c}" "business responsibility" "sustainability report" India filetype:pdf',
    '"{c}" "schedule vii" "amount spent" CSR filetype:pdf',
]

CSR_SPEND_QUERIES = [
    '"{c}" "CSR expenditure" crore India {fy}',
    '"{c}" "amount spent" "CSR" crore annual report India',
    '"{c}" "total CSR" budget crore India',
    '"{c}" "2% of average net profit" CSR India',
]

PARTNER_QUERIES = [
    '"{c}" CSR "implementation partner" OR "implementing partner" India NGO',
    '"{c}" CSR "partnered with" NGO OR foundation OR trust India',
    '"{c}" foundation "grant recipients" OR "funded organisations" India CSR',
    '"{c}" CSR annual report "our partners" OR "ngo partners" India',
    '"{c}" foundation "NGOs supported" OR "NGO partners list" India CSR',
    '"{c}" CSR "companies supported" OR "organisations we support" India',
    '"{c}" CSR "supported NGOs" OR "beneficiary organisations" India filetype:pdf',
]

PLAN_QUERIES = [
    '"{c}" CSR "partnered with" OR "partnership with" education India announced',
    '"{c}" foundation NGO collaboration education programme India launched',
    '"{c}" CSR education initiative "will invest" OR announces India',
    '"{c}" CSR head OR CEO statement education skilling India interview',
]

RFP_QUERIES = [
    '"{c}" CSR "request for proposal" OR "call for proposals" India',
    '"{c}" CSR "invite NGOs" OR "open call for partners" India',
    '"{c}" foundation "apply for grant" OR "grant application" education India',
]

LINKEDIN_PEOPLE_QUERIES = [
    'site:linkedin.com/in "{c}" "head of CSR" OR "CSR head"',
    'site:linkedin.com/in "{c}" "corporate social responsibility"',
    'site:linkedin.com/in "{c}" sustainability head OR director India',
    'site:linkedin.com/in "{c}" "CSR committee" OR foundation India',
]

SECTOR_QUERIES = [
    '"{c}" India sector industry business overview annual report',
    '"{c}" India revenue OR turnover OR "net worth" annual report {fy}',
]

GROUP_FOUNDATION_QUERIES = [
    '"{c}" "group foundation" CSR India',
    '"{c}" CSR "routed through" OR "implemented through" foundation India',
    '"{c}" "parent company" CSR foundation India trust',
]

MAX_PAGE_TEXT_CHARS = 9000
MAX_PDF_TEXT_CHARS = 20000
MAX_PDF_PAGES = 60
FINANCIAL_PDF_SCAN_PAGES = 90
CANDIDATE_EVAL_LIMIT = 6
MIN_ACCEPT_SCORE = 6
STRONG_ACCEPT_SCORE = 10


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
        "social responsibility",
    ]
    return sum(1 for kw in relevance_keywords if kw in lowered) >= 2


def has_financial_figures(text: str) -> bool:
    return bool(CURRENCY_FIGURE_PATTERN.search(text))


def count_financial_figures(text: str) -> int:
    return len(CURRENCY_FIGURE_PATTERN.findall(text))


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
    return [f"www.{slug}{tld}" for slug in slugs if slug for tld in (".com", ".in", ".co.in")]


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
    length_bonus = min(len(text) / 2000.0, 4.0)
    domain_bonus = 6.0 if url and any(gov in url.lower() for gov in OFFICIAL_GOV_DOMAINS) else 0.0
    pdf_bonus = 1.5 if url.lower().endswith(".pdf") else 0.0
    return csr_hits * 2.0 + figure_hits * 5.0 + length_bonus + domain_bonus + pdf_bonus


def ddgs_search_web(query: str, max_results: int = 5) -> list[dict]:
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results, backend="html"))
    except Exception as exc:
        logger.warning("ddgs search failed query=%r error=%s", query, exc)
        return []


async def search_web(query: str, max_results: int = 6, prefer_google: bool = True, quota_guard=None) -> list[dict]:
    if prefer_google and google_search.google_search_configured_and_available(quota_guard):
        results = await google_search.google_search_web(query, max_results=max_results, quota_guard=quota_guard)
        if results:
            return results
        logger.info("google search returned empty, falling back to ddgs query=%r", query)
    else:
        if prefer_google:
            logger.info("google search not available, using ddgs query=%r", query)
    return ddgs_search_web(query, max_results=max_results)


def fetch_page_text(url: str, max_chars: int = MAX_PAGE_TEXT_CHARS, verify_ssl: bool = True) -> str:
    try:
        response = get_session().get(url, timeout=15, verify=verify_ssl)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        return extract_main_text(soup, max_chars)
    except Exception as exc:
        if "404" in str(exc):
            logger.debug("fetch_page_text 404 (expected for guessed paths) url=%s", url)
        else:
            logger.info("fetch_page_text failed url=%s error=%s", url, exc)
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


def fetch_pdf_text(url: str, max_chars: int = MAX_PDF_TEXT_CHARS, max_pages: int = MAX_PDF_PAGES) -> str:
    try:
        import io
        import pdfplumber

        response = get_session().get(url, timeout=45)
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


def sitemap_csr_urls(domain: str, limit: int = 8) -> list[str]:
    try:
        response = get_session().get(f"https://{domain}/sitemap.xml", timeout=10)
        response.raise_for_status()
        urls = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", response.text)[:1200]
        matched = [url for url in urls if CSR_LINK_PATTERN.search(url)]
        if matched:
            return matched[:limit]
        nested_sitemaps = [url for url in urls if url.endswith(".xml")][:6]
        for nested_url in nested_sitemaps:
            try:
                nested_response = get_session().get(nested_url, timeout=10)
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


async def discover_company_domain(company: str, search_cfg: dict, quota_guard=None) -> str:
    domains = await discover_company_domains(company, search_cfg, quota_guard)
    return domains[0] if domains else ""


async def discover_company_domains(company: str, search_cfg: dict, quota_guard=None) -> list[str]:
    tokens = company_name_tokens(company)
    acronym = "".join(token[0] for token in tokens) if len(tokens) >= 2 else ""
    results = await search_web(
        f'"{company}" official website India',
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


async def fetch_india_csr_page(company: str, search_cfg: dict, quota_guard=None, max_fetches: int = 24) -> dict:
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
            best_candidate[0] = (score, source)

    def try_fetch(url: str, method: str, is_pdf: bool = False):
        if not url or url in tried_urls or remaining_budget[0] <= 0:
            return
        tried_urls.add(url)
        remaining_budget[0] -= 1
        text = fetch_pdf_text(url) if is_pdf else fetch_page_text(url)
        consider(url, method, text)

    discovered_domains = await discover_company_domains(company, search_cfg, quota_guard)
    domains = list(dict.fromkeys(discovered_domains + candidate_domains(company)))
    logger.info("india_csr_page discovered_domains company=%r domains=%s", company, domains)

    live_homepages = []
    for domain in domains:
        if remaining_budget[0] <= 0 or len(live_homepages) >= 3:
            break
        try:
            response = get_session().get(f"https://{domain}", timeout=12)
            remaining_budget[0] -= 1
            if response.ok and mentions_company(company, response.text):
                live_homepages.append((domain, response.text))
        except Exception as exc:
            logger.info("homepage fetch failed domain=%s error=%s", domain, exc)
            continue

    logger.info("india_csr_page live_homepages company=%r count=%d", company, len(live_homepages))

    if live_homepages:
        resolved_domain[0] = live_homepages[0][0]

    candidates_checked = 0
    for domain, homepage_html in live_homepages:
        links = csr_links_from_html(f"https://{domain}", homepage_html)
        for link in links:
            if candidates_checked >= CANDIDATE_EVAL_LIMIT and best_candidate[0] and best_candidate[0][0] >= STRONG_ACCEPT_SCORE:
                break
            try_fetch(link, "homepage_link", is_pdf=link.lower().endswith(".pdf"))
            candidates_checked += 1
        for path in CSR_PAGE_PATHS:
            if candidates_checked >= CANDIDATE_EVAL_LIMIT and best_candidate[0] and best_candidate[0][0] >= STRONG_ACCEPT_SCORE:
                break
            try_fetch(f"https://{domain}{path}", "direct")
            candidates_checked += 1
        for sitemap_url in sitemap_csr_urls(domain):
            if candidates_checked >= CANDIDATE_EVAL_LIMIT and best_candidate[0] and best_candidate[0][0] >= STRONG_ACCEPT_SCORE:
                break
            try_fetch(sitemap_url, "sitemap", is_pdf=sitemap_url.lower().endswith(".pdf"))
            candidates_checked += 1

    remaining_budget[0] = max(remaining_budget[0], 10)
    if not best_candidate[0] or best_candidate[0][0] < STRONG_ACCEPT_SCORE:
        query_pool = []
        for domain, _ in live_homepages:
            for template in CSR_PAGE_QUERIES:
                query_pool.append(template.format(c=company, d=domain))
        query_pool.extend(template.format(c=company) for template in CSR_PAGE_FALLBACK_QUERIES)

        for query in query_pool:
            results = await search_web(
                query, max_results=6, prefer_google=search_cfg.get("csr_pages", True), quota_guard=quota_guard
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
                try_fetch(url, "search", is_pdf=url.lower().endswith(".pdf"))
                consider(url, "snippet", snippet_body)
            if best_candidate[0] and best_candidate[0][0] >= STRONG_ACCEPT_SCORE:
                break

    if best_candidate[0]:
        logger.info(
            "india_csr_page DONE company=%r found=True score=%.1f url=%s",
            company, best_candidate[0][0], best_candidate[0][1].get("url", ""),
        )
        return best_candidate[0][1]

    logger.info("india_csr_page DONE company=%r found=False", company)
    fallback = make_source("india_csr_page", 1, status="NOT_FOUND")
    fallback["domain"] = resolved_domain[0]
    return fallback


async def find_company_cin(company: str, search_cfg: dict, quota_guard=None) -> str:
    for query_template in MCA_CIN_QUERIES:
        results = await search_web(
            query_template.format(c=company),
            max_results=5,
            prefer_google=search_cfg.get("mca", True),
            quota_guard=quota_guard,
        )
        for result in results:
            body = result.get("body", "") + " " + result.get("title", "") + " " + result.get("href", "")
            match = CIN_PATTERN.search(body)
            if match:
                return match.group(0)
    return ""


async def fetch_mca_company_data_gov_page(cin: str) -> str:
    if not cin:
        return ""
    candidate_urls = [
        f"https://www.mca.gov.in/mcafoportal/viewCompanyMasterData.do?cid={cin}",
        f"https://www.mca.gov.in/content/mca/global/en/mca/master-data/MDS.html?cin={cin}",
    ]
    for url in candidate_urls:
        text = fetch_page_text(url)
        if text and len(text) > 150:
            return text
    return ""


async def fetch_mca_portal(company: str, search_cfg: dict, quota_guard=None) -> dict:
    cin = await find_company_cin(company, search_cfg, quota_guard)

    if cin:
        mca_text = await fetch_mca_company_data_gov_page(cin)
        if mca_text and mentions_company(company, mca_text):
            source = make_source(
                "mca_portal", 2,
                f"https://www.mca.gov.in/mcafoportal/viewCompanyMasterData.do?cid={cin}",
                mca_text, "FOUND", "direct",
            )
            source["cin"] = cin
            return source

    best_candidate = None
    for query_template in MCA_FILING_QUERIES:
        results = await search_web(
            query_template.format(c=company), max_results=6,
            prefer_google=search_cfg.get("mca", True), quota_guard=quota_guard,
        )
        for result in results:
            url = result.get("href", "")
            body = result.get("body", "")
            if not url:
                continue
            text = fetch_pdf_text(url) if url.lower().endswith(".pdf") else fetch_page_text(url)
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
                        found_cin = cin_match.group(0)
                source = make_source("mca_via_search", 2, url, text, "FOUND", "search_proxy")
                if found_cin:
                    source["cin"] = found_cin
                best_candidate = (score, source)
        if best_candidate and best_candidate[0] >= STRONG_ACCEPT_SCORE:
            break

    if best_candidate:
        logger.info("mca_portal DONE company=%r found=True cin=%s", company, cin or best_candidate[1].get("cin", ""))
        return best_candidate[1]

    logger.info("mca_portal DONE company=%r found=False cin=%s", company, cin)
    return make_source("mca_portal", 2, status="NOT_FOUND")


async def fetch_national_csr_portal(company: str, search_cfg: dict, quota_guard=None) -> dict:
    company_query = company.replace(" ", "+")
    direct_urls = [
        f"https://www.csr.gov.in/content/csr/global/master/home/companydetail.html?companyName={company_query}",
        f"https://csr.gov.in/csr/companyprofile?company_name={company_query}",
    ]
    for url in direct_urls:
        text = fetch_page_text(url)
        if text and len(text) > 250 and mentions_company(company, text):
            return make_source("national_csr_portal", 3, url, text, "FOUND", "direct")

    best_candidate = None
    for query_template in NATIONAL_CSR_PORTAL_QUERIES:
        results = await search_web(
            query_template.format(c=company), max_results=6,
            prefer_google=search_cfg.get("mca", True), quota_guard=quota_guard,
        )
        for result in results[:6]:
            url = result.get("href", "")
            body = result.get("body", "")
            if not url:
                continue
            text = fetch_page_text(url) or body
            if text and mentions_company(company, text) and ("csr.gov.in" in url.lower() or is_csr_relevant(text)):
                score = score_candidate_text(company, text, url)
                if best_candidate is None or score > best_candidate[0]:
                    best_candidate = (score, make_source("national_csr_portal", 3, url, text, "FOUND", "search"))
        if best_candidate and best_candidate[0] >= STRONG_ACCEPT_SCORE:
            break

    if best_candidate:
        logger.info("national_csr_portal DONE company=%r found=True", company)
        return best_candidate[1]

    logger.info("national_csr_portal DONE company=%r found=False", company)
    return make_source("national_csr_portal", 3, status="NOT_FOUND")


async def fetch_annual_report(company: str, search_cfg: dict, quota_guard=None) -> dict:
    best_candidate = None
    urls_tried = 0
    for template in ANNUAL_REPORT_QUERIES:
        query = template.format(c=company, fy=CURRENT_FY_LABEL)
        results = await search_web(
            query,
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
                text = fetch_pdf_text(url)
                if not text and body and len(body) > 100 and mentions_company(company, body):
                    text = body
                if text and not pdf_is_csr_relevant(text):
                    logger.info("annual_report rejected non-csr pdf company=%r url=%s", company, url)
                    continue
            else:
                text = fetch_page_text(url) or body
            if not (text and len(text) > 250 and mentions_company(company, text)):
                continue
            score = score_candidate_text(company, text, url)
            if best_candidate is None or score > best_candidate[0]:
                fetch_method = "pdf" if url.lower().endswith(".pdf") else "search"
                best_candidate = (score, make_source("annual_report", 4, url, text, "FOUND", fetch_method))
        if best_candidate and best_candidate[0] >= STRONG_ACCEPT_SCORE + 4:
            break

    if not best_candidate or count_financial_figures(best_candidate[1].get("text", "")) == 0:
        for fy in PRIOR_FY_LABELS:
            query = f'"{company}" "annual report" {fy} CSR filetype:pdf'
            results = await search_web(
                query, max_results=6, prefer_google=search_cfg.get("annual_reports", True), quota_guard=quota_guard
            )
            for result in results:
                url = result.get("href", "")
                title = result.get("title", "")
                body = result.get("body", "")
                if not url or not url.lower().endswith(".pdf"):
                    continue
                if not mentions_company(company, f"{title} {body}") or not url_belongs_to_company(company, url):
                    continue
                text = fetch_pdf_text(url)
                if text and not pdf_is_csr_relevant(text):
                    continue
                if text and has_financial_figures(text) and mentions_company(company, text):
                    score = score_candidate_text(company, text, url)
                    if best_candidate is None or score > best_candidate[0]:
                        best_candidate = (score, make_source("annual_report", 4, url, text, "FOUND", "pdf_prior_fy"))
            if best_candidate and count_financial_figures(best_candidate[1].get("text", "")) > 0:
                break

    logger.info(
        "annual_report DONE company=%r urls_tried=%d found=%s figures=%d",
        company, urls_tried, bool(best_candidate),
        count_financial_figures(best_candidate[1].get("text", "")) if best_candidate else 0,
    )

    if best_candidate:
        return best_candidate[1]

    return make_source("annual_report", 4, status="NOT_FOUND")


async def fetch_partner_source(company: str, search_cfg: dict, quota_guard=None) -> dict:
    best_candidate = None
    urls_tried = 0
    for template in PARTNER_QUERIES:
        query = template.format(c=company)
        results = await search_web(query, max_results=6, prefer_google=search_cfg.get("partners", True), quota_guard=quota_guard)
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
            text = (fetch_pdf_text(url) if is_pdf else fetch_page_text(url)) or body
            if is_pdf and text and not pdf_is_csr_relevant(text):
                continue
            if not (text and len(text) > 250 and is_csr_relevant(text) and mentions_company(company, text)):
                continue
            score = score_candidate_text(company, text, url)
            if best_candidate is None or score > best_candidate[0]:
                best_candidate = (score, make_source("partner_search", 5, url, text, "FOUND", "search"))
        if best_candidate and best_candidate[0] >= STRONG_ACCEPT_SCORE:
            break

    logger.info("partner_search DONE company=%r urls_tried=%d found=%s", company, urls_tried, bool(best_candidate))

    if best_candidate:
        return best_candidate[1]

    return make_source("partner_search", 5, status="NOT_FOUND")


def _looks_like_current_csr_role(title_and_snippet: str) -> bool:
    if not CSR_ROLE_TITLE_PATTERN.search(title_and_snippet):
        return False
    if FORMER_ROLE_PATTERN.search(title_and_snippet):
        return False
    return True


async def fetch_linkedin_people(company: str, search_cfg: dict, quota_guard=None) -> dict:
    hits: list[dict] = []
    seen_urls: set[str] = set()

    if search_cfg.get("linkedin_people", True):
        for query_template in LINKEDIN_PEOPLE_QUERIES:
            role_hint = ""
            if '"' in query_template:
                parts = query_template.split('"')
                if len(parts) >= 4:
                    role_hint = parts[3]
            profiles = await google_search.google_search_linkedin_profiles(
                company, role_hint=role_hint, max_results=8, quota_guard=quota_guard,
            )
            for profile in profiles:
                url = profile.get("href", "")
                if not is_literal_linkedin_profile_url(url) or url in seen_urls:
                    continue
                title = profile.get("title", "")
                snippet = profile.get("body", "")
                if not mentions_company(company, f"{title} {snippet}"):
                    continue
                seen_urls.add(url)
                hits.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "confidence": "HIGH" if _looks_like_current_csr_role(f"{title} {snippet}") else "LOW",
                })
            if len(hits) >= 15:
                break

    if not hits:
        for query in LINKEDIN_PEOPLE_QUERIES:
            fallback_results = await search_web(
                query.format(c=company) if "{c}" in query else query.replace("{c}", company),
                max_results=8, prefer_google=search_cfg.get("linkedin_people", True), quota_guard=quota_guard,
            )
            for result in fallback_results:
                url = result.get("href", "")
                if not is_literal_linkedin_profile_url(url) or url in seen_urls:
                    continue
                title = result.get("title", "")
                snippet = result.get("body", "")
                if not mentions_company(company, f"{title} {snippet}"):
                    continue
                seen_urls.add(url)
                hits.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "confidence": "HIGH" if _looks_like_current_csr_role(f"{title} {snippet}") else "LOW",
                })
            if len(hits) >= 10:
                break

    if not hits:
        return make_source("people_search", 6, status="NOT_FOUND")

    hits.sort(key=lambda h: h.get("confidence") != "HIGH")
    high_confidence_hits = [h for h in hits if h.get("confidence") == "HIGH"]
    final_hits = high_confidence_hits[:10] if high_confidence_hits else hits[:6]

    combined_text = " || ".join(f"{hit['title']} — {hit['snippet']}" for hit in final_hits)
    source = make_source("people_search", 6, final_hits[0]["url"], clean_text(combined_text, 4000), "FOUND", "search_snippets")
    source["people_hits"] = final_hits
    return source


async def fetch_plans_source(company: str, search_cfg: dict, quota_guard=None, max_pages: int = 3) -> dict:
    hits, fetched_texts, first_url = [], [], ""
    all_queries = PLAN_QUERIES + RFP_QUERIES
    for query_template in all_queries:
        results = await search_web(
            query_template.format(c=company), max_results=5, prefer_google=search_cfg.get("partners", True), quota_guard=quota_guard
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
            if len(fetched_texts) < max_pages:
                text = (fetch_pdf_text(url) if url.lower().endswith(".pdf") else fetch_page_text(url)) or body
                if text and len(text) > 250 and is_csr_relevant(text) and mentions_company(company, text):
                    fetched_texts.append(text)
                    first_url = first_url or url
        if len(fetched_texts) >= max_pages:
            break

    if not hits and not fetched_texts:
        return make_source("plans_search", 7, status="NOT_FOUND")

    combined_text = " || ".join(fetched_texts) if fetched_texts else " || ".join(f"{hit['title']} — {hit['snippet']}" for hit in hits)
    source = make_source(
        "plans_search", 7, first_url or hits[0]["url"], clean_text(combined_text, 7000), "FOUND",
        "search" if fetched_texts else "search_snippets",
    )
    source["plan_hits"] = hits[:10]
    return source


async def fetch_sector_eligibility_source(company: str, search_cfg: dict, quota_guard=None) -> dict:
    hits, fetched_texts, first_url = [], [], ""
    queries = [t.format(c=company, fy=CURRENT_FY_LABEL) for t in SECTOR_QUERIES] + \
              [t.format(c=company) for t in GROUP_FOUNDATION_QUERIES]
    for query in queries:
        results = await search_web(
            query, max_results=5,
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
                text = fetch_page_text(url) or body
                if text and len(text) > 200 and mentions_company(company, text):
                    fetched_texts.append(text)
                    first_url = first_url or url
        if len(fetched_texts) >= 3:
            break

    if not hits and not fetched_texts:
        return make_source("sector_eligibility_search", 8, status="NOT_FOUND")

    combined_text = " || ".join(fetched_texts) if fetched_texts else " || ".join(f"{hit['title']} — {hit['snippet']}" for hit in hits)
    return make_source(
        "sector_eligibility_search", 8, first_url or hits[0]["url"], clean_text(combined_text, 6000), "FOUND",
        "search" if fetched_texts else "search_snippets",
    )


async def fetch_screen_sources(company: str, search_cfg: dict, quota_guard=None) -> list[dict]:
    source_1 = await fetch_india_csr_page(company, search_cfg, quota_guard)
    source_4 = await fetch_annual_report(company, search_cfg, quota_guard)
    source_2 = await fetch_mca_portal(company, search_cfg, quota_guard)
    source_5 = await fetch_partner_source(company, search_cfg, quota_guard)

    source_3 = make_source("national_csr_portal", 3, status="NOT_TRIED")
    source_6 = make_source("people_search", 6, status="NOT_TRIED")
    source_7 = make_source("plans_search", 7, status="NOT_TRIED")
    source_8 = make_source("sector_eligibility_search", 8, status="NOT_TRIED")

    sources = [source_1, source_2, source_3, source_4, source_5, source_6, source_7, source_8]
    found_count = sum(1 for s in sources if s.get("status") == "FOUND")
    logger.info("fetch_screen_sources DONE company=%r found=%d/8", company, found_count)
    return sources


async def fetch_deep_sources(company: str, search_cfg: dict, quota_guard=None, progress_cb=None) -> list[dict]:
    async def advance_step(message: str):
        if progress_cb:
            await progress_cb(message)

    await advance_step("Source 1/8 — India CSR page...")
    source_1 = await fetch_india_csr_page(company, search_cfg, quota_guard)
    logger.info(
        "deep source 1 company=%r status=%s figures=%d",
        company, source_1.get("status"), count_financial_figures(source_1.get("text", "")),
    )

    await advance_step("Source 2/8 — MCA portal + CIN lookup...")
    source_2 = await fetch_mca_portal(company, search_cfg, quota_guard)

    await advance_step("Source 3/8 — National CSR Portal...")
    source_3 = await fetch_national_csr_portal(company, search_cfg, quota_guard)

    await advance_step("Source 4/8 — Annual report...")
    source_4 = await fetch_annual_report(company, search_cfg, quota_guard)
    logger.info(
        "deep source 4 company=%r status=%s figures=%d",
        company, source_4.get("status"), count_financial_figures(source_4.get("text", "")),
    )

    await advance_step("Source 5/8 — Funded partners...")
    source_5 = await fetch_partner_source(company, search_cfg, quota_guard)

    await advance_step("Source 6/8 — CSR decision-makers (LinkedIn)...")
    source_6 = await fetch_linkedin_people(company, search_cfg, quota_guard)

    await advance_step("Source 7/8 — Partnerships, announced plans & open calls...")
    source_7 = await fetch_plans_source(company, search_cfg, quota_guard)

    await advance_step("Source 8/8 — Sector, eligibility & group-foundation signals...")
    source_8 = await fetch_sector_eligibility_source(company, search_cfg, quota_guard)

    sources = [source_1, source_2, source_3, source_4, source_5, source_6, source_7, source_8]
    total_figures = sum(count_financial_figures(s.get("text", "")) for s in sources)
    if total_figures == 0:
        for template in CSR_SPEND_QUERIES:
            query = template.format(c=company, fy=CURRENT_FY_LABEL)
            results = await search_web(
                query, max_results=6,
                prefer_google=search_cfg.get("csr_pages", True), quota_guard=quota_guard,
            )
            for result in results:
                url = result.get("href", "")
                body = result.get("body", "")
                if not url or not mentions_company(company, body):
                    continue
                text = (fetch_pdf_text(url) if url.lower().endswith(".pdf") else fetch_page_text(url)) or body
                if text and has_financial_figures(text) and mentions_company(company, text):
                    source_4 = make_source("annual_report", 4, url, text, "FOUND", "spend_fallback")
                    sources[3] = source_4
                    logger.info("deep fallback spend search recovered figures company=%r url=%s", company, url)
                    break
            if count_financial_figures(sources[3].get("text", "")) > 0:
                break

    return sources