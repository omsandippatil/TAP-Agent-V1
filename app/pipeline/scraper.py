
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from app.pipeline import google_search
from app.pipeline.utils import clean_text, extract_main_text, get_session, make_source

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
)

CSR_LINK_PATTERN = re.compile(
    r"(csr|corporate[\s_-]?social|social[\s_-]?responsib|sustainab|esg|"
    r"social[\s_-]?impact|citizenship|community[\s_-]?(initiativ|develop|engag)|"
    r"responsible[\s_-]?business|foundation|giving[\s_-]?back)",
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
]

CSR_KEYWORDS = [
    "csr", "corporate social", "philanthrop", "social responsibility",
    "schedule vii", "csr spend", "csr expenditure", "csr budget",
    "csr obligation", "csr fund", "community investment",
]

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

PARTNERSHIP_PLAN_QUERIES = [
    '"{c}" CSR "partnered with" OR "partnership with" NGO education India',
    '"{c}" foundation NGO collaboration education programme India announced',
    '"{c}" CSR education initiative announces OR launch OR "will invest" India',
    '"{c}" "CSR head" OR CEO education skilling India interview statement',
]

RFP_OPEN_CALL_QUERIES = [
    '"{c}" CSR "request for proposal" OR "call for proposals" India',
    '"{c}" CSR "looking for partners" OR "invite NGOs" OR "open call" India',
    '"{c}" foundation "apply for grant" OR "grant application" education India',
]

SECTOR_SIGNAL_QUERIES = [
    '"{c}" India sector industry business overview',
    '"{c}" India net worth OR turnover OR revenue annual report',
]

GROUP_FOUNDATION_QUERIES = [
    '"{c}" "group foundation" OR "parent foundation" CSR India',
    '"{c}" CSR "routed through" OR "central foundation" OR "group trust" India',
]

LINKEDIN_PEOPLE_QUERIES = [
    'site:linkedin.com/in "{c}" CSR OR "corporate social responsibility"',
    'site:linkedin.com/in "{c}" sustainability OR ESG',
    'site:linkedin.com/in "{c}" "head of CSR" OR "CSR head" OR "foundation"',
]

MCA_SEARCH_QUERIES = [
    '"{c}" India "CSR-2" OR "Form CSR-2" OR "CSR committee" site:mca.gov.in',
    '"{c}" India "CSR committee" "CSR obligation" crore annual report filing',
    '"{c}" CIN "corporate identification number" India',
]

NATIONAL_CSR_PORTAL_QUERIES = [
    'site:csr.gov.in "{c}"',
    '"{c}" site:csr.gov.in',
    '"{c}" "csr.gov.in" CSR portal India company profile',
]

MAX_PAGE_TEXT_CHARS = 6000
MAX_PDF_TEXT_CHARS = 9000


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


def accept_fetched_text(company: str, text: str, min_len: int = 400) -> bool:
    return bool(text) and len(text) > min_len and is_csr_relevant(text) and mentions_company(company, text)


def ddgs_search_web(query: str, max_results: int = 5) -> list[dict]:
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception:
        return []


async def search_web(query: str, max_results: int = 5, prefer_google: bool = False, quota_guard=None) -> list[dict]:
    if prefer_google and google_search.google_search_configured_and_available(quota_guard):
        results = await google_search.google_search_web(query, max_results=max_results, quota_guard=quota_guard)
        if results:
            return results
    return ddgs_search_web(query, max_results=max_results)


def fetch_page_text(url: str, max_chars: int = MAX_PAGE_TEXT_CHARS, verify_ssl: bool = True) -> str:
    try:
        response = get_session().get(url, timeout=12, verify=verify_ssl)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        return extract_main_text(soup, max_chars)
    except Exception:
        return ""


def fetch_pdf_text(url: str, max_chars: int = MAX_PDF_TEXT_CHARS, max_pages: int = 30) -> str:
    try:
        import io
        import pdfplumber

        response = get_session().get(url, timeout=35)
        response.raise_for_status()
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            pages_text = []
            total_len = 0
            for page in pdf.pages[:max_pages]:
                page_text = page.extract_text() or ""
                if page_text:
                    pages_text.append(page_text)
                    total_len += len(page_text)
                if total_len >= max_chars:
                    break
        return clean_text(" ".join(pages_text), max_chars)
    except Exception:
        return ""


def csr_links_from_html(base_url: str, html: str, limit: int = 6) -> list[str]:
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
            if score:
                scored_links.append((score, urljoin(base_url, href)))
    except Exception:
        pass
    scored_links.sort(key=lambda item: -item[0])
    seen_urls, ordered_urls = set(), []
    for _, url in scored_links:
        if url not in seen_urls:
            seen_urls.add(url)
            ordered_urls.append(url)
        if len(ordered_urls) >= limit:
            break
    return ordered_urls


def sitemap_csr_urls(domain: str, limit: int = 5) -> list[str]:
    try:
        response = get_session().get(f"https://{domain}/sitemap.xml", timeout=10)
        response.raise_for_status()
        urls = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", response.text)[:800]
        matched = [url for url in urls if CSR_LINK_PATTERN.search(url)]
        if matched:
            return matched[:limit]
        nested_sitemaps = [url for url in urls if url.endswith(".xml")][:5]
        for nested_url in nested_sitemaps:
            try:
                nested_response = get_session().get(nested_url, timeout=10)
                nested_response.raise_for_status()
                nested_urls = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", nested_response.text)[:800]
                nested_matched = [url for url in nested_urls if CSR_LINK_PATTERN.search(url)]
                if nested_matched:
                    return nested_matched[:limit]
            except Exception:
                continue
    except Exception:
        pass
    return []


async def discover_company_domain(company: str, search_cfg: dict, quota_guard=None) -> str:
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
    return matched_domains[0] if matched_domains else ""


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


async def fetch_india_csr_page(company: str, search_cfg: dict, quota_guard=None, max_fetches: int = 16) -> dict:
    tried_urls = set()
    remaining_budget = [max_fetches]
    resolved_domain = [""]

    def try_fetch(url: str, method: str, min_len: int = 400, is_pdf: bool = False):
        if not url or url in tried_urls or remaining_budget[0] <= 0:
            return None
        tried_urls.add(url)
        remaining_budget[0] -= 1
        text = fetch_pdf_text(url) if is_pdf else fetch_page_text(url)
        if accept_fetched_text(company, text, min_len):
            source = make_source("india_csr_page", 1, url, text, "FOUND", method)
            source["domain"] = urlparse(url).netloc.lower()
            return source
        return None

    discovered_domains = await discover_company_domains(company, search_cfg, quota_guard)
    domains = list(dict.fromkeys(discovered_domains + candidate_domains(company)))

    live_homepages = []
    for domain in domains:
        if remaining_budget[0] <= 0 or len(live_homepages) >= 3:
            break
        try:
            response = get_session().get(f"https://{domain}", timeout=10)
            remaining_budget[0] -= 1
            if response.ok and mentions_company(company, response.text):
                live_homepages.append((domain, response.text))
        except Exception:
            continue

    if live_homepages:
        resolved_domain[0] = live_homepages[0][0]

    for domain, homepage_html in live_homepages:
        for link in csr_links_from_html(f"https://{domain}", homepage_html):
            source = try_fetch(link, "homepage_link", is_pdf=link.lower().endswith(".pdf"))
            if source:
                return source
        for path in CSR_PAGE_PATHS:
            source = try_fetch(f"https://{domain}{path}", "direct")
            if source:
                return source
        for sitemap_url in sitemap_csr_urls(domain):
            source = try_fetch(sitemap_url, "sitemap", is_pdf=sitemap_url.lower().endswith(".pdf"))
            if source:
                return source

    remaining_budget[0] = max(remaining_budget[0], 5)
    for query in [
        f'"{company}" India CSR sustainability "corporate social"',
        f'"{company}" "corporate social responsibility" India initiatives',
        f'"{company}" CSR report India filetype:pdf',
        f'"{company}" sustainability report India filetype:pdf',
    ]:
        results = await search_web(
            query, max_results=6, prefer_google=search_cfg.get("csr_pages", True), quota_guard=quota_guard
        )
        for result in results:
            url = result.get("href", "")
            snippet_body = result.get("body", "")
            if not url or any(domain in url for domain in AGGREGATOR_DOMAINS):
                continue
            source = try_fetch(url, "search", min_len=250, is_pdf=url.lower().endswith(".pdf"))
            if source:
                return source
            if accept_fetched_text(company, snippet_body, 250):
                source = make_source("india_csr_page", 1, url, snippet_body, "FOUND", "snippet")
                source["domain"] = urlparse(url).netloc.lower()
                return source

    fallback = make_source("india_csr_page", 1, status="NOT_FOUND")
    fallback["domain"] = resolved_domain[0]
    return fallback


async def find_company_cin(company: str, search_cfg: dict, quota_guard=None) -> str:
    results = await search_web(
        f'"{company}" CIN "corporate identification number" India',
        max_results=4,
        prefer_google=search_cfg.get("mca", True),
        quota_guard=quota_guard,
    )
    for result in results:
        body = result.get("body", "") + result.get("title", "")
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

    for query_template in MCA_SEARCH_QUERIES:
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
            found_cin = cin or ""
            if not found_cin:
                cin_match = CIN_PATTERN.search(text) or CIN_PATTERN.search(body)
                if cin_match:
                    found_cin = cin_match.group(0)
            source = make_source("mca_via_search", 2, url, text, "FOUND", "search_proxy")
            if found_cin:
                source["cin"] = found_cin
            return source

    return make_source("mca_portal", 2, status="NOT_FOUND")


async def fetch_national_csr_portal(company: str, search_cfg: dict, quota_guard=None, verify_ssl: bool = True) -> dict:
    company_query = company.replace(" ", "+")
    direct_urls = [
        f"https://www.csr.gov.in/content/csr/global/master/home/companydetail.html?companyName={company_query}",
        f"https://csr.gov.in/csr/companyprofile?company_name={company_query}",
    ]
    for url in direct_urls:
        text = fetch_page_text(url, verify_ssl=verify_ssl)
        if not text and verify_ssl:
            text = fetch_page_text(url, verify_ssl=False)
        if text and len(text) > 250 and mentions_company(company, text):
            return make_source("national_csr_portal", 3, url, text, "FOUND", "direct")

    for query_template in NATIONAL_CSR_PORTAL_QUERIES:
        results = await search_web(
            query_template.format(c=company), max_results=6,
            prefer_google=search_cfg.get("mca", True), quota_guard=quota_guard,
        )
        for result in results[:5]:
            url = result.get("href", "")
            body = result.get("body", "")
            if not url:
                continue
            text = fetch_page_text(url, verify_ssl=verify_ssl) if url else ""
            if not text and url and verify_ssl:
                text = fetch_page_text(url, verify_ssl=False)
            if not text:
                text = body
            if text and mentions_company(company, text) and ("csr" in url.lower() or is_csr_relevant(text)):
                return make_source("national_csr_portal", 3, url, text, "FOUND", "search")

    return make_source("national_csr_portal", 3, status="NOT_FOUND")


async def fetch_annual_report(company: str, search_cfg: dict, quota_guard=None) -> dict:
    results = await search_web(
        f"{company} annual report sustainability CSR India",
        max_results=8,
        prefer_google=search_cfg.get("annual_reports", True),
        quota_guard=quota_guard,
    )
    for result in results:
        url = result.get("href", "")
        body = result.get("body", "")
        if not url:
            continue
        if url.lower().endswith(".pdf"):
            text = fetch_pdf_text(url)
            if text and len(text) > 350 and mentions_company(company, text):
                return make_source("annual_report", 4, url, text, "FOUND", "pdf")
            if body and len(body) > 100 and mentions_company(company, body):
                return make_source("annual_report", 4, url, body, "FOUND", "snippet")
            continue
        text = fetch_page_text(url) or body
        if text and len(text) > 250 and mentions_company(company, text):
            return make_source("annual_report", 4, url, text, "FOUND", "search")

    return make_source("annual_report", 4, status="NOT_FOUND")


async def fetch_partner_source(company: str, search_cfg: dict, quota_guard=None) -> dict:
    for query in [
        f'"{company}" CSR "implementation partner" OR "implementing partner" OR "NGO partner" India',
        f'"{company}" CSR "partnered with" foundation OR trust OR NGO India',
        f'"{company}" foundation grant recipients NGO India CSR',
    ]:
        results = await search_web(query, max_results=6, prefer_google=search_cfg.get("partners", True), quota_guard=quota_guard)
        for result in results:
            url = result.get("href", "")
            body = result.get("body", "")
            if not url or any(domain in url for domain in ["youtube", "twitter", "facebook"]):
                continue
            text = (fetch_pdf_text(url) if url.lower().endswith(".pdf") else fetch_page_text(url)) or body
            if text and len(text) > 250 and is_csr_relevant(text) and mentions_company(company, text):
                return make_source("partner_search", 5, url, text, "FOUND", "search")

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
            profiles = await google_search.google_search_linkedin_profiles(
                company, role_hint=query_template.split('"')[-2] if '"' in query_template else "",
                max_results=8, quota_guard=quota_guard,
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
        fallback_results = await search_web(
            f'"{company}" "CSR" OR "sustainability" head OR director OR manager LinkedIn India',
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

    if not hits:
        return make_source("people_search", 6, status="NOT_FOUND")

    hits.sort(key=lambda h: h.get("confidence") != "HIGH")
    high_confidence_hits = [h for h in hits if h.get("confidence") == "HIGH"]
    final_hits = high_confidence_hits[:10] if high_confidence_hits else hits[:6]

    combined_text = " || ".join(f"{hit['title']} — {hit['snippet']}" for hit in final_hits)
    source = make_source("people_search", 6, final_hits[0]["url"], clean_text(combined_text, 4000), "FOUND", "search_snippets")
    source["people_hits"] = final_hits
    return source


async def fetch_plans_source(company: str, search_cfg: dict, quota_guard=None, max_pages: int = 2) -> dict:
    hits, fetched_texts, first_url = [], [], ""
    all_queries = PARTNERSHIP_PLAN_QUERIES + RFP_OPEN_CALL_QUERIES
    for query_template in all_queries:
        results = await search_web(
            query_template.format(c=company), max_results=5, prefer_google=search_cfg.get("partners", True), quota_guard=quota_guard
        )
        for result in results:
            url = result.get("href", "")
            title = result.get("title", "")
            body = result.get("body", "")
            if not url or any(domain in url for domain in ["youtube", "twitter", "facebook", "instagram"]):
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
        "plans_search", 7, first_url or hits[0]["url"], clean_text(combined_text, 6000), "FOUND",
        "search" if fetched_texts else "search_snippets",
    )
    source["plan_hits"] = hits[:10]
    return source


async def fetch_sector_eligibility_source(company: str, search_cfg: dict, quota_guard=None) -> dict:
    hits, fetched_texts, first_url = [], [], ""
    for query_template in SECTOR_SIGNAL_QUERIES + GROUP_FOUNDATION_QUERIES:
        results = await search_web(
            query_template.format(c=company), max_results=5,
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
            if len(fetched_texts) < 2:
                text = fetch_page_text(url) or body
                if text and len(text) > 200 and mentions_company(company, text):
                    fetched_texts.append(text)
                    first_url = first_url or url
        if len(fetched_texts) >= 2:
            break

    if not hits and not fetched_texts:
        return make_source("sector_eligibility_search", 8, status="NOT_FOUND")

    combined_text = " || ".join(fetched_texts) if fetched_texts else " || ".join(f"{hit['title']} — {hit['snippet']}" for hit in hits)
    return make_source(
        "sector_eligibility_search", 8, first_url or hits[0]["url"], clean_text(combined_text, 5000), "FOUND",
        "search" if fetched_texts else "search_snippets",
    )


async def fetch_screen_sources(company: str, search_cfg: dict, quota_guard=None) -> list[dict]:
    source_1 = await fetch_india_csr_page(company, search_cfg, quota_guard)
    source_4 = await fetch_annual_report(company, search_cfg, quota_guard)

    source_2 = make_source("mca_portal", 2, status="NOT_TRIED")
    source_3 = make_source("national_csr_portal", 3, status="NOT_TRIED")
    source_5 = make_source("partner_search", 5, status="NOT_TRIED")
    source_6 = make_source("people_search", 6, status="NOT_TRIED")
    source_7 = make_source("plans_search", 7, status="NOT_TRIED")
    source_8 = make_source("sector_eligibility_search", 8, status="NOT_TRIED")
    return [source_1, source_2, source_3, source_4, source_5, source_6, source_7, source_8]


async def fetch_deep_sources(company: str, search_cfg: dict, quota_guard=None, progress_cb=None) -> list[dict]:
    async def advance_step(message: str):
        if progress_cb:
            await progress_cb(message)

    await advance_step("Source 1/8 — India CSR page...")
    source_1 = await fetch_india_csr_page(company, search_cfg, quota_guard)

    await advance_step("Source 2/8 — MCA portal + CIN lookup...")
    source_2 = await fetch_mca_portal(company, search_cfg, quota_guard)

    await advance_step("Source 3/8 — National CSR Portal...")
    source_3 = await fetch_national_csr_portal(company, search_cfg, quota_guard)

    await advance_step("Source 4/8 — Annual report...")
    source_4 = await fetch_annual_report(company, search_cfg, quota_guard)

    await advance_step("Source 5/8 — Funded partners...")
    source_5 = await fetch_partner_source(company, search_cfg, quota_guard)

    await advance_step("Source 6/8 — CSR decision-makers (LinkedIn)...")
    source_6 = await fetch_linkedin_people(company, search_cfg, quota_guard)

    await advance_step("Source 7/8 — Partnerships, announced plans & open calls...")
    source_7 = await fetch_plans_source(company, search_cfg, quota_guard)

    await advance_step("Source 8/8 — Sector, eligibility & group-foundation signals...")
    source_8 = await fetch_sector_eligibility_source(company, search_cfg, quota_guard)

    return [source_1, source_2, source_3, source_4, source_5, source_6, source_7, source_8]

