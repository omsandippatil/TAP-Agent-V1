import asyncio
import gc
import logging
import re
import time
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from app.pipeline import google_search
from app.pipeline.people_parser import parse_linkedin_hit
from app.pipeline.search_budget import SearchBudget
from app.pipeline.source_registry import SourceRegistry
from app.pipeline.utils import (
    classify_fetch_error,
    clean_text,
    domain_resolves,
    extract_main_text,
    get_session,
    get_with_referer_fallback,
    make_source,
    normalize_block_text,
)

logger = logging.getLogger("tap.scraper")

GENERIC_COMPANY_TOKENS = {
    "india", "limited", "ltd", "private", "pvt", "the", "and", "of",
    "company", "corp", "corporation", "inc", "group", "technologies",
    "solutions", "services", "international", "holdings", "enterprises",
    "industries", "systems", "global",
}

ENGLISH_COMMON_WORD_TOKENS = {
    "nice", "best", "good", "great", "prime", "apex", "peak", "smart",
    "bright", "ace", "spark", "sharp", "clear", "true", "real", "fresh",
    "fast", "swift", "solid", "sure", "safe", "trust", "value", "key",
    "core", "base", "edge", "link", "unity", "one", "first", "next",
    "target", "focus", "vision", "insight", "impact", "spring", "summit",
    "crown", "royal", "grand", "elite", "select", "choice", "premier",
    "pure", "vital", "active", "bold", "swift", "rise", "grow", "thrive",
    "ask", "wish", "hope", "dream", "amaze", "delight", "please",
}

MIN_RELIABLE_TOKEN_LENGTH = 5


def is_generic_company_name(company: str) -> bool:
    tokens = company_name_tokens(company)
    if not tokens:
        return True
    return all(
        token in ENGLISH_COMMON_WORD_TOKENS or len(token) < MIN_RELIABLE_TOKEN_LENGTH
        for token in tokens
    )

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

_DYNAMIC_BLOCKED_DOMAINS: dict[str, int] = {}
_DYNAMIC_BLOCK_THRESHOLD = 3

GOOGLE_CSE_BROKEN_COOLDOWN_SECONDS = 900

_GOOGLE_CSE_BROKEN_UNTIL = 0.0
_GOOGLE_CSE_BROKEN_LOCK = asyncio.Lock()


def _mark_google_cse_broken(reason: str) -> None:
    global _GOOGLE_CSE_BROKEN_UNTIL
    already_broken = time.monotonic() < _GOOGLE_CSE_BROKEN_UNTIL
    _GOOGLE_CSE_BROKEN_UNTIL = time.monotonic() + GOOGLE_CSE_BROKEN_COOLDOWN_SECONDS
    if not already_broken:
        logger.error(
            "google CSE marked broken for %ds cooldown reason=%s",
            GOOGLE_CSE_BROKEN_COOLDOWN_SECONDS, reason,
        )


def google_cse_is_broken() -> bool:
    return time.monotonic() < _GOOGLE_CSE_BROKEN_UNTIL


def reset_google_cse_broken_flag_for_tests() -> None:
    global _GOOGLE_CSE_BROKEN_UNTIL
    _GOOGLE_CSE_BROKEN_UNTIL = 0.0


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
    "/social-responsibility", "/esg", "/about/csr", "/about-us/csr",
    "/company/csr", "/social-impact", "/community", "/impact",
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
    "learning", "curriculum", "classroom", "student", "literacy", "ai",
    "artificial intelligence", "robotics", "government school", "teacher",
]

PRIORITY_EDUCATION_KEYWORDS = [
    "stem", "artificial intelligence", " ai ", "coding", "digital skills",
    "digital literacy", "robotics", "government school", "government schools",
    "public school", "teacher training", "teacher capacity", "girls in ai",
    "girls in data", "atal tinkering", "21st century skills", "21st-century skills",
    "e-learning", "elearning", "science fair", "science fairs",
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

NON_INDIA_CSR_GEO_PATTERN = re.compile(
    r"\b(finland|finnish|estonia|tallinn|sweden|swedish|norway|norwegian|denmark|danish|"
    r"ukraine|ukrainian|erasmus\+?|european\s+union|\beu\b\s+programme|ukraine\s+government|"
    r"government\s+of\s+ukraine|germany|german|netherlands|dutch|belgium|belgian|"
    r"united\s+kingdom|\buk\b(?!\w)|canada|canadian|australia|australian|"
    r"united\s+states|\busa\b|\bus\b(?!\w))\b",
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

RELATED_ENTITY_NAME_PATTERN = re.compile(
    r"\b([A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*){0,4}\s+"
    r"(?:Foundation|Trust|CSR\s+Foundation|Charitable\s+Trust))\b"
)
RELATED_INDIA_BRANCH_PATTERN = re.compile(
    r"\b([A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*){0,4}\s+"
    r"(?:AG|SE|N\.?V\.?|PLC)?\s*(?:India|Mumbai|Delhi|Bengaluru|Bangalore)\s+"
    r"(?:Branch|Private\s+Limited|Pvt\.?\s+Ltd\.?|Limited|Ltd\.?))\b"
)

NAMED_INITIATIVE_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,4}\s+"
    r"(?:Development Impact Bond|Outcomes Fund|Programme|Program|Initiative|Project|Mission|Scholarship))\b"
)
NAMED_NGO_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}\s+"
    r"(?:Foundation|Trust|Education Foundation|Infotech Foundation))\b"
)

GENERIC_PHRASE_STOPWORDS = {
    "explore", "organizations", "organisation", "organisations", "team", "our",
    "the", "about", "home", "menu", "navigation", "search", "contact", "click",
    "read", "more", "learn", "view", "see", "all", "browse", "filter", "sort",
    "share", "follow", "subscribe", "sign", "login", "register", "back", "next",
    "previous", "related", "recommended", "featured", "trending", "latest",
    "weekly", "newsletter", "post", "posts", "article", "articles", "page",
    "pages", "section", "category", "categories", "tag", "tags", "download",
    "copyright", "privacy", "terms", "cookie", "policy", "sitemap",
    "agency", "donor", "implementing", "discover", "why", "what", "who",
    "resources", "insights", "stories", "news", "media", "press", "events",
    "careers", "join", "connect", "network", "community",
}


def _is_plausible_entity_name(name: str, min_words: int = 2, max_words: int = 6) -> bool:
    if not name or len(name) > 90:
        return False
    words = name.split()
    if not (min_words <= len(words) <= max_words):
        return False
    lowered_words = [w.lower().strip(".,&-") for w in words]
    if any(not w for w in lowered_words):
        return False
    stopword_hits = sum(1 for w in lowered_words if w in GENERIC_PHRASE_STOPWORDS)
    if stopword_hits >= 1 and len(words) <= 3:
        return False
    if stopword_hits >= 2 or stopword_hits / len(words) > 0.35:
        return False
    if len(set(lowered_words)) < len(lowered_words) * 0.7:
        return False
    return True


def is_plausible_entity_name(name: str, min_words: int = 2, max_words: int = 6) -> bool:
    return _is_plausible_entity_name(name, min_words=min_words, max_words=max_words)


def _iter_candidate_lines(text: str, max_line_chars: int = 300):
    if not text:
        return
    for line in text.split("\n"):
        line = line.strip()
        if line and len(line) <= max_line_chars:
            yield line


def _extract_entities_from_lines(text: str, patterns, filter_fn) -> list[str]:
    if not text:
        return []
    found = []
    seen = set()
    for line in _iter_candidate_lines(text):
        for pattern in patterns:
            for match in pattern.finditer(line):
                name = re.sub(r"\s+", " ", match.group(1)).strip()
                if not filter_fn(name):
                    continue
                key = name.lower()
                if key in seen:
                    continue
                seen.add(key)
                found.append(name)
    return found


CURRENCY_NEAR_INDIA_WINDOW_CHARS = 200

CURRENT_FY_LABEL = "FY2025-26"
PRIOR_FY_LABELS = ["FY2024-25", "FY2023-24", "2024-25", "2023-24", "FY2022-23"]

FY_YEAR_TOKEN_PATTERN = re.compile(r"FY\s?20?\d{2}[-–]\d{2,4}|20\d{2}[-–]\d{2,4}", re.IGNORECASE)

EDUCATION_PROGRAMME_QUERIES = [
    '"{c}" ("school education" OR "government school" OR "public school") CSR India named programme students {site}',
    '"{c}" (STEM OR AI OR "artificial intelligence" OR coding OR "digital skills" OR "digital literacy") CSR India students named programme',
    '"{c}" ("government school" OR "public school" OR teachers OR students) CSR India skilling named programme beneficiaries',
    '"{c}" (STEM OR robotics OR "digital literacy" OR coding) CSR India schools named programme annual report filetype:pdf',
    '"{c}" ("government school" OR STEM OR "digital skills") CSR India NGO partner beneficiaries press release',
]

CSR_PAGE_QUERIES = [
    '"{c}" (corporate social responsibility OR "CSR policy" OR "sustainability report" OR "ESG report") India {site}',
    '"{c}" CSR India filetype:pdf {site}',
]

ANNUAL_REPORT_QUERIES = [
    '"{c}" ("annual report" OR "business responsibility and sustainability report") {fy} India CSR filetype:pdf {site}',
    '"{c}" annual report CSR India crore filetype:pdf',
]

MULTI_YEAR_FINANCIAL_QUERIES = [
    '"{c}" ("net profit" OR "profit after tax" OR "CSR expenditure") {fy1} {fy2} {fy3} crore',
]

CSR_SPEND_QUERIES = [
    '"{c}" ("CSR expenditure" OR "CSR spend" OR "amount spent" OR "total CSR") crore India {fy}',
    '"{c}" CSR expenditure India annual report crore',
]

MCA_CIN_QUERIES = [
    '"{c}" (CIN OR "corporate identification number") India',
]

MCA_FILING_QUERIES = [
    '"{c}" ("Form CSR-2" OR "MCA annual filing") CSR India filetype:pdf',
]

NATIONAL_CSR_PORTAL_QUERIES = [
    'site:csr.gov.in "{c}"',
]

LEGAL_ENTITY_RESOLUTION_QUERIES = [
    '"{c}" India "Private Limited" CIN MCA',
]

RELATED_ENTITY_DISCOVERY_QUERIES = [
    '"{c}" (foundation OR "India branch" OR subsidiary) CSR corporate social responsibility',
]

PARTNER_QUERIES = [
    '"{c}" CSR (NGO partner OR "implementation partner" OR "implementing partner") India education',
    'site:linkedin.com/company "{c}" (partnered with OR MoU) NGO CSR India',
    '"{c}" CSR partner NGO announcement press release India',
]

PARTNER_FOLLOWUP_QUERIES = [
    '"{partner}" "{c}" (partnership OR funded OR implementing)',
]

PLAN_QUERIES = [
    '"{c}" CSR (partnership education OR "request for proposal" OR "call for proposals") India',
]

LINKEDIN_PEOPLE_QUERIES = [
    'site:linkedin.com/in "{c}" (head of CSR OR CSR head OR sustainability director OR ESG) India',
    'site:linkedin.com/in "{c}" corporate social responsibility',
    'site:linkedin.com/in "{c}" (foundation OR trustee OR "social impact") India',
]

LINKEDIN_PEOPLE_NAME_ONLY_FALLBACK_QUERIES = [
    '"{c}" (CSR OR foundation) (head OR manager OR lead OR director OR trustee) India -job -jobs -vacancy',
]

SECTOR_QUERIES = [
    '"{c}" India sector industry business overview annual report',
]

PROGRAMME_DEEP_DIVE_QUERY_TEMPLATE = (
    '"{programme}" "{c}" (geography OR beneficiaries OR students OR partner OR scale OR outcomes)'
)

UNREADABLE_DOC_RECOVERY_QUERIES = [
    '"{c}" CSR (programme name OR expenditure OR "amount spent") India crore news press release',
    '"{c}" CSR (education OR STEM OR skilling) programme name students press release',
    '"{c}" CSR partner NGO announcement India',
    '"{c}" CSR India (students OR schools OR beneficiaries) named programme site:linkedin.com/company',
]

FOLLOWUP_QUERY_TEMPLATES = {
    "education_programme": [
        '"{c}" (education OR skilling OR STEM) programme India named',
    ],
    "csr_budget": [
        '"{c}" ("CSR expenditure" OR "amount spent") crore India annual report',
    ],
    "decision_maker": [
        'site:linkedin.com/in "{c}" (CSR OR sustainability OR ESG) head India',
    ],
    "ngo_partner": [
        '"{c}" CSR ("implementation partner" OR "implementing partner") education named',
    ],
    "csr_policy": [
        '"{c}" ("CSR policy" OR "CSR annual report") India filetype:pdf',
    ],
}

MAX_PAGE_TEXT_CHARS = 6000
MAX_PDF_TEXT_CHARS = 10000
MAX_PDF_PAGES = 15
FINANCIAL_PDF_SCAN_PAGES = 25
CANDIDATE_EVAL_LIMIT = 3
MIN_ACCEPT_SCORE = 4
STRONG_ACCEPT_SCORE = 10

PAGE_FETCH_TIMEOUT_SECONDS = 8
PDF_FETCH_TIMEOUT_SECONDS = 10
HOMEPAGE_FETCH_TIMEOUT_SECONDS = 6
DNS_CHECK_TIMEOUT_SECONDS = 1.5
SEARCH_TASK_TIMEOUT_SECONDS = 6
FETCH_TASK_TIMEOUT_SECONDS = 10
SOURCE_DEADLINE_SECONDS = 18
FOLLOWUP_DEADLINE_SECONDS = 10
CONCURRENT_FETCH_LIMIT = 3

DEEP_JOB_HARD_DEADLINE_SECONDS = 150
MAX_PDF_DOWNLOAD_BYTES = 15 * 1024 * 1024
PDF_STREAM_CHUNK_BYTES = 262144
MAX_PDF_PAGES_HARD_CAP = 40
SECOND_PASS_TEXT_LENGTH_FLOOR = 500
UNREADABLE_TEXT_LENGTH_FLOOR = 200

MAX_PARTNER_SOURCES_DEEP = 10
MAX_PARTNER_SOURCES_SCREEN = 5
MAX_PROGRAMME_SOURCES_DEEP = 7
MAX_PROGRAMME_SOURCES_SCREEN = 4
MAX_PARTNER_FOLLOWUP_NAMES = 3
MAX_PROGRAMME_DEEP_DIVE_NAMES = 3

_FETCH_SEMAPHORE = asyncio.Semaphore(CONCURRENT_FETCH_LIMIT)

_ENTITY_PROXIMITY_WINDOW_CHARS = 60

_PARTNER_RELEVANCE_KEYWORD_PATTERN = re.compile(
    r"\b(partner|partnered|partnership|ngo|foundation|mou|memorandum|collaborat|"
    r"implement|grant|csr|development impact bond|dib|outcomes fund)\b", re.IGNORECASE,
)


class DeepJobDeadlineExceeded(Exception):
    pass


def count_distinct_year_tokens(text: str) -> int:
    if not text:
        return 0
    return len({m.group(0).upper().replace(" ", "") for m in FY_YEAR_TOKEN_PATTERN.finditer(text)})


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


def count_priority_education_hits(text: str) -> int:
    if not text:
        return 0
    lowered = f" {text.lower()} "
    return sum(1 for kw in PRIORITY_EDUCATION_KEYWORDS if kw in lowered)


def has_india_or_education_signal(text: str) -> bool:
    if not text:
        return False
    if has_india_location_signal(text):
        return True
    lowered = text.lower()
    return any(kw in lowered for kw in EDUCATION_KEYWORDS)


def is_non_india_geo_dominant(text: str) -> bool:
    if not text:
        return False
    non_india_hits = len(NON_INDIA_CSR_GEO_PATTERN.findall(text))
    if non_india_hits == 0:
        return False
    india_hits = len(find_india_location_mentions(text))
    return non_india_hits > india_hits


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


_BUSINESS_CONTEXT_WORD_PATTERN = re.compile(
    r"\b(csr|corporate|company|ltd|limited|inc|pvt|private|india|subsidiary|"
    r"headquarter|founded|revenue|employee|annual report|sustainability|"
    r"foundation|ceo|software|technology|platform|solutions|customer)\b",
    re.IGNORECASE,
)

_GENERIC_NAME_CONTEXT_WINDOW_CHARS = 250


def _mentions_generic_company_name(company: str, text: str) -> bool:
    exact_case_positions = [m.start() for m in re.finditer(re.escape(company), text)]
    if not exact_case_positions:
        return False
    context_positions = [m.start() for m in _BUSINESS_CONTEXT_WORD_PATTERN.finditer(text)]
    if not context_positions:
        return False
    return any(
        abs(name_pos - ctx_pos) <= _GENERIC_NAME_CONTEXT_WINDOW_CHARS
        for name_pos in exact_case_positions
        for ctx_pos in context_positions
    )


def mentions_company(company: str, text: str) -> bool:
    if not text:
        return False
    if is_generic_company_name(company):
        return _mentions_generic_company_name(company, text)
    lowered = text.lower()
    tokens = company_name_tokens(company)
    if not tokens:
        return company.lower() in lowered
    return any(token in lowered for token in tokens)


def mentions_company_specifically(company: str, text: str) -> bool:
    if not text:
        return False
    if is_generic_company_name(company):
        return _mentions_generic_company_name(company, text)
    tokens = company_name_tokens(company)
    if len(tokens) < 2:
        return mentions_company(company, text)

    lowered = text.lower()
    positions = sorted(
        match.start()
        for token in tokens
        for match in re.finditer(re.escape(token), lowered)
    )
    if len(positions) < 2:
        return mentions_company(company, text)
    return any(b - a < _ENTITY_PROXIMITY_WINDOW_CHARS for a, b in zip(positions, positions[1:]))


def _looks_like_same_entity(company: str, candidate: str) -> bool:
    company_norm = re.sub(r"[^a-z0-9]", "", company.lower())
    candidate_norm = re.sub(r"[^a-z0-9]", "", candidate.lower())
    return company_norm == candidate_norm


def _domain_site_token(domains: list[str] | None) -> str:
    return f"site:{domains[0]}" if domains else ""


def _extract_related_entity_candidates(company: str, text: str) -> list[dict]:
    if not text:
        return []
    tokens = company_name_tokens(company)
    candidates: list[dict] = []
    seen = set()

    for pattern, entity_type in (
        (RELATED_ENTITY_NAME_PATTERN, "FOUNDATION"),
        (RELATED_INDIA_BRANCH_PATTERN, "INDIA_SUBSIDIARY"),
    ):
        for line in _iter_candidate_lines(text):
            for match in pattern.finditer(line):
                name = re.sub(r"\s+", " ", match.group(1)).strip()
                if not _is_plausible_entity_name(name):
                    continue
                if _looks_like_same_entity(company, name):
                    continue
                name_lower = name.lower()
                if tokens and not any(token in name_lower for token in tokens):
                    continue
                key = name_lower
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({"entity_name": name, "entity_type": entity_type})
    return candidates


def _extract_named_partner_candidates(company: str, text: str) -> list[str]:
    if not text:
        return []
    names = _extract_entities_from_lines(text, [NAMED_NGO_PATTERN], _is_plausible_entity_name)
    return [name for name in names if not _looks_like_same_entity(company, name)]


def _extract_named_programme_candidates(text: str) -> list[str]:
    return _extract_entities_from_lines(text, [NAMED_INITIATIVE_PATTERN], _is_plausible_entity_name)


async def discover_related_entities(company: str, search_cfg: dict, budget: SearchBudget,
                                     quota_guard=None, deadline: float | None = None) -> list[dict]:
    if getattr(budget, "related_entities_resolved", False):
        return budget.related_entities_cache or []

    discovered: dict[str, dict] = {}
    for query_template in RELATED_ENTITY_DISCOVERY_QUERIES:
        if deadline is not None and time.monotonic() >= deadline:
            break
        if not budget.google_has_budget("entity_resolution"):
            break
        query = query_template.format(c=company)
        results = await search_web(
            query, budget, max_results=6, quota_guard=quota_guard, category="entity_resolution",
        )
        for result in results:
            haystack = f"{result.get('title', '')}\n{result.get('body', '')}"
            for candidate in _extract_related_entity_candidates(company, haystack):
                key = candidate["entity_name"].lower()
                if key not in discovered:
                    discovered[key] = candidate

    resolved = list(discovered.values())
    if resolved:
        budget.mark_category_hit("entity_resolution", len(resolved))
    budget.related_entities_resolved = True
    budget.related_entities_cache = resolved
    logger.info(
        "discover_related_entities DONE company=%r found=%d names=%s",
        company, len(resolved), [e["entity_name"] for e in resolved],
    )
    return resolved


def related_entity_names(related_entities: list[dict]) -> list[str]:
    return [e.get("entity_name", "") for e in (related_entities or []) if e.get("entity_name")]


def candidate_domains(company: str) -> list[str]:
    tokens = company_name_tokens(company) or [re.sub(r"[^a-z0-9]", "", company.lower())]
    slugs = list(dict.fromkeys(["".join(tokens), tokens[0]]))
    ordered = []
    for tld in (".com", ".in", ".co.in", ".org"):
        for slug in slugs:
            if slug:
                ordered.append(f"www.{slug}{tld}")

    verified = [d for d in ordered if domain_resolves(d, timeout=DNS_CHECK_TIMEOUT_SECONDS)]
    if not verified:
        logger.info("candidate_domains: none of %d guessed domains resolved for company=%r", len(ordered), company)
    return verified


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
    if is_generic_company_name(company):
        return False
    tokens = company_name_tokens(company)
    if not tokens:
        return False
    host_base = host.replace("www.", "")
    return any(token in host_base for token in tokens)


def is_known_blocked_domain(url: str) -> bool:
    if not url:
        return False
    host = urlparse(url).netloc.lower()
    if any(domain in host for domain in BLOCKED_403_DOMAINS):
        return True
    return _DYNAMIC_BLOCKED_DOMAINS.get(host, 0) >= _DYNAMIC_BLOCK_THRESHOLD


def _record_blocked_response(url: str, status_code: int | None) -> None:
    if status_code != 403:
        return
    host = urlparse(url).netloc.lower()
    if not host:
        return
    count = _DYNAMIC_BLOCKED_DOMAINS.get(host, 0) + 1
    _DYNAMIC_BLOCKED_DOMAINS[host] = count
    if count == _DYNAMIC_BLOCK_THRESHOLD:
        logger.warning(
            "domain dynamically denylisted after %d consecutive 403s domain=%s", count, host,
        )


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
    education_priority_bonus = min(count_priority_education_hits(text) * 2.5, 10.0)
    length_bonus = min(len(text) / 2000.0, 4.0)
    domain_bonus = 6.0 if url and any(gov in url.lower() for gov in OFFICIAL_GOV_DOMAINS) else 0.0
    pdf_bonus = 1.5 if url.lower().endswith(".pdf") else 0.0
    return (
        csr_hits * 2.0 + figure_hits * 5.0 + india_figure_bonus + india_location_bonus
        + education_priority_bonus + length_bonus + domain_bonus + pdf_bonus
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


async def search_web(query: str, budget: SearchBudget, max_results: int = 6,
                      quota_guard=None, category: str = "") -> list[dict]:
    if google_cse_is_broken():
        logger.info("google search skipped, CSE marked broken query=%r category=%r", query, category)
        return []
    if not google_search.google_search_configured_and_available(quota_guard):
        return []
    if not budget.google_has_budget(category):
        logger.info("google search skipped, budget exhausted or category satisfied query=%r category=%r", query, category)
        return []

    budget.record_google_query(category)
    try:
        results = await asyncio.wait_for(
            google_search.google_search_web(query, max_results=max_results, quota_guard=quota_guard),
            timeout=SEARCH_TASK_TIMEOUT_SECONDS,
        )
    except google_search.GoogleCseInvalidArgumentError as exc:
        async with _GOOGLE_CSE_BROKEN_LOCK:
            _mark_google_cse_broken(str(exc))
        budget.record_query_results(category, 0)
        return []
    except asyncio.TimeoutError:
        logger.warning("google search timed out query=%r", query)
        budget.record_query_results(category, 0)
        return []

    budget.record_query_results(category, len(results))
    if not results:
        logger.info("google search returned empty query=%r category=%r", query, category)
    return results


def _fetch_page_text_sync(url: str, max_chars: int, verify_ssl: bool) -> tuple[str, str]:
    if is_known_blocked_domain(url):
        return "", "known_blocked_domain"
    try:
        response = get_with_referer_fallback(url, timeout=PAGE_FETCH_TIMEOUT_SECONDS, verify=verify_ssl)
        _record_blocked_response(url, response.status_code)
        if response.status_code == 403:
            return "", "http_4xx_403"
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_PDF_DOWNLOAD_BYTES:
            return "", "oversized_response"
        soup = BeautifulSoup(response.text, "lxml")
        text = extract_main_text(soup, max_chars)
        del soup
        return text, ""
    except Exception as exc:
        error_type = classify_fetch_error(exc)
        logger.info("fetch_page_text failed url=%s error_type=%s", url, error_type)
        return "", error_type
    finally:
        gc.collect()


async def fetch_page_text(url: str, max_chars: int = MAX_PAGE_TEXT_CHARS, verify_ssl: bool = True) -> str:
    async with _FETCH_SEMAPHORE:
        try:
            text, _error_type = await asyncio.wait_for(
                asyncio.to_thread(_fetch_page_text_sync, url, max_chars, verify_ssl),
                timeout=FETCH_TASK_TIMEOUT_SECONDS,
            )
            return text
        except asyncio.TimeoutError:
            logger.info("fetch_page_text timed out url=%s", url)
            return ""


def _select_financial_pdf_pages(pdf, max_pages: int, max_scan: int) -> list[int]:
    total_pages = len(pdf.pages)
    scan_upper = min(total_pages, max_scan)
    if total_pages <= max_pages:
        return list(range(total_pages))

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
        return list(range(min(max_pages, total_pages)))

    scored_indices.sort(key=lambda pair: pair[0], reverse=True)
    return sorted(idx for _, idx in scored_indices[:max_pages])


def _download_pdf_bytes(url: str) -> tuple[bytes, str]:
    response = get_with_referer_fallback(url, timeout=PDF_FETCH_TIMEOUT_SECONDS, stream=True)
    _record_blocked_response(url, response.status_code)
    if response.status_code == 403:
        response.close()
        return b"", "http_4xx_403"
    response.raise_for_status()

    content_length = response.headers.get("Content-Length")
    if content_length and int(content_length) > MAX_PDF_DOWNLOAD_BYTES:
        response.close()
        return b"", "oversized_pdf"

    chunks = []
    total = 0
    for chunk in response.iter_content(chunk_size=PDF_STREAM_CHUNK_BYTES):
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_PDF_DOWNLOAD_BYTES:
            response.close()
            return b"", "oversized_pdf"
        chunks.append(chunk)
    response.close()
    return b"".join(chunks), ""


def _ocr_pdf_pages(pdf_bytes: bytes, max_pages: int) -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
    except Exception:
        return ""
    try:
        images = convert_from_bytes(pdf_bytes, first_page=1, last_page=max(1, max_pages))
    except Exception:
        return ""
    texts = []
    for image in images:
        try:
            texts.append(pytesseract.image_to_string(image))
        except Exception:
            continue
        finally:
            image.close()
    return normalize_block_text(" ".join(texts), MAX_PDF_TEXT_CHARS)


def _fetch_pdf_text_sync(url: str, max_chars: int, max_pages: int) -> tuple[str, str]:
    if is_known_blocked_domain(url):
        return "", "known_blocked_domain"
    try:
        import io
        import pdfplumber

        pdf_bytes, error = _download_pdf_bytes(url)
        if error:
            return "", error
        if not pdf_bytes:
            return "", "empty_response"

        capped_pages = min(max_pages, MAX_PDF_PAGES_HARD_CAP)
        pages_text = []
        total_len = 0
        buffer = io.BytesIO(pdf_bytes)
        try:
            with pdfplumber.open(buffer) as pdf:
                selected_indices = _select_financial_pdf_pages(pdf, capped_pages, FINANCIAL_PDF_SCAN_PAGES)
                for idx in selected_indices:
                    page = pdf.pages[idx]
                    page_text = page.extract_text() or ""
                    page.flush_cache()
                    if page_text:
                        pages_text.append(page_text)
                        total_len += len(page_text)
                    if total_len >= max_chars:
                        break
        finally:
            buffer.close()

        combined_text = normalize_block_text("\n".join(pages_text), max_chars)
        if len(combined_text) < SECOND_PASS_TEXT_LENGTH_FLOOR:
            ocr_text = _ocr_pdf_pages(pdf_bytes, min(capped_pages, 8))
            if len(ocr_text) > len(combined_text):
                return ocr_text, ""
        return combined_text, ""
    except Exception as exc:
        error_type = classify_fetch_error(exc)
        logger.info("fetch_pdf_text failed url=%s error_type=%s", url, error_type)
        return "", error_type
    finally:
        gc.collect()


async def fetch_pdf_text(url: str, max_chars: int = MAX_PDF_TEXT_CHARS, max_pages: int = MAX_PDF_PAGES) -> str:
    async with _FETCH_SEMAPHORE:
        try:
            text, _error_type = await asyncio.wait_for(
                asyncio.to_thread(_fetch_pdf_text_sync, url, max_chars, max_pages),
                timeout=FETCH_TASK_TIMEOUT_SECONDS,
            )
            return text
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
        del soup
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
            except Exception:
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
            return []


async def discover_company_domain(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None) -> str:
    domains = await discover_company_domains(company, search_cfg, budget, quota_guard)
    return domains[0] if domains else ""


async def _direct_dotcom_probe(company: str) -> str:
    tokens = company_name_tokens(company)
    if not tokens:
        return ""
    slug = "".join(tokens)
    if not slug:
        return ""
    domain = f"www.{slug}.com"
    if not domain_resolves(domain, timeout=DNS_CHECK_TIMEOUT_SECONDS):
        return ""

    def _probe() -> bool:
        try:
            response = get_session().get(f"https://{domain}", timeout=HOMEPAGE_FETCH_TIMEOUT_SECONDS)
            return bool(response.ok)
        except Exception:
            return False

    try:
        confirmed = await asyncio.wait_for(asyncio.to_thread(_probe), timeout=FETCH_TASK_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return ""
    return domain if confirmed else ""


async def discover_company_domains(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None) -> list[str]:
    if budget.resolved_domains:
        return budget.resolved_domains

    tokens = company_name_tokens(company)
    acronym = "".join(token[0] for token in tokens) if len(tokens) >= 2 else ""

    matched_domains: list[str] = []

    direct_hit = await _direct_dotcom_probe(company)
    if direct_hit:
        matched_domains.append(direct_hit)

    results = await search_web(
        f'"{company}" official website India', budget, max_results=6,
        quota_guard=quota_guard, category="csr_page",
    )
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

    resolved = matched_domains[:4]
    budget.set_resolved_domains(resolved)
    return budget.resolved_domains


async def resolve_india_legal_entity_name(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None) -> str:
    if budget.legal_entity_name_resolved:
        return budget.legal_entity_name_cache or ""

    resolved_name = ""
    for query_template in LEGAL_ENTITY_RESOLUTION_QUERIES:
        if not budget.google_has_budget("legal_entity"):
            break
        query = query_template.format(c=company)
        results = await search_web(query, budget, max_results=6, quota_guard=quota_guard, category="legal_entity")
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

    if resolved_name:
        budget.mark_category_hit("legal_entity")
    budget.legal_entity_name_resolved = True
    budget.legal_entity_name_cache = resolved_name
    logger.info("resolve_india_legal_entity_name DONE company=%r resolved=%r", company, resolved_name or None)
    return resolved_name


async def _within_deadline(deadline: float) -> bool:
    return time.monotonic() < deadline


async def _check_job_deadline(job_deadline: float | None) -> None:
    if job_deadline is not None and time.monotonic() >= job_deadline:
        raise DeepJobDeadlineExceeded()


def _format_query_template(query_template: str, company: str) -> str:
    try:
        return query_template.format(c=company, fy=CURRENT_FY_LABEL)
    except KeyError as exc:
        logger.warning(
            "query template has unsupported placeholder %s, skipping template=%r", exc, query_template,
        )
        return ""


async def _recover_via_secondary_search(company: str, budget: SearchBudget, quota_guard, deadline: float,
                                         category: str, query_templates: list[str],
                                         min_len: int = 200) -> tuple[str, str] | None:
    for query_template in query_templates:
        if not await _within_deadline(deadline):
            break
        query = _format_query_template(query_template, company)
        if not query:
            continue
        results = await search_web(query, budget, max_results=6, quota_guard=quota_guard, category=category)
        for result in results:
            if not await _within_deadline(deadline):
                break
            url = result.get("href", "")
            title = result.get("title", "")
            body = result.get("body", "")
            if not url or any(domain in url for domain in AGGREGATOR_DOMAINS):
                continue
            if not mentions_company(company, f"{title} {body}"):
                continue
            is_pdf = url.lower().endswith(".pdf")
            text = await (fetch_pdf_text(url) if is_pdf else fetch_page_text(url)) or body
            if text and len(text) >= min_len and mentions_company(company, text) and is_csr_relevant(text):
                budget.mark_category_hit(category)
                return url, text
    return None


async def _recover_from_unreadable_document(company: str, budget: SearchBudget, quota_guard, deadline: float,
                                             category: str, seed_names: list[str] | None = None,
                                             min_len: int = UNREADABLE_TEXT_LENGTH_FLOOR) -> tuple[str, str] | None:
    """A document belonging to the company was found (a real URL, confirmed by
    title/snippet) but its text could not be extracted or extraction returned
    too little to use. Instead of giving up, chase the topic the document was
    about: named entities pulled from whatever snippet text is available,
    then a broadening set of generic recovery queries. Returns the first
    (url, text) pair that clears the relevance bar, or None.
    """
    for name in (seed_names or [])[:MAX_PARTNER_FOLLOWUP_NAMES]:
        if not await _within_deadline(deadline):
            return None
        query = f'"{name}" "{company}" (partnership OR funded OR implementing OR programme)'
        results = await search_web(query, budget, max_results=5, quota_guard=quota_guard, category=category)
        for result in results:
            url = result.get("href", "")
            title = result.get("title", "")
            body = result.get("body", "")
            if not url or any(domain in url for domain in AGGREGATOR_DOMAINS):
                continue
            if not mentions_company(company, f"{title} {body}"):
                continue
            is_pdf = url.lower().endswith(".pdf")
            text = await (fetch_pdf_text(url) if is_pdf else fetch_page_text(url)) or body
            if text and len(text) >= min_len and mentions_company(company, text) and is_csr_relevant(text):
                budget.mark_category_hit(category)
                return url, text

    return await _recover_via_secondary_search(
        company, budget, quota_guard, deadline, category, UNREADABLE_DOC_RECOVERY_QUERIES, min_len=min_len,
    )


async def fetch_india_csr_page(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None,
                                max_fetches: int = 20, registry: SourceRegistry | None = None,
                                job_deadline: float | None = None) -> dict:
    await _check_job_deadline(job_deadline)
    deadline = time.monotonic() + SOURCE_DEADLINE_SECONDS
    if job_deadline is not None:
        deadline = min(deadline, job_deadline)
    tried_urls = set()
    remaining_budget = [max_fetches]
    resolved_domain = [""]
    best_candidate = [None]
    weak_snippet_fallback = [None]
    document_found_unreadable = [False]

    def consider(url: str, method: str, text: str):
        if accept_fetched_text(company, text, 250):
            score = score_candidate_text(company, text, url)
            if best_candidate[0] is None or score > best_candidate[0][0]:
                source = make_source("india_csr_page", 1, url, text, "FOUND", method)
                source["domain"] = urlparse(url).netloc.lower()
                source["india_location_hits"] = find_india_location_mentions(text)[:10]
                best_candidate[0] = (score, source)
            return
        if text and len(text) > 80 and mentions_company(company, text) and mentions_csr_context(text):
            score = score_candidate_text(company, text, url)
            if weak_snippet_fallback[0] is None or score > weak_snippet_fallback[0][0]:
                source = make_source("india_csr_page", 1, url, text, "FOUND", method + "_snippet")
                source["domain"] = urlparse(url).netloc.lower()
                weak_snippet_fallback[0] = (score, source)

    domain_miss_streak: dict[str, int] = {}
    DOMAIN_MISS_ESCALATION_THRESHOLD = 4

    async def try_fetch(url: str, method: str, is_pdf: bool = False):
        if not url or url in tried_urls or remaining_budget[0] <= 0 or not await _within_deadline(deadline):
            return
        host = urlparse(url).netloc.lower()
        if budget.is_domain_dead(host) or budget.is_path_dead(url):
            return
        tried_urls.add(url)
        remaining_budget[0] -= 1
        text = await (fetch_pdf_text(url) if is_pdf else fetch_page_text(url))
        if not text or len(text) < UNREADABLE_TEXT_LENGTH_FLOOR:
            if url.lower().endswith(".pdf") or is_pdf:
                document_found_unreadable[0] = True
            budget.mark_path_dead(url)
            domain_miss_streak[host] = domain_miss_streak.get(host, 0) + 1
            if domain_miss_streak[host] >= DOMAIN_MISS_ESCALATION_THRESHOLD:
                budget.mark_domain_dead(host, "repeated_path_misses")
        else:
            domain_miss_streak[host] = 0
        consider(url, method, text)

    discovered_domains = await discover_company_domains(company, search_cfg, budget, quota_guard)
    domains = [d for d in dict.fromkeys(discovered_domains + candidate_domains(company)) if not budget.is_domain_dead(d)]

    async def check_homepage(domain: str):
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(get_session().get, f"https://{domain}", timeout=HOMEPAGE_FETCH_TIMEOUT_SECONDS),
                timeout=FETCH_TASK_TIMEOUT_SECONDS,
            )
            if response.ok:
                return domain, response.text
            if response.status_code == 404:
                budget.mark_domain_dead(domain, "homepage_404")
        except Exception as exc:
            error_type = classify_fetch_error(exc)
            logger.info("homepage fetch failed domain=%s error_type=%s", domain, error_type)
            if error_type in ("ssl", "dns"):
                budget.mark_domain_dead(domain, error_type)
        return None

    live_homepages = []
    for domain in domains[:6]:
        await _check_job_deadline(job_deadline)
        if budget.is_domain_dead(domain):
            continue
        result = await check_homepage(domain)
        if result:
            live_homepages.append(result)
        if len(live_homepages) >= 3:
            break

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
        for path in CSR_PAGE_PATHS:
            if best_candidate[0] and best_candidate[0][0] >= MIN_ACCEPT_SCORE:
                break
            if candidates_checked >= CANDIDATE_EVAL_LIMIT and best_candidate[0]:
                break
            if budget.is_domain_dead(domain):
                break
            await try_fetch(f"https://{domain}{path}", "direct")
            candidates_checked += 1
        if best_candidate[0] and best_candidate[0][0] >= MIN_ACCEPT_SCORE:
            break
        if budget.is_domain_dead(domain):
            continue
        for sitemap_url in await sitemap_csr_urls(domain):
            if best_candidate[0] and best_candidate[0][0] >= MIN_ACCEPT_SCORE:
                break
            await try_fetch(sitemap_url, "sitemap", is_pdf=sitemap_url.lower().endswith(".pdf"))
            candidates_checked += 1
        if best_candidate[0] and best_candidate[0][0] >= MIN_ACCEPT_SCORE:
            break

    remaining_budget[0] = max(remaining_budget[0], 8)
    if (not best_candidate[0] or best_candidate[0][0] < MIN_ACCEPT_SCORE) and await _within_deadline(deadline):
        site_token = _domain_site_token(discovered_domains)
        for query_template in CSR_PAGE_QUERIES:
            if not await _within_deadline(deadline):
                break
            if best_candidate[0] and best_candidate[0][0] >= MIN_ACCEPT_SCORE:
                break
            query = query_template.format(c=company, site=site_token).strip()
            results = await search_web(
                query, budget, max_results=6, quota_guard=quota_guard, category="csr_page",
            )
            for result in results:
                url = result.get("href", "")
                title = result.get("title", "")
                snippet_body = result.get("body", "")
                if not url or any(domain in url for domain in AGGREGATOR_DOMAINS):
                    continue
                if not mentions_company(company, f"{title} {snippet_body}"):
                    continue
                consider(url, "snippet", snippet_body)
                if not url_belongs_to_company(company, url, list(dict.fromkeys(discovered_domains))):
                    continue
                await try_fetch(url, "search", is_pdf=url.lower().endswith(".pdf"))
                if best_candidate[0] and best_candidate[0][0] >= MIN_ACCEPT_SCORE:
                    break

    chosen = best_candidate[0] or weak_snippet_fallback[0]

    if not chosen and document_found_unreadable[0] and await _within_deadline(deadline):
        recovered = await _recover_from_unreadable_document(
            company, budget, quota_guard, deadline, "csr_page",
        )
        if recovered:
            url, text = recovered
            score = score_candidate_text(company, text, url)
            source = make_source("india_csr_page", 1, url, text, "FOUND", "unreadable_recovery")
            source["domain"] = urlparse(url).netloc.lower()
            chosen = (score, source)
            logger.info("india_csr_page recovered via unreadable-document fallback company=%r url=%s", company, url)

    if chosen:
        result_source = chosen[1]
        if registry is not None:
            registry.register_core_source(result_source)
        budget.mark_category_hit("csr_page")
        logger.info("india_csr_page DONE company=%r found=True score=%.1f", company, chosen[0])
        return result_source

    logger.info("india_csr_page DONE company=%r found=False document_found_unreadable=%s", company, document_found_unreadable[0])
    fallback = make_source("india_csr_page", 1, status="NOT_FOUND")
    fallback["domain"] = resolved_domain[0]
    return fallback


async def find_company_cin(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None,
                            deadline: float | None = None) -> str:
    for query_template in MCA_CIN_QUERIES:
        if deadline is not None and not await _within_deadline(deadline):
            break
        results = await search_web(
            query_template.format(c=company), budget, max_results=5,
            quota_guard=quota_guard, category="cin",
        )
        for result in results:
            body = result.get("body", "") + " " + result.get("title", "") + " " + result.get("href", "")
            match = CIN_PATTERN.search(body)
            if match:
                budget.mark_category_hit("cin")
                logger.info("find_company_cin DONE company=%r cin=%s", company, match.group(0).upper())
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
                            registry: SourceRegistry | None = None, job_deadline: float | None = None) -> dict:
    await _check_job_deadline(job_deadline)
    deadline = time.monotonic() + SOURCE_DEADLINE_SECONDS
    if job_deadline is not None:
        deadline = min(deadline, job_deadline)
    legal_name = await resolve_india_legal_entity_name(company, search_cfg, budget, quota_guard)
    cin = await find_company_cin(company, search_cfg, budget, quota_guard, deadline=deadline)

    if cin:
        mca_text = await fetch_mca_company_data_gov_page(cin)
        if mca_text and mentions_company(company, mca_text):
            source = make_source(
                "mca_portal", 2, f"https://www.mca.gov.in/mcafoportal/viewCompanyMasterData.do?cid={cin}",
                mca_text, "FOUND", "direct",
            )
            source["cin"] = cin
            if legal_name:
                source["legal_entity_name"] = legal_name
            if registry is not None:
                registry.register_core_source(source)
            return source
        synthetic_text = (
            f"{company} is registered in India with Corporate Identification Number (CIN) {cin}."
            + (f" Registered legal entity name: {legal_name}." if legal_name else "")
        )
        source = make_source(
            "mca_portal", 2, f"https://www.mca.gov.in/mcafoportal/viewCompanyMasterData.do?cid={cin}",
            synthetic_text, "FOUND", "cin_confirmed_portal_blocked",
        )
        source["cin"] = cin
        if legal_name:
            source["legal_entity_name"] = legal_name
        if registry is not None:
            registry.register_core_source(source)
        logger.info("mca_portal DONE company=%r found=True (cin_only) cin=%s", company, cin)
        return source

    best_candidate = None
    for query_template in MCA_FILING_QUERIES:
        if not await _within_deadline(deadline):
            break
        if best_candidate and best_candidate[0] >= MIN_ACCEPT_SCORE:
            break
        results = await search_web(
            query_template.format(c=company), budget, max_results=6,
            quota_guard=quota_guard, category="mca_filing",
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
        if registry is not None:
            registry.register_core_source(best_candidate[1])
        budget.mark_category_hit("mca_filing")
        logger.info("mca_portal DONE company=%r found=True cin=%s", company, best_candidate[1].get("cin", ""))
        return best_candidate[1]

    logger.info("mca_portal DONE company=%r found=False", company)
    return make_source("mca_portal", 2, status="NOT_FOUND")


async def fetch_national_csr_portal(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None,
                                     registry: SourceRegistry | None = None, job_deadline: float | None = None) -> dict:
    await _check_job_deadline(job_deadline)
    deadline = time.monotonic() + SOURCE_DEADLINE_SECONDS
    if job_deadline is not None:
        deadline = min(deadline, job_deadline)
    company_query = company.replace(" ", "+")
    direct_url = f"https://www.csr.gov.in/content/csr/global/master/home/companydetail.html?companyName={company_query}"
    text = await fetch_page_text(direct_url)
    if text and len(text) > 250 and mentions_company(company, text):
        source = make_source("national_csr_portal", 3, direct_url, text, "FOUND", "direct")
        if registry is not None:
            registry.register_core_source(source)
        budget.mark_category_hit("national_csr_portal")
        return source

    best_candidate = None
    for query_template in NATIONAL_CSR_PORTAL_QUERIES:
        if not await _within_deadline(deadline):
            break
        results = await search_web(
            query_template.format(c=company), budget, max_results=6,
            quota_guard=quota_guard, category="national_csr_portal",
        )
        for result in results:
            url = result.get("href", "")
            body = result.get("body", "")
            if not url:
                continue
            page_text = await fetch_page_text(url) or body
            if page_text and mentions_company(company, page_text) and ("csr.gov.in" in url.lower() or is_csr_relevant(page_text)):
                score = score_candidate_text(company, page_text, url)
                if best_candidate is None or score > best_candidate[0]:
                    best_candidate = (score, make_source("national_csr_portal", 3, url, page_text, "FOUND", "search"))
        if best_candidate and best_candidate[0] >= MIN_ACCEPT_SCORE:
            break

    if best_candidate:
        if registry is not None:
            registry.register_core_source(best_candidate[1])
        budget.mark_category_hit("national_csr_portal")
        return best_candidate[1]

    return make_source("national_csr_portal", 3, status="NOT_FOUND")


async def fetch_annual_report(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None,
                               registry: SourceRegistry | None = None, job_deadline: float | None = None) -> dict:
    await _check_job_deadline(job_deadline)
    deadline = time.monotonic() + SOURCE_DEADLINE_SECONDS
    if job_deadline is not None:
        deadline = min(deadline, job_deadline)
    best_candidate = None
    weak_candidate = None
    urls_tried = 0
    pdf_found_but_unreadable = False
    site_token = _domain_site_token(budget.resolved_domains)

    for template in ANNUAL_REPORT_QUERIES:
        if not await _within_deadline(deadline):
            break
        if best_candidate and best_candidate[0] >= STRONG_ACCEPT_SCORE:
            break
        query = template.format(c=company, fy=CURRENT_FY_LABEL, site=site_token).strip()
        results = await search_web(
            query, budget, max_results=8, quota_guard=quota_guard, category="annual_report",
        )
        for result in results:
            if not await _within_deadline(deadline):
                break
            url = result.get("href", "")
            title = result.get("title", "")
            body = result.get("body", "")
            if not url or not mentions_company(company, f"{title} {body}") or not url_belongs_to_company(company, url):
                continue
            urls_tried += 1
            fetch_failed = False
            if url.lower().endswith(".pdf"):
                text = await fetch_pdf_text(url)
                if not text or len(text) < UNREADABLE_TEXT_LENGTH_FLOOR:
                    fetch_failed = True
                    pdf_found_but_unreadable = True
                    text = body if body and len(body) > 100 and mentions_company(company, body) else ""
                if text and not pdf_is_csr_relevant(text) and not fetch_failed:
                    continue
                is_india_specific = has_india_specific_financial_figure(text) if text else False
                if text and count_financial_figures(text) > 0 and not is_india_specific and not fetch_failed:
                    score = score_candidate_text(company, text, url)
                    if weak_candidate is None or score > weak_candidate[0]:
                        weak_candidate = (score, make_source("annual_report", 4, url, text, "FOUND", "pdf_weak_locale"))
                    continue
            else:
                text = await fetch_page_text(url) or body
            if not (text and len(text) > 200 and mentions_company(company, text)):
                continue
            if not is_csr_relevant(text) and not has_financial_figures(text):
                continue
            score = score_candidate_text(company, text, url)
            if best_candidate is None or score > best_candidate[0]:
                fetch_method = "pdf" if url.lower().endswith(".pdf") else "search"
                best_candidate = (score, make_source("annual_report", 4, url, text, "FOUND", fetch_method))
            if best_candidate and best_candidate[0] >= STRONG_ACCEPT_SCORE:
                break

    if (not best_candidate or count_financial_figures(best_candidate[1].get("text", "")) == 0) and await _within_deadline(deadline):
        for fy in PRIOR_FY_LABELS[:2]:
            if not await _within_deadline(deadline):
                break
            if best_candidate and best_candidate[0] >= MIN_ACCEPT_SCORE:
                break
            query = f'"{company}" "annual report" {fy} CSR filetype:pdf'
            results = await search_web(query, budget, max_results=6, quota_guard=quota_guard, category="annual_report")
            for result in results:
                if not await _within_deadline(deadline):
                    break
                url = result.get("href", "")
                title = result.get("title", "")
                body = result.get("body", "")
                if not url or not url.lower().endswith(".pdf"):
                    continue
                if not mentions_company(company, f"{title} {body}") or not url_belongs_to_company(company, url):
                    continue
                text = await fetch_pdf_text(url)
                if not text or len(text) < UNREADABLE_TEXT_LENGTH_FLOOR:
                    pdf_found_but_unreadable = True
                    continue
                if not pdf_is_csr_relevant(text):
                    continue
                if has_financial_figures(text) and mentions_company(company, text):
                    score = score_candidate_text(company, text, url)
                    if best_candidate is None or score > best_candidate[0]:
                        best_candidate = (score, make_source("annual_report", 4, url, text, "FOUND", "pdf_prior_fy"))
            if best_candidate and count_financial_figures(best_candidate[1].get("text", "")) > 0:
                break

    if pdf_found_but_unreadable and (not best_candidate or count_financial_figures(best_candidate[1].get("text", "")) == 0) and await _within_deadline(deadline):
        recovered = await _recover_from_unreadable_document(
            company, budget, quota_guard, deadline, "annual_report",
        )
        if recovered:
            url, text = recovered
            score = score_candidate_text(company, text, url)
            if best_candidate is None or score > best_candidate[0]:
                best_candidate = (score, make_source("annual_report", 4, url, text, "FOUND", "unreadable_recovery"))
            logger.info("annual_report recovered via unreadable-document fallback company=%r url=%s", company, url)

    chosen = best_candidate or weak_candidate
    if chosen and pdf_found_but_unreadable:
        chosen[1]["fetch_method"] = f"{chosen[1].get('fetch_method', '')}_unreadable_recovered".strip("_")
    logger.info(
        "annual_report DONE company=%r urls_tried=%d found=%s pdf_found_but_unreadable=%s",
        company, urls_tried, bool(chosen), pdf_found_but_unreadable,
    )

    if chosen:
        if registry is not None:
            registry.register_core_source(chosen[1])
        budget.mark_category_hit("annual_report")
        return chosen[1]

    fallback = make_source("annual_report", 4, status="NOT_FOUND")
    if pdf_found_but_unreadable:
        fallback["fetch_method"] = "pdf_found_unreadable"
    return fallback


async def fetch_multi_year_financials(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None,
                                       registry: SourceRegistry | None = None, job_deadline: float | None = None,
                                       annual_report_source: dict | None = None) -> dict:
    await _check_job_deadline(job_deadline)
    deadline = time.monotonic() + SOURCE_DEADLINE_SECONDS
    if job_deadline is not None:
        deadline = min(deadline, job_deadline)

    annual_report_years = count_distinct_year_tokens((annual_report_source or {}).get("text", ""))
    if annual_report_years >= 2:
        return make_source("multi_year_financials", 10, status="NOT_TRIED")

    fy1, fy2, fy3 = CURRENT_FY_LABEL, PRIOR_FY_LABELS[0], PRIOR_FY_LABELS[1]
    best_candidate = None

    for template in MULTI_YEAR_FINANCIAL_QUERIES:
        if not await _within_deadline(deadline):
            break
        query = template.format(c=company, fy1=fy1, fy2=fy2, fy3=fy3)
        results = await search_web(
            query, budget, max_results=8, quota_guard=quota_guard, category="multi_year_financials",
        )
        for result in results:
            if not await _within_deadline(deadline):
                break
            url = result.get("href", "")
            title = result.get("title", "")
            body = result.get("body", "")
            if not url or not mentions_company(company, f"{title} {body}"):
                continue
            if url.lower().endswith(".pdf"):
                text = await fetch_pdf_text(url)
                if text and not pdf_is_csr_relevant(text):
                    continue
            else:
                text = await fetch_page_text(url) or body
            if not (text and len(text) > 200 and mentions_company(company, text)):
                continue
            if count_financial_figures(text) < 2:
                continue
            year_tokens = count_distinct_year_tokens(text)
            score = score_candidate_text(company, text, url)
            candidate = (year_tokens, score, make_source("multi_year_financials", 10, url, text, "FOUND", "search"))
            if best_candidate is None:
                best_candidate = candidate
            elif year_tokens >= 2 and best_candidate[0] < 2:
                best_candidate = candidate
            elif (year_tokens >= 2) == (best_candidate[0] >= 2) and score > best_candidate[1]:
                best_candidate = candidate
        if best_candidate and best_candidate[0] >= 2:
            break

    if not best_candidate and await _within_deadline(deadline):
        recovered = await _recover_via_secondary_search(
            company, budget, quota_guard, deadline, "multi_year_financials", CSR_SPEND_QUERIES, min_len=150,
        )
        if recovered:
            url, text = recovered
            year_tokens = count_distinct_year_tokens(text)
            score = score_candidate_text(company, text, url)
            best_candidate = (year_tokens, score, make_source("multi_year_financials", 10, url, text, "FOUND", "csr_spend_recovery"))

    if best_candidate:
        chosen = best_candidate[2]
        if registry is not None:
            registry.register_core_source(chosen)
        budget.mark_category_hit("multi_year_financials")
        return chosen

    return make_source("multi_year_financials", 10, status="NOT_FOUND")


async def fetch_partner_source(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None,
                                registry: SourceRegistry | None = None, job_deadline: float | None = None,
                                related_entities: list[dict] | None = None, mode: str = "deep") -> dict:
    await _check_job_deadline(job_deadline)
    deadline = time.monotonic() + SOURCE_DEADLINE_SECONDS
    if job_deadline is not None:
        deadline = min(deadline, job_deadline)
    max_partner_sources = MAX_PARTNER_SOURCES_DEEP if mode == "deep" else MAX_PARTNER_SOURCES_SCREEN
    candidates: list[tuple[float, str, str]] = []
    seen_urls: set[str] = set()
    urls_tried = 0

    search_targets = [company] + related_entity_names(related_entities)[:2]

    for target in search_targets:
        for template in PARTNER_QUERIES:
            if not await _within_deadline(deadline):
                break
            if len(candidates) >= max_partner_sources * 2:
                break
            query = template.format(c=target)
            results = await search_web(query, budget, max_results=6, quota_guard=quota_guard, category="partner_search")
            for result in results:
                if not await _within_deadline(deadline):
                    break
                url = result.get("href", "")
                title = result.get("title", "")
                body = result.get("body", "")
                if not url or url in seen_urls:
                    continue
                if not mentions_company_specifically(company, f"{title} {body}") and not mentions_company_specifically(target, f"{title} {body}"):
                    continue

                if "linkedin.com" in url:
                    snippet_text = f"{title}. {body}".strip()
                    if len(snippet_text) < 40 or not _PARTNER_RELEVANCE_KEYWORD_PATTERN.search(snippet_text):
                        continue
                    seen_urls.add(url)
                    urls_tried += 1
                    score = score_candidate_text(company, snippet_text, url)
                    candidates.append((score, url, snippet_text))
                    continue

                if any(domain in url for domain in AGGREGATOR_DOMAINS):
                    continue
                seen_urls.add(url)
                urls_tried += 1
                is_pdf = url.lower().endswith(".pdf")
                text = await (fetch_pdf_text(url) if is_pdf else fetch_page_text(url)) or body
                if is_pdf and text and not pdf_is_csr_relevant(text):
                    continue
                if not text or len(text) < 150 or not mentions_company(company, text):
                    continue
                if not is_csr_relevant(text) and not _PARTNER_RELEVANCE_KEYWORD_PATTERN.search(text):
                    continue
                if is_non_india_geo_dominant(text) and not has_india_location_signal(text):
                    continue
                score = score_candidate_text(company, text, url)
                if has_india_location_signal(text):
                    score += 3.0
                candidates.append((score, url, text))

            if len(candidates) >= max_partner_sources * 2:
                break
        if len(candidates) >= max_partner_sources * 2:
            break

    candidate_names: list[str] = []
    for _, _, text in candidates:
        for name in _extract_named_partner_candidates(company, text):
            if name not in candidate_names:
                candidate_names.append(name)

    followup_budget = MAX_PARTNER_FOLLOWUP_NAMES if mode == "deep" else max(1, MAX_PARTNER_FOLLOWUP_NAMES - 1)
    if candidate_names and await _within_deadline(deadline):
        for partner_name in candidate_names[:followup_budget]:
            if not await _within_deadline(deadline):
                break
            for template in PARTNER_FOLLOWUP_QUERIES:
                query = template.format(partner=partner_name, c=company)
                results = await search_web(query, budget, max_results=4, quota_guard=quota_guard, category="partner_search")
                for result in results:
                    url = result.get("href", "")
                    title = result.get("title", "")
                    body = result.get("body", "")
                    if not url or url in seen_urls or any(domain in url for domain in AGGREGATOR_DOMAINS):
                        continue
                    if not mentions_company_specifically(company, f"{title} {body}"):
                        continue
                    seen_urls.add(url)
                    urls_tried += 1
                    text = await fetch_page_text(url) or body
                    if not text or len(text) < 120 or not mentions_company(company, text):
                        continue
                    score = score_candidate_text(company, text, url) + 3.0
                    candidates.append((score, url, text))

    if not candidates and await _within_deadline(deadline):
        recovered = await _recover_from_unreadable_document(
            company, budget, quota_guard, deadline, "partner_search", seed_names=candidate_names, min_len=150,
        )
        if recovered:
            url, text = recovered
            score = score_candidate_text(company, text, url)
            candidates.append((score, url, text))

    logger.info(
        "partner_search DONE company=%r urls_tried=%d candidates_found=%d mode=%s",
        company, urls_tried, len(candidates), mode,
    )

    if not candidates:
        return make_source("partner_search", 5, status="NOT_FOUND")

    candidates.sort(key=lambda c: c[0], reverse=True)
    top = candidates[:max_partner_sources]
    combined_text = "\n\n---\n\n".join(f"[{url}]\n{text[:2500]}" for _, url, text in top)
    primary_url = top[0][1]
    source = make_source("partner_search", 5, primary_url, normalize_block_text(combined_text, 10000), "FOUND", "search")

    if registry is not None:
        registry.register_core_source(source)
        for _, url, text in top[1:]:
            registry.register_child_hit(
                source_name="partner_search", url=url, label="Partner search result", excerpt=text[:200],
            )

    budget.mark_category_hit("partner_search", min(len(top), 2))
    return source


async def fetch_education_programme_source(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None,
                                            registry: SourceRegistry | None = None, job_deadline: float | None = None,
                                            related_entities: list[dict] | None = None, mode: str = "deep") -> dict:
    await _check_job_deadline(job_deadline)
    deadline = time.monotonic() + SOURCE_DEADLINE_SECONDS
    if job_deadline is not None:
        deadline = min(deadline, job_deadline)

    max_programme_sources = MAX_PROGRAMME_SOURCES_DEEP if mode == "deep" else MAX_PROGRAMME_SOURCES_SCREEN
    candidates: list[tuple[float, str, str]] = []
    seen_urls: set[str] = set()
    urls_tried = 0

    search_targets = [company] + related_entity_names(related_entities)[:2]

    for target in search_targets:
        site_token = _domain_site_token(budget.resolved_domains) if target == company else ""
        for template in EDUCATION_PROGRAMME_QUERIES:
            if not await _within_deadline(deadline):
                break
            if len(candidates) >= max_programme_sources * 2:
                break
            query = template.format(c=target, site=site_token).strip()
            results = await search_web(query, budget, max_results=6, quota_guard=quota_guard, category="education_programme_search")
            for result in results:
                if not await _within_deadline(deadline):
                    break
                url = result.get("href", "")
                title = result.get("title", "")
                body = result.get("body", "")
                if not url or url in seen_urls or any(domain in url for domain in AGGREGATOR_DOMAINS):
                    continue
                if not mentions_company(company, f"{title} {body}") and not mentions_company(target, f"{title} {body}"):
                    continue
                seen_urls.add(url)
                urls_tried += 1
                is_pdf = url.lower().endswith(".pdf")
                text = await (fetch_pdf_text(url) if is_pdf else fetch_page_text(url)) or body
                if not text or len(text) < 150 or not mentions_company(company, text):
                    continue
                if not has_india_or_education_signal(text):
                    continue
                if is_non_india_geo_dominant(text) and not has_india_location_signal(text):
                    continue
                score = score_candidate_text(company, text, url) + 5.0
                if has_india_location_signal(text):
                    score += 3.0
                candidates.append((score, url, text))

            if len(candidates) >= max_programme_sources * 2:
                break
        if len(candidates) >= max_programme_sources * 2:
            break

    programme_names: list[str] = []
    for _, _, text in candidates:
        for name in _extract_named_programme_candidates(text):
            if name not in programme_names:
                programme_names.append(name)

    deep_dive_budget = MAX_PROGRAMME_DEEP_DIVE_NAMES if mode == "deep" else max(1, MAX_PROGRAMME_DEEP_DIVE_NAMES - 1)
    if programme_names and await _within_deadline(deadline):
        for programme_name in programme_names[:deep_dive_budget]:
            if not await _within_deadline(deadline):
                break
            query = PROGRAMME_DEEP_DIVE_QUERY_TEMPLATE.format(programme=programme_name, c=company)
            results = await search_web(query, budget, max_results=5, quota_guard=quota_guard, category="education_programme_search")
            for result in results:
                url = result.get("href", "")
                title = result.get("title", "")
                body = result.get("body", "")
                if not url or url in seen_urls or any(domain in url for domain in AGGREGATOR_DOMAINS):
                    continue
                if not mentions_company(company, f"{title} {body}"):
                    continue
                seen_urls.add(url)
                urls_tried += 1
                text = await fetch_page_text(url) or body
                if not text or len(text) < 120 or not mentions_company(company, text):
                    continue
                if is_non_india_geo_dominant(text) and not has_india_location_signal(text):
                    continue
                score = score_candidate_text(company, text, url) + 4.0
                candidates.append((score, url, text))

    if not candidates and await _within_deadline(deadline):
        recovered = await _recover_from_unreadable_document(
            company, budget, quota_guard, deadline, "education_programme_search",
            seed_names=programme_names, min_len=150,
        )
        if recovered:
            url, text = recovered
            score = score_candidate_text(company, text, url) + 2.0
            candidates.append((score, url, text))

    logger.info(
        "education_programme_search DONE company=%r urls_tried=%d candidates_found=%d mode=%s",
        company, urls_tried, len(candidates), mode,
    )

    if not candidates:
        return make_source("education_programme_search", 9, status="NOT_FOUND")

    candidates.sort(key=lambda c: c[0], reverse=True)
    top = candidates[:max_programme_sources]
    combined_text = "\n\n---\n\n".join(f"[{url}]\n{text[:2500]}" for _, url, text in top)
    primary_url = top[0][1]
    source = make_source("education_programme_search", 9, primary_url, normalize_block_text(combined_text, 10000), "FOUND", "search")

    if registry is not None:
        registry.register_core_source(source)
        for _, url, text in top[1:]:
            registry.register_child_hit(
                source_name="education_programme_search", url=url, label="Programme search result", excerpt=text[:200],
            )

    budget.mark_category_hit("education_programme_search", min(len(top), 2))
    return source


async def _run_linkedin_query_batch(company: str, queries: list[str], budget: SearchBudget,
                                     quota_guard, deadline: float, add_hit_fn, max_hits: int) -> int:
    collected = 0
    for query_template in queries:
        if not await _within_deadline(deadline) or collected >= max_hits:
            break
        if not budget.google_has_budget("people_search"):
            break
        if google_cse_is_broken():
            break
        role_hint = ""
        if '"' in query_template:
            parts = query_template.split('"')
            if len(parts) >= 4:
                role_hint = parts[3]
        budget.record_google_query("people_search")
        try:
            profiles = await google_search.google_search_linkedin_profiles(
                company, role_hint=role_hint, max_results=8, quota_guard=quota_guard,
            )
        except google_search.GoogleCseInvalidArgumentError as exc:
            async with _GOOGLE_CSE_BROKEN_LOCK:
                _mark_google_cse_broken(str(exc))
            break
        budget.record_query_results("people_search", len(profiles))
        for profile in profiles:
            url = profile.get("href", "")
            if not is_literal_linkedin_profile_url(url):
                continue
            if add_hit_fn(profile.get("title", ""), profile.get("body", ""), url):
                collected += 1
    return collected


async def _search_named_csr_contact_snippets(company: str, budget: SearchBudget, quota_guard,
                                              deadline: float, add_hit_fn, max_hits: int) -> int:
    collected = 0
    for query_template in LINKEDIN_PEOPLE_NAME_ONLY_FALLBACK_QUERIES:
        if not await _within_deadline(deadline) or collected >= max_hits:
            break
        results = await search_web(
            query_template.format(c=company), budget, max_results=6, quota_guard=quota_guard, category="people_search",
        )
        for result in results:
            url = result.get("href", "")
            title = result.get("title", "")
            body = result.get("body", "")
            if not url or "linkedin.com" in url:
                continue
            if add_hit_fn(title, body, url):
                collected += 1
    return collected


async def fetch_linkedin_people(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None,
                                 registry: SourceRegistry | None = None, job_deadline: float | None = None) -> dict:
    await _check_job_deadline(job_deadline)
    deadline = time.monotonic() + SOURCE_DEADLINE_SECONDS
    if job_deadline is not None:
        deadline = min(deadline, job_deadline)
    hits: list[dict] = []
    seen_urls: set[str] = set()

    def _add_hit(raw_title: str, snippet: str, url: str) -> bool:
        if url in seen_urls:
            return False
        parsed = parse_linkedin_hit(raw_title, snippet, url, company)
        if not parsed["name"] or not parsed["has_csr_signal"]:
            return False
        if not mentions_company_specifically(company, f"{raw_title} {snippet}"):
            return False
        seen_urls.add(url)
        if registry is not None:
            parsed["source_number"] = registry.register_child_hit(
                source_name="people_search", url=url, label=f"LinkedIn — {parsed['name']}",
                excerpt=f"{parsed['title']} — {parsed['snippet']}"[:280],
            )
        hits.append(parsed)
        return True

    await _run_linkedin_query_batch(company, LINKEDIN_PEOPLE_QUERIES, budget, quota_guard, deadline, _add_hit, 15)

    strong_hits = [h for h in hits if h.get("confidence") in ("HIGH", "MEDIUM")]
    india_signal_hits = [h for h in hits if h.get("india_location_signal")]

    if not strong_hits and await _within_deadline(deadline):
        await _search_named_csr_contact_snippets(company, budget, quota_guard, deadline, _add_hit, 8)

    if not hits:
        return make_source("people_search", 6, status="NOT_FOUND")

    hits.sort(key=lambda h: (
        h.get("confidence") != "HIGH",
        h.get("confidence") != "MEDIUM",
        not h.get("india_location_signal"),
    ))
    high_confidence_hits = [h for h in hits if h.get("confidence") == "HIGH"]
    medium_confidence_hits = [h for h in hits if h.get("confidence") == "MEDIUM"]
    low_confidence_hits = [h for h in hits if h.get("confidence") not in ("HIGH", "MEDIUM")]
    final_hits = (high_confidence_hits + medium_confidence_hits)[:10] or low_confidence_hits[:6]

    combined_text = " || ".join(f"{hit['name']} — {hit['title']} — {hit['snippet']}" for hit in final_hits)
    source = make_source("people_search", 6, final_hits[0]["url"], clean_text(combined_text, 4000), "FOUND", "search_snippets")
    source["people_hits"] = final_hits
    source["used_global_fallback"] = not bool(india_signal_hits) and bool(final_hits)
    if registry is not None:
        registry.register_core_source(source)
    budget.mark_category_hit("people_search", min(len(high_confidence_hits) + len(medium_confidence_hits), 2) or 1)
    logger.info(
        "people_search DONE company=%r hits_total=%d high_confidence=%d medium_confidence=%d",
        company, len(hits), len(high_confidence_hits), len(medium_confidence_hits),
    )
    return source


async def fetch_plans_source(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None,
                              max_pages: int = 3, registry: SourceRegistry | None = None,
                              job_deadline: float | None = None) -> dict:
    await _check_job_deadline(job_deadline)
    deadline = time.monotonic() + SOURCE_DEADLINE_SECONDS
    if job_deadline is not None:
        deadline = min(deadline, job_deadline)
    hits, fetched_texts, first_url = [], [], ""
    for query_template in PLAN_QUERIES:
        if not await _within_deadline(deadline):
            break
        if len(fetched_texts) >= max_pages:
            break
        results = await search_web(
            query_template.format(c=company), budget, max_results=5, quota_guard=quota_guard, category="plans_search",
        )
        for result in results:
            if not await _within_deadline(deadline):
                break
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
                if text and len(text) > 200 and is_csr_relevant(text) and mentions_company(company, text):
                    fetched_texts.append(text)
                    first_url = first_url or url

    if not hits and not fetched_texts:
        return make_source("plans_search", 7, status="NOT_FOUND")

    combined_text = " || ".join(fetched_texts) if fetched_texts else " || ".join(f"{hit['title']} — {hit['snippet']}" for hit in hits)
    source = make_source(
        "plans_search", 7, first_url or hits[0]["url"], normalize_block_text(combined_text, 7000), "FOUND",
        "search" if fetched_texts else "search_snippets",
    )
    source["plan_hits"] = hits[:10]
    if registry is not None:
        registry.register_core_source(source)
    budget.mark_category_hit("plans_search")
    return source


async def fetch_sector_eligibility_source(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None,
                                           registry: SourceRegistry | None = None,
                                           job_deadline: float | None = None) -> dict:
    await _check_job_deadline(job_deadline)
    deadline = time.monotonic() + SOURCE_DEADLINE_SECONDS
    if job_deadline is not None:
        deadline = min(deadline, job_deadline)
    hits, fetched_texts, first_url = [], [], ""
    for query_template in SECTOR_QUERIES:
        if not await _within_deadline(deadline):
            break
        if len(fetched_texts) >= 3:
            break
        results = await search_web(
            query_template.format(c=company), budget, max_results=5,
            quota_guard=quota_guard, category="sector_eligibility_search",
        )
        for result in results:
            if not await _within_deadline(deadline):
                break
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
                if text and len(text) > 150 and mentions_company(company, text):
                    fetched_texts.append(text)
                    first_url = first_url or url

    if not hits and not fetched_texts:
        return make_source("sector_eligibility_search", 8, status="NOT_FOUND")

    combined_text = " || ".join(fetched_texts) if fetched_texts else " || ".join(f"{hit['title']} — {hit['snippet']}" for hit in hits)
    source = make_source(
        "sector_eligibility_search", 8, first_url or hits[0]["url"], normalize_block_text(combined_text, 6000), "FOUND",
        "search" if fetched_texts else "search_snippets",
    )
    if registry is not None:
        registry.register_core_source(source)
    budget.mark_category_hit("sector_eligibility_search")
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
        results = await search_web(
            query, budget, max_results=5, quota_guard=quota_guard, category=f"followup_{question_category}",
        )
        for result in results:
            if not await _within_deadline(deadline):
                break
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
    budget = SearchBudget(company, max_google_queries=24)

    source_9 = await fetch_education_programme_source(company, search_cfg, budget, quota_guard, registry=registry, mode="screen")
    source_1 = await fetch_india_csr_page(company, search_cfg, budget, quota_guard, registry=registry)
    source_4 = await fetch_annual_report(company, search_cfg, budget, quota_guard, registry=registry)
    source_10 = await fetch_multi_year_financials(company, search_cfg, budget, quota_guard, registry=registry, annual_report_source=source_4)
    source_6 = await fetch_linkedin_people(company, search_cfg, budget, quota_guard, registry=registry)
    source_2 = await fetch_mca_portal(company, search_cfg, budget, quota_guard, registry=registry)
    source_5 = await fetch_partner_source(company, search_cfg, budget, quota_guard, registry=registry, mode="screen")

    source_3 = make_source("national_csr_portal", 3, status="NOT_TRIED")
    source_7 = make_source("plans_search", 7, status="NOT_TRIED")
    source_8 = make_source("sector_eligibility_search", 8, status="NOT_TRIED")

    sources = [source_1, source_2, source_3, source_4, source_5, source_6, source_7, source_8, source_9, source_10]
    found_count = sum(1 for s in sources if s.get("status") == "FOUND")
    logger.info(
        "fetch_screen_sources DONE company=%r found=%d/10 google_used=%d category_breakdown=%s",
        company, found_count, budget.google_queries_used, budget.category_used,
    )
    return sources


async def fetch_deep_sources(company: str, search_cfg: dict, quota_guard=None, progress_cb=None,
                              registry: SourceRegistry | None = None) -> list[dict]:
    registry = registry or SourceRegistry(company)
    budget = SearchBudget(company)
    job_deadline = time.monotonic() + DEEP_JOB_HARD_DEADLINE_SECONDS

    def not_tried(name: str, num: int) -> dict:
        return make_source(name, num, status="NOT_TRIED")

    async def advance_step(message: str):
        if progress_cb:
            await progress_cb(message)

    related_entities: list[dict] = []

    try:
        await advance_step("Education programmes and decision-makers first...")
        source_9 = await fetch_education_programme_source(
            company, search_cfg, budget, quota_guard, registry=registry, job_deadline=job_deadline,
            related_entities=related_entities, mode="deep",
        )
        source_6 = await fetch_linkedin_people(company, search_cfg, budget, quota_guard, registry=registry, job_deadline=job_deadline)

        await advance_step("Mapping related entities — parent, India branch, foundation...")
        related_entities = await discover_related_entities(company, search_cfg, budget, quota_guard, deadline=job_deadline)
        if registry is not None and related_entities:
            for entity in related_entities:
                registry.register_child_hit(
                    source_name="entity_resolution", url="",
                    label=f"Related entity — {entity.get('entity_name', '')} ({entity.get('entity_type', '')})",
                    excerpt="",
                )

        await advance_step("CSR page, MCA, National CSR Portal, annual report...")
        source_1 = await fetch_india_csr_page(company, search_cfg, budget, quota_guard, registry=registry, job_deadline=job_deadline)
        source_2 = await fetch_mca_portal(company, search_cfg, budget, quota_guard, registry=registry, job_deadline=job_deadline)
        source_3 = await fetch_national_csr_portal(company, search_cfg, budget, quota_guard, registry=registry, job_deadline=job_deadline)
        source_4 = await fetch_annual_report(company, search_cfg, budget, quota_guard, registry=registry, job_deadline=job_deadline)
        source_10 = await fetch_multi_year_financials(company, search_cfg, budget, quota_guard, registry=registry, job_deadline=job_deadline, annual_report_source=source_4)

        await advance_step("Partners, plans, sector, education programmes follow-up...")
        source_5 = await fetch_partner_source(
            company, search_cfg, budget, quota_guard, registry=registry, job_deadline=job_deadline,
            related_entities=related_entities, mode="deep",
        )
        source_7 = await fetch_plans_source(company, search_cfg, budget, quota_guard, registry=registry, job_deadline=job_deadline)
        source_8 = await fetch_sector_eligibility_source(company, search_cfg, budget, quota_guard, registry=registry, job_deadline=job_deadline)
        if related_entities:
            source_9 = await fetch_education_programme_source(
                company, search_cfg, budget, quota_guard, registry=registry, job_deadline=job_deadline,
                related_entities=related_entities, mode="deep",
            )
    except DeepJobDeadlineExceeded:
        logger.warning("fetch_deep_sources hit hard job deadline company=%r", company)
        existing = locals()
        source_1 = existing.get("source_1") or not_tried("india_csr_page", 1)
        source_2 = existing.get("source_2") or not_tried("mca_portal", 2)
        source_3 = existing.get("source_3") or not_tried("national_csr_portal", 3)
        source_4 = existing.get("source_4") or not_tried("annual_report", 4)
        source_5 = existing.get("source_5") or not_tried("partner_search", 5)
        source_6 = existing.get("source_6") or not_tried("people_search", 6)
        source_7 = existing.get("source_7") or not_tried("plans_search", 7)
        source_8 = existing.get("source_8") or not_tried("sector_eligibility_search", 8)
        source_9 = existing.get("source_9") or not_tried("education_programme_search", 9)
        source_10 = existing.get("source_10") or not_tried("multi_year_financials", 10)

    sources = [source_1, source_2, source_3, source_4, source_5, source_6, source_7, source_8, source_9, source_10]

    total_figures = sum(count_financial_figures(s.get("text", "")) for s in sources)
    if total_figures == 0 and budget.google_has_budget("csr_budget") and time.monotonic() < job_deadline:
        spend_deadline = min(time.monotonic() + SOURCE_DEADLINE_SECONDS, job_deadline)
        for template in CSR_SPEND_QUERIES:
            if time.monotonic() >= spend_deadline or not budget.google_has_budget("csr_budget"):
                break
            results = await search_web(
                template.format(c=company, fy=CURRENT_FY_LABEL), budget, max_results=6,
                quota_guard=quota_guard, category="csr_budget",
            )
            for result in results:
                if time.monotonic() >= spend_deadline:
                    break
                url = result.get("href", "")
                body = result.get("body", "")
                if not url or not mentions_company(company, body):
                    continue
                text = await (fetch_pdf_text(url) if url.lower().endswith(".pdf") else fetch_page_text(url)) or body
                if text and has_financial_figures(text) and mentions_company(company, text):
                    source_4 = make_source("annual_report", 4, url, text, "FOUND", "spend_fallback")
                    registry.register_core_source(source_4)
                    sources[3] = source_4
                    budget.mark_category_hit("csr_budget")
                    break
            if count_financial_figures(sources[3].get("text", "")) > 0:
                break

    found_count = sum(1 for s in sources if s.get("status") == "FOUND")
    logger.info(
        "fetch_deep_sources DONE company=%r found=%d/10 total_financial_figures=%d source_bank_entries=%d "
        "google_used=%d related_entities=%d category_breakdown=%s",
        company, found_count, sum(count_financial_figures(s.get("text", "")) for s in sources),
        len(registry.entries()), budget.google_queries_used, len(related_entities), budget.category_used,
    )
    gc.collect()
    return sources