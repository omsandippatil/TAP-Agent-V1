import asyncio
import gc
import logging
import re
import time
from datetime import datetime
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from app.config import settings
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


def _verbose() -> bool:
    return bool(getattr(settings, "verbose_pipeline_logging", True))


def _vlog(level: int, msg: str, *args) -> None:
    if _verbose():
        logger.log(level, msg, *args)


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


def is_unreliable_for_blind_guessing(company: str) -> bool:
    return False


is_generic_company_name = is_unreliable_for_blind_guessing

AGGREGATOR_DOMAINS = (
    "youtube.", "twitter.", "x.com", "facebook.", "instagram.", "linkedin.",
    "wikipedia.", "glassdoor.", "indeed.", "crunchbase.", "bloomberg.",
    "zaubacorp", "tofler.", "justdial.", "indiamart.", "ambitionbox.",
    "moneycontrol.", "economictimes.", "livemint.", "reuters.",
    "apkpure.", "h1bgrader.", "quora.", "reddit.", "pinterest.",
    "medium.com", "slideshare.", "scribd.", "vimeo.", "tiktok.",
    "naukri.", "shine.com", "timesjobs.", "monsterindia.", "tracxn.",
    "pdfcoffee.", "coursehero.", "studocu.", "yumpu.", "docplayer.",
    "academia.edu", "researchgate.", "issuu.",
)

UNTRUSTED_ENTITY_RESOLUTION_DOMAINS = AGGREGATOR_DOMAINS + (
    "pdfcoffee.", "scribd.", "coursehero.", "studocu.", "yumpu.",
    "docplayer.", "issuu.", "slideshare.",
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

CSR_PAGE_PATHS_LOCALE_PREFIXED = [
    "/india/en/india/esg", "/india/en/csr", "/india/en/esg",
    "/in/en/csr", "/en-in/csr", "/en/india/csr", "/global/india/csr",
    "/india/en/csr", "/india/en/esg", "/in/en/csr", "/in/en/esg",
    "/en-in/csr", "/en-in/esg", "/india/csr", "/india/esg",
    "/india/en/about/csr", "/india/en/about/esg",
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

PROGRAMME_NARRATIVE_KEYWORDS = [
    "students", "student", "school", "schools", "beneficiar", "stem",
    "teacher", "curriculum", "government school", "public school",
    "classroom", "learning", "skilling", "skill development", "digital literacy",
    "robotics", "coding", "workshop", "scholarship", "fellowship", "mentorship",
    "training programme", "training program", "capacity building",
]

FINANCIAL_ANNEXURE_KEYWORDS = [
    "schedule vii", "annexure", "amount spent", "csr expenditure", "csr committee",
    "csr policy", "board report", "prescribed csr", "unspent amount",
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

KNOWN_MULTI_COMPANY_BRANDS = frozenset([
    "ibm", "accenture", "tcs", "infosys", "wipro", "hcl", "capgemini", "cognizant",
    "genpact", "deloitte", "kpmg", "ey", "pwc", "microsoft", "google", "amazon",
    "oracle", "sap", "salesforce", "adobe", "intel", "dell", "hp", "hewlett packard",
    "cerner", "cisco", "vmware", "nvidia", "meta", "apple", "samsung", "tech mahindra",
    "mindtree", "mphasis", "ltimindtree", "larsen", "reliance", "tata", "birla",
    "mahindra", "bajaj", "adani", "hindustan unilever", "itc limited", "ntpc",
])


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


def current_fiscal_year_label(as_of: datetime | None = None) -> str:
    reference = as_of or datetime.utcnow()
    start_year = reference.year if reference.month >= 4 else reference.year - 1
    end_year = (start_year + 1) % 100
    return f"FY{start_year}-{end_year:02d}"


def prior_fiscal_year_labels(current_label: str, count: int) -> list[str]:
    match = re.match(r"FY(\d{4})-(\d{2})", current_label)
    if not match:
        return []
    start_year = int(match.group(1))
    labels = []
    for offset in range(1, count + 1):
        prior_start = start_year - offset
        prior_end = (prior_start + 1) % 100
        labels.append(f"FY{prior_start}-{prior_end:02d}")
        labels.append(f"{prior_start}-{prior_end:02d}")
    return labels


CURRENT_FY_LABEL = current_fiscal_year_label()
FISCAL_YEAR_LOOKBACK_COUNT = 5
PRIOR_FY_LABELS = prior_fiscal_year_labels(CURRENT_FY_LABEL, FISCAL_YEAR_LOOKBACK_COUNT)
CALENDAR_YEAR_LOOKBACK_COUNT = 5


def recent_calendar_years(as_of: datetime | None = None, count: int = CALENDAR_YEAR_LOOKBACK_COUNT) -> list[str]:
    reference = as_of or datetime.utcnow()
    return [str(reference.year - offset) for offset in range(count)]


RECENT_CALENDAR_YEARS = recent_calendar_years()

FY_YEAR_TOKEN_PATTERN = re.compile(r"FY\s?20?\d{2}[-–]\d{2,4}|20\d{2}[-–]\d{2,4}", re.IGNORECASE)

EDUCATION_PROGRAMME_QUERIES = [
    '"{c}" ("school education" OR "government school" OR "public school") CSR India named programme students {site}',
    '"{c}" (STEM OR AI OR "artificial intelligence" OR coding OR "digital skills" OR "digital literacy") CSR India students named programme {site}',
    '"{c}" ("government school" OR "public school" OR teachers OR students) CSR India skilling named programme beneficiaries',
    '"{c}" (STEM OR robotics OR "digital literacy" OR coding) CSR India schools named programme annual report filetype:pdf',
    '"{c}" ("government school" OR STEM OR "digital skills") CSR India NGO partner beneficiaries press release',
    '"{c}" CSR India programme name students schools {site}',
]

CSR_PAGE_QUERIES = [
    '"{c}" (corporate social responsibility OR "CSR policy" OR "sustainability report" OR "ESG report") India {site}',
    '"{c}" CSR India filetype:pdf {site}',
    '"{c}" ("CSR" OR "ESG" OR "sustainability") India {site}',
]

ANNUAL_REPORT_QUERIES_YEAR_AGNOSTIC = [
    '"{c}" ("annual report" OR "sustainability report" OR "CSR report" OR "ESG report") India filetype:pdf {site}',
    '"{c}" ("annual report" OR "sustainability report" OR "CSR report" OR "ESG report") filetype:pdf {site}',
]

ANNUAL_REPORT_QUERIES_FY_SPECIFIC = [
    '"{c}" ("annual report" OR "business responsibility and sustainability report") {fy} India CSR filetype:pdf {site}',
]

ANNUAL_REPORT_QUERIES_CALENDAR_YEAR = [
    '"{c}" ("annual report" OR "sustainability report" OR "CSR report") {year} India filetype:pdf',
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
    '"{c}" India "Private Limited" CIN MCA site:mca.gov.in',
    '"{c}" India "Private Limited" CIN MCA',
    '"{c}" India registered office "Private Limited"',
]

RELATED_ENTITY_DISCOVERY_QUERIES = [
    '"{c}" (foundation OR "India branch" OR subsidiary) CSR corporate social responsibility',
]

PARTNER_QUERIES = [
    '"{c}" CSR (NGO partner OR "implementation partner" OR "implementing partner") India education {site}',
    'site:linkedin.com/company "{c}" (partnered with OR MoU) NGO CSR India',
    '"{c}" CSR partner NGO announcement press release India',
]

PARTNER_FOLLOWUP_QUERIES = [
    '"{partner}" "{c}" (partnership OR funded OR implementing)',
    '"{partner}" "{c}" CSR India',
]

PLAN_QUERIES = [
    '"{c}" CSR (partnership education OR "request for proposal" OR "call for proposals") India {site}',
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
    '"{c}" India sector industry business overview annual report {site}',
]

PROGRAMME_DEEP_DIVE_QUERY_TEMPLATE = (
    '"{programme}" "{c}" (geography OR beneficiaries OR students OR partner OR scale OR outcomes)'
)

CHAINED_FOLLOWUP_TITLE_TEMPLATE = '"{title}" "{c}" (geography OR beneficiaries OR students OR partner OR scale OR outcomes OR press release)'
CHAINED_FOLLOWUP_DOMAIN_TEMPLATE = '"{title}" {site}'
CHAINED_FOLLOWUP_NGO_TEMPLATE = '"{title}" NGO CSR partnership India'
CHAINED_FOLLOWUP_PRESS_TEMPLATE = '"{c}" "{title}" press release announcement'

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
MAX_PDF_PAGES = 20
FINANCIAL_PDF_SCAN_PAGES = 40
CANDIDATE_EVAL_LIMIT = 3
MIN_ACCEPT_SCORE = 4
STRONG_ACCEPT_SCORE = 10

PAGE_FETCH_TIMEOUT_SECONDS = 8
PDF_FETCH_TIMEOUT_SECONDS = 10
HOMEPAGE_FETCH_TIMEOUT_SECONDS = 6
DNS_CHECK_TIMEOUT_SECONDS = 1.5
SEARCH_TASK_TIMEOUT_SECONDS = 6
FETCH_TASK_TIMEOUT_SECONDS = 10
SOURCE_DEADLINE_SECONDS = 20
FOLLOWUP_DEADLINE_SECONDS = 10
CONCURRENT_FETCH_LIMIT = 3

DEEP_JOB_HARD_DEADLINE_SECONDS = 170
MAX_PDF_DOWNLOAD_BYTES = 15 * 1024 * 1024
PDF_STREAM_CHUNK_BYTES = 262144
MAX_PDF_PAGES_HARD_CAP = 50
SECOND_PASS_TEXT_LENGTH_FLOOR = 500
UNREADABLE_TEXT_LENGTH_FLOOR = 200

MAX_PARTNER_SOURCES_DEEP = 10
MAX_PARTNER_SOURCES_SCREEN = 5
MAX_PROGRAMME_SOURCES_DEEP = 7
MAX_PROGRAMME_SOURCES_SCREEN = 4
MAX_PARTNER_FOLLOWUP_NAMES = 3
MAX_PROGRAMME_DEEP_DIVE_NAMES = 3

MAX_GUESSED_PATH_ATTEMPTS_PER_DOMAIN = 10
DOMAIN_MISS_ESCALATION_THRESHOLD = 7
DOMAIN_KILLING_ERROR_TYPES = {"dns", "ssl", "connection_error"}

MAX_TEASER_NGO_FOLLOWUP_QUERIES = 2
MAX_CHAINED_FOLLOWUP_TARGETS = 4
MAX_CHAINED_FOLLOWUP_QUERIES_PER_TARGET = 3

_FETCH_SEMAPHORE = asyncio.Semaphore(CONCURRENT_FETCH_LIMIT)

_ENTITY_PROXIMITY_WINDOW_CHARS = 60

_PARTNER_RELEVANCE_KEYWORD_PATTERN = re.compile(
    r"\b(partner|partnered|partnership|ngo|foundation|mou|memorandum|collaborat|"
    r"implement|grant|csr|development impact bond|dib|outcomes fund)\b", re.IGNORECASE,
)

QUANTIFIED_BENEFICIARY_PATTERN = re.compile(
    r"\b([\d][\d,]{2,})\s*\+?\s*(students|beneficiaries|children|individuals|lives)\b",
    re.IGNORECASE,
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


def count_programme_narrative_hits(text: str) -> int:
    if not text:
        return 0
    lowered = text.lower()
    return sum(1 for kw in PROGRAMME_NARRATIVE_KEYWORDS if kw in lowered)


def count_financial_annexure_hits(text: str) -> int:
    if not text:
        return 0
    lowered = text.lower()
    return sum(1 for kw in FINANCIAL_ANNEXURE_KEYWORDS if kw in lowered)


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

_SEARCH_DERIVED_BUSINESS_CONTEXT_PATTERN = re.compile(
    r"\b(official|csr|corporate|\.com|website|india\s+\w+|sustainability|foundation|"
    r"annual report|company|subsidiary|headquarter)\b",
    re.IGNORECASE,
)


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


def is_search_derived_domain_trustworthy(company: str, host: str, search_result_title: str,
                                          search_result_body: str) -> bool:
    haystack = f"{search_result_title} {search_result_body}"
    lowered_haystack = haystack.lower()
    lowered_host = (host or "").lower()
    tokens = company_name_tokens(company)
    if not tokens:
        return False
    token_hit = any(token in lowered_host or token in lowered_haystack for token in tokens)
    if not token_hit:
        return False
    return bool(_SEARCH_DERIVED_BUSINESS_CONTEXT_PATTERN.search(haystack))


def mentions_company(company: str, text: str, known_domains: list[str] | None = None,
                      url: str = "") -> bool:
    if not text:
        return False
    if known_domains and url:
        host = urlparse(url).netloc.lower()
        if host and any(host == d or host.endswith("." + d) or d.endswith("." + host) for d in known_domains):
            tokens = company_name_tokens(company)
            if tokens:
                lowered = text.lower()
                if any(token in lowered for token in tokens):
                    return True
            elif company.lower() in text.lower():
                return True
    if is_unreliable_for_blind_guessing(company):
        return _mentions_generic_company_name(company, text)
    lowered = text.lower()
    tokens = company_name_tokens(company)
    if not tokens:
        return company.lower() in lowered
    return any(token in lowered for token in tokens)


def mentions_company_specifically(company: str, text: str) -> bool:
    if not text:
        return False
    if is_unreliable_for_blind_guessing(company):
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


def extract_named_programme_candidates_with_titles(text: str) -> list[dict]:
    if not text:
        return []
    found = []
    seen = set()
    for line in _iter_candidate_lines(text):
        for match in NAMED_INITIATIVE_PATTERN.finditer(line):
            name = re.sub(r"\s+", " ", match.group(1)).strip()
            if not _is_plausible_entity_name(name):
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append({"title": name, "kind": "programme"})
    return found


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
            url = result.get("href", "")
            if any(domain in url for domain in UNTRUSTED_ENTITY_RESOLUTION_DOMAINS):
                continue
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
    if is_unreliable_for_blind_guessing(company):
        _vlog(
            logging.INFO,
            "candidate_domains: skipping brute-force domain guessing for generic/ambiguous "
            "company=%r — domain discovery continues via search-derived discovery "
            "(discover_company_domains) and the CSR_PAGE_QUERIES search-query fallback; "
            "this is a deferral to a still-active path, not a dead end",
            company,
        )
        return []

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
    if is_unreliable_for_blind_guessing(company):
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


def accept_fetched_text(company: str, text: str, min_len: int = 400, known_domains: list[str] | None = None,
                         url: str = "") -> bool:
    return (
        bool(text) and len(text) > min_len and is_csr_relevant(text)
        and mentions_company(company, text, known_domains=known_domains, url=url)
    )


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
    narrative_bonus = min(count_programme_narrative_hits(text) * 1.5, 8.0)
    length_bonus = min(len(text) / 2000.0, 4.0)
    domain_bonus = 6.0 if url and any(gov in url.lower() for gov in OFFICIAL_GOV_DOMAINS) else 0.0
    pdf_bonus = 1.5 if url.lower().endswith(".pdf") else 0.0
    return (
        csr_hits * 2.0 + figure_hits * 5.0 + india_figure_bonus + india_location_bonus
        + education_priority_bonus + narrative_bonus + length_bonus + domain_bonus + pdf_bonus
    )


def _contains_unrelated_brand_token(company: str, candidate: str) -> str:
    company_tokens_set = set(company_name_tokens(company))
    candidate_words = re.findall(r"[A-Z][a-zA-Z]*", candidate)
    for i in range(len(candidate_words)):
        for span in (1, 2):
            if i + span > len(candidate_words):
                continue
            phrase = " ".join(candidate_words[i:i + span]).lower()
            single = candidate_words[i].lower()
            if single in KNOWN_MULTI_COMPANY_BRANDS or phrase in KNOWN_MULTI_COMPANY_BRANDS:
                if single not in company_tokens_set and phrase not in company_tokens_set:
                    return candidate_words[i]
    return ""


def _has_duplicate_legal_suffix_pattern(candidate: str) -> bool:
    suffix_hits = re.findall(
        r"(private\s+limited|pvt\.?\s*ltd\.?|limited|ltd\.?)", candidate, re.IGNORECASE
    )
    return len(suffix_hits) > 1


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
    if _has_duplicate_legal_suffix_pattern(candidate):
        _vlog(
            logging.INFO,
            "is_plausible_legal_entity_name REJECTED company=%r candidate=%r reason=duplicate_legal_suffix",
            company, candidate,
        )
        return False
    unrelated_brand = _contains_unrelated_brand_token(company, candidate)
    if unrelated_brand:
        _vlog(
            logging.INFO,
            "is_plausible_legal_entity_name REJECTED company=%r candidate=%r reason=unrelated_brand_token "
            "unrelated_token=%r",
            company, candidate, unrelated_brand,
        )
        return False
    return True


async def search_web(query: str, budget: SearchBudget, max_results: int = 6,
                      quota_guard=None, category: str = "") -> list[dict]:
    if google_cse_is_broken():
        logger.info("google search skipped, CSE marked broken query=%r category=%r", query, category)
        return []
    if google_search.daily_quota_is_exhausted():
        budget.mark_quota_exhausted_globally()
        logger.info(
            "google search skipped, daily quota exhausted query=%r category=%r", query, category,
        )
        return []
    if not google_search.google_search_configured_and_available(quota_guard):
        return []
    if not budget.google_has_budget(category):
        reason = "quota_exhausted_globally" if budget.quota_exhausted_globally else "budget_exhausted_or_category_satisfied"
        logger.info("google search skipped query=%r category=%r reason=%s", query, category, reason)
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
    else:
        _vlog(
            logging.INFO,
            "google search results query=%r category=%r count=%d urls=%s",
            query, category, len(results), [r.get("href", "") for r in results],
        )
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
    text, _ = await fetch_page_text_with_error(url, max_chars, verify_ssl)
    return text


async def fetch_page_text_with_error(url: str, max_chars: int = MAX_PAGE_TEXT_CHARS,
                                      verify_ssl: bool = True) -> tuple[str, str]:
    async with _FETCH_SEMAPHORE:
        try:
            text, error_type = await asyncio.wait_for(
                asyncio.to_thread(_fetch_page_text_sync, url, max_chars, verify_ssl),
                timeout=FETCH_TASK_TIMEOUT_SECONDS,
            )
            _vlog(
                logging.INFO,
                "fetch_page_text DONE url=%s chars=%d preview=%r",
                url, len(text or ""), (text or "")[:200].replace("\n", " "),
            )
            return text, error_type
        except asyncio.TimeoutError:
            logger.info("fetch_page_text timed out url=%s", url)
            return "", "timeout"


def _score_pdf_page_text(snippet: str) -> float:
    if not snippet:
        return 0.0
    lowered = snippet.lower()
    financial_score = count_financial_figures(snippet) * 3.0
    if has_india_location_signal(snippet):
        financial_score += 2.0
    if "csr" in lowered:
        financial_score += 2.0
    financial_score += count_financial_annexure_hits(snippet) * 3.0

    narrative_score = count_programme_narrative_hits(snippet) * 2.5
    narrative_score += count_priority_education_hits(snippet) * 2.0

    return financial_score + narrative_score


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
        score = _score_pdf_page_text(snippet)
        if score > 0:
            scored_indices.append((score, idx))

    if not scored_indices:
        front_slice = list(range(min(max_pages // 2, total_pages)))
        back_start = max(0, total_pages - (max_pages - len(front_slice)))
        back_slice = list(range(back_start, total_pages))
        return sorted(set(front_slice + back_slice))[:max_pages]

    scored_indices.sort(key=lambda pair: pair[0], reverse=True)
    top_by_score = [idx for _, idx in scored_indices[:max_pages]]

    financial_kept = 0
    narrative_kept = 0
    for idx in top_by_score:
        try:
            snippet = pdf.pages[idx].extract_text() or ""
        except Exception:
            snippet = ""
        if count_financial_annexure_hits(snippet) > 0 or count_financial_figures(snippet) > 2:
            financial_kept += 1
        if count_programme_narrative_hits(snippet) > 0:
            narrative_kept += 1

    if narrative_kept == 0 and len(scored_indices) > max_pages:
        narrative_only_scored = []
        for idx in range(scan_upper):
            if idx in top_by_score:
                continue
            try:
                snippet = pdf.pages[idx].extract_text() or ""
            except Exception:
                snippet = ""
            narrative_signal = count_programme_narrative_hits(snippet) * 2.5 + count_priority_education_hits(snippet) * 2.0
            if narrative_signal > 0:
                narrative_only_scored.append((narrative_signal, idx))
        narrative_only_scored.sort(key=lambda pair: pair[0], reverse=True)
        swap_count = max(1, max_pages // 4)
        for _, idx in narrative_only_scored[:swap_count]:
            if idx not in top_by_score:
                if len(top_by_score) >= max_pages:
                    top_by_score.pop()
                top_by_score.append(idx)

    return sorted(set(top_by_score))


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


_OCR_DEPENDENCIES_CHECKED = False
_OCR_DEPENDENCIES_AVAILABLE = False


def ocr_dependencies_available() -> bool:
    global _OCR_DEPENDENCIES_CHECKED, _OCR_DEPENDENCIES_AVAILABLE
    if _OCR_DEPENDENCIES_CHECKED:
        return _OCR_DEPENDENCIES_AVAILABLE
    _OCR_DEPENDENCIES_CHECKED = True
    try:
        import pytesseract  # noqa: F401
        from pdf2image import convert_from_bytes  # noqa: F401
        _OCR_DEPENDENCIES_AVAILABLE = True
    except Exception as exc:
        _OCR_DEPENDENCIES_AVAILABLE = False
        logger.warning(
            "OCR DEPENDENCIES UNAVAILABLE — pytesseract and/or pdf2image are not importable in "
            "this environment (error=%s). Scanned or image-based PDF CSR filings will return no "
            "text and will be indistinguishable from a genuinely-empty document unless this is "
            "fixed. Install pytesseract, pdf2image, and a poppler runtime to enable OCR fallback.",
            exc,
        )
    return _OCR_DEPENDENCIES_AVAILABLE


def _ocr_pdf_pages(pdf_bytes: bytes, max_pages: int) -> str:
    if not ocr_dependencies_available():
        return ""
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
    except Exception:
        return ""
    try:
        images = convert_from_bytes(pdf_bytes, first_page=1, last_page=max(1, max_pages))
    except Exception as exc:
        logger.info("OCR conversion failed error=%s", exc)
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
            if ocr_dependencies_available():
                ocr_text = _ocr_pdf_pages(pdf_bytes, min(capped_pages, 10))
                if len(ocr_text) > len(combined_text):
                    return ocr_text, ""
            else:
                logger.warning(
                    "PDF url=%s extracted only %d chars of text and OCR is unavailable in this "
                    "environment — this document may be scanned/image-based and its content is "
                    "being silently lost. See ocr_dependencies_available() warning at startup.",
                    url, len(combined_text),
                )
        return combined_text, ""
    except Exception as exc:
        error_type = classify_fetch_error(exc)
        logger.info("fetch_pdf_text failed url=%s error_type=%s", url, error_type)
        return "", error_type
    finally:
        gc.collect()


async def fetch_pdf_text(url: str, max_chars: int = MAX_PDF_TEXT_CHARS, max_pages: int = MAX_PDF_PAGES) -> str:
    text, _ = await fetch_pdf_text_with_error(url, max_chars, max_pages)
    return text


async def fetch_pdf_text_with_error(url: str, max_chars: int = MAX_PDF_TEXT_CHARS,
                                     max_pages: int = MAX_PDF_PAGES) -> tuple[str, str]:
    async with _FETCH_SEMAPHORE:
        try:
            text, error_type = await asyncio.wait_for(
                asyncio.to_thread(_fetch_pdf_text_sync, url, max_chars, max_pages),
                timeout=FETCH_TASK_TIMEOUT_SECONDS,
            )
            _vlog(
                logging.INFO,
                "fetch_pdf_text DONE url=%s chars=%d preview=%r",
                url, len(text or ""), (text or "")[:200].replace("\n", " "),
            )
            return text, error_type
        except asyncio.TimeoutError:
            logger.info("fetch_pdf_text timed out url=%s", url)
            return "", "timeout"


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
    matched: list[str] = []
    try:
        response = get_session().get(f"https://{domain}/sitemap.xml", timeout=PAGE_FETCH_TIMEOUT_SECONDS)
        response.raise_for_status()
        urls = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", response.text)[:1200]
        matched = [url for url in urls if CSR_LINK_PATTERN.search(url)]
        if matched:
            return matched[:limit]
        nested_sitemaps = [url for url in urls if url.endswith(".xml")][:6]
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
    return matched[:limit]


async def sitemap_csr_urls(domain: str, limit: int = 12) -> list[str]:
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
        title = result.get("title", "")
        body = result.get("body", "")
        if not host or any(domain in host for domain in AGGREGATOR_DOMAINS):
            continue
        if NEGATIVE_LINK_PATTERN.search(host):
            continue
        trustworthy = is_search_derived_domain_trustworthy(company, host, title, body)
        _vlog(
            logging.INFO,
            "discover_company_domains host=%s check=search_derived (blind-guess check never "
            "applied on this path) trustworthy=%s",
            host, trustworthy,
        )
        if trustworthy and host not in matched_domains:
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
            url = result.get("href", "")
            if any(domain in url for domain in UNTRUSTED_ENTITY_RESOLUTION_DOMAINS):
                _vlog(
                    logging.INFO,
                    "resolve_india_legal_entity_name SKIPPED_SOURCE company=%r url=%r "
                    "reason=untrusted_aggregator_domain",
                    company, url,
                )
                continue
            haystack = f"{result.get('title', '')} {result.get('body', '')}"
            match = INDIA_LEGAL_ENTITY_PATTERN.search(haystack)
            if match:
                candidate = re.sub(r"\s+", " ", match.group(1)).strip()
                plausible = is_plausible_legal_entity_name(company, candidate)
                _vlog(
                    logging.INFO,
                    "resolve_india_legal_entity_name CANDIDATE company=%r candidate=%r plausible=%s "
                    "source_url=%r haystack_preview=%r",
                    company, candidate, plausible, url, haystack[:250],
                )
                if plausible:
                    resolved_name = candidate
                    break
        if resolved_name:
            break

    if resolved_name:
        budget.mark_category_hit("legal_entity")
    else:
        _vlog(
            logging.INFO,
            "resolve_india_legal_entity_name UNRESOLVED company=%r — leaving legal_entity_name "
            "empty rather than accepting a weak/rejected candidate",
            company,
        )
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


def discover_chained_followup_targets(company: str, text: str) -> list[dict]:
    if not text:
        return []
    targets: list[dict] = []
    seen_keys: set[str] = set()

    for name in _extract_named_partner_candidates(company, text):
        key = f"partner:{name.lower()}"
        if key not in seen_keys:
            seen_keys.add(key)
            targets.append({"title": name, "kind": "partner"})

    for programme in extract_named_programme_candidates_with_titles(text):
        key = f"programme:{programme['title'].lower()}"
        if key not in seen_keys:
            seen_keys.add(key)
            targets.append(programme)

    for candidate in _extract_related_entity_candidates(company, text):
        key = f"entity:{candidate['entity_name'].lower()}"
        if key not in seen_keys:
            seen_keys.add(key)
            targets.append({"title": candidate["entity_name"], "kind": "entity"})

    return targets[:MAX_CHAINED_FOLLOWUP_TARGETS]


async def run_chained_followup_retrieval(company: str, budget: SearchBudget, quota_guard, deadline: float,
                                          category: str, source_text: str,
                                          resolved_domains: list[str] | None = None,
                                          registry: SourceRegistry | None = None,
                                          parent_source_name: str = "") -> list[dict]:
    targets = discover_chained_followup_targets(company, source_text)
    if not targets:
        return []

    recovered: list[dict] = []
    site_token = _domain_site_token(resolved_domains)

    for target in targets:
        if not await _within_deadline(deadline):
            break
        if not budget.google_has_budget(category):
            break
        title = target["title"]
        kind = target["kind"]

        query_candidates = [CHAINED_FOLLOWUP_TITLE_TEMPLATE.format(title=title, c=company)]
        if site_token:
            query_candidates.append(CHAINED_FOLLOWUP_DOMAIN_TEMPLATE.format(title=title, site=site_token))
        if kind == "partner" or kind == "entity":
            query_candidates.append(CHAINED_FOLLOWUP_NGO_TEMPLATE.format(title=title))
        query_candidates.append(CHAINED_FOLLOWUP_PRESS_TEMPLATE.format(c=company, title=title))

        queries_run_for_target = 0
        for query in query_candidates[:MAX_CHAINED_FOLLOWUP_QUERIES_PER_TARGET]:
            if not await _within_deadline(deadline) or not budget.google_has_budget(category):
                break
            queries_run_for_target += 1
            results = await search_web(query, budget, max_results=5, quota_guard=quota_guard, category=category)
            for result in results:
                url = result.get("href", "")
                title_snippet = result.get("title", "")
                body = result.get("body", "")
                if not url or any(domain in url for domain in AGGREGATOR_DOMAINS):
                    continue
                if not mentions_company(company, f"{title_snippet} {body}") and title.lower() not in f"{title_snippet} {body}".lower():
                    continue
                is_pdf = url.lower().endswith(".pdf")
                text = await (fetch_pdf_text(url) if is_pdf else fetch_page_text(url)) or body
                if not text or len(text) < SECOND_PASS_TEXT_LENGTH_FLOOR:
                    continue
                if not is_csr_relevant(text) and title.lower() not in text.lower():
                    continue
                new_source = make_source(
                    "second_pass_recovery", 11, url, text, "FOUND", f"chained_followup_{kind}",
                )
                if registry is not None:
                    registry.register_child_hit(
                        source_name=parent_source_name or "second_pass_recovery", url=url,
                        label=f"Chained follow-up — {title}", excerpt=text[:200],
                    )
                recovered.append(new_source)
                budget.mark_category_hit(category)
                break
            if recovered and recovered[-1].get("fetch_method", "").startswith("chained_followup"):
                break

    return recovered


def spawn_teaser_ngo_followup_queries(company: str, text: str) -> list[str]:
    names = _extract_named_partner_candidates(company, text)
    return [f'"{name}" "{company}" partnership India' for name in names]


def spawn_quantified_benefit_followup_query(company: str, text: str) -> str:
    if not text:
        return ""
    match = QUANTIFIED_BENEFICIARY_PATTERN.search(text)
    if not match:
        return ""
    number = match.group(1)
    return f'"{company}" "{number}" students OR beneficiaries CSR India'


async def _run_teaser_and_benefit_followups(company: str, text: str, budget: SearchBudget, quota_guard,
                                             deadline: float, registry: SourceRegistry | None,
                                             parent_source_name: str) -> None:
    teaser_queries = spawn_teaser_ngo_followup_queries(company, text)[:MAX_TEASER_NGO_FOLLOWUP_QUERIES]
    queries_used = 0

    for query in teaser_queries:
        if not await _within_deadline(deadline):
            return
        results = await search_web(query, budget, max_results=4, quota_guard=quota_guard, category="csr_page")
        queries_used += 1
        for result in results:
            url = result.get("href", "")
            title = result.get("title", "")
            body = result.get("body", "")
            if not url or any(domain in url for domain in AGGREGATOR_DOMAINS):
                continue
            if not mentions_company(company, f"{title} {body}"):
                continue
            fetched_text = await (
                fetch_pdf_text(url) if url.lower().endswith(".pdf") else fetch_page_text(url)
            ) or body
            if fetched_text and len(fetched_text) > 200 and is_csr_relevant(fetched_text) and mentions_company(company, fetched_text):
                if registry is not None:
                    registry.register_child_hit(
                        source_name=parent_source_name, url=url,
                        label="NGO teaser follow-up", excerpt=fetched_text[:200],
                    )
                break

    if queries_used >= MAX_TEASER_NGO_FOLLOWUP_QUERIES or not await _within_deadline(deadline):
        return

    benefit_query = spawn_quantified_benefit_followup_query(company, text)
    if not benefit_query:
        return
    results = await search_web(benefit_query, budget, max_results=4, quota_guard=quota_guard, category="csr_page")
    for result in results:
        url = result.get("href", "")
        title = result.get("title", "")
        body = result.get("body", "")
        if not url or any(domain in url for domain in AGGREGATOR_DOMAINS):
            continue
        if not mentions_company(company, f"{title} {body}"):
            continue
        fetched_text = await (
            fetch_pdf_text(url) if url.lower().endswith(".pdf") else fetch_page_text(url)
        ) or body
        if fetched_text and len(fetched_text) > 200 and is_csr_relevant(fetched_text) and mentions_company(company, fetched_text):
            if registry is not None:
                registry.register_child_hit(
                    source_name=parent_source_name, url=url,
                    label="Quantified beneficiary follow-up", excerpt=fetched_text[:200],
                )
            break


async def fetch_india_csr_page(company: str, search_cfg: dict, budget: SearchBudget, quota_guard=None,
                                max_fetches: int = 24, registry: SourceRegistry | None = None,
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
    known_company_domains: list[str] = []

    def candidate_clears_bar() -> bool:
        return bool(best_candidate[0] and best_candidate[0][0] >= MIN_ACCEPT_SCORE)

    def consider(url: str, method: str, text: str):
        text_preview = (text or "")[:200].replace("\n", " ")
        url_host = urlparse(url).netloc.lower() if url else ""
        domain_is_confirmed = bool(url_host) and any(
            url_host == d or url_host.endswith("." + d) for d in known_company_domains
        )
        if accept_fetched_text(company, text, 250, known_domains=known_company_domains, url=url):
            score = score_candidate_text(company, text, url)
            _vlog(
                logging.INFO,
                "india_csr_page CANDIDATE ACCEPTED company=%r url=%s method=%s score=%.1f "
                "text_len=%d domain_confirmed=%s preview=%r",
                company, url, method, score, len(text or ""), domain_is_confirmed, text_preview,
            )
            if best_candidate[0] is None or score > best_candidate[0][0]:
                source = make_source("india_csr_page", 1, url, text, "FOUND", method)
                source["domain"] = urlparse(url).netloc.lower()
                source["india_location_hits"] = find_india_location_mentions(text)[:10]
                best_candidate[0] = (score, source)
            return
        rejection_reason = (
            "no_text" if not text else
            "too_short" if len(text) <= 250 else
            "fails_csr_relevance_check" if not is_csr_relevant(text) else
            "company_not_mentioned"
        )
        _vlog(
            logging.INFO,
            "india_csr_page CANDIDATE REJECTED company=%r url=%s method=%s reason=%s "
            "text_len=%d domain_confirmed=%s preview=%r",
            company, url, method, rejection_reason, len(text or ""), domain_is_confirmed, text_preview,
        )
        if (
            text and len(text) > 80 and is_csr_relevant(text) and mentions_csr_context(text)
            and (domain_is_confirmed or mentions_company(company, text))
        ):
            score = score_candidate_text(company, text, url)
            _vlog(
                logging.INFO,
                "india_csr_page CANDIDATE WEAK_FALLBACK company=%r url=%s method=%s score=%.1f "
                "text_len=%d",
                company, url, method, score, len(text),
            )
            if weak_snippet_fallback[0] is None or score > weak_snippet_fallback[0][0]:
                source = make_source("india_csr_page", 1, url, text, "FOUND", method + "_snippet")
                source["domain"] = urlparse(url).netloc.lower()
                weak_snippet_fallback[0] = (score, source)

    domain_miss_streak: dict[str, int] = {}

    async def try_fetch(url: str, method: str, is_pdf: bool = False, is_guessed_path: bool = False):
        if not url or url in tried_urls or remaining_budget[0] <= 0 or not await _within_deadline(deadline):
            return
        host = urlparse(url).netloc.lower()
        if budget.is_domain_dead(host) or budget.is_path_dead(url):
            return
        if is_guessed_path and budget.guessed_path_miss_count.get(host, 0) >= MAX_GUESSED_PATH_ATTEMPTS_PER_DOMAIN:
            return
        tried_urls.add(url)
        remaining_budget[0] -= 1
        text, error_type = await (
            fetch_pdf_text_with_error(url) if is_pdf else fetch_page_text_with_error(url)
        )
        if not text or len(text) < UNREADABLE_TEXT_LENGTH_FLOOR:
            if url.lower().endswith(".pdf") or is_pdf:
                document_found_unreadable[0] = True
            budget.mark_path_dead(url)
            if is_guessed_path and error_type not in DOMAIN_KILLING_ERROR_TYPES and not error_type.startswith("http_4xx_403"):
                count = budget.record_guessed_path_miss(host)
                _vlog(
                    logging.INFO,
                    "india_csr_page guessed_path_miss host=%s count=%d error_type=%s "
                    "(does not count toward domain-dead escalation)",
                    host, count, error_type,
                )
            else:
                domain_miss_streak[host] = domain_miss_streak.get(host, 0) + 1
                if domain_miss_streak[host] >= DOMAIN_MISS_ESCALATION_THRESHOLD:
                    budget.mark_domain_dead(host, f"repeated_genuine_failures:{error_type}")
        else:
            domain_miss_streak[host] = 0
        consider(url, method, text)

    discovered_domains = await discover_company_domains(company, search_cfg, budget, quota_guard)
    domains = [d for d in dict.fromkeys(discovered_domains + candidate_domains(company)) if not budget.is_domain_dead(d)]
    known_company_domains = list(dict.fromkeys(discovered_domains + domains))

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
        if candidate_clears_bar():
            break

        sitemap_urls = await sitemap_csr_urls(domain)
        for sitemap_url in sitemap_urls:
            if candidate_clears_bar():
                break
            if not await _within_deadline(deadline):
                break
            await try_fetch(sitemap_url, "sitemap", is_pdf=sitemap_url.lower().endswith(".pdf"))
            candidates_checked += 1
        if candidate_clears_bar():
            break

        links = csr_links_from_html(f"https://{domain}", homepage_html)
        for link in links:
            if candidate_clears_bar():
                break
            if not await _within_deadline(deadline):
                break
            await try_fetch(link, "homepage_link", is_pdf=link.lower().endswith(".pdf"))
            candidates_checked += 1
        if candidate_clears_bar():
            break

        if not budget.is_domain_dead(domain):
            for path in CSR_PAGE_PATHS:
                if candidate_clears_bar():
                    break
                if budget.is_domain_dead(domain) or not await _within_deadline(deadline):
                    break
                await try_fetch(f"https://{domain}{path}", "direct", is_guessed_path=True)
                candidates_checked += 1
        if candidate_clears_bar():
            break

        if not budget.is_domain_dead(domain):
            for path in CSR_PAGE_PATHS_LOCALE_PREFIXED:
                if candidate_clears_bar():
                    break
                if budget.is_domain_dead(domain) or not await _within_deadline(deadline):
                    break
                await try_fetch(f"https://{domain}{path}", "locale_prefixed", is_guessed_path=True)
                candidates_checked += 1
        if candidate_clears_bar():
            break

    remaining_budget[0] = max(remaining_budget[0], 8)
    if not candidate_clears_bar() and await _within_deadline(deadline):
        site_token = _domain_site_token(discovered_domains)
        for query_template in CSR_PAGE_QUERIES:
            if not await _within_deadline(deadline):
                break
            if candidate_clears_bar():
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
                if candidate_clears_bar():
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
        if await _within_deadline(deadline):
            await _run_teaser_and_benefit_followups(
                company, result_source.get("text", ""), budget, quota_guard, deadline, registry, "india_csr_page",
            )
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
                mca_text, "FOUND", "direct", is_synthetic=False,
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
            synthetic_text, "FOUND", "cin_confirmed_portal_blocked", is_synthetic=True,
        )
        source["cin"] = cin
        if legal_name:
            source["legal_entity_name"] = legal_name
        if registry is not None:
            registry.register_core_source(source)
        logger.info(
            "mca_portal DONE company=%r found=True (cin_only, SYNTHETIC — does not count as a "
            "genuine fetch) cin=%s", company, cin,
        )
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

    ordered_templates: list[str] = []
    if site_token:
        ordered_templates.extend(ANNUAL_REPORT_QUERIES_YEAR_AGNOSTIC)
    ordered_templates.extend(ANNUAL_REPORT_QUERIES_FY_SPECIFIC)
    if not site_token:
        ordered_templates.extend(ANNUAL_REPORT_QUERIES_YEAR_AGNOSTIC)

    for template in ordered_templates:
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
        for fy in PRIOR_FY_LABELS:
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

    if (not best_candidate or count_financial_figures(best_candidate[1].get("text", "")) == 0) and await _within_deadline(deadline):
        for year in RECENT_CALENDAR_YEARS:
            if not await _within_deadline(deadline):
                break
            if best_candidate and best_candidate[0] >= MIN_ACCEPT_SCORE:
                break
            for template in ANNUAL_REPORT_QUERIES_CALENDAR_YEAR:
                query = template.format(c=company, year=year)
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
                            best_candidate = (score, make_source("annual_report", 4, url, text, "FOUND", "pdf_calendar_year"))
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
        "annual_report DONE company=%r urls_tried=%d found=%s pdf_found_but_unreadable=%s "
        "fiscal_year_lookback=%d calendar_year_lookback=%d",
        company, urls_tried, bool(chosen), pdf_found_but_unreadable,
        FISCAL_YEAR_LOOKBACK_COUNT, CALENDAR_YEAR_LOOKBACK_COUNT,
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

    fy1 = CURRENT_FY_LABEL
    fy2 = PRIOR_FY_LABELS[0] if PRIOR_FY_LABELS else fy1
    fy3 = PRIOR_FY_LABELS[2] if len(PRIOR_FY_LABELS) > 2 else fy2
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
    site_token = _domain_site_token(budget.resolved_domains)

    for target in search_targets:
        target_site_token = site_token if target == company else ""
        for template in PARTNER_QUERIES:
            if not await _within_deadline(deadline):
                break
            if len(candidates) >= max_partner_sources * 2:
                break
            query = template.format(c=target, site=target_site_token).strip()
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
            company, budget, quota_guard, deadline, "second_pass_unreadable_doc", seed_names=candidate_names, min_len=150,
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
    site_token = _domain_site_token(budget.resolved_domains)

    for target in search_targets:
        target_site_token = site_token if target == company else ""
        for template in EDUCATION_PROGRAMME_QUERIES:
            if not await _within_deadline(deadline):
                break
            if len(candidates) >= max_programme_sources * 2:
                break
            query = template.format(c=target, site=target_site_token).strip()
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
            company, budget, quota_guard, deadline, "second_pass_unreadable_doc",
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
                              job_deadline: float | None = None, cheap_mode: bool = False) -> dict:
    await _check_job_deadline(job_deadline)
    deadline = time.monotonic() + (SOURCE_DEADLINE_SECONDS if not cheap_mode else FOLLOWUP_DEADLINE_SECONDS)
    if job_deadline is not None:
        deadline = min(deadline, job_deadline)
    site_token = _domain_site_token(budget.resolved_domains)
    hits, fetched_texts, first_url = [], [], ""
    templates = PLAN_QUERIES[:1] if cheap_mode else PLAN_QUERIES
    max_pages = 1 if cheap_mode else max_pages
    for query_template in templates:
        if not await _within_deadline(deadline):
            break
        if len(fetched_texts) >= max_pages:
            break
        query = query_template.format(c=company, site=site_token).strip()
        results = await search_web(
            query, budget, max_results=5, quota_guard=quota_guard, category="plans_search",
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
                                           job_deadline: float | None = None, cheap_mode: bool = False) -> dict:
    await _check_job_deadline(job_deadline)
    deadline = time.monotonic() + (SOURCE_DEADLINE_SECONDS if not cheap_mode else FOLLOWUP_DEADLINE_SECONDS)
    if job_deadline is not None:
        deadline = min(deadline, job_deadline)
    site_token = _domain_site_token(budget.resolved_domains)
    hits, fetched_texts, first_url = [], [], ""
    max_fetch = 1 if cheap_mode else 3
    for query_template in SECTOR_QUERIES:
        if not await _within_deadline(deadline):
            break
        if len(fetched_texts) >= max_fetch:
            break
        query = query_template.format(c=company, site=site_token).strip()
        results = await search_web(
            query, budget, max_results=5,
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
            if len(fetched_texts) < max_fetch:
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

async def fetch_screen_sources(company: str, search_cfg: dict, registry: SourceRegistry | None = None) -> list[dict]:
    budget = SearchBudget(company, mode="screen")
    job_deadline = time.monotonic() + DEEP_JOB_HARD_DEADLINE_SECONDS

    sources: list[dict] = []

    try:
        india_csr_page_source = await fetch_india_csr_page(
            company, search_cfg, budget, registry=registry, job_deadline=job_deadline,
        )
        sources.append(india_csr_page_source)

        related_entities = await discover_related_entities(
            company, search_cfg, budget, deadline=job_deadline,
        )

        mca_portal_source = await fetch_mca_portal(
            company, search_cfg, budget, registry=registry, job_deadline=job_deadline,
        )
        sources.append(mca_portal_source)

        annual_report_source = await fetch_annual_report(
            company, search_cfg, budget, registry=registry, job_deadline=job_deadline,
        )
        sources.append(annual_report_source)

        people_source = await fetch_linkedin_people(
            company, search_cfg, budget, registry=registry, job_deadline=job_deadline,
        )
        sources.append(people_source)

        partner_source = await fetch_partner_source(
            company, search_cfg, budget, registry=registry, job_deadline=job_deadline,
            related_entities=related_entities, mode="screen",
        )
        sources.append(partner_source)

        education_programme_source = await fetch_education_programme_source(
            company, search_cfg, budget, registry=registry, job_deadline=job_deadline,
            related_entities=related_entities, mode="screen",
        )
        sources.append(education_programme_source)

    except DeepJobDeadlineExceeded:
        logger.warning("fetch_screen_sources hit hard deadline company=%r sources_so_far=%d", company, len(sources))

    logger.info(
        "fetch_screen_sources DONE company=%r sources=%d found=%d",
        company, len(sources), sum(1 for s in sources if s.get("status") == "FOUND"),
    )
    return sources


async def fetch_deep_sources(company: str, search_cfg: dict, registry: SourceRegistry | None = None) -> list[dict]:
    budget = SearchBudget(company, mode="deep")
    job_deadline = time.monotonic() + DEEP_JOB_HARD_DEADLINE_SECONDS

    sources: list[dict] = []

    try:
        india_csr_page_source = await fetch_india_csr_page(
            company, search_cfg, budget, registry=registry, job_deadline=job_deadline,
        )
        sources.append(india_csr_page_source)

        related_entities = await discover_related_entities(
            company, search_cfg, budget, deadline=job_deadline,
        )

        mca_portal_source = await fetch_mca_portal(
            company, search_cfg, budget, registry=registry, job_deadline=job_deadline,
        )
        sources.append(mca_portal_source)

        national_csr_portal_source = await fetch_national_csr_portal(
            company, search_cfg, budget, registry=registry, job_deadline=job_deadline,
        )
        sources.append(national_csr_portal_source)

        annual_report_source = await fetch_annual_report(
            company, search_cfg, budget, registry=registry, job_deadline=job_deadline,
        )
        sources.append(annual_report_source)

        multi_year_financials_source = await fetch_multi_year_financials(
            company, search_cfg, budget, registry=registry, job_deadline=job_deadline,
            annual_report_source=annual_report_source,
        )
        sources.append(multi_year_financials_source)

        people_source = await fetch_linkedin_people(
            company, search_cfg, budget, registry=registry, job_deadline=job_deadline,
        )
        sources.append(people_source)

        partner_source = await fetch_partner_source(
            company, search_cfg, budget, registry=registry, job_deadline=job_deadline,
            related_entities=related_entities, mode="deep",
        )
        sources.append(partner_source)

        education_programme_source = await fetch_education_programme_source(
            company, search_cfg, budget, registry=registry, job_deadline=job_deadline,
            related_entities=related_entities, mode="deep",
        )
        sources.append(education_programme_source)

        plans_source = await fetch_plans_source(
            company, search_cfg, budget, registry=registry, job_deadline=job_deadline,
        )
        sources.append(plans_source)

        sector_eligibility_source = await fetch_sector_eligibility_source(
            company, search_cfg, budget, registry=registry, job_deadline=job_deadline,
        )
        sources.append(sector_eligibility_source)

    except DeepJobDeadlineExceeded:
        logger.warning("fetch_deep_sources hit hard deadline company=%r sources_so_far=%d", company, len(sources))

    logger.info(
        "fetch_deep_sources DONE company=%r sources=%d found=%d",
        company, len(sources), sum(1 for s in sources if s.get("status") == "FOUND"),
    )
    return sources