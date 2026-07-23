import hashlib
import json
import logging
import re
import time

import certifi
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("tap.utils")

BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"^(home|about us?|contact us?|careers?|sign in|log ?in|sign up|register)$", re.IGNORECASE),
    re.compile(r"^(privacy policy|terms( of (use|service))?|cookie policy|disclaimer|sitemap)$", re.IGNORECASE),
    re.compile(r"^(all rights reserved|copyright ©|©\s*\d{4})", re.IGNORECASE),
    re.compile(r"^(share|tweet|follow us|subscribe|read more|load more|back to top)$", re.IGNORECASE),
    re.compile(r"^\d+$"),
]

BOILERPLATE_CONTAINS_PATTERNS = [
    re.compile(r"javascript is disabled", re.IGNORECASE),
    re.compile(r"enable cookies", re.IGNORECASE),
    re.compile(r"click here to", re.IGNORECASE),
]

STRIP_TAGS = [
    "script", "style", "nav", "footer", "header", "aside", "noscript", "svg",
    "form", "button", "iframe", "img", "picture", "video", "audio",
    "input", "select", "textarea", "meta", "link",
]

MAIN_CONTENT_SELECTORS = ["main", "article", "[role=main]", "#content", ".content", ".main-content"]

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

RETRY_STATUS_FORCELIST = (403, 429, 500, 502, 503, 504)
RETRY_TOTAL = 2
RETRY_BACKOFF_FACTOR = 1.2


def build_http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Upgrade-Insecure-Requests": "1",
        "Connection": "keep-alive",
        "DNT": "1",
    })
    session.verify = certifi.where()

    retry = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=RETRY_STATUS_FORCELIST,
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_SESSION_SINGLETON: requests.Session | None = None


def get_session() -> requests.Session:
    global _SESSION_SINGLETON
    if _SESSION_SINGLETON is None:
        _SESSION_SINGLETON = build_http_session()
    return _SESSION_SINGLETON


def get_with_referer_fallback(url: str, timeout: float, **kwargs) -> requests.Response:
    """GET a URL, and if it 403s with no referer, retry once with a same-origin referer.

    Some WAFs specifically block direct-to-resource requests (especially PDFs) that
    arrive with no referer chain, but allow the same request if it looks like it came
    from a click on the site's own homepage.
    """
    session = get_session()
    response = session.get(url, timeout=timeout, **kwargs)
    if response.status_code == 403:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            homepage = f"{parsed.scheme}://{parsed.netloc}/"
            if homepage != url:
                retry_headers = dict(kwargs.pop("headers", {}) or {})
                retry_headers["Referer"] = homepage
                logger.info("retrying with referer after 403 url=%s referer=%s", url, homepage)
                response = session.get(url, timeout=timeout, headers=retry_headers, **kwargs)
        except Exception as exc:
            logger.info("referer retry failed url=%s error=%s", url, exc)
    return response


def classify_fetch_error(exc: Exception) -> str:
    """Classify a fetch exception into a coarse error_type for structured logging."""
    text = str(exc)
    exc_type = type(exc).__name__
    if "NameResolutionError" in text or "NameResolutionError" in exc_type or "getaddrinfo" in text:
        return "dns"
    if isinstance(exc, requests.exceptions.SSLError) or "SSLError" in exc_type:
        return "ssl"
    if isinstance(exc, requests.exceptions.Timeout) or "Timeout" in exc_type:
        return "timeout"
    if isinstance(exc, requests.exceptions.HTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status is not None:
            if 400 <= status < 500:
                return f"http_4xx_{status}"
            if 500 <= status < 600:
                return f"http_5xx_{status}"
        return "http_error"
    match = re.search(r"\b(4\d{2}|5\d{2})\b", text)
    if match:
        code = int(match.group(1))
        return f"http_4xx_{code}" if code < 500 else f"http_5xx_{code}"
    if "ConnectionError" in exc_type:
        return "connection_error"
    return "other"


def domain_resolves(domain: str, timeout: float = 1.5) -> bool:
    """Fast DNS-only check so we don't burn fetch slots/time on domains that don't exist."""
    import socket
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(domain)
        return True
    except Exception:
        return False
    finally:
        socket.setdefaulttimeout(None)


def make_source(source_name: str, priority: int, url: str = "", text: str = "",
                 status: str = "NOT_FOUND", fetch_method: str = "search") -> dict:
    return {
        "source_name": source_name,
        "priority": priority,
        "url": url,
        "text": text,
        "status": status,
        "fetch_method": fetch_method,
    }


def clean_text(raw_text: str, max_chars: int = 15000) -> str:
    collapsed = re.sub(r"\s+", " ", raw_text).strip()
    return collapsed[:max_chars]


def _is_boilerplate_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 2:
        return True
    if any(pattern.match(stripped) for pattern in BOILERPLATE_LINE_PATTERNS):
        return True
    if any(pattern.search(stripped) for pattern in BOILERPLATE_CONTAINS_PATTERNS):
        return True
    return False


def extract_main_text(soup, max_chars: int = 16000) -> str:
    for tag_name in STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    root = None
    for selector in MAIN_CONTENT_SELECTORS:
        found = soup.select_one(selector)
        if found and len(found.get_text(strip=True)) > 400:
            root = found
            break
    if root is None:
        root = soup.body or soup

    lines = []
    seen_lines = set()
    for element in root.find_all(["p", "li", "h1", "h2", "h3", "h4", "td", "blockquote"]):
        text = element.get_text(" ", strip=True)
        if not text or _is_boilerplate_line(text):
            continue
        key = text.lower()[:120]
        if key in seen_lines:
            continue
        seen_lines.add(key)
        lines.append(text)

    if not lines:
        return clean_text(root.get_text(" ", strip=True), max_chars)

    combined = "\n".join(lines)
    return clean_text(combined, max_chars)


def extract_table_rows(soup, max_rows: int = 200) -> list[list[str]]:
    rows = []
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if cells:
                rows.append(cells)
            if len(rows) >= max_rows:
                return rows
    return rows


def combine_source_texts(sources: list) -> str:
    return "\n\n".join(
        source["text"] for source in sources
        if source.get("status") == "FOUND" and source.get("text")
    )


CSR_SIGNAL_KEYWORDS = [
    "csr", "corporate social", "philanthrop", "social responsibility",
    "schedule vii", "csr spend", "csr expenditure", "csr budget",
    "csr obligation", "csr fund", "community investment", "sustainability",
    "foundation", "ngo", "education", "skill", "crore", "lakh",
    "partnered", "partnership", "initiative", "programme", "program",
]

SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
COMPANY_STOPWORDS = {
    "india", "limited", "ltd", "private", "pvt", "the", "and", "of",
    "company", "corp", "corporation", "inc", "group", "technologies",
    "solutions", "services", "international",
}


def company_tokens(company: str) -> list[str]:
    return [
        token for token in re.sub(r"[^a-z0-9 ]", " ", company.lower()).split()
        if len(token) > 2 and token not in COMPANY_STOPWORDS
    ]


def sentence_mentions_company(sentence_lower: str, tokens: list[str]) -> bool:
    if not tokens:
        return True
    return any(token in sentence_lower for token in tokens)


def sentence_csr_signal_count(sentence_lower: str) -> int:
    return sum(1 for kw in CSR_SIGNAL_KEYWORDS if kw in sentence_lower)


def score_sentence(sentence: str, tokens: list[str]) -> int:
    lowered = sentence.lower()
    if len(sentence) < 25:
        return -1
    score = sentence_csr_signal_count(lowered) * 2
    if sentence_mentions_company(lowered, tokens):
        score += 3
    return score


def relevant_excerpt_from_text(text: str, company: str, max_chars: int) -> str:
    tokens = company_tokens(company)
    sentences = SPLIT_PATTERN.split(text)
    scored = [(score_sentence(s, tokens), s.strip()) for s in sentences]
    scored = [(score, s) for score, s in scored if score >= 0 and s]

    if not scored:
        return clean_text(text, max_chars)

    scored.sort(key=lambda pair: pair[0], reverse=True)

    kept, used_chars = [], 0
    for score, sentence in scored:
        if score <= 0 and used_chars > 0:
            break
        addition = len(sentence) + 1
        if used_chars + addition > max_chars:
            break
        kept.append(sentence)
        used_chars += addition

    if not kept:
        return clean_text(text, max_chars)
    return clean_text(" ".join(kept), max_chars)


def trim_source_for_relevance(source: dict, company: str, per_source_budget: int) -> str:
    text = source.get("text", "")
    if not text:
        return ""
    if len(text) <= per_source_budget:
        return text
    return relevant_excerpt_from_text(text, company, per_source_budget)


def build_relevant_evidence_text(sources: list, company: str, total_budget: int = 9000) -> str:
    found_sources = [s for s in sources if s.get("status") == "FOUND" and s.get("text")]
    if not found_sources:
        return ""

    per_source_budget = max(600, total_budget // max(len(found_sources), 1))
    chunks = []
    for source in found_sources:
        trimmed = trim_source_for_relevance(source, company, per_source_budget)
        if trimmed:
            label = source.get("source_name", "source")
            chunks.append(f"[{label}]\n{trimmed}")

    combined = "\n\n".join(chunks)
    return combined[:total_budget]


def build_sources_manifest(sources: list) -> str:
    lines = []
    for source in sources:
        if source.get("status") == "NOT_TRIED":
            continue
        number = source.get("source_number")
        prefix = f"[{number}] " if number else ""
        lines.append(
            f"{prefix}{source.get('source_name', '')} | {source.get('status', '')} | {source.get('url', '')}"
        )
    return "\n".join(lines)


def merge_manifest_with_registry(sources_manifest: str, registry) -> str:
    manifest_lines = registry.as_manifest_lines()
    if not manifest_lines:
        return sources_manifest
    registry_block = (
        "NUMBERED SOURCE INDEX — cite facts using the bracketed number exactly as shown, "
        "e.g. [3], and never invent a number that is not listed here:\n"
        + "\n".join(manifest_lines)
    )
    if not sources_manifest:
        return registry_block
    return sources_manifest + "\n\n" + registry_block


def to_json(value) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def evidence_hash(sources: list) -> str:
    combined = combine_source_texts(sources)
    digest_input = combined.encode("utf-8", errors="ignore")
    return hashlib.sha256(digest_input).hexdigest()


def mission_hash(mission: str) -> str:
    digest_input = (mission or "").strip().encode("utf-8", errors="ignore")
    return hashlib.sha256(digest_input).hexdigest()