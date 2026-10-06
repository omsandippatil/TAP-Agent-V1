import hashlib
import logging
import re

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

BLOCK_LEVEL_TEXT_TAGS = [
    "p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "td", "th", "blockquote",
    "dd", "dt", "figcaption", "summary", "caption", "pre",
]

GENERIC_CONTAINER_TAGS = ["div", "span", "section", "article"]

MIN_CONTAINER_TEXT_LENGTH = 20
MAX_CONTAINER_TEXT_LENGTH = 2000

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
                 status: str = "NOT_FOUND", fetch_method: str = "search",
                 is_synthetic: bool = False) -> dict:
    return {
        "source_name": source_name,
        "priority": priority,
        "url": url,
        "text": text,
        "status": status,
        "fetch_method": fetch_method,
        "is_synthetic": is_synthetic,
    }


def clean_text(raw_text: str, max_chars: int = 15000) -> str:
    collapsed = re.sub(r"\s+", " ", raw_text).strip()
    return collapsed[:max_chars]


def normalize_block_text(raw_text: str, max_chars: int = 15000) -> str:
    if not raw_text:
        return ""
    lines = []
    for raw_line in raw_text.splitlines():
        collapsed = re.sub(r"[ \t\u00a0]+", " ", raw_line).strip()
        if collapsed:
            lines.append(collapsed)
    joined = "\n".join(lines)
    return joined[:max_chars]


def _is_boilerplate_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 2:
        return True
    if any(pattern.match(stripped) for pattern in BOILERPLATE_LINE_PATTERNS):
        return True
    if any(pattern.search(stripped) for pattern in BOILERPLATE_CONTAINS_PATTERNS):
        return True
    return False


def _direct_text_of_container(tag) -> str:
    direct_pieces = []
    for child in tag.children:
        if getattr(child, "name", None) in GENERIC_CONTAINER_TAGS:
            continue
        if hasattr(child, "get_text"):
            direct_pieces.append(child.get_text(" ", strip=True))
        else:
            piece = str(child).strip()
            if piece:
                direct_pieces.append(piece)
    return " ".join(p for p in direct_pieces if p)


def _extract_generic_container_text(root, seen_lines: set) -> list[str]:
    lines = []
    for tag_name in GENERIC_CONTAINER_TAGS:
        for tag in root.find_all(tag_name):
            if tag.find(BLOCK_LEVEL_TEXT_TAGS):
                continue
            text = _direct_text_of_container(tag)
            if not text or len(text) < MIN_CONTAINER_TEXT_LENGTH:
                continue
            if len(text) > MAX_CONTAINER_TEXT_LENGTH:
                continue
            if _is_boilerplate_line(text):
                continue
            key = text.lower()[:120]
            if key in seen_lines:
                continue
            seen_lines.add(key)
            lines.append(text)
    return lines


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
    for element in root.find_all(BLOCK_LEVEL_TEXT_TAGS):
        text = element.get_text(" ", strip=True)
        if not text or _is_boilerplate_line(text):
            continue
        key = text.lower()[:120]
        if key in seen_lines:
            continue
        seen_lines.add(key)
        lines.append(text)

    lines.extend(_extract_generic_container_text(root, seen_lines))

    if not lines:
        return normalize_block_text(root.get_text("\n", strip=True), max_chars)

    combined = "\n".join(lines)
    return normalize_block_text(combined, max_chars)


def combine_source_texts(sources: list) -> str:
    return "\n\n".join(
        source["text"] for source in sources
        if source.get("status") == "FOUND" and source.get("text")
    )


def build_sources_manifest(sources: list) -> str:
    lines = []
    for source in sources:
        if source.get("status") == "NOT_TRIED":
            continue
        number = source.get("source_number")
        prefix = f"[{number}] " if number else ""
        synthetic_tag = " | SYNTHETIC (not a direct fetch)" if source.get("is_synthetic") else ""
        lines.append(
            f"{prefix}{source.get('source_name', '')} | {source.get('status', '')}{synthetic_tag} | {source.get('url', '')}"
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


def evidence_hash(sources: list) -> str:
    combined = combine_source_texts(sources)
    digest_input = combined.encode("utf-8", errors="ignore")
    return hashlib.sha256(digest_input).hexdigest()


def mission_hash(mission: str) -> str:
    digest_input = (mission or "").strip().encode("utf-8", errors="ignore")
    return hashlib.sha256(digest_input).hexdigest()