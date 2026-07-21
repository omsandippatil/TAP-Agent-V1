import re
from urllib.parse import urlparse

LINKEDIN_TITLE_SEPARATOR_PATTERN = re.compile(r"\s*[\|\u2014\u2013\-]\s*")

LINKEDIN_LOCALE_SUBDOMAIN_PATTERN = re.compile(r"^([a-z]{2,3})\.linkedin\.com$", re.IGNORECASE)

LINKEDIN_INDIA_SUBDOMAIN_TOKENS = frozenset(["in"])
LINKEDIN_NON_INDIA_SUBDOMAIN_TOKENS = frozenset([
    "au", "uk", "us", "ca", "sg", "de", "fr", "nl", "ae", "hk", "cn", "jp",
    "za", "nz", "ie", "es", "it", "ch", "se", "no", "dk", "br", "mx", "my", "id", "ph",
])

LINKEDIN_BOILERPLATE_SUFFIX_PATTERN = re.compile(
    r"\s*-\s*linkedin\s*$|\s*\|\s*linkedin\s*$", re.IGNORECASE
)

LINKEDIN_NAME_PREFIX_PATTERN = re.compile(
    r"^([A-Z][A-Za-z\.\'\u2019\-]+(?:\s+[A-Z][A-Za-z\.\'\u2019\-]+){0,4})"
)

CSR_ROLE_KEYWORD_PATTERN = re.compile(
    r"(chief\s+csr\s+officer|head[\s,]*(?:of\s+)?csr|csr\s+head|csr\s+director|"
    r"chief\s+sustainability\s+officer|sustainability\s+head|head\s+of\s+sustainability|"
    r"vp[\s,\-]*csr|csr\s+manager|csr\s+lead|csr\s+specialist|csr\s+executive|"
    r"foundation\s+(?:ceo|director|head|manager)|"
    r"esg\s+head|head\s+of\s+esg|esg\s+manager|esg\s+lead|"
    r"social\s+impact\s+(?:head|lead|director|manager)|"
    r"community\s+(?:engagement|relations|development)\s+(?:head|lead|manager|director)|"
    r"corporate\s+social\s+responsibility|"
    r"inclusion\s+(?:head|lead|manager|director)|"
    r"diversity\s*(?:,|&|and)?\s*inclusion|"
    r"philanthropy\s+(?:head|lead|manager|director))",
    re.IGNORECASE,
)

FORMER_ROLE_KEYWORD_PATTERN = re.compile(
    r"\b(former|ex[\s\-]|previously|until\s+\d{4}|retired|alumnus|alumni|"
    r"past\s+(?:employee|role)|no\s+longer)\b",
    re.IGNORECASE,
)

COMPANY_AT_PATTERN = re.compile(r"\bat\s+([A-Z][\w&.,\'\-]+(?:\s+[A-Z][\w&.,\'\-]+){0,5})", re.IGNORECASE)

INDIA_LOCATION_TOKENS = frozenset([
    "india", "bharat", "delhi", "new delhi", "mumbai", "bombay", "bengaluru", "bangalore",
    "chennai", "madras", "kolkata", "calcutta", "hyderabad", "pune", "ahmedabad", "surat",
    "jaipur", "lucknow", "kanpur", "nagpur", "indore", "thane", "bhopal", "visakhapatnam",
    "patna", "vadodara", "ghaziabad", "ludhiana", "agra", "nashik", "faridabad", "meerut",
    "rajkot", "kalyan", "vasai", "varanasi", "srinagar", "aurangabad", "dhanbad", "amritsar",
    "navi mumbai", "allahabad", "prayagraj", "ranchi", "howrah", "coimbatore", "jabalpur",
    "gwalior", "vijayawada", "jodhpur", "madurai", "raipur", "kota", "guwahati", "chandigarh",
    "solapur", "hubli", "mysore", "mysuru", "tiruchirappalli", "bareilly", "aligarh", "gurgaon",
    "gurugram", "noida", "moradabad", "jalandhar", "bhubaneswar", "salem", "warangal",
    "maharashtra", "karnataka", "tamil nadu", "gujarat", "rajasthan", "uttar pradesh",
    "west bengal", "telangana", "kerala", "punjab", "haryana", "bihar", "odisha", "assam",
    "goa", "jharkhand", "chhattisgarh", "uttarakhand", "himachal pradesh", "andhra pradesh",
])

NON_INDIA_LOCATION_TOKENS = frozenset([
    "united states", "usa", "u.s.", "uk", "united kingdom", "australia", "canada",
    "singapore", "germany", "france", "netherlands", "uae", "dubai", "abu dhabi",
    "hong kong", "china", "japan", "south africa", "new zealand", "ireland", "spain",
    "italy", "switzerland", "sweden", "norway", "denmark", "brazil", "mexico",
])


def strip_linkedin_suffix(raw_title: str) -> str:
    return LINKEDIN_BOILERPLATE_SUFFIX_PATTERN.sub("", raw_title or "").strip()


def split_linkedin_title(raw_title: str) -> list[str]:
    cleaned = strip_linkedin_suffix(raw_title)
    parts = [p.strip() for p in LINKEDIN_TITLE_SEPARATOR_PATTERN.split(cleaned) if p.strip()]
    return parts


def extract_person_name(raw_title: str, parts: list[str] | None = None) -> str:
    parts = parts if parts is not None else split_linkedin_title(raw_title)
    if parts:
        candidate = parts[0]
        match = LINKEDIN_NAME_PREFIX_PATTERN.match(candidate)
        if match:
            return match.group(1).strip()
        if len(candidate.split()) <= 5 and not CSR_ROLE_KEYWORD_PATTERN.search(candidate):
            return candidate.strip()
    fallback_match = LINKEDIN_NAME_PREFIX_PATTERN.match(strip_linkedin_suffix(raw_title))
    return fallback_match.group(1).strip() if fallback_match else ""


def extract_job_title(raw_title: str, snippet: str, parts: list[str] | None = None) -> str:
    parts = parts if parts is not None else split_linkedin_title(raw_title)
    role_segments = [p for p in parts[1:] if p and not COMPANY_AT_PATTERN.fullmatch(p)]
    if role_segments:
        return role_segments[0].strip(" .")
    match = CSR_ROLE_KEYWORD_PATTERN.search(f"{raw_title} {snippet}")
    return match.group(0).strip() if match else ""


def extract_company_affiliation(raw_title: str, snippet: str, parts: list[str] | None = None,
                                 company: str = "") -> str:
    parts = parts if parts is not None else split_linkedin_title(raw_title)

    if company:
        tokens = [t for t in re.sub(r"[^a-z0-9 ]", " ", company.lower()).split() if len(t) > 2]
        for part in parts[1:]:
            lowered_part = part.lower()
            if tokens and any(token in lowered_part for token in tokens):
                return part.strip(" .")

    for part in parts[1:]:
        match = COMPANY_AT_PATTERN.search(part)
        if match:
            return match.group(1).strip(" .")

    haystack = f"{raw_title} {snippet}"
    match = COMPANY_AT_PATTERN.search(haystack)
    if match:
        return match.group(1).strip(" .")

    non_role_segments = [p for p in parts[1:] if not CSR_ROLE_KEYWORD_PATTERN.search(p)]
    if non_role_segments:
        return non_role_segments[-1].strip(" .")
    return ""


def linkedin_url_locale(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    match = LINKEDIN_LOCALE_SUBDOMAIN_PATTERN.match(host)
    return match.group(1) if match else ""


def location_mentions_india(raw_title: str, snippet: str, url: str = "") -> bool:
    locale = linkedin_url_locale(url)
    if locale in LINKEDIN_INDIA_SUBDOMAIN_TOKENS:
        return True
    if locale in LINKEDIN_NON_INDIA_SUBDOMAIN_TOKENS:
        haystack = f"{raw_title} {snippet}".lower()
        return any(token in haystack for token in INDIA_LOCATION_TOKENS)

    haystack = f"{raw_title} {snippet}".lower()
    if any(token in haystack for token in NON_INDIA_LOCATION_TOKENS):
        if not any(india_token in haystack for india_token in INDIA_LOCATION_TOKENS):
            return False
    return any(token in haystack for token in INDIA_LOCATION_TOKENS)


def is_current_csr_role(raw_title: str, snippet: str) -> bool:
    haystack = f"{raw_title} {snippet}"
    if not CSR_ROLE_KEYWORD_PATTERN.search(haystack):
        return False
    return not FORMER_ROLE_KEYWORD_PATTERN.search(haystack)


def parse_linkedin_hit(raw_title: str, snippet: str, url: str, company: str) -> dict:
    parts = split_linkedin_title(raw_title)
    name = extract_person_name(raw_title, parts)
    job_title = extract_job_title(raw_title, snippet, parts)
    affiliation = extract_company_affiliation(raw_title, snippet, parts, company=company)
    india_signal = location_mentions_india(raw_title, snippet, url)
    current_role = is_current_csr_role(raw_title, snippet)
    has_csr_signal = bool(CSR_ROLE_KEYWORD_PATTERN.search(f"{raw_title} {snippet}"))

    if current_role and india_signal:
        confidence = "HIGH"
    elif current_role or (has_csr_signal and india_signal):
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "name": name,
        "title": job_title,
        "company_affiliation": affiliation,
        "url": url,
        "raw_title": strip_linkedin_suffix(raw_title),
        "snippet": (snippet or "").strip(),
        "india_location_signal": india_signal,
        "is_current_csr_role": current_role,
        "has_csr_signal": has_csr_signal,
        "confidence": confidence,
    }
