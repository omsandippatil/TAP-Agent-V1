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

LINKEDIN_NAME_WITH_CREDENTIALS_PATTERN = re.compile(
    r"^([A-Z][A-Za-z\.\'\u2019\-]+(?:\s+[A-Z][A-Za-z\.\'\u2019\-]+){0,4})"
    r"(?:\s*,?\s*(?:PhD|Ph\.D\.?|MBA|CFA|CPA|PMP|MSc|MS|MA|BSc))*",
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

SENIORITY_KEYWORD_PATTERN_ORDER = [
    ("C_SUITE", re.compile(r"\b(chief\s+\w+\s+officer|c[a-z]o)\b", re.IGNORECASE)),
    ("VP", re.compile(r"\b(vice\s+president|vp)\b", re.IGNORECASE)),
    ("DIRECTOR", re.compile(r"\bdirector\b", re.IGNORECASE)),
    ("HEAD", re.compile(r"\bhead\b", re.IGNORECASE)),
    ("MANAGER", re.compile(r"\bmanager\b", re.IGNORECASE)),
    ("LEAD", re.compile(r"\blead\b", re.IGNORECASE)),
    ("SPECIALIST", re.compile(r"\b(specialist|executive|associate|officer)\b", re.IGNORECASE)),
    ("ANALYST", re.compile(r"\banalyst\b", re.IGNORECASE)),
    ("INTERN", re.compile(r"\bintern(ship)?\b", re.IGNORECASE)),
]

DEPARTMENT_KEYWORD_PATTERN = re.compile(
    r"\b(csr|corporate\s+social\s+responsibility|sustainability|esg|"
    r"social\s+impact|community\s+(?:engagement|relations|development)|"
    r"philanthropy|foundation|diversity\s*(?:,|&|and)?\s*inclusion|inclusion)\b",
    re.IGNORECASE,
)

FORMER_ROLE_KEYWORD_PATTERN = re.compile(
    r"\b(former|ex[\s\-]|previously|until\s+\d{4}|retired|alumnus|alumni|"
    r"past\s+(?:employee|role)|no\s+longer)\b",
    re.IGNORECASE,
)

NON_CSR_TITLE_KEYWORD_PATTERN = re.compile(
    r"\b(software\s+(?:engineer|developer)|sales\s+(?:manager|executive|director|"
    r"representative|associate|lead)|regional\s+(?:operations|sales)|business\s+"
    r"development|product\s+(?:manager|engineer)|account\s+manager|"
    r"marketing\s+(?:manager|executive)(?!\s*,?\s*csr)|(?<!csr\s)finance\s+"
    r"(?:manager|executive|analyst)|procurement|supply\s+chain|it\s+support|"
    r"software\s+testing|human\s+resources?\s+(?:generalist|executive|associate)|"
    r"talent\s+acquisition|recruiter|plant\s+(?:manager|operations)|factory\s+"
    r"operations)\b",
    re.IGNORECASE,
)

COMPANY_AT_PATTERN = re.compile(r"\bat\s+([A-Z][\w&.\'\-]+(?:\s+[A-Z][\w&.\'\-]+){0,5})")

MULTI_COMPANY_AT_PATTERN = re.compile(r"\bat\s+([A-Z][\w&.\'\-]+(?:\s+[A-Z][\w&.\'\-]+){0,5})")

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

PHONE_PATTERN = re.compile(
    r"(?:\+91[\s\-]?)?(?:\(?\d{3,5}\)?[\s\-]?)?\d{5}[\s\-]?\d{5}\b|"
    r"\+\d{1,3}[\s\-]?\d{3,4}[\s\-]?\d{3,4}[\s\-]?\d{3,4}\b"
)

EDUCATION_HINT_PATTERN = re.compile(
    r"\b(IIM|IIT|BITS|XLRI|MBA|Masters?\s+in|B\.?Tech|M\.?Tech|B\.?A\.?|M\.?A\.?|"
    r"University\s+of\s+\w+|College)\b"
)

YEARS_EXPERIENCE_PATTERN = re.compile(r"\b(\d{1,2})\+?\s*years?\s+(?:of\s+)?experience\b", re.IGNORECASE)

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

CITY_LOOKUP_PATTERN = re.compile(
    r"\b(mumbai|bombay|new\s*delhi|delhi|bengaluru|bangalore|chennai|madras|kolkata|"
    r"calcutta|hyderabad|pune|ahmedabad|surat|jaipur|lucknow|kanpur|nagpur|indore|"
    r"thane|bhopal|visakhapatnam|patna|vadodara|ghaziabad|ludhiana|agra|nashik|"
    r"faridabad|meerut|rajkot|gurgaon|gurugram|noida|chandigarh|coimbatore)\b",
    re.IGNORECASE,
)

COMPANY_STOPWORDS = frozenset([
    "the", "and", "inc", "ltd", "llc", "co", "corp", "corporation", "company",
    "limited", "group", "pvt", "private", "plc", "llp",
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
        match = LINKEDIN_NAME_WITH_CREDENTIALS_PATTERN.match(candidate) or LINKEDIN_NAME_PREFIX_PATTERN.match(candidate)
        if match:
            return match.group(1).strip()
        if len(candidate.split()) <= 5 and not CSR_ROLE_KEYWORD_PATTERN.search(candidate):
            return candidate.strip()
    fallback_match = LINKEDIN_NAME_PREFIX_PATTERN.match(strip_linkedin_suffix(raw_title))
    return fallback_match.group(1).strip() if fallback_match else ""


def extract_name_from_url(url: str) -> str:
    try:
        path = urlparse(url).path
    except Exception:
        return ""
    match = re.search(r"/in/([^/?#]+)", path)
    if not match:
        return ""
    slug = re.sub(r"-[a-f0-9]{6,}$", "", match.group(1))
    slug = slug.replace("-", " ").strip()
    tokens = [t.capitalize() for t in slug.split() if t and not t.isdigit()]
    return " ".join(tokens[:4])


def extract_job_title(raw_title: str, snippet: str, parts: list[str] | None = None) -> str:
    parts = parts if parts is not None else split_linkedin_title(raw_title)
    role_segments = [p for p in parts[1:] if p and not COMPANY_AT_PATTERN.fullmatch(p)]
    if role_segments:
        candidate = role_segments[0].strip(" .")
        at_split = COMPANY_AT_PATTERN.search(candidate)
        if at_split:
            candidate = candidate[:at_split.start()].strip(" .,")
        return candidate
    match = CSR_ROLE_KEYWORD_PATTERN.search(f"{raw_title} {snippet}")
    return match.group(0).strip() if match else ""


def extract_seniority_level(job_title: str, raw_title: str, snippet: str) -> str:
    haystack = f"{job_title} {raw_title}".strip() or snippet
    for level, pattern in SENIORITY_KEYWORD_PATTERN_ORDER:
        if pattern.search(haystack):
            return level
    if snippet and any(pattern.search(snippet) for _, pattern in SENIORITY_KEYWORD_PATTERN_ORDER):
        for level, pattern in SENIORITY_KEYWORD_PATTERN_ORDER:
            if pattern.search(snippet):
                return level
    return "UNKNOWN"


def extract_department(job_title: str, snippet: str) -> str:
    match = DEPARTMENT_KEYWORD_PATTERN.search(job_title)
    if match:
        return match.group(0).strip().upper()
    match = DEPARTMENT_KEYWORD_PATTERN.search(snippet or "")
    return match.group(0).strip().upper() if match else ""


def _company_tokens(company: str) -> list[str]:
    return [t for t in re.sub(r"[^a-z0-9 ]", " ", company.lower()).split()
            if len(t) > 2 and t not in COMPANY_STOPWORDS]


def extract_company_affiliation(raw_title: str, snippet: str, parts: list[str] | None = None,
                                 company: str = "") -> str:
    parts = parts if parts is not None else split_linkedin_title(raw_title)

    if company:
        tokens = _company_tokens(company)
        for part in parts[1:]:
            lowered_part = part.lower()
            if not (tokens and any(token in lowered_part for token in tokens)):
                continue
            at_match = COMPANY_AT_PATTERN.search(part)
            if at_match:
                return at_match.group(1).strip(" .")
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


def extract_all_company_mentions(raw_title: str, snippet: str, primary_affiliation: str = "") -> list[str]:
    haystack = f"{raw_title} {snippet}"
    seen = set()
    ordered = []
    if primary_affiliation:
        seen.add(primary_affiliation.strip(" .").lower())
        ordered.append(primary_affiliation.strip(" ."))
    for match in MULTI_COMPANY_AT_PATTERN.finditer(haystack):
        candidate = match.group(1).strip(" .")
        key = candidate.lower()
        if key and key not in seen:
            seen.add(key)
            ordered.append(candidate)
    return ordered


def extract_contact_hints(snippet: str, raw_title: str) -> dict:
    haystack = f"{raw_title} {snippet}"
    email_match = EMAIL_PATTERN.search(haystack)
    phone_match = PHONE_PATTERN.search(haystack)
    return {
        "email": email_match.group(0) if email_match else "",
        "phone": re.sub(r"\s+", " ", phone_match.group(0)).strip() if phone_match else "",
    }


def extract_education_hints(snippet: str) -> list[str]:
    if not snippet:
        return []
    hits = []
    seen = set()
    for match in EDUCATION_HINT_PATTERN.finditer(snippet):
        value = match.group(0).strip()
        key = value.lower()
        if key not in seen:
            seen.add(key)
            hits.append(value)
    return hits


def extract_years_experience(snippet: str) -> int | None:
    if not snippet:
        return None
    match = YEARS_EXPERIENCE_PATTERN.search(snippet)
    return int(match.group(1)) if match else None


def extract_city(raw_title: str, snippet: str) -> str:
    haystack = f"{raw_title} {snippet}"
    match = CITY_LOOKUP_PATTERN.search(haystack)
    return match.group(0).strip().title() if match else ""


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


def is_currently_at_company(raw_title: str, snippet: str, affiliation: str, company: str) -> bool:
    if not company:
        return False
    tokens = _company_tokens(company)
    if not tokens:
        return False

    affiliation_lower = (affiliation or "").lower()
    if affiliation_lower and any(token in affiliation_lower for token in tokens):
        pass
    elif not affiliation_lower:
        return False
    else:
        return False

    haystack = f"{raw_title} {snippet}"
    if FORMER_ROLE_KEYWORD_PATTERN.search(haystack):
        return False

    sentences = re.split(r"(?<=[.!?])\s+", snippet or "")
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(token in sentence_lower for token in tokens) and FORMER_ROLE_KEYWORD_PATTERN.search(sentence):
            return False

    return True


def parse_linkedin_hit(raw_title: str, snippet: str, url: str, company: str) -> dict:
    parts = split_linkedin_title(raw_title)
    name = extract_person_name(raw_title, parts) or extract_name_from_url(url)
    job_title = extract_job_title(raw_title, snippet, parts)
    affiliation = extract_company_affiliation(raw_title, snippet, parts, company=company)
    all_companies = extract_all_company_mentions(raw_title, snippet, affiliation)
    india_signal = location_mentions_india(raw_title, snippet, url)
    current_role = is_current_csr_role(raw_title, snippet)
    has_csr_signal = bool(CSR_ROLE_KEYWORD_PATTERN.search(f"{raw_title} {snippet}"))
    company_match = is_currently_at_company(raw_title, snippet, affiliation, company)
    seniority = extract_seniority_level(job_title, raw_title, snippet)
    department = extract_department(job_title, snippet)
    city = extract_city(raw_title, snippet)
    contact = extract_contact_hints(snippet, raw_title)
    education = extract_education_hints(snippet)
    years_experience = extract_years_experience(snippet)

    title_csr_match = bool(CSR_ROLE_KEYWORD_PATTERN.search(job_title))
    title_blocked = bool(NON_CSR_TITLE_KEYWORD_PATTERN.search(job_title)) and not title_csr_match
    role_verified = has_csr_signal and not title_blocked

    if current_role and company_match and india_signal:
        confidence = "HIGH"
    elif company_match and (current_role or has_csr_signal):
        confidence = "MEDIUM"
    elif current_role and india_signal:
        confidence = "LOW"
    else:
        confidence = "LOW"

    profile_completeness = sum([
        bool(name), bool(job_title), bool(affiliation), bool(city),
        bool(contact["email"]) or bool(contact["phone"]), bool(education), years_experience is not None,
    ])

    return {
        "name": name,
        "title": job_title,
        "seniority": seniority,
        "department": department,
        "company_affiliation": affiliation,
        "all_company_mentions": all_companies,
        "city": city,
        "url": url,
        "raw_title": strip_linkedin_suffix(raw_title),
        "snippet": (snippet or "").strip(),
        "contact_email": contact["email"],
        "contact_phone": contact["phone"],
        "education_hints": education,
        "years_experience": years_experience,
        "india_location_signal": india_signal,
        "is_current_csr_role": current_role,
        "has_csr_signal": has_csr_signal,
        "role_verified": role_verified,
        "is_current_company_match": company_match,
        "profile_completeness": profile_completeness,
        "confidence": confidence,
    }