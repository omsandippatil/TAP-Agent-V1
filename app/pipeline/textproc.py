import functools
import re
from collections import Counter

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORD_SPLIT = re.compile(r"[a-zA-Z][a-zA-Z\-']{1,}")
CLAUSE_BOUNDARY = re.compile(r"[.!?;,\n]")

STOPWORDS = frozenset("""
a an the and or but if then else for of to in on at by with from as is are was were be been being
this that these those it its it's their his her they he she we you your our i me my mine yours
ours theirs him them us not no nor so such than too very can will just about into over after before
under again further once here there when where why how all any both each few more most other some
such only own same s t can will don should now also may might must shall would could
company companies limited ltd pvt private inc corp corporation group india indian
""".split())

CSR_SIGNAL_WEIGHTS = {
    "csr": 5, "corporate social responsibility": 6, "philanthropy": 4,
    "social responsibility": 5, "schedule vii": 5, "csr spend": 6,
    "csr expenditure": 6, "csr budget": 5, "csr obligation": 5,
    "csr fund": 5, "community investment": 4, "sustainability": 3,
    "foundation": 3, "ngo": 4, "implementing partner": 5,
    "implementation partner": 5, "education": 3, "skill": 3,
    "skilling": 3, "crore": 4, "lakh": 3, "partnered": 4,
    "partnership": 4, "initiative": 2, "programme": 3, "program": 3,
    "csr committee": 5, "csr policy": 4, "csr-2": 6, "mca": 4,
    "annual report": 3, "digital literacy": 4, "government school": 5,
    "stem": 3, "coding": 3, "financial literacy": 4, "21st century": 3,
    "net worth": 5, "turnover": 4, "net profit": 4, "section 135": 6,
    "group foundation": 6, "parent foundation": 6, "central trust": 5,
    "routed through": 4, "csr trust": 5,
    "appointed": 3, "joined as": 4, "since 20": 3, "new head": 4,
    "took over": 3, "tenure": 3,
    "board of directors": 4, "promoter": 4, "chairman": 3,
    "chairperson": 3, "personal foundation": 5, "philanthropist": 4,
    "employee volunteering": 6, "payroll giving": 6, "volunteering programme": 5,
    "volunteer hours": 4, "employee giving": 5,
    "request for proposal": 6, "call for proposals": 6, "open call": 4,
    "looking for partners": 5, "invite ngos": 5, "grant application": 4,
    "apply for grant": 4,
    "sector": 2, "industry": 2, "revenue": 3, "profit": 3,
    "rising": 2, "declining": 2, "year-on-year": 3, "yoy": 3,
    "multi-year": 4, "three-year": 4, "five-year": 4,
}

FINANCIAL_TERMS = frozenset([
    "crore", "cr", "lakh", "lac", "rupees", "inr", "rs", "budget",
    "spent", "spend", "expenditure", "invest", "investment", "fund",
    "funding", "allocation", "outlay", "net worth", "turnover", "profit",
])

GEOGRAPHY_TERMS = frozenset([
    "delhi", "mumbai", "maharashtra", "karnataka", "bangalore", "bengaluru",
    "chennai", "kolkata", "pune", "hyderabad", "gujarat", "rajasthan",
    "punjab", "haryana", "up", "bihar", "odisha", "telangana", "kerala",
    "tamil nadu", "west bengal", "andhra pradesh", "assam", "goa",
])

TENURE_TERMS = frozenset([
    "appointed", "joined", "since", "took over", "tenure", "years at",
    "new head", "newly appointed", "former", "ex-", "until",
])

GOVERNANCE_TERMS = frozenset([
    "board of directors", "promoter", "chairman", "chairperson",
    "group foundation", "parent foundation", "central trust",
    "routed through", "csr trust", "csr committee",
])

SOURCE_PRIORITY_WEIGHT = {
    "india_csr_page": 1.35,
    "annual_report": 1.3,
    "global_annual_report": 1.3,
    "mca_portal": 1.25,
    "national_csr_portal": 1.2,
    "mca_via_search": 1.05,
    "partner_search": 1.0,
    "plans_search": 1.0,
    "people_search": 0.95,
    "sector_eligibility_search": 0.95,
    "web_search_snippet": 0.8,
}

BOILERPLATE_SENTENCE_PATTERNS = [
    re.compile(r"^(home|about us?|contact us?|careers?|sign in|log ?in|sign up|register)\b", re.IGNORECASE),
    re.compile(r"^(privacy policy|terms( of (use|service))?|cookie policy|disclaimer|sitemap)\b", re.IGNORECASE),
    re.compile(r"(all rights reserved|copyright ©|©\s*\d{4})", re.IGNORECASE),
    re.compile(r"^(share|tweet|follow us|subscribe|read more|load more|back to top)\b", re.IGNORECASE),
    re.compile(r"javascript is disabled|enable cookies|click here to|accept cookies|we use cookies", re.IGNORECASE),
    re.compile(r"^\W*$"),
]


def is_boilerplate_sentence(sentence: str) -> bool:
    stripped = sentence.strip()
    if len(stripped) < 20:
        return True
    if sum(1 for ch in stripped if ch.isalpha()) < len(stripped) * 0.4:
        return True
    return any(pattern.search(stripped) for pattern in BOILERPLATE_SENTENCE_PATTERNS)


def normalize_document(raw_text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw_text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    normalized = normalize_document(text)
    if not normalized:
        return []
    pieces = SENTENCE_SPLIT.split(normalized)
    return [p.strip() for p in pieces if p.strip()]


def _clause_safe_bigrams(normalized_text: str) -> list[str]:
    bigrams = []
    for clause in CLAUSE_BOUNDARY.split(normalized_text.lower()):
        words = [w for w in WORD_SPLIT.findall(clause) if w not in STOPWORDS and len(w) > 2]
        for i in range(len(words) - 1):
            bigrams.append(f"{words[i]} {words[i + 1]}")
    return bigrams


def extract_keywords(text: str, top_n: int = 40, document_frequency: Counter | None = None, corpus_size: int = 1) -> list[tuple]:
    normalized = normalize_document(text).lower()
    words = WORD_SPLIT.findall(normalized)
    filtered = [w for w in words if w not in STOPWORDS and len(w) > 2]
    unigram_counts = Counter(filtered)
    bigram_counts = Counter(bg for bg in _clause_safe_bigrams(normalized) if bg.count(" ") == 1)

    scored = Counter()
    for word, count in unigram_counts.items():
        scored[word] += count

    for phrase, count in bigram_counts.items():
        if count >= 2:
            scored[phrase] += count * 1.5

    for phrase, weight in CSR_SIGNAL_WEIGHTS.items():
        occurrences = normalized.count(phrase)
        if occurrences:
            scored[phrase] += occurrences * weight

    if document_frequency and corpus_size > 1:
        for term in list(scored.keys()):
            df = document_frequency.get(term, 1)
            idf = max(0.15, 1.0 - (df - 1) / corpus_size)
            scored[term] = scored[term] * (0.5 + 0.5 * idf)

    ranked = scored.most_common(top_n)
    return ranked


def score_sentence_relevance(sentence: str, company_tokens: list[str]) -> float:
    lowered = sentence.lower()
    length = len(sentence)
    if length < 25:
        return -1.0

    score = 0.0
    distinct_signal_hits = 0
    for phrase, weight in CSR_SIGNAL_WEIGHTS.items():
        if phrase in lowered:
            score += weight
            distinct_signal_hits += 1

    if company_tokens and any(token in lowered for token in company_tokens):
        score += 4.0

    if any(term in lowered for term in FINANCIAL_TERMS):
        score += 3.0

    if any(term in lowered for term in GEOGRAPHY_TERMS):
        score += 1.5

    if any(term in lowered for term in TENURE_TERMS):
        score += 1.5

    if any(term in lowered for term in GOVERNANCE_TERMS):
        score += 1.5

    digit_density = sum(1 for ch in sentence if ch.isdigit())
    if digit_density >= 2:
        score += 1.0

    if distinct_signal_hits >= 2:
        score += min(distinct_signal_hits - 1, 3) * 0.75

    length_penalty = max(1.0, length / 220.0)
    return score / length_penalty


def company_name_tokens(company: str) -> list[str]:
    stop = {"india", "limited", "ltd", "private", "pvt", "the", "and", "of", "company",
            "corp", "corporation", "inc", "group", "technologies", "solutions", "services", "international"}
    return [t for t in re.sub(r"[^a-z0-9 ]", " ", company.lower()).split() if len(t) > 2 and t not in stop]


def structure_source_text(source_name: str, raw_text: str, company: str, max_sentences: int = 40,
                           document_frequency: Counter | None = None, corpus_size: int = 1) -> dict:
    sentences = split_sentences(raw_text)
    company_tokens = company_name_tokens(company)

    kept = []
    seen_sentence_keys = set()
    for sentence in sentences:
        if is_boilerplate_sentence(sentence):
            continue
        key = re.sub(r"\s+", " ", sentence.lower()).strip()[:140]
        if key in seen_sentence_keys:
            continue
        relevance = score_sentence_relevance(sentence, company_tokens)
        if relevance <= 0:
            continue
        seen_sentence_keys.add(key)
        kept.append((relevance, sentence))

    kept.sort(key=lambda pair: pair[0], reverse=True)
    top_sentences = [s for _, s in kept[:max_sentences]]

    keywords = extract_keywords(
        " ".join(top_sentences) or raw_text, top_n=15,
        document_frequency=document_frequency, corpus_size=corpus_size,
    )

    return {
        "source_name": source_name,
        "sentences": top_sentences,
        "keywords": [kw for kw, _ in keywords],
        "sentence_count": len(top_sentences),
        "priority_weight": SOURCE_PRIORITY_WEIGHT.get(source_name, 1.0),
    }


def _document_frequency_across_sources(sources: list, company: str) -> Counter:
    document_frequency = Counter()
    for source in sources:
        if source.get("status") != "FOUND" or not source.get("text"):
            continue
        normalized = normalize_document(source["text"]).lower()
        words = {w for w in WORD_SPLIT.findall(normalized) if w not in STOPWORDS and len(w) > 2}
        for word in words:
            document_frequency[word] += 1
    return document_frequency


def structure_all_sources(sources: list, company: str, max_sentences_per_source: int = 25) -> list[dict]:
    found_sources = [s for s in sources if s.get("status") == "FOUND" and s.get("text")]
    if not found_sources:
        return []

    document_frequency = _document_frequency_across_sources(found_sources, company)
    corpus_size = max(len(found_sources), 1)

    structured = []
    for source in found_sources:
        structured.append(
            structure_source_text(
                source.get("source_name", "source"),
                source["text"],
                company,
                max_sentences=max_sentences_per_source,
                document_frequency=document_frequency,
                corpus_size=corpus_size,
            )
        )
    return structured


@functools.lru_cache(maxsize=1)
def _tiktoken_encoding():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


@functools.lru_cache(maxsize=2048)
def _estimate_tokens_cached(text: str) -> int:
    try:
        return len(_tiktoken_encoding().encode(text))
    except Exception:
        return max(1, len(text) // 4)


def estimate_tokens(text: str) -> int:
    return _estimate_tokens_cached(text)


def build_token_budgeted_evidence(structured_sources: list[dict], company: str, token_budget: int) -> str:
    if not structured_sources:
        return ""

    global_keywords = Counter()
    for source in structured_sources:
        for kw in source["keywords"]:
            global_keywords[kw] += 1
    top_global_keywords = [kw for kw, _ in global_keywords.most_common(25)]

    header = f"KEY TERMS ACROSS SOURCES: {', '.join(top_global_keywords)}\n\n"
    used_tokens = estimate_tokens(header)
    chunks = [header]

    total_weight = sum(s.get("priority_weight", 1.0) for s in structured_sources) or 1.0
    remaining_budget = max(0, token_budget - used_tokens)

    ordered_sources = sorted(structured_sources, key=lambda s: s.get("priority_weight", 1.0), reverse=True)

    for source in ordered_sources:
        if used_tokens >= token_budget:
            break
        share = source.get("priority_weight", 1.0) / total_weight
        source_token_allowance = max(150, int(remaining_budget * share))
        block_lines = [f"[{source['source_name']}]"]
        block_tokens = estimate_tokens(block_lines[0])
        for sentence in source["sentences"]:
            candidate_line = f"- {sentence}"
            candidate_tokens = estimate_tokens(candidate_line)
            if block_tokens + candidate_tokens > source_token_allowance:
                break
            if used_tokens + candidate_tokens > token_budget:
                break
            block_lines.append(candidate_line)
            block_tokens += candidate_tokens
            used_tokens += candidate_tokens
        if len(block_lines) > 1:
            chunks.append("\n".join(block_lines))

    return "\n\n".join(chunks)

