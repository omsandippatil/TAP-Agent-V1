import functools
import logging
import re

logger = logging.getLogger("tap.textproc")

STOPWORDS = frozenset("""
a an the and or but if then else for of to in on at by with from as is are
was were be been being this that these those it its it's their his her
they he she we you your our i me my mine yours ours theirs him them us not
no nor so such than too very can will just about into over after before
under again further once here there when where why how all any both each
few more most other some such only own same s t can will don should now
also may might must shall would could
""".split())

BOILERPLATE_LINE_PATTERNS = (
    re.compile(r"^(home|about us?|contact us?|careers?|sign in|log ?in|sign up|register)\b", re.IGNORECASE),
    re.compile(r"^(privacy policy|terms( of (use|service))?|cookie policy|disclaimer|sitemap)\b", re.IGNORECASE),
    re.compile(r"(all rights reserved|copyright ©|©\s*\d{4})", re.IGNORECASE),
    re.compile(r"^(share|tweet|follow us|subscribe|read more|load more|back to top)\b", re.IGNORECASE),
    re.compile(r"javascript is disabled|enable cookies|click here to|accept cookies|we use cookies", re.IGNORECASE),
    re.compile(r"^\W*$"),
)

CLAUSE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\u20b9\"'(])")

_ABBREVIATIONS = frozenset([
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "eg", "ie",
    "no", "rs", "inr", "co", "ltd", "pvt", "govt", "dept", "univ", "assn",
    "fig", "approx", "est", "u.s", "u.k", "vol", "resp", "rev",
])
_ABBREV_GUARD = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in sorted(_ABBREVIATIONS, key=len, reverse=True)) + r")\.$",
    re.IGNORECASE,
)
_DECIMAL_NUMBER = re.compile(r"\d\.\d")
_ELLIPSIS_PLACEHOLDER = "\u0000ELLIPSIS\u0000"

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"http\S+")
_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]")

MIN_SENTENCE_LENGTH = 15
MIN_ALPHA_RATIO = 0.4
MIN_TRUNCATE_CHARS = 200
FINGERPRINT_WORD_LIMIT = 20

LOG_SOURCE_TEXT_PREVIEW_CHARS = 400

RELEVANCE_KEYWORD_WEIGHTS = {
    "stem": 4, "artificial intelligence": 4, " ai ": 3, "coding": 4, "robotics": 3,
    "digital skill": 4, "digital literacy": 4, "government school": 5, "public school": 4,
    "teacher training": 4, "teacher capacity": 3, "curriculum": 3, "student": 3,
    "beneficiar": 3, "school": 2, "21st century": 3, "21st-century": 3,
    "e-learning": 3, "elearning": 3, "science fair": 3, "girls in": 3, "atal tinkering": 4,
    "csr expenditure": 3, "csr spend": 3, "crore": 2, "lakh": 2, "foundation": 2,
    "ngo partner": 3, "implementing partner": 3, "partnership": 2, "programme": 2,
    "program": 2, "initiative": 2, "csr committee": 1, "sustainability report": 1,
    "annexure": 1, "schedule vii": 1,
}

def normalize_whitespace_and_html(raw_text):
    if not raw_text:
        return ""
    text = _HTML_TAG.sub(" ", raw_text)
    text = _URL.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def split_sentences(text):
    normalized = normalize_whitespace_and_html(text)
    if not normalized:
        return []

    protected = normalized.replace("...", _ELLIPSIS_PLACEHOLDER)
    boundaries = []
    for match in CLAUSE_END.finditer(protected):
        prefix = protected[:match.start()]
        if _ABBREV_GUARD.search(prefix):
            continue
        window = protected[max(0, match.start() - 2):match.start() + 2]
        if _DECIMAL_NUMBER.search(window):
            continue
        boundaries.append(match.start())

    pieces = []
    start = 0
    for boundary in boundaries:
        pieces.append(protected[start:boundary])
        start = boundary
    pieces.append(protected[start:])

    restored = (p.replace(_ELLIPSIS_PLACEHOLDER, "...").strip() for p in pieces)
    return [p for p in restored if p]


def is_boilerplate_sentence(sentence):
    stripped = sentence.strip()
    if len(stripped) < MIN_SENTENCE_LENGTH:
        return True
    if not stripped:
        return True
    alpha_count = sum(1 for ch in stripped if ch.isalpha())
    if alpha_count < len(stripped) * MIN_ALPHA_RATIO:
        return True
    return any(pattern.search(stripped) for pattern in BOILERPLATE_LINE_PATTERNS)


def _stopword_fingerprint(sentence):
    lowered = _NON_ALNUM.sub(" ", sentence.lower())
    words = sorted({w for w in lowered.split() if w not in STOPWORDS and len(w) > 2})
    return " ".join(words[:FINGERPRINT_WORD_LIMIT])


def clean_source_text(raw_text, seen_fingerprints=None, source_name=""):
    if not raw_text:
        return ""
    if seen_fingerprints is None:
        seen_fingerprints = set()

    total_sentences = 0
    boilerplate_dropped = 0
    duplicate_dropped = 0
    kept = []

    for sentence in split_sentences(raw_text):
        total_sentences += 1
        if is_boilerplate_sentence(sentence):
            boilerplate_dropped += 1
            continue
        fingerprint = _stopword_fingerprint(sentence)
        if fingerprint:
            if fingerprint in seen_fingerprints:
                duplicate_dropped += 1
                continue
            seen_fingerprints.add(fingerprint)
        kept.append(sentence)

    result = " ".join(kept)
    logger.info(
        "clean_source_text source=%r sentences_total=%d kept=%d boilerplate_dropped=%d "
        "duplicate_dropped=%d raw_chars=%d cleaned_chars=%d",
        source_name or "unknown", total_sentences, len(kept), boilerplate_dropped,
        duplicate_dropped, len(raw_text), len(result),
    )
    return result


@functools.lru_cache(maxsize=1)
def _tiktoken_encoding():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


@functools.lru_cache(maxsize=4096)
def _estimate_tokens_cached(text):
    try:
        return len(_tiktoken_encoding().encode(text))
    except Exception:
        return max(1, len(text) // 4)


def estimate_tokens(text):
    if not text:
        return 0
    return _estimate_tokens_cached(text)


def _sentence_relevance_score(sentence_lower):
    return sum(weight for keyword, weight in RELEVANCE_KEYWORD_WEIGHTS.items() if keyword in sentence_lower)


def relevance_ranked_sentences(text):
    sentences = split_sentences(text)
    scored = []
    for position, sentence in enumerate(sentences):
        score = _sentence_relevance_score(sentence.lower())
        scored.append((score, position, sentence))
    return scored


def truncate_preserving_relevant_content(text, target_chars):
    if not text or len(text) <= target_chars:
        return text or ""

    scored = relevance_ranked_sentences(text)
    if not scored:
        return text[:target_chars]

    ordered_by_relevance = sorted(scored, key=lambda item: (-item[0], item[1]))

    kept_positions = set()
    used_chars = 0
    for score, position, sentence in ordered_by_relevance:
        addition = len(sentence) + 1
        if used_chars + addition > target_chars and kept_positions:
            continue
        kept_positions.add(position)
        used_chars += addition
        if used_chars >= target_chars:
            break

    if not kept_positions:
        return text[:target_chars]

    ordered_sentences = [sentence for _, position, sentence in scored if position in kept_positions]
    result = " ".join(ordered_sentences)
    return result[:target_chars] if len(result) > target_chars else result


def clean_and_budget_sources(sources, token_budget):
    if not sources:
        logger.info("clean_and_budget_sources no sources provided token_budget=%s", token_budget)
        return sources

    found_sources = [s for s in sources if s.get("status") == "FOUND" and s.get("text")]
    if not found_sources:
        logger.info(
            "clean_and_budget_sources sources_total=%d found_sources=0 token_budget=%s — nothing to send to LLM",
            len(sources), token_budget,
        )
        return sources

    token_budget = max(0, int(token_budget or 0))

    seen_fingerprints = set()
    cleaned = []
    pre_clean_lengths = {}
    for source in found_sources:
        name = source.get("source_name", "")
        pre_clean_lengths[name] = len(source.get("text", ""))
        cleaned_text = clean_source_text(source.get("text", ""), seen_fingerprints, source_name=name)
        cleaned.append({**source, "text": cleaned_text})

    per_source_tokens = {s.get("source_name", ""): estimate_tokens(s["text"]) for s in cleaned}
    total_tokens = sum(per_source_tokens.values())

    truncated = False
    if total_tokens == 0 or total_tokens <= token_budget:
        result_by_name = {s.get("source_name"): s for s in cleaned}
        logger.info(
            "clean_and_budget_sources company_evidence FITS budget token_budget=%d total_tokens=%d "
            "sources_found=%d",
            token_budget, total_tokens, len(cleaned),
        )
    else:
        truncated = True
        keep_ratio = token_budget / total_tokens if total_tokens else 0
        result_by_name = {}
        for source in cleaned:
            name = source.get("source_name")
            text = source["text"]
            target_chars = max(MIN_TRUNCATE_CHARS, int(len(text) * keep_ratio))
            truncated_text = truncate_preserving_relevant_content(text, target_chars)
            result_by_name[name] = {**source, "text": truncated_text}
            logger.info(
                "clean_and_budget_sources TRUNCATED source=%r cleaned_chars=%d kept_chars=%d "
                "pre_clean_chars=%d keep_ratio=%.3f relevance_ranked=True",
                name, len(text), len(truncated_text), pre_clean_lengths.get(name, 0), keep_ratio,
            )
        logger.info(
            "clean_and_budget_sources company_evidence EXCEEDED budget token_budget=%d total_tokens=%d "
            "keep_ratio=%.3f sources_found=%d",
            token_budget, total_tokens, keep_ratio, len(cleaned),
        )

    output = []
    for source in sources:
        name = source.get("source_name")
        output.append(result_by_name.get(name, source))

    final_found = [s for s in output if s.get("status") == "FOUND" and s.get("text")]
    final_total_chars = sum(len(s.get("text", "")) for s in final_found)
    final_total_tokens = sum(estimate_tokens(s.get("text", "")) for s in final_found)
    logger.info(
        "clean_and_budget_sources FINAL PAYLOAD TO LLM sources_included=%d total_chars=%d "
        "est_total_tokens=%d truncated=%s breakdown=%s",
        len(final_found), final_total_chars, final_total_tokens, truncated,
        [(s.get("source_name", ""), len(s.get("text", ""))) for s in final_found],
    )
    for source in final_found:
        text = source.get("text", "")
        preview = text[:LOG_SOURCE_TEXT_PREVIEW_CHARS].replace("\n", " ")
        suffix = "..." if len(text) > LOG_SOURCE_TEXT_PREVIEW_CHARS else ""
        logger.debug(
            "clean_and_budget_sources LLM_INPUT source=%r chars=%d preview=%r%s",
            source.get("source_name", ""), len(text), preview, suffix,
        )

    return output


def combine_evidence_text(sources):
    if not sources:
        return ""
    chunks = []
    for source in sources:
        if source.get("status") != "FOUND" or not source.get("text"):
            continue
        chunks.append(f"[{source.get('source_name', 'source')}]\n{source['text']}")
    combined = "\n\n".join(chunks)
    logger.info(
        "combine_evidence_text sources_included=%d combined_chars=%d",
        sum(1 for s in sources if s.get("status") == "FOUND" and s.get("text")), len(combined),
    )
    return combined