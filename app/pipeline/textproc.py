import functools
import re

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


def clean_source_text(raw_text, seen_fingerprints=None):
    """Strip boilerplate/nav/cookie-banner lines and cross-source duplicate
    sentences. This never rewrites, reorders, or drops sentences based on
    topical relevance — it only removes junk and exact-duplicate boilerplate
    that appears verbatim across multiple pages. The stopword fingerprint is
    used purely as a cheap dedup key; the sentence text that is kept is
    returned completely verbatim, so the model reads real grammatical
    sentences rather than a stopword-stripped bag of words.
    """
    if not raw_text:
        return ""
    if seen_fingerprints is None:
        seen_fingerprints = set()

    kept = []
    for sentence in split_sentences(raw_text):
        if is_boilerplate_sentence(sentence):
            continue
        fingerprint = _stopword_fingerprint(sentence)
        if fingerprint:
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
        kept.append(sentence)
    return " ".join(kept)


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


def remove_stopwords_and_boilerplate(sources, company=""):
    """The single entry point this module exposes to the rest of the
    pipeline. For every FOUND source, strips boilerplate/nav junk and
    cross-source duplicate sentences via clean_source_text(), and leaves
    everything else about the source dict untouched. No relevance scoring,
    no keyword weighting, no per-source priority — every source is treated
    identically. `company` is accepted for interface symmetry with the rest
    of the pipeline but is intentionally unused: which sentences survive
    here must not depend on which company is being screened, only on
    whether a sentence is boilerplate or a duplicate.
    """
    if not sources:
        return sources

    seen_fingerprints = set()
    cleaned = []
    for source in sources:
        if source.get("status") != "FOUND" or not source.get("text"):
            cleaned.append(source)
            continue
        cleaned_text = clean_source_text(source["text"], seen_fingerprints)
        cleaned.append({**source, "text": cleaned_text})
    return cleaned


def clean_and_budget_sources(sources, token_budget):
    """Clean sources, then if the combined evidence still exceeds
    token_budget, truncate every source proportionally to its own length.
    This is deliberately dumb and transparent: no source is judged more
    important than another here, so nothing about which source gets more
    room depends on a code-level opinion about relevance. If content is
    lost, it's lost evenly across all sources, not selectively.
    """
    if not sources:
        return sources

    found_sources = [s for s in sources if s.get("status") == "FOUND" and s.get("text")]
    if not found_sources:
        return sources

    token_budget = max(0, int(token_budget or 0))

    seen_fingerprints = set()
    cleaned = []
    for source in found_sources:
        cleaned_text = clean_source_text(source.get("text", ""), seen_fingerprints)
        cleaned.append({**source, "text": cleaned_text})

    total_tokens = sum(estimate_tokens(s["text"]) for s in cleaned)

    if total_tokens == 0 or total_tokens <= token_budget:
        result_by_name = {s.get("source_name"): s for s in cleaned}
    else:
        keep_ratio = token_budget / total_tokens if total_tokens else 0
        result_by_name = {}
        for source in cleaned:
            text = source["text"]
            target_chars = max(MIN_TRUNCATE_CHARS, int(len(text) * keep_ratio))
            result_by_name[source.get("source_name")] = {**source, "text": text[:target_chars]}

    output = []
    for source in sources:
        name = source.get("source_name")
        output.append(result_by_name.get(name, source))
    return output


def combine_evidence_text(sources):
    if not sources:
        return ""
    chunks = []
    for source in sources:
        if source.get("status") != "FOUND" or not source.get("text"):
            continue
        chunks.append(f"[{source.get('source_name', 'source')}]\n{source['text']}")
    return "\n\n".join(chunks)