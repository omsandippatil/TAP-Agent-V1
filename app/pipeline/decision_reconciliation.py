import re

CSR_ROLE_TERMS = (
    "csr", "corporate social responsibility", "esg", "sustainability",
    "foundation", "social impact", "corporate citizenship", "philanthropy",
)

FORMER_ROLE_PATTERN = re.compile(
    r"\b(previously|formerly|former|ex-|past|until\s+\d{4})\b", re.IGNORECASE,
)

NARRATIVE_PERSON_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*(?:[-\u2013\u2014,]|\()\s*(?:is\s+)?(?:the\s+)?"
    r"(Head\s+of\s+CSR|CSR\s+Head|Head\s*[-,]?\s*CSR|Chief\s+Sustainability\s+Officer|"
    r"Head\s+of\s+Sustainability|Sustainability\s+Head|Head\s+of\s+Foundation|"
    r"Foundation\s+Director|CSR\s+Director|CSR\s+Manager|CSR\s+Lead)\b",
    re.IGNORECASE,
)

NAMED_PROGRAMME_MENTION_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,5}\s+"
    r"(?:Programme|Program|Initiative|Project|Mission|Scholarship|Lab|Labs|Academy))\b"
)

NO_CSR_HEAD_CONTRADICTION_PATTERN = re.compile(
    r"no\s+(?:named\s+)?(?:csr|sustainability)\s+head\s+(?:has\s+been\s+)?identified|"
    r"no\s+csr\s+head\s+found|no\s+decision[\s-]makers?\s+found",
    re.IGNORECASE,
)


def _name_key(name):
    return re.sub(r"[^a-z]", "", (name or "").lower())


def _is_former_role(*texts):
    combined = " ".join(t for t in texts if t)
    return bool(combined) and bool(FORMER_ROLE_PATTERN.search(combined))


def extract_narrative_person_mentions(*texts):
    found, seen = [], set()
    for text in texts:
        if not text:
            continue
        for match in NARRATIVE_PERSON_PATTERN.finditer(text):
            name = match.group(1).strip()
            key = _name_key(name)
            if key and key not in seen:
                seen.add(key)
                found.append({"name": name, "title": match.group(2).strip()})
    return found


def people_hits_as_decision_makers(sources):
    people_source = next(
        (s for s in (sources or []) if s.get("source_name") == "people_search"), None
    )
    hits = (people_source or {}).get("people_hits", []) or []
    candidates = []
    for hit in hits:
        if hit.get("confidence") not in ("HIGH", "MEDIUM"):
            continue
        haystack = f"{hit.get('title', '')} {hit.get('snippet', '')}".lower()
        if not any(term in haystack for term in CSR_ROLE_TERMS):
            continue
        candidates.append({
            "name": hit.get("name", ""),
            "title": hit.get("title", ""),
            "tenure_status": "UNKNOWN",
            "tenure_evidence": "",
            "is_india_specific": bool(hit.get("india_location_signal")),
            "source_excerpt": (hit.get("snippet", "") or "")[:260],
            "source": "people_search",
            "linkedin_url": hit.get("url", ""),
        })
    return candidates


def extract_named_programme_mentions(*texts):
    found, seen = [], set()
    for text in texts:
        if not text:
            continue
        for match in NAMED_PROGRAMME_MENTION_PATTERN.finditer(text):
            name = re.sub(r"\s+", " ", match.group(1)).strip()
            key = _name_key(name)
            if key and key not in seen:
                seen.add(key)
                found.append(name)
    return found


def reconcile_extraction(extraction, sources):
    if not isinstance(extraction, dict):
        return extraction

    existing_people = list(extraction.get("decision_makers") or [])
    existing_keys = {_name_key(p.get("name")) for p in existing_people if p.get("name")}

    narrative_texts = [
        extraction.get("csr_head_note", ""),
        extraction.get("key_facts_summary", ""),
    ]
    candidates = extract_narrative_person_mentions(*narrative_texts) + people_hits_as_decision_makers(sources)

    for candidate in candidates:
        key = _name_key(candidate.get("name"))
        if not key or key in existing_keys:
            continue
        if _is_former_role(
            candidate.get("title", ""), candidate.get("tenure_evidence", ""), candidate.get("source_excerpt", "")
        ):
            continue
        existing_keys.add(key)
        existing_people.append({
            "name": candidate.get("name", ""),
            "title": candidate.get("title", ""),
            "public_facing_score": 0,
            "tenure_status": candidate.get("tenure_status", "UNKNOWN"),
            "tenure_evidence": candidate.get("tenure_evidence", ""),
            "is_india_specific": candidate.get("is_india_specific", False),
            "source_excerpt": candidate.get("source_excerpt", ""),
            "source": candidate.get("source", ""),
            "linkedin_url": candidate.get("linkedin_url", ""),
        })
    extraction["decision_makers"] = existing_people

    existing_programmes = list(extraction.get("programmes") or [])
    existing_programme_keys = {_name_key(p.get("name")) for p in existing_programmes if p.get("name")}

    narrative_programme_texts = [
        extraction.get("fit_rationale", ""),
        extraction.get("strategic_insight", ""),
        extraction.get("csr_head_note", ""),
        extraction.get("key_facts_summary", ""),
        extraction.get("delivery_model_evidence", ""),
    ]
    for name in extract_named_programme_mentions(*narrative_programme_texts):
        key = _name_key(name)
        if key in existing_programme_keys:
            continue
        existing_programme_keys.add(key)
        existing_programmes.append({
            "name": name,
            "what_is_funded": "",
            "beneficiary_group": "",
            "beneficiary_type": "OTHER",
            "description": (
                "Named in narrative text elsewhere in this report; not separately "
                "extracted in full — verify manually."
            ),
            "is_multi_year": False,
            "cohort_or_scale": "",
            "funded_by_entity": "",
            "chain_missing_elements": [
                "beneficiaries", "geography", "partner",
                "government_school_involvement", "scale_or_outcomes", "funding_amount",
            ],
            "source_excerpt": "",
            "source": "",
            "confidence": "probable",
        })
    extraction["programmes"] = existing_programmes

    if existing_people and extraction.get("csr_head_note") and NO_CSR_HEAD_CONTRADICTION_PATTERN.search(extraction["csr_head_note"]):
        extraction["csr_head_note"] = ""

    return extraction