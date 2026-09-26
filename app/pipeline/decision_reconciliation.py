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
    r"Foundation\s+Director|CSR\s+Director|CSR\s+Manager|CSR\s+Lead|"
    r"VP\s+(?:of\s+)?CSR|CSR\s+Committee\s+(?:Chair|Member)|Trustee)\b",
    re.IGNORECASE,
)

NAMED_PROGRAMME_MENTION_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,5}\s+"
    r"(?:Programme|Program|Initiative|Project|Mission|Scholarship|Lab|Labs|Academy|"
    r"Fellowship|Curriculum|Workshop))\b"
)

NO_CSR_HEAD_CONTRADICTION_PATTERN = re.compile(
    r"no\s+(?:named\s+)?(?:csr|sustainability)\s+head\s+(?:has\s+been\s+)?identified|"
    r"no\s+csr\s+head\s+found|no\s+decision[\s-]makers?\s+found|"
    r"insufficient\s+(?:evidence|data)\s+(?:on|for)\s+decision[\s-]makers?",
    re.IGNORECASE,
)

NARRATIVE_FIELDS = (
    "csr_head_note", "key_facts_summary", "strategic_insight", "fit_rationale",
    "delivery_model_evidence", "programme_depth_evidence", "partnership_evidence",
)

PROGRAMME_NARRATIVE_FIELDS = (
    "fit_rationale", "strategic_insight", "csr_head_note", "key_facts_summary",
    "delivery_model_evidence", "programme_depth_evidence",
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
        if _is_former_role(hit.get("title", ""), hit.get("snippet", "")):
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


def child_hits_as_decision_makers(registry):
    if registry is None:
        return []
    candidates = []
    for entry in registry.entries():
        if entry.get("kind") != "child" or entry.get("source_name") != "people_search":
            continue
        label = entry.get("label", "")
        name = label.split("—", 1)[-1].strip() if "—" in label else ""
        if not name:
            continue
        excerpt = entry.get("excerpt", "")
        haystack = f"{name} {excerpt}".lower()
        if not any(term in haystack for term in CSR_ROLE_TERMS):
            continue
        if _is_former_role(excerpt):
            continue
        title = excerpt.split("—", 1)[0].strip() if "—" in excerpt else ""
        candidates.append({
            "name": name,
            "title": title,
            "tenure_status": "UNKNOWN",
            "tenure_evidence": "",
            "is_india_specific": False,
            "source_excerpt": excerpt[:260],
            "source": "people_search",
            "linkedin_url": entry.get("url", ""),
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


def _collect_all_decision_maker_candidates(extraction, sources, registry):
    narrative_texts = [extraction.get(field, "") for field in NARRATIVE_FIELDS]
    candidates = []
    candidates.extend(extract_narrative_person_mentions(*narrative_texts))
    candidates.extend(people_hits_as_decision_makers(sources))
    candidates.extend(child_hits_as_decision_makers(registry))
    return candidates


def _merge_decision_makers(extraction, sources, registry):
    existing_people = list(extraction.get("decision_makers") or [])
    existing_keys = {_name_key(p.get("name")) for p in existing_people if p.get("name")}

    for candidate in _collect_all_decision_maker_candidates(extraction, sources, registry):
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
    return existing_people


def _merge_programmes(extraction):
    existing_programmes = list(extraction.get("programmes") or [])
    existing_programme_keys = {_name_key(p.get("name")) for p in existing_programmes if p.get("name")}

    narrative_texts = [extraction.get(field, "") for field in PROGRAMME_NARRATIVE_FIELDS]
    for name in extract_named_programme_mentions(*narrative_texts):
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
    return existing_programmes


def _reconcile_narrative_contradictions(extraction, has_decision_makers):
    if has_decision_makers:
        lead = extraction["decision_makers"][0]
        lead_line = f"{lead.get('name', '')} — {lead.get('title', '')}".strip(" —")
        for field in NARRATIVE_FIELDS:
            value = extraction.get(field, "")
            if value and NO_CSR_HEAD_CONTRADICTION_PATTERN.search(value):
                if field == "csr_head_note":
                    extraction[field] = lead_line
                else:
                    extraction[field] = NO_CSR_HEAD_CONTRADICTION_PATTERN.sub("", value).strip()
    else:
        note = extraction.get("csr_head_note", "")
        if note and NO_CSR_HEAD_CONTRADICTION_PATTERN.search(note):
            extraction["csr_head_note"] = ""


def reconcile_extraction(extraction, sources, registry=None):
    if not isinstance(extraction, dict):
        return extraction

    extraction["decision_makers"] = _merge_decision_makers(extraction, sources, registry)
    extraction["programmes"] = _merge_programmes(extraction)
    _reconcile_narrative_contradictions(extraction, bool(extraction["decision_makers"]))

    return extraction