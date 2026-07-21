from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="app/render/templates")


SOURCE_LABELS = {
    "india_csr_page": "Company CSR page",
    "mca_portal": "MCA portal",
    "mca_via_search": "MCA (via search)",
    "national_csr_portal": "National CSR Portal",
    "annual_report": "Annual / sustainability report",
    "global_annual_report": "Annual / sustainability report",
    "partner_search": "Partner search",
    "people_search": "LinkedIn people search",
    "plans_search": "Partnerships & plans search",
    "sector_eligibility_search": "Sector & eligibility search",
    "important_link": "Important link",
}


def tier_color(result: dict) -> str:
    return (result.get("scoring_tier") or {}).get("color", "#6B7280")


def source_url_by_name(sources: list) -> dict:
    lookup = {}
    for source in sources or []:
        name = source.get("source_name", "")
        url = source.get("url", "")
        if name and url and name not in lookup:
            lookup[name] = url
    return lookup


def source_bank_by_number(result: dict) -> dict:
    return {entry["number"]: entry for entry in result.get("source_bank", []) or [] if entry.get("number")}


def source_bank_by_name(result: dict) -> dict:
    lookup = {}
    for entry in result.get("source_bank", []) or []:
        name = entry.get("source_name", "")
        if name and name not in lookup:
            lookup[name] = entry
    return lookup


def build_source_links(sources: list) -> list:
    links = []
    for source in sources or []:
        name = source.get("source_name", "")
        url = source.get("url", "")
        links.append({
            "name": name,
            "label": SOURCE_LABELS.get(name, name),
            "status": source.get("status", "NOT_TRIED"),
            "url": url,
            "is_pdf": url.lower().endswith(".pdf") if url else False,
        })
    return links


def is_linkedin_profile_url(url: str) -> bool:
    return bool(url) and "linkedin.com/in/" in url.lower()


def merge_decision_makers(result: dict) -> list:
    analysis = result.get("analysis") or {}
    llm_people = analysis.get("decision_makers") or []
    scraped_people = result.get("decision_makers") or []

    scraped_by_name = {}
    for person in scraped_people:
        key = (person.get("name") or "").strip().lower()
        if key and key not in scraped_by_name:
            scraped_by_name[key] = person

    merged = []
    seen_names = set()

    for person in llm_people:
        name = (person.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        seen_names.add(key)
        scraped_match = scraped_by_name.get(key, {})
        linkedin_url = person.get("linkedin_url") or scraped_match.get("url", "")
        if not is_linkedin_profile_url(linkedin_url):
            linkedin_url = ""
        merged.append({
            "name": name,
            "title": person.get("title") or scraped_match.get("title", ""),
            "company_affiliation": scraped_match.get("company_affiliation", ""),
            "linkedin_url": linkedin_url,
            "url": linkedin_url,
            "tenure_status": person.get("tenure_status", "UNKNOWN"),
            "tenure_evidence": person.get("tenure_evidence", ""),
            "public_facing_score": person.get("public_facing_score", 0),
            "source_excerpt": person.get("source_excerpt", ""),
            "source": person.get("source", ""),
            "india_location_signal": bool(scraped_match.get("india_location_signal")),
            "confidence": scraped_match.get("confidence", "MEDIUM"),
        })

    for person in scraped_people:
        name = (person.get("name") or "").strip()
        key = name.lower()
        if not name or key in seen_names:
            continue
        seen_names.add(key)
        linkedin_url = person.get("url", "")
        if not is_linkedin_profile_url(linkedin_url):
            linkedin_url = ""
        merged.append({
            "name": name,
            "title": person.get("title", ""),
            "company_affiliation": person.get("company_affiliation", ""),
            "linkedin_url": linkedin_url,
            "url": linkedin_url,
            "tenure_status": "UNKNOWN",
            "tenure_evidence": "",
            "public_facing_score": 0,
            "source_excerpt": person.get("snippet", "") or person.get("title", ""),
            "source": "people_search",
            "india_location_signal": bool(person.get("india_location_signal")),
            "confidence": person.get("confidence", "LOW"),
        })

    return merged


def resolve_source_ref(source_ref, source_bank_numbers: dict, source_bank_names: dict, source_urls: dict, source_labels: dict) -> dict | None:
    if source_ref is None or source_ref == "":
        return None
    if isinstance(source_ref, int) or (isinstance(source_ref, str) and source_ref.isdigit()):
        entry = source_bank_numbers.get(int(source_ref))
        if entry:
            return {"label": entry.get("label", ""), "url": entry.get("url", ""), "number": entry.get("number")}
        return None
    entry = source_bank_names.get(source_ref)
    if entry:
        return {"label": entry.get("label", ""), "url": entry.get("url", ""), "number": entry.get("number")}
    url = source_urls.get(source_ref, "")
    label = source_labels.get(source_ref, source_ref)
    if url or label:
        return {"label": label, "url": url, "number": None}
    return None


def build_context(request, company: str, mode: str, result: dict, files: dict) -> dict:
    file_ctx = {
        label: {"filename": fn, "mime": mime, "b64": b64}
        for label, (fn, mime, b64) in (files or {}).items()
    }
    analysis = result.get("analysis") or {}
    sources = result.get("sources", []) or []
    return {
        "request": request,
        "company": company,
        "mode": mode,
        "result": result,
        "analysis": analysis,
        "criteria": analysis.get("criteria", []),
        "score_breakdown": result.get("score_breakdown", {}) or {},
        "tier": result.get("scoring_tier", {}) or {},
        "tier_color": tier_color(result),
        "fit": result.get("fit_score", 0),
        "fit_rationale": analysis.get("fit_rationale", ""),
        "decision_makers": merge_decision_makers(result),
        "files": file_ctx,
        "important_links": result.get("important_links") or [],
        "eligibility": analysis.get("eligibility", {}) or {},
        "sector": analysis.get("sector", {}) or {},
        "group_foundation": analysis.get("group_foundation", {}) or {},
        "rfp_signal": analysis.get("rfp_signal", {}) or {},
        "board_affinity": analysis.get("board_affinity", {}) or {},
        "volunteering": analysis.get("volunteering", {}) or {},
        "contact_pathway": analysis.get("contact_pathway", {}) or {},
        "spend": analysis.get("spend", {}) or {},
        "source_urls": source_url_by_name(sources),
        "source_labels": SOURCE_LABELS,
        "source_links": build_source_links(sources),
        "source_bank": source_bank_by_number(result),
        "source_bank_by_name": source_bank_by_name(result),
    }


def render_results(request, company: str, mode: str, result: dict, files: dict):
    context = build_context(request, company, mode, result, files)
    return templates.TemplateResponse("results.html", context)


def render_results_page(request, company: str, mode: str, result: dict, files: dict):
    context = build_context(request, company, mode, result, files)
    return templates.TemplateResponse("results_page.html", context)