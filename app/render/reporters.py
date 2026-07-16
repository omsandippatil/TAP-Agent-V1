from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="app/render/templates")


def tier_color(result: dict) -> str:
    return (result.get("scoring_tier") or {}).get("color", "#6B7280")


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
}


def source_url_by_name(sources: list) -> dict:
    lookup = {}
    for source in sources or []:
        name = source.get("source_name", "")
        url = source.get("url", "")
        if name and url and name not in lookup:
            lookup[name] = url
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
        "decision_makers": result.get("decision_makers", []) or [],
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
    }


def render_results(request, company: str, mode: str, result: dict, files: dict):
    context = build_context(request, company, mode, result, files)
    return templates.TemplateResponse("results.html", context)