import logging

from app.pipeline import google_search, llm, logo
from app.pipeline.scraper import mentions_company
from app.pipeline.source_registry import SourceRegistry, extract_cited_numbers, strip_unknown_citation_tokens
from app.pipeline.textproc import build_token_budgeted_evidence, structure_all_sources
from app.pipeline.utils import build_sources_manifest, evidence_hash, merge_manifest_with_registry, mission_hash

logger = logging.getLogger("tap.scorer")

TIER_DEFAULT = [
    {"min": 90, "tier": 1, "label": "Immediate Target", "color": "#0F3D3E", "key": "IMMEDIATE_TARGET",
     "action": "Assign relationship manager. Personalised CEO-to-CEO outreach within 7 days.",
     "description": "Mission-critical alignment. Fast-track partnership."},
    {"min": 80, "tier": 2, "label": "Strong Fit", "color": "#146B65", "key": "STRONG_FIT",
     "action": "Prepare full partnership pitch. Schedule discovery call.",
     "description": "High alignment. Partnership team lead — prioritise."},
    {"min": 65, "tier": 3, "label": "Conditional", "color": "#20B2AA", "key": "CONDITIONAL",
     "action": "Strengthen evidence. Identify warmest introduction path.",
     "description": "Solid signals. Needs tailored case before outreach."},
    {"min": 45, "tier": 4, "label": "Watchlist", "color": "#F5C518", "key": "WATCHLIST",
     "action": "Monitor CSR policy updates quarterly. Nurture relationship.",
     "description": "Partial alignment. Not partnership-ready yet."},
    {"min": 0, "tier": 0, "label": "Not a Target", "color": "#9CA3A3", "key": "REJECT",
     "action": "Deprioritise. Redirect effort to higher-fit companies.",
     "description": "Low fit with TAP's 21st-century skills mission."},
]

TIER_UNSCORED = {
    "tier": None, "label": "Insufficient Data", "color": "#9CA3A3", "key": "UNSCORED",
    "action": "Gather more evidence before scoring — try direct outreach to the company's India CSR office.",
    "description": "Not enough public evidence to score fit. This is not a negative signal.",
}

SCORE_BANDS = [
    {"min": 75, "key": "HIGH", "label": "Strong fit — prioritise", "color": "#146B65"},
    {"min": 40, "key": "MID", "label": "Partial fit — monitor", "color": "#F5C518"},
    {"min": 0, "key": "LOW", "label": "Low fit — deprioritise", "color": "#9CA3A3"},
]

BAND_UNSCORED = {"key": "UNSCORED", "label": "Not enough evidence to score", "color": "#9CA3A3"}


def get_scoring_tier(score, cfg: dict) -> dict:
    if score is None:
        return dict(TIER_UNSCORED)
    tiers = cfg.get("decision_tiers_v7", TIER_DEFAULT) or TIER_DEFAULT
    for tier in tiers:
        if score >= tier.get("min", 0):
            return tier
    return tiers[-1]


def score_band(score, cfg: dict) -> dict:
    if score is None:
        return dict(BAND_UNSCORED)
    bands = cfg.get("score_bands", SCORE_BANDS) or SCORE_BANDS
    for band in bands:
        if score >= band.get("min", 0):
            return band
    return bands[-1]


def determine_state(sources: list) -> str:
    tried_sources = [s for s in sources if s.get("status") != "NOT_TRIED"]
    if any(s.get("status") == "FOUND" for s in sources):
        return "FOUND"
    if len(tried_sources) >= 4:
        return "CONFIRMED_ABSENT"
    return "NOT_FOUND_IN_SOURCE"


def build_evidence_for_analysis(sources: list, company: str) -> tuple[str, list[dict]]:
    structured_sources = structure_all_sources(sources, company)
    if not structured_sources:
        return "", []
    token_budget = llm.analysis_input_token_budget()
    evidence_text = build_token_budgeted_evidence(structured_sources, company, token_budget)
    return evidence_text, structured_sources


def build_score_breakdown(analysis: dict) -> dict:
    criteria = analysis.get("criteria", [])
    average_confidence = (
        sum(c.get("confidence", 0) for c in criteria) / len(criteria) if criteria else 0
    )
    return {
        "average_confidence_pct": round(average_confidence, 1),
        "criteria_weighted": [
            {
                "id": c["id"],
                "name": c["name"],
                "score": c["score"],
                "confidence": c["confidence"],
                "evidence": c["evidence"],
                "reasoning": c["reasoning"],
                "cited_sources": extract_cited_numbers(c.get("evidence", "") + " " + c.get("reasoning", "")),
            }
            for c in criteria
        ],
    }


def resolve_decision_makers(sources: list) -> list[dict]:
    people_source = next((s for s in sources if s.get("source_name") == "people_search"), None)
    hits = (people_source or {}).get("people_hits", [])
    out = []
    seen_names = set()
    for hit in hits:
        name = (hit.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        out.append({
            "name": name,
            "title": (hit.get("title") or "").strip(),
            "company_affiliation": (hit.get("company_affiliation") or "").strip(),
            "url": (hit.get("url") or "").strip(),
            "india_location_signal": bool(hit.get("india_location_signal")),
            "is_current_csr_role": bool(hit.get("is_current_csr_role")),
            "confidence": hit.get("confidence", "LOW"),
            "source_number": hit.get("source_number"),
        })
    return out


def build_source_links(sources: list) -> list[dict]:
    labels = {
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
    out = []
    for source in sources:
        if source.get("status") == "NOT_TRIED":
            continue
        url = source.get("url", "")
        out.append({
            "label": labels.get(source.get("source_name", ""), source.get("source_name", "")),
            "url": url,
            "status": source.get("status", ""),
            "is_pdf": url.lower().endswith(".pdf"),
            "source_number": source.get("source_number"),
        })
    return out


async def gather_important_links(company: str, quota_guard=None, registry: SourceRegistry | None = None) -> list[dict]:
    if not google_search.google_search_configured_and_available(quota_guard):
        return []

    queries = [
        f'"{company}" official CSR OR sustainability page India',
        f'"{company}" India "CSR-2" OR "Form CSR-2" OR MCA filing',
        f'"{company}" annual report OR sustainability report CSR India filetype:pdf',
        f'"{company}" CSR "request for proposal" OR "open call" OR "looking for partners" India',
        f'"{company}" board of directors CSR committee India',
        f'"{company}" CSR press release India {"news"}',
    ]

    all_results: list[dict] = []
    seen_urls: set[str] = set()
    for query in queries:
        try:
            hits = await google_search.google_search_web(query, max_results=5, quota_guard=quota_guard)
        except Exception:
            hits = []
        for hit in hits:
            url = hit.get("href", "")
            if not url or url in seen_urls:
                continue
            if not is_primarily_about_company(company, hit):
                continue
            seen_urls.add(url)
            all_results.append(hit)

    if not all_results:
        return []

    try:
        selected = await llm.select_important_links(company, all_results)
    except Exception:
        selected = []

    if registry is not None:
        for link in selected:
            link["source_number"] = registry.register_child_hit(
                source_name="important_link",
                url=link.get("url", ""),
                label=link.get("label", "") or link.get("url", ""),
                excerpt=link.get("relevance", ""),
            )

    return selected


def is_primarily_about_company(company: str, hit: dict) -> bool:
    title = hit.get("title", "")
    body = hit.get("body", "")
    if not mentions_company(company, f"{title} {body}"):
        return False
    if mentions_company(company, title):
        return True
    url = hit.get("href", "")
    tokens = [t for t in company.lower().split() if len(t) > 2]
    if tokens and any(token in url.lower() for token in tokens):
        return True
    return False


async def resolve_logo(company: str, sources: list, cfg: dict, quota_guard=None) -> str:
    search_cfg = cfg.get("search_source_toggles", {})
    try:
        return await logo.resolve_company_logo_url(company, search_cfg, quota_guard, sources)
    except Exception as exc:
        logger.warning("resolve_logo failed company=%r error=%s", company, exc)
        return ""


def _unscored_result(state: str, insight: str, sources: list, source_links: list, logo_url: str,
                      registry: SourceRegistry, decision_makers: list | None = None) -> dict:
    return {
        "state": state,
        "fit_score": None,
        "strategic_insight": insight,
        "band": dict(BAND_UNSCORED),
        "scoring_tier": dict(TIER_UNSCORED),
        "analysis": None,
        "score_breakdown": {},
        "decision_makers": decision_makers or [],
        "sources": sources,
        "source_links": source_links,
        "important_links": [],
        "logo_url": logo_url,
        "source_bank": registry.as_source_bank(),
    }


async def score(company: str, sources: list, cfg: dict, quota_guard=None,
                 registry: SourceRegistry | None = None) -> dict:
    registry = registry or SourceRegistry(company)
    for source in sources:
        if source.get("status") == "FOUND" and not source.get("source_number"):
            registry.register_core_source(source)

    state = determine_state(sources)
    logger.info("score START company=%r state=%s source_bank_size=%d", company, state, len(registry.entries()))

    source_links = build_source_links(sources)
    logo_url = await resolve_logo(company, sources, cfg, quota_guard)

    mission = cfg.get("org_mission") or llm.DEFAULT_MISSION
    sources_manifest = merge_manifest_with_registry(build_sources_manifest(sources), registry)
    relevant_evidence_preview, _ = build_evidence_for_analysis(sources, company)

    if state == "CONFIRMED_ABSENT" and not relevant_evidence_preview.strip():
        logger.info("score UNSCORED company=%r reason=confirmed_absent no_anthropic_calls", company)
        insight = (
            f"No publicly available India CSR data was found for {company} across the sources checked. "
            "This does not mean {company} is a poor fit — it may simply mean their CSR activity isn't "
            "publicly documented, or it sits behind channels this search doesn't reach. "
            "Recommended: direct outreach to their India CSR office to confirm fit before deprioritising."
        ).format(company=company)
        return _unscored_result(state, insight, sources, source_links, logo_url, registry)

    analysis = None
    if relevant_evidence_preview.strip():
        try:
            analysis = await llm.analyze_company(company, mission, sources, sources_manifest)
        except Exception as exc:
            logger.error("score analyze_company raised company=%r error=%s", company, exc)
            analysis = None
    else:
        logger.info("score UNSCORED company=%r reason=no_relevant_evidence no_anthropic_calls", company)

    valid_numbers = {entry["number"] for entry in registry.entries()}

    if not analysis:
        cooldown_remaining = llm.anthropic_cooldown_remaining_seconds()
        if cooldown_remaining > 0:
            insight = (
                f"{llm.LLM_UNAVAILABLE_EVIDENCE} — Anthropic rate limit is active, "
                f"try again in about {int(cooldown_remaining // 60)}m {int(cooldown_remaining % 60)}s. "
                "This is a temporary infrastructure gap, not a reflection of the company's fit."
            )
            logger.warning(
                "score UNSCORED company=%r reason=anthropic_cooldown_active seconds_left=%.0f",
                company, cooldown_remaining,
            )
        else:
            insight = (
                f"{llm.LLM_UNAVAILABLE_EVIDENCE} This is a temporary gap in evidence processing, "
                "not a reflection of the company's fit — re-run scoring once evidence is available."
            )
            logger.warning("score UNSCORED company=%r reason=no_analysis", company)
        return _unscored_result(state, insight, sources, source_links, logo_url, registry, resolve_decision_makers(sources))

    final_score = int(round(min(max(analysis.get("fit_score", 0), 0), 100)))
    breakdown = build_score_breakdown(analysis)
    tier = get_scoring_tier(final_score, cfg)

    logger.info(
        "score eligibility company=%r plausibly_mandated=%s routed_through_group=%s spend_trend=%s",
        company,
        (analysis.get("eligibility") or {}).get("plausibly_mandated"),
        (analysis.get("group_foundation") or {}).get("routed_through_group"),
        (analysis.get("spend") or {}).get("trend_direction"),
    )

    try:
        insight = await llm.generate_strategic_insight_narrative(company, mission, state, final_score, tier["label"], analysis)
        insight = strip_unknown_citation_tokens(insight, valid_numbers)
    except Exception as exc:
        logger.error("score strategic_insight raised company=%r error=%s", company, exc)
        insight = (
            f"{llm.LLM_UNAVAILABLE_EVIDENCE} A numeric fit score was generated from available evidence, "
            "but the narrative summary could not be produced this run."
        )

    decision_makers = resolve_decision_makers(sources)

    try:
        important_links = await gather_important_links(company, quota_guard=quota_guard, registry=registry)
    except Exception as exc:
        logger.error("score gather_important_links raised company=%r error=%s", company, exc)
        important_links = []

    logger.info(
        "score DONE company=%r fit_score=%d tier=%s source_bank_size=%d",
        company, final_score, tier.get("label"), len(registry.entries()),
    )

    return {
        "state": state,
        "fit_score": final_score,
        "strategic_insight": insight,
        "band": score_band(final_score, cfg),
        "scoring_tier": tier,
        "analysis": analysis,
        "score_breakdown": breakdown,
        "decision_makers": decision_makers,
        "sources": sources,
        "source_links": source_links,
        "important_links": important_links,
        "logo_url": logo_url,
        "source_bank": registry.as_source_bank(),
        "cache_key": (company.strip().lower(), evidence_hash(sources), mission_hash(mission)),
    }