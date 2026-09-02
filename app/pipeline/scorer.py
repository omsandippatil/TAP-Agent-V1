import logging
import re

from app.pipeline import google_search, llm, logo
from app.pipeline.source_registry import SourceRegistry, extract_cited_numbers, strip_unknown_citation_tokens
from app.pipeline.textproc import clean_and_budget_sources
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

IMPORTANT_LINK_QUERIES = (
    '"{company}" official CSR OR sustainability page India',
    '"{company}" India "CSR-2" OR "Form CSR-2" OR MCA filing',
    '"{company}" annual report OR sustainability report CSR India filetype:pdf',
    '"{company}" CSR "request for proposal" OR "open call" OR "looking for partners" India',
)
MAX_IMPORTANT_LINKS = 8
IMPORTANT_LINK_SEARCH_RESULTS = 5

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

_COMPANY_SUFFIX_PATTERN = re.compile(
    r"\b(private\s+limited|pvt\.?\s*ltd\.?|limited|ltd\.?|inc\.?|incorporated|corp\.?|"
    r"corporation|llp|india)\b",
    re.IGNORECASE,
)


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


def _normalize_company_name(name: str) -> str:
    cleaned = _COMPANY_SUFFIX_PATTERN.sub("", name or "")
    return re.sub(r"[^a-z0-9]", "", cleaned.lower())


def is_existing_tap_partner(company: str, cfg: dict) -> bool:
    existing_partners = cfg.get("tap_existing_partners", []) or []
    normalized_company = _normalize_company_name(company)
    if not normalized_company:
        return False
    for partner in existing_partners:
        normalized_partner = _normalize_company_name(partner)
        if normalized_partner and (normalized_partner == normalized_company or normalized_partner in normalized_company):
            return True
    return False


def determine_state(sources: list) -> str:
    tried_sources = [s for s in sources if s.get("status") != "NOT_TRIED"]
    if any(s.get("status") == "FOUND" for s in sources):
        return "FOUND"
    if len(tried_sources) >= 4:
        return "CONFIRMED_ABSENT"
    return "NOT_FOUND_IN_SOURCE"


def build_score_breakdown(analysis: dict) -> dict:
    criteria = analysis.get("criteria", [])
    average_confidence = (
        sum(c.get("confidence", 0) for c in criteria) / len(criteria) if criteria else 0
    )
    weighted_confidence = llm.weighted_average_criteria_confidence(criteria)
    return {
        "average_confidence_pct": round(average_confidence, 1),
        "weighted_confidence_pct": round(weighted_confidence, 1),
        "criteria_weighted": [
            {
                "id": c.get("id", ""),
                "name": c.get("name") or llm.CRITERIA_TITLES.get(c.get("id", ""), c.get("id", "")),
                "score": c.get("score", 0),
                "confidence": c.get("confidence", 0),
                "evidence": c.get("evidence", ""),
                "reasoning": c.get("reasoning", ""),
                "cited_sources": extract_cited_numbers(c.get("evidence", "") + " " + c.get("reasoning", "")),
            }
            for c in criteria
        ],
    }


def build_source_links(sources: list) -> list[dict]:
    out = []
    for source in sources:
        if source.get("status") == "NOT_TRIED":
            continue
        url = source.get("url", "")
        out.append({
            "label": SOURCE_LABELS.get(source.get("source_name", ""), source.get("source_name", "")),
            "url": url,
            "status": source.get("status", ""),
            "is_pdf": url.lower().endswith(".pdf"),
            "source_number": source.get("source_number"),
        })
    return out


def _name_key(name: str) -> str:
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def attach_linkedin_urls(decision_makers: list[dict], sources: list) -> list[dict]:
    people_source = next((s for s in sources if s.get("source_name") == "people_search"), None)
    hits = (people_source or {}).get("people_hits", [])
    if not hits:
        return decision_makers

    hits_by_name = {}
    for hit in hits:
        key = _name_key(hit.get("name"))
        url = (hit.get("url") or "").strip()
        if key and url and key not in hits_by_name:
            hits_by_name[key] = url

    for person in decision_makers:
        if person.get("linkedin_url"):
            continue
        key = _name_key(person.get("name"))
        matched_url = hits_by_name.get(key)
        if not matched_url:
            for candidate_key, candidate_url in hits_by_name.items():
                if key and (key in candidate_key or candidate_key in key):
                    matched_url = candidate_url
                    break
        if matched_url:
            person["linkedin_url"] = matched_url
    return decision_makers


async def gather_important_links(company: str, quota_guard=None, registry: SourceRegistry | None = None) -> list[dict]:
    if not google_search.google_search_configured_and_available(quota_guard):
        return []

    seen_urls: set[str] = set()
    results: list[dict] = []
    for query_template in IMPORTANT_LINK_QUERIES:
        query = query_template.format(company=company)
        try:
            hits = await google_search.google_search_web(query, max_results=IMPORTANT_LINK_SEARCH_RESULTS, quota_guard=quota_guard)
        except Exception:
            hits = []
        for hit in hits:
            url = hit.get("href", "")
            title = hit.get("title", "")
            body = hit.get("body", "")
            if not url or url in seen_urls:
                continue
            if company.lower() not in f"{title} {body}".lower():
                continue
            seen_urls.add(url)
            results.append({"label": title[:80] or url, "url": url, "relevance": body[:140]})
            if len(results) >= MAX_IMPORTANT_LINKS:
                break
        if len(results) >= MAX_IMPORTANT_LINKS:
            break

    if registry is not None:
        for link in results:
            link["source_number"] = registry.register_child_hit(
                source_name="important_link",
                url=link.get("url", ""),
                label=link.get("label", "") or link.get("url", ""),
                excerpt=link.get("relevance", ""),
            )
    return results


async def resolve_logo(company: str, sources: list, cfg: dict, quota_guard=None) -> str:
    search_cfg = cfg.get("search_source_toggles", {})
    try:
        return await logo.resolve_company_logo_url(company, search_cfg, quota_guard, sources)
    except Exception as exc:
        logger.warning("resolve_logo failed company=%r error=%s", company, exc)
        return ""


def _unscored_result(state: str, insight: str, sources: list, source_links: list, logo_url: str,
                      registry: SourceRegistry, existing_partner: bool = False,
                      analysis: dict | None = None, score_breakdown: dict | None = None,
                      decision_makers: list | None = None) -> dict:
    return {
        "state": state,
        "fit_score": None,
        "strategic_insight": insight,
        "band": dict(BAND_UNSCORED),
        "scoring_tier": dict(TIER_UNSCORED),
        "analysis": analysis,
        "score_breakdown": score_breakdown or {},
        "decision_makers": decision_makers or [],
        "sources": sources,
        "source_links": source_links,
        "important_links": [],
        "logo_url": logo_url,
        "is_existing_tap_partner": existing_partner,
        "source_bank": registry.as_source_bank(),
    }


def _existing_partner_prefix(company: str, note: str) -> str:
    return f"**Existing TAP partner** — {company} is on TAP's active donor/partner list. {note} "


async def score(company: str, sources: list, cfg: dict, quota_guard=None,
                 registry: SourceRegistry | None = None, mode: str = "deep") -> dict:
    registry = registry or SourceRegistry(company)
    for source in sources:
        if source.get("status") == "FOUND" and not source.get("source_number"):
            registry.register_core_source(source)

    state = determine_state(sources)
    logger.info("score START company=%r mode=%r state=%s source_bank_size=%d", company, mode, state, len(registry.entries()))

    existing_partner = is_existing_tap_partner(company, cfg)
    source_links = build_source_links(sources)
    logo_url = await resolve_logo(company, sources, cfg, quota_guard)

    mission = cfg.get("org_mission") or llm.DEFAULT_MISSION
    sources_manifest = merge_manifest_with_registry(build_sources_manifest(sources), registry)

    found_count = sum(1 for s in sources if s.get("status") == "FOUND")
    if found_count == 0:
        insight = (
            f"No publicly available India CSR data was found for {company} across the sources "
            "checked. This does not mean the company is a poor fit — it may simply mean their "
            "CSR activity isn't publicly documented, or it sits behind channels this search "
            "doesn't reach. Recommended: direct outreach to their India CSR office to confirm "
            "fit before deprioritising."
        )
        if existing_partner:
            insight = _existing_partner_prefix(
                company, "No public CSR evidence surfaced in this run, but this must never be "
                "read as 'Not a Target' — follow up internally rather than deprioritising."
            ) + insight
        logger.info("score UNSCORED company=%r mode=%r reason=no_sources_found no_anthropic_call", company, mode)
        return _unscored_result(state, insight, sources, source_links, logo_url, registry, existing_partner)

    cleaned_sources = clean_and_budget_sources(
        sources, llm.evidence_token_budget(company, mission, sources_manifest)
    )

    analysis = await llm.analyze_and_score_company(company, mission, cleaned_sources, sources_manifest)

    if not analysis:
        cooldown_remaining = llm.anthropic_cooldown_remaining_seconds()
        if cooldown_remaining > 0:
            insight = (
                f"{llm.LLM_UNAVAILABLE_EVIDENCE} — Anthropic rate limit is active, try again in "
                f"about {int(cooldown_remaining // 60)}m {int(cooldown_remaining % 60)}s. This is "
                "a temporary infrastructure gap, not a reflection of the company's fit."
            )
        else:
            insight = (
                f"{llm.LLM_UNAVAILABLE_EVIDENCE} This is a temporary gap in evidence processing, "
                "not a reflection of the company's fit — re-run scoring once evidence is available."
            )
        if existing_partner:
            insight = _existing_partner_prefix(
                company, "Scoring could not run this time, but this must never be read as "
                "'Not a Target'."
            ) + insight
        logger.warning("score UNSCORED company=%r mode=%r reason=analysis_call_failed", company, mode)
        return _unscored_result(state, insight, sources, source_links, logo_url, registry, existing_partner)

    if analysis.get("evidence_coverage_insufficient"):
        coverage_reason = analysis.get("evidence_coverage_reason", "").strip()
        avg_conf = analysis.get("average_criteria_confidence_pct", 0)
        weighted_conf = analysis.get("weighted_criteria_confidence_pct", avg_conf)
        authenticity = analysis.get("overall_authenticity_score", 0)
        reason_clause = coverage_reason or "coverage was too thin to score fit confidently."
        insight = (
            f"Evidence coverage for {company} was too thin to score fit confidently: "
            f"{reason_clause} (weighted criteria confidence: {weighted_conf:.0f}%, source "
            f"authenticity: {authenticity}%). This means the research pass did not retrieve "
            f"enough public evidence to judge fit either way — it is **not** a Low Fit "
            f"determination. Recommended: broaden the manual search or reach out directly to "
            f"the company's India CSR office before deprioritising."
        )
        if existing_partner:
            insight = _existing_partner_prefix(
                company, "Evidence coverage was too thin to score confidently in this "
                "run, but this must never be read as 'Not a Target'."
            ) + insight
        logger.warning(
            "score UNSCORED company=%r mode=%r reason=evidence_coverage_insufficient "
            "coverage_reason=%r avg_confidence=%.1f weighted_confidence=%.1f authenticity=%d",
            company, mode, coverage_reason, avg_conf, weighted_conf, authenticity,
        )
        decision_makers = attach_linkedin_urls(list(analysis.get("decision_makers", [])), sources)
        return _unscored_result(
            state, insight, sources, source_links, logo_url, registry, existing_partner,
            analysis=analysis,
            score_breakdown=build_score_breakdown(analysis),
            decision_makers=decision_makers,
        )

    final_score = analysis["fit_score"]
    tier = get_scoring_tier(final_score, cfg)
    band = score_band(final_score, cfg)
    breakdown = build_score_breakdown(analysis)

    valid_numbers = {entry["number"] for entry in registry.entries()}
    insight = strip_unknown_citation_tokens(analysis.get("strategic_insight", ""), valid_numbers)
    if existing_partner:
        insight = _existing_partner_prefix(
            company, "This score reflects that established relationship."
        ) + insight

    decision_makers = attach_linkedin_urls(list(analysis.get("decision_makers", [])), sources)

    try:
        important_links = await gather_important_links(company, quota_guard=quota_guard, registry=registry)
    except Exception as exc:
        logger.error("score gather_important_links raised company=%r error=%s", company, exc)
        important_links = []

    logger.info(
        "score DONE company=%r mode=%r fit_score=%d tier=%s source_bank_size=%d",
        company, mode, final_score, tier.get("label"), len(registry.entries()),
    )

    return {
        "state": state,
        "fit_score": final_score,
        "strategic_insight": insight,
        "band": band,
        "scoring_tier": tier,
        "analysis": analysis,
        "score_breakdown": breakdown,
        "decision_makers": decision_makers,
        "sources": sources,
        "source_links": source_links,
        "important_links": important_links,
        "logo_url": logo_url,
        "source_bank": registry.as_source_bank(),
        "is_existing_tap_partner": existing_partner,
        "cache_key": (mode, company.strip().lower(), evidence_hash(sources), mission_hash(mission)),
    }