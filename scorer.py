# scorer.py — v11: 6-dimension scoring engine + LLM semantic alignment (Groq)
# Calibrated so companies that SUPPORT education (funding programmes/NGOs)
# score properly, not just education-first companies.
# Weighted dims: Focus (45) + Adjacency (25) + Geography (10) +
#                CSR Maturity (10) + Budget (5) + Source Quality (5) = 100
import os, re, yaml
from utils import all_sources_tried, any_source_found, best_source_quality, combine_source_texts

# Semantic scoring (Groq LLM) — lazy import so the tool works without it
try:
    from llm import semantic_alignment
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False
    def semantic_alignment(*a, **kw): return None

_DEFAULT_MISSION = (
    "The Apprentice Project (TAP) develops 21st-century skills (critical thinking, "
    "creativity, confidence, communication, problem-solving, self-awareness, "
    "financial literacy) for low-income middle and high school students in India, "
    "delivered through TAP Buddy — an AI-powered WhatsApp chatbot with video "
    "electives (Coding, Science, Visual Arts, Financial Literacy). TAP works "
    "exclusively in government schools with partners like MCD, DoE Delhi, BMC "
    "Mumbai and SCERT Maharashtra. TAP does NOT run vocational training or job "
    "placement."
)


def _kw_in(text_lower: str, kw: str) -> bool:
    """Word-boundary keyword check (no substring false positives)."""
    return re.search(r"(?<![a-z0-9])" + re.escape(kw.lower()) + r"(?![a-z0-9])",
                     text_lower) is not None


def _cfg():
    p = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _score_focus(focus_facts: list, cfg: dict, adj_fired: list = None) -> tuple:
    """
    Calibrated scoring (v11, education-supporter aware).
      - strongest match dominates (70%), average fills in (30%)
      - softened depth penalty: 1 match→×0.75, 2→×0.9, 3+→×1.0
      - adjacency-confirmed uplift: companies whose education involvement is
        indirect (generic keywords, but 2+ adjacency clusters firing) get a
        floor on the core score — funding/supporting education programmes
        counts as real alignment even when education is not their main focus
      - generic cap only for weak matches (best < 55) with zero adjacency
      - no keyword match at all + 2+ adjacency clusters → partial credit
        instead of 0 (indirect education supporters are no longer zeroed out)
    """
    tap_focus = cfg.get("tap_focus_areas", {})
    adj_fired = adj_fired or []
    n_adj = len(adj_fired)
    if not focus_facts and n_adj < 2:
        return 0, [], False
    weights, matched = [], []
    tap_lower = {k.lower(): v for k, v in tap_focus.items()}
    for fact in focus_facts or []:
        kw = fact.get("value", "").lower()
        pts = tap_lower.get(kw)
        if pts is None:
            for k, v in tap_lower.items():
                if k in kw or kw in k:
                    pts = v
                    break
        if pts is not None:
            weights.append(pts)
            matched.append(f"{fact['value']} ({pts}pts)")
    if not weights:
        if n_adj >= 2:
            # Indirect education supporter: no direct keyword match, but
            # multiple adjacency clusters confirm education engagement
            core = min(45 + 8 * n_adj, 72)
            matched = [f"education supporter (via {n_adj} adjacency clusters)"]
            return min(round(40 * core / 100), 40), matched, True
        return 0, [], False
    weights.sort(reverse=True)
    best  = weights[0]
    avg   = sum(weights) / len(weights)
    core  = 0.7 * best + 0.3 * avg
    # adjacency-confirmed uplift: multiple adjacent education programmes
    # confirm genuine engagement even when focus keywords are generic.
    # If the uplift fires, alignment is INDIRECT → final score capped below
    # the Immediate Target tier (see indirect_alignment_cap).
    indirect = False
    if n_adj >= 3 and core < 82:
        core, indirect = 82, True
    elif n_adj == 2 and core < 72:
        core, indirect = 72, True
    effective_depth = len(weights) + n_adj
    depth = 0.75 if effective_depth == 1 else 0.9 if effective_depth == 2 else 1.0
    score = round(40 * core / 100 * depth)
    # generic cap only for weak matches with NO confirming adjacency
    if best < 55 and n_adj == 0:
        score = min(score, 15)
    return min(score, 40), matched, indirect


def _score_adjacency(adj_signals: dict, cfg: dict) -> tuple:
    max_boost = cfg.get("max_adjacency_boost", 20)
    fired, total = [], 0
    for cid, info in sorted(adj_signals.items(),
                             key=lambda x: x[1].get("boost", 0), reverse=True):
        if info.get("fires") and total < max_boost:
            boost = min(info["boost"], max_boost - total)
            total += boost
            fired.append({
                "id": cid, "label": info["label"],
                "keywords_found": info["keywords_found"],
                "tap_reasoning":  info["tap_reasoning"],
                "boost_applied":  boost,
                "evidence_excerpts": info.get("evidence_excerpts", []),
            })
    return min(total, max_boost), fired


def _score_geography(geography: list, cfg: dict) -> tuple:
    geo_scores = cfg.get("geo_scores", {})
    tap_states = set(s.lower() for s in cfg.get("tap_states", []))
    geo_lower  = set(g.get("place", "").lower() for g in geography)
    overlap    = geo_lower & tap_states
    if len(overlap) >= 3:
        return geo_scores.get("tap_state_count_3plus", 10), \
               f"{len(overlap)} TAP-presence states"
    if len(overlap) >= 1:
        return geo_scores.get("tap_state_count_1_2", 7), \
               f"TAP state(s): {', '.join(sorted(overlap)).title()}"
    if geo_lower:
        return geo_scores.get("other_india_only", 4), \
               f"India presence ({', '.join(list(geo_lower)[:3]).title()})"
    return geo_scores.get("india_mentioned_only", 2), "India mentioned"


def _score_maturity(sources: list, cfg: dict) -> tuple:
    signals_cfg = cfg.get("maturity_signals", {})
    cap  = cfg.get("maturity_cap", 10)
    text = combine_source_texts(sources).lower()
    total, found = 0, []
    for sig, pts in signals_cfg.items():
        if _kw_in(text, sig):
            total += pts
            found.append(sig)
        if total >= cap:
            break
    return min(total, cap), found


def _score_budget(spend: dict, cfg: dict) -> tuple:
    tiers         = cfg.get("india_budget_tiers", [])
    unknown_score = cfg.get("budget_unknown_score", 5)
    inr = spend.get("inr_crore")
    if inr is None:
        return unknown_score, "Not publicly disclosed (neutral 5)"
    for tier in sorted(tiers, key=lambda t: t["min_crore"], reverse=True):
        if inr >= tier["min_crore"]:
            return tier["score"], f"{spend['display']} — {tier.get('label', '')}"
    return 3, f"{spend['display']}"


def _education_dominance(sources: list, cfg: dict) -> tuple:
    """
    Returns (multiplier, education_share). Measures how central education is
    in the CSR evidence vs other domains (health, environment, road safety…).
    A health-dominant portfolio with stray education mentions gets its
    education-related dimensions dampened — proportion matters, not just
    keyword presence.
    """
    dd = cfg.get("domain_dominance", {})
    edu_terms   = dd.get("education_terms", [])
    other_terms = dd.get("other_domain_terms", [])
    if not edu_terms or not other_terms:
        return 1.0, None
    text = combine_source_texts(sources).lower()

    def _count(terms):
        total = 0
        for t in terms:
            total += len(re.findall(
                r"(?<![a-z0-9])" + re.escape(t.lower()) + r"(?![a-z0-9])", text))
        return total

    edu, oth = _count(edu_terms), _count(other_terms)
    if edu + oth == 0:
        return 1.0, None
    share = edu / (edu + oth)
    bands = sorted(dd.get("bands", []),
                   key=lambda b: b.get("min_share", 0), reverse=True)
    for band in bands:
        if share >= band.get("min_share", 0):
            return float(band.get("multiplier", 1.0)), round(share, 2)
    return 1.0, round(share, 2)


def _score_partner_similarity(sources: list, cfg: dict) -> tuple:
    """
    Scores similarity to TAP's ACTUAL partners (native 0-15).
    Traits derived from researched CSR profiles of real TAP partners
    (Tenerity, Capgemini, Bajaj Finserv, NRB Bearings, SVP India, Project
    Tech4Dev) — see partner_exemplar_traits in config.yaml.
    """
    traits = cfg.get("partner_exemplar_traits", {})
    cap    = cfg.get("partner_similarity_cap", 15)
    text   = combine_source_texts(sources).lower()
    total, matched = 0, []
    for tid, t in traits.items():
        kws_found = [kw for kw in t.get("keywords", []) if _kw_in(text, str(kw))]
        if kws_found:
            total += t.get("points", 2)
            matched.append({"id": tid, "label": t.get("label", tid),
                            "keywords_found": kws_found[:4],
                            "evidence_from": t.get("evidence_from", "").strip()})
    return min(total, cap), matched


def _score_source_quality(sources: list, cfg: dict) -> tuple:
    q_cfg = dict(cfg.get("source_quality", {}))
    q_cfg.setdefault("annual_report", q_cfg.get("global_annual_report", 6))
    return best_source_quality(sources, q_cfg)


def get_scoring_tier(score: int) -> dict:
    """
    5-tier TAP decision framework (v9).
    Tier 1 = Immediate Target (90+)
    Tier 2 = Strong Fit (80–89)
    Tier 3 = Conditional (65–79)
    Tier 4 = Watchlist (50–64)
    Tier 0 = Not a Target (<50)
    """
    _TIERS = [
        {"min": 90, "tier": 1, "label": "Immediate Target",
         "color": "#7C3AED", "key": "IMMEDIATE_TARGET",
         "action": "Assign relationship manager. Personalised CEO-to-CEO outreach within 7 days.",
         "description": "Mission-critical alignment. Fast-track partnership."},
        {"min": 80, "tier": 2, "label": "Strong Fit",
         "color": "#16A34A", "key": "STRONG_FIT",
         "action": "Prepare full partnership pitch. Schedule discovery call.",
         "description": "High alignment. Partnership team lead — prioritise."},
        {"min": 65, "tier": 3, "label": "Conditional",
         "color": "#0EA5E9", "key": "CONDITIONAL",
         "action": "Strengthen evidence. Identify warmest introduction path.",
         "description": "Solid signals. Needs tailored case before outreach."},
        {"min": 50, "tier": 4, "label": "Watchlist",
         "color": "#D97706", "key": "WATCHLIST",
         "action": "Monitor CSR policy updates quarterly. Nurture relationship.",
         "description": "Partial alignment. Not partnership-ready yet."},
        {"min": 0,  "tier": 0, "label": "Not a Target",
         "color": "#DC2626", "key": "REJECT",
         "action": "Deprioritise. Redirect effort to higher-fit companies.",
         "description": "Low fit with TAP's 21st-century skills mission."},
    ]
    cfg = _cfg()
    tiers = cfg.get("decision_tiers_v7", _TIERS) or _TIERS
    for t in tiers:
        if score >= t.get("min", 0):
            return t
    return _TIERS[-1]


def apply_strict_penalties(total: int, parsed: dict, cfg: dict) -> tuple:
    """
    Apply penalties for structural misalignment.
    IMPORTANT policy (v9):
      - no_school_level → 0 deduction (indirect alignment is still valid)
      - higher_education_only → cap at 70 (not TAP's model, but note only)
      - employability_only → cap at 60 (vocational ≠ 21CS skills)
      - infra_donations_only → -15 (cash/infra gives, not programme partnership)
    """
    penalties_cfg = cfg.get("penalties", {})
    applied = []
    ai_flags = parsed.get("ai_flags", {})

    # Higher-ed only (university focus with no school component)
    if ai_flags.get("is_higher_ed_only") or parsed.get("is_higher_ed_only"):
        cap = penalties_cfg.get("higher_education_only", {}).get("cap", 70)
        if total > cap:
            applied.append({"reason": "Higher-education only CSR (no school-level)",
                             "deduction": total - cap, "cap_applied": cap})
            total = cap

    # Employability/vocational detection — rule-based, so it works even
    # without the LLM (Diageo "Learning for Life" fix): if vocational terms
    # outweigh school-level terms, this is downstream employability CSR,
    # which TAP explicitly does not do.
    text = combine_source_texts(parsed.get("_sources", [])).lower()
    voc_terms    = ["vocational", "employability", "job readiness", "placement",
                    "apprenticeship", "hospitality", "workforce",
                    "job training", "livelihood training"]
    school_terms = ["school", "schools", "classroom", "teacher", "teachers",
                    "government school", "middle school", "secondary school"]
    voc_hits    = sum(1 for t in voc_terms    if _kw_in(text, t))
    school_hits = sum(1 for t in school_terms if _kw_in(text, t))
    rule_employability = voc_hits >= 2 and voc_hits > school_hits

    # Employability/vocational only (not TAP's mission)
    if (ai_flags.get("is_employability_only")
            or parsed.get("is_employability_only")
            or rule_employability):
        cap = penalties_cfg.get("employability_only", {}).get("cap", 60)
        if total > cap:
            reason = "Employability/vocational only (not 21CS skills)"
            if rule_employability:
                reason += (f" — rule-based: {voc_hits} vocational vs "
                           f"{school_hits} school-level signals")
            applied.append({"reason": reason,
                             "deduction": total - cap, "cap_applied": cap})
            total = cap

    # Infrastructure / cash donations only (no programme partnership potential)
    infra_signals = ["school building", "toilet construction", "infrastructure grant",
                     "sanitation", "clean water", "drinking water facility"]
    prog_signals  = ["programme", "program", "curriculum", "training", "workshop",
                     "mentoring", "coaching", "digital", "skill", "learning"]
    infra_hit  = sum(1 for s in infra_signals if _kw_in(text, s))
    prog_hit   = sum(1 for s in prog_signals  if _kw_in(text, s))
    if infra_hit >= 2 and prog_hit == 0:
        ded = penalties_cfg.get("infra_donations_only", {}).get("deduction", 15)
        applied.append({"reason": "Infrastructure/cash donations only — no programme angle",
                        "deduction": ded})
        total = max(0, total - ded)

    # NOTE: no_school_level is intentionally NOT penalised (indirect alignment is valid)

    return total, applied


def _score_delivery_model_fit(parsed: dict, cfg: dict) -> tuple:
    """
    Scores the CSR delivery model:
      HYBRID (funds NGOs + runs own) = 15  ← best partnership potential
      FUNDER (grants to NGOs)        = 12  ← very high value — pitch TAP as grantee
      IMPLEMENTER (in-house only)    = 6   ← harder sell, needs different angle
      UNCLEAR                        = 4   ← incomplete data
    Note: FUNDER scores high. Being a pure funder is HIGHLY VALUABLE for TAP.
    """
    dm_scores = cfg.get("delivery_model_scores",
                        {"HYBRID": 15, "FUNDER": 12, "IMPLEMENTER": 6, "UNCLEAR": 4})
    dm = parsed.get("csr_delivery_model", {})
    model = (dm.get("model") or "UNCLEAR").upper()
    score = dm_scores.get(model, 4)
    note  = dm.get("note", "")
    return score, model, note


def _score_csr_depth(parsed: dict, cfg: dict) -> tuple:
    """
    Scores depth / breadth of CSR programme evidence (0–20).
    Evaluates: number of distinct programmes, budget transparency, annual
    reporting quality, MCA filing completeness.
    """
    depth = 0
    signals = []

    progs = parsed.get("programs", [])
    if len(progs) >= 5:
        depth += 8; signals.append(f"{len(progs)} distinct programmes")
    elif len(progs) >= 2:
        depth += 5; signals.append(f"{len(progs)} programmes")
    elif len(progs) == 1:
        depth += 2; signals.append("1 programme found")

    spend = parsed.get("spend", {})
    if spend.get("inr_crore") and spend["inr_crore"] > 0:
        depth += 4; signals.append("India CSR spend publicly disclosed")

    maturity = parsed.get("maturity_signals", [])
    if "csr committee" in " ".join(maturity).lower():
        depth += 3; signals.append("CSR committee mentioned")
    if "annual report" in " ".join(maturity).lower():
        depth += 3; signals.append("Annual report CSR section")
    if "mca" in " ".join(maturity).lower() or len(maturity) >= 3:
        depth += 2; signals.append("MCA filing / structured reporting")

    return min(depth, 20), signals


def _score_evidence_strength(sources: list, cfg: dict) -> tuple:
    """
    Scores the quality and diversity of evidence (0–10).
    Primary sources (annual report, company CSR page, MCA) score highest.
    Multiple corroborating sources increase confidence.
    """
    found_sources = [s for s in sources if s.get("status") == "FOUND"]
    primary = [s for s in found_sources
               if s.get("source_type","") in
               ("company_csr_page","annual_report","mca_portal","national_csr_portal")]
    score = 0
    if len(primary) >= 3:
        score = 10
    elif len(primary) == 2:
        score = 8
    elif len(primary) == 1:
        score = 5
    elif found_sources:
        score = 3
    label = f"{len(found_sources)} sources found ({len(primary)} primary)"
    return score, label


def _score_strategic_fit(parsed: dict, cfg: dict) -> tuple:
    """
    Bonus for direct TAP-signal factors (0–5):
      +2 has_govt_school_work (MCD/DoE/BMC partner fit)
      +2 has_adolescent_focus (target demographic match)
      +1 geographic overlap in TAP delivery states
    """
    ai_flags = parsed.get("ai_flags", {})
    score, signals = 0, []
    if ai_flags.get("has_govt_school_work"):
        score += 2; signals.append("Govt school work (MCD/DoE/BMC alignment)")
    if ai_flags.get("has_adolescent_focus"):
        score += 2; signals.append("Adolescent/youth focus")
    geo = [g.get("place","").lower() for g in parsed.get("geography",[])]
    tap_states = set(s.lower() for s in cfg.get("tap_states",[]))
    if set(geo) & tap_states:
        score += 1; signals.append("Active in TAP delivery state(s)")
    return min(score, 5), signals


def score_band(fit: int, cfg: dict = None) -> dict:
    """Returns the scoring-gradient band for a fit score (strict, config-driven)."""
    cfg = cfg or _cfg()
    for band in cfg.get("score_bands", []):
        if fit >= band.get("min", 0):
            return band
    return {"min": 0, "key": "LOW", "label": "Low fit — deprioritise",
            "color": "#DC2626"}


def determine_state(sources: list) -> str:
    tried = [s for s in sources if s.get("status") != "NOT_TRIED"]
    if any(s.get("status") == "FOUND" for s in sources):
        return "FOUND"
    if len(tried) >= 4:
        return "CONFIRMED_ABSENT"
    if len(tried) > 0:
        return "NOT_FOUND_IN_SOURCE"
    return "NOT_FOUND_IN_SOURCE"


def generate_strategic_insight(company, state, focus_facts, adj_fired, geography,
                                fit_score, csr_vertical=None, sources=None,
                                delivery=None):
    """
    Rule-based strategic insight (fallback when Claude AI is unavailable).
    csr_vertical: alias for delivery model (accepts both param names).
    """
    delivery = delivery or csr_vertical   # accept either keyword
    lines = []
    focus_vals   = [f.get("value", "") for f in focus_facts]
    fired_labels = [c["label"] for c in adj_fired]
    geo_places   = [g.get("place", "") for g in geography][:3]

    if state == "CONFIRMED_ABSENT":
        return (
            f"{company} has no publicly available India CSR data across four sources. "
            f"This may indicate no India CSR obligation or undisclosed programmes. "
            f"Recommended: direct outreach to their India office."
        )

    if focus_vals:
        lines.append(
            f"{company} shows CSR alignment with TAP's 21st-century skills mission "
            f"(TAP Buddy: life skills, coding, financial literacy for MCD/DoE/BMC "
            f"middle and high-school students), with evidence in: "
            f"{', '.join(focus_vals[:4])}."
        )
    elif fired_labels:
        lines.append(
            f"{company} does not currently fund programmes exactly matching TAP's "
            f"model, but shows investment in adjacent areas: "
            f"{', '.join(fired_labels[:3])}."
        )
    else:
        lines.append(
            f"Limited public CSR data was found for {company}. "
            f"Data found does not yet confirm direct alignment with TAP's programmes."
        )

    # Delivery model framing (funder = highly valuable for TAP)
    if delivery and isinstance(delivery, dict):
        model = delivery.get("model", "")
        note  = delivery.get("note", "")
        if model == "FUNDER":
            lines.append(
                f"Delivery model: FUNDER — grants to NGO partners. "
                f"TAP is an ideal grantee: proven, government-integrated, "
                f"measurable impact. {note}"
            )
        elif model == "HYBRID":
            lines.append(
                f"Delivery model: HYBRID — funds partners AND runs own programmes. "
                f"Pitch TAP as a delivery-excellence partner to strengthen their "
                f"existing education portfolio. {note}"
            )
        elif model == "IMPLEMENTER":
            lines.append(
                f"Delivery model: IN-HOUSE IMPLEMENTER. "
                f"Angle: TAP as a specialist curriculum/tech partner rather than "
                f"a grant recipient. {note}"
            )

    if adj_fired:
        top = adj_fired[0]
        kws = top["keywords_found"][:3]
        lines.append(
            f"Key adjacency: {top['label']} "
            f"({'evidence: ' + ', '.join(kws) if kws else ''}). "
            f"{top['tap_reasoning']}"
        )
        for c in adj_fired:
            if c["id"] == "government_schools":
                lines.append(
                    "Government school presence means TAP can integrate directly "
                    "into their existing delivery pipeline — zero new infrastructure needed."
                )
                break

    geo_str = ", ".join(geo_places) if geo_places else "India"
    band    = score_band(fit_score)
    tier    = get_scoring_tier(fit_score)
    lines.append(
        f"Assessment: {tier['label']} ({fit_score}/100) — "
        f"{tier.get('description', band.get('label',''))}. "
        f"Active in {geo_str}."
    )
    lines.append(f"Recommended action: {tier.get('action', '')}")
    return " ".join(lines)


def score(company: str, sources: list, parsed: dict) -> dict:
    cfg   = _cfg()
    state = determine_state(sources)

    if state == "CONFIRMED_ABSENT":
        insight = generate_strategic_insight(company, state, [], [], [], 0)
        tier    = get_scoring_tier(0)
        return {"state": state, "fit_score": 0, "strategic_insight": insight,
                "band": score_band(0, cfg), "scoring_tier": tier,
                "breakdown": {}, "data": parsed, "sources": sources}

    # Run adjacency FIRST so focus scoring can use adj_fired for depth calc
    adj_score, adj_fired = _score_adjacency(parsed.get("adjacency_signals", {}), cfg)

    # ALL firing clusters (not just those within the boost cap) — used by
    # focus scoring to recognise indirect education supporters
    adj_all_fired = [
        {"id": cid, "label": info.get("label", cid)}
        for cid, info in parsed.get("adjacency_signals", {}).items()
        if info.get("fires")
    ]

    # Focus alignment with adjacency-aware depth
    fa_score, fa_matched, fa_indirect = _score_focus(
        parsed.get("focus_areas", []), cfg, adj_fired=adj_all_fired)

    # ── Semantic alignment (LLM) — lifts focus score when keywords miss meaning
    # e.g. "digital inclusion" ≈ TAP's digital literacy but scores 0 on keywords.
    # The LLM score can only RAISE the focus dimension, never lower it, and the
    # whole step degrades silently to keyword-only scoring on any failure.
    sem = None
    sem_cfg = cfg.get("semantic_scoring", {})
    if _LLM_AVAILABLE and sem_cfg.get("enabled", True):
        evidence = combine_source_texts(sources)
        if evidence.strip():
            mission = cfg.get("org_mission", _DEFAULT_MISSION)
            pctx = cfg.get("partner_pattern_context", "").strip()
            if pctx:
                mission = f"{mission}\n\nKnown-partner pattern: {pctx}"
            sem = semantic_alignment(company, mission, evidence)
    if sem:
        sem_focus = round(40 * sem["score"] / 100)
        if sem_focus > fa_score:
            fa_matched = list(fa_matched) + \
                         [f"semantic: {t}" for t in sem.get("themes", [])[:4]]
            fa_score = sem_focus
        # LLM-detected misalignment feeds the strict-penalty pass
        flags = [f.lower() for f in sem.get("flags", [])]
        if any("vocational" in f or "employability" in f for f in flags):
            parsed.setdefault("ai_flags", {})["is_employability_only"] = True
        if any("higher_ed" in f or "higher-ed" in f for f in flags):
            parsed.setdefault("ai_flags", {})["is_higher_ed_only"] = True
    geo_score, geo_label  = _score_geography(parsed.get("geography", []), cfg)
    mat_score, mat_found  = _score_maturity(sources, cfg)
    bud_score, bud_label  = _score_budget(parsed.get("spend", {}), cfg)
    src_score, src_name   = _score_source_quality(sources, cfg)
    sim_score, sim_matched = _score_partner_similarity(sources, cfg)

    # Rescale native sub-scores (fa/40, adj/20, sim/15, others/10) to weights
    _w    = cfg.get("weights", {}) or {}
    W_FA  = _w.get("focus_alignment", 40)
    W_ADJ = _w.get("adjacency_boost", 18)
    W_SIM = _w.get("partner_similarity", 15)
    W_GEO = _w.get("geography_fit", 9)
    W_MAT = _w.get("csr_maturity", 8)
    W_BUD = _w.get("budget_size", 5)
    W_SRC = _w.get("source_quality", 5)
    sim_cap   = cfg.get("partner_similarity_cap", 15)
    fa_score  = round(fa_score  / 40 * W_FA)
    adj_score = round(adj_score / 20 * W_ADJ)
    sim_score = round(sim_score / sim_cap * W_SIM)
    geo_score = round(geo_score / 10 * W_GEO)
    mat_score = round(mat_score / 10 * W_MAT)
    bud_score = round(bud_score / 10 * W_BUD)
    src_score = round(src_score / 10 * W_SRC)

    # Education-dominance dampener: stray education keywords inside a
    # health/environment-dominant portfolio must not score like a real
    # education funder. Dampens only the education-driven dimensions.
    dom_mult, dom_share = _education_dominance(sources, cfg)
    if dom_mult < 1.0:
        fa_score  = round(fa_score  * dom_mult)
        adj_score = round(adj_score * dom_mult)
        sim_score = round(sim_score * dom_mult)

    raw_total = (fa_score + adj_score + sim_score + geo_score + mat_score +
                 bud_score + src_score)
    if state == "NOT_FOUND_IN_SOURCE":
        raw_total = max(raw_total, 10)
    raw_total = min(raw_total, 100)

    # Strict penalties pass
    parsed["_sources"] = sources  # needed by penalty infra checker
    total, penalties_applied = apply_strict_penalties(raw_total, parsed, cfg)

    # Known existing TAP partner? Ground truth beats the model.
    company_l = company.lower().strip()
    is_tap_partner = any(
        p.lower() in company_l or company_l in p.lower()
        for p in cfg.get("tap_existing_partners", [])
    )

    # Indirect-alignment ceiling: education SUPPORTERS score well (Strong
    # Fit territory) but the 90+ Immediate Target tier is reserved for
    # companies whose CSR focus DIRECTLY matches TAP's mission.
    # Existing TAP partners are EXEMPT — collaboration is already proven.
    if fa_indirect and not is_tap_partner:
        ind_cap = cfg.get("indirect_alignment_cap", 89)
        if total > ind_cap:
            penalties_applied.append({
                "reason": "Indirect education alignment — Immediate Target "
                          "tier reserved for direct mission matches",
                "deduction": total - ind_cap, "cap_applied": ind_cap})
            total = ind_cap

    # Partner floor: proven TAP collaboration guarantees Immediate Target
    partner_note = None
    if is_tap_partner:
        floor = cfg.get("tap_partner_floor", 92)
        if total < floor:
            partner_note = (f"Existing TAP partner — score floored at {floor} "
                            f"(model scored {total})")
            total = floor
        else:
            partner_note = "Existing TAP partner — proven collaboration"

    tier = get_scoring_tier(total)

    breakdown = {
        "focus_alignment": {"score": fa_score,  "max": W_FA,
                             "matched": fa_matched, "label": f"{fa_score}/{W_FA}"},
        "adjacency_boost": {"score": adj_score, "max": W_ADJ,
                             "fired_clusters": adj_fired, "label": f"{adj_score}/{W_ADJ}"},
        "partner_similarity": {"score": sim_score, "max": W_SIM,
                                "matched_traits": sim_matched,
                                "label": f"{sim_score}/{W_SIM}"},
        "geography_fit":   {"score": geo_score, "max": W_GEO, "label": geo_label},
        "csr_maturity":    {"score": mat_score, "max": W_MAT,
                             "signals": mat_found, "label": f"{mat_score}/{W_MAT}"},
        "budget_size":     {"score": bud_score, "max": W_BUD, "label": bud_label},
        "source_quality":  {"score": src_score, "max": W_SRC,
                             "source": src_name, "label": f"{src_score}/{W_SRC}"},
        "penalties":       penalties_applied,
        "raw_score":       raw_total,
        "education_dominance": {"share": dom_share, "multiplier": dom_mult,
                                 "applied": dom_mult < 1.0},
        "existing_partner": {"is_partner": is_tap_partner,
                              "note": partner_note or ""},
        "semantic_alignment": {
            "used":      bool(sem),
            "score":     sem.get("score") if sem else None,
            "rationale": sem.get("rationale", "") if sem else "",
            "themes":    sem.get("themes", []) if sem else [],
        },
    }

    insight = generate_strategic_insight(
        company, state,
        parsed.get("focus_areas", []),
        adj_fired,
        parsed.get("geography", []),
        total,
        delivery=parsed.get("csr_delivery_model"),
    )
    if sem and sem.get("rationale"):
        insight = f"{insight} AI semantic analysis: {sem['rationale']}"
    if is_tap_partner:
        insight = (f"{company} is an EXISTING TAP COLLABORATING PARTNER — "
                   f"relationship already proven. {insight}")

    return {
        "state":            state,
        "fit_score":        total,
        "strategic_insight": insight,
        "band":             score_band(total, cfg),
        "scoring_tier":     tier,
        "breakdown":        breakdown,
        "data":             parsed,
        "sources":          sources,
    }
