# methodology.py — TAP CSR Research Methodology (8 criteria, 0–5 scale)
# Implements the methodology template: eight criteria each scored 0–5,
# the average becomes the verdict tier. Derived from the same evidence the
# 0–100 engine uses, so the two views always agree on facts.
# Scale: 5 strong · 3–4 partial · 1–2 weak · 0 none/to confirm.
# Tiers: Priority Hunt 4.0–5.0 · Worth Hunting 3.0–3.9 ·
#        Conditional Fit 2.0–2.9 · Low Priority < 2.0
import re
from utils import combine_source_texts

TO_CONFIRM = "To confirm"

METHOD_TIERS = [
    {"min": 4.0, "label": "Priority Hunt",   "color": "#7C3AED"},
    {"min": 3.0, "label": "Worth Hunting",   "color": "#16A34A"},
    {"min": 2.0, "label": "Conditional Fit", "color": "#0EA5E9"},
    {"min": 0.0, "label": "Low Priority",    "color": "#DC2626"},
]


def _kw(text: str, kw: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(kw.lower()) + r"(?![a-z0-9])",
                     text) is not None


def _rating(score: float) -> str:
    if score >= 4.5:
        return "Strong"
    if score >= 2.5:
        return "Partial"
    if score >= 0.5:
        return "Weak"
    return "None / to confirm"


def method_tier(avg: float) -> dict:
    for t in METHOD_TIERS:
        if avg >= t["min"]:
            return t
    return METHOD_TIERS[-1]


def derive_criteria(company: str, result: dict, cfg: dict) -> dict:
    """
    Returns {"criteria": [8 dicts], "average": float, "tier": dict,
             "csr_head_note": str, "open_questions": [str]}.
    Each criterion: {id, name, score, rating, evidence}.
    Gaps are marked 'To confirm' with where to confirm them — never filled.
    """
    parsed  = result.get("data", {}) or {}
    sources = result.get("sources", []) or []
    bd      = result.get("breakdown", {}) or {}
    text    = combine_source_texts(sources).lower()
    firing  = {cid for cid, i in (parsed.get("adjacency_signals") or {}).items()
               if i.get("fires")}
    focus_vals = [f.get("value", "").lower() for f in parsed.get("focus_areas", [])]
    criteria, open_qs = [], []

    def _dim_ratio(key):
        d = bd.get(key, {})
        mx = d.get("max") or 1
        return (d.get("score") or 0) / mx

    # 1 ── Education: intervention not scholarship (the decisive filter)
    prog_hits = sum(1 for k in ("programme", "program", "curriculum",
                                 "intervention", "classroom", "pedagogy",
                                 "training", "workshop") if _kw(text, k))
    sch = _kw(text, "scholarship") or _kw(text, "scholarships")
    score = round(5 * _dim_ratio("focus_alignment"), 1)
    if sch and prog_hits < 2:
        score = min(score, 1.5)
        ev = ("Scholarship-led education portfolio, few programme signals — "
              "scores low even on a big budget (the one filter that sorts fastest)")
    elif score >= 2.5:
        ev = ("Programme/intervention evidence: "
              + (", ".join(focus_vals[:4]) if focus_vals else f"{prog_hits} programme signals"))
    elif score > 0:
        ev = "Limited education-intervention evidence in latest-year sources"
    else:
        ev = f"{TO_CONFIRM} — check latest CSR-2 / annual report education line items"
        open_qs.append("Exact education allocation and whether it is "
                       "intervention- or scholarship-led (CSR-2 filing)")
    criteria.append({"id": "education_intervention",
                     "name": "1. Education: intervention not scholarship",
                     "score": score, "rating": _rating(score), "evidence": ev})

    # 2 ── STEM
    if "stem_exposure" in firing:
        score = 4.0 + (1.0 if any(k in " ".join(focus_vals)
                                   for k in ("stem", "coding", "science")) else 0)
        ev = "STEM/coding/robotics programmes found in CSR evidence"
    elif _kw(text, "stem") or _kw(text, "science"):
        score, ev = 2.0, "STEM mentioned but no dedicated programme found"
    else:
        score, ev = 0.0, f"{TO_CONFIRM} — no STEM signal in fetched sources"
    criteria.append({"id": "stem", "name": "2. STEM",
                     "score": round(min(score, 5), 1),
                     "rating": _rating(score), "evidence": ev})

    # 3 ── Technology & 21st-century skills
    t21 = [k for k in ("21st century skills", "21st-century skills", "life skills",
                        "critical thinking", "problem solving", "digital literacy")
           if any(k in v for v in focus_vals) or _kw(text, k)]
    if "digital_education" in firing and t21:
        score, ev = 5.0, "Digital/edtech delivery + 21CS signals: " + ", ".join(t21[:3])
    elif "digital_education" in firing:
        score, ev = 4.0, "Digital/edtech-delivered learning programmes found"
    elif t21:
        score, ev = 3.0, "21st-century skills signals: " + ", ".join(t21[:3])
    elif _kw(text, "digital"):
        score, ev = 1.5, "Generic digital mentions only"
    else:
        score, ev = 0.0, f"{TO_CONFIRM} — no technology-for-learning signal"
        open_qs.append("Whether any current programme uses technology for "
                       "learning (programme pages of implementing partners)")
    criteria.append({"id": "tech_21cs", "name": "3. Technology & 21st-century skills",
                     "score": score, "rating": _rating(score), "evidence": ev})

    # 4 ── Public-schooling understanding
    if "government_schools" in firing:
        score, ev = 5.0, "Active government-school outreach (institutional access exists)"
    elif _kw(text, "school") or _kw(text, "schools"):
        score, ev = 2.5, "School-level work found; government-school channel unclear"
    else:
        score, ev = 0.0, f"{TO_CONFIRM} — no school-system evidence in sources"
    criteria.append({"id": "public_schooling", "name": "4. Public-schooling understanding",
                     "score": score, "rating": _rating(score), "evidence": ev})

    # 5 ── Systems-change orientation
    sys_kws = [k for k in ("systems change", "system change", "scale", "policy",
                            "teacher training", "learning outcomes", "capacity building")
               if _kw(text, k)]
    base = 0.0
    if {"teacher_training", "learning_quality"} & firing:
        base = 3.5
    elif sys_kws:
        base = 2.0
    if _dim_ratio("csr_maturity") >= 0.7:
        base += 1.0
    score = round(min(base, 5), 1)
    ev = ("Signals: " + ", ".join(sys_kws[:4])) if sys_kws else (
        f"{TO_CONFIRM} — no systems-change language found")
    criteria.append({"id": "systems_change", "name": "5. Systems-change orientation",
                     "score": score, "rating": _rating(score), "evidence": ev})

    # 6 ── Funds TAP-like organisations (the best single tell)
    peers = [p for p in (parsed.get("ngo_partners") or []) if p.get("is_peer_ngo")]
    similar = [p for p in (parsed.get("ngo_partners") or []) if p.get("tap_similar")]
    if peers:
        score = 5.0
        ev = "Already funds TAP-peer NGO(s): " + ", ".join(p["name"] for p in peers[:3])
    elif similar:
        score = 4.0
        ev = "Funds education NGOs similar to TAP: " + ", ".join(p["name"] for p in similar[:3])
    elif "ngo_collaboration" in firing:
        score, ev = 3.0, "Funds NGOs through implementation partners; education-NGO overlap to confirm"
    else:
        score, ev = 0.0, f"{TO_CONFIRM} — check NGO Darpan and partner pages"
        open_qs.append("Which NGOs they currently fund (NGO Darpan, partner pages)")
    criteria.append({"id": "funds_tap_like", "name": "6. Funds TAP-like organisations",
                     "score": score, "rating": _rating(score), "evidence": ev})

    # 7 ── Budget size and trend
    spend = parsed.get("spend") or {}
    if spend.get("inr_crore"):
        score = round(5 * _dim_ratio("budget_size"), 1)
        ev = (f"Latest-year India CSR spend: {spend.get('display', '')} — "
              f"multi-year trend {TO_CONFIRM.lower()} (needs consecutive CSR-2 filings)")
        open_qs.append("Multi-year CSR obligation trend (consecutive CSR-2 filings)")
    else:
        score = 0.0
        ev = f"{TO_CONFIRM} — spend not found; read from CSR-2 via CIN"
        open_qs.append("Exact latest-year CSR spend in ₹ crore (MCA CSR-2 by CIN)")
    criteria.append({"id": "budget", "name": "7. Budget size and trend",
                     "score": score, "rating": _rating(score), "evidence": ev})

    # 8 ── Geography overlap with TAP states
    score = round(5 * _dim_ratio("geography_fit"), 1)
    geo_label = bd.get("geography_fit", {}).get("label", "")
    ev = geo_label or f"{TO_CONFIRM} — India state footprint unclear"
    criteria.append({"id": "geography", "name": "8. Geography overlap with TAP states",
                     "score": score, "rating": _rating(score), "evidence": ev})

    # ── Average and verdict
    avg  = round(sum(c["score"] for c in criteria) / len(criteria), 1)
    tier = method_tier(avg)

    # Standing open questions from the methodology
    open_qs.append("Current CSR decision-maker and their philosophy "
                   "(public profile / LinkedIn) — qualitative deciding factor")

    return {
        "criteria": criteria,
        "average": avg,
        "tier": tier,
        "csr_head_note": ("CSR head philosophy: not scored, but a deciding "
                          "factor — read their public profile for what they "
                          "champion and fund."),
        "open_questions": open_qs,
    }
